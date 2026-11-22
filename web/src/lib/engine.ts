import ForceGraph3D from '3d-force-graph';
import { Group, Mesh, MeshLambertMaterial } from 'three';
import {
  XY_SCALE, NODE_PADDING, FIT_PADDING, PAN_SLACK, FLY_MS, LEFT_INSET_PX, BOTTOM_INSET_PX, DETAILS_PANEL_PX,
  MIN_DISTANCE, BUBBLE_MAX, BUBBLE_MIN, CLUSTER_FIT, LOAD_PX, OPEN_PX, FULL_PX, FADE_START_PX, FADE_END_PX,
  BUBBLE_ALPHA, FADED_ALPHA, VIEW_MARGIN, MAX_LOADS, MAX_OPEN, MAX_LOADED, OPEN_HYSTERESIS, INNER_RADIUS,
  INNER_MAX, FILL, R_MIN, R_MAX, SIZE_CURVE, TITLE_FONT, TITLE_MIN_PX, TITLE_MAX_PX, TITLE_FIT_W, TITLE_FIT_H,
  TITLE_MOVE_START_PX, TITLE_MOVE_END_PX, TITLE_TOP_PX, EDGE_PICK_PX, EDGE_DIM, EDGE_HIGHLIGHT,
  EDGE_WIDTH_BOOST, HALO_WIDTH, HALO_MIN, HALO_MIN_PX, UNIT_SPHERE, HALO_MATERIAL, DIMMED_COLOR, PAPER_LIGHTEN,
} from '../constants';
import type {
  BubbleNode, ClusterFile, ClusterSummary, EdgeTip, FieldFilter, GraphLink, GraphNode, LayoutLink,
  LoadedCluster, Overview, PaperNode, YearSeries,
} from '../types';
import { dim, lighten, withAlpha } from './color';
import { createFieldHelpers } from './fields';
import { escapeHtml, fetchJson, stripPrefix } from './html';
import { expandClusterFile } from './slim';
import type { SlimClusterFile } from './slim';
import { resolveOverlaps } from './layout';
import { distanceToSegment, lerp, smoothstep } from './math';
import { noRaycast } from './raycast';
import { makeTitleSprite, titleDims } from './titleSprite';

// ---------------------------------------------------------------------------
// One continuous map. Each cluster ("chunk") is a bubble sized by how many papers it holds.
// Zoom into a bubble (or click it) and the bubble fades to a faint outline while its most
// influential papers appear inside it, so you can pan from cluster to cluster at full detail.
// Citation lines run between papers of different clusters when both are open, and otherwise end
// at the other cluster's bubble.
//
// The engine owns the 3D scene and everything that depends on the camera. React owns the panels:
// they read the engine's state (re-rendering whenever it calls emit) and call its actions.
// Data comes from export_web_data.py: /data/overview.json and /data/clusters/<id>.json
// ---------------------------------------------------------------------------

export type Engine = ReturnType<typeof createEngine>;

export interface EngineOptions {
  /** Folder the cluster files load from (ends with "/"): data/ for the original map, data/maps/<id>/ for drill-down maps. */
  dataDir: string;
  /**
   * Called to switch maps: a bubble that leads to another map (its `map` field) was clicked, or a node standing for
   * a cluster of another map (its `opens` field), in which case `cluster` is the cluster to fly to there.
   */
  onOpenMap?: (mapId: string, cluster?: number) => void;
  /** The cluster files are in the compact drill-down format (lib/slim.ts). */
  slim?: boolean;
}

export function createEngine(container: HTMLElement, overview: Overview, options: EngineOptions) {
  const { meta, clusters, fields } = overview;
  const clusterById = new Map(clusters.map((c) => [c.id, c]));
  const fieldHelpers = createFieldHelpers(fields);
  const { key: fieldKey, name: fieldName, clusterColor } = fieldHelpers;

  // ---- Change notification for React ----
  let version = 0;
  const listeners = new Set<() => void>();
  const emit = () => {
    version++;
    listeners.forEach((l) => l());
  };
  const subscribe = (l: () => void) => {
    listeners.add(l);
    return () => {
      listeners.delete(l);
    };
  };

  // ---- State ----
  let nodes: GraphNode[] = []; // everything the graph is drawing: all bubbles + the papers of every loaded cluster
  let nodeById = new Map<string | number, GraphNode>();
  let hovered: GraphNode | null = null;
  let detailsFor: PaperNode | null = null; // the paper shown in the details panel
  let searchHover: BubbleNode | null = null; // the bubble whose search result the mouse is over
  let focused: BubbleNode | null = null; // the open bubble nearest the middle of the screen, if any
  let routed: number | null = null; // cluster id the URL hash asks for
  let fieldFilter: FieldFilter = null; // null, an arXiv field code, or 'other'
  let attached = new Set<GraphLink>(); // citation lines touching the selected paper
  let edgeHoverNode: GraphNode | null = null; // the paper/bubble at the far end of the line under the mouse
  let edgeTip: EdgeTip | null = null;
  const loaded = new Map<number, LoadedCluster>();
  const loading = new Set<number>();
  const failed = new Set<number>();
  let destroyed = false;

  // ---- Time window (set by the slider at the bottom) ----
  const T0 = meta.yearMin;
  const T1 = meta.yearMax;
  let windowLo = T0;
  let windowHi = T1;
  const windowIsFull = () => windowLo <= T0 && windowHi >= T1;

  // Per-year paper counts of a cluster for the current field filter.
  const countsOf = (c: ClusterSummary) => (fieldFilter ? c.fy[fieldFilter] : c.count);
  // How many papers of those counts fall inside the window. Counts are per whole year, so a year
  // that is only partly inside the window counts proportionally.
  function inWindow(counts: number[]): number {
    let total = 0;
    overview.years.forEach((y, i) => {
      const a = Math.max(y, T0);
      const b = Math.min(y + 1, T1);
      if (b <= a) return;
      const overlap = Math.min(b, windowHi) - Math.max(a, windowLo);
      if (overlap > 0) total += counts[i] * Math.min(1, overlap / (b - a));
    });
    return total;
  }

  const yearVisible = (n: { year: number | null }) =>
    n.year == null ? windowIsFull() : n.year >= windowLo && n.year <= windowHi;
  const isVisible = (n: GraphNode): boolean => {
    if (n.kind === 'cluster') return n.inWindow >= 0.5;
    return n.bubble.open && yearVisible(n); // a paper shows only while its bubble is open
  };
  // While a graph is being built the library may still hold a link's ends as plain ids rather than
  // node objects, so link functions look the nodes up themselves.
  const endpoint = (e: GraphLink['source']): GraphNode | undefined =>
    typeof e === 'object' ? e : nodeById.get(e);
  function linkVisible(l: GraphLink): boolean {
    const a = endpoint(l.source);
    const b = endpoint(l.target);
    if (!a || !b || !isVisible(a) || !isVisible(b)) return false;
    // The lines between bubbles would run across the papers of an open bubble, so they hide.
    if (l.kind === 'cluster') return !(a as BubbleNode).open && !(b as BubbleNode).open;
    return true;
  }

  // ---- Bubbles (built once; positions never change) ----
  const maxSize = Math.max(...clusters.map((c) => c.size));
  const bubbleNodes: BubbleNode[] = clusters.map((c) => {
    const x = c.x * XY_SCALE;
    const y = c.y * XY_SCALE;
    const fullRadius = BUBBLE_MAX * Math.sqrt(c.size / maxSize);
    const title = stripPrefix(c.title);
    const lines = title.split(' / '); // the clustering names a cluster with a few keywords separated by " / "
    return {
      id: c.id, kind: 'cluster', cluster: c, x, y, z: 0, fx: x, fy: y, fz: 0,
      fullRadius, shrunk: fullRadius, radius: fullRadius, inWindow: c.size, open: false, px: 0, inView: false,
      title: { lines, centerDims: titleDims(lines), oneLine: [title], topDims: titleDims([title]) },
    };
  });
  // UMAP knows nothing about bubble sizes, so nudge overlapping ones apart, then re-pin.
  resolveOverlaps(bubbleNodes, { padding: NODE_PADDING });
  bubbleNodes.forEach((n) => { n.fx = n.x; n.fy = n.y; });
  const bubbleOf = new Map(bubbleNodes.map((n) => [n.id, n]));
  const maxWeight = Math.max(...overview.links.map((l) => l.weight));
  const bubbleLinks: GraphLink[] = overview.links.map((l) => ({
    ...l,
    kind: 'cluster',
    color: `rgba(255, 255, 255, ${(0.14 + 0.5 * Math.sqrt(l.weight / maxWeight)).toFixed(2)})`,
  }));

  // Narrowing the time window or picking a field shrinks bubbles (never grows them, so no new
  // overlaps appear). A bubble's area is its share of papers inside the window that match the field.
  function sizeBubbles() {
    bubbleNodes.forEach((n) => {
      n.inWindow = inWindow(countsOf(n.cluster));
      n.shrunk = Math.max(BUBBLE_MIN, n.fullRadius * Math.sqrt(n.inWindow / n.cluster.size));
    });
  }

  // ---- Node + link appearance ----
  function nodeColor(n: GraphNode): string {
    if (n.kind === 'cluster') return clusterColor(n.cluster);
    // Papers outside the selected field fade out; the rest take their cluster's color.
    if (fieldFilter && fieldKey(n.categoryCode) !== fieldFilter) return DIMMED_COLOR;
    return lighten(clusterColor(n.bubble.cluster), PAPER_LIGHTEN); // lighter than the bubble, so they stand out inside it
  }
  let pxPerUnitNow = 1; // current zoom, kept up to date by updateLOD
  const haloRadius = (n: GraphNode) => n.radius + Math.max(HALO_MIN, n.radius * HALO_WIDTH, HALO_MIN_PX / pxPerUnitNow);
  const updateHalo = (n: GraphNode | null | undefined) => {
    if (!n?.halo) return;
    n.halo.visible = n === hovered || n === detailsFor || n === searchHover || n === edgeHoverNode;
    n.halo.scale.setScalar(haloRadius(n));
  };
  function makeHalo(n: GraphNode): Mesh {
    const halo = new Mesh(UNIT_SPHERE, HALO_MATERIAL);
    halo.raycast = noRaycast; // the halo shouldn't count as part of the node for hover/click
    n.halo = halo;
    updateHalo(n); // the library rebuilds a node's objects when it re-enters the window
    return halo;
  }
  // A bubble is our own object, not the library's sphere, so we can fade it as you zoom in:
  // it must not write to the depth buffer, or it would hide the papers inside it.
  function makeBubble(n: BubbleNode): Group {
    const group = new Group();
    const sphere = new Mesh(
      UNIT_SPHERE,
      new MeshLambertMaterial({ color: clusterColor(n.cluster), transparent: true, opacity: BUBBLE_ALPHA, depthWrite: false }),
    );
    sphere.renderOrder = -1; // drawn before the papers
    sphere.scale.setScalar(n.radius);
    n.sphere = sphere;
    n.group = group;
    n.titleCenter = n.titleTop = null; // (re)built on demand: the library recreates this object when the bubble re-enters the time window
    group.add(sphere);
    group.add(makeHalo(n));
    return group;
  }
  const nodeObject = (n: GraphNode) => (n.kind === 'cluster' ? makeBubble(n) : makeHalo(n));

  function nodeLabel(n: GraphNode): string {
    if (n.kind === 'cluster') {
      const c = n.cluster;
      const filtered = fieldFilter !== null;
      const inRange = Math.round(n.inWindow);
      const note = windowIsFull() && !filtered
        ? ''
        : ` ${filtered ? `${fieldName(fieldFilter!)} ` : ''}in range (of ${c.size.toLocaleString()})`;
      return `
        <b>${escapeHtml(stripPrefix(c.title))}</b><br/>
        ${inRange.toLocaleString()} papers${note}<br/>
        Mostly ${escapeHtml(c.categoryName)} (${Math.round(c.categoryShare * 100)}%)<br/>
        Top paper: ${escapeHtml(c.topPaper)}<br/>
        <span style="color:#8b92a5">${c.map ? 'Click to open its map · zoom in to see its clusters' : 'Click or zoom in to open'}</span>`;
    }
    if (n.opens) {
      const whole = n.opens.cluster === undefined; // the node stands for a whole category map, not one cluster
      return `
        <b>${escapeHtml(n.title)}</b><br/>
        ${n.opens.papers.toLocaleString()} papers${whole ? '' : ` · a cluster of ${escapeHtml(n.opens.map)}`}<br/>
        <span style="color:#8b92a5">Click to open ${whole ? 'its map' : `it in the ${escapeHtml(n.opens.map)} map`}</span>`;
    }
    // Citation counts and influence aren't in every dataset yet; leave out what is missing.
    const stats = [
      n.citations ? `${n.citations.toLocaleString()} citations` : '',
      n.influence ? `influence ${n.influence.toFixed(2)}` : '',
    ].filter(Boolean).join(' · ');
    return `
      <b>${escapeHtml(n.title)}</b><br/>
      ${escapeHtml(n.date ?? 'unknown date')} · ${escapeHtml(n.category)}<br/>
      ${stats ? `${stats}<br/>` : ''}
      <span style="color:#8b92a5">${escapeHtml(stripPrefix(n.bubble.cluster.title))}</span>`;
  }

  // Lines between bubbles light up for the hovered bubble; the rest stay faint.
  function linkColor(l: GraphLink): string {
    if (attached.has(l)) return EDGE_HIGHLIGHT;
    let color: string;
    if (l.kind === 'cluster') {
      const touches = hovered && (endpoint(l.source) === hovered || endpoint(l.target) === hovered);
      color = touches ? 'rgba(255, 255, 255, 0.95)' : l.color!;
    } else {
      color = l.color ?? 'rgba(255, 255, 255, 0.55)'; // citations to other clusters take that cluster's color
    }
    return attached.size ? dim(color, EDGE_DIM) : color; // while a paper is selected, its lines stand out
  }
  // Paper-to-paper lines scale with their cluster (their papers are tiny in world units); lines between
  // bubbles, or from a paper to a far-away bubble, are hairlines so they look right at any zoom.
  const linkWidth = (l: GraphLink) => (l.kind === 'paper' ? 2 * l.scale! * (attached.has(l) ? EDGE_WIDTH_BOOST : 1) : 0);
  const arrowLength = (l: GraphLink) => (l.kind === 'paper' ? 8 * l.scale! * (attached.has(l) ? EDGE_WIDTH_BOOST : 1) : 0);

  // ---- The graph ----
  // The library's typings are loose about node/link shapes, so the instance is untyped; everything we
  // hand it and read back from it is typed on our side.
  // Orbit controls let us switch rotate off and clamp zoom (the default trackball can't).
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const graph: any = new (ForceGraph3D as any)(container, { controlType: 'orbit' })
    .backgroundColor('#0b0d12')
    .nodeColor(nodeColor)
    .nodeVisibility(isVisible)
    .linkVisibility(linkVisible)
    // Opaque nodes, so the white halo behind them doesn't show through.
    .nodeOpacity(1)
    .nodeThreeObject(nodeObject)
    .nodeThreeObjectExtend((n: GraphNode) => n.kind !== 'cluster') // papers: keep the default sphere and add a halo
    // The library's radius = cbrt(nodeVal) * nodeRelSize, so with relSize 1, nodeVal = radius³.
    .nodeVal((n: GraphNode) => n.radius ** 3)
    .nodeRelSize(1)
    .nodeResolution(12)
    .enableNodeDrag(false)
    .nodeLabel(nodeLabel)
    .linkColor(linkColor)
    .linkWidth(linkWidth)
    .linkDirectionalArrowLength(arrowLength) // arrows only for paper-to-paper citations
    .linkDirectionalArrowRelPos(1)
    .onNodeHover((n: GraphNode | null, prev: GraphNode | null) => {
      hovered = n;
      updateHalo(prev);
      updateHalo(n);
      if (n?.kind === 'cluster' || prev?.kind === 'cluster') graph.linkColor(linkColor); // re-apply so the lines re-color
    })
    .onNodeClick((n: GraphNode) => {
      // Drill down a level: a category bubble opens its map; one of its cluster nodes opens that map at that cluster.
      if (n.kind === 'cluster' && n.cluster.map && options.onOpenMap) options.onOpenMap(n.cluster.map);
      else if (n.kind === 'paper' && n.opens && options.onOpenMap) options.onOpenMap(n.opens.map, n.opens.cluster);
      else if (n.kind === 'cluster') navigate(`#cluster=${n.id}`);
      else selectPaper(n);
    })
    // A click on empty space follows a highlighted line if one is right there; otherwise it deselects.
    .onBackgroundClick((event: MouseEvent) => handleEmptyClick(event))
    .onLinkClick((_link: GraphLink, event: MouseEvent) => handleEmptyClick(event))
    // Positions are fixed, so there is nothing for the physics engine to do.
    .warmupTicks(0)
    .cooldownTicks(0);

  // Flat map: look straight down the z-axis. No rotating; drag to pan, scroll to zoom toward the pointer.
  const controls = graph.controls();
  controls.enableRotate = false;
  controls.enablePan = true;
  controls.screenSpacePanning = true; // pan moves in the screen plane (= the map plane here)
  controls.zoomToCursor = true; // scrolling zooms toward the mouse, so you can dive into any bubble
  // Three.js MOUSE / TOUCH enum values: MOUSE.PAN = 2, MOUSE.DOLLY = 1, TOUCH.PAN = 1, TOUCH.DOLLY_PAN = 2
  controls.mouseButtons = { LEFT: 2, MIDDLE: 1, RIGHT: 2 };
  controls.touches = { ONE: 1, TWO: 2 };

  const camera = graph.camera();
  const tanHalfFov = () => Math.tan((camera.fov * Math.PI) / 360);
  const aspect = () => window.innerWidth / window.innerHeight; // camera.aspect updates a tick late on resize

  // Bounding box of the bubble map (circle edges included); camera limits are based on it.
  const bounds = (() => {
    const minX = Math.min(...bubbleNodes.map((n) => n.x - n.fullRadius));
    const maxX = Math.max(...bubbleNodes.map((n) => n.x + n.fullRadius));
    const minY = Math.min(...bubbleNodes.map((n) => n.y - n.fullRadius));
    const maxY = Math.max(...bubbleNodes.map((n) => n.y + n.fullRadius));
    return { cx: (minX + maxX) / 2, cy: (minY + maxY) / 2, halfW: (maxX - minX) / 2, halfH: (maxY - minY) / 2 };
  })();

  let flight: object | null = null;
  // Every animation frame we ask for is tracked, so destroy() can cancel the ones still pending.
  const pendingFrames = new Set<number>();
  const frame = (fn: FrameRequestCallback) => {
    const id = requestAnimationFrame((t) => {
      pendingFrames.delete(id);
      if (!destroyed) fn(t);
    });
    pendingFrames.add(id);
  };

  // The info panel covers the left edge and the timeline covers the bottom, so views are fitted
  // into the free space (wide/tall screens only; on small windows the panels just overlay the map).
  const leftInset = () => (window.innerWidth >= 1100 ? LEFT_INSET_PX : 0);
  const bottomInset = () => (window.innerHeight >= 600 ? BOTTOM_INSET_PX : 0);
  const freeSize = () => ({ w: window.innerWidth - leftInset(), h: window.innerHeight - bottomInset() });
  interface View { x: number; y: number; z: number }
  // A camera position that puts world point (x, y) in the middle of the free area at `distance`.
  // `rightPx` reserves extra room on the right (for the paper details panel).
  function viewAt(x: number, y: number, distance: number, rightPx = 0): View {
    const unitsPerPx = (2 * distance * tanHalfFov()) / window.innerHeight;
    // The free area's center is right of / above the screen center, so the camera aims left of / below the point.
    return { x: x - ((leftInset() - rightPx) / 2) * unitsPerPx, y: y - (bottomInset() / 2) * unitsPerPx, z: distance };
  }
  // The view that shows the whole map: as close as possible while everything still fits.
  function homeView(): View {
    const t = tanHalfFov();
    const { w, h } = freeSize();
    const distance = FIT_PADDING * Math.max(bounds.halfH / (t * (h / window.innerHeight)), bounds.halfW / (t * (w / window.innerHeight)));
    return viewAt(bounds.cx, bounds.cy, distance);
  }
  // The view that makes one bubble fill most of the free area.
  function clusterView(bubble: BubbleNode, rightPx = 0): View {
    const { w, h } = freeSize();
    const pxPerUnit = (CLUSTER_FIT * Math.min(w - rightPx, h)) / (2 * bubble.fullRadius);
    const distance = Math.max(MIN_DISTANCE, window.innerHeight / (2 * tanHalfFov() * pxPerUnit));
    return viewAt(bubble.x, bubble.y, distance, rightPx);
  }
  function fitToWindow() {
    flight = null;
    graph.width(window.innerWidth).height(window.innerHeight);
    const home = homeView();
    graph.cameraPosition(home, { x: home.x, y: home.y, z: 0 });
    controls.maxDistance = home.z; // can't zoom out past the full map
    controls.minDistance = MIN_DISTANCE;
  }

  // Smooth camera move that keeps the camera straight above its target the whole way.
  // (The library's own cameraPosition() animation moves the camera and its look-at point on
  // different timers, which tilts the view mid-flight.) Zoom is interpolated in log space so it
  // feels even; any user pan/zoom cancels the flight.
  function flyCamera(to: View, ms: number) {
    const from = { x: camera.position.x, y: camera.position.y, z: camera.position.z };
    const start = performance.now();
    const mine = (flight = {});
    const step = (now: number) => {
      if (flight !== mine) return; // cancelled or superseded
      const p = Math.min(1, (now - start) / ms);
      const e = 1 - Math.pow(1 - p, 3); // ease-out cubic
      const x = lerp(from.x, to.x, e);
      const y = lerp(from.y, to.y, e);
      const z = from.z * Math.pow(to.z / from.z, e);
      camera.position.set(x, y, z);
      controls.target.set(x, y, 0);
      scheduleLOD();
      if (p < 1) frame(step);
      else flight = null;
    };
    frame(step);
  }
  const cancelFlight = () => { flight = null; };
  const canvas: HTMLCanvasElement = graph.renderer().domElement;
  canvas.addEventListener('pointerdown', cancelFlight);
  canvas.addEventListener('wheel', cancelFlight, { passive: true });

  // Pan bound: the visible area may go past the (padded) map edges by PAN_SLACK of a half-view.
  // The further you zoom in, the further from the center you can go, and you can never
  // pan far enough to lose the map entirely.
  function clampPan() {
    if (flight) return; // the fly-to animation drives the camera itself; clamping mid-flight causes jitter
    const viewHalfH = camera.position.z * tanHalfFov();
    const viewHalfW = viewHalfH * aspect();
    const roomX = Math.max(0, bounds.halfW * FIT_PADDING - viewHalfW) + PAN_SLACK * viewHalfW;
    const roomY = Math.max(0, bounds.halfH * FIT_PADDING - viewHalfH) + PAN_SLACK * viewHalfH;
    const cx = bounds.cx - (leftInset() / window.innerWidth) * viewHalfW; // same shifts as viewAt()
    const cy = bounds.cy - (bottomInset() / window.innerHeight) * viewHalfH;
    const x = Math.min(cx + roomX, Math.max(cx - roomX, controls.target.x));
    const y = Math.min(cy + roomY, Math.max(cy - roomY, controls.target.y));
    if (x === controls.target.x && y === controls.target.y) return;
    // Keep the camera directly above its target so the view stays flat.
    camera.position.x = controls.target.x = x;
    camera.position.y = controls.target.y = y;
  }
  const onControlsChange = () => { clampPan(); scheduleLOD(); }; // fires on every pan and zoom
  controls.addEventListener('change', onControlsChange);
  const onResize = () => { fitToWindow(); clampPan(); scheduleLOD(); };
  window.addEventListener('resize', onResize);

  // The library's built-in hint still describes rotating, which is turned off.
  const navHint = container.querySelector('.scene-nav-info');
  if (navHint) navHint.textContent = 'Left or Right-click: pan, Mouse-wheel/middle-click: zoom toward the pointer';

  // ---- Papers inside bubbles ----
  // Lay out one cluster's papers inside its bubble. Returns the paper nodes and their links.
  function buildCluster(bubble: BubbleNode, data: ClusterFile): LoadedCluster {
    // The cluster's own top papers set the layout and the sizes. "Extra" papers (included because another
    // cluster's line leads to them) are placed in the same frame, at the smallest size.
    const core = data.nodes.filter((p) => !p.extra);
    // Size by influence rank (0 = most influential).
    const ranked = [...core].sort((a, b) => b.influence - a.influence);
    const rank = new Map(ranked.map((p, i) => [p.id, i]));

    // Center on the median paper and scale so 90% of papers fill INNER_RADIUS; the few far outliers
    // are squashed inward, so the dense core stays readable.
    const median = (values: number[]) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
    const mx = median(core.map((p) => p.x));
    const my = median(core.map((p) => p.y));
    const dists = core.map((p) => Math.hypot(p.x - mx, p.y - my)).sort((a, b) => a - b);
    const r90 = dists[Math.floor(dists.length * 0.9)] || 1;
    const place = (p: { x: number; y: number }) => {
      const dx = p.x - mx;
      const dy = p.y - my;
      const r = Math.hypot(dx, dy);
      const scaled = (r / r90) * INNER_RADIUS;
      const shown = scaled <= INNER_RADIUS ? scaled : Math.min(INNER_RADIUS + (scaled - INNER_RADIUS) * 0.25, INNER_MAX);
      const k = r ? shown / r : 0;
      return { x: dx * k, y: dy * k };
    };
    const papers: PaperNode[] = data.nodes.map((p) => {
      const pct = p.extra ? 0 : ranked.length > 1 ? 1 - rank.get(p.id)! / (ranked.length - 1) : 1;
      const { x, y } = place(p);
      return {
        ...p, kind: 'paper', x, y, z: 0, fx: 0, fy: 0, fz: 0, bubble,
        radius: R_MIN + (R_MAX - R_MIN) * Math.pow(pct, SIZE_CURVE),
      };
    });
    resolveOverlaps(papers, { padding: NODE_PADDING }); // papers can't overlap each other

    // Shrink the whole layout to fit inside the bubble and move it to the bubble's position.
    const extent = Math.max(...papers.map((p) => Math.hypot(p.x, p.y) + p.radius));
    const scale = (bubble.fullRadius * FILL) / extent;
    papers.forEach((p) => {
      p.x = bubble.x + p.x * scale;
      p.y = bubble.y + p.y * scale;
      p.radius *= scale;
      Object.assign(p, { z: 0, fx: p.x, fy: p.y, fz: 0, bubble });
    });

    // Remember which outside papers each citation line leads to.
    const outside = new Map(data.external.map((p) => [p.id, p]));
    const links: LayoutLink[] = data.links.map((l) => {
      if (!l.cross) return { source: l.source, target: l.target };
      const ext = (outside.get(l.source) ?? outside.get(l.target))!;
      return { source: l.source, target: l.target, cross: true, outsideId: ext.id, neighbor: ext.cluster, ext };
    });
    return { papers, links, scale, lastUsed: performance.now(), linkCache: new Map() };
  }

  // Rebuild the graph's node and link lists after a cluster loads. Existing node objects are reused,
  // so nothing already on screen is recreated.
  function rebuildGraph() {
    nodes = [...bubbleNodes, ...[...loaded.values()].flatMap((e) => e.papers)];
    nodeById = new Map(nodes.map((n) => [n.id, n]));

    // Line objects are reused between rebuilds (the library only keeps drawings for objects it has
    // seen before), so a new cluster loading doesn't make it redraw every line on screen.
    const reuse = (entry: LoadedCluster, key: string, make: () => GraphLink) => {
      let link = entry.linkCache.get(key);
      if (!link) entry.linkCache.set(key, (link = make()));
      return link;
    };
    const links: GraphLink[] = [...bubbleLinks];
    const seen = new Set<string>();
    for (const entry of loaded.values()) {
      for (const l of entry.links) {
        if (!l.cross) {
          links.push(reuse(entry, `i|${l.source}|${l.target}`, () => ({ source: l.source, target: l.target, kind: 'paper', scale: entry.scale })));
          continue;
        }
        const neighborColor = clusterColor(clusterById.get(l.neighbor!)!);
        const color = withAlpha(neighborColor, 0.7);
        const stubColor = withAlpha(neighborColor, 0.4);
        if (nodeById.has(l.outsideId!)) {
          // Both papers are on screen (the other cluster is loaded too): a real paper-to-paper line.
          const key = `${l.source}>${l.target}`;
          if (seen.has(key)) continue; // the same citation is listed in both clusters' files
          seen.add(key);
          links.push({ source: l.source, target: l.target, kind: 'paper', cross: true, color, scale: entry.scale });
        } else {
          // The other end isn't loaded (or isn't one of its top papers): end the line at that cluster's bubble.
          const mine = l.source === l.outsideId ? l.target : l.source;
          const info = { kind: 'stub' as const, color: stubColor, scale: entry.scale, neighbor: l.neighbor, ext: l.ext };
          links.push(reuse(entry, `s|${l.source}|${l.target}`, () => (l.source === l.outsideId
            ? { source: l.neighbor!, target: mine, ...info }
            : { source: mine, target: l.neighbor!, ...info })));
        }
      }
    }
    graph.graphData({ nodes, links });
    refreshHighlight();
    emit(); // the focused cluster's paper count may have changed
  }

  // Several clusters often finish loading together; rebuild the scene once for all of them.
  let rebuildTimer: ReturnType<typeof setTimeout> | null = null;
  function scheduleRebuild() {
    if (rebuildTimer === null) rebuildTimer = setTimeout(flushRebuild, 60);
  }
  function flushRebuild() {
    if (rebuildTimer === null || destroyed) return;
    clearTimeout(rebuildTimer);
    rebuildTimer = null;
    rebuildGraph();
    scheduleLOD();
  }

  // Keep memory and scene size bounded: drop the least recently open clusters beyond MAX_LOADED.
  function evictOld(keepId: number) {
    while (loaded.size > MAX_LOADED) {
      let oldest: { id: number; entry: LoadedCluster } | null = null;
      for (const [id, entry] of loaded) {
        if (id === keepId || bubbleOf.get(id)!.open || detailsFor?.bubble.id === id) continue;
        if (!oldest || entry.lastUsed < oldest.entry.lastUsed) oldest = { id, entry };
      }
      if (!oldest) return;
      loaded.delete(oldest.id);
    }
  }

  // Fetch a cluster's papers (once). Resolves when they are loaded.
  const pending = new Map<number, Promise<LoadedCluster | void>>();
  function ensureLoaded(id: number): Promise<LoadedCluster | void> {
    if (loaded.has(id)) return Promise.resolve(loaded.get(id));
    if (!pending.has(id)) {
      loading.add(id);
      pending.set(id, fetchJson<ClusterFile | SlimClusterFile>(`${options.dataDir}clusters/${id}.json`)
        .then((raw) => {
          // Drill-down maps store their files compactly (lib/slim.ts); the original map's files are used as they are.
          const data = options.slim ? expandClusterFile(raw as SlimClusterFile, id, overview) : (raw as ClusterFile);
          if (destroyed) return;
          loaded.set(id, buildCluster(bubbleOf.get(id)!, data));
          evictOld(id);
          scheduleRebuild();
          return loaded.get(id);
        })
        .catch((err) => { console.error(err); failed.add(id); })
        .finally(() => { loading.delete(id); pending.delete(id); if (!destroyed) scheduleLOD(); }));
    }
    return pending.get(id)!;
  }

  // ---- Level of detail ----
  // How big a bubble looks on screen decides how much of it we show: far away it is a solid
  // bubble; up close it is a faint outline around its papers.
  let lodQueued = false;
  function scheduleLOD() {
    if (lodQueued) return;
    lodQueued = true;
    frame(() => { lodQueued = false; updateLOD(); });
  }

  interface ViewBox { left: number; right: number; top: number; bottom: number }
  // Show a bubble's title: in its middle while it is closed, above its outline once it opens.
  function updateTitle(n: BubbleNode, pxPerUnit: number, view: ViewBox) {
    if (!n.group) return; // the bubble isn't being drawn (outside the time window)
    if (!n.inView) { // off screen: draw nothing, and don't build the text image until it is needed
      if (n.titleCenter) n.titleCenter.visible = false;
      if (n.titleTop) n.titleTop.visible = false;
      return;
    }
    const t = n.title;
    const moved = n.open ? smoothstep(TITLE_MOVE_START_PX, TITLE_MOVE_END_PX, n.px) : 0; // 0 = middle, 1 = top

    // -- Middle: sized to fit inside the bubble, hidden when it would be too small to read --
    let centerAlpha = 0;
    let centerScale = 0;
    if (moved < 1) {
      const fit = Math.min((TITLE_FIT_W * n.radius) / t.centerDims.w, (TITLE_FIT_H * n.radius) / t.centerDims.h);
      const fontPx = fit * TITLE_FONT * pxPerUnit; // how big the text would look on screen
      centerScale = fontPx > TITLE_MAX_PX ? (fit * TITLE_MAX_PX) / fontPx : fit;
      centerAlpha = smoothstep(TITLE_MIN_PX - 1, TITLE_MIN_PX + 2, Math.min(fontPx, TITLE_MAX_PX)) * (1 - moved);
    }
    if (centerAlpha > 0.01) {
      if (!n.titleCenter) n.group.add((n.titleCenter = makeTitleSprite(t.lines, t.centerDims)));
      n.titleCenter.visible = true;
      n.titleCenter.material.opacity = centerAlpha;
      n.titleCenter.scale.set(t.centerDims.w * centerScale, t.centerDims.h * centerScale, 1);
    } else if (n.titleCenter) {
      n.titleCenter.visible = false;
    }

    // -- Top: constant on-screen size, just above the outline, kept inside the visible area --
    if (moved > 0.01) {
      if (!n.titleTop) n.group.add((n.titleTop = makeTitleSprite(t.oneLine, t.topDims)));
      const k = TITLE_TOP_PX / (TITLE_FONT * pxPerUnit);
      const w = t.topDims.w * k;
      const h = t.topDims.h * k;
      const z = camera.position.z;
      // The outline as the camera sees it (a sphere looks a bit bigger than its flat radius when close).
      const outline = z > n.radius * 1.05 ? (n.radius * z) / Math.sqrt(z * z - n.radius * n.radius) : n.radius * 2;
      const margin = 6 / pxPerUnit;
      let y = outline + margin + h / 2;
      y = Math.min(y, view.top - n.y - margin - h / 2); // stick to the top of the screen if the outline is above it
      y = Math.max(y, -outline + h / 2); // ...but never below the bubble
      const minX = view.left - n.x + w / 2 + margin;
      const maxX = view.right - n.x - w / 2 - margin;
      let x = minX <= maxX ? Math.min(Math.max(0, minX), maxX) : 0; // keep it on screen unless it is wider than the view
      const span = outline - w / 2; // ...and above the bubble horizontally
      x = span > 0 ? Math.min(Math.max(x, -span), span) : 0;
      n.titleTop.visible = true;
      n.titleTop.material.opacity = moved;
      n.titleTop.scale.set(w, h, 1);
      n.titleTop.position.set(x, y, 0);
    } else if (n.titleTop) {
      n.titleTop.visible = false;
    }
  }

  let openKey = '';
  function updateLOD() {
    if (destroyed) return;
    const tan = tanHalfFov();
    const pxPerUnit = (pxPerUnitNow = window.innerHeight / (2 * camera.position.z * tan));
    const halfH = camera.position.z * tan * VIEW_MARGIN;
    const halfW = halfH * aspect();
    // The middle of the free screen area (not hidden behind the panels), in world units.
    const fx = camera.position.x + leftInset() / 2 / pxPerUnit;
    const fy = camera.position.y + bottomInset() / 2 / pxPerUnit;

    // Candidates: bubbles on screen that are big enough to matter, nearest the middle first.
    const candidates: { n: BubbleNode; d: number }[] = [];
    for (const n of bubbleNodes) {
      n.px = n.fullRadius * pxPerUnit;
      const inView = (n.inView =
        Math.abs(n.x - camera.position.x) < halfW + n.fullRadius && Math.abs(n.y - camera.position.y) < halfH + n.fullRadius);
      if (inView && n.px >= LOAD_PX && n.inWindow >= 0.5) {
        candidates.push({ n, d: Math.hypot(n.x - fx, n.y - fy) * (n.open ? OPEN_HYSTERESIS : 1) });
      }
    }
    candidates.sort((a, b) => a.d - b.d);

    // Only the MAX_OPEN nearest big-enough bubbles may open. A few more are fetched ahead of time,
    // so panning to the next bubble doesn't wait on the network.
    const open = new Set<number>();
    let slots = MAX_OPEN;
    candidates.forEach(({ n }, rank) => {
      const wantsOpen = n.px >= OPEN_PX && slots > 0;
      if (wantsOpen) slots--;
      if ((wantsOpen || rank < MAX_OPEN + 2) && !loaded.has(n.id) && !failed.has(n.id) && !loading.has(n.id) && loading.size < MAX_LOADS) ensureLoaded(n.id);
      if (wantsOpen && loaded.has(n.id)) open.add(n.id);
    });

    // The part of the map visible between the panels, in world units (for keeping titles on screen).
    const upp = 1 / pxPerUnit;
    const view: ViewBox = {
      left: camera.position.x - halfW / VIEW_MARGIN + leftInset() * upp,
      right: camera.position.x + halfW / VIEW_MARGIN,
      top: camera.position.y + halfH / VIEW_MARGIN,
      bottom: camera.position.y - halfH / VIEW_MARGIN + bottomInset() * upp,
    };

    // While flying to a cluster, open it straight away (it may still look tiny): the paper we are heading to
    // is then already drawn, selected and outlined by the time the camera arrives.
    if (flight && routed !== null && loaded.has(routed) && bubbleOf.get(routed)!.inWindow >= 0.5 && !open.has(routed)) {
      if (open.size >= MAX_OPEN) {
        // Stay within MAX_OPEN: make room by closing the open bubble farthest from the middle.
        const farthest = [...candidates].reverse().find(({ n }) => open.has(n.id));
        if (farthest) open.delete(farthest.n.id);
      }
      open.add(routed);
    }

    const now = performance.now();
    for (const n of bubbleNodes) {
      n.open = open.has(n.id);
      if (n.open) loaded.get(n.id)!.lastUsed = now;

      // Expand from the (possibly shrunken) size to the full size as it opens, then fade out.
      n.radius = Math.max(BUBBLE_MIN, lerp(n.shrunk, n.fullRadius, n.open ? smoothstep(OPEN_PX, FULL_PX, n.px) : 0));
      if (n.sphere) {
        n.sphere.scale.setScalar(n.radius);
        (n.sphere.material as MeshLambertMaterial).opacity = n.open
          ? lerp(BUBBLE_ALPHA, FADED_ALPHA, smoothstep(FADE_START_PX, FADE_END_PX, n.px))
          : BUBBLE_ALPHA;
        // An open bubble must not catch the mouse, or you could never hover the papers inside it.
        n.sphere.raycast = n.open ? noRaycast : Mesh.prototype.raycast;
      }
      updateHalo(n);
      updateTitle(n, pxPerUnit, view);
    }

    updateHalo(detailsFor); // paper outlines keep their on-screen thickness as you zoom
    updateHalo(hovered);
    updateHalo(edgeHoverNode);

    // Papers and lines come and go as bubbles open and close.
    const key = [...open].join(',');
    if (key !== openKey) {
      openKey = key;
      graph.nodeVisibility(isVisible).linkVisibility(linkVisible);
    }

    // The "focused" cluster is the open bubble nearest the middle; the left panel and histogram describe it.
    const best = candidates.find(({ n }) => open.has(n.id))?.n ?? null;
    if (best !== focused) {
      focused = best;
      emit();
    }
  }

  // ---- Navigation ----
  // The URL hash says which cluster was asked for (#cluster=12), so a cluster can be bookmarked or
  // shared, and the browser's back button flies back out. #cluster=12&paper=ARXIV:1234.5678 also
  // opens that paper's details. Zooming by hand doesn't change the URL.
  function route() {
    const m = /^#cluster=(\d+)(?:&paper=(.+))?$/.exec(location.hash);
    if (m) return goToCluster(Number(m[1]), m[2] ? decodeURIComponent(m[2]) : null);
    if (routed !== null) goHome();
    routed = null;
  }
  function navigate(hash: string, { replace = false } = {}) {
    const url = hash || location.pathname + location.search;
    // Moving between papers of the same cluster shouldn't fill the back button's history.
    if (replace || hash === location.hash) history.replaceState(null, '', url);
    else history.pushState(null, '', url);
    route();
  }
  const onRoute = () => { route(); };
  window.addEventListener('popstate', onRoute);
  window.addEventListener('hashchange', onRoute);

  function goHome() {
    flyCamera(homeView(), FLY_MS);
  }

  async function goToCluster(id: number, paperId: string | null) {
    const bubble = bubbleOf.get(id);
    if (!bubble) return;
    routed = id;
    // If a paper's details are about to open on the right, frame the cluster in the space left of them.
    flyCamera(clusterView(bubble, paperId && window.innerWidth >= 1100 ? DETAILS_PANEL_PX : 0), FLY_MS);
    await ensureLoaded(id);
    flushRebuild(); // its papers must be in the graph before we can open one
    if (destroyed || routed !== id || !paperId) return;
    const paper = nodeById.get(paperId);
    if (paper?.kind === 'paper') openDetails(paper);
  }

  // ---- Field filter + time window ----
  // Total papers per year for each field, summed over all clusters (for the histogram).
  const fieldTimelines: Record<string, YearSeries> = {};
  const timelineFor = (key: FieldFilter): YearSeries => {
    if (!key) return overview.timeline;
    if (!fieldTimelines[key]) {
      const count = overview.years.map((_, i) => clusters.reduce((s, c) => s + c.fy[key][i], 0));
      const inf = overview.years.map((_, i) =>
        count[i] ? clusters.reduce((s, c) => s + c.fy[key][i] * c.fi[key][i], 0) / count[i] : 0);
      fieldTimelines[key] = { count, inf };
    }
    return fieldTimelines[key];
  };
  /** The per-year numbers the histogram shows: the focused cluster if there is one, else everything. */
  function histogram(): YearSeries {
    if (focused) {
      const c = focused.cluster;
      return fieldFilter ? { count: c.fy[fieldFilter], inf: c.fi[fieldFilter] } : { count: c.count, inf: c.inf };
    }
    return timelineFor(fieldFilter);
  }
  /** The line under the time slider, e.g. "98,623 of 98,623 papers". */
  function timelineSummary(): string {
    const filterNote = fieldFilter ? ` ${fieldFilter}` : '';
    if (!focused) {
      const inRange = Math.round(bubbleNodes.reduce((s, n) => s + n.inWindow, 0));
      const total = fieldFilter
        ? clusters.reduce((s, c) => s + c.fy[fieldFilter!].reduce((a, b) => a + b, 0), 0)
        : meta.clusteredPapers;
      return `${inRange.toLocaleString()} of ${total.toLocaleString()}${filterNote} papers`;
    }
    const own = loaded.get(focused.id)?.papers.filter((p) => !p.extra) ?? [];
    const matching = own.filter((n) => yearVisible(n) && (!fieldFilter || fieldKey(n.categoryCode) === fieldFilter)).length;
    const what = focused.cluster.inner && focused.cluster.inner !== 'papers' ? focused.cluster.inner : 'top papers';
    return `${matching} of ${own.length} ${what}${filterNote ? ` (${filterNote.trim()})` : ''} shown`;
  }
  /** How many of the focused cluster's own top papers are loaded (0 until its file arrives). */
  const focusedPaperCount = () =>
    focused ? (loaded.get(focused.id)?.papers.filter((p) => !p.extra).length ?? 0) : 0;

  function setFieldFilter(key: string | null) {
    fieldFilter = key === fieldFilter ? null : key;
    applyWindow();
    emit();
  }

  // Called whenever the time window or the field filter changes.
  function applyWindow() {
    sizeBubbles();
    graph.nodeColor(nodeColor);
    openKey = ''; // force the visibility functions to be re-applied
    updateLOD();
  }

  let windowQueued = false;
  /** Move the time window's ends (years, fractional). The slider calls this on every drag step. */
  function setWindow(lo: number, hi: number) {
    windowLo = lo;
    windowHi = hi;
    emit(); // the slider handles move right away; the scene catches up on the next frame
    if (windowQueued) return;
    windowQueued = true;
    frame(() => {
      windowQueued = false;
      applyWindow();
      emit();
    });
  }

  // ---- Paper details panel ----
  function openDetails(n: PaperNode) {
    const prev = detailsFor;
    detailsFor = n;
    updateHalo(prev);
    updateHalo(n);
    refreshHighlight();
    emit();
  }

  function closeDetails() {
    const prev = detailsFor;
    detailsFor = null;
    updateHalo(prev);
    refreshHighlight();
    emit();
  }

  // ---- Citation lines of the selected paper: highlight, hover, follow ----
  // Selecting a paper (clicking it) opens its details and frames its cluster.
  function selectPaper(n: PaperNode) {
    openDetails(n);
    navigate(`#cluster=${n.bubble.id}&paper=${encodeURIComponent(n.id)}`, { replace: routed === n.bubble.id });
  }

  // Work out which lines touch the selected paper and restyle: they go bright and thick, the rest dim.
  function refreshHighlight() {
    attached = new Set();
    if (detailsFor && nodeById.get(detailsFor.id) === detailsFor) {
      for (const l of graph.graphData().links as GraphLink[]) {
        if (endpoint(l.source) === detailsFor || endpoint(l.target) === detailsFor) attached.add(l);
      }
    }
    clearEdgeHover();
    graph.linkColor(linkColor).linkWidth(linkWidth).linkDirectionalArrowLength(arrowLength);
    updateEdgeHover(); // the mouse may be resting on one of the new lines
  }

  // The highlighted, visible line nearest the mouse (within EDGE_PICK_PX), found in screen space so it
  // works for hairlines and at any zoom.
  function pickEdge(clientX: number, clientY: number): GraphLink | null {
    if (!attached.size) return null;
    const rect = canvas.getBoundingClientRect();
    const px = clientX - rect.left;
    const py = clientY - rect.top;
    let best: GraphLink | null = null;
    let bestDistance = EDGE_PICK_PX;
    for (const l of attached) {
      if (!linkVisible(l)) continue;
      const a = endpoint(l.source)!;
      const b = endpoint(l.target)!;
      const d = distanceToSegment(px, py, graph.graph2ScreenCoords(a.x, a.y, 0), graph.graph2ScreenCoords(b.x, b.y, 0));
      if (d < bestDistance) { best = l; bestDistance = d; }
    }
    return best;
  }
  const farEnd = (l: GraphLink) => (endpoint(l.source) === detailsFor ? endpoint(l.target) : endpoint(l.source))!;
  const farEndTitle = (l: GraphLink) => { const n = farEnd(l); return n.kind === 'cluster' ? l.ext!.title : n.title; };

  // Go to the other end of a line: another paper is selected; a line that ends at a neighboring cluster's
  // bubble flies there and shows the paper it leads to.
  function followEdge(l: GraphLink) {
    const other = farEnd(l);
    if (!other) return;
    if (other.kind === 'paper') {
      selectPaper(other);
    } else {
      // The paper is drawn once that cluster opens (it is in the cluster's file even if it isn't a top paper).
      navigate(`#cluster=${l.neighbor}&paper=${encodeURIComponent(l.ext!.id)}`);
      // Safety net for data exported before that was the case: show its details without a node.
      ensureLoaded(l.neighbor!).then(() => {
        flushRebuild();
        if (!destroyed && routed === l.neighbor && !nodeById.has(l.ext!.id)) {
          openDetails({ ...l.ext!, kind: 'paper', bubble: bubbleOf.get(l.neighbor!)! } as unknown as PaperNode);
        }
      });
    }
  }
  function handleEmptyClick(event?: MouseEvent) {
    const l = event && pickEdge(event.clientX, event.clientY);
    if (l) followEdge(l);
    else closeDetails();
  }

  // Hovering a highlighted line shows where it leads (React draws the tooltip from edgeTip).
  // (The library resets the canvas's own cursor every frame, so the cursor is set on its container.)
  function clearEdgeHover() {
    if (edgeHoverNode) {
      const prev = edgeHoverNode;
      edgeHoverNode = null;
      updateHalo(prev);
    }
    if (edgeTip) {
      edgeTip = null;
      emit();
    }
    container.style.cursor = '';
  }
  let lastPointer: { x: number; y: number } | null = null; // where the mouse last was, in window coordinates
  function updateEdgeHover() {
    if (!lastPointer || !attached.size || hovered) { clearEdgeHover(); return; } // a paper under the mouse takes precedence
    const l = pickEdge(lastPointer.x, lastPointer.y);
    if (!l) { clearEdgeHover(); return; }
    const other = farEnd(l);
    if (other !== edgeHoverNode) {
      const prev = edgeHoverNode;
      edgeHoverNode = other;
      updateHalo(prev);
      updateHalo(other);
    }
    edgeTip = {
      verb: endpoint(l.source) === detailsFor ? 'Cites' : 'Cited by',
      title: farEndTitle(l),
      x: Math.min(lastPointer.x + 14, window.innerWidth - 340),
      y: lastPointer.y + 14,
    };
    emit();
    container.style.cursor = 'pointer';
  }
  let edgeHoverQueued = false;
  const onPointerMove = (e: PointerEvent) => {
    lastPointer = { x: e.clientX, y: e.clientY };
    if (e.buttons) { clearEdgeHover(); return; } // dragging the map
    if (edgeHoverQueued) return;
    edgeHoverQueued = true;
    // Two frames later, so the library has updated which paper (if any) is under the mouse.
    frame(() => frame(() => { edgeHoverQueued = false; updateEdgeHover(); }));
  };
  const onPointerLeave = () => { lastPointer = null; clearEdgeHover(); };
  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('pointerleave', onPointerLeave);

  // Esc closes the details panel first; pressing it again flies back out to the whole map.
  const onKeyDown = (e: KeyboardEvent) => {
    if (e.key !== 'Escape' || e.target instanceof HTMLInputElement) return;
    if (detailsFor) closeDetails();
    else if (focused || routed !== null) navigate('');
  };
  window.addEventListener('keydown', onKeyDown);

  // ---- Actions for the panels ----
  /** Highlight a bubble while its search result is under the mouse (null clears it). */
  function setSearchHover(id: number | null) {
    const prev = searchHover;
    searchHover = id === null ? null : (bubbleOf.get(id) ?? null);
    updateHalo(prev);
    updateHalo(searchHover);
  }

  /** Back to the starting state: whole map in view, no time or field filter, nothing open. */
  function resetAll() {
    if (fieldFilter) fieldFilter = null;
    windowLo = T0;
    windowHi = T1;
    applyWindow();
    closeDetails();
    if (routed !== null || location.hash) navigate('');
    else goHome();
    emit();
  }

  function destroy() {
    destroyed = true;
    pendingFrames.forEach(cancelAnimationFrame);
    pendingFrames.clear();
    if (rebuildTimer !== null) clearTimeout(rebuildTimer);
    window.removeEventListener('resize', onResize);
    window.removeEventListener('popstate', onRoute);
    window.removeEventListener('hashchange', onRoute);
    window.removeEventListener('keydown', onKeyDown);
    canvas.removeEventListener('pointerdown', cancelFlight);
    canvas.removeEventListener('wheel', cancelFlight);
    canvas.removeEventListener('pointermove', onPointerMove);
    canvas.removeEventListener('pointerleave', onPointerLeave);
    controls.removeEventListener('change', onControlsChange);
    listeners.clear();
    graph._destructor();
    container.replaceChildren();
    container.style.cursor = '';
  }

  // ---- Start ----
  rebuildGraph();
  sizeBubbles();
  fitToWindow();
  updateLOD();
  route();

  // Handy for poking at things from the browser console (and for the headless browser tests).
  (window as unknown as Record<string, unknown>).graph = graph;
  (window as unknown as Record<string, unknown>).lod = {
    bubbleNodes,
    loaded,
    get attached() { return attached; },
    get selected() { return detailsFor; },
    get nodeById() { return nodeById; },
  };

  return {
    overview,
    fieldHelpers,
    bubbleNodes,
    timeRange: { min: T0, max: T1 },
    subscribe,
    getVersion: () => version,
    // Read-only views of the state the panels display.
    get focused() { return focused; },
    get detailsFor() { return detailsFor; },
    get fieldFilter() { return fieldFilter; },
    get windowLo() { return windowLo; },
    get windowHi() { return windowHi; },
    get edgeTip() { return edgeTip; },
    histogram,
    timelineSummary,
    focusedPaperCount,
    // Actions.
    navigate,
    /** Switch to another map (drill-down maps only). */
    openMap: (mapId: string, cluster?: number) => options.onOpenMap?.(mapId, cluster),
    goHome,
    setFieldFilter,
    setWindow,
    closeDetails,
    setSearchHover,
    resetAll,
    destroy,
  };
}
