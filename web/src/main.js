import ForceGraph3D from '3d-force-graph';
import { resolveOverlaps } from './layout.js';
import {
  BackSide, CanvasTexture, Group, Mesh, MeshBasicMaterial, MeshLambertMaterial, SphereGeometry, SRGBColorSpace, Sprite, SpriteMaterial,
} from 'three';

// ---------------------------------------------------------------------------
// One continuous map. Each cluster ("chunk") is a bubble sized by how many papers it holds.
// Zoom into a bubble (or click it) and the bubble fades to a faint outline while its most
// influential papers appear inside it, so you can pan from cluster to cluster at full detail.
// Citation lines run between papers of different clusters when both are open, and otherwise end
// at the other cluster's bubble.
// Data comes from export_web_data.py: /data/overview.json and /data/clusters/<id>.json
// ---------------------------------------------------------------------------

// ---- Layout constants (tweak these to change how things look) ----
const XY_SCALE = 300;       // overview positions arrive normalized to [-1, 1]; this sets world size
const NODE_PADDING = 0.6;   // minimum empty gap kept between any two circles (world units)
const FIT_PADDING = 1.08;   // >1 leaves a margin around the map when fitting it to the window
const PAN_SLACK = 0.4;      // extra pan room past the map edge, as a fraction of half the visible area
const FLY_MS = 800;         // camera animation time
const LEFT_INSET_PX = 400;  // width the left info panel covers on wide screens (panel + margin)
const BOTTOM_INSET_PX = 150; // height the timeline box covers at the bottom (box + margin)
const DETAILS_PANEL_PX = 412; // width the paper details panel covers on the right (panel + margin)
const MIN_DISTANCE = 10;    // closest the camera can get (small bubbles need to fill the screen)

// Bubbles: radius ~ sqrt(papers), so AREA is proportional to the number of papers.
const BUBBLE_MAX = 30;      // radius of the biggest cluster
const BUBBLE_MIN = 2;       // smallest a bubble can shrink to when the time window / field filter is narrowed
const CLUSTER_FIT = 0.85;   // clicking a bubble zooms until it fills this fraction of the free screen area

// Level of detail, in on-screen pixels of a bubble's radius:
const LOAD_PX = 45;         // start fetching the bubble's papers
const OPEN_PX = 75;         // show the papers (the bubble starts to expand to its full size)
const FULL_PX = 115;        // fully expanded
const FADE_START_PX = 75;   // from here the bubble fades to a faint outline...
const FADE_END_PX = 130;    // ...reached at this size
const BUBBLE_ALPHA = 0.95;
const FADED_ALPHA = 0.1;
const VIEW_MARGIN = 1.15;   // bubbles this far outside the screen still count as "in view"
const MAX_LOADS = 4;        // cluster files fetched at the same time
const MAX_OPEN = 5;         // at most this many bubbles show their papers at once: the ones nearest the middle of the screen
const MAX_LOADED = 14;      // cluster files kept in memory; the least recently open ones are dropped beyond this
const OPEN_HYSTERESIS = 0.8; // an already-open bubble counts as this much closer, so two similar bubbles don't flicker

// Papers inside an open bubble: laid out at a comfortable size, then shrunk to fit the bubble.
const INNER_RADIUS = 190;   // 90% of a cluster's papers fit within this distance of the center
const INNER_MAX = 255;      // outliers are squashed to stay within this distance
const FILL = 0.9;           // papers fill this fraction of the bubble's radius
// Paper size is set by influence RANK (the raw scores of a cluster's top papers are too similar).
const R_MIN = 5;
const R_MAX = 28;
const SIZE_CURVE = 1.5;     // >1 keeps most papers smallish and lets the top few stand out

// Cluster titles drawn on the map. Zoomed out, a bubble's title sits in its middle (split into short lines
// and sized to fit); once the bubble opens it moves above the outline at a constant readable size.
const TITLE_FONT = 44;      // text size on the title's canvas (px); also sets how crisp it looks
const TITLE_LINE = 54;      // line height on that canvas
const TITLE_MIN_PX = 9;     // a centered title only appears once it would be at least this big on screen...
const TITLE_MAX_PX = 24;    // ...and never gets bigger than this
const TITLE_FIT_W = 1.6;    // a centered title may fill this many radii across (the bubble is 2 wide)...
const TITLE_FIT_H = 1.1;    // ...and this many radii tall
const TITLE_MOVE_START_PX = 105; // the title starts moving from the middle to the top at this bubble radius...
const TITLE_MOVE_END_PX = 140;   // ...and is fully at the top at this one
const TITLE_TOP_PX = 15;    // on-screen text size of a title above an open bubble

// Citation lines of the selected paper (the one whose details are open) are highlighted and can be clicked.
const EDGE_PICK_PX = 9;          // a click this close (in screen pixels) to a highlighted line follows it
const EDGE_DIM = 0.3;            // other lines fade to this fraction of their brightness while a paper is selected
const EDGE_HIGHLIGHT = 'rgba(255, 255, 255, 0.95)';
const EDGE_WIDTH_BOOST = 2.4;    // highlighted paper-to-paper lines are this much thicker (and their arrows bigger)

// Outline = a slightly bigger white sphere behind the node ("halo").
const HALO_WIDTH = 0.18;    // outline thickness as a fraction of the node's radius
const HALO_MIN = 0.05;      // minimum outline thickness in world units...
const HALO_MIN_PX = 3;      // ...and on screen, so even the smallest paper's outline is easy to see
const UNIT_SPHERE = new SphereGeometry(1, 20, 14); // shared by every bubble and halo; scaled per node
// BackSide renders only the inside of the shell, so it never covers the node in front of it.
const HALO_MATERIAL = new MeshBasicMaterial({ color: 0xffffff, side: BackSide });

// One color per arXiv field (the exporter lists the biggest fields first; the rest are "other").
const FIELD_COLORS = [
  '#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948',
  '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac', '#86bcb6', '#d37295',
];
const OTHER_COLOR = '#6c7386';
const DIMMED_COLOR = 'rgba(95, 100, 115, 0.3)'; // papers outside the selected field
const PAPER_LIGHTEN = 0.4;  // papers are their cluster's color mixed this far toward white

const $ = (id) => document.getElementById(id);
const escapeHtml = (s) =>
  String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const stripPrefix = (title) => title.replace(/^Cluster \d+:\s*/, ''); // "Cluster 5: Foo / Bar" -> "Foo / Bar"
const swatch = (color) => `<span class="swatch" style="background:${color}"></span>`;
const fetchJson = async (url) => {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return res.json();
};
const withAlpha = (hex, a) => {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
};
const smoothstep = (a, b, x) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};
const lerp = (a, b, t) => a + (b - a) * t;
// Mix a #rrggbb color toward white (amount 0..1).
const lighten = (hex, amount) => {
  const n = parseInt(hex.slice(1), 16);
  const mix = (c) => Math.round(lerp(c, 255, amount));
  return `#${[(n >> 16) & 255, (n >> 8) & 255, n & 255].map((c) => mix(c).toString(16).padStart(2, '0')).join('')}`;
};
const noRaycast = () => {};
// Scale the opacity of an "rgba(r, g, b, a)" color.
const dim = (color, factor) => color.replace(/rgba\(([^)]*),\s*([\d.]+)\)$/, (_, rgb, a) => `rgba(${rgb}, ${(a * factor).toFixed(2)})`);
// Shortest distance from point (px, py) to the segment a-b (all in screen pixels).
function distanceToSegment(px, py, a, b) {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const len2 = dx * dx + dy * dy;
  const t = len2 ? Math.min(1, Math.max(0, ((px - a.x) * dx + (py - a.y) * dy) / len2)) : 0;
  return Math.hypot(px - (a.x + t * dx), py - (a.y + t * dy));
}

// ---- Title text (drawn on canvases, shown as sprites: flat images that always face the camera) ----
const TITLE_FAMILY = 'system-ui, -apple-system, "Segoe UI", sans-serif';
const measureCtx = document.createElement('canvas').getContext('2d');
measureCtx.font = `600 ${TITLE_FONT}px ${TITLE_FAMILY}`;
const TITLE_PAD = 14;
function titleDims(lines) {
  return {
    w: Math.ceil(Math.max(...lines.map((l) => measureCtx.measureText(l).width))) + TITLE_PAD * 2,
    h: lines.length * TITLE_LINE + TITLE_PAD * 2,
  };
}
function makeTitleSprite(lines, dims) {
  const canvas = document.createElement('canvas');
  canvas.width = dims.w;
  canvas.height = dims.h;
  const ctx = canvas.getContext('2d');
  ctx.font = `600 ${TITLE_FONT}px ${TITLE_FAMILY}`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.lineJoin = 'round';
  ctx.lineWidth = TITLE_FONT * 0.2;
  ctx.strokeStyle = 'rgba(8, 10, 16, 0.92)'; // a dark outline keeps the text readable on any color
  ctx.fillStyle = '#ffffff';
  lines.forEach((line, i) => {
    const y = TITLE_PAD + TITLE_LINE * (i + 0.5) + 2;
    ctx.strokeText(line, dims.w / 2, y);
    ctx.fillText(line, dims.w / 2, y);
  });
  const texture = new CanvasTexture(canvas);
  texture.colorSpace = SRGBColorSpace;
  // Always drawn on top, even where an open bubble's papers are.
  const sprite = new Sprite(new SpriteMaterial({ map: texture, transparent: true, depthTest: false, depthWrite: false }));
  sprite.renderOrder = 20;
  sprite.raycast = noRaycast; // text must not block hovering or clicking what's underneath
  return sprite;
}

async function main() {
  const overview = await fetchJson('/data/overview.json');
  const { meta, clusters, fields } = overview;
  const clusterById = new Map(clusters.map((c) => [c.id, c]));

  // ---- arXiv fields ----
  const fieldIndex = new Map(fields.map((f, i) => [f.code, i]));
  const fieldKey = (code) => (fieldIndex.has(code) ? code : 'other'); // anything outside the biggest fields
  const fieldColor = (key) => {
    const i = fieldIndex.get(key);
    return i !== undefined && i < FIELD_COLORS.length ? FIELD_COLORS[i] : OTHER_COLOR;
  };
  const fieldName = (key) => (key === 'other' ? 'Other fields' : fields[fieldIndex.get(key)]?.name ?? key);
  const clusterColor = (c) => fieldColor(c.field);

  // ---- State ----
  let nodes = [];           // everything the graph is drawing: all bubbles + the papers of every loaded cluster
  let nodeById = new Map();
  let hovered = null;
  let detailsFor = null;    // the paper shown in the details panel
  let searchHover = null;   // the bubble whose search result the mouse is over
  let focused = null;       // the open bubble nearest the middle of the screen, if any
  let routed = null;        // cluster id the URL hash asks for
  let fieldFilter = null;   // null, an arXiv field code, or 'other'
  let attached = new Set(); // citation lines touching the selected paper
  let edgeHoverNode = null; // the paper/bubble at the far end of the line under the mouse
  const loaded = new Map(); // cluster id -> { papers, links, scale }
  const loading = new Set();
  const failed = new Set();

  // ---- Time window (set by the slider at the bottom) ----
  const T0 = meta.yearMin;
  const T1 = meta.yearMax;
  let windowLo = T0;
  let windowHi = T1;
  const windowIsFull = () => windowLo <= T0 && windowHi >= T1;

  // Per-year paper counts of a cluster for the current field filter.
  const countsOf = (c) => (fieldFilter ? c.fy[fieldFilter] : c.count);
  // How many papers of those counts fall inside the window. Counts are per whole year, so a year
  // that is only partly inside the window counts proportionally.
  function inWindow(counts) {
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

  const yearVisible = (n) => (n.year == null ? windowIsFull() : n.year >= windowLo && n.year <= windowHi);
  const isVisible = (n) => {
    if (n.kind === 'cluster') return n.inWindow >= 0.5;
    return n.bubble.open && yearVisible(n); // a paper shows only while its bubble is open
  };
  // While a graph is being built the library may still hold a link's ends as plain ids rather than
  // node objects, so link functions look the nodes up themselves.
  const endpoint = (e) => (typeof e === 'object' ? e : nodeById.get(e));
  function linkVisible(l) {
    const a = endpoint(l.source);
    const b = endpoint(l.target);
    if (!a || !b || !isVisible(a) || !isVisible(b)) return false;
    // The lines between bubbles would run across the papers of an open bubble, so they hide.
    if (l.kind === 'cluster') return !a.open && !b.open;
    return true;
  }

  // ---- Bubbles (built once; positions never change) ----
  const maxSize = Math.max(...clusters.map((c) => c.size));
  const bubbleNodes = clusters.map((c) => {
    const x = c.x * XY_SCALE;
    const y = c.y * XY_SCALE;
    const fullRadius = BUBBLE_MAX * Math.sqrt(c.size / maxSize);
    const title = stripPrefix(c.title);
    const lines = title.split(' / '); // the clustering names a cluster with a few keywords separated by " / "
    return {
      id: c.id, kind: 'cluster', cluster: c, x, y, z: 0, fx: x, fy: y, fz: 0,
      fullRadius, shrunk: fullRadius, radius: fullRadius, inWindow: c.size, open: false,
      title: { lines, centerDims: titleDims(lines), oneLine: [title], topDims: titleDims([title]) },
    };
  });
  // UMAP knows nothing about bubble sizes, so nudge overlapping ones apart, then re-pin.
  resolveOverlaps(bubbleNodes, { padding: NODE_PADDING });
  bubbleNodes.forEach((n) => { n.fx = n.x; n.fy = n.y; });
  const bubbleOf = new Map(bubbleNodes.map((n) => [n.id, n]));
  const maxWeight = Math.max(...overview.links.map((l) => l.weight));
  const bubbleLinks = overview.links.map((l) => ({ ...l, kind: 'cluster', color: `rgba(255, 255, 255, ${(0.14 + 0.5 * Math.sqrt(l.weight / maxWeight)).toFixed(2)})` }));

  // Narrowing the time window or picking a field shrinks bubbles (never grows them, so no new
  // overlaps appear). A bubble's area is its share of papers inside the window that match the field.
  function sizeBubbles() {
    bubbleNodes.forEach((n) => {
      n.inWindow = inWindow(countsOf(n.cluster));
      n.shrunk = Math.max(BUBBLE_MIN, n.fullRadius * Math.sqrt(n.inWindow / n.cluster.size));
    });
  }

  // ---- Node + link appearance ----
  function nodeColor(n) {
    if (n.kind === 'cluster') return clusterColor(n.cluster);
    // Papers outside the selected field fade out; the rest take their cluster's color.
    if (fieldFilter && fieldKey(n.categoryCode) !== fieldFilter) return DIMMED_COLOR;
    return lighten(clusterColor(n.bubble.cluster), PAPER_LIGHTEN); // lighter than the bubble, so they stand out inside it
  }
  let pxPerUnitNow = 1; // current zoom, kept up to date by updateLOD
  const haloRadius = (n) => n.radius + Math.max(HALO_MIN, n.radius * HALO_WIDTH, HALO_MIN_PX / pxPerUnitNow);
  const updateHalo = (n) => {
    if (!n?.halo) return;
    n.halo.visible = n === hovered || n === detailsFor || n === searchHover || n === edgeHoverNode;
    n.halo.scale.setScalar(haloRadius(n));
  };
  function makeHalo(n) {
    const halo = new Mesh(UNIT_SPHERE, HALO_MATERIAL);
    halo.raycast = noRaycast; // the halo shouldn't count as part of the node for hover/click
    n.halo = halo;
    updateHalo(n);            // the library rebuilds a node's objects when it re-enters the window
    return halo;
  }
  // A bubble is our own object, not the library's sphere, so we can fade it as you zoom in:
  // it must not write to the depth buffer, or it would hide the papers inside it.
  function makeBubble(n) {
    const group = new Group();
    const sphere = new Mesh(UNIT_SPHERE, new MeshLambertMaterial({
      color: clusterColor(n.cluster), transparent: true, opacity: BUBBLE_ALPHA, depthWrite: false,
    }));
    sphere.renderOrder = -1; // drawn before the papers
    sphere.scale.setScalar(n.radius);
    n.sphere = sphere;
    n.group = group;
    n.titleCenter = n.titleTop = null; // (re)built on demand: the library recreates this object when the bubble re-enters the time window
    group.add(sphere);
    group.add(makeHalo(n));
    return group;
  }
  const nodeObject = (n) => (n.kind === 'cluster' ? makeBubble(n) : makeHalo(n));

  function nodeLabel(n) {
    if (n.kind === 'cluster') {
      const c = n.cluster;
      const filtered = fieldFilter !== null;
      const inRange = Math.round(n.inWindow);
      const note = windowIsFull() && !filtered ? '' : ` ${filtered ? `${fieldName(fieldFilter)} ` : ''}in range (of ${c.size.toLocaleString()})`;
      return `
        <b>${escapeHtml(stripPrefix(c.title))}</b><br/>
        ${inRange.toLocaleString()} papers${note}<br/>
        Mostly ${escapeHtml(c.categoryName)} (${Math.round(c.categoryShare * 100)}%)<br/>
        Top paper: ${escapeHtml(c.topPaper)}<br/>
        <span style="color:#8b92a5">Click or zoom in to open</span>`;
    }
    return `
      <b>${escapeHtml(n.title)}</b><br/>
      ${escapeHtml(n.date ?? 'unknown date')} · ${escapeHtml(n.category)}<br/>
      ${n.citations.toLocaleString()} citations · influence ${n.influence.toFixed(2)}<br/>
      <span style="color:#8b92a5">${escapeHtml(stripPrefix(n.bubble.cluster.title))}</span>`;
  }

  // Lines between bubbles light up for the hovered bubble; the rest stay faint.
  function linkColor(l) {
    if (attached.has(l)) return EDGE_HIGHLIGHT;
    let color;
    if (l.kind === 'cluster') {
      const touches = hovered && (endpoint(l.source) === hovered || endpoint(l.target) === hovered);
      color = touches ? 'rgba(255, 255, 255, 0.95)' : l.color;
    } else {
      color = l.color ?? 'rgba(255, 255, 255, 0.55)'; // citations to other clusters take that cluster's color
    }
    return attached.size ? dim(color, EDGE_DIM) : color; // while a paper is selected, its lines stand out
  }
  // Paper-to-paper lines scale with their cluster (their papers are tiny in world units); lines between
  // bubbles, or from a paper to a far-away bubble, are hairlines so they look right at any zoom.
  const linkWidth = (l) => (l.kind === 'paper' ? 2 * l.scale * (attached.has(l) ? EDGE_WIDTH_BOOST : 1) : 0);
  const arrowLength = (l) => (l.kind === 'paper' ? 8 * l.scale * (attached.has(l) ? EDGE_WIDTH_BOOST : 1) : 0);

  // ---- The graph ----
  // Orbit controls let us switch rotate off and clamp zoom (the default trackball can't).
  const graph = ForceGraph3D({ controlType: 'orbit' })($('graph'))
    .backgroundColor('#0b0d12')
    .nodeColor(nodeColor)
    .nodeVisibility(isVisible)
    .linkVisibility(linkVisible)
    // Opaque nodes, so the white halo behind them doesn't show through.
    .nodeOpacity(1)
    .nodeThreeObject(nodeObject)
    .nodeThreeObjectExtend((n) => n.kind !== 'cluster') // papers: keep the default sphere and add a halo
    // The library's radius = cbrt(nodeVal) * nodeRelSize, so with relSize 1, nodeVal = radius³.
    .nodeVal((n) => n.radius ** 3)
    .nodeRelSize(1)
    .nodeResolution(12)
    .enableNodeDrag(false)
    .nodeLabel(nodeLabel)
    .linkColor(linkColor)
    .linkWidth(linkWidth)
    .linkDirectionalArrowLength(arrowLength) // arrows only for paper-to-paper citations
    .linkDirectionalArrowRelPos(1)
    .onNodeHover((n, prev) => {
      hovered = n;
      updateHalo(prev);
      updateHalo(n);
      if (n?.kind === 'cluster' || prev?.kind === 'cluster') graph.linkColor(linkColor); // re-apply so the lines re-color
    })
    .onNodeClick((n) => {
      if (n.kind === 'cluster') navigate(`#cluster=${n.id}`);
      else selectPaper(n);
    })
    // A click on empty space follows a highlighted line if one is right there; otherwise it deselects.
    .onBackgroundClick((event) => handleEmptyClick(event))
    .onLinkClick((link, event) => handleEmptyClick(event))
    // Positions are fixed, so there is nothing for the physics engine to do.
    .warmupTicks(0)
    .cooldownTicks(0);

  // Flat map: look straight down the z-axis. No rotating; drag to pan, scroll to zoom toward the pointer.
  const controls = graph.controls();
  controls.enableRotate = false;
  controls.enablePan = true;
  controls.screenSpacePanning = true; // pan moves in the screen plane (= the map plane here)
  controls.zoomToCursor = true;       // scrolling zooms toward the mouse, so you can dive into any bubble
  // Three.js MOUSE / TOUCH enum values: MOUSE.PAN = 2, MOUSE.DOLLY = 1, TOUCH.PAN = 1, TOUCH.DOLLY_PAN = 2
  controls.mouseButtons = { LEFT: 2, MIDDLE: 1, RIGHT: 2 };
  controls.touches = { ONE: 1, TWO: 2 };

  const camera = graph.camera();
  const tanHalfFov = () => Math.tan((camera.fov * Math.PI) / 360);
  const aspect = () => window.innerWidth / window.innerHeight; // camera.aspect updates a tick late on resize

  // Bounding box of the bubble map (circle edges included); camera limits are based on it.
  let bounds = { cx: 0, cy: 0, halfW: 1, halfH: 1 };
  {
    const minX = Math.min(...bubbleNodes.map((n) => n.x - n.fullRadius));
    const maxX = Math.max(...bubbleNodes.map((n) => n.x + n.fullRadius));
    const minY = Math.min(...bubbleNodes.map((n) => n.y - n.fullRadius));
    const maxY = Math.max(...bubbleNodes.map((n) => n.y + n.fullRadius));
    bounds = { cx: (minX + maxX) / 2, cy: (minY + maxY) / 2, halfW: (maxX - minX) / 2, halfH: (maxY - minY) / 2 };
  }

  let flight = null;

  // The info panel covers the left edge and the timeline covers the bottom, so views are fitted
  // into the free space (wide/tall screens only; on small windows the panels just overlay the map).
  const leftInset = () => (window.innerWidth >= 1100 ? LEFT_INSET_PX : 0);
  const bottomInset = () => (window.innerHeight >= 600 ? BOTTOM_INSET_PX : 0);
  const freeSize = () => ({ w: window.innerWidth - leftInset(), h: window.innerHeight - bottomInset() });
  // A camera position that puts world point (x, y) in the middle of the free area at `distance`.
  // `rightPx` reserves extra room on the right (for the paper details panel).
  function viewAt(x, y, distance, rightPx = 0) {
    const unitsPerPx = (2 * distance * tanHalfFov()) / window.innerHeight;
    // The free area's center is right of / above the screen center, so the camera aims left of / below the point.
    return { x: x - ((leftInset() - rightPx) / 2) * unitsPerPx, y: y - (bottomInset() / 2) * unitsPerPx, z: distance };
  }
  // The view that shows the whole map: as close as possible while everything still fits.
  function homeView() {
    const t = tanHalfFov();
    const { w, h } = freeSize();
    const distance = FIT_PADDING * Math.max(bounds.halfH / (t * (h / window.innerHeight)), bounds.halfW / (t * (w / window.innerHeight)));
    return viewAt(bounds.cx, bounds.cy, distance);
  }
  // The view that makes one bubble fill most of the free area.
  function clusterView(bubble, rightPx = 0) {
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
  function flyCamera(to, ms) {
    const from = { x: camera.position.x, y: camera.position.y, z: camera.position.z };
    const start = performance.now();
    const mine = (flight = {});
    const step = (now) => {
      if (flight !== mine) return; // cancelled or superseded
      const p = Math.min(1, (now - start) / ms);
      const e = 1 - Math.pow(1 - p, 3); // ease-out cubic
      const x = lerp(from.x, to.x, e);
      const y = lerp(from.y, to.y, e);
      const z = from.z * Math.pow(to.z / from.z, e);
      camera.position.set(x, y, z);
      controls.target.set(x, y, 0);
      scheduleLOD();
      if (p < 1) requestAnimationFrame(step);
      else flight = null;
    };
    requestAnimationFrame(step);
  }
  const cancelFlight = () => { flight = null; };
  const canvas = graph.renderer().domElement;
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
    const cx = bounds.cx - (leftInset() / window.innerWidth) * viewHalfW;   // same shifts as viewAt()
    const cy = bounds.cy - (bottomInset() / window.innerHeight) * viewHalfH;
    const x = Math.min(cx + roomX, Math.max(cx - roomX, controls.target.x));
    const y = Math.min(cy + roomY, Math.max(cy - roomY, controls.target.y));
    if (x === controls.target.x && y === controls.target.y) return;
    // Keep the camera directly above its target so the view stays flat.
    camera.position.x = controls.target.x = x;
    camera.position.y = controls.target.y = y;
  }
  controls.addEventListener('change', () => { clampPan(); scheduleLOD(); }); // fires on every pan and zoom
  window.addEventListener('resize', () => { fitToWindow(); clampPan(); scheduleLOD(); });

  // The library's built-in hint still describes rotating, which is turned off.
  const navHint = document.querySelector('.scene-nav-info');
  if (navHint) navHint.textContent = 'Left or Right-click: pan, Mouse-wheel/middle-click: zoom toward the pointer';

  // ---- Papers inside bubbles ----
  // Lay out one cluster's papers inside its bubble. Returns the paper nodes and their links.
  function buildCluster(bubble, data) {
    // The cluster's own top papers set the layout and the sizes. "Extra" papers (included because another
    // cluster's line leads to them) are placed in the same frame, at the smallest size.
    const core = data.nodes.filter((p) => !p.extra);
    // Size by influence rank (0 = most influential).
    const ranked = [...core].sort((a, b) => b.influence - a.influence);
    const rank = new Map(ranked.map((p, i) => [p.id, i]));

    // Center on the median paper and scale so 90% of papers fill INNER_RADIUS; the few far outliers
    // are squashed inward, so the dense core stays readable.
    const median = (values) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
    const mx = median(core.map((p) => p.x));
    const my = median(core.map((p) => p.y));
    const dists = core.map((p) => Math.hypot(p.x - mx, p.y - my)).sort((a, b) => a - b);
    const r90 = dists[Math.floor(dists.length * 0.9)] || 1;
    const place = (p) => {
      const dx = p.x - mx;
      const dy = p.y - my;
      const r = Math.hypot(dx, dy);
      const scaled = (r / r90) * INNER_RADIUS;
      const shown = scaled <= INNER_RADIUS ? scaled : Math.min(INNER_RADIUS + (scaled - INNER_RADIUS) * 0.25, INNER_MAX);
      const k = r ? shown / r : 0;
      return { x: dx * k, y: dy * k };
    };
    const papers = data.nodes.map((p) => {
      const pct = p.extra ? 0 : ranked.length > 1 ? 1 - rank.get(p.id) / (ranked.length - 1) : 1;
      const { x, y } = place(p);
      return { ...p, kind: 'paper', x, y, radius: R_MIN + (R_MAX - R_MIN) * Math.pow(pct, SIZE_CURVE) };
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
    const links = data.links.map((l) => {
      if (!l.cross) return { source: l.source, target: l.target };
      const ext = outside.get(l.source) ?? outside.get(l.target);
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
    const reuse = (entry, key, make) => {
      let link = entry.linkCache.get(key);
      if (!link) entry.linkCache.set(key, (link = make()));
      return link;
    };
    const links = [...bubbleLinks];
    const seen = new Set();
    for (const entry of loaded.values()) {
      for (const l of entry.links) {
        if (!l.cross) {
          links.push(reuse(entry, `i|${l.source}|${l.target}`, () => ({ source: l.source, target: l.target, kind: 'paper', scale: entry.scale })));
          continue;
        }
        const color = withAlpha(clusterColor(clusterById.get(l.neighbor)), 0.7);
        const stubColor = withAlpha(clusterColor(clusterById.get(l.neighbor)), 0.4);
        if (nodeById.has(l.outsideId)) {
          // Both papers are on screen (the other cluster is loaded too): a real paper-to-paper line.
          const key = `${l.source}>${l.target}`;
          if (seen.has(key)) continue; // the same citation is listed in both clusters' files
          seen.add(key);
          links.push({ source: l.source, target: l.target, kind: 'paper', cross: true, color, scale: entry.scale });
        } else {
          // The other end isn't loaded (or isn't one of its top papers): end the line at that cluster's bubble.
          const mine = l.source === l.outsideId ? l.target : l.source;
          const info = { kind: 'stub', color: stubColor, scale: entry.scale, neighbor: l.neighbor, ext: l.ext };
          links.push(reuse(entry, `s|${l.source}|${l.target}`, () => (l.source === l.outsideId
            ? { source: l.neighbor, target: mine, ...info }
            : { source: mine, target: l.neighbor, ...info })));
        }
      }
    }
    graph.graphData({ nodes, links });
    refreshHighlight();
  }

  // Several clusters often finish loading together; rebuild the scene once for all of them.
  let rebuildTimer = null;
  function scheduleRebuild() {
    if (rebuildTimer === null) rebuildTimer = setTimeout(flushRebuild, 60);
  }
  function flushRebuild() {
    if (rebuildTimer === null) return;
    clearTimeout(rebuildTimer);
    rebuildTimer = null;
    rebuildGraph();
    scheduleLOD();
  }

  // Keep memory and scene size bounded: drop the least recently open clusters beyond MAX_LOADED.
  function evictOld(keepId) {
    while (loaded.size > MAX_LOADED) {
      let oldest = null;
      for (const [id, entry] of loaded) {
        if (id === keepId || bubbleOf.get(id).open || detailsFor?.bubble.id === id) continue;
        if (!oldest || entry.lastUsed < oldest.entry.lastUsed) oldest = { id, entry };
      }
      if (!oldest) return;
      loaded.delete(oldest.id);
    }
  }

  // Fetch a cluster's papers (once). Resolves when they are loaded.
  const pending = new Map();
  function ensureLoaded(id) {
    if (loaded.has(id)) return Promise.resolve(loaded.get(id));
    if (!pending.has(id)) {
      loading.add(id);
      pending.set(id, fetchJson(`/data/clusters/${id}.json`)
        .then((data) => {
          loaded.set(id, buildCluster(bubbleOf.get(id), data));
          evictOld(id);
          scheduleRebuild();
          return loaded.get(id);
        })
        .catch((err) => { console.error(err); failed.add(id); })
        .finally(() => { loading.delete(id); pending.delete(id); scheduleLOD(); }));
    }
    return pending.get(id);
  }

  // ---- Level of detail ----
  // How big a bubble looks on screen decides how much of it we show: far away it is a solid
  // bubble; up close it is a faint outline around its papers.
  let lodQueued = false;
  function scheduleLOD() {
    if (lodQueued) return;
    lodQueued = true;
    requestAnimationFrame(() => { lodQueued = false; updateLOD(); });
  }

  // Show a bubble's title: in its middle while it is closed, above its outline once it opens.
  function updateTitle(n, pxPerUnit, view) {
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
      y = Math.min(y, view.top - n.y - margin - h / 2);   // stick to the top of the screen if the outline is above it
      y = Math.max(y, -outline + h / 2);                  // ...but never below the bubble
      const minX = view.left - n.x + w / 2 + margin;
      const maxX = view.right - n.x - w / 2 - margin;
      let x = minX <= maxX ? Math.min(Math.max(0, minX), maxX) : 0; // keep it on screen unless it is wider than the view
      const span = outline - w / 2;                       // ...and above the bubble horizontally
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
    const tan = tanHalfFov();
    const pxPerUnit = (pxPerUnitNow = window.innerHeight / (2 * camera.position.z * tan));
    const halfH = camera.position.z * tan * VIEW_MARGIN;
    const halfW = halfH * aspect();
    // The middle of the free screen area (not hidden behind the panels), in world units.
    const fx = camera.position.x + (leftInset() / 2) / pxPerUnit;
    const fy = camera.position.y + (bottomInset() / 2) / pxPerUnit;

    // Candidates: bubbles on screen that are big enough to matter, nearest the middle first.
    const candidates = [];
    for (const n of bubbleNodes) {
      n.px = n.fullRadius * pxPerUnit;
      const inView = (n.inView = Math.abs(n.x - camera.position.x) < halfW + n.fullRadius && Math.abs(n.y - camera.position.y) < halfH + n.fullRadius);
      if (inView && n.px >= LOAD_PX && n.inWindow >= 0.5) {
        candidates.push({ n, d: Math.hypot(n.x - fx, n.y - fy) * (n.open ? OPEN_HYSTERESIS : 1) });
      }
    }
    candidates.sort((a, b) => a.d - b.d);

    // Only the MAX_OPEN nearest big-enough bubbles may open. A few more are fetched ahead of time,
    // so panning to the next bubble doesn't wait on the network.
    const open = new Set();
    let slots = MAX_OPEN;
    candidates.forEach(({ n }, rank) => {
      const wantsOpen = n.px >= OPEN_PX && slots > 0;
      if (wantsOpen) slots--;
      if ((wantsOpen || rank < MAX_OPEN + 2) && !loaded.has(n.id) && !failed.has(n.id) && !loading.has(n.id) && loading.size < MAX_LOADS) ensureLoaded(n.id);
      if (wantsOpen && loaded.has(n.id)) open.add(n.id);
    });

    // The part of the map visible between the panels, in world units (for keeping titles on screen).
    const upp = 1 / pxPerUnit;
    const view = {
      left: camera.position.x - halfW / VIEW_MARGIN + leftInset() * upp,
      right: camera.position.x + halfW / VIEW_MARGIN,
      top: camera.position.y + halfH / VIEW_MARGIN,
      bottom: camera.position.y - halfH / VIEW_MARGIN + bottomInset() * upp,
    };

    // While flying to a cluster, open it straight away (it may still look tiny): the paper we are heading to
    // is then already drawn, selected and outlined by the time the camera arrives.
    if (flight && routed !== null && loaded.has(routed) && bubbleOf.get(routed).inWindow >= 0.5 && !open.has(routed)) {
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
      if (n.open) loaded.get(n.id).lastUsed = now;

      // Expand from the (possibly shrunken) size to the full size as it opens, then fade out.
      n.radius = Math.max(BUBBLE_MIN, lerp(n.shrunk, n.fullRadius, n.open ? smoothstep(OPEN_PX, FULL_PX, n.px) : 0));
      if (n.sphere) {
        n.sphere.scale.setScalar(n.radius);
        n.sphere.material.opacity = n.open ? lerp(BUBBLE_ALPHA, FADED_ALPHA, smoothstep(FADE_START_PX, FADE_END_PX, n.px)) : BUBBLE_ALPHA;
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
      updateChrome();
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
  function navigate(hash, { replace = false } = {}) {
    const url = hash || location.pathname + location.search;
    // Moving between papers of the same cluster shouldn't fill the back button's history.
    if (replace || hash === location.hash) history.replaceState(null, '', url);
    else history.pushState(null, '', url);
    route();
  }
  window.addEventListener('popstate', route);
  window.addEventListener('hashchange', route);

  function goHome() {
    flyCamera(homeView(), FLY_MS);
  }

  async function goToCluster(id, paperId) {
    const bubble = bubbleOf.get(id);
    if (!bubble) return;
    routed = id;
    // If a paper's details are about to open on the right, frame the cluster in the space left of them.
    flyCamera(clusterView(bubble, paperId && window.innerWidth >= 1100 ? DETAILS_PANEL_PX : 0), FLY_MS);
    await ensureLoaded(id);
    flushRebuild(); // its papers must be in the graph before we can open one
    if (routed !== id || !paperId) return;
    const paper = nodeById.get(paperId);
    if (paper?.kind === 'paper') openDetails(paper);
  }

  // ---- Field filter + time window ----
  // Total papers per year for each field, summed over all clusters (for the histogram).
  const fieldTimelines = {};
  const timelineFor = (key) => {
    if (!key) return overview.timeline;
    if (!fieldTimelines[key]) {
      const count = overview.years.map((_, i) => clusters.reduce((s, c) => s + c.fy[key][i], 0));
      const inf = overview.years.map((_, i) => (count[i] ? clusters.reduce((s, c) => s + c.fy[key][i] * c.fi[key][i], 0) / count[i] : 0));
      fieldTimelines[key] = { count, inf };
    }
    return fieldTimelines[key];
  };
  const histogramData = () => {
    if (focused) {
      const c = focused.cluster;
      return fieldFilter ? { count: c.fy[fieldFilter], inf: c.fi[fieldFilter] } : { count: c.count, inf: c.inf };
    }
    return timelineFor(fieldFilter);
  };

  function setFieldFilter(key) {
    fieldFilter = key === fieldFilter ? null : key;
    document.querySelectorAll('#legend .chip').forEach((chip) => chip.classList.toggle('active', chip.dataset.field === fieldFilter));
    $('legend').classList.toggle('filtering', fieldFilter !== null);
    applyWindow();
    timeline.setData(histogramData());
    timeline.refresh();
  }

  // Called whenever the time window or the field filter changes.
  function applyWindow() {
    sizeBubbles();
    graph.nodeColor(nodeColor);
    openKey = ''; // force the visibility functions to be re-applied
    updateLOD();
  }

  // ---- Panels that depend on what's in focus ----
  function updateChrome() {
    if (!focused) {
      $('stats').textContent =
        `${meta.clusterCount} clusters · ${meta.clusteredPapers.toLocaleString()} clustered papers · ` +
        `${meta.noisePapers.toLocaleString()} unclustered (not shown)`;
      $('hint').textContent =
        'Bubble size = papers in the cluster · color = main arXiv field · lines = citations between clusters. ' +
        'Scroll to zoom toward the pointer (or click a bubble) to see inside it; drag to pan.';
      $('selection').hidden = true;
    } else {
      const c = focused.cluster;
      $('stats').textContent = `${c.size.toLocaleString()} papers in this cluster`;
      $('hint').textContent =
        `Showing its ${loaded.get(c.id).papers.filter((p) => !p.extra).length} most influential papers (size = influence rank). Arrows point to the ` +
        'cited paper; lines to other clusters end at their bubble until you zoom into it. Drag to pan to a neighbor.';
      $('selection').innerHTML =
        `<button id="back" type="button">← All clusters</button>` +
        `<div class="banner-title">${swatch(clusterColor(c))}${escapeHtml(stripPrefix(c.title))}</div>`;
      $('back').addEventListener('click', () => navigate(''));
      $('selection').hidden = false;
    }
    timeline.setData(histogramData());
    timeline.refresh();
  }

  // ---- Paper details panel ----
  function openDetails(n) {
    const prev = detailsFor;
    detailsFor = n;
    const cluster = n.bubble.cluster;
    $('details-title').textContent = n.title;
    $('details-meta').textContent = `${n.date ?? 'Unknown date'} · ${n.category}`;
    $('details-stats').innerHTML =
      `<div><b>${n.citations.toLocaleString()}</b><span>citations</span></div>` +
      `<div><b>${n.influence.toFixed(2)}</b><span>influence</span></div>`;
    $('details-cluster').innerHTML = swatch(clusterColor(cluster)) + escapeHtml(stripPrefix(cluster.title));
    const MAX_AUTHORS = 8;
    const extra = n.authors.length - MAX_AUTHORS;
    $('details-authors').textContent = (n.authors.slice(0, MAX_AUTHORS).join(', ') || 'Unknown') + (extra > 0 ? ` and ${extra} more` : '');
    $('details-abstract').textContent = n.abstract || 'No abstract available.';
    $('details-link').href = `https://arxiv.org/abs/${n.arxivId}`;
    $('details').hidden = false;
    $('details').scrollTop = 0;
    updateHalo(prev);
    updateHalo(n);
    refreshHighlight();
  }

  function closeDetails() {
    const prev = detailsFor;
    detailsFor = null;
    $('details').hidden = true;
    updateHalo(prev);
    refreshHighlight();
  }

  // ---- Citation lines of the selected paper: highlight, hover, follow ----
  // Selecting a paper (clicking it) opens its details and frames its cluster.
  function selectPaper(n) {
    openDetails(n);
    navigate(`#cluster=${n.bubble.id}&paper=${encodeURIComponent(n.id)}`, { replace: routed === n.bubble.id });
  }

  // Work out which lines touch the selected paper and restyle: they go bright and thick, the rest dim.
  function refreshHighlight() {
    attached = new Set();
    if (detailsFor && nodeById.get(detailsFor.id) === detailsFor) {
      for (const l of graph.graphData().links) {
        if (endpoint(l.source) === detailsFor || endpoint(l.target) === detailsFor) attached.add(l);
      }
    }
    clearEdgeHover();
    graph.linkColor(linkColor).linkWidth(linkWidth).linkDirectionalArrowLength(arrowLength);
    updateEdgeHover(); // the mouse may be resting on one of the new lines
  }

  // The highlighted, visible line nearest the mouse (within EDGE_PICK_PX), found in screen space so it
  // works for hairlines and at any zoom.
  function pickEdge(clientX, clientY) {
    if (!attached.size) return null;
    const rect = canvas.getBoundingClientRect();
    const px = clientX - rect.left;
    const py = clientY - rect.top;
    let best = null;
    let bestDistance = EDGE_PICK_PX;
    for (const l of attached) {
      if (!linkVisible(l)) continue;
      const a = endpoint(l.source);
      const b = endpoint(l.target);
      const d = distanceToSegment(px, py, graph.graph2ScreenCoords(a.x, a.y, 0), graph.graph2ScreenCoords(b.x, b.y, 0));
      if (d < bestDistance) { best = l; bestDistance = d; }
    }
    return best;
  }
  const farEnd = (l) => (endpoint(l.source) === detailsFor ? endpoint(l.target) : endpoint(l.source));
  const farEndTitle = (l) => { const n = farEnd(l); return n.kind === 'cluster' ? l.ext.title : n.title; };

  // Go to the other end of a line: another paper is selected; a line that ends at a neighboring cluster's
  // bubble flies there and shows the paper it leads to.
  function followEdge(l) {
    const other = farEnd(l);
    if (!other) return;
    if (other.kind === 'paper') {
      selectPaper(other);
    } else {
      // The paper is drawn once that cluster opens (it is in the cluster's file even if it isn't a top paper).
      navigate(`#cluster=${l.neighbor}&paper=${encodeURIComponent(l.ext.id)}`);
      // Safety net for data exported before that was the case: show its details without a node.
      ensureLoaded(l.neighbor).then(() => {
        flushRebuild();
        if (routed === l.neighbor && !nodeById.has(l.ext.id)) openDetails({ ...l.ext, kind: 'paper', bubble: bubbleOf.get(l.neighbor) });
      });
    }
  }
  function handleEmptyClick(event) {
    const l = event && pickEdge(event.clientX, event.clientY);
    if (l) followEdge(l);
    else closeDetails();
  }

  // Hovering a highlighted line shows where it leads.
  const edgeTip = document.createElement('div');
  edgeTip.id = 'edge-tip';
  edgeTip.hidden = true;
  document.body.appendChild(edgeTip);
  // (The library resets the canvas's own cursor every frame, so the cursor is set on its container.)
  function clearEdgeHover() {
    if (edgeHoverNode) {
      const prev = edgeHoverNode;
      edgeHoverNode = null;
      updateHalo(prev);
    }
    edgeTip.hidden = true;
    $('graph').style.cursor = '';
  }
  let lastPointer = null; // where the mouse last was, in window coordinates
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
    const verb = endpoint(l.source) === detailsFor ? 'Cites' : 'Cited by';
    edgeTip.innerHTML = `${verb} <b>${escapeHtml(farEndTitle(l))}</b><br/><span>Click to go there</span>`;
    edgeTip.style.left = `${Math.min(lastPointer.x + 14, window.innerWidth - 340)}px`;
    edgeTip.style.top = `${lastPointer.y + 14}px`;
    edgeTip.hidden = false;
    $('graph').style.cursor = 'pointer';
  }
  let edgeHoverQueued = false;
  canvas.addEventListener('pointermove', (e) => {
    lastPointer = { x: e.clientX, y: e.clientY };
    if (e.buttons) { clearEdgeHover(); return; } // dragging the map
    if (edgeHoverQueued) return;
    edgeHoverQueued = true;
    // Two frames later, so the library has updated which paper (if any) is under the mouse.
    requestAnimationFrame(() => requestAnimationFrame(() => { edgeHoverQueued = false; updateEdgeHover(); }));
  });
  canvas.addEventListener('pointerleave', () => { lastPointer = null; clearEdgeHover(); });
  $('details-close').addEventListener('click', closeDetails);
  // Esc closes the details panel first; pressing it again flies back out to the whole map.
  window.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape' || e.target instanceof HTMLInputElement) return;
    if (detailsFor) closeDetails();
    else if (focused || routed !== null) navigate('');
  });

  // ---- arXiv field chips (color key + filter) ----
  const chipFields = [...fields.map((f) => ({ key: f.code, label: f.code, title: `${f.name} · ${f.papers.toLocaleString()} clustered papers` })),
    { key: 'other', label: 'other', title: 'All remaining arXiv fields' }];
  $('legend').innerHTML = chipFields
    .map((f) => `<button type="button" class="chip" data-field="${escapeHtml(f.key)}" title="${escapeHtml(f.title)}">${swatch(fieldColor(f.key))}${escapeHtml(f.label)}</button>`)
    .join('');
  $('legend').addEventListener('click', (e) => {
    const chip = e.target.closest('.chip');
    if (chip) setFieldFilter(chip.dataset.field);
  });

  // ---- Cluster search (type a topic, or pick from the biggest clusters) ----
  const searchIndex = clusters.map((c) => ({
    c,
    text: `${stripPrefix(c.title)} ${c.topPaper} ${c.categoryName} ${c.category}`.toLowerCase(),
  }));
  const input = $('search-input');
  const results = $('search-results');
  let shown = [];     // clusters currently listed
  let active = -1;    // keyboard-highlighted result

  function setSearchHover(node) {
    const prev = searchHover;
    searchHover = node;
    updateHalo(prev);
    updateHalo(node);
  }
  function renderResults() {
    const q = input.value.trim().toLowerCase();
    const terms = q.split(/\s+/).filter(Boolean);
    const hits = searchIndex.filter((e) => terms.every((t) => e.text.includes(t))).map((e) => e.c).sort((a, b) => b.size - a.size);
    shown = hits.slice(0, 8);
    active = shown.length && q ? 0 : -1;
    const heading = q ? '' : '<li class="heading">Largest clusters</li>';
    const rows = shown.map((c, i) =>
      `<li data-i="${i}" class="${i === active ? 'active' : ''}"><button type="button">${swatch(clusterColor(c))}` +
      `<span class="name">${escapeHtml(stripPrefix(c.title))}</span><span class="count">${c.size.toLocaleString()}</span></button></li>`);
    results.innerHTML = shown.length ? heading + rows.join('') : '<li class="empty">No clusters match.</li>';
    results.hidden = false;
  }
  function openResult(i) {
    const c = shown[i];
    if (!c) return;
    setSearchHover(null);
    input.value = '';
    results.hidden = true;
    input.blur();
    navigate(`#cluster=${c.id}`);
  }
  input.addEventListener('focus', renderResults);
  input.addEventListener('input', renderResults);
  input.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      if (!shown.length) return;
      active = (active + (e.key === 'ArrowDown' ? 1 : -1) + shown.length) % shown.length;
      results.querySelectorAll('li[data-i]').forEach((li) => li.classList.toggle('active', Number(li.dataset.i) === active));
    } else if (e.key === 'Enter') {
      openResult(active >= 0 ? active : 0);
    } else if (e.key === 'Escape') {
      input.value = '';
      results.hidden = true;
      input.blur();
    }
  });
  // mousedown (not click) so the input doesn't lose focus and hide the list before the click lands.
  results.addEventListener('mousedown', (e) => {
    e.preventDefault();
    const li = e.target.closest('li[data-i]');
    if (li) openResult(Number(li.dataset.i));
  });
  results.addEventListener('mouseover', (e) => {
    const li = e.target.closest('li[data-i]');
    setSearchHover(li ? bubbleOf.get(shown[Number(li.dataset.i)].id) : null);
  });
  results.addEventListener('mouseleave', () => setSearchHover(null));
  input.addEventListener('blur', () => { results.hidden = true; setSearchHover(null); });

  // ---- Time window slider ----
  // Two-handle slider: drag either end inward to hide papers outside that date.
  function setupTimeline() {
    const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const fmt = (y) => `${MONTHS[Math.min(11, Math.floor((y - Math.floor(y)) * 12))]} ${Math.floor(y)}`;
    const span = T1 - T0;
    const MIN_GAP = span * 0.01; // handles can't cross or fully overlap
    const pct = (y) => `${((y - T0) / span) * 100}%`;

    const panel = $('timeline');
    const track = $('tl-track');
    const loEl = $('tl-lo');
    const hiEl = $('tl-hi');
    const rangeEl = $('tl-range');
    const labelEl = $('tl-label');

    // One bar per year. Two views of the same bars (flip with the switch):
    //   volume     = papers that year (sqrt-scaled so the recent spike doesn't flatten the rest)
    //   importance = average influence score of that year's papers.
    //                Average rather than total, so it doesn't just repeat the volume chart.
    // setData() swaps in the numbers for whatever is in focus (all clusters or the focused one,
    // for all fields or the selected one).
    const bars = $('tl-bars');
    const barEls = overview.years.map((year) => {
      const a = Math.max(year, T0);
      const b = Math.min(year + 1, T1);
      const el = document.createElement('div');
      el.style.left = pct(a);
      el.style.width = `calc(${((b - a) / span) * 100}% - 1px)`;
      bars.appendChild(el);
      return { el, year };
    });

    let data = { count: overview.years.map(() => 0), inf: overview.years.map(() => 0) };
    let barMode = 'volume';
    function styleBars() {
      const maxCount = Math.max(1, ...data.count);
      const maxImportance = Math.max(1e-9, ...data.inf);
      barEls.forEach(({ el, year }, i) => {
        const h = barMode === 'volume' ? Math.sqrt(data.count[i] / maxCount) : data.inf[i] / maxImportance;
        el.style.height = `${data.count[i] === 0 ? 0 : Math.max(6, h * 100)}%`;
        el.classList.toggle('importance', barMode === 'importance');
        el.title = `${year}: ${data.count[i].toLocaleString()} papers · avg importance ${data.inf[i].toFixed(2)}`;
      });
    }

    const modeBtn = $('tl-mode');
    modeBtn.addEventListener('click', () => {
      barMode = barMode === 'volume' ? 'importance' : 'volume';
      modeBtn.setAttribute('aria-checked', String(barMode === 'importance'));
      $('tl-mode-text').textContent = barMode === 'volume' ? 'Paper volume' : 'Paper importance';
      styleBars();
    });

    // Year ticks every 5 years.
    const ticks = $('tl-ticks');
    for (let y = Math.ceil(T0 / 5) * 5; y <= T1; y += 5) {
      const t = document.createElement('span');
      t.style.left = pct(y);
      t.textContent = y;
      ticks.appendChild(t);
    }

    function refresh() {
      loEl.style.left = pct(windowLo);
      hiEl.style.left = pct(windowHi);
      rangeEl.style.left = pct(windowLo);
      rangeEl.style.width = `${((windowHi - windowLo) / span) * 100}%`;
      const filterNote = fieldFilter ? ` ${fieldFilter}` : '';
      let summary;
      if (!focused) {
        const inRange = Math.round(bubbleNodes.reduce((s, n) => s + n.inWindow, 0));
        const total = fieldFilter ? clusters.reduce((s, c) => s + c.fy[fieldFilter].reduce((a, b) => a + b, 0), 0) : meta.clusteredPapers;
        summary = `${inRange.toLocaleString()} of ${total.toLocaleString()}${filterNote} papers`;
      } else {
        const own = loaded.get(focused.id).papers.filter((p) => !p.extra);
        const matching = own.filter((n) => yearVisible(n) && (!fieldFilter || fieldKey(n.categoryCode) === fieldFilter)).length;
        summary = `${matching} of ${own.length} top papers${filterNote ? ` (${filterNote.trim()})` : ''} shown`;
      }
      labelEl.innerHTML = `${fmt(windowLo)} – ${fmt(windowHi)} <span class="hint">· ${summary}</span>`;
    }

    let queued = false;
    const scheduleUpdate = () => {
      if (queued) return;
      queued = true;
      requestAnimationFrame(() => {
        queued = false;
        applyWindow();
        refresh();
      });
    };

    function bindHandle(el, which) {
      el.addEventListener('pointerdown', (e) => {
        el.setPointerCapture(e.pointerId);
        el.classList.add('dragging');
        e.preventDefault();
      });
      el.addEventListener('pointermove', (e) => {
        if (!el.hasPointerCapture(e.pointerId)) return;
        const rect = track.getBoundingClientRect();
        const p = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
        const y = T0 + p * span;
        if (which === 'lo') windowLo = Math.max(T0, Math.min(y, windowHi - MIN_GAP));
        else windowHi = Math.min(T1, Math.max(y, windowLo + MIN_GAP));
        scheduleUpdate();
      });
      const release = () => el.classList.remove('dragging');
      el.addEventListener('pointerup', release);
      el.addEventListener('pointercancel', release);
    }
    bindHandle(loEl, 'lo');
    bindHandle(hiEl, 'hi');

    panel.hidden = false;
    return {
      setData(d) { data = d; styleBars(); },
      refresh,
      resetWindow() {
        windowLo = T0;
        windowHi = T1;
        applyWindow();
        refresh();
      },
    };
  }
  const timeline = setupTimeline();

  // Back to the starting state: whole map in view, no time or field filter, nothing open.
  $('tl-reset').addEventListener('click', () => {
    if (fieldFilter) setFieldFilter(fieldFilter); // toggling the active field clears it
    timeline.resetWindow();
    closeDetails();
    if (routed !== null || location.hash) navigate('');
    else goHome();
  });

  // ---- Start ----
  rebuildGraph();
  sizeBubbles();
  fitToWindow();
  updateLOD();
  updateChrome();
  route();

  // Handy for poking at things from the browser console while we build.
  window.graph = graph;
  window.lod = { bubbleNodes, loaded, get attached() { return attached; }, get selected() { return detailsFor; }, get nodeById() { return nodeById; } };
}

main().catch((err) => {
  console.error(err);
  $('stats').textContent = `Failed to load data: ${err.message}`;
});
