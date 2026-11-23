import {
  BufferAttribute, BufferGeometry, Color, LineSegments, Mesh, MeshBasicMaterial,
  Matrix4, OrthographicCamera, Points, RingGeometry, Scene, ShaderMaterial, Vector3, WebGLRenderer,
} from 'three';
import { MapControls } from 'three/examples/jsm/controls/MapControls.js';
import { FIELD_COLORS, OTHER_COLOR } from './util';
import { createShardLoaders } from './data';
import { createGrid } from './grid';
import type { EdgeEntry, PaperDetail, PointData } from './types';

// ---------------------------------------------------------------------------
// Point-cloud prototype: every paper is one dot at its fixed UMAP position, noise included.
// One THREE.Points draws all of them in a single call; an orthographic camera with MapControls
// gives 2D pan and zoom. Hover and click pick dots through a uniform grid (grid.ts).
//
// Like lib/engine.ts, the engine owns the scene and React owns the panels: React reads the
// engine's state (re-rendering whenever it calls emit) and calls its actions.
// ---------------------------------------------------------------------------

const BACKGROUND = 0x0b0d12;
const NOISE_RGBA = [120, 126, 142, 70];   // faint grey haze for unclustered papers
const CLUSTER_ALPHA = 210;
const BASE_PX = 1.6;        // dot diameter (CSS px) at the starting zoom, for an average paper...
const MAX_PX = 14;          // ...growing with zoom, up to this
const ZOOM_GROWTH = 0.6;    // dot size grows as zoom^ZOOM_GROWTH, so zooming in also separates dots
const INFLUENCE_SIZE = 2.5; // the most influential papers are up to (1 + this) times bigger
const PICK_PX = 6;          // hover/click reach around the cursor, in CSS px
const CLICK_SLOP_PX = 4;    // a press that moves further than this is a drag, not a click
const FIT_MARGIN = 1.05;
const SELECTED_FADE = 0.18; // while a paper is selected, every other dot keeps this much of its opacity
export const REF_COLOR = '#5fb4ff';     // lines to papers the selected paper cites
export const CITER_COLOR = '#ff8a5c';   // lines to papers that cite it
const LINE_OPACITY = 0.55;
const PANEL_PX = 380; // width of the details panel (#details in styles.css)
// Past this many clusters, per-cluster hues stop being telling apart -- 8,490 clusters on the
// full dataset would be confetti. Above it, dots take their arXiv field's colour instead, which
// shows the macro structure; the cluster title still appears on hover and in the panel.
const MANY_CLUSTERS = 200;

// ---- Level of detail ----
// The GPU draws 3.2M dots in one call without breaking a sweat; the reason to draw fewer when
// zoomed out is that they overplot into a solid mass and you lose the landmarks. The draw-order
// index is already sorted least-to-most influential, so "show only the top K" is a draw range
// over its tail -- O(1), no buffer rewrite, no fetching.
//
// How many is "too many" is decided by INK COVERAGE: the share of the screen the dots paint
// over. A fixed count is the wrong target, because dot diameter also grows with zoom, so
// holding the count steady lets the ink keep climbing -- the old curve peaked near 150%
// coverage around zoom 3-6, where dots are mostly drawing on top of each other, and it hit
// every dot by zoom 5.7 of a 500x range. Budgeting coverage instead makes the two knobs
// (how many dots, how big) consistent by construction: raise MAX_PX and fewer dots are drawn.
// Neighbouring dots are kept at least this many dot-diameters apart on screen. It sets the
// pyramid level, and so the on-screen density, directly. Derived analytically it would be
// sqrt(pi / (4 * coverage)), but the pyramid stacks a cell's point together with its parents'
// in the same patch of screen and cells are not uniformly occupied, so the honest value is
// measured: 4.3 holds the dense computer-science blob near 15% ink across the zoom range,
// where the analytic 2.37 left it at 50-70%.
const DOT_GAP = 4.3;
// Floor on the dot count. The geometry above is conservative at the extremes -- fully zoomed
// out it asks for ~26k, and raising the time axis spreads the map over more screen and asks for
// less still -- which leaves the overview thinner than it needs to be. Measured ink at this
// floor is ~9%, well under the ~22% peak elsewhere in the zoom range.
const LOD_MIN = 50_000;
// Below this many drawn dots, tilted picking just scores them all (exact, ~1.5 ms).
// Above it the ray's shadow is walked through the grid instead.
const BRUTE_PICK_MAX = 400_000;

// ---- Time axis ----
// Papers lift off the plane by publication year, turning the map into a chronological landscape.
// Height lives in position.z and is scaled by a uniform, so toggling and animating it costs
// nothing -- no 39 MB position rewrite.
const Z_SPAN = 1.4;         // world units from the oldest paper to the newest (x/y span 2.0)
const Z_ANIM_MS = 600;

const VERTEX = /* glsl */ `
  attribute vec4 aColor;
  attribute float aSize;
  uniform float uPx;
  uniform float uDpr;
  uniform float uFade;
  uniform float uZ;
  varying vec4 vColor;
  varying float vSize;
  void main() {
    vColor = vec4(aColor.rgb, aColor.a * uFade);
    vSize = uPx * aSize * uDpr;
    gl_PointSize = vSize;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position.xy, position.z * uZ, 1.0);
  }
`;

const FRAGMENT = /* glsl */ `
  varying vec4 vColor;
  varying float vSize;
  void main() {
    float r = length(gl_PointCoord - 0.5);
    if (r > 0.5) discard;
    // Soften only the outermost pixel, so even tiny dots stay visible.
    float a = vColor.a * smoothstep(0.5, 0.5 - 1.0 / max(vSize, 1.0), r);
    gl_FragColor = vec4(vColor.rgb, a);
  }
`;

const LINE_VERTEX = /* glsl */ `
  attribute vec3 aLineColor;
  uniform float uZ;
  varying vec3 vColor;
  void main() {
    vColor = aLineColor;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position.xy, position.z * uZ, 1.0);
  }
`;

const LINE_FRAGMENT = /* glsl */ `
  uniform float uOpacity;
  varying vec3 vColor;
  void main() {
    gl_FragColor = vec4(vColor, uOpacity);
  }
`;

export type PointEngine = ReturnType<typeof createPointEngine>;

export interface HoverInfo {
  index: number;
  /** Cursor position in CSS px, relative to the container. */
  screenX: number;
  screenY: number;
  /** undefined while the title shard loads. */
  title?: string;
}

export interface Selection {
  index: number;
  /** undefined while loading; null when the paper has no citation data at all. */
  edges?: EdgeEntry | null;
  detail?: PaperDetail | null;
  /** The neighbors, most influential first. */
  refs: number[];
  citers: number[];
  error?: string;
}

export function createPointEngine(container: HTMLElement, data: PointData) {
  const { meta, x, y, influence, cluster } = data;
  const n = meta.count;
  const loaders = createShardLoaders(meta);
  const grid = createGrid(x, y);
  const clusterById = new Map(meta.clusters.map((c) => [c.id, c]));

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
  let hovered: HoverInfo | null = null;
  let selection: Selection | null = null;
  let highlighted: number | null = null; // the paper whose row the mouse is over in the details panel
  let destroyed = false;

  // ---- Colors ----
  // Few clusters: one golden-angle hue each (biggest first), which keeps neighbouring clusters
  // far apart on the colour wheel. Many clusters: fall back to the arXiv field palette.
  const colorByField = meta.clusters.length > MANY_CLUSTERS;
  const tmp = new Color();
  const toRgb = (css: string): [number, number, number] => {
    tmp.set(css);
    return [Math.round(tmp.r * 255), Math.round(tmp.g * 255), Math.round(tmp.b * 255)];
  };

  const clusterRgb = new Map<number, [number, number, number]>();
  meta.clusters.forEach((c, k) => {
    tmp.setHSL(((k * 137.508) % 360) / 360, 0.65, 0.6);
    clusterRgb.set(c.id, [Math.round(tmp.r * 255), Math.round(tmp.g * 255), Math.round(tmp.b * 255)]);
  });
  const fieldRgb = meta.fields.map((_, i) => toRgb(FIELD_COLORS[i] ?? OTHER_COLOR));
  const noiseCss = `rgb(${NOISE_RGBA.slice(0, 3).join(', ')})`;

  /** The colour a given paper's dot is drawn in. */
  const rgbOf = (i: number): [number, number, number] | undefined =>
    colorByField ? fieldRgb[data.field[i]] : clusterRgb.get(cluster[i]);

  /** Same colour as a CSS string, for the swatches beside titles in the panels. */
  const paperCss = (i: number): string => {
    if (cluster[i] === -1 && !colorByField) return noiseCss;
    const rgb = rgbOf(i);
    return rgb ? `rgb(${rgb.join(', ')})` : noiseCss;
  };

  const clusterCss = (id: number): string => {
    const rgb = clusterRgb.get(id);
    return rgb ? `rgb(${rgb.join(', ')})` : noiseCss;
  };

  // ---- Geometry: one vertex per paper ----
  const positions = new Float32Array(n * 3);
  const colors = new Uint8Array(n * 4);
  const sizes = new Float32Array(n);
  // Dots are sized by influence RANK rather than raw influence. The raw scores are extremely
  // skewed -- the median paper has 0 citations and ~62% of them sit on the score's 0.1 floor, so
  // scaling by value puts 42% of dots at exactly the minimum size. Ranking spreads the papers that
  // do have citations across the full size range. export_web_data.py ranks for the same reason.
  // Equal scores share a rank, so the floor stays one flat group instead of inventing an order.
  const byScore = new Uint32Array(n);
  for (let i = 0; i < n; i++) byScore[i] = i;
  byScore.sort((a, b) => influence[a] - influence[b]);
  const rank = new Float32Array(n);
  for (let k = 0; k < n; ) {
    let j = k;
    while (j < n && influence[byScore[j]] === influence[byScore[k]]) j++;
    const shared = n > 1 ? k / (n - 1) : 0; // every tied paper takes the run's first position
    for (let m = k; m < j; m++) rank[byScore[m]] = shared;
    k = j;
  }
  // Height by publication year. Papers with no year (0) sit on the floor with the oldest.
  const { yearMin, yearMax } = meta;
  const yearSpan = Math.max(1, yearMax - yearMin);
  const zOf = (i: number) => (data.year[i] ? ((data.year[i] - yearMin) / yearSpan) * Z_SPAN : 0);

  for (let i = 0; i < n; i++) {
    positions[i * 3] = x[i];
    positions[i * 3 + 1] = y[i];
    positions[i * 3 + 2] = zOf(i);
    const rgb = cluster[i] === -1 ? undefined : rgbOf(i);
    if (rgb) colors.set([rgb[0], rgb[1], rgb[2], CLUSTER_ALPHA], i * 4);
    else colors.set(NOISE_RGBA, i * 4);
    sizes[i] = 1 + INFLUENCE_SIZE * Math.sqrt(rank[i]);
  }
  // Draw order, which doubles as the level-of-detail order: the renderer draws a tail of this
  // index buffer, so whatever sits at the end is what survives when zoomed out.
  //
  // Points are bucketed into a quadtree-style pyramid. Level 0 is a 1x1 grid, level 1 a 2x2,
  // and so on; each cell accepts exactly ONE point, and points are offered to the pyramid most
  // influential first, so the most important paper in a region claims the coarsest free cell.
  // Drawing levels 0..L therefore yields at most one dot per cell of a 2^L grid -- an evenly
  // spread sample whose density is CAPPED everywhere.
  //
  // That cap is the point. Two simpler schemes failed on this data:
  //   - by influence alone: citation data only covers the 128,606 scraped papers, nearly all
  //     computer science, so the zoomed-out map looked like arXiv was almost entirely CS;
  //   - a flat random sample: honest globally, but it thins dense and sparse regions by the
  //     same ratio, so the dense CS blob stayed a solid mass while the rest went bare.
  // The pyramid keeps the landmarks (they win the coarse cells) and fixes both.
  const MAX_LEVEL = 11; // 4^11 = 4.2M cells at the finest level, for 3.2M points
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (let i = 0; i < n; i++) {
    if (x[i] < minX) minX = x[i];
    if (x[i] > maxX) maxX = x[i];
    if (y[i] < minY) minY = y[i];
    if (y[i] > maxY) maxY = y[i];
  }
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;

  const occupied: Uint8Array[] = [];
  for (let l = 0; l <= MAX_LEVEL; l++) occupied.push(new Uint8Array(1 << (2 * l)));
  const level = new Uint8Array(n);
  // byScore runs ascending by influence, so walking it backwards offers the best papers first.
  for (let k = n - 1; k >= 0; k--) {
    const i = byScore[k];
    const fx = (x[i] - minX) / spanX;
    const fy = (y[i] - minY) / spanY;
    let placed = MAX_LEVEL + 1; // overflow: a cell was taken at every level
    for (let l = 0; l <= MAX_LEVEL; l++) {
      const side = 1 << l;
      const col = Math.min(side - 1, (fx * side) | 0);
      const row = Math.min(side - 1, (fy * side) | 0);
      const cell = row * side + col;
      if (!occupied[l][cell]) {
        occupied[l][cell] = 1;
        placed = l;
        break;
      }
    }
    level[i] = placed;
  }

  // Running totals, so "draw levels 0..L" becomes a draw range. levelCumulative[L] is how many
  // points sit at level L or coarser; the last entry is every point.
  const levelCumulative = new Uint32Array(MAX_LEVEL + 2);
  for (let i = 0; i < n; i++) levelCumulative[level[i]]++;
  for (let l = 1; l <= MAX_LEVEL + 1; l++) levelCumulative[l] += levelCumulative[l - 1];

  // Sorted deepest level first, so the TAIL of the buffer -- the part that stays drawn when
  // zoomed out -- holds the coarse levels. Within a level, least influential first, so the
  // important dots are painted last and land on top.
  const order = new Uint32Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  order.sort((a, b) => level[b] - level[a] || influence[a] - influence[b]);

  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new BufferAttribute(positions, 3));
  geometry.setAttribute('aColor', new BufferAttribute(colors, 4, true));
  geometry.setAttribute('aSize', new BufferAttribute(sizes, 1));
  const drawPos = new Uint32Array(n);
  for (let k = 0; k < n; k++) drawPos[order[k]] = k;

  geometry.setIndex(new BufferAttribute(order, 1));
  geometry.computeBoundingSphere();

  const makeDotMaterial = () => new ShaderMaterial({
    vertexShader: VERTEX,
    fragmentShader: FRAGMENT,
    uniforms: { uPx: { value: BASE_PX }, uDpr: { value: window.devicePixelRatio }, uFade: { value: 1 },
               uZ: { value: 0 } },
    transparent: true,
    depthTest: false,
    depthWrite: false,
  });
  const material = makeDotMaterial();
  const points = new Points(geometry, material);

  // While a paper is selected: lines to its neighbors, then the neighbors redrawn at full opacity on top
  // of the faded map. Both are rebuilt on every selection (a few to a few thousand vertices).
  const lineMaterial = new ShaderMaterial({
    vertexShader: LINE_VERTEX,
    fragmentShader: LINE_FRAGMENT,
    uniforms: { uZ: { value: 0 }, uOpacity: { value: LINE_OPACITY } },
    transparent: true, depthTest: false, depthWrite: false,
  });
  const lines = new LineSegments(new BufferGeometry(), lineMaterial);
  lines.renderOrder = 1;
  lines.frustumCulled = false;
  const overlayMaterial = makeDotMaterial();
  const overlay = new Points(new BufferGeometry(), overlayMaterial);
  overlay.renderOrder = 2;
  overlay.frustumCulled = false;
  const refRgb = new Color(REF_COLOR);
  const citerRgb = new Color(CITER_COLOR);

  /** Rebuild the lines and neighbor dots for the current selection (or clear them). */
  const buildSelectionLayers = () => {
    lines.geometry.dispose();
    overlay.geometry.dispose();
    lines.geometry = new BufferGeometry();
    overlay.geometry = new BufferGeometry();
    if (!selection) {
      material.uniforms.uFade.value = 1;
      return;
    }
    material.uniforms.uFade.value = SELECTED_FADE;
    const s = selection.index;
    const neighbors = [...selection.refs.map((j) => [j, refRgb] as const), ...selection.citers.map((j) => [j, citerRgb] as const)];
    const linePos = new Float32Array(neighbors.length * 6);
    const lineCol = new Float32Array(neighbors.length * 6);
    neighbors.forEach(([j, rgb], k) => {
      linePos.set([x[s], y[s], zOf(s), x[j], y[j], zOf(j)], k * 6);
      lineCol.set([rgb.r, rgb.g, rgb.b, rgb.r, rgb.g, rgb.b], k * 6);
    });
    lines.geometry.setAttribute('position', new BufferAttribute(linePos, 3));
    lines.geometry.setAttribute('aLineColor', new BufferAttribute(lineCol, 3));

    // Neighbors first, the selected paper last so it sits on top.
    const dots = [...new Set([...neighbors.map(([j]) => j), s])].filter((j) => j !== s).concat(s);
    const pos = new Float32Array(dots.length * 3);
    const col = new Uint8Array(dots.length * 4);
    const size = new Float32Array(dots.length);
    dots.forEach((j, k) => {
      pos.set([x[j], y[j], zOf(j)], k * 3);
      col.set([colors[j * 4], colors[j * 4 + 1], colors[j * 4 + 2], 255], k * 4);
      size[k] = Math.max(sizes[j], 1.6) * (j === s ? 1.6 : 1);
    });
    overlay.geometry.setAttribute('position', new BufferAttribute(pos, 3));
    overlay.geometry.setAttribute('aColor', new BufferAttribute(col, 4, true));
    overlay.geometry.setAttribute('aSize', new BufferAttribute(size, 1));
  };

  // Rings around the hovered and the selected paper; scaled every frame to stay a fixed size on screen.
  const ringGeometry = new RingGeometry(0.75, 1, 40);
  const makeRing = (color: number) => {
    const ring = new Mesh(ringGeometry, new MeshBasicMaterial({ color, depthTest: false, transparent: true }));
    ring.renderOrder = 1;
    ring.visible = false;
    return ring;
  };
  const hoverRing = makeRing(0xffffff);
  const selectRing = makeRing(0xffd166);

  // ---- Scene ----
  const renderer = new WebGLRenderer({ antialias: false });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setClearColor(BACKGROUND);
  container.appendChild(renderer.domElement);

  const scene = new Scene();
  hoverRing.renderOrder = 3;
  selectRing.renderOrder = 3;
  scene.add(points, lines, overlay, hoverRing, selectRing);
  const camera = new OrthographicCamera(-1, 1, 1, -1, -100, 100);
  // The map lies in the XY plane and the time axis rises along +Z, so +Z is "up" for the orbit.
  // With three's default +Y up, starting overhead would put the camera on the equator and
  // right-dragging would roll the map sideways instead of tilting it.
  camera.up.set(0, 0, 1);
  camera.position.set(0, 0, 5);

  const controls = new MapControls(camera, renderer.domElement);
  controls.enableRotate = false;
  controls.screenSpacePanning = true;
  controls.zoomToCursor = true;
  controls.enableDamping = false;
  controls.minZoom = 0.5;
  controls.maxZoom = 500;
  // Polar angle 0 is straight down. Never reach the horizon: picking divides by the view
  // direction's z, and a grazing view of a flat map shows nothing anyway.
  controls.minPolarAngle = 0;
  controls.maxPolarAngle = 1.35;

  let width = 1;
  let height = 1;
  /** Fit the whole map: half-height of the view at zoom 1, in world units. */
  const fitHalfHeight = () => {
    let maxX = 0, maxY = 0;
    for (let i = 0; i < n; i++) {
      maxX = Math.max(maxX, Math.abs(x[i]));
      maxY = Math.max(maxY, Math.abs(y[i]));
    }
    return (aspect: number) => FIT_MARGIN * Math.max(maxY, maxX / aspect);
  };
  const halfHeightFor = fitHalfHeight();

  /** Angle between the view direction and straight-down, in radians. 0 = flat map. */
  const currentPolar = () => {
    const dx = camera.position.x - controls.target.x;
    const dy = camera.position.y - controls.target.y;
    const dz = camera.position.z - controls.target.z;
    return Math.atan2(Math.hypot(dx, dy), dz);
  };

  /**
   * Size the orthographic frustum to the map. Once the time axis is raised the landscape is
   * taller than the plane, and a tilted camera sees that height as extra vertical extent, so the
   * frustum has to grow by it or the newest papers fall off the top of the screen.
   */
  const updateProjection = () => {
    const aspect = width / height;
    const lift = 0.5 * Z_SPAN * zNow * Math.sin(currentPolar());
    const h = halfHeightFor(aspect) + lift;
    camera.left = -h * aspect;
    camera.right = h * aspect;
    camera.top = h;
    camera.bottom = -h;
    camera.updateProjectionMatrix();
  };

  const resize = () => {
    width = Math.max(1, container.clientWidth);
    height = Math.max(1, container.clientHeight);
    renderer.setSize(width, height);
    updateProjection();
    requestRender();
  };

  /** CSS px per world unit at the current zoom. */
  const pxPerWorld = () => (height * camera.zoom) / (camera.top - camera.bottom);
  const dotPx = () => Math.min(MAX_PX, BASE_PX * Math.pow(camera.zoom, ZOOM_GROWTH));

  // ---- Level of detail ----
  // `order` runs least-to-most influential (noise first), so the most influential papers are its
  // tail. Drawing the last `drawCount` entries therefore shows the top papers and hides the rest.
  let drawCount = n;

  /**
   * How many dots to draw, chosen by picking a pyramid level from screen geometry.
   *
   * Pick the level whose cells project to `spacing` pixels, and the view can hold at most one
   * dot per spacing x spacing square -- everywhere, whatever the local density, because the
   * pyramid already allows only one point per cell. On-screen density is then fixed by
   * geometry, not by how crowded that part of the map happens to be.
   *
   * Deriving the count from a global budget instead (dots = coverage * area * zoom^2) assumes
   * density is uniform across the map. It is not: that put the dense computer-science blob at
   * 68% ink while the sparse regions sat at 22%.
   */
  const worldSpan = Math.max(spanX, spanY);
  const lodFor = (_zoom: number) => {
    const spacing = DOT_GAP * dotPx();
    // 2^level cells across the map, each one `spacing` px wide on screen
    const wanted = Math.log2((worldSpan * pxPerWorld()) / spacing);
    const clamped = Math.max(0, Math.min(MAX_LEVEL + 1, wanted));
    // Interpolate between whole levels, or the dot count would jump as you scroll.
    const lo = Math.floor(clamped);
    const hi = Math.min(MAX_LEVEL + 1, lo + 1);
    const blend = clamped - lo;
    const count = levelCumulative[lo] + blend * (levelCumulative[hi] - levelCumulative[lo]);
    return Math.max(LOD_MIN, Math.min(n, Math.round(count)));
  };

  /** Is paper i currently drawn? Used so hover and click can't land on a hidden dot. */
  const isDrawn = (i: number) => drawPos[i] >= n - drawCount;

  function applyLod() {
    const next = lodFor(camera.zoom);
    if (next === drawCount) return false;
    drawCount = next;
    geometry.setDrawRange(n - drawCount, drawCount);
    // A hidden dot must not stay hovered.
    if (hovered && !isDrawn(hovered.index)) hovered = null;
    emit(); // the HUD shows how many dots are currently drawn
    return true;
  }
  geometry.setDrawRange(n - drawCount, drawCount);

  // ---- Time axis ----
  // zTarget/zNow animate the lift so the landscape rises rather than snapping.
  let zTarget = 0;
  let zNow = 0;
  let zAnimStart = 0;
  let zAnimFrom = 0;

  const setZUniform = (v: number) => {
    material.uniforms.uZ.value = v;
    overlayMaterial.uniforms.uZ.value = v;
    lineMaterial.uniforms.uZ.value = v;
  };

  const placeRing = (ring: Mesh, i: number | null) => {
    ring.visible = i !== null;
    if (i === null) return;
    ring.position.set(x[i], y[i], zOf(i) * zNow);
    ring.scale.setScalar((dotPx() * sizes[i]) / 2 / pxPerWorld() + 4 / pxPerWorld());
    // Rings are flat quads: face the camera so they read as circles from any angle.
    ring.quaternion.copy(camera.quaternion);
  };

  // ---- Render on demand ----
  let frame = 0;
  function requestRender() {
    if (frame || destroyed) return;
    frame = requestAnimationFrame(() => {
      frame = 0;
      if (zNow !== zTarget) {
        const t = Math.min(1, (performance.now() - zAnimStart) / Z_ANIM_MS);
        const eased = t * t * (3 - 2 * t);
        zNow = zAnimFrom + (zTarget - zAnimFrom) * eased;
        if (t >= 1) zNow = zTarget;
        setZUniform(zNow);
        updateProjection();
      }
      applyLod();
      material.uniforms.uPx.value = dotPx();
      overlayMaterial.uniforms.uPx.value = dotPx();
      placeRing(hoverRing, hovered?.index ?? highlighted);
      placeRing(selectRing, selection?.index ?? null);
      renderer.render(scene, camera);
      if (zNow !== zTarget) requestRender(); // keep animating
    });
  }
  controls.addEventListener('change', () => {
    // The view moved under the cursor, so whatever was hovered no longer is; the next pointermove re-picks.
    if (hovered) {
      hovered = null;
      emit();
    }
    requestRender();
  });

  // ---- Picking ----
  // Flat: unproject the cursor onto z=0 and ask the grid (which is indexed on x/y) for the
  // nearest dot. Tilted: a dot's screen position depends on its height too, so the cursor no
  // longer maps to one x/y point but to a SEGMENT -- the ray's shadow on the plane, from the
  // floor of the landscape to its ceiling. Walk that segment, ask the grid at each step, and
  // score the candidates by true screen distance.
  const cursor = new Vector3();
  const probe = new Vector3();

  // Project (x, y, z*zNow) to pixels with the camera's combined matrix, without allocating.
  const viewProj = new Matrix4();
  const refreshViewProj = () => {
    camera.updateMatrixWorld();
    viewProj.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
  };

  const pick = (clientX: number, clientY: number): number => {
    const rect = renderer.domElement.getBoundingClientRect();
    const sx = clientX - rect.left;
    const sy = clientY - rect.top;
    const reachPx = Math.max(PICK_PX, (dotPx() * (1 + INFLUENCE_SIZE)) / 2);

    // Flat map: the grid is indexed on x/y and the cursor unprojects straight onto the plane,
    // so one spatial query answers it.
    if (zNow <= 0) {
      cursor.set((sx / rect.width) * 2 - 1, -(sy / rect.height) * 2 + 1, 0).unproject(camera);
      return grid.nearest(cursor.x, cursor.y, reachPx / pxPerWorld(), isDrawn);
    }

    // Tilted: a dot's screen position depends on its height, so the x/y grid can no longer
    // answer "what is under the cursor" -- the candidates lie along the ray's shadow, a band
    // several world units long. Scanning that band through the grid was both slow and wrong
    // (it returns the nearest dot in WORLD space, which is often not the nearest on screen).
    // Scoring every drawn dot in screen space is exact, and only the drawn ones can be hit.
    refreshViewProj();
    const m = viewProj.elements;
    const halfW = width / 2;
    const halfH = height / 2;
    let best = -1;
    let bestD = reachPx * reachPx;

    const consider = (i: number) => {
      if (!isDrawn(i)) return;
      const px = x[i];
      const py = y[i];
      const pz = zOf(i) * zNow;
      const w = m[3] * px + m[7] * py + m[11] * pz + m[15];
      if (w === 0) return;
      const dx = ((m[0] * px + m[4] * py + m[8] * pz + m[12]) / w + 1) * halfW - sx;
      if (dx > reachPx || dx < -reachPx) return; // cheap reject before the second axis
      const dy = (1 - (m[1] * px + m[5] * py + m[9] * pz + m[13]) / w) * halfH - sy;
      const d = dx * dx + dy * dy;
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    };

    if (drawCount <= BRUTE_PICK_MAX) {
      // Few enough dots that scoring all of them is exact and quicker than any index.
      for (let k = n - drawCount; k < n; k++) consider(order[k]);
      return best;
    }

    // Zoomed in: the tube around the cursor's ray is narrow in world terms, so walking the
    // grid cells its shadow crosses touches far fewer points than the whole drawn set.
    cursor.set((sx / rect.width) * 2 - 1, -(sy / rect.height) * 2 + 1, 0).unproject(camera);
    const dir = new Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
    if (Math.abs(dir.z) < 1e-6) {
      for (let k = n - drawCount; k < n; k++) consider(order[k]);
      return best;
    }
    const at = (z: number) => {
      const t = (z - cursor.z) / dir.z;
      return { x: cursor.x + dir.x * t, y: cursor.y + dir.y * t };
    };
    const a = at(0);
    const b = at(Z_SPAN * zNow);
    const polar = Math.acos(Math.min(1, Math.abs(dir.z)));
    const searchWorld = (reachPx / pxPerWorld()) / Math.max(Math.cos(polar), 0.2);
    grid.eachAlongSegment(a.x, a.y, b.x, b.y, searchWorld, consider);
    return best;
  };

  const setHover = (i: number, sx: number, sy: number) => {
    if (i === -1) {
      if (hovered) {
        hovered = null;
        emit();
        requestRender();
      }
      return;
    }
    if (hovered?.index === i) {
      hovered = { ...hovered, screenX: sx, screenY: sy };
      emit();
      return;
    }
    hovered = { index: i, screenX: sx, screenY: sy };
    loaders.title(i).then((title) => {
      if (hovered?.index === i && !destroyed) {
        hovered = { ...hovered, title: title ?? '(untitled)' };
        emit();
      }
    }).catch((err) => console.error(err));
    emit();
    requestRender();
  };

  let moveQueued = false;
  let lastMove: PointerEvent | null = null;
  let pressX = 0, pressY = 0, pressed = false;
  const onMove = (e: PointerEvent) => {
    lastMove = e;
    if (moveQueued) return;
    moveQueued = true;
    requestAnimationFrame(() => {
      moveQueued = false;
      if (!lastMove || destroyed) return;
      if (pressed) return setHover(-1, 0, 0); // no tooltips while dragging
      const rect = renderer.domElement.getBoundingClientRect();
      setHover(pick(lastMove.clientX, lastMove.clientY), lastMove.clientX - rect.left, lastMove.clientY - rect.top);
    });
  };
  const onDown = (e: PointerEvent) => {
    pressed = true;
    pressX = e.clientX;
    pressY = e.clientY;
  };
  const onUp = (e: PointerEvent) => {
    pressed = false;
    if (Math.hypot(e.clientX - pressX, e.clientY - pressY) > CLICK_SLOP_PX) return;
    select(pick(e.clientX, e.clientY));
  };
  const onLeave = () => setHover(-1, 0, 0);
  const onKey = (e: KeyboardEvent) => {
    if (e.key === 'Escape') select(null);
  };
  const canvas = renderer.domElement;
  canvas.addEventListener('pointermove', onMove);
  canvas.addEventListener('pointerdown', onDown);
  canvas.addEventListener('pointerup', onUp);
  canvas.addEventListener('pointerleave', onLeave);
  window.addEventListener('keydown', onKey);

  const observer = new ResizeObserver(resize);
  observer.observe(container);
  resize();

  // ---- Actions ----
  const byInfluence = (a: number, b: number) => influence[b] - influence[a];

  /** Select paper i (-1 or null clears), then load its citation links and details. */
  function select(i: number | null) {
    if (i === null || i < 0) {
      selection = null;
      highlighted = null;
    } else if (selection?.index !== i) {
      selection = { index: i, refs: [], citers: [] };
      highlighted = null;
      const current = selection;
      const update = (patch: Partial<Selection>) => {
        if (selection !== current || destroyed) return; // a newer selection replaced this one
        selection = Object.assign(current, patch);
        if (patch.edges !== undefined) buildSelectionLayers();
        emit();
        requestRender();
      };
      loaders.edges(i).then(
        (e) => update({ edges: e ?? null, refs: [...(e?.r ?? [])].sort(byInfluence), citers: [...(e?.c ?? [])].sort(byInfluence) }),
        (err: Error) => update({ edges: null, error: err.message }),
      );
      loaders.detail(i).then(
        (d) => update({ detail: d ?? null }),
        (err: Error) => update({ detail: null, error: err.message }),
      );
    }
    buildSelectionLayers();
    emit();
    requestRender();
  }

  /** Ring a paper while the mouse is over its row in the details panel (null clears). */
  function highlight(i: number | null) {
    if (highlighted === i) return;
    highlighted = i;
    requestRender();
  }

  /** Pan (keeping the zoom) so paper i is in the middle of the part of the screen the details panel leaves free. */
  function centerOn(i: number) {
    const panelPx = Math.min(PANEL_PX, width - 32) + 16;
    const cx = x[i] + panelPx / 2 / pxPerWorld();
    camera.position.set(cx, y[i], camera.position.z);
    controls.target.set(cx, y[i], 0);
    controls.update();
    requestRender();
  }

  const DEFAULT_TILT = 1.0; // radians off straight-down when the time axis is switched on

  /**
   * Point the camera `polar` radians off straight-down, keeping its distance and azimuth.
   * Straight down (0) is the flat map; larger angles look across the landscape from the side.
   */
  function setTilt(polar: number) {
    const t = controls.target;
    const dx = camera.position.x - t.x;
    const dy = camera.position.y - t.y;
    const dz = camera.position.z - t.z;
    const d = Math.hypot(dx, dy, dz) || 5;
    // Keep whatever azimuth the user had; default to looking from -Y when straight overhead.
    const flat = Math.hypot(dx, dy);
    const ax = flat > 1e-6 ? dx / flat : 0;
    const ay = flat > 1e-6 ? dy / flat : -1;
    const s = Math.sin(polar);
    camera.position.set(t.x + ax * s * d, t.y + ay * s * d, t.z + Math.cos(polar) * d);
    controls.update();
    updateProjection();
    requestRender();
  }

  /** Raise or lower the publication-year axis. Rotation only makes sense once there is height. */
  function setTimeAxis(on: boolean) {
    const next = on ? 1 : 0;
    if (next === zTarget) return;
    zAnimFrom = zNow;
    zAnimStart = performance.now();
    zTarget = next;
    controls.enableRotate = on;
    // Orbit around the middle of the landscape's height, not its floor, or the cloud drifts off
    // the top of the screen as it rises.
    controls.target.z = on ? (Z_SPAN * 0.5) : 0;
    // Height is invisible from straight overhead, so switching the axis on has to tilt the
    // camera too, or the toggle would appear to do nothing.
    setTilt(on ? DEFAULT_TILT : 0);
    emit();
    requestRender();
  }

  function resetView() {
    camera.position.set(0, 0, zTarget > 0 ? Z_SPAN * 0.5 + 5 : 5);
    controls.target.set(0, 0, zTarget > 0 ? Z_SPAN * 0.5 : 0);
    if (zTarget > 0) setTilt(DEFAULT_TILT);
    camera.zoom = 1;
    camera.updateProjectionMatrix();
    controls.update();
    requestRender();
  }

  function destroy() {
    destroyed = true;
    if (frame) cancelAnimationFrame(frame);
    observer.disconnect();
    canvas.removeEventListener('pointermove', onMove);
    canvas.removeEventListener('pointerdown', onDown);
    canvas.removeEventListener('pointerup', onUp);
    canvas.removeEventListener('pointerleave', onLeave);
    window.removeEventListener('keydown', onKey);
    controls.dispose();
    geometry.dispose();
    material.dispose();
    lines.geometry.dispose();
    lineMaterial.dispose();
    overlay.geometry.dispose();
    overlayMaterial.dispose();
    ringGeometry.dispose();
    (hoverRing.material as MeshBasicMaterial).dispose();
    (selectRing.material as MeshBasicMaterial).dispose();
    renderer.dispose();
    canvas.remove();
    listeners.clear();
  }

  return {
    data,
    loaders,
    subscribe,
    getVersion: () => version,
    get hovered() {
      return hovered;
    },
    get selection() {
      return selection;
    },
    clusterOf: (i: number) => clusterById.get(cluster[i]) ?? null,
    clusterCss,
    paperCss,
    colorByField,
    setTimeAxis,
    setTilt,
    /** How crowded the current view actually is. Used to tune DOT_GAP against real data. */
    debugStats() {
      refreshViewProj();
      const m = viewProj.elements;
      let onScreen = 0;
      for (let k = n - drawCount; k < n; k++) {
        const i = order[k];
        const pz = zOf(i) * zNow;
        const w = m[3] * x[i] + m[7] * y[i] + m[11] * pz + m[15];
        const ndcX = (m[0] * x[i] + m[4] * y[i] + m[8] * pz + m[12]) / w;
        const ndcY = (m[1] * x[i] + m[5] * y[i] + m[9] * pz + m[13]) / w;
        if (ndcX >= -1 && ndcX <= 1 && ndcY >= -1 && ndcY <= 1) onScreen++;
      }
      const d = dotPx();
      return {
        zoom: +camera.zoom.toFixed(2),
        drawn: drawCount,
        onScreen,
        dotPx: +d.toFixed(2),
        inkCoverage: +((onScreen * Math.PI * 0.25 * d * d) / (width * height)).toFixed(3),
      };
    },
    get timeAxis() {
      return zTarget > 0;
    },
    get drawn() {
      return drawCount;
    },
    select,
    highlight,
    centerOn,
    resetView,
    destroy,
  };
}
