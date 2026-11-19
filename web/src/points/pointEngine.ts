import {
  BufferAttribute, BufferGeometry, Color, LineBasicMaterial, LineSegments, Mesh, MeshBasicMaterial,
  OrthographicCamera, Points, RingGeometry, Scene, ShaderMaterial, Vector3, WebGLRenderer,
} from 'three';
import { MapControls } from 'three/examples/jsm/controls/MapControls.js';
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

const VERTEX = /* glsl */ `
  attribute vec4 aColor;
  attribute float aSize;
  uniform float uPx;
  uniform float uDpr;
  uniform float uFade;
  varying vec4 vColor;
  varying float vSize;
  void main() {
    vColor = vec4(aColor.rgb, aColor.a * uFade);
    vSize = uPx * aSize * uDpr;
    gl_PointSize = vSize;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
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

  // ---- Colors: one per cluster (golden-angle hues, biggest clusters first), grey for noise ----
  const clusterRgb = new Map<number, [number, number, number]>();
  const tmp = new Color();
  meta.clusters.forEach((c, k) => {
    tmp.setHSL(((k * 137.508) % 360) / 360, 0.65, 0.6);
    clusterRgb.set(c.id, [Math.round(tmp.r * 255), Math.round(tmp.g * 255), Math.round(tmp.b * 255)]);
  });
  const clusterCss = (id: number): string => {
    const rgb = clusterRgb.get(id);
    return rgb ? `rgb(${rgb.join(', ')})` : `rgb(${NOISE_RGBA.slice(0, 3).join(', ')})`;
  };

  // ---- Geometry: one vertex per paper ----
  const positions = new Float32Array(n * 3);
  const colors = new Uint8Array(n * 4);
  const sizes = new Float32Array(n);
  let maxInfluence = 0;
  for (let i = 0; i < n; i++) maxInfluence = Math.max(maxInfluence, influence[i]);
  const minInfluence = influence.reduce((m, v) => Math.min(m, v), Infinity);
  const span = maxInfluence - minInfluence || 1;
  for (let i = 0; i < n; i++) {
    positions[i * 3] = x[i];
    positions[i * 3 + 1] = y[i];
    const rgb = clusterRgb.get(cluster[i]);
    if (rgb) colors.set([rgb[0], rgb[1], rgb[2], CLUSTER_ALPHA], i * 4);
    else colors.set(NOISE_RGBA, i * 4);
    sizes[i] = 1 + INFLUENCE_SIZE * Math.sqrt((influence[i] - minInfluence) / span);
  }
  // Draw order: noise first (underneath), then clustered papers from least to most influential.
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => {
    const na = cluster[a] === -1 ? 0 : 1, nb = cluster[b] === -1 ? 0 : 1;
    return na - nb || influence[a] - influence[b];
  });

  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new BufferAttribute(positions, 3));
  geometry.setAttribute('aColor', new BufferAttribute(colors, 4, true));
  geometry.setAttribute('aSize', new BufferAttribute(sizes, 1));
  geometry.setIndex(new BufferAttribute(Uint32Array.from(order), 1));
  geometry.computeBoundingSphere();

  const makeDotMaterial = () => new ShaderMaterial({
    vertexShader: VERTEX,
    fragmentShader: FRAGMENT,
    uniforms: { uPx: { value: BASE_PX }, uDpr: { value: window.devicePixelRatio }, uFade: { value: 1 } },
    transparent: true,
    depthTest: false,
    depthWrite: false,
  });
  const material = makeDotMaterial();
  const points = new Points(geometry, material);

  // While a paper is selected: lines to its neighbors, then the neighbors redrawn at full opacity on top
  // of the faded map. Both are rebuilt on every selection (a few to a few thousand vertices).
  const lineMaterial = new LineBasicMaterial({
    vertexColors: true, transparent: true, opacity: LINE_OPACITY, depthTest: false, depthWrite: false,
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
      linePos.set([x[s], y[s], 0, x[j], y[j], 0], k * 6);
      lineCol.set([rgb.r, rgb.g, rgb.b, rgb.r, rgb.g, rgb.b], k * 6);
    });
    lines.geometry.setAttribute('position', new BufferAttribute(linePos, 3));
    lines.geometry.setAttribute('color', new BufferAttribute(lineCol, 3));

    // Neighbors first, the selected paper last so it sits on top.
    const dots = [...new Set([...neighbors.map(([j]) => j), s])].filter((j) => j !== s).concat(s);
    const pos = new Float32Array(dots.length * 3);
    const col = new Uint8Array(dots.length * 4);
    const size = new Float32Array(dots.length);
    dots.forEach((j, k) => {
      pos.set([x[j], y[j], 0], k * 3);
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
  const camera = new OrthographicCamera(-1, 1, 1, -1, -10, 10);
  camera.position.set(0, 0, 5);

  const controls = new MapControls(camera, renderer.domElement);
  controls.enableRotate = false;
  controls.screenSpacePanning = true;
  controls.zoomToCursor = true;
  controls.enableDamping = false;
  controls.minZoom = 0.5;
  controls.maxZoom = 500;

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

  const resize = () => {
    width = Math.max(1, container.clientWidth);
    height = Math.max(1, container.clientHeight);
    renderer.setSize(width, height);
    const aspect = width / height;
    const h = halfHeightFor(aspect);
    camera.left = -h * aspect;
    camera.right = h * aspect;
    camera.top = h;
    camera.bottom = -h;
    camera.updateProjectionMatrix();
    requestRender();
  };

  /** CSS px per world unit at the current zoom. */
  const pxPerWorld = () => (height * camera.zoom) / (camera.top - camera.bottom);
  const dotPx = () => Math.min(MAX_PX, BASE_PX * Math.pow(camera.zoom, ZOOM_GROWTH));

  const placeRing = (ring: Mesh, i: number | null) => {
    ring.visible = i !== null;
    if (i === null) return;
    ring.position.set(x[i], y[i], 0);
    ring.scale.setScalar((dotPx() * sizes[i]) / 2 / pxPerWorld() + 4 / pxPerWorld());
  };

  // ---- Render on demand ----
  let frame = 0;
  function requestRender() {
    if (frame || destroyed) return;
    frame = requestAnimationFrame(() => {
      frame = 0;
      material.uniforms.uPx.value = dotPx();
      overlayMaterial.uniforms.uPx.value = dotPx();
      placeRing(hoverRing, hovered?.index ?? highlighted);
      placeRing(selectRing, selection?.index ?? null);
      renderer.render(scene, camera);
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
  const cursor = new Vector3();
  const pick = (clientX: number, clientY: number): number => {
    const rect = renderer.domElement.getBoundingClientRect();
    cursor.set(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1, 0);
    cursor.unproject(camera);
    const reachPx = Math.max(PICK_PX, dotPx() * (1 + INFLUENCE_SIZE) / 2);
    return grid.nearest(cursor.x, cursor.y, reachPx / pxPerWorld());
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

  function resetView() {
    camera.position.set(0, 0, 5);
    controls.target.set(0, 0, 0);
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
    select,
    highlight,
    centerOn,
    resetView,
    destroy,
  };
}
