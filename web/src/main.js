import ForceGraph3D from '3d-force-graph';
import { resolveOverlaps } from './layout.js';
import { BackSide, Mesh, MeshBasicMaterial, SphereGeometry } from 'three';

// ---------------------------------------------------------------------------
// Layout constants (tweak these to change how the landscape looks)
// ---------------------------------------------------------------------------
const XY_SCALE = 300;    // UMAP coords arrive normalized to [-1, 1]; this sets world size
const USE_TIMELINE = false; // false = flat 2D map (z = 0). true = z-axis is publication date.
const Z_PER_YEAR = 12;   // world units per year along the timeline axis (only when USE_TIMELINE)
const FIT_PADDING = 1.08; // >1 leaves a margin around the map when fitting it to the window
const CLUSTER_FIT_PADDING = 1.25; // >1 leaves a margin around a cluster when flying to it
const CLUSTER_MIN_EXTENT = 30;    // never zoom in tighter than this (world units around the cluster center)
const DETAILS_PANEL_PX = 412;     // width the details panel covers on the right (panel + margins)
const FLY_MS = 700;               // camera animation time
const PAN_SLACK = 0.4;   // extra pan room past the map edge, as a fraction of half the visible area
                          // (0 = map edge stops at the window edge, 1 = map edge can reach screen center)

// Node sizes, as radii in world units (the map is ~600 units wide).
const R_MIN = 2;        // papers with the lowest influence (most have 0 citations)
const R_MAX = 30;       // the most influential paper
const NODE_PADDING = 0.6; // minimum empty gap kept between any two circles (world units)
const SIZE_CURVE = 1;   // 1 = radius grows evenly with score; <1 spreads low scores apart, >1 keeps them small
const NOISE_COLOR = 'rgba(140, 146, 165, 0.25)';
const DIMMED_COLOR = 'rgba(95, 100, 115, 0.3)'; // nodes outside the selected cluster
// Outline = a slightly bigger white sphere behind the node ("halo").
const HALO_WIDTH = 0.18; // outline thickness as a fraction of the node's radius
const HALO_MIN = 0.8;    // minimum outline thickness in world units, so tiny nodes still show one
// BackSide renders only the inside of the shell, so it never covers the node in front of it.
const HALO_MATERIAL = new MeshBasicMaterial({ color: 0xffffff, side: BackSide });
const CLUSTER_COLORS = [
  '#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
  '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac',
  '#86bcb6', '#d37295', '#fabfd2', '#b6992d', '#499894',
];

const clusterColor = (id) => (id === -1 ? NOISE_COLOR : CLUSTER_COLORS[id % CLUSTER_COLORS.length]);

const escapeHtml = (s) =>
  String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);

async function main() {
  // Files in web/public/ are served from the site root, so this is web/public/data/graph.json.
  const res = await fetch('/data/graph.json');
  const data = await res.json();
  const { meta, clusters } = data;
  const clusterTitle = Object.fromEntries(clusters.map((c) => [c.id, c.title]));
  const midYear = (meta.yearMin + meta.yearMax) / 2;

  // Pin every node to its ML-computed position. fx/fy/fz tell the force engine
  // "don't move this node", so the layout is the UMAP map, not a physics sim.
  const nodes = data.nodes.map((n) => {
    const x = n.x * XY_SCALE;
    const y = n.y * XY_SCALE;
    const z = USE_TIMELINE ? ((n.year ?? midYear) - midYear) * Z_PER_YEAR : 0;
    return { ...n, x, y, z, fx: x, fy: y, fz: z };
  });

  // One continuous scale: the lowest influence gets R_MIN, the highest gets R_MAX, and everything
  // in between is placed proportionally (SIZE_CURVE bends the middle of the scale).
  const influences = nodes.map((n) => n.influence);
  const minInf = Math.min(...influences);
  const maxInf = Math.max(...influences);
  nodes.forEach((n) => {
    const t = (n.influence - minInf) / (maxInf - minInf || 1);
    n.radius = R_MIN + (R_MAX - R_MIN) * Math.pow(t, SIZE_CURVE);
  });

  // UMAP doesn't know how big the circles are, so many overlap. Nudge them apart (minimal
  // movement, big papers move least), then re-pin each node at its final spot.
  const layoutStats = resolveOverlaps(nodes, { padding: NODE_PADDING });
  nodes.forEach((n) => { n.fx = n.x; n.fy = n.y; });
  console.log(
    `Overlap removal: avg move ${layoutStats.moved.toFixed(1)}, max move ${layoutStats.maxMoved.toFixed(1)}, ` +
    `overlaps left: ${layoutStats.overlaps ?? 'not checked'}`);

  // ---- Hover + cluster selection state ----
  let hovered = null;
  let selectedCluster = null; // cluster id, or null for the normal view

  // ---- Time window (set by the slider at the bottom) ----
  const T0 = meta.yearMin;
  const T1 = meta.yearMax;
  let windowLo = T0;
  let windowHi = T1;
  // Papers with no date only show while the window covers everything.
  const isVisible = (n) =>
    n.year == null ? windowLo <= T0 && windowHi >= T1 : n.year >= windowLo && n.year <= windowHi;

  const inSelection = (n) => selectedCluster !== null && n.cluster === selectedCluster;
  const nodeColor = (n) => (selectedCluster === null || inSelection(n) ? clusterColor(n.cluster) : DIMMED_COLOR);
  const updateHalo = (n) => { if (n?.halo) n.halo.visible = n === hovered || inSelection(n); };

  // A white sphere slightly larger than the node, centered on it. Hidden until hovered/selected.
  // Being a sphere, it stays concentric from any camera angle (so it will still work once the
  // timeline tilts the view).
  function makeHalo(n) {
    const r = n.radius + Math.max(HALO_MIN, n.radius * HALO_WIDTH);
    const halo = new Mesh(new SphereGeometry(r, 24, 16), HALO_MATERIAL);
    // The library rebuilds a node's objects when it re-enters the time window, so start in the right state.
    halo.visible = n === hovered || inSelection(n);
    halo.raycast = () => {}; // the halo shouldn't count as part of the node for hover/click
    n.halo = halo;
    return halo;
  }

  function setSelectedCluster(id) {
    selectedCluster = id;
    document.querySelectorAll('#cluster-list li').forEach((li) =>
      li.classList.toggle('active', Number(li.dataset.id) === id));
    graph.nodeColor(nodeColor); // re-applying the accessor recolors every node
    nodes.forEach(updateHalo);
    const sel = document.getElementById('selection');
    if (id === null) {
      sel.hidden = true;
    } else {
      const c = clusters.find((c) => c.id === id);
      sel.innerHTML = `<span class="swatch" style="background:${clusterColor(id)}"></span>` +
        `${escapeHtml(c?.title ?? `Cluster ${id}`)} · ${c?.size ?? '?'} papers<br/>` +
        `<span class="hint">Click the background to clear</span>`;
      sel.hidden = false;
    }
  }

  // ---- Paper details panel ----
  // Abstracts and authors live in a separate file (1.4 MB), fetched the first time it's needed.
  let papersPromise = null;
  const loadPapers = () => (papersPromise ??= fetch('/data/papers.json').then((r) => r.json()));
  const $ = (id) => document.getElementById(id);
  let detailsFor = null; // the node currently shown in the panel

  async function openDetails(n) {
    detailsFor = n;
    const c = clusters.find((c) => c.id === n.cluster);
    $('details-title').textContent = n.title;
    $('details-meta').textContent = `${n.date ?? 'Unknown date'} · ${n.category}`;
    $('details-stats').innerHTML =
      `<div><b>${n.citations}</b><span>citations</span></div>` +
      `<div><b>${n.influence.toFixed(2)}</b><span>influence</span></div>`;
    $('details-cluster').innerHTML =
      `<span class="swatch" style="background:${clusterColor(n.cluster)}"></span>` +
      escapeHtml(n.cluster === -1 ? 'Not in a cluster' : (c?.title ?? `Cluster ${n.cluster}`));
    $('details-authors').textContent = 'Loading…';
    $('details-abstract').textContent = 'Loading…';
    $('details-link').hidden = true;
    $('details').hidden = false;
    $('details').scrollTop = 0;

    try {
      const papers = await loadPapers();
      if (detailsFor !== n) return; // user clicked something else while this loaded
      const p = papers[n.id];
      if (!p) throw new Error('paper not found in papers.json');
      const MAX_AUTHORS = 8;
      const shown = p.authors.slice(0, MAX_AUTHORS).join(', ');
      const extra = p.authors.length - MAX_AUTHORS;
      $('details-authors').textContent = (shown || 'Unknown') + (extra > 0 ? ` and ${extra} more` : '');
      $('details-abstract').textContent = p.abstract || 'No abstract available.';
      $('details-link').href = `https://arxiv.org/abs/${p.arxivId}`;
      $('details-link').hidden = false;
    } catch (err) {
      console.error(err);
      if (detailsFor !== n) return;
      $('details-authors').textContent = 'Could not load details.';
      $('details-abstract').textContent = '';
    }
  }

  function closeDetails() {
    detailsFor = null;
    $('details').hidden = true;
  }
  $('details-close').addEventListener('click', closeDetails);
  window.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeDetails(); });

  // Clicking any paper opens its details. What happens to the cluster selection:
  //   paper inside the selected cluster -> keep the selection and the view, just switch the details
  //   anything else                     -> select that paper's cluster and fly the camera to it
  //                                        (noise papers have no cluster, so they just clear the selection)
  function onNodeClick(n) {
    if (!inSelection(n)) {
      const next = n.cluster === -1 ? null : n.cluster;
      setSelectedCluster(next);
      if (next !== null) flyToCluster(next);
    }
    openDetails(n);
  }

  // Center the view on a cluster's average position and zoom so all of it fits.
  function flyToCluster(id) {
    const inCluster = nodes.filter((n) => n.cluster === id);
    const members = inCluster.filter(isVisible); // fit what's actually drawn if the time window hides some
    const pts = members.length ? members : inCluster;
    const cx = pts.reduce((s, n) => s + n.x, 0) / pts.length;
    const cy = pts.reduce((s, n) => s + n.y, 0) / pts.length;
    // Half-extent around the mean, so every member is inside the view; + radius so big dots aren't clipped.
    const extX = Math.max(...pts.map((n) => Math.abs(n.x - cx) + n.radius), CLUSTER_MIN_EXTENT);
    const extY = Math.max(...pts.map((n) => Math.abs(n.y - cy) + n.radius), CLUSTER_MIN_EXTENT);

    // The details panel covers the right edge on wide screens; fit and center within the space left of it.
    const reserved = window.innerWidth >= 900 ? DETAILS_PANEL_PX : 0;
    const viewAspect = (window.innerWidth - reserved) / window.innerHeight;
    const t = Math.tan((graph.camera().fov * Math.PI) / 360);
    const wanted = CLUSTER_FIT_PADDING * Math.max(extY / t, extX / (t * viewAspect));
    const distance = Math.min(controls.maxDistance, Math.max(controls.minDistance, wanted));

    // Shift the camera right so the cluster lands in the middle of the uncovered area.
    const unitsPerPx = (2 * distance * t) / window.innerHeight;
    const x = cx + (reserved / 2) * unitsPerPx;
    flyCamera({ x, y: cy, z: distance }, FLY_MS);
  }

  // Orbit controls let us switch rotate/pan off individually (the default trackball can't clamp zoom).
  const graph = ForceGraph3D({ controlType: 'orbit' })(document.getElementById('graph'))
    .backgroundColor('#0b0d12')
    .graphData({ nodes, links: data.links })
    .nodeColor(nodeColor)
    .nodeVisibility(isVisible)
    .linkVisibility((l) => isVisible(l.source) && isVisible(l.target))
    // Opaque nodes, so the white halo behind them doesn't show through. Colors with their own
    // alpha (noise, dimmed) stay see-through.
    .nodeOpacity(1)
    .nodeThreeObject(makeHalo)
    .nodeThreeObjectExtend(true) // keep the default sphere, add the halo to it
    .onNodeHover((n, prev) => { hovered = n; updateHalo(prev); updateHalo(n); })
    .onNodeClick(onNodeClick)
    .onBackgroundClick(() => { setSelectedCluster(null); closeDetails(); })
    // The library's radius = cbrt(nodeVal) * nodeRelSize, so with relSize 1, nodeVal = radius³.
    .nodeVal((n) => n.radius ** 3)
    .nodeRelSize(1)
    .nodeResolution(12)
    .enableNodeDrag(false)
    .nodeLabel((n) => `
      <b>${escapeHtml(n.title)}</b><br/>
      ${escapeHtml(n.date ?? 'unknown date')} · ${escapeHtml(n.category)}<br/>
      ${n.citations} citations · influence ${n.influence.toFixed(2)}<br/>
      <span style="color:${clusterColor(n.cluster)}">${escapeHtml(clusterTitle[n.cluster] ?? '')}</span>`)
    .linkColor(() => 'rgba(255, 255, 255, 0.5)')
    .linkDirectionalArrowLength(4)
    .linkDirectionalArrowRelPos(1)
    // Positions are fixed, so there is nothing for the physics engine to do.
    .warmupTicks(0)
    .cooldownTicks(0);

  // Flat map: look straight down the z-axis. No rotating; drag to pan, scroll to zoom.
  const controls = graph.controls();
  controls.enableRotate = false;
  controls.enablePan = true;
  controls.screenSpacePanning = true; // pan moves in the screen plane (= the map plane here)
  // Three.js MOUSE / TOUCH enum values: MOUSE.PAN = 2, MOUSE.DOLLY = 1, TOUCH.PAN = 1, TOUCH.DOLLY_PAN = 2
  controls.mouseButtons = { LEFT: 2, MIDDLE: 1, RIGHT: 2 };
  controls.touches = { ONE: 1, TWO: 2 };

  // Bounding box of the map, centered on its middle.
  const xs = nodes.map((n) => n.x);
  const ys = nodes.map((n) => n.y);
  const center = { x: (Math.min(...xs) + Math.max(...xs)) / 2, y: (Math.min(...ys) + Math.max(...ys)) / 2 };
  const halfW = (Math.max(...xs) - Math.min(...xs)) / 2;
  const halfH = (Math.max(...ys) - Math.min(...ys)) / 2;

  const camera = graph.camera();
  const tanHalfFov = () => Math.tan((camera.fov * Math.PI) / 360);
  const aspect = () => window.innerWidth / window.innerHeight; // camera.aspect updates a tick late on resize

  // Camera distance at which the whole map just fits the window, given the camera's field of view.
  function fitToWindow() {
    flight = null;
    graph.width(window.innerWidth).height(window.innerHeight);
    const distance = FIT_PADDING * Math.max(halfH / tanHalfFov(), halfW / (tanHalfFov() * aspect()));
    graph.cameraPosition({ x: center.x, y: center.y, z: distance }, { x: center.x, y: center.y, z: 0 });
    controls.maxDistance = distance; // can't zoom out past the full map
    controls.minDistance = distance * 0.05;
  }

  // Smooth camera move that keeps the camera straight above its target the whole way.
  // (The library's own cameraPosition() animation moves the camera and its look-at point on
  // different timers, which tilts the view mid-flight.) Zoom is interpolated in log space so it
  // feels even; any user pan/zoom cancels the flight.
  let flight = null;
  function flyCamera(to, ms) {
    const from = { x: camera.position.x, y: camera.position.y, z: camera.position.z };
    const start = performance.now();
    const mine = (flight = {});
    const step = (now) => {
      if (flight !== mine) return; // cancelled or superseded
      const p = Math.min(1, (now - start) / ms);
      const e = 1 - Math.pow(1 - p, 3); // ease-out cubic
      const x = from.x + (to.x - from.x) * e;
      const y = from.y + (to.y - from.y) * e;
      const z = from.z * Math.pow(to.z / from.z, e);
      camera.position.set(x, y, z);
      controls.target.set(x, y, 0);
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
    const roomX = Math.max(0, halfW * FIT_PADDING - viewHalfW) + PAN_SLACK * viewHalfW;
    const roomY = Math.max(0, halfH * FIT_PADDING - viewHalfH) + PAN_SLACK * viewHalfH;
    const x = Math.min(center.x + roomX, Math.max(center.x - roomX, controls.target.x));
    const y = Math.min(center.y + roomY, Math.max(center.y - roomY, controls.target.y));
    if (x === controls.target.x && y === controls.target.y) return;
    // Keep the camera directly above its target so the view stays flat.
    camera.position.x = controls.target.x = x;
    camera.position.y = controls.target.y = y;
  }

  fitToWindow();
  controls.addEventListener('change', clampPan); // fires on every pan and zoom
  window.addEventListener('resize', () => { fitToWindow(); clampPan(); });

  // The library's built-in hint still describes rotating, which is turned off.
  const navHint = document.querySelector('.scene-nav-info');
  if (navHint) navHint.textContent = 'Left or Right-click: pan, Mouse-wheel/middle-click: zoom';

  // Cluster list in the top-left panel: biggest first. Click one to select it and fly there;
  // click the active one again to clear. (Noise isn't a real cluster, so it isn't listed.)
  const list = $('cluster-list');
  [...clusters]
    .filter((c) => c.id !== -1)
    .sort((a, b) => b.size - a.size)
    .forEach((c) => {
      const name = c.title.replace(/^Cluster \d+:\s*/, ''); // drop the "Cluster 5:" prefix
      const li = document.createElement('li');
      li.dataset.id = c.id;
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.title = c.title;
      btn.innerHTML =
        `<span class="swatch" style="background:${clusterColor(c.id)}"></span>` +
        `<span class="name">${escapeHtml(name)}</span><span class="count">${c.size}</span>`;
      btn.addEventListener('click', () => {
        if (selectedCluster === c.id) {
          setSelectedCluster(null);
          return;
        }
        setSelectedCluster(c.id);
        closeDetails(); // the open paper may belong to a different cluster
        flyToCluster(c.id);
      });
      li.appendChild(btn);
      list.appendChild(li);
    });

  const timeline = setupTimeline();

  // Back to the starting state: whole map in view, nothing selected, nothing filtered.
  $('tl-reset').addEventListener('click', () => {
    setSelectedCluster(null);
    closeDetails();
    timeline.resetWindow();
    flyCamera({ x: center.x, y: center.y, z: controls.maxDistance }, FLY_MS); // maxDistance = whole map fits
  });

  // Two-handle slider: drag either end inward to hide papers outside that date.
  function setupTimeline() {
    const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const fmt = (y) => `${MONTHS[Math.min(11, Math.floor((y - Math.floor(y)) * 12))]} ${Math.floor(y)}`;
    const span = T1 - T0;
    const MIN_GAP = span * 0.01; // handles can't cross or fully overlap
    const pct = (y) => `${((y - T0) / span) * 100}%`;

    const panel = document.getElementById('timeline');
    const track = document.getElementById('tl-track');
    const loEl = document.getElementById('tl-lo');
    const hiEl = document.getElementById('tl-hi');
    const rangeEl = document.getElementById('tl-range');
    const labelEl = document.getElementById('tl-label');

    // One bar per year, with two views of the same bars (flip with the switch):
    //   volume     = papers that year (sqrt-scaled so the 2025-26 spike doesn't flatten the rest)
    //   importance = average influence score of that year's papers (the value that sets node size).
    //                Average rather than total, so it doesn't just repeat the volume chart.
    const byYear = new Map();
    nodes.forEach((n) => {
      if (n.year == null) return;
      const y = Math.floor(n.year);
      const e = byYear.get(y) ?? { count: 0, influence: 0 };
      e.count += 1;
      e.influence += n.influence;
      byYear.set(y, e);
    });
    const yearStats = [...byYear].map(([year, e]) => ({ year, count: e.count, importance: e.influence / e.count }));
    const maxCount = Math.max(...yearStats.map((s) => s.count));
    const maxImportance = Math.max(...yearStats.map((s) => s.importance));

    const bars = document.getElementById('tl-bars');
    const barEls = yearStats.map((s) => {
      const a = Math.max(s.year, T0);
      const b = Math.min(s.year + 1, T1);
      const el = document.createElement('div');
      el.style.left = pct(a);
      el.style.width = `calc(${((b - a) / span) * 100}% - 1px)`;
      bars.appendChild(el);
      return { el, s };
    });

    let barMode = 'volume';
    function styleBars() {
      barEls.forEach(({ el, s }) => {
        const h = barMode === 'volume' ? Math.sqrt(s.count / maxCount) : s.importance / maxImportance;
        el.style.height = `${Math.max(6, h * 100)}%`;
        el.classList.toggle('importance', barMode === 'importance');
        el.title = `${s.year}: ${s.count} papers · avg importance ${s.importance.toFixed(2)}`;
      });
    }
    styleBars();

    const modeBtn = document.getElementById('tl-mode');
    modeBtn.addEventListener('click', () => {
      barMode = barMode === 'volume' ? 'importance' : 'volume';
      modeBtn.setAttribute('aria-checked', String(barMode === 'importance'));
      document.getElementById('tl-mode-text').textContent = barMode === 'volume' ? 'Paper volume' : 'Paper importance';
      styleBars();
    });

    // Year ticks every 5 years.
    const ticks = document.getElementById('tl-ticks');
    for (let y = Math.ceil(T0 / 5) * 5; y <= T1; y += 5) {
      const t = document.createElement('span');
      t.style.left = pct(y);
      t.textContent = y;
      ticks.appendChild(t);
    }

    let queued = false;
    function render() {
      loEl.style.left = pct(windowLo);
      hiEl.style.left = pct(windowHi);
      rangeEl.style.left = pct(windowLo);
      rangeEl.style.width = `${((windowHi - windowLo) / span) * 100}%`;
      const shown = nodes.filter(isVisible).length;
      labelEl.innerHTML = `${fmt(windowLo)} – ${fmt(windowHi)} <span class="hint">· ${shown} of ${nodes.length} papers</span>`;
      // Re-apply the accessors so the library re-checks which nodes/links to draw.
      graph.nodeVisibility(isVisible).linkVisibility((l) => isVisible(l.source) && isVisible(l.target));
    }
    const scheduleRender = () => {
      if (queued) return;
      queued = true;
      requestAnimationFrame(() => { queued = false; render(); });
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
        scheduleRender();
      });
      const release = () => el.classList.remove('dragging');
      el.addEventListener('pointerup', release);
      el.addEventListener('pointercancel', release);
    }
    bindHandle(loEl, 'lo');
    bindHandle(hiEl, 'hi');

    panel.hidden = false;
    render();

    return {
      resetWindow() {
        windowLo = T0;
        windowHi = T1;
        render();
      },
    };
  }

  document.getElementById('stats').textContent =
    `${meta.paperCount}${meta.totalPapers > meta.paperCount ? ` of ${meta.totalPapers.toLocaleString()}` : ''} papers · ${meta.linkCount} citation links · ${clusters.filter((c) => c.id !== -1).length} clusters · ` +
    `${Math.floor(meta.yearMin)}–${Math.floor(meta.yearMax)}`;

  // Handy for poking at things from the browser console while we build.
  window.graph = graph;
}

main().catch((err) => {
  console.error(err);
  document.getElementById('stats').textContent = `Failed to load data: ${err.message}`;
});
