import { useCallback, useEffect, useImperativeHandle, useMemo, useRef } from 'react';
import ForceGraph3D, { type ForceGraphMethods, type LinkObject } from 'react-force-graph-3d';
import { BackSide, Mesh, MeshBasicMaterial, SphereGeometry, Vector3, type PerspectiveCamera } from 'three';
import {
  CLUSTER_FIT_PADDING,
  CLUSTER_MIN_EXTENT,
  DETAILS_PANEL_PX,
  DIMMED_COLOR,
  FIT_PADDING,
  FLY_MS,
  HALO_MIN,
  HALO_WIDTH,
  PAN_SLACK,
  TIMELINE_VIEW_DIR,
  clusterColor,
} from '../constants';
import {
  MOUSE,
  TOUCH,
  createFlightController,
  type CameraPose,
  type Coords,
  type FlightController,
  type OrbitLikeControls,
} from '../lib/camera';
import { computeBounds, type Bounds } from '../lib/graph';
import { escapeHtml } from '../lib/html';
import type { PaperNode, RawLink, ResolvedLink, ViewMode } from '../types';

// Outline = a slightly bigger white sphere behind the node ("halo").
// BackSide renders only the inside of the shell, so it never covers the node in front of it.
const HALO_MATERIAL = new MeshBasicMaterial({ color: 0xffffff, side: BackSide });

const tanHalfFov = (camera: PerspectiveCamera) => Math.tan((camera.fov * Math.PI) / 360);
// camera.aspect updates a tick late on resize, so measure the window instead.
const windowAspect = () => window.innerWidth / window.innerHeight;

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/**
 * Distance at which a sphere of `radius` fits the view. The narrower of the two screen
 * dimensions is the binding one, hence min(1, aspect).
 */
function distanceToFit(radius: number, camera: PerspectiveCamera, aspect: number, padding: number) {
  return (padding * radius) / (tanHalfFov(camera) * Math.min(1, aspect));
}

/** Unit vector from the target toward the camera — the direction the map is being viewed from. */
function viewDirection(camera: PerspectiveCamera, controls: OrbitLikeControls): Coords {
  const dx = camera.position.x - controls.target.x;
  const dy = camera.position.y - controls.target.y;
  const dz = camera.position.z - controls.target.z;
  const len = Math.hypot(dx, dy, dz) || 1;
  return { x: dx / len, y: dy / len, z: dz / len };
}

/** The pose that frames the whole map: straight down in flat mode, tilted in timeline mode. */
function homePose(bounds: Bounds, camera: PerspectiveCamera, mode: ViewMode): CameraPose {
  const { center } = bounds;
  if (mode === 'flat') {
    const t = tanHalfFov(camera);
    const distance =
      FIT_PADDING * Math.max(bounds.halfH / t, bounds.halfW / (t * windowAspect()));
    return { position: { x: center.x, y: center.y, z: distance }, target: { ...center, z: 0 } };
  }
  const distance = distanceToFit(bounds.radius, camera, windowAspect(), FIT_PADDING);
  const d = TIMELINE_VIEW_DIR;
  const len = Math.hypot(d.x, d.y, d.z);
  return {
    position: {
      x: center.x + (d.x / len) * distance,
      y: center.y + (d.y / len) * distance,
      z: center.z + (d.z / len) * distance,
    },
    target: { ...center },
  };
}

export interface ResearchGraphHandle {
  /** Center the view on a cluster's average position and zoom so all of it fits. */
  flyToCluster(id: number): void;
  /** Pull back until the whole map is in view again. */
  resetView(): void;
}

interface Props {
  nodes: PaperNode[];
  links: RawLink[];
  mode: ViewMode;
  clusterTitle: Record<number, string>;
  selectedCluster: number | null;
  isVisible: (n: PaperNode) => boolean;
  width: number;
  height: number;
  onNodeClick: (node: PaperNode) => void;
  onBackgroundClick: () => void;
  ref?: React.Ref<ResearchGraphHandle>;
}

export function ResearchGraph({
  nodes,
  links,
  mode,
  clusterTitle,
  selectedCluster,
  isVisible,
  width,
  height,
  onNodeClick,
  onBackgroundClick,
  ref,
}: Props) {
  const fgRef = useRef<ForceGraphMethods<PaperNode, RawLink> | undefined>(undefined);
  const flightRef = useRef<FlightController | null>(null);

  // Hover and the current selection are read from inside imperative three.js callbacks, so they
  // live in refs as well as props: re-rendering React on every mousemove would be far too costly.
  const hoveredRef = useRef<PaperNode | null>(null);
  const selectedRef = useRef(selectedCluster);
  const isVisibleRef = useRef(isVisible);
  useEffect(() => {
    selectedRef.current = selectedCluster;
  }, [selectedCluster]);
  useEffect(() => {
    isVisibleRef.current = isVisible;
  }, [isVisible]);

  // Bounding box of the map, centered on its middle.
  const bounds = useMemo(() => computeBounds(nodes), [nodes]);

  // Stable identity, or the library re-ingests the whole dataset on every render.
  const graphData = useMemo(() => ({ nodes, links }), [nodes, links]);

  // ---- Node appearance -----------------------------------------------------
  const inSelection = useCallback(
    (n: PaperNode | null | undefined) =>
      n != null && selectedCluster !== null && n.cluster === selectedCluster,
    [selectedCluster],
  );

  const nodeColor = useCallback(
    (n: PaperNode) =>
      selectedCluster === null || n.cluster === selectedCluster ? clusterColor(n.cluster) : DIMMED_COLOR,
    [selectedCluster],
  );

  // A white sphere slightly larger than the node, centered on it. Hidden until hovered/selected.
  // Being a sphere, it stays concentric from any camera angle (so it still works once the
  // timeline tilts the view). Kept identity-stable: changing this accessor would make the
  // library rebuild every node's three.js objects.
  const makeHalo = useCallback((n: PaperNode) => {
    const r = n.radius + Math.max(HALO_MIN, n.radius * HALO_WIDTH);
    const halo = new Mesh(new SphereGeometry(r, 24, 16), HALO_MATERIAL);
    // The library rebuilds a node's objects when it re-enters the time window, so start in the right state.
    halo.visible =
      n === hoveredRef.current || (selectedRef.current !== null && n.cluster === selectedRef.current);
    halo.raycast = () => {}; // the halo shouldn't count as part of the node for hover/click
    n.halo = halo;
    return halo;
  }, []);

  const updateHalo = useCallback(
    (n: PaperNode | null | undefined) => {
      if (n?.halo) n.halo.visible = n === hoveredRef.current || inSelection(n);
    },
    [inSelection],
  );

  const handleHover = useCallback(
    (n: PaperNode | null, prev: PaperNode | null) => {
      hoveredRef.current = n;
      updateHalo(prev);
      updateHalo(n);
    },
    [updateHalo],
  );

  // Selecting a cluster outlines its members.
  useEffect(() => {
    for (const n of nodes) updateHalo(n);
  }, [nodes, updateHalo]);

  const nodeLabel = useCallback(
    (n: PaperNode) => `
      <b>${escapeHtml(n.title)}</b><br/>
      ${escapeHtml(n.date ?? 'unknown date')} · ${escapeHtml(n.category)}<br/>
      ${n.citations} citations · influence ${n.influence.toFixed(2)}<br/>
      <span style="color:${clusterColor(n.cluster)}">${escapeHtml(clusterTitle[n.cluster] ?? '')}</span>`,
    [clusterTitle],
  );

  // The library's radius = cbrt(nodeVal) * nodeRelSize, so with relSize 1, nodeVal = radius³.
  const nodeVal = useCallback((n: PaperNode) => n.radius ** 3, []);
  const nodeVisibility = useCallback((n: PaperNode) => isVisible(n), [isVisible]);
  // The library rewrites each link's source/target from an id string to the node object as it
  // ingests the data, but its types still allow the id form, so narrow it here.
  const linkVisibility = useCallback(
    (l: LinkObject<PaperNode, RawLink>) => {
      const { source, target } = l as unknown as ResolvedLink;
      return isVisible(source) && isVisible(target);
    },
    [isVisible],
  );

  // ---- Camera --------------------------------------------------------------
  // Flat mode keeps the camera square to the map, so "distance" is just camera.position.z and
  // panning is clamped in x/y. Timeline mode lets the camera orbit, so neither holds: distance
  // is measured to the orbit target, and it's the target that gets kept inside the map.
  const clampPan = useCallback(() => {
    const graph = fgRef.current;
    if (!graph || flightRef.current?.isFlying()) return; // the fly-to animation drives the camera
    const camera = graph.camera() as PerspectiveCamera; //  itself; clamping mid-flight causes jitter
    const controls = graph.controls() as OrbitLikeControls;
    const { center } = bounds;

    if (mode === 'flat') {
      // Pan bound: the visible area may go past the (padded) map edges by PAN_SLACK of a half-view.
      // The further you zoom in, the further from the center you can go, and you can never
      // pan far enough to lose the map entirely.
      const viewHalfH = camera.position.z * tanHalfFov(camera);
      const viewHalfW = viewHalfH * windowAspect();
      const roomX = Math.max(0, bounds.halfW * FIT_PADDING - viewHalfW) + PAN_SLACK * viewHalfW;
      const roomY = Math.max(0, bounds.halfH * FIT_PADDING - viewHalfH) + PAN_SLACK * viewHalfH;
      const x = clamp(controls.target.x, center.x - roomX, center.x + roomX);
      const y = clamp(controls.target.y, center.y - roomY, center.y + roomY);
      if (x === controls.target.x && y === controls.target.y) return;
      // Keep the camera directly above its target so the view stays flat.
      camera.position.x = controls.target.x = x;
      camera.position.y = controls.target.y = y;
      return;
    }

    // Timeline mode: keep the point being orbited inside the map's bounding box (plus slack), and
    // carry the camera along by the same delta so the viewing angle and zoom survive the clamp.
    const slack = 1 + PAN_SLACK;
    const dx = clamp(controls.target.x, center.x - bounds.halfW * slack, center.x + bounds.halfW * slack) - controls.target.x;
    const dy = clamp(controls.target.y, center.y - bounds.halfH * slack, center.y + bounds.halfH * slack) - controls.target.y;
    const dz = clamp(controls.target.z, center.z - bounds.halfD * slack, center.z + bounds.halfD * slack) - controls.target.z;
    if (dx === 0 && dy === 0 && dz === 0) return;
    controls.target.set(controls.target.x + dx, controls.target.y + dy, controls.target.z + dz);
    camera.position.set(camera.position.x + dx, camera.position.y + dy, camera.position.z + dz);
  }, [bounds, mode]);

  /** Snap (no animation) to the view that frames the whole map. */
  const fitToWindow = useCallback(() => {
    const graph = fgRef.current;
    if (!graph) return;
    flightRef.current?.cancel();
    const camera = graph.camera() as PerspectiveCamera;
    const controls = graph.controls() as OrbitLikeControls;
    const pose = homePose(bounds, camera, mode);
    graph.cameraPosition(pose.position, pose.target);
    // Can't zoom out past the whole map.
    controls.maxDistance = Math.hypot(
      pose.position.x - pose.target.x,
      pose.position.y - pose.target.y,
      pose.position.z - pose.target.z,
    );
    controls.minDistance = controls.maxDistance * 0.05;
  }, [bounds, mode]);

  const flyToCluster = useCallback(
    (id: number) => {
      const graph = fgRef.current;
      const flight = flightRef.current;
      if (!graph || !flight) return;
      const inCluster = nodes.filter((n) => n.cluster === id);
      if (!inCluster.length) return;
      // Fit what's actually drawn if the time window hides some of the cluster.
      const members = inCluster.filter((n) => isVisibleRef.current(n));
      const pts = members.length ? members : inCluster;

      const cx = pts.reduce((s, n) => s + n.x, 0) / pts.length;
      const cy = pts.reduce((s, n) => s + n.y, 0) / pts.length;
      const cz = pts.reduce((s, n) => s + n.z, 0) / pts.length;

      const camera = graph.camera() as PerspectiveCamera;
      const controls = graph.controls() as OrbitLikeControls;
      // The details panel covers the right edge on wide screens; fit and center within the space left of it.
      const reserved = window.innerWidth >= 900 ? DETAILS_PANEL_PX : 0;
      const viewAspect = (window.innerWidth - reserved) / window.innerHeight;
      const t = tanHalfFov(camera);

      let distance: number;
      if (mode === 'flat') {
        // Half-extent around the mean, so every member is inside the view; + radius so big dots aren't clipped.
        let extX = CLUSTER_MIN_EXTENT;
        let extY = CLUSTER_MIN_EXTENT;
        for (const n of pts) {
          extX = Math.max(extX, Math.abs(n.x - cx) + n.radius);
          extY = Math.max(extY, Math.abs(n.y - cy) + n.radius);
        }
        distance = CLUSTER_FIT_PADDING * Math.max(extY / t, extX / (t * viewAspect));
      } else {
        // Tilted view, so fit the sphere around the cluster rather than an axis-aligned box.
        let ext = CLUSTER_MIN_EXTENT;
        for (const n of pts) {
          ext = Math.max(ext, Math.hypot(n.x - cx, n.y - cy, n.z - cz) + n.radius);
        }
        distance = distanceToFit(ext, camera, viewAspect, CLUSTER_FIT_PADDING);
      }
      distance = clamp(distance, controls.minDistance, controls.maxDistance);

      // Shift the view right so the cluster lands in the middle of the uncovered area. In flat
      // mode "right" is +x; once the camera can orbit it's the camera's own right vector.
      const unitsPerPx = (2 * distance * t) / window.innerHeight;
      const nudge = (reserved / 2) * unitsPerPx;
      const target =
        mode === 'flat'
          ? { x: cx + nudge, y: cy, z: 0 }
          : (() => {
              const right = new Vector3().setFromMatrixColumn(camera.matrixWorld, 0).normalize();
              return { x: cx + right.x * nudge, y: cy + right.y * nudge, z: cz + right.z * nudge };
            })();

      // Keep looking from wherever the user has rotated to; only move and zoom.
      const dir = mode === 'flat' ? { x: 0, y: 0, z: 1 } : viewDirection(camera, controls);
      flight.flyTo(
        {
          target,
          position: {
            x: target.x + dir.x * distance,
            y: target.y + dir.y * distance,
            z: target.z + dir.z * distance,
          },
        },
        FLY_MS,
      );
    },
    [nodes, mode],
  );

  const resetView = useCallback(() => {
    const graph = fgRef.current;
    const flight = flightRef.current;
    if (!graph || !flight) return;
    flight.flyTo(homePose(bounds, graph.camera() as PerspectiveCamera, mode), FLY_MS);
  }, [bounds, mode]);

  useImperativeHandle(ref, () => ({ flyToCluster, resetView }), [flyToCluster, resetView]);

  // One-time wiring of the three.js controls that the React props don't cover.
  useEffect(() => {
    const graph = fgRef.current;
    if (!graph) return;
    const camera = graph.camera() as PerspectiveCamera;
    const controls = graph.controls() as OrbitLikeControls;
    const flight = createFlightController(camera, controls);
    flightRef.current = flight;

    controls.enablePan = true;
    controls.screenSpacePanning = true; // pan moves in the screen plane
    if (mode === 'flat') {
      // Look straight down the z-axis. No rotating; drag to pan, scroll to zoom.
      controls.enableRotate = false;
      controls.mouseButtons = { LEFT: MOUSE.PAN, MIDDLE: MOUSE.DOLLY, RIGHT: MOUSE.PAN };
      controls.touches = { ONE: TOUCH.PAN, TWO: TOUCH.DOLLY_PAN };
    } else {
      // The whole point of the timeline is seeing the map from an angle, so drag now orbits
      // and panning moves to the right button.
      controls.enableRotate = true;
      controls.mouseButtons = { LEFT: MOUSE.ROTATE, MIDDLE: MOUSE.DOLLY, RIGHT: MOUSE.PAN };
      controls.touches = { ONE: TOUCH.ROTATE, TWO: TOUCH.DOLLY_PAN };
    }
    controls.addEventListener('change', clampPan); // fires on every pan, orbit and zoom

    const cancelFlight = () => flight.cancel();
    const canvas = graph.renderer().domElement;
    canvas.addEventListener('pointerdown', cancelFlight);
    canvas.addEventListener('wheel', cancelFlight, { passive: true });

    // The library's built-in hint doesn't match either of our control schemes.
    const navHint = document.querySelector('.scene-nav-info');
    if (navHint) {
      navHint.textContent =
        mode === 'flat'
          ? 'Left or Right-click: pan, Mouse-wheel/middle-click: zoom'
          : 'Left-click: rotate, Right-click: pan, Mouse-wheel/middle-click: zoom';
    }

    // Handy for poking at things from the browser console while we build. The React ref exposes
    // the library's methods but not its data, so hand over the node array too.
    Object.assign(window, { graph, graphNodes: nodes });

    return () => {
      controls.removeEventListener('change', clampPan);
      canvas.removeEventListener('pointerdown', cancelFlight);
      canvas.removeEventListener('wheel', cancelFlight);
      flightRef.current = null;
    };
  }, [clampPan, mode, nodes]);

  // Frame the map on load, on resize, and whenever the view mode changes.
  useEffect(() => {
    fitToWindow();
    clampPan();
  }, [width, height, fitToWindow, clampPan]);

  // The wrapper carries #graph, which the stylesheet pins to the viewport.
  return (
    <div id="graph">
      <ForceGraph3D<PaperNode, RawLink>
        ref={fgRef}
        controlType="orbit" // orbit lets us switch rotate/pan off individually (trackball can't clamp zoom)
        width={width}
        height={height}
        backgroundColor="#0b0d12"
        graphData={graphData}
        nodeColor={nodeColor}
        nodeVisibility={nodeVisibility}
        linkVisibility={linkVisibility}
        // Opaque nodes, so the white halo behind them doesn't show through. Colors with their own
        // alpha (noise, dimmed) stay see-through.
        nodeOpacity={1}
        nodeThreeObject={makeHalo}
        nodeThreeObjectExtend // keep the default sphere, add the halo to it
        nodeVal={nodeVal}
        nodeRelSize={1}
        nodeResolution={12}
        nodeLabel={nodeLabel}
        enableNodeDrag={false}
        onNodeHover={handleHover}
        onNodeClick={onNodeClick}
        onBackgroundClick={onBackgroundClick}
        linkColor={() => 'rgba(255, 255, 255, 0.5)'}
        linkDirectionalArrowLength={4}
        linkDirectionalArrowRelPos={1}
        // Positions are fixed, so there is nothing for the physics engine to do.
        warmupTicks={0}
        cooldownTicks={0}
      />
    </div>
  );
}
