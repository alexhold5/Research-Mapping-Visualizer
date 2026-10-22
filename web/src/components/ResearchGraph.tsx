import { useCallback, useEffect, useImperativeHandle, useMemo, useRef } from 'react';
import ForceGraph3D, { type ForceGraphMethods, type LinkObject } from 'react-force-graph-3d';
import { BackSide, Mesh, MeshBasicMaterial, SphereGeometry, type PerspectiveCamera } from 'three';
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
  clusterColor,
} from '../constants';
import { createFlightController, type FlightController, type OrbitLikeControls } from '../lib/camera';
import { computeBounds } from '../lib/graph';
import { escapeHtml } from '../lib/html';
import type { PaperNode, RawLink, ResolvedLink } from '../types';

// Outline = a slightly bigger white sphere behind the node ("halo").
// BackSide renders only the inside of the shell, so it never covers the node in front of it.
const HALO_MATERIAL = new MeshBasicMaterial({ color: 0xffffff, side: BackSide });

const tanHalfFov = (camera: PerspectiveCamera) => Math.tan((camera.fov * Math.PI) / 360);
// camera.aspect updates a tick late on resize, so measure the window instead.
const windowAspect = () => window.innerWidth / window.innerHeight;

export interface ResearchGraphHandle {
  /** Center the view on a cluster's average position and zoom so all of it fits. */
  flyToCluster(id: number): void;
  /** Pull back until the whole map is in view again. */
  resetView(): void;
}

interface Props {
  nodes: PaperNode[];
  links: RawLink[];
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
  // Pan bound: the visible area may go past the (padded) map edges by PAN_SLACK of a half-view.
  // The further you zoom in, the further from the center you can go, and you can never
  // pan far enough to lose the map entirely.
  const clampPan = useCallback(() => {
    const graph = fgRef.current;
    if (!graph || flightRef.current?.isFlying()) return; // the fly-to animation drives the camera
    const camera = graph.camera() as PerspectiveCamera; //  itself; clamping mid-flight causes jitter
    const controls = graph.controls() as OrbitLikeControls;
    const viewHalfH = camera.position.z * tanHalfFov(camera);
    const viewHalfW = viewHalfH * windowAspect();
    const roomX = Math.max(0, bounds.halfW * FIT_PADDING - viewHalfW) + PAN_SLACK * viewHalfW;
    const roomY = Math.max(0, bounds.halfH * FIT_PADDING - viewHalfH) + PAN_SLACK * viewHalfH;
    const x = Math.min(bounds.center.x + roomX, Math.max(bounds.center.x - roomX, controls.target.x));
    const y = Math.min(bounds.center.y + roomY, Math.max(bounds.center.y - roomY, controls.target.y));
    if (x === controls.target.x && y === controls.target.y) return;
    // Keep the camera directly above its target so the view stays flat.
    camera.position.x = controls.target.x = x;
    camera.position.y = controls.target.y = y;
  }, [bounds]);

  /** Camera distance at which the whole map just fits the window, given the camera's field of view. */
  const fitToWindow = useCallback(() => {
    const graph = fgRef.current;
    if (!graph) return;
    flightRef.current?.cancel();
    const camera = graph.camera() as PerspectiveCamera;
    const controls = graph.controls() as OrbitLikeControls;
    const t = tanHalfFov(camera);
    const distance = FIT_PADDING * Math.max(bounds.halfH / t, bounds.halfW / (t * windowAspect()));
    graph.cameraPosition(
      { x: bounds.center.x, y: bounds.center.y, z: distance },
      { x: bounds.center.x, y: bounds.center.y, z: 0 },
    );
    controls.maxDistance = distance; // can't zoom out past the full map
    controls.minDistance = distance * 0.05;
  }, [bounds]);

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
      // Half-extent around the mean, so every member is inside the view; + radius so big dots aren't clipped.
      let extX = CLUSTER_MIN_EXTENT;
      let extY = CLUSTER_MIN_EXTENT;
      for (const n of pts) {
        extX = Math.max(extX, Math.abs(n.x - cx) + n.radius);
        extY = Math.max(extY, Math.abs(n.y - cy) + n.radius);
      }

      // The details panel covers the right edge on wide screens; fit and center within the space left of it.
      const reserved = window.innerWidth >= 900 ? DETAILS_PANEL_PX : 0;
      const viewAspect = (window.innerWidth - reserved) / window.innerHeight;
      const camera = graph.camera() as PerspectiveCamera;
      const controls = graph.controls() as OrbitLikeControls;
      const t = tanHalfFov(camera);
      const wanted = CLUSTER_FIT_PADDING * Math.max(extY / t, extX / (t * viewAspect));
      const distance = Math.min(controls.maxDistance, Math.max(controls.minDistance, wanted));

      // Shift the camera right so the cluster lands in the middle of the uncovered area.
      const unitsPerPx = (2 * distance * t) / window.innerHeight;
      flight.flyTo({ x: cx + (reserved / 2) * unitsPerPx, y: cy, z: distance }, FLY_MS);
    },
    [nodes],
  );

  const resetView = useCallback(() => {
    const graph = fgRef.current;
    const flight = flightRef.current;
    if (!graph || !flight) return;
    const controls = graph.controls() as OrbitLikeControls;
    // maxDistance is the distance at which the whole map fits, set by fitToWindow().
    flight.flyTo({ x: bounds.center.x, y: bounds.center.y, z: controls.maxDistance }, FLY_MS);
  }, [bounds]);

  useImperativeHandle(ref, () => ({ flyToCluster, resetView }), [flyToCluster, resetView]);

  // One-time wiring of the three.js controls that the React props don't cover.
  useEffect(() => {
    const graph = fgRef.current;
    if (!graph) return;
    const camera = graph.camera() as PerspectiveCamera;
    const controls = graph.controls() as OrbitLikeControls;
    const flight = createFlightController(camera, controls);
    flightRef.current = flight;

    // Flat map: look straight down the z-axis. No rotating; drag to pan, scroll to zoom.
    controls.enableRotate = false;
    controls.enablePan = true;
    controls.screenSpacePanning = true; // pan moves in the screen plane (= the map plane here)
    // Three.js MOUSE / TOUCH enum values: MOUSE.PAN = 2, MOUSE.DOLLY = 1, TOUCH.PAN = 1, TOUCH.DOLLY_PAN = 2
    controls.mouseButtons = { LEFT: 2, MIDDLE: 1, RIGHT: 2 };
    controls.touches = { ONE: 1, TWO: 2 };
    controls.addEventListener('change', clampPan); // fires on every pan and zoom

    const cancelFlight = () => flight.cancel();
    const canvas = graph.renderer().domElement;
    canvas.addEventListener('pointerdown', cancelFlight);
    canvas.addEventListener('wheel', cancelFlight, { passive: true });

    // The library's built-in hint still describes rotating, which is turned off.
    const navHint = document.querySelector('.scene-nav-info');
    if (navHint) navHint.textContent = 'Left or Right-click: pan, Mouse-wheel/middle-click: zoom';

    // Handy for poking at things from the browser console while we build.
    (window as unknown as { graph: unknown }).graph = graph;

    return () => {
      controls.removeEventListener('change', clampPan);
      canvas.removeEventListener('pointerdown', cancelFlight);
      canvas.removeEventListener('wheel', cancelFlight);
      flightRef.current = null;
    };
  }, [clampPan]);

  // Initial fit, and again whenever the window changes size.
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
