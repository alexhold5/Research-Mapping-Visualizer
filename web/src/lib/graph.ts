import { NODE_PADDING, R_MAX, R_MIN, SIZE_CURVE, XY_SCALE, Z_PER_YEAR } from '../constants';
import { resolveOverlaps } from './layout';
import type { GraphMeta, PaperNode, RawNode, ViewMode } from '../types';

/**
 * Min/max over a big array without `Math.min(...arr)` — the spread form blows the call
 * stack somewhere around 100k arguments, which is exactly the scale we're aiming for.
 */
function extent(values: ArrayLike<number>): [number, number] {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < values.length; i++) {
    if (values[i] < min) min = values[i];
    if (values[i] > max) max = values[i];
  }
  return [min, max];
}

export interface Bounds {
  center: { x: number; y: number; z: number };
  halfW: number;
  halfH: number;
  /** Half the depth of the map along the timeline axis; 0 in flat mode. */
  halfD: number;
  /** Radius of a sphere around the whole map, used to frame it from any angle. */
  radius: number;
}

export function computeBounds(nodes: PaperNode[]): Bounds {
  const [minX, maxX] = extent(nodes.map((n) => n.x));
  const [minY, maxY] = extent(nodes.map((n) => n.y));
  const [minZ, maxZ] = extent(nodes.map((n) => n.z));
  const halfW = (maxX - minX) / 2;
  const halfH = (maxY - minY) / 2;
  const halfD = (maxZ - minZ) / 2;
  return {
    center: { x: (minX + maxX) / 2, y: (minY + maxY) / 2, z: (minZ + maxZ) / 2 },
    halfW,
    halfH,
    halfD,
    radius: Math.hypot(halfW, halfH, halfD),
  };
}

/**
 * Turns exported nodes into renderable ones: scale UMAP coords into world space, pin each node
 * so the force engine can't move it, size it by influence, then nudge overlapping spheres apart.
 *
 * In timeline mode the z coordinate is the publication date and is never adjusted — only x/y
 * absorb the de-overlapping, so the chronology stays exact.
 *
 * This is the expensive step (the overlap pass is hundreds of relaxation rounds), so it runs
 * once per dataset per view mode, not per render.
 */
export function buildNodes(rawNodes: RawNode[], meta: GraphMeta, mode: ViewMode): PaperNode[] {
  const midYear = (meta.yearMin + meta.yearMax) / 2;

  // Pin every node to its ML-computed position. fx/fy/fz tell the force engine
  // "don't move this node", so the layout is the UMAP map, not a physics sim.
  const nodes: PaperNode[] = rawNodes.map((n) => {
    const x = n.x * XY_SCALE;
    const y = n.y * XY_SCALE;
    // Papers with no parseable date sit at the middle of the timeline rather than at year 0.
    const z = mode === 'timeline' ? ((n.year ?? midYear) - midYear) * Z_PER_YEAR : 0;
    return { ...n, x, y, z, fx: x, fy: y, fz: z, radius: R_MIN };
  });

  // One continuous scale: the lowest influence gets R_MIN, the highest gets R_MAX, and everything
  // in between is placed proportionally (SIZE_CURVE bends the middle of the scale).
  const [minInf, maxInf] = extent(nodes.map((n) => n.influence));
  const span = maxInf - minInf || 1;
  for (const n of nodes) {
    const t = (n.influence - minInf) / span;
    n.radius = R_MIN + (R_MAX - R_MIN) * Math.pow(t, SIZE_CURVE);
  }

  // UMAP doesn't know how big the spheres are, so many overlap. Nudge them apart (minimal
  // movement, big papers move least), then re-pin each node at its final spot.
  const started = performance.now();
  const stats = resolveOverlaps(nodes, { padding: NODE_PADDING });
  for (const n of nodes) {
    n.fx = n.x;
    n.fy = n.y;
  }
  console.log(
    `Overlap removal (${mode}, ${nodes.length} nodes, ${Math.round(performance.now() - started)}ms): ` +
      `avg move ${stats.moved.toFixed(1)}, max move ${stats.maxMoved.toFixed(1)}, ` +
      `overlaps left: ${stats.overlaps ?? 'not checked'}`,
  );

  return nodes;
}
