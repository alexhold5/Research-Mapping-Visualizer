// Nudges overlapping spheres apart in x/y while keeping each one as close as possible to its
// original (UMAP) position. Run once after node radii are known; it edits n.x / n.y in place.
//
// How it works, each round:
//   1. Push apart every pair of spheres that overlap. The smaller one moves further, so the
//      biggest papers stay closest to where the ML put them.
//   2. Pull every sphere slightly back toward its original spot, so clusters don't drift or balloon.
// After the rounds, a few pure separation passes squeeze out any leftover overlap.
// Deterministic: same input always gives the same layout.
//
// z is read but never written. In timeline mode it carries the publication date, which the
// proposal wants kept undistorted, so two papers far enough apart in time already count as
// separated and need no nudge at all. In flat mode every z is 0 and this reduces exactly to
// the 2D circle-packing it replaced.

interface Positioned {
  x: number;
  y: number;
  z: number;
  radius: number;
}

export interface LayoutOptions {
  padding?: number;
  rounds?: number;
  pull?: number;
  finalPasses?: number;
}

export interface LayoutStats {
  /** Mean distance a node travelled from its UMAP position. */
  moved: number;
  maxMoved: number;
  /** Overlaps remaining, or null when the O(n^2) check was skipped for being too slow. */
  overlaps: number | null;
}

export function resolveOverlaps(
  nodes: Positioned[],
  { padding = 0.6, rounds = 200, pull = 0.03, finalPasses = 150 }: LayoutOptions = {},
): LayoutStats {
  const count = nodes.length;
  if (count < 2) return { moved: 0, maxMoved: 0, overlaps: 0 };

  const hx = Float64Array.from(nodes, (n) => n.x); // original positions
  const hy = Float64Array.from(nodes, (n) => n.y);
  const x = Float64Array.from(hx);
  const y = Float64Array.from(hy);
  const z = Float64Array.from(nodes, (n) => n.z); // fixed: the timeline axis
  const r = Float64Array.from(nodes, (n) => n.radius);
  let maxRadius = 0;
  for (let i = 0; i < count; i++) if (r[i] > maxRadius) maxRadius = r[i];
  const cell = 2 * maxRadius + padding; // any overlapping pair is in the same or an adjacent cell
  const GOLDEN = 2.399963; // used to pick a direction for circles sitting exactly on top of each other

  const key = (cx: number, cy: number) => (cx + 50000) * 100000 + (cy + 50000);

  /** Returns the deepest overlap found (0 when everything is clear). */
  function separate(): number {
    const grid = new Map<number, number[]>();
    for (let i = 0; i < count; i++) {
      const k = key(Math.floor(x[i] / cell), Math.floor(y[i] / cell));
      const bucket = grid.get(k);
      if (bucket) bucket.push(i);
      else grid.set(k, [i]);
    }

    let deepest = 0;
    for (let i = 0; i < count; i++) {
      const cx = Math.floor(x[i] / cell);
      const cy = Math.floor(y[i] / cell);
      for (let gx = cx - 1; gx <= cx + 1; gx++) {
        for (let gy = cy - 1; gy <= cy + 1; gy++) {
          const bucket = grid.get(key(gx, gy));
          if (!bucket) continue;
          for (const j of bucket) {
            if (j <= i) continue; // handle each pair once
            const minDist = r[i] + r[j] + padding;
            const dz = z[i] - z[j];
            if (Math.abs(dz) >= minDist) continue; // already clear of each other along the timeline
            // Separation still owed in the xy plane once the gap in z is accounted for.
            const needXY = Math.sqrt(minDist * minDist - dz * dz);
            let dx = x[i] - x[j];
            let dy = y[i] - y[j];
            let d = Math.hypot(dx, dy);
            const overlap = needXY - d;
            if (overlap <= 0) continue;
            if (d < 1e-9) {
              dx = Math.cos(i * GOLDEN);
              dy = Math.sin(i * GOLDEN);
              d = 1;
            }
            const nx = dx / d;
            const ny = dy / d;
            const share = (r[j] * r[j]) / (r[i] * r[i] + r[j] * r[j]); // fraction of the move that i takes
            x[i] += nx * overlap * share;
            y[i] += ny * overlap * share;
            x[j] -= nx * overlap * (1 - share);
            y[j] -= ny * overlap * (1 - share);
            if (overlap > deepest) deepest = overlap;
          }
        }
      }
    }
    return deepest;
  }

  for (let round = 0; round < rounds; round++) {
    separate();
    for (let i = 0; i < count; i++) {
      x[i] += (hx[i] - x[i]) * pull;
      y[i] += (hy[i] - y[i]) * pull;
    }
  }
  for (let pass = 0; pass < finalPasses; pass++) {
    if (separate() < 0.01) break;
  }

  // Report, then write back.
  let total = 0;
  let maxMoved = 0;
  for (let i = 0; i < count; i++) {
    const moved = Math.hypot(x[i] - hx[i], y[i] - hy[i]);
    total += moved;
    if (moved > maxMoved) maxMoved = moved;
    nodes[i].x = x[i];
    nodes[i].y = y[i];
  }

  // Brute-force check of the result; skipped for big datasets where n^2 would be slow.
  let overlaps: number | null = null;
  if (count <= 5000) {
    overlaps = 0;
    for (let i = 0; i < count; i++) {
      for (let j = i + 1; j < count; j++) {
        if (Math.hypot(x[i] - x[j], y[i] - y[j], z[i] - z[j]) < r[i] + r[j] - 0.05) overlaps++;
      }
    }
  }
  return { moved: total / count, maxMoved, overlaps };
}
