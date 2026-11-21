/**
 * Uniform grid over the points, so hover and click only check the dots near the cursor instead of
 * all ~287k. Built once with a counting sort: the indices of cell k's points are
 * items[start[k] .. start[k + 1]).
 */
export function createGrid(x: Float32Array, y: Float32Array, res = 512) {
  const n = x.length;
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (let i = 0; i < n; i++) {
    if (x[i] < minX) minX = x[i];
    if (x[i] > maxX) maxX = x[i];
    if (y[i] < minY) minY = y[i];
    if (y[i] > maxY) maxY = y[i];
  }
  const cellW = (maxX - minX) / res || 1;
  const cellH = (maxY - minY) / res || 1;
  const col = (v: number) => Math.min(res - 1, Math.max(0, Math.floor((v - minX) / cellW)));
  const row = (v: number) => Math.min(res - 1, Math.max(0, Math.floor((v - minY) / cellH)));

  const cellOf = new Uint32Array(n);
  const start = new Uint32Array(res * res + 1);
  for (let i = 0; i < n; i++) {
    const k = row(y[i]) * res + col(x[i]);
    cellOf[i] = k;
    start[k + 1]++;
  }
  for (let k = 0; k < res * res; k++) start[k + 1] += start[k];
  const fill = start.slice(0, res * res);
  const items = new Uint32Array(n);
  for (let i = 0; i < n; i++) items[fill[cellOf[i]]++] = i;

  /**
   * The point nearest (px, py) within `radius` (world units), or -1.
   *
   * `visible` filters to the points actually on screen: the renderer draws only a slice of the
   * index buffer at low zoom, and hovering a dot that isn't drawn would be a lie.
   */
  const nearest = (px: number, py: number, radius: number, visible?: (i: number) => boolean): number => {
    const c0 = col(px - radius), c1 = col(px + radius);
    const r0 = row(py - radius), r1 = row(py + radius);
    let best = -1;
    let bestD = radius * radius;
    for (let r = r0; r <= r1; r++) {
      for (let c = c0; c <= c1; c++) {
        const k = r * res + c;
        for (let j = start[k]; j < start[k + 1]; j++) {
          const i = items[j];
          if (visible && !visible(i)) continue;
          const dx = x[i] - px, dy = y[i] - py;
          const d = dx * dx + dy * dy;
          if (d <= bestD) {
            bestD = d;
            best = i;
          }
        }
      }
    }
    return best;
  };

  // Visit every point whose cell lies within `radius` of the segment (ax,ay)-(bx,by), each
  // point exactly once. Used for picking a tilted view: the cursor's ray is perpendicular to an
  // orthographic image plane, so every point along it lands on the same pixel, and the
  // candidates form a thin tube whose shadow on the plane is this segment.
  const stamp = new Int32Array(res * res);
  let token = 0;
  const eachAlongSegment = (
    ax: number, ay: number, bx: number, by: number, radius: number, fn: (i: number) => void,
  ) => {
    token++;
    const len = Math.hypot(bx - ax, by - ay);
    const steps = Math.max(1, Math.ceil(len / Math.min(cellW, cellH)));
    const rc = Math.ceil(radius / cellW);
    const rr = Math.ceil(radius / cellH);
    for (let s = 0; s <= steps; s++) {
      const f = s / steps;
      const c = col(ax + (bx - ax) * f);
      const r = row(ay + (by - ay) * f);
      for (let dr = -rr; dr <= rr; dr++) {
        const rowI = r + dr;
        if (rowI < 0 || rowI >= res) continue;
        for (let dc = -rc; dc <= rc; dc++) {
          const colI = c + dc;
          if (colI < 0 || colI >= res) continue;
          const k = rowI * res + colI;
          if (stamp[k] === token) continue;
          stamp[k] = token;
          for (let j = start[k]; j < start[k + 1]; j++) fn(items[j]);
        }
      }
    }
  };

  return { nearest, eachAlongSegment };
}
