import { fetchJson } from './util';
import type { ColumnInfo, PointData, PointsMeta } from './types';

export const DATA_URL = `${import.meta.env.BASE_URL}data/points/`;

const ARRAY_TYPES = {
  float32: Float32Array,
  int16: Int16Array,
  uint16: Uint16Array,
  uint8: Uint8Array,
} as const;

/** meta.json + points.bin: one dot per paper. Each column of points.bin becomes a typed array (no copying). */
export async function loadPoints(): Promise<PointData> {
  const started = performance.now();
  const meta = await fetchJson<PointsMeta>(`${DATA_URL}meta.json`);
  const res = await fetch(`${DATA_URL}points.bin`);
  if (!res.ok) throw new Error(`points.bin: ${res.status}`);
  const buffer = await res.arrayBuffer();

  const column = <K extends ColumnInfo['name']>(name: K) => {
    const info = meta.columns.find((c) => c.name === name);
    if (!info) throw new Error(`points.bin has no "${name}" column`);
    return new ARRAY_TYPES[info.type](buffer, info.offset, meta.count);
  };

  return {
    meta,
    x: column('x') as Float32Array,
    y: column('y') as Float32Array,
    influence: column('influence') as Float32Array,
    cluster: column('cluster') as Int16Array,
    year: column('year') as Uint16Array,
    field: column('field') as Uint8Array,
    bytes: buffer.byteLength,
    loadMs: performance.now() - started,
  };
}
