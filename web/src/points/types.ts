// ---- What export_points.py writes to data/v2/ ----

/** One arXiv field (or archive) that gets its own colour; the last entry is always "other". */
export interface Field {
  code: string;
  name: string;
  papers: number;
}

export interface ColumnInfo {
  name: 'x' | 'y' | 'influence' | 'cluster' | 'year' | 'field';
  type: 'float32' | 'int16' | 'uint16' | 'uint8';
  offset: number;
}

export interface PointCluster {
  id: number;
  title: string;
  size: number;
  x: number;
  y: number;
  /** Index into meta.fields of the cluster's most common field. */
  field: number;
}

export interface PointsMeta {
  count: number;
  columns: ColumnInfo[];
  edgeShards: number;
  titleShards: number;
  detailShards: number;
  noisePapers: number;
  /** False when the export dropped abstracts to keep the payload small. */
  hasAbstracts?: boolean;
  yearMin: number;
  yearMax: number;
  /** The biggest fields, then a final "other" entry. */
  fields: Field[];
  clusters: PointCluster[];
  source: string;
}

/** One value per paper, indexed by paper index `i`. */
export interface PointData {
  meta: PointsMeta;
  x: Float32Array;
  y: Float32Array;
  influence: Float32Array;
  /** -1 = noise. */
  cluster: Int16Array;
  /** Whole year; 0 = unknown. */
  year: Uint16Array;
  field: Uint8Array;
  /** Bytes downloaded and time taken, for the HUD. */
  bytes: number;
  loadMs: number;
}

/** edges/<i % edgeShards>.json */
export interface EdgeEntry {
  /** Papers this one cites. */
  r: number[];
  /** Papers that cite this one. */
  c: number[];
  /** References to papers outside the dataset (no dot to draw a line to). */
  o?: number;
}

/** details/<i % detailShards>.json */
export interface PaperDetail {
  id: string;
  title: string;
  authors: string[];
  date: string | null;
  citations: number;
  category: string;
  /** Absent when the export ran with --abstract-chars -1. */
  abstract?: string;
  /** The abstract was trimmed by export_points.py; the full text is on arXiv. */
  truncated?: boolean;
}
