import type { Mesh } from 'three';

/** A node exactly as `export_web_data.py` writes it into graph.json. */
export interface RawNode {
  id: string;
  title: string;
  /** UMAP coordinates, normalized by the exporter to [-1, 1]. */
  x: number;
  y: number;
  /** Decimal publication year (2021.5 = mid-2021), or null when the date was unparseable. */
  year: number | null;
  date: string | null;
  /** HDBSCAN label; -1 means "noise", i.e. not in any cluster. */
  cluster: number;
  influence: number;
  citations: number;
  category: string;
  categoryCode: string;
}

/**
 * A node after `buildNodes()` scales it into world space and sizes it.
 * fx/fy/fz pin the node so the force engine leaves it where the ML put it.
 */
export interface PaperNode extends RawNode {
  z: number;
  fx: number;
  fy: number;
  fz: number;
  radius: number;
  /** The white outline sphere, attached by the graph component. */
  halo?: Mesh;
}

export interface RawLink {
  source: string;
  target: string;
}

/** 3d-force-graph replaces the id strings with the node objects once it ingests the data. */
export interface ResolvedLink {
  source: PaperNode;
  target: PaperNode;
}

export interface Cluster {
  id: number;
  title: string;
  size: number;
  centroid: { x: number; y: number; year: number };
}

export interface GraphMeta {
  paperCount: number;
  linkCount: number;
  yearMin: number;
  yearMax: number;
  source: string;
}

export interface RawGraphData {
  meta: GraphMeta;
  clusters: Cluster[];
  nodes: RawNode[];
  links: RawLink[];
}

/** One entry of papers.json, loaded lazily for the details panel. */
export interface PaperDetail {
  abstract: string;
  authors: string[];
  arxivId: string;
}
