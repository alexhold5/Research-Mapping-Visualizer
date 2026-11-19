import type { Group, Mesh, Sprite } from 'three';
import type { TitleDims } from './lib/titleSprite';

// ---- What export_web_data.py writes ----

export interface Meta {
  totalPapers: number;
  clusteredPapers: number;
  noisePapers: number;
  clusterCount: number;
  perCluster: number;
  yearMin: number;
  yearMax: number;
  source: string;
}

/** Per-year numbers behind the histogram: paper counts and average influence. */
export interface YearSeries {
  count: number[];
  inf: number[];
}

export interface Field {
  code: string;
  name: string;
  papers: number;
}

export interface ClusterSummary {
  id: number;
  title: string;
  size: number;
  influence: number;
  category: string;
  categoryName: string;
  field: string;
  categoryShare: number;
  topPaper: string;
  count: number[];
  inf: number[];
  /** Per-field, per-year paper counts / average influence. */
  fy: Record<string, number[]>;
  fi: Record<string, number[]>;
  x: number;
  y: number;
}

export interface ClusterLinkRaw {
  source: number;
  target: number;
  weight: number;
}

export interface Overview {
  meta: Meta;
  years: number[];
  timeline: YearSeries;
  fields: Field[];
  clusters: ClusterSummary[];
  links: ClusterLinkRaw[];
}

/** The light record of one paper (what clusters/<id>.json holds). Papers are listed in influence order. */
export interface PaperInfo {
  id: string;
  title: string;
  year: number | null;
  date: string | null;
  influence: number;
  citations: number;
  category: string;
  categoryCode: string;
  /** 0 = the most influential paper of its cluster. */
  rank: number;
}

export interface RawPaper extends PaperInfo {
  x: number;
  y: number;
}

/** The heavy part (details/<cluster>/<block>.json), fetched when a paper is selected. */
export interface PaperDetail {
  authors: string[];
  abstract: string;
}

export type DetailsState =
  | { status: 'loading' }
  | { status: 'failed' }
  | ({ status: 'ready' } & PaperDetail);

export interface ExternalPaper extends PaperInfo {
  cluster: number;
  inTop: boolean;
}

export interface ClusterFileLink {
  source: string;
  target: string;
  cross?: boolean;
}

export interface ClusterFile {
  id: number;
  nodes: RawPaper[];
  external: ExternalPaper[];
  neighbors: unknown;
  links: ClusterFileLink[];
}

// ---- What the graph draws ----

export interface TitleInfo {
  lines: string[];
  centerDims: TitleDims;
  oneLine: string[];
  topDims: TitleDims;
}

interface NodeBase {
  x: number;
  y: number;
  z: number;
  fx: number;
  fy: number;
  fz: number;
  radius: number;
  halo?: Mesh;
}

export interface BubbleNode extends NodeBase {
  id: number;
  kind: 'cluster';
  cluster: ClusterSummary;
  fullRadius: number;
  /** Radius after the time window / field filter shrink it. */
  shrunk: number;
  /** Papers inside the window that match the field filter. */
  inWindow: number;
  open: boolean;
  /** Rank of the first paper of the window of papers drawn while this bubble is open. */
  pageStart: number;
  title: TitleInfo;
  px: number;
  inView: boolean;
  sphere?: Mesh;
  group?: Group;
  titleCenter?: Sprite | null;
  titleTop?: Sprite | null;
}

export interface PaperNode extends NodeBase, RawPaper {
  kind: 'paper';
  bubble: BubbleNode;
  arxivId: string;
}

export type GraphNode = BubbleNode | PaperNode;

type LinkEnd = string | number | GraphNode;

export interface GraphLink {
  source: LinkEnd;
  target: LinkEnd;
  kind?: 'cluster' | 'paper' | 'stub';
  color?: string;
  scale?: number;
  weight?: number;
  cross?: boolean;
  neighbor?: number;
  ext?: ExternalPaper;
}

/** A cluster's papers once laid out inside its bubble. */
export interface LoadedCluster {
  papers: PaperNode[];
  links: LayoutLink[];
  scale: number;
  lastUsed: number;
  linkCache: Map<string, GraphLink>;
}

export interface LayoutLink {
  source: string;
  target: string;
  cross?: boolean;
  outsideId?: string;
  neighbor?: number;
  ext?: ExternalPaper;
}

/** What the paper-to-paper line under the mouse points at (shown as a small tooltip). */
export interface EdgeTip {
  x: number;
  y: number;
  verb: string;
  title: string;
}

export type FieldFilter = string | null;

/** Which papers of the focused cluster are drawn: ranks start+1 .. end of total. */
export interface PageInfo {
  start: number;
  end: number;
  total: number;
  canPrev: boolean;
  canNext: boolean;
}
