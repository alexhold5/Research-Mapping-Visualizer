import type { Group, Mesh, Sprite } from 'three';
import type { TitleDims } from './lib/titleSprite';

// ---- What export_web_data.py writes ----

export interface Meta {
  /** Drill-down maps only (export_maps.py): which map this is, its name, and the map "back" goes to. */
  mapId?: string;
  title?: string;
  /** root = all of arXiv (bubbles are groups), group = e.g. cs (categories), category = e.g. cs.AI (clusters). */
  level?: 'root' | 'group' | 'category';
  parent?: string | null;
  parentTitle?: string | null;
  /** How each cluster's papers were picked: by influence score, or a random sample (before there were scores). */
  ranked?: 'influence' | 'random';
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
  /** Drill-down maps: clicking this bubble opens that map (e.g. a category's own cluster map). */
  map?: string;
  /** Drill-down maps: what zooming into this bubble shows. */
  inner?: 'categories' | 'clusters' | 'papers';
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
  /** Drill-down maps: arXiv ids of the papers in each cluster's file, in file order (names the far end of a cross-cluster line). */
  shownIds?: Record<string, string[]>;
}

export interface PaperInfo {
  id: string;
  title: string;
  year: number | null;
  date: string | null;
  influence: number;
  citations: number;
  category: string;
  categoryCode: string;
  authors: string[];
  abstract: string;
  arxivId: string;
}

export interface RawPaper extends PaperInfo {
  x: number;
  y: number;
  /** Included only because another cluster's line leads to it. */
  extra?: boolean;
  /** Drill-down maps: this node stands for a whole cluster of another map, which clicking it opens. */
  opens?: { map: string; cluster?: number; papers: number };
}

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
