import type { ClusterFile, ClusterFileLink, ClusterLinkRaw, ClusterSummary, ExternalPaper, Field, Meta, Overview, RawPaper } from '../types';

// ---------------------------------------------------------------------------
// The drill-down maps (export_maps.py) are stored in a compact form: papers as short arrays and only the
// per-year numbers that are non-zero. These functions expand them into the shapes the engine already uses
// (types.ts), so the engine draws the original map and the drill-down maps the same way.
// Authors and abstracts are not in these files; they are fetched for the one selected paper.
// ---------------------------------------------------------------------------

export interface SlimBubble {
  id: number;
  title: string;
  size: number;
  x: number;
  y: number;
  /** Index into fields; fields.length = "other". */
  field: number;
  /** Share of the bubble's papers in that field. */
  share: number;
  /** Its top paper's title, or a line like "Largest cluster: ...". */
  top: string;
  count: number[];
  /** Papers per year for each field index that has any. */
  fy: Record<string, number[]>;
  /** Mean influence per year (only once the data has influence scores). */
  inf?: number[];
  /** The map clicking this bubble opens. */
  map?: string;
  /** What zooming into it shows. */
  inner: 'categories' | 'clusters' | 'papers';
}

export interface SlimOverview {
  slim: 1;
  meta: Meta;
  years: number[];
  timeline: { count: number[]; inf?: number[] };
  fields: Field[];
  bubbles: SlimBubble[];
  /** [bubble id, bubble id, citations between them] */
  links: [number, number, number][];
  /** Category maps: cluster id -> the arXiv ids of its file's papers, in order. */
  shown?: Record<string, string[]>;
}

/** [arXiv id, title, x, y, year, influence, field index, citations] */
export type SlimPaperRow = [string, string, number, number, number, number, number, number];
/** [row, the other paper's cluster, the other paper's row there, 1 if this row's paper cites it, 0 if it cites this one] */
export type SlimCrossRow = [number, number, number, 0 | 1];
/** [map, cluster id or null (= the whole map), title, x, y, papers, year] */
export type SlimOpensRow = [string, number | null, string, number, number, number, number];

export interface SlimClusterFile {
  papers?: SlimPaperRow[];
  /** [row, row]: the paper in the first row cites the one in the second. */
  links?: [number, number][];
  /** Citations between a paper here and a paper in another cluster of the same map. */
  cross?: SlimCrossRow[];
  opens?: SlimOpensRow[];
}

export const isSlimOverview = (raw: unknown): raw is SlimOverview => (raw as SlimOverview).slim === 1;

export function expandOverview(raw: SlimOverview): Overview {
  const { fields, years } = raw;
  const zeros = years.map(() => 0);
  const keys = [...fields.map((f) => f.code), 'other'];
  const codeOf = (i: number) => keys[Math.min(i, fields.length)];
  const nameOf = (i: number) => fields[i]?.name ?? 'Other';

  const clusters: ClusterSummary[] = raw.bubbles.map((b) => ({
    id: b.id,
    title: b.title,
    size: b.size,
    x: b.x,
    y: b.y,
    influence: 0,
    category: codeOf(b.field),
    categoryName: nameOf(b.field),
    field: codeOf(b.field),
    categoryShare: b.share,
    topPaper: b.top,
    count: b.count,
    inf: b.inf ?? zeros,
    fy: Object.fromEntries(keys.map((k, i) => [k, b.fy[i] ?? zeros])),
    fi: Object.fromEntries(keys.map((k) => [k, zeros])),
    map: b.map,
    inner: b.inner,
  }));
  const links: ClusterLinkRaw[] = raw.links.map(([source, target, weight]) => ({ source, target, weight }));
  return { meta: raw.meta, years, timeline: { count: raw.timeline.count, inf: raw.timeline.inf ?? zeros }, fields, clusters, links, shownIds: raw.shown };
}

/** "0704.0001" -> "2007-04", "hep-ex/0111022" -> "2001-11" (arXiv ids encode the submission month). */
function submissionMonth(arxivId: string): string | null {
  const m = /(\d{2})(\d{2})[.\d]*$/.exec(arxivId.split('/').pop() ?? '');
  if (!m) return null;
  const yy = Number(m[1]);
  return `${yy >= 91 ? 1900 + yy : 2000 + yy}-${m[2]}`;
}

export function expandClusterFile(raw: SlimClusterFile, bubbleId: number, overview: Overview): ClusterFile {
  const { fields } = overview;
  const bubble = overview.clusters.find((c) => c.id === bubbleId);
  const nodes: RawPaper[] = [];
  for (const [id, title, x, y, year, influence, field, citations] of raw.papers ?? []) {
    const f = fields[field];
    nodes.push({
      id, title, x, y, year, influence, citations: citations ?? 0, date: submissionMonth(id),
      category: f?.name ?? 'Other', categoryCode: f?.code ?? 'other', authors: [], abstract: '', arxivId: id,
    });
  }
  for (const [map, cluster, title, x, y, papers, year] of raw.opens ?? []) {
    nodes.push({
      id: `${map}:${cluster ?? 'map'}`, title, x, y, year, date: null,
      influence: papers, // sizes follow influence rank, so the biggest clusters/categories are the biggest nodes
      citations: 0, category: bubble?.categoryName ?? '', categoryCode: bubble?.field ?? 'other', authors: [],
      abstract: '', arxivId: '', opens: { map, cluster: cluster ?? undefined, papers },
    });
  }
  const rows = raw.papers ?? [];
  const links: ClusterFileLink[] = (raw.links ?? []).map(([a, b]) => ({ source: rows[a][0], target: rows[b][0] }));

  // Lines to papers in other clusters. The other paper's details aren't in this file, so it is named by its arXiv id
  // until its cluster opens (then the real paper is drawn and the line joins the two papers).
  const external = new Map<string, ExternalPaper>();
  for (const [row, cluster, otherRow, cites] of raw.cross ?? []) {
    const otherId = overview.shownIds?.[cluster]?.[otherRow];
    if (!otherId) continue;
    if (!external.has(otherId)) {
      external.set(otherId, {
        id: otherId, title: `arXiv:${otherId}`, year: null, date: submissionMonth(otherId), influence: 0, citations: 0,
        category: '', categoryCode: 'other', authors: [], abstract: '', arxivId: otherId, cluster, inTop: true,
      });
    }
    const mine = rows[row][0];
    links.push(cites ? { source: mine, target: otherId, cross: true } : { source: otherId, target: mine, cross: true });
  }
  return { id: bubbleId, nodes, external: [...external.values()], neighbors: [], links };
}
