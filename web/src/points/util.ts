/** Small helpers the point map needs. */

export const fetchJson = async <T>(url: string): Promise<T> => {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return res.json() as Promise<T>;
};

/** "Cluster 5: Foo / Bar" -> "Foo / Bar" */
export const stripPrefix = (title: string): string => title.replace(/^Cluster \d+:\s*/, '');

/**
 * One colour per arXiv field, in the order the exporter lists them (biggest first); anything
 * past the end falls back to OTHER_COLOR. Tableau 10/20 hues, extended to 19 because the
 * full-arXiv export colours by archive (cs, math, cond-mat, ...) rather than by category.
 */
export const FIELD_COLORS = [
  '#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948',
  '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac', '#86bcb6', '#d37295',
  '#8cd17d', '#b6992d', '#499894', '#fabfd2', '#d7b5a6', '#79706e', '#a0cbe8',
];
export const OTHER_COLOR = '#6c7386';
