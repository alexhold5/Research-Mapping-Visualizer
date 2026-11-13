import { FIELD_COLORS, OTHER_COLOR } from '../constants';
import type { ClusterSummary, Field } from '../types';

/** Color and name lookups for arXiv fields. The biggest fields get their own color; the rest are "other". */
export function createFieldHelpers(fields: Field[]) {
  const index = new Map(fields.map((f, i) => [f.code, i]));
  /** Anything outside the biggest fields collapses to 'other'. */
  const key = (code: string): string => (index.has(code) ? code : 'other');
  const color = (k: string): string => {
    const i = index.get(k);
    return i !== undefined && i < FIELD_COLORS.length ? FIELD_COLORS[i] : OTHER_COLOR;
  };
  const name = (k: string): string =>
    k === 'other' ? 'Other fields' : (fields[index.get(k) ?? -1]?.name ?? k);
  const clusterColor = (c: ClusterSummary): string => color(c.field);
  return { key, color, name, clusterColor };
}

export type FieldHelpers = ReturnType<typeof createFieldHelpers>;
