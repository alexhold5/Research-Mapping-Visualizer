const ENTITIES: Record<string, string> = {
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#39;',
};

/** 3d-force-graph injects node labels as raw HTML, so paper titles have to be escaped. */
export const escapeHtml = (s: unknown): string => String(s).replace(/[&<>"']/g, (c) => ENTITIES[c]);
