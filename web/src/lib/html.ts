export const escapeHtml = (s: unknown): string =>
  String(s).replace(
    /[&<>"']/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!,
  );

/** "Cluster 5: Foo / Bar" -> "Foo / Bar" */
export const stripPrefix = (title: string): string => title.replace(/^Cluster \d+:\s*/, '');

export const swatch = (color: string): string =>
  `<span class="swatch" style="background:${color}"></span>`;

export const fetchJson = async <T = any>(url: string): Promise<T> => {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return res.json() as Promise<T>;
};

export const $ = (id: string): HTMLElement => document.getElementById(id)!;
