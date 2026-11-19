import { lerp } from './math';

/** "#rrggbb" + alpha -> "rgba(r, g, b, a)". */
export const withAlpha = (hex: string, a: number): string => {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
};

/** Mix a #rrggbb color toward white (amount 0..1). */
export const lighten = (hex: string, amount: number): string => {
  const n = parseInt(hex.slice(1), 16);
  const mix = (c: number) => Math.round(lerp(c, 255, amount));
  return `#${[(n >> 16) & 255, (n >> 8) & 255, n & 255].map((c) => mix(c).toString(16).padStart(2, '0')).join('')}`;
};

/** Scale the opacity of an "rgba(r, g, b, a)" color. */
export const dim = (color: string, factor: number): string =>
  color.replace(
    /rgba\(([^)]*),\s*([\d.]+)\)$/,
    (_, rgb: string, a: string) => `rgba(${rgb}, ${(Number(a) * factor).toFixed(2)})`,
  );
