export function Swatch({ color }: { color: string }) {
  return <span className="swatch" style={{ background: color }} />;
}
