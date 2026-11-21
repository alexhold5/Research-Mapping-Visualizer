/** The small colour dot shown beside a cluster or paper title. */
export const Swatch = ({ color }: { color: string }) => (
  <span className="swatch" style={{ background: color }} />
);
