import { useMemo } from 'react';
import { clusterColor, clusterName } from '../constants';
import type { Cluster } from '../types';

interface Props {
  clusters: Cluster[];
  selectedCluster: number | null;
  onSelect: (id: number) => void;
}

/**
 * Cluster list in the top-left panel: biggest first. Click one to select it and fly there;
 * click the active one again to clear. (Noise isn't a real cluster, so it isn't listed.)
 */
export function ClusterList({ clusters, selectedCluster, onSelect }: Props) {
  const ordered = useMemo(
    () => clusters.filter((c) => c.id !== -1).sort((a, b) => b.size - a.size),
    [clusters],
  );

  return (
    <ul id="cluster-list">
      {ordered.map((c) => (
        <li key={c.id} className={c.id === selectedCluster ? 'active' : undefined}>
          <button type="button" title={c.title} onClick={() => onSelect(c.id)}>
            <span className="swatch" style={{ background: clusterColor(c.id) }} />
            <span className="name">{clusterName(c.title)}</span>
            <span className="count">{c.size}</span>
          </button>
        </li>
      ))}
    </ul>
  );
}
