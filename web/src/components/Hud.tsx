import { clusterColor } from '../constants';
import { ClusterList } from './ClusterList';
import type { Cluster, GraphMeta } from '../types';

interface Props {
  meta: GraphMeta | null;
  clusters: Cluster[];
  selectedCluster: number | null;
  onSelectCluster: (id: number) => void;
  status: string | null;
}

export function Hud({ meta, clusters, selectedCluster, onSelectCluster, status }: Props) {
  const selected = clusters.find((c) => c.id === selectedCluster);
  const realClusters = clusters.filter((c) => c.id !== -1).length;

  return (
    <div id="hud">
      <h1>Research Map</h1>
      <p id="stats">
        {status ??
          (meta
            ? `${meta.paperCount} papers · ${meta.linkCount} citation links · ${realClusters} clusters · ` +
              `${Math.floor(meta.yearMin)}–${Math.floor(meta.yearMax)}`
            : 'Loading…')}
      </p>
      <p className="hint">Position = topic similarity (UMAP) · color = cluster · size = influence</p>
      <p className="hint">Scroll to zoom · drag to pan · click a paper to select its cluster</p>

      {selectedCluster !== null && (
        <p id="selection" aria-live="polite">
          <span className="swatch" style={{ background: clusterColor(selectedCluster) }} />
          {selected?.title ?? `Cluster ${selectedCluster}`} · {selected?.size ?? '?'} papers
          <br />
          <span className="hint">Click the background to clear</span>
        </p>
      )}

      <h2 id="clusters-heading">Clusters</h2>
      <ClusterList clusters={clusters} selectedCluster={selectedCluster} onSelect={onSelectCluster} />
    </div>
  );
}
