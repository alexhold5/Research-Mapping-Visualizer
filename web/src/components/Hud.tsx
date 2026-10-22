import { clusterColor } from '../constants';
import { ClusterList } from './ClusterList';
import type { Cluster, GraphMeta, ViewMode } from '../types';

interface Props {
  meta: GraphMeta | null;
  clusters: Cluster[];
  selectedCluster: number | null;
  onSelectCluster: (id: number) => void;
  mode: ViewMode;
  onModeChange: (mode: ViewMode) => void;
  status: string | null;
}

export function Hud({
  meta,
  clusters,
  selectedCluster,
  onSelectCluster,
  mode,
  onModeChange,
  status,
}: Props) {
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
      <p className="hint">
        {mode === 'flat'
          ? 'Position = topic similarity (UMAP) · color = cluster · size = influence'
          : 'X/Y = topic similarity (UMAP) · Z = publication date · size = influence'}
      </p>
      <p className="hint">
        {mode === 'flat'
          ? 'Scroll to zoom · drag to pan · click a paper to select its cluster'
          : 'Drag to rotate · right-drag to pan · scroll to zoom'}
      </p>

      <div id="view-mode" role="radiogroup" aria-label="View mode">
        <button
          type="button"
          role="radio"
          aria-checked={mode === 'flat'}
          className={mode === 'flat' ? 'active' : undefined}
          onClick={() => onModeChange('flat')}
          title="Flat topic map, viewed from straight above"
        >
          Flat map
        </button>
        <button
          type="button"
          role="radio"
          aria-checked={mode === 'timeline'}
          className={mode === 'timeline' ? 'active' : undefined}
          onClick={() => onModeChange('timeline')}
          title="Spread papers along the Z axis by publication date"
        >
          Timeline (3D)
        </button>
      </div>

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
