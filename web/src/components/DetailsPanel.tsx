import { useEffect } from 'react';
import { clusterColor } from '../constants';
import type { DetailState } from '../hooks/usePaperDetail';
import type { Cluster, PaperNode } from '../types';

const MAX_AUTHORS = 8;

interface Props {
  node: PaperNode;
  detail: DetailState;
  clusters: Cluster[];
  onClose: () => void;
}

function authorLine(authors: string[]): string {
  const shown = authors.slice(0, MAX_AUTHORS).join(', ');
  const extra = authors.length - MAX_AUTHORS;
  return (shown || 'Unknown') + (extra > 0 ? ` and ${extra} more` : '');
}

export function DetailsPanel({ node, detail, clusters, onClose }: Props) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  const cluster = clusters.find((c) => c.id === node.cluster);
  const failed = detail.status === 'error';

  return (
    // key={node.id} resets the scroll position when a different paper is opened.
    <aside id="details" key={node.id} aria-live="polite">
      <button id="details-close" type="button" aria-label="Close details" title="Close (Esc)" onClick={onClose}>
        ×
      </button>

      <h2 id="details-title">{node.title}</h2>
      <p id="details-meta">
        {node.date ?? 'Unknown date'} · {node.category}
      </p>

      <div id="details-stats">
        <div>
          <b>{node.citations}</b>
          <span>citations</span>
        </div>
        <div>
          <b>{node.influence.toFixed(2)}</b>
          <span>influence</span>
        </div>
      </div>

      <p id="details-cluster">
        <span className="swatch" style={{ background: clusterColor(node.cluster) }} />
        {node.cluster === -1 ? 'Not in a cluster' : (cluster?.title ?? `Cluster ${node.cluster}`)}
      </p>

      <h3>Authors</h3>
      <p id="details-authors">
        {detail.status === 'ready'
          ? authorLine(detail.detail.authors)
          : failed
            ? 'Could not load details.'
            : 'Loading…'}
      </p>

      <h3>Abstract</h3>
      <p id="details-abstract">
        {detail.status === 'ready'
          ? detail.detail.abstract || 'No abstract available.'
          : failed
            ? ''
            : 'Loading…'}
      </p>

      {detail.status === 'ready' && (
        <a
          id="details-link"
          href={`https://arxiv.org/abs/${detail.detail.arxivId}`}
          target="_blank"
          rel="noopener noreferrer"
        >
          View on arXiv ↗
        </a>
      )}
    </aside>
  );
}
