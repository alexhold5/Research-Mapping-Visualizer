import { useEffect, useRef } from 'react';
import { useEngineState } from '../hooks/useEngineState';
import type { Engine } from '../lib/engine';
import { stripPrefix } from '../lib/html';
import { Swatch } from './Swatch';

const MAX_AUTHORS = 8;

/** The right panel: everything about the selected paper. */
export function DetailsPanel({ engine }: { engine: Engine }) {
  useEngineState(engine);
  const paper = engine.detailsFor;
  const ref = useRef<HTMLElement>(null);

  // Start at the top whenever a different paper is opened.
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = 0;
  }, [paper?.id]);

  const cluster = paper?.bubble.cluster;
  const detail = paper ? engine.detailsOf(paper) : null;
  const authors = detail?.status === 'ready' ? detail.authors : [];
  const extraAuthors = authors.length - MAX_AUTHORS;

  return (
    <aside id="details" ref={ref} hidden={!paper} aria-live="polite">
      <button id="details-close" type="button" aria-label="Close details" title="Close (Esc)" onClick={() => engine.closeDetails()}>
        ×
      </button>
      {paper && cluster && (
        <>
          <h2 id="details-title">{paper.title}</h2>
          <p id="details-meta">
            {paper.date ?? 'Unknown date'} · {paper.category}
          </p>
          <div id="details-stats">
            <div>
              <b>{paper.citations.toLocaleString()}</b>
              <span>citations</span>
            </div>
            <div>
              <b>{paper.influence.toFixed(2)}</b>
              <span>influence</span>
            </div>
          </div>
          <p id="details-cluster">
            <Swatch color={engine.fieldHelpers.clusterColor(cluster)} />
            {stripPrefix(cluster.title)}
          </p>
          <h3>Authors</h3>
          <p id="details-authors">
            {detail?.status === 'loading' ? 'Loading…' : authors.slice(0, MAX_AUTHORS).join(', ') || 'Unknown'}
            {extraAuthors > 0 ? ` and ${extraAuthors} more` : ''}
          </p>
          <h3>Abstract</h3>
          <p id="details-abstract">{detail?.status === 'loading'
              ? 'Loading…'
              : detail?.status === 'failed'
                ? 'Could not load the abstract.'
                : detail?.status === 'ready' && detail.abstract
                  ? detail.abstract
                  : 'No abstract available.'}</p>
          <a id="details-link" href={`https://arxiv.org/abs/${paper.arxivId}`} target="_blank" rel="noopener noreferrer">
            View on arXiv ↗
          </a>
        </>
      )}
    </aside>
  );
}
