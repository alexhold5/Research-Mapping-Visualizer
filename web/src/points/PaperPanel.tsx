import { useEffect, useRef, useState, useSyncExternalStore } from 'react';
import { Swatch } from './Swatch';
import { stripPrefix } from './util';
import { CITER_COLOR, REF_COLOR } from './pointEngine';
import type { PointEngine } from './pointEngine';

const MAX_AUTHORS = 8;
const FIRST_ROWS = 20; // neighbor rows shown at first; titles load per row, so keep the first batch small
const MORE_ROWS = 50;

/** The right panel: the selected paper, then its references and citers (each row hops to that paper). */
export function PaperPanel({ engine }: { engine: PointEngine }) {
  useSyncExternalStore(engine.subscribe, engine.getVersion);
  const sel = engine.selection;
  const ref = useRef<HTMLElement>(null);

  useEffect(() => {
    if (ref.current) ref.current.scrollTop = 0;
  }, [sel?.index]);

  if (!sel) return null;
  const { index, detail, edges } = sel;
  const cluster = engine.clusterOf(index);
  const extraAuthors = detail ? detail.authors.length - MAX_AUTHORS : 0;
  const arxivId = detail?.id.replace(/^ARXIV:/, '');

  return (
    <aside id="details" ref={ref} style={{ bottom: 16 }} aria-live="polite">
      <button id="details-close" type="button" aria-label="Close details" title="Close (Esc)" onClick={() => engine.select(null)}>
        ×
      </button>
      {detail === undefined ? (
        <h2>Loading…</h2>
      ) : detail === null ? (
        <h2>No details for paper #{index}</h2>
      ) : (
        <>
          <h2 id="details-title">{detail.title}</h2>
          <p id="details-meta">
            {detail.date ?? 'Unknown date'} · {detail.category}
          </p>
          <div id="details-stats">
            <div>
              <b>{detail.citations.toLocaleString()}</b>
              <span>citations</span>
            </div>
            <div>
              <b>{engine.data.influence[index].toFixed(2)}</b>
              <span>influence</span>
            </div>
          </div>
        </>
      )}
      <p id="details-cluster">
        <Swatch color={engine.paperCss(index)} />
        {cluster ? stripPrefix(cluster.title) : 'Unclustered (noise)'}
      </p>

      {edges === undefined ? (
        <p className="hint" style={{ marginTop: 12 }}>Loading citation links…</p>
      ) : (
        <>
          <NeighborList engine={engine} title="References" color={REF_COLOR} ids={sel.refs} key={`r${index}`} />
          <NeighborList engine={engine} title="Cited by" color={CITER_COLOR} ids={sel.citers} key={`c${index}`} />
          {edges?.o ? (
            <p style={{ marginTop: 8, color: '#8b92a5' }}>
              +{edges.o.toLocaleString()} references to papers outside this dataset (not drawn)
            </p>
          ) : null}
          {sel.refs.length + sel.citers.length === 0 && (
            <p style={{ marginTop: 8, color: '#8b92a5' }}>
              {edges === null
                ? 'No citation data for this paper.'
                : 'None of its citations connect to other papers in this dataset.'}
            </p>
          )}
        </>
      )}
      {sel.error && <p style={{ marginTop: 8, color: '#ff8a8a' }}>Failed to load: {sel.error}</p>}

      {detail && (
        <>
          <h3>Authors</h3>
          <p id="details-authors">
            {detail.authors.slice(0, MAX_AUTHORS).join(', ') || 'Unknown'}
            {extraAuthors > 0 ? ` and ${extraAuthors} more` : ''}
          </p>
          {engine.data.meta.hasAbstracts !== false && (
            <>
              <h3>Abstract</h3>
              <p id="details-abstract">{detail.abstract || 'No abstract available.'}</p>
              {detail.truncated && (
                <p className="hint" style={{ marginTop: -4 }}>
                  Trimmed to keep the dataset small — the full abstract is on arXiv.
                </p>
              )}
            </>
          )}
          <a id="details-link" href={`https://arxiv.org/abs/${arxivId}`} target="_blank" rel="noopener noreferrer">
            View on arXiv ↗
          </a>
        </>
      )}
    </aside>
  );
}

function NeighborList({ engine, title, color, ids }: { engine: PointEngine; title: string; color: string; ids: number[] }) {
  const [shown, setShown] = useState(FIRST_ROWS);
  if (ids.length === 0) return null;
  return (
    <>
      <h3>
        <span style={{ color }}>━</span> {title} ({ids.length.toLocaleString()})
      </h3>
      <ul style={{ listStyle: 'none', margin: 0, padding: 0 }}>
        {ids.slice(0, shown).map((j) => (
          <NeighborRow engine={engine} index={j} key={j} />
        ))}
      </ul>
      {ids.length > shown && (
        <button type="button" className="more-rows" onClick={() => setShown(shown + MORE_ROWS)}>
          Show {Math.min(MORE_ROWS, ids.length - shown)} more
        </button>
      )}
    </>
  );
}

function NeighborRow({ engine, index }: { engine: PointEngine; index: number }) {
  const [title, setTitle] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    engine.loaders.title(index).then((t) => live && setTitle(t ?? '(untitled)'), () => live && setTitle('(failed to load title)'));
    return () => {
      live = false;
    };
  }, [engine, index]);

  const cluster = engine.clusterOf(index);
  const year = engine.data.year[index];
  return (
    <li>
      <button
        type="button"
        className="neighbor-row"
        title={cluster ? stripPrefix(cluster.title) : 'Unclustered (noise)'}
        onMouseEnter={() => engine.highlight(index)}
        onMouseLeave={() => engine.highlight(null)}
        onClick={() => {
          engine.select(index);
          engine.centerOn(index);
        }}
      >
        <Swatch color={engine.paperCss(index)} />
        <span>{title ?? '…'}</span>
        {year > 0 && <span className="year">{year}</span>}
      </button>
    </li>
  );
}
