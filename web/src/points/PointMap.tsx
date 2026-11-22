import { useEffect, useRef, useState, useSyncExternalStore } from 'react';
import { stripPrefix } from './util';
import { loadPoints } from './data';
import { createPointEngine } from './pointEngine';
import type { PointEngine } from './pointEngine';
import type { PointData } from './types';

type LoadState = { status: 'loading' } | { status: 'ready'; data: PointData } | { status: 'error'; message: string };

const usePointEngine = (engine: PointEngine) => useSyncExternalStore(engine.subscribe, engine.getVersion);

/** The global view: every paper as a dot, for seeing how arXiv clusters. Data comes from export_points.py. */
export function PointMap() {
  const [state, setState] = useState<LoadState>({ status: 'loading' });

  useEffect(() => {
    let cancelled = false;
    loadPoints()
      .then((data) => !cancelled && setState({ status: 'ready', data }))
      .catch((err: Error) => {
        console.error(err);
        if (!cancelled) setState({ status: 'error', message: err.message });
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (state.status === 'ready') return <PointScene data={state.data} />;
  return (
    <div id="hud">
      <h1>Research Map · global view</h1>
      <p id="stats">
        {state.status === 'loading'
          ? 'Loading every paper…'
          : `Failed to load data: ${state.message}. Run "python export_points.py" from the repo root first.`}
      </p>
    </div>
  );
}

function PointScene({ data }: { data: PointData }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [engine, setEngine] = useState<PointEngine | null>(null);

  useEffect(() => {
    const e = createPointEngine(containerRef.current!, data);
    setEngine(e);
    if (import.meta.env.DEV) (window as unknown as { pointEngine: typeof e }).pointEngine = e; // for debugging in the console
    return () => {
      e.destroy();
      setEngine(null);
    };
  }, [data]);

  return (
    <>
      <div id="graph" ref={containerRef} />
      {engine && (
        <>
          <PointHud engine={engine} />
          <HoverTip engine={engine} />
        </>
      )}
    </>
  );
}

function PointHud({ engine }: { engine: PointEngine }) {
  usePointEngine(engine);
  const { meta } = engine.data;
  const noiseShare = Math.round((100 * meta.noisePapers) / meta.count);

  return (
    <div id="hud" style={{ pointerEvents: 'auto' }}>
      <h1>Research Map · global view</h1>
      <p id="stats">
        {meta.count.toLocaleString()} papers · {meta.clusters.length.toLocaleString()} clusters ·{' '}
        {meta.noisePapers.toLocaleString()} unclustered ({noiseShare}%)
      </p>
      <p className="hint">
        Every paper is a dot, coloured by {engine.colorByField ? 'its arXiv group' : 'its cluster'}; grey haze = unclustered. Hover a dot to see its
        cluster. To find individual papers, switch to the Clustered View.
      </p>
      <p className="hint">
        Drawing <b>{engine.drawn.toLocaleString()}</b> of {meta.count.toLocaleString()} dots (
        {Math.round((100 * engine.drawn) / meta.count)}%): the most influential papers plus an even sample of the
        rest. Zoom in for more.
      </p>
      <label className="hint" style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}>
        <input type="checkbox" checked={engine.timeAxis} onChange={(e) => engine.setTimeAxis(e.target.checked)} />
        <span>
          Time axis: lift papers by submission year ({meta.yearMin}–{meta.yearMax})
          {engine.timeAxis && <>; <b>right-drag to tilt</b></>}
        </span>
      </label>
      <p className="hint">
        <a href="#" onClick={(e) => { e.preventDefault(); engine.resetView(); }} style={{ color: 'inherit' }}>
          reset view
        </a>
      </p>
    </div>
  );
}

function HoverTip({ engine }: { engine: PointEngine }) {
  usePointEngine(engine);
  const h = engine.hovered;
  if (!h) return null;
  const c = engine.clusterOf(h.index);
  const year = engine.data.year[h.index];
  return (
    <div
      style={{
        position: 'fixed',
        left: h.screenX + 14,
        top: h.screenY + 14,
        maxWidth: 360,
        padding: '6px 9px',
        background: 'rgba(15, 18, 26, 0.92)',
        border: '1px solid rgba(255, 255, 255, 0.12)',
        borderRadius: 6,
        fontSize: 12,
        lineHeight: 1.4,
        pointerEvents: 'none',
      }}
    >
      <div style={{ color: engine.paperCss(h.index) }}>{c ? stripPrefix(c.title) : 'Unclustered'}</div>
      <div style={{ color: '#8b92a5', marginTop: 2 }}>
        {engine.fieldOf(h.index)}
        {year > 0 && ` · ${year}`}
      </div>
    </div>
  );
}
