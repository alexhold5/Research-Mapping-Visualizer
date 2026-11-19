import { useEffect, useRef, useState, useSyncExternalStore } from 'react';
import { stripPrefix } from '../lib/html';
import { loadPoints } from './data';
import { PaperPanel } from './PaperPanel';
import { CITER_COLOR, REF_COLOR, createPointEngine } from './pointEngine';
import type { PointEngine } from './pointEngine';
import type { PointData } from './types';

type LoadState = { status: 'loading' } | { status: 'ready'; data: PointData } | { status: 'error'; message: string };

const usePointEngine = (engine: PointEngine) => useSyncExternalStore(engine.subscribe, engine.getVersion);

/** Point-cloud prototype (?view=points): every paper as a dot. Data comes from export_points.py. */
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
      <h1>Research Map · all papers</h1>
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
  const [firstFrameMs, setFirstFrameMs] = useState<number | null>(null);

  useEffect(() => {
    const started = performance.now();
    const e = createPointEngine(containerRef.current!, data);
    setEngine(e);
    if (import.meta.env.DEV) (window as unknown as { pointEngine: typeof e }).pointEngine = e; // for debugging in the console
    requestAnimationFrame(() => setFirstFrameMs(data.loadMs + performance.now() - started));
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
          <PointHud engine={engine} firstFrameMs={firstFrameMs} />
          <PaperPanel engine={engine} />
          <HoverTip engine={engine} />
        </>
      )}
    </>
  );
}

function PointHud({ engine, firstFrameMs }: { engine: PointEngine; firstFrameMs: number | null }) {
  usePointEngine(engine);
  const { meta, bytes } = engine.data;
  const noiseShare = Math.round((100 * meta.noisePapers) / meta.count);

  return (
    <div id="hud" style={{ pointerEvents: 'auto' }}>
      <h1>Research Map · all papers</h1>
      <p id="stats">
        {meta.count.toLocaleString()} papers · {meta.clusters.length} clusters ·{' '}
        {meta.noisePapers.toLocaleString()} unclustered ({noiseShare}%)
      </p>
      <p className="hint">
        Every paper is a dot; color = cluster, grey haze = unclustered (noise). Bigger dots are more influential.
        Scroll to zoom, drag to pan, click a dot to select it.
      </p>
      <p className="hint">
        Prototype · {(bytes / 1048576).toFixed(1)} MB
        {firstFrameMs !== null && ` · first frame in ${Math.round(firstFrameMs)} ms`} ·{' '}
        <a href="#" onClick={(e) => { e.preventDefault(); engine.resetView(); }} style={{ color: 'inherit' }}>
          reset view
        </a>{' '}
        · <a href={import.meta.env.BASE_URL} style={{ color: 'inherit' }}>back to cluster view</a>
      </p>
      <p className="hint">
        Click a dot to draw its citations: <span style={{ color: REF_COLOR }}>blue</span> = papers it cites,{' '}
        <span style={{ color: CITER_COLOR }}>orange</span> = papers citing it. Esc or empty space clears.
      </p>
    </div>
  );
}

function HoverTip({ engine }: { engine: PointEngine }) {
  usePointEngine(engine);
  const h = engine.hovered;
  if (!h) return null;
  const c = engine.clusterOf(h.index);
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
      <div>{h.title ?? 'Loading…'}</div>
      <div style={{ color: engine.clusterCss(c?.id ?? -1), marginTop: 2 }}>
        {c ? stripPrefix(c.title) : 'Unclustered (noise)'}
        {engine.data.year[h.index] > 0 && <span style={{ color: '#8b92a5' }}> · {engine.data.year[h.index]}</span>}
      </div>
    </div>
  );
}
