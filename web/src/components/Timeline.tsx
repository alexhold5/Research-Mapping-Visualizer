import { useRef, useState } from 'react';
import type { PointerEvent as ReactPointerEvent } from 'react';
import { useEngineState } from '../hooks/useEngineState';
import type { Engine } from '../lib/engine';

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const fmt = (y: number) => `${MONTHS[Math.min(11, Math.floor((y - Math.floor(y)) * 12))]} ${Math.floor(y)}`;

type BarMode = 'volume' | 'importance';

/**
 * The time window at the bottom: a histogram with two draggable ends. Dragging an end inward hides
 * papers outside that date. The bars describe whatever is in focus (all clusters, or the focused
 * one, for all fields or the selected one).
 */
export function Timeline({ engine }: { engine: Engine }) {
  useEngineState(engine);
  const { years } = engine.overview;
  const { min: T0, max: T1 } = engine.timeRange;
  const span = T1 - T0;
  const minGap = span * 0.01; // handles can't cross or fully overlap
  const pct = (y: number) => `${((y - T0) / span) * 100}%`;

  const { windowLo, windowHi } = engine;
  const data = engine.histogram();
  const [mode, setMode] = useState<BarMode>('volume');
  const trackRef = useRef<HTMLDivElement>(null);

  // volume = papers that year (sqrt-scaled so the recent spike doesn't flatten the rest);
  // importance = average influence of that year's papers (an average, so it doesn't just repeat the volume chart).
  const maxCount = Math.max(1, ...data.count);
  const maxImportance = Math.max(1e-9, ...data.inf);

  const ticks: number[] = [];
  for (let y = Math.ceil(T0 / 5) * 5; y <= T1; y += 5) ticks.push(y);

  const handleProps = (which: 'lo' | 'hi') => ({
    onPointerDown: (e: ReactPointerEvent<HTMLDivElement>) => {
      e.currentTarget.setPointerCapture(e.pointerId);
      e.currentTarget.classList.add('dragging');
      e.preventDefault();
    },
    onPointerMove: (e: ReactPointerEvent<HTMLDivElement>) => {
      if (!e.currentTarget.hasPointerCapture(e.pointerId) || !trackRef.current) return;
      const rect = trackRef.current.getBoundingClientRect();
      const p = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
      const y = T0 + p * span;
      if (which === 'lo') engine.setWindow(Math.max(T0, Math.min(y, engine.windowHi - minGap)), engine.windowHi);
      else engine.setWindow(engine.windowLo, Math.min(T1, Math.max(y, engine.windowLo + minGap)));
    },
    onPointerUp: (e: ReactPointerEvent<HTMLDivElement>) => e.currentTarget.classList.remove('dragging'),
    onPointerCancel: (e: ReactPointerEvent<HTMLDivElement>) => e.currentTarget.classList.remove('dragging'),
  });

  return (
    <div id="timeline">
      <div id="tl-head">
        <button
          id="tl-mode"
          type="button"
          role="switch"
          aria-checked={mode === 'importance'}
          title="Switch what the bars show"
          onClick={() => setMode(mode === 'volume' ? 'importance' : 'volume')}
        >
          <span className="sw" />
          <span id="tl-mode-text">{mode === 'volume' ? 'Paper volume' : 'Paper importance'}</span>
        </button>
        <div id="tl-label">
          {fmt(windowLo)} – {fmt(windowHi)} <span className="hint">· {engine.timelineSummary()}</span>
        </div>
        <button id="tl-reset" type="button" title="Show the whole map again and clear all selections and filters" onClick={() => engine.resetAll()}>
          Reset view
        </button>
      </div>
      <div id="tl-track" ref={trackRef}>
        <div id="tl-bars">
          {years.map((year, i) => {
            const a = Math.max(year, T0);
            const b = Math.min(year + 1, T1);
            const h = mode === 'volume' ? Math.sqrt(data.count[i] / maxCount) : data.inf[i] / maxImportance;
            return (
              <div
                key={year}
                className={mode === 'importance' ? 'importance' : undefined}
                style={{
                  left: pct(a),
                  width: `calc(${((b - a) / span) * 100}% - 1px)`,
                  height: `${data.count[i] === 0 ? 0 : Math.max(6, h * 100)}%`,
                }}
                title={`${year}: ${data.count[i].toLocaleString()} papers · avg importance ${data.inf[i].toFixed(2)}`}
              />
            );
          })}
        </div>
        <div id="tl-range" style={{ left: pct(windowLo), width: `${((windowHi - windowLo) / span) * 100}%` }} />
        <div className="tl-handle" id="tl-lo" title="Earliest date shown" style={{ left: pct(windowLo) }} {...handleProps('lo')} />
        <div className="tl-handle" id="tl-hi" title="Latest date shown" style={{ left: pct(windowHi) }} {...handleProps('hi')} />
      </div>
      <div id="tl-ticks">
        {ticks.map((y) => (
          <span key={y} style={{ left: pct(y) }}>
            {y}
          </span>
        ))}
      </div>
    </div>
  );
}
