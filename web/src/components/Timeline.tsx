import { useCallback, useMemo, useRef, useState } from 'react';
import type { PaperNode } from '../types';

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const fmt = (y: number) => `${MONTHS[Math.min(11, Math.floor((y - Math.floor(y)) * 12))]} ${Math.floor(y)}`;

type BarMode = 'volume' | 'importance';

interface YearStat {
  year: number;
  count: number;
  /** Mean influence score of that year's papers. */
  importance: number;
}

interface Props {
  nodes: PaperNode[];
  yearMin: number;
  yearMax: number;
  lo: number;
  hi: number;
  visibleCount: number;
  onWindowChange: (lo: number, hi: number) => void;
  onReset: () => void;
}

/** Two-handle slider: drag either end inward to hide papers outside that date. */
export function Timeline({
  nodes,
  yearMin: T0,
  yearMax: T1,
  lo,
  hi,
  visibleCount,
  onWindowChange,
  onReset,
}: Props) {
  const [barMode, setBarMode] = useState<BarMode>('volume');
  const trackRef = useRef<HTMLDivElement>(null);

  const span = T1 - T0;
  const MIN_GAP = span * 0.01; // handles can't cross or fully overlap
  const pct = useCallback((y: number) => `${((y - T0) / span) * 100}%`, [T0, span]);

  // One bar per year, with two views of the same bars (flip with the switch):
  //   volume     = papers that year (sqrt-scaled so the 2025-26 spike doesn't flatten the rest)
  //   importance = average influence score of that year's papers (the value that sets node size).
  //                Average rather than total, so it doesn't just repeat the volume chart.
  const { yearStats, maxCount, maxImportance } = useMemo(() => {
    const byYear = new Map<number, { count: number; influence: number }>();
    for (const n of nodes) {
      if (n.year == null) continue;
      const y = Math.floor(n.year);
      const e = byYear.get(y) ?? { count: 0, influence: 0 };
      e.count += 1;
      e.influence += n.influence;
      byYear.set(y, e);
    }
    const stats: YearStat[] = [...byYear].map(([year, e]) => ({
      year,
      count: e.count,
      importance: e.influence / e.count,
    }));
    return {
      yearStats: stats,
      maxCount: Math.max(...stats.map((s) => s.count)),
      maxImportance: Math.max(...stats.map((s) => s.importance)),
    };
  }, [nodes]);

  // Year ticks every 5 years.
  const ticks = useMemo(() => {
    const out: number[] = [];
    for (let y = Math.ceil(T0 / 5) * 5; y <= T1; y += 5) out.push(y);
    return out;
  }, [T0, T1]);

  // Dragging fires far faster than we want to re-filter the graph, so coalesce to one
  // update per animation frame (same as the vanilla version did).
  const pending = useRef<[number, number] | null>(null);
  const queued = useRef(false);
  const schedule = useCallback(
    (nextLo: number, nextHi: number) => {
      pending.current = [nextLo, nextHi];
      if (queued.current) return;
      queued.current = true;
      requestAnimationFrame(() => {
        queued.current = false;
        if (pending.current) onWindowChange(...pending.current);
      });
    },
    [onWindowChange],
  );

  const makeHandleProps = (which: 'lo' | 'hi') => ({
    onPointerDown: (e: React.PointerEvent<HTMLDivElement>) => {
      e.currentTarget.setPointerCapture(e.pointerId);
      e.currentTarget.classList.add('dragging');
      e.preventDefault();
    },
    onPointerMove: (e: React.PointerEvent<HTMLDivElement>) => {
      if (!e.currentTarget.hasPointerCapture(e.pointerId)) return;
      const rect = trackRef.current!.getBoundingClientRect();
      const p = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
      const y = T0 + p * span;
      if (which === 'lo') schedule(Math.max(T0, Math.min(y, hi - MIN_GAP)), hi);
      else schedule(lo, Math.min(T1, Math.max(y, lo + MIN_GAP)));
    },
    onPointerUp: (e: React.PointerEvent<HTMLDivElement>) => e.currentTarget.classList.remove('dragging'),
    onPointerCancel: (e: React.PointerEvent<HTMLDivElement>) => e.currentTarget.classList.remove('dragging'),
  });

  return (
    <div id="timeline">
      <div id="tl-head">
        <button
          id="tl-mode"
          type="button"
          role="switch"
          aria-checked={barMode === 'importance'}
          title="Switch what the bars show"
          onClick={() => setBarMode((m) => (m === 'volume' ? 'importance' : 'volume'))}
        >
          <span className="sw" />
          <span id="tl-mode-text">{barMode === 'volume' ? 'Paper volume' : 'Paper importance'}</span>
        </button>

        <div id="tl-label">
          {fmt(lo)} – {fmt(hi)}{' '}
          <span className="hint">
            · {visibleCount} of {nodes.length} papers
          </span>
        </div>

        <button
          id="tl-reset"
          type="button"
          title="Show the whole map again and clear all selections and filters"
          onClick={onReset}
        >
          Reset view
        </button>
      </div>

      <div id="tl-track" ref={trackRef}>
        <div id="tl-bars">
          {yearStats.map((s) => {
            const a = Math.max(s.year, T0);
            const b = Math.min(s.year + 1, T1);
            const h = barMode === 'volume' ? Math.sqrt(s.count / maxCount) : s.importance / maxImportance;
            return (
              <div
                key={s.year}
                className={barMode === 'importance' ? 'importance' : undefined}
                style={{
                  left: pct(a),
                  width: `calc(${((b - a) / span) * 100}% - 1px)`,
                  height: `${Math.max(6, h * 100)}%`,
                }}
                title={`${s.year}: ${s.count} papers · avg importance ${s.importance.toFixed(2)}`}
              />
            );
          })}
        </div>

        <div id="tl-range" style={{ left: pct(lo), width: `${((hi - lo) / span) * 100}%` }} />
        <div className="tl-handle" id="tl-lo" title="Earliest date shown" style={{ left: pct(lo) }} {...makeHandleProps('lo')} />
        <div className="tl-handle" id="tl-hi" title="Latest date shown" style={{ left: pct(hi) }} {...makeHandleProps('hi')} />
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
