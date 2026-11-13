import { useEffect, useRef, useState } from 'react';
import { createEngine } from '../lib/engine';
import type { Engine } from '../lib/engine';
import type { Overview } from '../types';
import { DetailsPanel } from './DetailsPanel';
import { EdgeTipView } from './EdgeTipView';
import { Hud } from './Hud';
import { Timeline } from './Timeline';

/** The map and its panels. The engine draws the 3D scene into the #graph div; the panels are React. */
export function ResearchMap({ overview }: { overview: Overview }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [engine, setEngine] = useState<Engine | null>(null);

  useEffect(() => {
    const e = createEngine(containerRef.current!, overview);
    setEngine(e);
    return () => {
      e.destroy();
      setEngine(null);
    };
  }, [overview]);

  return (
    <>
      <div id="graph" ref={containerRef} />
      {engine && (
        <>
          <Hud engine={engine} />
          <DetailsPanel engine={engine} />
          <Timeline engine={engine} />
          <EdgeTipView engine={engine} />
        </>
      )}
    </>
  );
}
