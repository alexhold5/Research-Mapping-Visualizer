import { useEffect, useRef, useState } from 'react';
import { createEngine } from '../lib/engine';
import type { Engine } from '../lib/engine';
import { mapDir } from '../hooks/useOverview';
import type { Overview } from '../types';
import { DetailsPanel } from './DetailsPanel';
import { EdgeTipView } from './EdgeTipView';
import { Hud } from './Hud';
import { Timeline } from './Timeline';

/** The map and its panels. The engine draws the 3D scene into the #graph div; the panels are React. */
export function ResearchMap({ overview, onOpenMap }: { overview: Overview; onOpenMap: (id: string) => void }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [engine, setEngine] = useState<Engine | null>(null);

  useEffect(() => {
    // A drill-down map's cluster files sit next to its overview (data/maps/<id>/); the original map's in data/.
    const e = createEngine(containerRef.current!, overview, {
      dataDir: mapDir(overview.meta.mapId ?? null),
      onOpenMap,
      slim: overview.meta.mapId !== undefined,
    });
    setEngine(e);
    return () => {
      e.destroy();
      setEngine(null);
    };
  }, [overview, onOpenMap]);

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
