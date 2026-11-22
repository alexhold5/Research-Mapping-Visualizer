import { useCallback, useEffect, useState } from 'react';
import { ResearchMap } from './components/ResearchMap';
import { ViewSwitch } from './components/ViewSwitch';
import type { View } from './components/ViewSwitch';
import { useOverview } from './hooks/useOverview';
import { PointMap } from './points/PointMap';

/** The top of the drill-down hierarchy, which the Clustered View opens at. */
const ROOT_MAP = 'arxiv';

/** ?view=global is the point map; anything else is the drill-down maps. */
const viewFromUrl = (): View =>
  new URLSearchParams(window.location.search).get('view') === 'global' ? 'global' : 'clustered';

/** ?map=cs.AI picks a drill-down map (data/maps/cs.AI/). Without it: all of arXiv. */
const mapFromUrl = (): string => new URLSearchParams(window.location.search).get('map') ?? ROOT_MAP;

export function App() {
  const [view, setView] = useState(viewFromUrl);
  const [mapId, setMapId] = useState(mapFromUrl);

  // The back button can move between views and maps: re-read both from the URL.
  useEffect(() => {
    const onPop = () => {
      setView(viewFromUrl());
      setMapId(mapFromUrl());
    };
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, []);

  /**
   * Switch to another map, optionally flying to one of its clusters. The URL gets ?map=<id>, plus #cluster=<n> if
   * given (any old #cluster=... belonged to the old map); the new map's engine reads that hash when it starts.
   */
  const openMap = useCallback((id: string, cluster?: number) => {
    const params = new URLSearchParams(window.location.search);
    params.delete('view');
    params.set('map', id);
    history.pushState(null, '', `${window.location.pathname}?${params}${cluster === undefined ? '' : `#cluster=${cluster}`}`);
    setView('clustered');
    setMapId(id);
  }, []);

  /** Global -> Clustered always lands on the top map (all of arXiv); Clustered -> Global opens the point map. */
  const changeView = useCallback((next: View) => {
    if (next === 'global') {
      history.pushState(null, '', `${window.location.pathname}?view=global`);
      setView('global');
    } else {
      openMap(ROOT_MAP);
    }
  }, [openMap]);

  return (
    <>
      {view === 'global' ? <PointMap /> : <DrillDown mapId={mapId} onOpenMap={openMap} />}
      <ViewSwitch view={view} onChange={changeView} />
    </>
  );
}

function DrillDown({ mapId, onOpenMap }: { mapId: string; onOpenMap: (id: string, cluster?: number) => void }) {
  const state = useOverview(mapId);
  if (state.status === 'ready') return <ResearchMap overview={state.overview} onOpenMap={onOpenMap} />;
  return (
    <div id="hud">
      <h1>Research Map</h1>
      <p id="stats">{state.status === 'loading' ? 'Loading…' : `Failed to load data: ${state.message}`}</p>
    </div>
  );
}
