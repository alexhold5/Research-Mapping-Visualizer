import { useCallback, useEffect, useState } from 'react';
import { ResearchMap } from './components/ResearchMap';
import { useOverview } from './hooks/useOverview';

/** ?map=cs.AI picks a drill-down map (data/maps/cs.AI/); without it, the original single map (data/). */
const mapFromUrl = (): string | null => new URLSearchParams(window.location.search).get('map');

export function App() {
  const [mapId, setMapId] = useState(mapFromUrl);
  const state = useOverview(mapId);

  // The back button can move between maps: re-read the map from the URL.
  useEffect(() => {
    const onPop = () => setMapId(mapFromUrl());
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, []);

  /**
   * Switch to another map, optionally flying to one of its clusters. The URL gets ?map=<id>, plus #cluster=<n> if
   * given (any old #cluster=... belonged to the old map); the new map's engine reads that hash when it starts.
   */
  const openMap = useCallback((id: string, cluster?: number) => {
    const params = new URLSearchParams(window.location.search);
    params.set('map', id);
    history.pushState(null, '', `${window.location.pathname}?${params}${cluster === undefined ? '' : `#cluster=${cluster}`}`);
    setMapId(id);
  }, []);

  if (state.status === 'ready') return <ResearchMap overview={state.overview} onOpenMap={openMap} />;

  return (
    <div id="hud">
      <h1>Research Map</h1>
      <p id="stats">{state.status === 'loading' ? 'Loading…' : `Failed to load data: ${state.message}`}</p>
    </div>
  );
}
