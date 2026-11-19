import { ResearchMap } from './components/ResearchMap';
import { useOverview } from './hooks/useOverview';
import { PointMap } from './points/PointMap';

/** ?view=points opens the point-cloud prototype; anything else is the cluster view. */
const POINTS_VIEW = new URLSearchParams(window.location.search).get('view') === 'points';

export function App() {
  return POINTS_VIEW ? <PointMap /> : <ClusterApp />;
}

function ClusterApp() {
  const state = useOverview();

  if (state.status === 'ready') return <ResearchMap overview={state.overview} />;

  return (
    <div id="hud">
      <h1>Research Map</h1>
      <p id="stats">{state.status === 'loading' ? 'Loading…' : `Failed to load data: ${state.message}`}</p>
    </div>
  );
}
