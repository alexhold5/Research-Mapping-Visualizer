import { ResearchMap } from './components/ResearchMap';
import { useOverview } from './hooks/useOverview';

export function App() {
  const state = useOverview();

  if (state.status === 'ready') return <ResearchMap overview={state.overview} />;

  return (
    <div id="hud">
      <h1>Research Map</h1>
      <p id="stats">{state.status === 'loading' ? 'Loading…' : `Failed to load data: ${state.message}`}</p>
    </div>
  );
}
