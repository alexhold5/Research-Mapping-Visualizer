import { useEngineState } from '../hooks/useEngineState';
import type { Engine } from '../lib/engine';
import { stripPrefix } from '../lib/html';
import { ClusterSearch } from './ClusterSearch';
import { FieldChips } from './FieldChips';
import { Swatch } from './Swatch';

/** The left panel: title, what you are looking at, the field filter and the cluster search. */
export function Hud({ engine }: { engine: Engine }) {
  useEngineState(engine);
  const { meta } = engine.overview;
  const focused = engine.focused;

  return (
    <div id="hud">
      <h1>Research Map</h1>
      {focused ? (
        <>
          <p id="stats">{focused.cluster.size.toLocaleString()} papers in this cluster</p>
          <p id="hint" className="hint">
            Showing its {engine.focusedPaperCount()} most influential papers (size = influence rank). Arrows point to
            the cited paper; lines to other clusters end at their bubble until you zoom into it. Drag to pan to a
            neighbor.
          </p>
          <div id="selection">
            <button id="back" type="button" onClick={() => engine.navigate('')}>
              ← All clusters
            </button>
            <div className="banner-title">
              <Swatch color={engine.fieldHelpers.clusterColor(focused.cluster)} />
              {stripPrefix(focused.cluster.title)}
            </div>
          </div>
        </>
      ) : (
        <>
          <p id="stats">
            {meta.clusterCount} clusters · {meta.clusteredPapers.toLocaleString()} clustered papers ·{' '}
            {meta.noisePapers.toLocaleString()} unclustered (not shown)
          </p>
          <p id="hint" className="hint">
            Bubble size = papers in the cluster · color = main arXiv field · lines = citations between clusters. Scroll
            to zoom toward the pointer (or click a bubble) to see inside it; drag to pan.
          </p>
          <div id="selection" hidden />
        </>
      )}
      <h2 className="section">Filter by arXiv field</h2>
      <FieldChips engine={engine} />
      <h2 className="section">Find a cluster</h2>
      <ClusterSearch engine={engine} />
    </div>
  );
}
