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
  // Drill-down maps (export_maps.py) name themselves and know their parent map; the original map does neither.
  const withMaps = engine.overview.clusters.filter((c) => c.map);
  const isGroupMap = withMaps.length > 0; // its bubbles open other maps (the arXiv map, or a group like cs)
  const unit = meta.level === 'root' ? 'groups' : isGroupMap ? 'categories' : 'clusters';
  const one = meta.level === 'root' ? 'group' : isGroupMap ? 'category' : 'cluster';

  return (
    <div id="hud">
      <h1>Research Map{meta.title ? ` · ${meta.title}` : ''}</h1>
      {meta.parent && (
        <button type="button" className="map-back" onClick={() => engine.openMap(meta.parent!)}>
          ← Up to {meta.parentTitle ?? meta.parent}
        </button>
      )}
      {focused ? (
        <>
          <p id="stats">
            {focused.cluster.size.toLocaleString()} papers in this {one}
          </p>
          <p id="hint" className="hint">
            {isGroupMap
              ? `Showing its ${engine.focusedPaperCount()} ${focused.cluster.inner ?? 'clusters'} (size = papers). Click one to open it.`
              : `Showing ${meta.ranked === 'random' ? `a random sample of ${engine.focusedPaperCount()} papers (no influence scores yet)` : `its ${engine.focusedPaperCount()} most influential papers (size = influence rank)`}. Arrows point to the cited paper; lines to other clusters end at their bubble until you zoom into it. Drag to pan to a neighbor.`}
          </p>
          <div id="selection">
            <button id="back" type="button" onClick={() => engine.navigate('')}>
              ← All {unit}
            </button>
            <div className="banner-title">
              <Swatch color={engine.fieldHelpers.clusterColor(focused.cluster)} />
              {stripPrefix(focused.cluster.title)}
            </div>
            {focused.cluster.map && (
              <button type="button" className="map-back" onClick={() => engine.openMap(focused.cluster.map!)}>
                Open the {focused.cluster.map} map →
              </button>
            )}
          </div>
        </>
      ) : (
        <>
          <p id="stats">
            {meta.clusterCount} {unit} · {meta.clusteredPapers.toLocaleString()} {isGroupMap ? '' : 'clustered '}papers
            {meta.noisePapers > 0 && <> · {meta.noisePapers.toLocaleString()} unclustered (not shown)</>}
          </p>
          {isGroupMap && (
            <p className="hint">
              Click a {one} to open its map, or zoom into it to see its {meta.level === 'root' ? 'categories' : 'clusters'} here
              {withMaps.length < meta.clusterCount && ` (${withMaps.length} of ${meta.clusterCount} have a map so far)`}.
            </p>
          )}
          <p id="hint" className="hint">
            {isGroupMap
              ? `Bubble size = papers in the ${one} · color = ${one}. Scroll to zoom toward the pointer; drag to pan.`
              : 'Bubble size = papers in the cluster · color = main arXiv field · lines = citations between clusters. Scroll to zoom toward the pointer (or click a bubble) to see inside it; drag to pan.'}
          </p>
          <div id="selection" hidden />
        </>
      )}
      <h2 className="section">Filter by arXiv field</h2>
      <FieldChips engine={engine} />
      <h2 className="section">Find a {one}</h2>
      <ClusterSearch engine={engine} />
    </div>
  );
}
