import { useCallback, useMemo, useRef, useState } from 'react';
import { DetailsPanel } from './components/DetailsPanel';
import { Hud } from './components/Hud';
import { ResearchGraph, type ResearchGraphHandle } from './components/ResearchGraph';
import { Timeline } from './components/Timeline';
import { useGraphData, usePreparedNodes } from './hooks/useGraphData';
import { usePaperDetail } from './hooks/usePaperDetail';
import { useWindowSize } from './hooks/useWindowSize';
import type { PaperNode, ViewMode } from './types';

const NO_CLUSTERS: never[] = [];

export function App() {
  const state = useGraphData();
  const { width, height } = useWindowSize();
  const graphRef = useRef<ResearchGraphHandle>(null);

  const [mode, setMode] = useState<ViewMode>('flat');
  const [selectedCluster, setSelectedCluster] = useState<number | null>(null);
  const [selectedPaper, setSelectedPaper] = useState<PaperNode | null>(null);
  // null = the full range. Kept as an override so it resets cleanly and survives data loading.
  const [range, setRange] = useState<[number, number] | null>(null);

  const detail = usePaperDetail(selectedPaper);

  const graph = state.status === 'ready' ? state.graph : null;
  const nodes = usePreparedNodes(graph, mode);
  const clusters = graph?.clusters ?? NO_CLUSTERS;
  const clusterTitle = useMemo(
    () => Object.fromEntries(clusters.map((c) => [c.id, c.title])) as Record<number, string>,
    [clusters],
  );

  // ---- Time window ---------------------------------------------------------
  const T0 = graph?.meta.yearMin ?? 0;
  const T1 = graph?.meta.yearMax ?? 0;
  const [lo, hi] = range ?? [T0, T1];

  // Papers with no date only show while the window covers everything.
  const isVisible = useCallback(
    (n: PaperNode) => (n.year == null ? lo <= T0 && hi >= T1 : n.year >= lo && n.year <= hi),
    [lo, hi, T0, T1],
  );
  const visibleCount = useMemo(() => (nodes ? nodes.filter(isVisible).length : 0), [nodes, isVisible]);

  // ---- Interaction ---------------------------------------------------------
  const selectCluster = useCallback(
    (id: number) => {
      if (selectedCluster === id) {
        setSelectedCluster(null); // clicking the active cluster clears it
        return;
      }
      setSelectedCluster(id);
      setSelectedPaper(null); // the open paper may belong to a different cluster
      graphRef.current?.flyToCluster(id);
    },
    [selectedCluster],
  );

  // Clicking any paper opens its details. What happens to the cluster selection:
  //   paper inside the selected cluster -> keep the selection and the view, just switch the details
  //   anything else                     -> select that paper's cluster and fly the camera to it
  //                                        (noise papers have no cluster, so they just clear the selection)
  const handleNodeClick = useCallback(
    (n: PaperNode) => {
      if (selectedCluster === null || n.cluster !== selectedCluster) {
        const next = n.cluster === -1 ? null : n.cluster;
        setSelectedCluster(next);
        if (next !== null) graphRef.current?.flyToCluster(next);
      }
      setSelectedPaper(n);
    },
    [selectedCluster],
  );

  const handleBackgroundClick = useCallback(() => {
    setSelectedCluster(null);
    setSelectedPaper(null);
  }, []);

  const closeDetails = useCallback(() => setSelectedPaper(null), []);

  const handleWindowChange = useCallback((nextLo: number, nextHi: number) => {
    setRange([nextLo, nextHi]);
  }, []);

  // Back to the starting state: whole map in view, nothing selected, nothing filtered.
  const handleReset = useCallback(() => {
    setSelectedCluster(null);
    setSelectedPaper(null);
    setRange(null);
    graphRef.current?.resetView();
  }, []);

  return (
    <>
      {graph && nodes && (
        <ResearchGraph
          ref={graphRef}
          nodes={nodes}
          links={graph.links}
          mode={mode}
          clusterTitle={clusterTitle}
          selectedCluster={selectedCluster}
          isVisible={isVisible}
          width={width}
          height={height}
          onNodeClick={handleNodeClick}
          onBackgroundClick={handleBackgroundClick}
        />
      )}

      <Hud
        meta={graph?.meta ?? null}
        clusters={clusters}
        selectedCluster={selectedCluster}
        onSelectCluster={selectCluster}
        mode={mode}
        onModeChange={setMode}
        status={state.status === 'error' ? `Failed to load data: ${state.message}` : null}
      />

      {selectedPaper && (
        <DetailsPanel node={selectedPaper} detail={detail} clusters={clusters} onClose={closeDetails} />
      )}

      {graph && nodes && (
        <Timeline
          nodes={nodes}
          yearMin={T0}
          yearMax={T1}
          lo={lo}
          hi={hi}
          visibleCount={visibleCount}
          onWindowChange={handleWindowChange}
          onReset={handleReset}
        />
      )}
    </>
  );
}
