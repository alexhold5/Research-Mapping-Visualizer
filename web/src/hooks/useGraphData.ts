import { useEffect, useMemo, useRef, useState } from 'react';
import { buildNodes } from '../lib/graph';
import type { PaperNode, RawGraphData, ViewMode } from '../types';

type State =
  | { status: 'loading' }
  | { status: 'ready'; graph: RawGraphData }
  | { status: 'error'; message: string };

/**
 * Loads graph.json once on mount.
 * Files in web/public/ are served from the site root, so this is web/public/data/graph.json.
 */
export function useGraphData(): State {
  const [state, setState] = useState<State>({ status: 'loading' });

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const res = await fetch('/data/graph.json');
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        const data: RawGraphData = await res.json();
        if (cancelled) return;
        setState({ status: 'ready', graph: data });
      } catch (err) {
        console.error(err);
        if (!cancelled) setState({ status: 'error', message: (err as Error).message });
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  return state;
}

/**
 * The renderable nodes for one view mode, built on demand and kept.
 *
 * Switching modes re-runs the overlap pass because the spacing papers need depends on how far
 * apart they already are in z, so the two modes genuinely have different layouts. Each result is
 * cached, so a mode is only ever laid out once and flipping back is instant.
 */
export function usePreparedNodes(graph: RawGraphData | null, mode: ViewMode): PaperNode[] | null {
  const cache = useRef(new Map<ViewMode, PaperNode[]>());

  // A new dataset invalidates every layout built from the old one.
  useEffect(() => {
    cache.current.clear();
  }, [graph]);

  return useMemo(() => {
    if (!graph) return null;
    const hit = cache.current.get(mode);
    if (hit) return hit;
    const built = buildNodes(graph.nodes, graph.meta, mode);
    cache.current.set(mode, built);
    return built;
  }, [graph, mode]);
}
