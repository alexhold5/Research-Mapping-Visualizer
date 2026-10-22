import { useEffect, useState } from 'react';
import { buildNodes } from '../lib/graph';
import type { PaperNode, RawGraphData } from '../types';

export interface PreparedGraph extends RawGraphData {
  /** The exported nodes, scaled into world space, sized, and de-overlapped. */
  paperNodes: PaperNode[];
}

type State =
  | { status: 'loading' }
  | { status: 'ready'; graph: PreparedGraph }
  | { status: 'error'; message: string };

/**
 * Loads graph.json once on mount and prepares it for rendering.
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
        const paperNodes = buildNodes(data.nodes, data.meta);
        if (cancelled) return;
        setState({ status: 'ready', graph: { ...data, paperNodes } });
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
