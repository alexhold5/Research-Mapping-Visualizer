import { useEffect, useState } from 'react';
import type { PaperDetail, PaperNode } from '../types';

// Abstracts and authors live in a separate file (1.4 MB), fetched the first time one is needed
// and then kept for the rest of the session.
let papersPromise: Promise<Record<string, PaperDetail>> | null = null;
const loadPapers = () => (papersPromise ??= fetch('/data/papers.json').then((r) => r.json()));

export type DetailState =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'ready'; detail: PaperDetail }
  | { status: 'error' };

/** Fetches the abstract and authors for the selected paper, ignoring stale responses. */
export function usePaperDetail(node: PaperNode | null): DetailState {
  const [state, setState] = useState<DetailState>({ status: 'idle' });

  useEffect(() => {
    if (!node) {
      setState({ status: 'idle' });
      return;
    }
    let cancelled = false; // the user clicked something else while this was loading
    setState({ status: 'loading' });

    loadPapers()
      .then((papers) => {
        if (cancelled) return;
        const detail = papers[node.id];
        if (!detail) throw new Error(`${node.id} not found in papers.json`);
        setState({ status: 'ready', detail });
      })
      .catch((err) => {
        console.error(err);
        if (!cancelled) setState({ status: 'error' });
      });

    return () => {
      cancelled = true;
    };
  }, [node]);

  return state;
}
