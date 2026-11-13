import { useEffect, useState } from 'react';
import { fetchJson } from '../lib/html';
import type { Overview } from '../types';

type State =
  | { status: 'loading' }
  | { status: 'ready'; overview: Overview }
  | { status: 'error'; message: string };

/** Loads overview.json (the bubbles, fields and timeline) once on mount. Files in web/public/ are served from the site root. */
export function useOverview(): State {
  const [state, setState] = useState<State>({ status: 'loading' });

  useEffect(() => {
    let cancelled = false;
    fetchJson<Overview>('/data/overview.json')
      .then((overview) => {
        if (!cancelled) setState({ status: 'ready', overview });
      })
      .catch((err: Error) => {
        console.error(err);
        if (!cancelled) setState({ status: 'error', message: err.message });
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return state;
}
