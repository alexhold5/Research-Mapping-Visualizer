import { useEffect, useState } from 'react';
import { fetchJson } from '../lib/html';
import { expandOverview, isSlimOverview } from '../lib/slim';
import type { Overview } from '../types';

type State =
  | { status: 'loading' }
  | { status: 'ready'; overview: Overview }
  | { status: 'error'; message: string };
type Loaded = Exclude<State, { status: 'loading' }> & { mapId: string | null };

/** Folder (under the site root) holding a map's overview.json and clusters/. null = the original single map. */
export const mapDir = (mapId: string | null): string =>
  `${import.meta.env.BASE_URL}data/${mapId ? `maps/${mapId}/` : ''}`;

/**
 * Loads a map's overview.json (the bubbles, fields and timeline), again whenever the map changes.
 * Files in web/public/ are served from the site root.
 */
export function useOverview(mapId: string | null): State {
  // What finished loading last, and for which map. Right after the map changes it still holds the old
  // map's data, which must not be shown (or handed to the engine) as the new map's.
  const [loaded, setLoaded] = useState<Loaded | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchJson<unknown>(`${mapDir(mapId)}overview.json`)
      .then((raw) => {
        const overview = isSlimOverview(raw) ? expandOverview(raw) : (raw as Overview);
        if (!cancelled) setLoaded({ status: 'ready', overview, mapId });
      })
      .catch((err: Error) => {
        console.error(err);
        if (!cancelled) setLoaded({ status: 'error', message: err.message, mapId });
      });
    return () => {
      cancelled = true;
    };
  }, [mapId]);

  return loaded && loaded.mapId === mapId ? loaded : { status: 'loading' };
}
