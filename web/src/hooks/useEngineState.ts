import { useSyncExternalStore } from 'react';
import type { Engine } from '../lib/engine';

/**
 * Re-render the calling component whenever the engine reports a change. The engine's state is read
 * straight off the engine object during render; this hook only supplies the "something changed" signal.
 */
export function useEngineState(engine: Engine): void {
  useSyncExternalStore(engine.subscribe, engine.getVersion);
}
