import { useEngineState } from '../hooks/useEngineState';
import type { Engine } from '../lib/engine';

/** Small tooltip that follows the mouse while it rests on a highlighted citation line. */
export function EdgeTipView({ engine }: { engine: Engine }) {
  useEngineState(engine);
  const tip = engine.edgeTip;
  return (
    <div id="edge-tip" hidden={!tip} style={tip ? { left: tip.x, top: tip.y } : undefined}>
      {tip && (
        <>
          {tip.verb} <b>{tip.title}</b>
          <br />
          <span>Click to go there</span>
        </>
      )}
    </div>
  );
}
