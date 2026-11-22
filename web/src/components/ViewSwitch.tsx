export type View = 'clustered' | 'global';

/** The slider at the top of the screen: ( Clustered View | Global View ). Clicking anywhere on it swaps to the other view. */
export function ViewSwitch({ view, onChange }: { view: View; onChange: (view: View) => void }) {
  const next: View = view === 'clustered' ? 'global' : 'clustered';
  return (
    <button
      id="view-switch"
      type="button"
      role="switch"
      aria-checked={view === 'global'}
      aria-label={`Switch to ${next === 'global' ? 'Global' : 'Clustered'} View`}
      data-view={view}
      onClick={() => onChange(next)}
    >
      <span className="knob" />
      <span className={view === 'clustered' ? 'on' : ''}>Clustered View</span>
      <span className={view === 'global' ? 'on' : ''}>Global View</span>
    </button>
  );
}
