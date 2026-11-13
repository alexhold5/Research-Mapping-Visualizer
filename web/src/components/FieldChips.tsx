import { useEngineState } from '../hooks/useEngineState';
import type { Engine } from '../lib/engine';
import { Swatch } from './Swatch';

/** The arXiv field chips: they are the color key and also the field filter. */
export function FieldChips({ engine }: { engine: Engine }) {
  useEngineState(engine);
  const { fields } = engine.overview;
  const chips = [
    ...fields.map((f) => ({
      key: f.code,
      label: f.code,
      title: `${f.name} · ${f.papers.toLocaleString()} clustered papers`,
    })),
    { key: 'other', label: 'other', title: 'All remaining arXiv fields' },
  ];
  const active = engine.fieldFilter;

  return (
    <div id="legend" className={active !== null ? 'filtering' : undefined}>
      {chips.map((f) => (
        <button
          key={f.key}
          type="button"
          className={`chip${f.key === active ? ' active' : ''}`}
          data-field={f.key}
          title={f.title}
          onClick={() => engine.setFieldFilter(f.key)}
        >
          <Swatch color={engine.fieldHelpers.color(f.key)} />
          {f.label}
        </button>
      ))}
    </div>
  );
}
