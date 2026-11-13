import { useMemo, useState } from 'react';
import type { Engine } from '../lib/engine';
import { stripPrefix } from '../lib/html';
import { Swatch } from './Swatch';

const MAX_RESULTS = 8;

/** Type a topic (or just focus the box for the biggest clusters) and jump to a cluster. */
export function ClusterSearch({ engine }: { engine: Engine }) {
  const { clusters } = engine.overview;
  const [query, setQuery] = useState('');
  const [listOpen, setListOpen] = useState(false);
  const [active, setActive] = useState(-1); // keyboard-highlighted result

  const searchIndex = useMemo(
    () =>
      clusters.map((c) => ({
        c,
        text: `${stripPrefix(c.title)} ${c.topPaper} ${c.categoryName} ${c.category}`.toLowerCase(),
      })),
    [clusters],
  );

  const q = query.trim().toLowerCase();
  const shown = useMemo(() => {
    const terms = q.split(/\s+/).filter(Boolean);
    return searchIndex
      .filter((e) => terms.every((t) => e.text.includes(t)))
      .map((e) => e.c)
      .sort((a, b) => b.size - a.size)
      .slice(0, MAX_RESULTS);
  }, [searchIndex, q]);

  const close = () => {
    setListOpen(false);
    engine.setSearchHover(null);
  };
  const open = (i: number) => {
    const c = shown[i];
    if (!c) return;
    setQuery('');
    close();
    (document.activeElement as HTMLElement | null)?.blur();
    engine.navigate(`#cluster=${c.id}`);
  };
  // The first match is highlighted as soon as there is something typed.
  const activeIndex = active >= 0 && active < shown.length ? active : shown.length && q ? 0 : -1;

  return (
    <div id="search">
      <input
        id="search-input"
        type="search"
        placeholder="Search topics, e.g. federated, diffusion…"
        autoComplete="off"
        spellCheck={false}
        value={query}
        onFocus={() => setListOpen(true)}
        onBlur={close}
        onChange={(e) => {
          setQuery(e.target.value);
          setActive(-1);
          setListOpen(true);
        }}
        onKeyDown={(e) => {
          if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
            e.preventDefault();
            if (!shown.length) return;
            const step = e.key === 'ArrowDown' ? 1 : -1;
            setActive((Math.max(activeIndex, 0) + (activeIndex < 0 ? 0 : step) + shown.length) % shown.length);
          } else if (e.key === 'Enter') {
            open(activeIndex >= 0 ? activeIndex : 0);
          } else if (e.key === 'Escape') {
            setQuery('');
            close();
            e.currentTarget.blur();
          }
        }}
      />
      <ul
        id="search-results"
        hidden={!listOpen}
        // mousedown (not click) so the input doesn't lose focus and hide the list before the click lands.
        onMouseDown={(e) => {
          e.preventDefault();
          const li = (e.target as HTMLElement).closest('li[data-i]') as HTMLElement | null;
          if (li) open(Number(li.dataset.i));
        }}
        onMouseOver={(e) => {
          const li = (e.target as HTMLElement).closest('li[data-i]') as HTMLElement | null;
          engine.setSearchHover(li ? shown[Number(li.dataset.i)].id : null);
        }}
        onMouseLeave={() => engine.setSearchHover(null)}
      >
        {shown.length === 0 ? (
          <li className="empty">No clusters match.</li>
        ) : (
          <>
            {!q && <li className="heading">Largest clusters</li>}
            {shown.map((c, i) => (
              <li key={c.id} data-i={i} className={i === activeIndex ? 'active' : ''}>
                <button type="button">
                  <Swatch color={engine.fieldHelpers.clusterColor(c)} />
                  <span className="name">{stripPrefix(c.title)}</span>
                  <span className="count">{c.size.toLocaleString()}</span>
                </button>
              </li>
            ))}
          </>
        )}
      </ul>
    </div>
  );
}
