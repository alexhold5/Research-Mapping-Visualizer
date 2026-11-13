import { useEffect } from 'react';
import { startLegacyApp } from './legacy-main.js';

/**
 * Step 1 of the React port: the page skeleton is now JSX, but the graph and every panel are still
 * driven by the old imperative code (legacy-main.js), which finds these elements by id.
 * Later steps replace each block below with a real component.
 */
export function App() {
  useEffect(() => {
    startLegacyApp();
  }, []);

  return (
    <>
      <div id="graph" />
      <div id="hud">
        <h1>Research Map</h1>
        <p id="stats">Loading…</p>
        <p id="hint" className="hint" />
        <div id="selection" hidden />
        <h2 className="section">Filter by arXiv field</h2>
        <div id="legend" />
        <h2 className="section">Find a cluster</h2>
        <div id="search">
          <input
            id="search-input"
            type="search"
            placeholder="Search topics, e.g. federated, diffusion…"
            autoComplete="off"
            spellCheck={false}
          />
          <ul id="search-results" hidden />
        </div>
      </div>
      <aside id="details" hidden aria-live="polite">
        <button id="details-close" type="button" aria-label="Close details" title="Close (Esc)">
          ×
        </button>
        <h2 id="details-title" />
        <p id="details-meta" />
        <div id="details-stats" />
        <p id="details-cluster" />
        <h3>Authors</h3>
        <p id="details-authors" />
        <h3>Abstract</h3>
        <p id="details-abstract" />
        <a id="details-link" target="_blank" rel="noopener noreferrer">
          View on arXiv ↗
        </a>
      </aside>
      <div id="timeline" hidden>
        <div id="tl-head">
          <button
            id="tl-mode"
            type="button"
            role="switch"
            aria-checked="false"
            title="Switch what the bars show"
          >
            <span className="sw" />
            <span id="tl-mode-text">Paper volume</span>
          </button>
          <div id="tl-label" />
          <button
            id="tl-reset"
            type="button"
            title="Show the whole map again and clear all selections and filters"
          >
            Reset view
          </button>
        </div>
        <div id="tl-track">
          <div id="tl-bars" />
          <div id="tl-range" />
          <div className="tl-handle" id="tl-lo" title="Earliest date shown" />
          <div className="tl-handle" id="tl-hi" title="Latest date shown" />
        </div>
        <div id="tl-ticks" />
      </div>
    </>
  );
}
