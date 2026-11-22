# Cluster dynamics

Tracks each cluster's **relative paper volume** (its share of all papers published that year) across publication years. That separates rapidly expanding subfields from mature or contracting ones. Shares are used instead of raw counts because the arXiv dataset is heavily weighted toward 2025–2026.

## Files

- `../cluster_dynamics.py`: computes per-cluster yearly counts and shares, the annual change in share over the last N years (default 5), and an expanding / stable / contracting label. Writes `cluster_dynamics.json`.
- `index.html`: interactive view of that JSON. It has a growth-vs-maturity scatter, a UMAP map with a year scrubber, small multiples of yearly share, and a table.
- `cluster_dynamics.json`: the current output. It comes from the **full dataset** (287,421 papers) in `enriched_arXiv_scientific_dataset.csv`, run through `ml_pipeline.py`'s SciBERT → UMAP → HDBSCAN steps with `min_cluster_size=500`. That gives 47 clusters plus a noise group holding about 18.5% of papers. The map view draws a 15,000-paper random sample (seed 42).

## Regenerate

```bash
python ml_pipeline.py                      # writes enriched_<csv>.csv with cluster_label / cluster_title
python cluster_dynamics.py enriched_arXiv_scientific_dataset.csv --out visualization/cluster_dynamics.json
cd visualization && python -m http.server  # then open http://localhost:8000
```

The page loads the JSON with `fetch`, so it must be served over HTTP. Opening it with `file://` won't work.

# Presentation analyses

`presentation.html` is a one-page summary for the project presentation (not the production site). It loads `presentation_data.json`, which `../presentation_analysis.py` builds from the full clustered dataset (`enriched_arXiv_scientific_full_dataset.csv`, ~3.2M papers, 8,490 clusters within 152 primary categories) and the full citation graph (`citation_graph_adjacency_full.json`, 35M references between papers in the dataset).

1. **Cluster dynamics**: each cluster's share of each year's papers, with a log-linear growth rate over the last 5 years (expanding at least +10%/yr, contracting at most -10%/yr). Publication year comes from the arXiv ID, not `updated_date`.
2. **Influence efficiency**: in-corpus citations divided by the mean for the same primary category and publication year, averaged per cluster, then compared across size quartiles and growth status. Normalizing within each field matters because citation habits differ a lot (astro-ph papers list ~29 arXiv references, cs papers ~5).
3. **Cross-disciplinary breakthroughs**: top-2% papers by normalized citations, ranked by a boundary score that combines embedding-neighborhood cluster entropy with the cluster and top-level-field entropy of their citers.

`../citation_graph.py` streams the adjacency JSON into integer edge arrays so the 786 MB file never has to fit in memory as Python objects.

```bash
python presentation_analysis.py enriched_arXiv_scientific_full_dataset.csv --edge-cache edges.npz --out visualization/presentation_data.json
cd visualization && python -m http.server   # open http://localhost:8000/presentation.html
```

The script also accepts a Parquet copy of the CSV, which loads much faster. `--edge-cache` saves the parsed graph so later runs skip the JSON.
