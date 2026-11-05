# Cluster dynamics

Tracks each cluster's **relative paper volume** (its share of all papers published that year) across publication years. That separates rapidly expanding subfields from mature or contracting ones. Shares are used instead of raw counts because the arXiv dataset is heavily weighted toward 2025–2026.

## Files

- `../cluster_dynamics.py`: computes per-cluster yearly counts and shares, the annual change in share over the last N years (default 5), and an expanding / stable / contracting label. Writes `cluster_dynamics.json`.
- `index.html`: interactive view of that JSON. It has a growth-vs-maturity scatter, a UMAP map with a year scrubber, small multiples of yearly share, and a table.
- `cluster_dynamics.json`: the current output. It comes from a **12,000-paper random sample** (seed 42) of `arXiv_scientific_dataset.csv`, run through `ml_pipeline.py`'s SciBERT → UMAP → HDBSCAN steps. For the sample, HDBSCAN's `min_cluster_size` was lowered from 500 to 100, giving 22 clusters.

## Regenerate

```bash
python ml_pipeline.py                      # writes enriched_<csv>.csv with cluster_label / cluster_title
python cluster_dynamics.py enriched_arXiv_scientific_dataset_with_citations.csv --out visualization/cluster_dynamics.json
cd visualization && python -m http.server  # then open http://localhost:8000
```

The page loads the JSON with `fetch`, so it must be served over HTTP. Opening it with `file://` won't work.
