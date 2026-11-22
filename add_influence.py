"""
Compute influence_score for every paper and save it as a small sidecar CSV (id, influence_score), instead of
rewriting the 5 GB dataset CSV. export_maps.py and export_points.py pick the sidecar up automatically.

Same formula as ML_scripts/3.23M_dir/ml_pipeline.calculate_influence_scores: 50% log-scaled citations_count and
50% PageRank (alpha 0.85) over the citation graph, each min-max scaled, then rescaled to [0.1, 1.0].
Inputs are the CSV's own columns: id, s2_paper_id, citations_count and referenced_paper_ids (the same 35M edges as
citation_graph_adjacency_full.json, so that file is not needed). PageRank is done with scipy on a sparse matrix:
networkx would need several GB for 3.2M nodes.

Usage:
  python add_influence.py
  python add_influence.py --csv path/to/enriched_arXiv_scientific_full_dataset.csv --out influence_full.csv
"""
import argparse
import os
import re
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

CSV = os.path.join("web", "public", "data", "enriched_arXiv_scientific_full_dataset.csv")
OUT = os.path.join("web", "public", "data", "influence_full.csv")
CHUNK = 400_000
REF = re.compile(r'"(ARXIV:[^"]+)"')


def minmax(v):
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)


def pagerank(rows, cols, n, alpha=0.85, tol=1e-10, max_iter=200):
    """PageRank with networkx semantics: uniform teleport, papers citing nothing spread their rank evenly."""
    m = sp.csr_matrix((np.ones(len(rows), dtype=np.float64), (rows, cols)), shape=(n, n))
    out = np.asarray(m.sum(axis=1)).ravel()
    dangling = out == 0
    mt = (sp.diags(np.where(dangling, 0.0, 1.0 / np.maximum(out, 1))) @ m).T.tocsr()
    x = np.full(n, 1.0 / n)
    for i in range(max_iter):
        prev = x
        x = alpha * (mt @ x + x[dangling].sum() / n) + (1 - alpha) / n
        if np.abs(x - prev).sum() < n * tol:
            break
    print(f"  PageRank converged after {i + 1} iterations")
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=CSV)
    p.add_argument("--out", default=OUT)
    args = p.parse_args()
    started = time.time()

    ids, s2, cites, refs = [], [], [], []
    for chunk in pd.read_csv(args.csv, usecols=["id", "s2_paper_id", "citations_count", "referenced_paper_ids"],
                             dtype={"id": str, "s2_paper_id": str}, chunksize=CHUNK):
        ids.append(chunk["id"])
        s2.append(chunk["s2_paper_id"])
        cites.append(chunk["citations_count"].fillna(0).to_numpy(dtype=np.float64))
        refs.append(chunk["referenced_paper_ids"].fillna("[]").to_numpy())
        print(f"  read {sum(len(c) for c in cites):,} rows", end="\r")
    ids = pd.concat(ids, ignore_index=True)
    s2 = pd.concat(s2, ignore_index=True)
    cites = np.concatenate(cites)
    n = len(ids)
    print(f"Read {n:,} rows in {time.time() - started:.0f}s")

    row_of = {pid: i for i, pid in enumerate(s2)}
    src, dst = [], []
    for i, text in enumerate(np.concatenate(refs)):
        if len(text) > 2:
            for ref in REF.findall(text):
                j = row_of.get(ref)
                if j is not None and j != i:
                    src.append(i)
                    dst.append(j)
    del row_of
    print(f"{len(src):,} in-dataset citation edges ({time.time() - started:.0f}s)")

    pr = pagerank(np.array(src, dtype=np.int32), np.array(dst, dtype=np.int32), n)
    del src, dst
    raw = 0.5 * minmax(np.log1p(cites)) + 0.5 * minmax(pr)
    score = 0.1 + 0.9 * minmax(raw)

    pd.DataFrame({"id": ids, "influence_score": score.astype(np.float32)}).to_csv(args.out, index=False)
    print(f"influence_score: min {score.min():.3f}, median {np.median(score):.3f}, max {score.max():.3f}")
    print(f"Wrote {args.out} ({os.path.getsize(args.out) / 1e6:.0f} MB) in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
