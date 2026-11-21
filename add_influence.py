"""
Compute the `influence_score` the exporters need, for either enriched CSV layout.

Influence blends two signals: log-scaled citation counts (Semantic Scholar) and PageRank over
the in-dataset citation graph. Same formula as ml_pipeline.calculate_influence_scores -- that
function can't be imported here because ml_pipeline.py pulls in torch/transformers at module
level, which this computation doesn't need.

By default it writes a small SIDECAR csv (s2_paper_id, citations_count, influence_score) that
export_points.py reads with --influence. That matters for the 3.2M-row dataset, where rewriting
the whole 4 GB CSV just to append one column costs minutes and gigabytes. Use --inline to get a
full copy of the input with the column appended instead.

Usage:
  # 287k "scientific" dataset (already carries citations_count)
  python add_influence.py --csv enriched_arXiv_scientific_dataset.csv

  # 3.2M "full" dataset: no citation counts of its own, so borrow them from the scraped CSV
  python add_influence.py --csv enriched_arXiv_scientific_full_dataset.csv \
      --citations enriched_arXiv_scientific_dataset.csv --out influence_full.csv
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import MinMaxScaler

import schema

CHUNK = 250_000


def read_ids(path):
    """s2_paper_id (and citations_count when the file has one) for every row, in file order."""
    header = pd.read_csv(path, nrows=0).columns
    cols = [c for c in ("s2_paper_id", "id", "citations_count") if c in header]
    frames = [c for c in pd.read_csv(path, usecols=cols, chunksize=CHUNK, dtype=str)]
    df = pd.concat(frames, ignore_index=True)
    if "s2_paper_id" not in df.columns:
        df["s2_paper_id"] = [schema.to_s2_id(v) for v in df["id"]]
    df = df.dropna(subset=["s2_paper_id"]).drop_duplicates("s2_paper_id").reset_index(drop=True)
    if "citations_count" in df.columns:
        df["citations_count"] = pd.to_numeric(df["citations_count"], errors="coerce").fillna(0)
    return df[["s2_paper_id"] + (["citations_count"] if "citations_count" in df.columns else [])]


def pagerank(edges, n, alpha=0.85, tol=1.0e-10, max_iter=200):
    """
    PageRank over a sparse citation graph, matching networkx.pagerank(alpha=0.85) semantics:
    uniform personalization, and dangling nodes (papers that cite nothing in the dataset)
    redistributing their mass uniformly.

    Done with scipy rather than networkx because building a 3.2M-node nx.DiGraph costs several
    GB before PageRank even starts; the sparse matrix is ~50 MB.
    """
    if not len(edges[0]):
        return np.zeros(n)
    rows, cols = edges
    m = sp.csr_matrix((np.ones(len(rows), dtype=np.float64), (rows, cols)), shape=(n, n))
    out_degree = np.asarray(m.sum(axis=1)).ravel()
    dangling = out_degree == 0
    inv = np.where(dangling, 0.0, 1.0 / np.maximum(out_degree, 1))
    # Row-normalize, then iterate on the transpose so each step is one sparse matvec.
    mt = (sp.diags(inv) @ m).T.tocsr()

    x = np.full(n, 1.0 / n)
    teleport = (1.0 - alpha) / n
    for _ in range(max_iter):
        prev = x
        x = alpha * (mt @ x + x[dangling].sum() / n) + teleport
        if np.abs(x - prev).sum() < n * tol:
            break
    return x


def influence_scores(citations, pageranks):
    """
    Combine citation volume and graph centrality into one score per paper.

    Citation counts are power-law distributed, so they are log-scaled before any linear
    normalization -- otherwise a handful of 100k-citation papers would flatten everything else to
    zero. The two signals are normalized SEPARATELY before being averaged because PageRank values
    are ~1e-6 and would vanish next to log-citations in a shared scale. The final rescale to
    [0.1, 1.0] (rather than [0, 1]) keeps the least-cited papers visible: the renderer sizes dots
    from this value, and a true zero would make them disappear.
    """
    log_citations = np.log1p(np.asarray(citations, dtype=float))
    pageranks = np.asarray(pageranks, dtype=float)
    # MinMaxScaler returns all-zeros for a constant column rather than dividing by zero, which is
    # the behaviour we want for an empty graph or a dataset with no citation counts at all.
    norm_citations = MinMaxScaler().fit_transform(log_citations.reshape(-1, 1)).flatten()
    norm_pageranks = MinMaxScaler().fit_transform(pageranks.reshape(-1, 1)).flatten()
    raw = 0.5 * norm_citations + 0.5 * norm_pageranks
    return MinMaxScaler(feature_range=(0.1, 1.0)).fit_transform(raw.reshape(-1, 1)).flatten()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="enriched_arXiv_scientific_dataset.csv")
    p.add_argument("--graph", default="citation_graph_adjacency.json")
    p.add_argument("--citations", help="CSV to borrow citations_count from, when --csv has none")
    p.add_argument("--out", default=None, help="default: influence_<csv stem>.csv")
    p.add_argument("--inline", action="store_true",
                   help="write a full copy of --csv with the column appended, not a sidecar")
    args = p.parse_args()
    stem = os.path.splitext(os.path.basename(args.csv))[0]
    out = args.out or (f"{stem}_with_influence.csv" if args.inline else f"influence_{stem}.csv")
    started = time.time()

    keys = read_ids(args.csv)
    n = len(keys)
    print(f"{n:,} unique papers ({time.time() - started:.0f}s)")

    # ---- Citation counts ----
    if "citations_count" not in keys.columns:
        if args.citations:
            src = read_ids(args.citations)
            if "citations_count" not in src.columns:
                raise SystemExit(f"{args.citations} has no citations_count column either.")
            counts = dict(zip(src["s2_paper_id"], src["citations_count"]))
            keys["citations_count"] = keys["s2_paper_id"].map(counts).fillna(0)
            known = int((keys["s2_paper_id"].isin(counts)).sum())
            print(f"Citation counts borrowed from {os.path.basename(args.citations)}: "
                  f"{known:,} papers ({100 * known / n:.1f}%); the rest get 0")
        else:
            keys["citations_count"] = 0
            print("No citations_count anywhere -- influence will be PageRank only.")

    # ---- PageRank over the citation graph ----
    index_of = {pid: i for i, pid in enumerate(keys["s2_paper_id"])}
    with open(args.graph, "r") as f:
        adj = json.load(f)
    rows, cols = [], []
    total = 0
    for source, targets in adj.items():
        s = index_of.get(source)
        total += len(targets)
        if s is None:
            continue
        for target in targets:
            t = index_of.get(target)
            if t is not None and t != s:
                rows.append(s)
                cols.append(t)
    print(f"Citation graph: {len(rows):,} of {total:,} references have both papers in this CSV "
          f"({100 * len(rows) / max(total, 1):.0f}%)")

    pr = pagerank((np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32)), n)
    linked = int((pr > pr.min()).sum())
    print(f"PageRank done ({time.time() - started:.0f}s); {linked:,} papers "
          f"({100 * linked / n:.0f}%) sit above the floor")

    scores = influence_scores(keys["citations_count"].to_numpy(), pr)
    keys["influence_score"] = scores
    print(f"influence_score: min {scores.min():.4f}, median {np.median(scores):.4f}, "
          f"max {scores.max():.4f}")

    # ---- Write ----
    if args.inline:
        first = True
        rows_out = 0
        for chunk in pd.read_csv(args.csv, dtype={"s2_paper_id": str}, chunksize=CHUNK,
                                 low_memory=False):
            if "s2_paper_id" not in chunk.columns:
                chunk["s2_paper_id"] = [schema.to_s2_id(v) for v in chunk["id"]]
            chunk["influence_score"] = chunk["s2_paper_id"].map(
                dict(zip(keys["s2_paper_id"], scores))).fillna(0.0)
            chunk.to_csv(out, mode="w" if first else "a", header=first, index=False)
            first = False
            rows_out += len(chunk)
            print(f"  ...wrote {rows_out:,} rows", end="\r", flush=True)
        print()
    else:
        keys.to_csv(out, index=False)

    print(f"Wrote {out} ({os.path.getsize(out) / 1048576:.0f} MB) in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
