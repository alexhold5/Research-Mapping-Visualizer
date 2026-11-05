"""
Longitudinal cluster dynamics.

Tracks each cluster's *relative* paper volume (its share of all papers published
that year) across publication years, then separates rapidly expanding subfields
from mature or contracting ones.

Shares are used instead of raw counts so that uneven collection across years
(the arXiv dataset is heavily weighted toward 2025-2026) does not make every
cluster look like it is growing.

Input: an enriched CSV from ml_pipeline.py with at least
    cluster_label, cluster_title, published_date (or updated_date)
and optionally x_2d / y_2d for the map view.

Usage:
    python cluster_dynamics.py enriched_arXiv_scientific_dataset_with_citations.csv \
        --out visualization/cluster_dynamics.json
"""
import argparse
import json

import numpy as np
import pandas as pd

# Annual growth in share above/below which a cluster is called expanding/contracting.
EXPANDING_RATE = 0.15
CONTRACTING_RATE = -0.10


def add_publication_year(df, date_columns=("published_date", "updated_date")):
    """Parse the first available date column into an integer `pub_year`."""
    for col in date_columns:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
            df = df.assign(pub_year=parsed.dt.year)
            return df.dropna(subset=["pub_year"]).astype({"pub_year": int})
    raise ValueError(f"None of {date_columns} found in the dataframe.")


def yearly_cluster_volume(df, min_papers_per_year=50):
    """
    Returns (counts, shares): cluster x year tables of paper counts and of each
    cluster's share of that year's papers. Years with fewer than
    `min_papers_per_year` papers in total are dropped as too noisy to compare.
    """
    counts = pd.crosstab(df["cluster_label"], df["pub_year"])
    totals = counts.sum(axis=0)
    keep = totals[totals >= min_papers_per_year].index
    counts = counts[keep]
    shares = counts / counts.sum(axis=0)
    return counts, shares


def share_growth_rate(share_series, window):
    """
    Annual growth rate of a cluster's share over the last `window` years,
    from a least-squares fit of log(share) against year: exp(slope) - 1.
    """
    recent = share_series.iloc[-window:]
    years = recent.index.to_numpy(dtype=float)
    # Floor zero shares at 0.01% so log() stays finite.
    log_share = np.log(np.maximum(recent.to_numpy(dtype=float), 1e-4))
    slope = np.polyfit(years, log_share, 1)[0] if len(years) >= 2 else 0.0
    return float(np.exp(slope) - 1)


def classify(growth_rate):
    if growth_rate >= EXPANDING_RATE:
        return "expanding"
    if growth_rate <= CONTRACTING_RATE:
        return "contracting"
    return "stable"


def compute_cluster_dynamics(df, window=5, min_papers_per_year=50):
    """
    Per-cluster summary of longitudinal volume.

    growth_rate   annual change in share over the last `window` years
    share_recent  mean share over the last `window` years
    share_prior   mean share over the `window` years before that
    median_year   median publication year
    mean_year     mean publication year (lower = more established; spreads
                  better than the median when recent years dominate the data)
    status        expanding / stable / contracting from growth_rate
    """
    df = add_publication_year(df)
    counts, shares = yearly_cluster_volume(df, min_papers_per_year)
    years = [int(y) for y in shares.columns]
    titles = df.groupby("cluster_label")["cluster_title"].first()
    corpus_median_year = float(df["pub_year"].median())

    clusters = []
    for label in shares.index:
        s = shares.loc[label]
        rate = share_growth_rate(s, window)
        members = df[df["cluster_label"] == label]
        top_categories = []
        if "category_code" in members.columns:
            cat_share = members["category_code"].value_counts(normalize=True).head(3)
            top_categories = [[str(k), round(float(v), 3)] for k, v in cat_share.items()]
        clusters.append({
            "id": int(label),
            "title": str(titles.get(label, f"Cluster {label}")),
            "size": int(len(members)),
            "counts": [int(c) for c in counts.loc[label]],
            "shares": [float(v) for v in s],
            "growth_rate": rate,
            "share_recent": float(s.iloc[-window:].mean()),
            "share_prior": float(s.iloc[-2 * window:-window].mean()) if len(s) > window else None,
            "median_year": float(members["pub_year"].median()),
            "mean_year": round(float(members["pub_year"].mean()), 2),
            "top_categories": top_categories,
            "status": "noise" if label == -1 else classify(rate),
        })

    clusters.sort(key=lambda c: c["growth_rate"], reverse=True)
    return {
        "years": years,
        "year_totals": [int(t) for t in counts.sum(axis=0)],
        "corpus_median_year": corpus_median_year,
        "params": {
            "window": window,
            "min_papers_per_year": min_papers_per_year,
            "expanding_rate": EXPANDING_RATE,
            "contracting_rate": CONTRACTING_RATE,
            "n_papers": int(len(df)),
        },
        "clusters": clusters,
    }


def map_points(df, max_points=15000, seed=42):
    """Compact [x, y, cluster, year] rows for the map view."""
    df = add_publication_year(df)
    if not {"x_2d", "y_2d"}.issubset(df.columns):
        return []
    if len(df) > max_points:
        df = df.sample(n=max_points, random_state=seed)
    return [
        [round(float(x), 3), round(float(y), 3), int(c), int(yr)]
        for x, y, c, yr in df[["x_2d", "y_2d", "cluster_label", "pub_year"]].itertuples(index=False)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", help="Enriched CSV with cluster_label / cluster_title / published_date")
    parser.add_argument("--out", default="visualization/cluster_dynamics.json")
    parser.add_argument("--window", type=int, default=5, help="Years used for the growth fit")
    parser.add_argument("--min-papers-per-year", type=int, default=50)
    parser.add_argument("--max-points", type=int, default=15000, help="Points kept for the map view")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    result = compute_cluster_dynamics(df, args.window, args.min_papers_per_year)
    result["points"] = map_points(df, args.max_points)
    with open(args.out, "w") as f:
        json.dump(result, f, separators=(",", ":"))

    print(f"Wrote {args.out}: {len(result['clusters'])} clusters, years {result['years'][0]}-{result['years'][-1]}")
    for c in result["clusters"]:
        print(f"  {c['status']:<11} {c['growth_rate']:+7.1%}/yr  share {c['share_recent']:6.1%}  {c['title']}")


if __name__ == "__main__":
    main()
