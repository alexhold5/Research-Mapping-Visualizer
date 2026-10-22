"""
Convert the ML pipeline's enriched CSV into JSON files the web frontend loads.

Outputs (in web/public/data/):
  graph.json   {nodes: [...], links: [...], clusters: [...], meta: {...}}
               The {nodes, links} shape is what 3d-force-graph expects.
  papers.json  {id: {abstract, authors, arxiv_id}}  - loaded lazily for the detail modal

The website only gets a capped set of papers (MAX_PAPERS, 1000 by default) so it never gets
overloaded, however big the ML run was. See select_papers() for how they are chosen.

Usage:
  python export_web_data.py
  python export_web_data.py --max-papers 2000          # show more (0 = no cap)
  python export_web_data.py --csv some_other_enriched.csv --graph citation_graph_adjacency.json
"""
import argparse
import ast
import json
import os

import numpy as np
import pandas as pd

OUT_DIR = os.path.join("web", "public", "data")
MAX_PAPERS = 1000   # most papers sent to the website (0 = no cap)
TOP_SHARE = 0.25    # fraction of those slots reserved for the highest-influence papers overall


def select_papers(df, max_papers=MAX_PAPERS, top_share=TOP_SHARE, seed=42):
    """
    Pick the papers worth showing when there are more than the website can handle.

      1. Landmarks: the top `top_share` of the slots go to the highest-influence papers overall.
      2. Coverage: the rest are split across clusters (and the unclustered group) in proportion to
         their size, taking the highest-influence papers in each. Every real cluster gets at least
         one slot, so the map keeps its shape instead of becoming only old, well-cited papers.
      Most papers share the minimum influence score, so ties are broken with a seeded random
      nudge: the same input always gives the same selection.
    Returns the chosen rows in their original order.
    """
    if not max_papers or len(df) <= max_papers:
        return df

    rng = np.random.default_rng(seed)
    prio = pd.Series(df["influence_score"].fillna(0).to_numpy() + rng.random(len(df)) * 1e-6, index=df.index)

    landmarks = prio.nlargest(int(max_papers * top_share)).index
    groups = df.loc[df.index.difference(landmarks), "cluster_label"]
    sizes = groups.value_counts()
    remaining = max_papers - len(landmarks)

    # Slots per cluster: proportional to size (floored), at least 1 for each real cluster...
    exact = sizes / sizes.sum() * remaining
    quota = exact.astype(int)
    quota[(quota == 0) & (quota.index != -1)] = 1
    # ...then hand out or take back slots until the total is exactly `remaining`.
    leftover = remaining - int(quota.sum())
    by_fraction = (exact - exact.astype(int)).sort_values(ascending=False).index
    while leftover > 0:
        gave = False
        for label in by_fraction:
            if leftover == 0:
                break
            if quota[label] < sizes[label]:
                quota[label] += 1
                leftover -= 1
                gave = True
        if not gave:
            break
    while leftover < 0:
        quota[quota.idxmax()] -= 1
        leftover += 1

    picked = set(landmarks)
    for label, slots in quota.items():
        members = groups.index[groups == label]
        picked.update(prio[members].nlargest(int(slots)).index)
    return df[df.index.isin(picked)]


def to_decimal_year(dates):
    """2021-07-01 -> 2021.5 (roughly). Used for the z-axis (timeline)."""
    start = pd.to_datetime(dates.dt.year.astype("Int64").astype(str) + "-01-01", errors="coerce")
    return dates.dt.year + (dates - start).dt.days / 365.25


def parse_authors(value):
    """The CSV stores authors as the text of a Python list, e.g. "['A B', 'C D']". Return a real list."""
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)):
                return [str(a).strip() for a in parsed]
        except (ValueError, SyntaxError):
            pass
        return [value.strip()] if value.strip() else []
    return []


def normalize_xy(x, y):
    """Center UMAP coords and scale to fit [-1, 1], keeping the aspect ratio."""
    cx, cy = (x.max() + x.min()) / 2, (y.max() + y.min()) / 2
    half = max(x.max() - x.min(), y.max() - y.min()) / 2 or 1.0
    return (x - cx) / half, (y - cy) / half


def load_links(graph_path, ids):
    """Citation edges where both papers are in the dataset (same rule as ml_pipeline.load_graph_from_json)."""
    if not os.path.exists(graph_path):
        print(f"No citation graph at {graph_path}; exporting without links.")
        return []
    with open(graph_path, "r") as f:
        adj = json.load(f)
    links = []
    for source, targets in adj.items():
        if source in ids:
            for target in targets:
                if target in ids and target != source:
                    links.append({"source": source, "target": target})
    return links


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="enriched_arXiv_scientific_dataset_with_citations.csv")
    parser.add_argument("--graph", default="citation_graph_adjacency.json")
    parser.add_argument("--max-papers", type=int, default=MAX_PAPERS,
                        help="cap on papers sent to the website (0 = no cap)")
    parser.add_argument("--top-share", type=float, default=TOP_SHARE,
                        help="fraction of the cap reserved for the highest-influence papers")
    parser.add_argument("--out", default=OUT_DIR, help="output folder")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df.dropna(subset=["s2_paper_id"]).drop_duplicates("s2_paper_id")
    total_papers = len(df)
    print(f"Loaded {total_papers} papers from {args.csv}")

    df = select_papers(df, args.max_papers, args.top_share)
    if len(df) < total_papers:
        n_clusters = df.loc[df["cluster_label"] != -1, "cluster_label"].nunique()
        print(f"Selected {len(df)} of {total_papers} papers for the website "
              f"({n_clusters} clusters represented, cap {args.max_papers}).")

    # Pipeline used to name these umap_x/umap_y; newer runs use x_2d/y_2d.
    xcol = "x_2d" if "x_2d" in df.columns else "umap_x"
    ycol = "y_2d" if "y_2d" in df.columns else "umap_y"
    nx_, ny_ = normalize_xy(df[xcol], df[ycol])

    # Proposal: z-axis = publication date.
    pub = pd.to_datetime(df["published_date"], errors="coerce", format="mixed")
    year = to_decimal_year(pub)

    nodes = []
    for i, row in enumerate(df.itertuples(index=False)):
        nodes.append({
            "id": row.s2_paper_id,
            "title": " ".join(str(row.title).split()),
            "x": round(float(nx_.iloc[i]), 5),
            "y": round(float(ny_.iloc[i]), 5),
            "year": None if pd.isna(year.iloc[i]) else round(float(year.iloc[i]), 3),
            "date": None if pd.isna(pub.iloc[i]) else pub.iloc[i].strftime("%Y-%m-%d"),
            "cluster": int(row.cluster_label),
            "influence": round(float(row.influence_score), 4),
            "citations": int(0 if pd.isna(row.citations_count) else row.citations_count),
            "category": row.category,
            "categoryCode": row.category_code,
        })

    # One entry per cluster, with a centroid so we can float labels in 3D later.
    clusters = []
    for label, g in df.assign(_x=nx_.values, _y=ny_.values, _year=year.values).groupby("cluster_label"):
        clusters.append({
            "id": int(label),
            "title": g["cluster_title"].iloc[0],
            "size": int(len(g)),
            "centroid": {"x": round(float(g["_x"].mean()), 4),
                         "y": round(float(g["_y"].mean()), 4),
                         "year": round(float(np.nanmean(g["_year"])), 3)},
        })

    ids = {n["id"] for n in nodes}
    links = load_links(args.graph, ids)

    years = [n["year"] for n in nodes if n["year"] is not None]
    graph = {
        "meta": {"paperCount": len(nodes), "totalPapers": total_papers, "linkCount": len(links),
                 "yearMin": min(years), "yearMax": max(years), "source": os.path.basename(args.csv)},
        "clusters": clusters,
        "nodes": nodes,
        "links": links,
    }

    papers = {
        row.s2_paper_id: {
            "abstract": " ".join(str(row.summary).split()),
            "authors": parse_authors(row.authors),
            "arxivId": str(row.s2_paper_id).replace("ARXIV:", ""),
        }
        for row in df.itertuples(index=False)
    }

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "graph.json"), "w", encoding="utf-8") as f:
        json.dump(graph, f, separators=(",", ":"))
    with open(os.path.join(args.out, "papers.json"), "w", encoding="utf-8") as f:
        json.dump(papers, f, separators=(",", ":"))

    print(f"Wrote {len(nodes)} nodes, {len(links)} links, {len(clusters)} clusters "
          f"(years {graph['meta']['yearMin']:.1f}-{graph['meta']['yearMax']:.1f}) to {args.out}/")


if __name__ == "__main__":
    main()
