"""
Convert the ML pipeline's enriched CSV into JSON files the web frontend loads.

Outputs (in web/public/data/):
  graph.json   {nodes: [...], links: [...], clusters: [...], meta: {...}}
               The {nodes, links} shape is what 3d-force-graph expects.
  papers.json  {id: {abstract, authors, arxiv_id}}  - loaded lazily for the detail modal

Usage:
  python export_web_data.py
  python export_web_data.py --csv some_other_enriched.csv --graph citation_graph_adjacency.json
"""
import argparse
import ast
import json
import os

import numpy as np
import pandas as pd

OUT_DIR = os.path.join("web", "public", "data")


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
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df.dropna(subset=["s2_paper_id"]).drop_duplicates("s2_paper_id")
    print(f"Loaded {len(df)} papers from {args.csv}")

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
        "meta": {"paperCount": len(nodes), "linkCount": len(links),
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

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "graph.json"), "w", encoding="utf-8") as f:
        json.dump(graph, f, separators=(",", ":"))
    with open(os.path.join(OUT_DIR, "papers.json"), "w", encoding="utf-8") as f:
        json.dump(papers, f, separators=(",", ":"))

    print(f"Wrote {len(nodes)} nodes, {len(links)} links, {len(clusters)} clusters "
          f"(years {graph['meta']['yearMin']:.1f}-{graph['meta']['yearMax']:.1f}) to {OUT_DIR}/")


if __name__ == "__main__":
    main()
