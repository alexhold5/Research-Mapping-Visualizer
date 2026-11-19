"""
Export EVERY paper (noise included) for the point-cloud prototype (web/?view=points).

Unlike export_web_data.py, which only ships each cluster's top papers, this writes one compact
binary file with a dot per paper, and puts everything else in small shards the website fetches
on demand. Each paper gets an index `i` (its position in points.bin); every file is keyed by `i`,
and the shard a paper lives in is just `i % SHARDS`, so the website needs no lookup table.

Outputs (in web/public/data/v2/):
  meta.json                 paper count, column layout of points.bin, clusters, fields, shard counts
  points.bin                little-endian columns, N values each, in this order:
                              x Float32, y Float32, influence Float32, cluster Int16 (-1 = noise),
                              year Uint16 (0 = unknown), field Uint8 (index into meta.fields)
  edges/<i % 256>.json      {i: {"r": [refs], "c": [cited by], "o": refs outside the dataset}}
                            (only papers with at least one link are listed)
  titles/<i % 2048>.json    {i: title}
  details/<i % 4096>.json   {i: {id, title, authors, date, citations, category, abstract}}

Only citations with both papers in the CSV become lines, since only those have a dot at each end.
References to papers outside the dataset are counted per paper ("o") so the website can mention them.

Usage:
  python export_points.py
  python export_points.py --csv enriched_arXiv_scientific_dataset.csv --graph citation_graph_adjacency.json
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd

from export_web_data import normalize_xy, parse_authors, parse_dates, to_decimal_year

OUT_DIR = os.path.join("web", "public", "data", "v2")
EDGE_SHARDS = 256
TITLE_SHARDS = 2048   # titles alone (~12 KB per file), for hover tooltips and neighbor lists (many papers per click)
DETAIL_SHARDS = 4096  # full records with abstracts, fetched for the one selected paper
FIELD_COUNT = 12  # arXiv fields that get their own color; the rest share "other"

WANTED = {
    "s2_paper_id", "title", "category", "category_code", "published_date", "authors", "summary",
    "citations_count", "influence_score", "x_2d", "y_2d", "umap_x", "umap_y", "cluster_label", "cluster_title",
}

# Column order in points.bin. Wider types come first so every column starts aligned for its typed array.
COLUMNS = [("x", "<f4"), ("y", "<f4"), ("influence", "<f4"), ("cluster", "<i2"), ("year", "<u2"), ("field", "u1")]


def clean(text):
    return " ".join(str(text).split()) if isinstance(text, str) else ""


def clear_dir(path):
    """Make `path` exist and remove the JSON files an earlier run left in it."""
    os.makedirs(path, exist_ok=True)
    for stale in glob.glob(os.path.join(path, "*.json")):
        os.remove(stale)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"), ensure_ascii=False)
    return os.path.getsize(path)


def build_edges(graph_path, index_of, n):
    """refs[i] / cited_by[i]: sets of paper indices; outside[i]: references to papers not in the CSV."""
    refs = [set() for _ in range(n)]
    cited_by = [set() for _ in range(n)]
    outside = np.zeros(n, dtype=np.int32)
    total = unknown_source = self_links = 0
    with open(graph_path, "r") as f:
        adj = json.load(f)
    for source, targets in adj.items():
        total += len(targets)
        s = index_of.get(source)
        if s is None:
            unknown_source += len(targets)
            continue
        for target in targets:
            t = index_of.get(target)
            if t is None:
                outside[s] += 1
            elif t == s:
                self_links += 1
            else:
                refs[s].add(t)
                cited_by[t].add(s)
    kept = sum(len(r) for r in refs)
    print(f"Citation graph: {total:,} references from {len(adj):,} papers.")
    print(f"  kept {kept:,} links with both papers in the CSV ({100 * kept / max(total, 1):.0f}%)")
    print(f"  dropped {int(outside.sum()):,} to papers outside the CSV, {unknown_source:,} from papers outside it, "
          f"{self_links:,} self-citations, {total - kept - int(outside.sum()) - unknown_source - self_links:,} duplicates")
    return refs, cited_by, outside


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="enriched_arXiv_scientific_dataset.csv")
    parser.add_argument("--graph", default="citation_graph_adjacency.json")
    parser.add_argument("--out", default=OUT_DIR, help="output folder")
    args = parser.parse_args()
    started = time.time()

    df = pd.read_csv(args.csv, usecols=lambda c: c in WANTED)
    rows = len(df)
    df = df.dropna(subset=["s2_paper_id"]).drop_duplicates("s2_paper_id").reset_index(drop=True)
    n = len(df)
    print(f"Loaded {rows:,} rows from {args.csv}; {n:,} unique papers ({time.time() - started:.0f}s)")

    xcol = "x_2d" if "x_2d" in df.columns else "umap_x"
    ycol = "y_2d" if "y_2d" in df.columns else "umap_y"
    if df[xcol].isna().any() or df[ycol].isna().any():
        raise SystemExit(f"{int(df[xcol].isna().sum())} papers have no position; every paper needs one for a dot.")
    df["cluster_label"] = df["cluster_label"].fillna(-1).astype(int)
    df["citations_count"] = df["citations_count"].fillna(0).astype(int)
    df["influence_score"] = df["influence_score"].fillna(0.0)
    pub = parse_dates(df["published_date"])
    year = to_decimal_year(pub)
    date = pub.dt.strftime("%Y-%m-%d")

    # ---- Fields: the biggest get their own index; the rest share the last one ("other") ----
    field_counts = df["category_code"].value_counts()
    cat_name = df.drop_duplicates("category_code").set_index("category_code")["category"].to_dict()
    top_fields = list(field_counts.head(FIELD_COUNT).index)
    field_index = {c: i for i, c in enumerate(top_fields)}
    other = len(top_fields)
    fields = [{"code": c, "name": cat_name.get(c, c), "papers": int(field_counts[c])} for c in top_fields]
    fields.append({"code": "other", "name": "Other", "papers": int(n - field_counts.head(FIELD_COUNT).sum())})

    # ---- points.bin ----
    x, y = normalize_xy(df[xcol].astype(float), df[ycol].astype(float))
    values = {
        "x": x.to_numpy(),
        "y": y.to_numpy(),
        "influence": df["influence_score"].to_numpy(),
        "cluster": df["cluster_label"].to_numpy(),
        "year": np.floor(year).fillna(0).astype(int).to_numpy(),
        "field": df["category_code"].map(field_index).fillna(other).astype(int).to_numpy(),
    }
    os.makedirs(args.out, exist_ok=True)
    layout, offset = [], 0
    with open(os.path.join(args.out, "points.bin"), "wb") as f:
        for name, dtype in COLUMNS:
            data = np.ascontiguousarray(values[name], dtype=dtype)
            f.write(data.tobytes())
            layout.append({"name": name, "type": np.dtype(dtype).name, "offset": offset})
            offset += data.nbytes
    points_bytes = offset

    # ---- Clusters (noise is reported separately) ----
    clusters = []
    clustered = df["cluster_label"] != -1
    for cid, g in df[clustered].groupby("cluster_label"):
        clusters.append({
            "id": int(cid),
            "title": clean(g["cluster_title"].iloc[0]),
            "size": int(len(g)),
            "x": round(float(x[g.index].mean()), 5),
            "y": round(float(y[g.index].mean()), 5),
            "field": int(pd.Series(values["field"][g.index]).mode().iloc[0]),
        })
    clusters.sort(key=lambda c: -c["size"])
    noise = int((~clustered).sum())
    print(f"{len(clusters)} clusters, {n - noise:,} clustered papers, {noise:,} noise ({100 * noise / n:.0f}%)")

    # ---- Edge shards ----
    index_of = {pid: i for i, pid in enumerate(df["s2_paper_id"])}
    refs, cited_by, outside = build_edges(args.graph, index_of, n)
    edge_dir = os.path.join(args.out, "edges")
    clear_dir(edge_dir)
    edge_bytes, linked = 0, 0
    for s in range(EDGE_SHARDS):
        shard = {}
        for i in range(s, n, EDGE_SHARDS):
            if refs[i] or cited_by[i] or outside[i]:
                entry = {"r": sorted(refs[i]), "c": sorted(cited_by[i])}
                if outside[i]:
                    entry["o"] = int(outside[i])
                shard[i] = entry
                linked += bool(refs[i] or cited_by[i])
        edge_bytes += write_json(os.path.join(edge_dir, f"{s}.json"), shard)
    degree = np.array([len(r) + len(c) for r, c in zip(refs, cited_by)])
    print(f"{linked:,} papers ({100 * linked / n:.0f}%) have at least one drawable link; "
          f"median {int(np.median(degree[degree > 0]))}, max {int(degree.max()):,} links per linked paper")
    del refs, cited_by

    # ---- Title shards ----
    title_dir = os.path.join(args.out, "titles")
    clear_dir(title_dir)
    titles = [clean(t) for t in df["title"]]
    title_bytes = sum(
        write_json(os.path.join(title_dir, f"{s}.json"), {i: titles[i] for i in range(s, n, TITLE_SHARDS)})
        for s in range(TITLE_SHARDS)
    )

    # ---- Detail shards ----
    detail_dir = os.path.join(args.out, "details")
    clear_dir(detail_dir)
    records = [None] * n
    for i, r in enumerate(df[["s2_paper_id", "title", "authors", "citations_count", "category", "summary"]]
                          .itertuples(index=False, name=None)):
        pid, title, authors, citations, category, summary = r
        records[i] = {
            "id": pid,
            "title": titles[i],
            "authors": parse_authors(authors),
            "date": date.iat[i] if isinstance(date.iat[i], str) else None,
            "citations": int(citations),
            "category": category if isinstance(category, str) else "",
            "abstract": clean(summary),
        }
    detail_bytes = sum(
        write_json(os.path.join(detail_dir, f"{s}.json"), {i: records[i] for i in range(s, n, DETAIL_SHARDS)})
        for s in range(DETAIL_SHARDS)
    )

    meta = {
        "count": n,
        "columns": layout,
        "edgeShards": EDGE_SHARDS,
        "titleShards": TITLE_SHARDS,
        "detailShards": DETAIL_SHARDS,
        "noisePapers": noise,
        "yearMin": int(values["year"][values["year"] > 0].min()),
        "yearMax": int(values["year"].max()),
        "fields": fields,
        "clusters": clusters,
        "source": os.path.basename(args.csv),
    }
    write_json(os.path.join(args.out, "meta.json"), meta)

    mb = 1048576
    print(f"Wrote to {args.out}/ in {time.time() - started:.0f}s:")
    print(f"  points.bin   {points_bytes / mb:.1f} MB ({points_bytes / n:.0f} bytes per paper)")
    print(f"  edges/       {edge_bytes / mb:.1f} MB in {EDGE_SHARDS} files (~{edge_bytes / EDGE_SHARDS / 1024:.0f} KB each)")
    print(f"  titles/      {title_bytes / mb:.1f} MB in {TITLE_SHARDS} files (~{title_bytes / TITLE_SHARDS / 1024:.0f} KB each)")
    print(f"  details/    {detail_bytes / mb:.1f} MB in {DETAIL_SHARDS} files (~{detail_bytes / DETAIL_SHARDS / 1024:.0f} KB each)")


if __name__ == "__main__":
    main()
