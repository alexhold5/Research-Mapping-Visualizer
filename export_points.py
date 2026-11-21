"""
Export EVERY paper (noise included) for the point-cloud prototype (web/?view=points).

This writes one compact binary file with a dot per paper and puts everything else in small
shards the website fetches on demand. Each paper gets an index `i` (its position in points.bin); every file is keyed by `i`,
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
                            Abstracts are trimmed to --abstract-chars (full text is 340 MB of
                            the 420 MB this folder would otherwise take); trimmed records carry
                            "truncated": true and the website links out to arXiv for the rest.

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

import schema
from schema import normalize_xy, parse_dates, to_decimal_year

OUT_DIR = os.path.join("web", "public", "data", "v2")
EDGE_SHARDS = 256
TITLE_SHARDS = 2048   # titles alone (~12 KB per file), for hover tooltips and neighbor lists (many papers per click)
DETAIL_SHARDS = 4096  # full records with abstracts, fetched for the one selected paper
FIELD_COUNT = 19  # arXiv fields that get their own color; the rest share "other"
FIELD_COVERAGE = 0.70  # if the top FIELD_COUNT codes cover less than this, group by archive
                       # instead: across all of arXiv, 19 codes reach only ~45% of papers,
                       # while 19 archives (cs, math, cond-mat, ...) reach ~99%.
ABSTRACT_CHARS = 400  # keep a readable teaser; "View on arXiv" in the panel has the rest
                      # (0 = full text, -1 = no abstracts at all -- see --abstract-chars)

WANTED = {
    "s2_paper_id", "title", "category", "category_code", "published_date", "authors", "summary",
    "citations_count", "influence_score", "x_2d", "y_2d", "umap_x", "umap_y", "cluster_label", "cluster_title",
}

# Column order in points.bin. Wider types come first so every column starts aligned for its typed array.
COLUMNS = [("x", "<f4"), ("y", "<f4"), ("influence", "<f4"), ("cluster", "<i2"), ("year", "<u2"), ("field", "u1")]


def clean(text):
    return " ".join(str(text).split()) if isinstance(text, str) else ""


def trim(text, limit):
    """Cut `text` to at most `limit` characters on a word boundary. Returns (text, was_trimmed)."""
    if not limit or len(text) <= limit:
        return text, False
    cut = text[:limit]
    space = cut.rfind(" ")
    if space > limit * 0.6:  # don't strand a long unbroken token
        cut = cut[:space]
    return cut.rstrip(" .,;:") + "\u2026", True


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
    parser.add_argument("--abstract-chars", type=int, default=ABSTRACT_CHARS,
                        help="trim abstracts to this many characters "
                             "(0 keeps the full text, -1 drops abstracts entirely)")
    parser.add_argument("--influence", help="sidecar CSV from add_influence.py "
                                            "(s2_paper_id, influence_score) to merge in")
    args = parser.parse_args()
    started = time.time()

    # Abstracts dominate the payload AND the memory cost of loading, so when they are switched
    # off don't even read the column -- that is 3.1 GB of text on the 3.2M-row dataset.
    wanted = set(WANTED)
    if args.abstract_chars < 0:
        wanted.discard("summary")
    available = set(pd.read_csv(args.csv, nrows=0).columns)
    needed = schema.source_columns(wanted & (wanted | available)) & available

    df = pd.read_csv(args.csv, usecols=sorted(needed), low_memory=False)
    rows = len(df)
    schema.normalize(df)
    df = df.dropna(subset=["s2_paper_id"]).drop_duplicates("s2_paper_id").reset_index(drop=True)
    n = len(df)
    print(f"Loaded {rows:,} rows from {args.csv}; {n:,} unique papers ({time.time() - started:.0f}s)")

    if args.influence:
        side = pd.read_csv(args.influence, dtype={"s2_paper_id": str})
        df["influence_score"] = df["s2_paper_id"].map(
            dict(zip(side["s2_paper_id"], side["influence_score"])))
        missing = int(df["influence_score"].isna().sum())
        note = ""
        # The sidecar also carries the citation counts it scored from, which the full dataset's
        # CSV has no column for -- take those too rather than leaving every paper on 0.
        if "citations_count" in side.columns and not df["citations_count"].any():
            counts = dict(zip(side["s2_paper_id"], side["citations_count"]))
            df["citations_count"] = df["s2_paper_id"].map(counts).fillna(0)
            known = int((df["citations_count"] > 0).sum())
            note = f"; citation counts for {known:,} papers ({100 * known / len(df):.0f}%)"
        print(f"Merged influence from {os.path.basename(args.influence)}{note}"
              + (f"; {missing:,} papers not listed there get 0" if missing else ""))
    if df["influence_score"].isna().all():
        raise SystemExit("No influence_score in the CSV and none supplied with --influence. "
                         "Run add_influence.py first -- it sets dot size and orders the panels.")

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
    codes = df["category_code"]
    field_counts = codes.value_counts()
    coverage = field_counts.head(FIELD_COUNT).sum() / max(len(df), 1)
    if coverage < FIELD_COVERAGE:
        archives = codes.map(schema.archive_of)
        arch_counts = archives.value_counts()
        arch_coverage = arch_counts.head(FIELD_COUNT).sum() / max(len(df), 1)
        print(f"Top {FIELD_COUNT} categories cover only {100 * coverage:.0f}% of papers; "
              f"colouring by arXiv archive instead ({100 * arch_coverage:.0f}% covered, "
              f"{len(arch_counts)} archives)")
        codes = archives
        field_counts = arch_counts
        cat_name = {a: schema.ARCHIVE_NAMES.get(a, a) for a in field_counts.index}
    else:
        cat_name = df.drop_duplicates("category_code").set_index("category_code")["category"].to_dict()
    top_fields = list(field_counts.head(FIELD_COUNT).index)
    field_index = {c: i for i, c in enumerate(top_fields)}
    other = len(top_fields)
    fields = [{"code": c, "name": cat_name.get(c, c), "papers": int(field_counts[c])} for c in top_fields]
    fields.append({"code": "other", "name": "Other fields", "papers": int(n - field_counts.head(FIELD_COUNT).sum())})

    # ---- points.bin ----
    x, y = normalize_xy(df[xcol].astype(float), df[ycol].astype(float))
    values = {
        "x": x.to_numpy(),
        "y": y.to_numpy(),
        "influence": df["influence_score"].to_numpy(),
        "cluster": df["cluster_label"].to_numpy(),
        "year": np.floor(year).fillna(0).astype(int).to_numpy(),
        "field": codes.map(field_index).fillna(other).astype(int).to_numpy(),
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
    # Built one shard at a time: materializing all N records at once costs over a gigabyte on the
    # 3.2M-row dataset, and they are written and thrown away immediately anyway.
    ids = df["s2_paper_id"].tolist()
    authors_raw = df["authors"].tolist()
    citations = df["citations_count"].tolist()
    categories = df["category"].tolist()
    dates = date.tolist()
    summaries = df["summary"].tolist() if args.abstract_chars >= 0 else None

    def record(i):
        rec = {
            "id": ids[i],
            "title": titles[i],
            "authors": schema.parse_author_field(authors_raw[i]),
            "date": dates[i] if isinstance(dates[i], str) else None,
            "citations": int(citations[i]),
            "category": categories[i] if isinstance(categories[i], str) else "",
        }
        if summaries is not None:
            rec["abstract"], truncated = trim(clean(summaries[i]), args.abstract_chars)
            if truncated:
                rec["truncated"] = True
        return rec

    detail_bytes = sum(
        write_json(os.path.join(detail_dir, f"{s}.json"), {i: record(i) for i in range(s, n, DETAIL_SHARDS)})
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
        "hasAbstracts": args.abstract_chars >= 0,
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
