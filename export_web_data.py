"""
Convert the ML pipeline's enriched CSV into JSON files the web frontend loads.

The website has two levels:
  1. Overview: one bubble per cluster ("chunk"), sized by how many papers it holds, with the
     strongest citation links between clusters.
  2. Cluster view: click a bubble to see that cluster's most influential papers. Only a window of them
     is drawn at a time (the arrow keys slide it), together with papers from other clusters that are
     cited by / cite the papers you select (so you can see HOW clusters connect).

Outputs (in web/public/data/):
  overview.json          clusters (position, size, per-year and per-field counts, ...) + cluster-to-cluster links
  clusters/<id>.json     the top PER_CLUSTER papers of one cluster, in influence order (light records: no
                         abstract or authors), the papers of OTHER clusters that have a citation link to them
                         ("external"), and the citation links. A citation is kept only when both papers are
                         among their clusters' top PER_CLUSTER, and a link between two clusters is listed in
                         both clusters' files. Loaded only when that cluster is opened.
  details/<id>/<k>.json  authors and abstracts of that cluster's papers ranked k*DETAIL_BLOCK ..
                         (k+1)*DETAIL_BLOCK - 1, keyed by paper id. Loaded when a paper there is selected.

Papers the clustering left unclustered (noise) are not shown; the overview reports how many there are.

Usage:
  python export_web_data.py --csv CURRENT_BEST_OUTPUT.csv
  python export_web_data.py --csv "RUN_#1.csv" --per-cluster 80
  python export_web_data.py --csv my_run.csv --graph citation_graph_adjacency.json --out some/folder
"""
import argparse
import ast
import glob
import json
import os
import shutil
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

OUT_DIR = os.path.join("web", "public", "data")
PER_CLUSTER = 500         # papers exported per cluster (the website draws a window of them at a time)
DETAIL_BLOCK = 50         # papers per details file (matches the website's window size)
MAX_CLUSTER_LINKS = 300   # strongest cluster-to-cluster links kept in the overview
FIELD_COUNT = 12          # arXiv fields that get their own color / filter chip; the rest are "other"

# Only these columns are read: the CSV also holds the full text twice, which we don't need.
WANTED = {
    "s2_paper_id", "title", "category", "category_code", "published_date", "authors", "summary",
    "citations_count", "influence_score", "x_2d", "y_2d", "umap_x", "umap_y", "cluster_label", "cluster_title",
}


def parse_dates(series):
    """Fast path for the usual M/D/YY format, falling back to slow per-value parsing for oddballs."""
    dates = pd.to_datetime(series, format="%m/%d/%y", errors="coerce")
    bad = dates.isna() & series.notna()
    if bad.any():
        dates[bad] = pd.to_datetime(series[bad], errors="coerce", format="mixed")
    return dates


def to_decimal_year(dates):
    """2021-07-01 -> 2021.5 (roughly). Used for the timeline."""
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
    """Center coordinates and scale them to fit [-1, 1], keeping the aspect ratio."""
    cx, cy = (x.max() + x.min()) / 2, (y.max() + y.min()) / 2
    half = max(x.max() - x.min(), y.max() - y.min()) / 2 or 1.0
    return (x - cx) / half, (y - cy) / half


def paper_record(r):
    """The light fields the website needs to draw one paper. `r` is a dict of one CSV row (plus year/date)."""
    year, date = r["year"], r["date"]
    return {
        "id": r["s2_paper_id"],
        "title": " ".join(str(r["title"]).split()),
        "year": None if pd.isna(year) else round(float(year), 3),
        "date": None if pd.isna(date) else date,
        "influence": round(float(r["influence_score"]), 4),
        "citations": int(r["citations_count"]),
        "category": r["category"],
        "categoryCode": r["category_code"],
    }


def detail_record(r):
    """The heavy fields, loaded only when a paper is selected."""
    return {
        "authors": parse_authors(r["authors"]),
        "abstract": " ".join(str(r["summary"]).split()),
    }


def load_citations(graph_path):
    """{citing paper: [cited papers]} or None. (The scraper stores each paper's *references*.)"""
    if not os.path.exists(graph_path):
        print(f"No citation graph at {graph_path}; exporting without links.")
        return None
    with open(graph_path, "r") as f:
        return json.load(f)


def scan_citations(adj, cluster_of):
    """
    First pass over the citation graph. Returns {(clusterA, clusterB): number of citations between them},
    counting every clustered paper, not just the exported ones. (Citations within a cluster don't count.)
    """
    pair_weights = Counter()
    if adj is None:
        return pair_weights
    for source, targets in adj.items():
        cs = cluster_of.get(source)
        if cs is None:
            continue
        for target in targets:
            ct = cluster_of.get(target)
            if ct is None or target == source or ct == cs:
                continue
            pair_weights[(min(cs, ct), max(cs, ct))] += 1
    return pair_weights


def scan_links(adj, selected):
    """
    Second pass, over the exported papers only. Returns
      inner: {cluster: [(citing, cited), ...]} between two papers of that cluster
      cross: {cluster: [(citing, cited), ...]} between a paper of that cluster and one of another cluster
             (listed under both clusters)
    """
    inner, cross = defaultdict(list), defaultdict(list)
    if adj is None:
        return inner, cross
    for source, targets in adj.items():
        cs = selected.get(source)
        if cs is None:
            continue
        for target in targets:
            ct = selected.get(target)
            if ct is None or target == source:
                continue
            if cs == ct:
                inner[cs].append((source, target))
            else:
                cross[cs].append((source, target))
                cross[ct].append((source, target))
    return inner, cross


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="enriched_arXiv_scientific_dataset_with_citations.csv")
    parser.add_argument("--graph", default="citation_graph_adjacency.json")
    parser.add_argument("--per-cluster", type=int, default=PER_CLUSTER, help="papers shown per cluster")
    parser.add_argument("--max-links", type=int, default=MAX_CLUSTER_LINKS, help="cluster-to-cluster links kept")
    parser.add_argument("--out", default=OUT_DIR, help="output folder")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, usecols=lambda c: c in WANTED)
    df = df.dropna(subset=["s2_paper_id"]).drop_duplicates("s2_paper_id")
    total_papers = len(df)
    print(f"Loaded {total_papers} papers from {args.csv}")

    xcol = "x_2d" if "x_2d" in df.columns else "umap_x"
    ycol = "y_2d" if "y_2d" in df.columns else "umap_y"
    df["cluster_label"] = df["cluster_label"].astype(int)
    df["citations_count"] = df["citations_count"].fillna(0).astype(int)
    df["influence_score"] = df["influence_score"].fillna(0.0)

    pub = parse_dates(df["published_date"])
    df["year"] = to_decimal_year(pub)
    df["date"] = pub.dt.strftime("%Y-%m-%d")

    clustered = df[df["cluster_label"] != -1].copy()
    if clustered.empty:
        raise SystemExit("This CSV has no clusters (every paper is noise), so there is nothing to show. "
                         "Pick a run that found clusters.")
    noise_papers = total_papers - len(clustered)
    print(f"{clustered['cluster_label'].nunique()} clusters, {len(clustered)} clustered papers, "
          f"{noise_papers} unclustered ({100 * noise_papers / total_papers:.0f}%)")

    # ---- arXiv fields: the biggest ones get their own color / filter chip, the rest are "other" ----
    field_counts = clustered["category_code"].value_counts()
    cat_name = clustered.drop_duplicates("category_code").set_index("category_code")["category"].to_dict()
    top_fields = list(field_counts.head(FIELD_COUNT).index)
    fields = [{"code": c, "name": cat_name.get(c, c), "papers": int(field_counts[c])} for c in top_fields]
    clustered["field_key"] = clustered["category_code"].where(clustered["category_code"].isin(top_fields), "other")
    field_keys = top_fields + ["other"]

    # ---- Year bins (whole years) shared by every histogram ----
    clustered["year_int"] = np.floor(clustered["year"])
    y0, y1 = int(clustered["year_int"].min()), int(clustered["year_int"].max())
    years = list(range(y0, y1 + 1))
    year_pos = {y: i for i, y in enumerate(years)}

    def per_year(frame):
        """papers per year and mean influence per year, aligned with `years`."""
        g = frame.dropna(subset=["year_int"]).groupby("year_int")["influence_score"].agg(["size", "mean"])
        counts = [int(g["size"].get(y, 0)) for y in years]
        infl = [round(float(g["mean"].get(y, 0.0)), 4) for y in years]
        return counts, infl

    def per_field_year(frame):
        """{field: counts per year} and {field: mean influence per year} for the field filter."""
        fy = {k: [0] * len(years) for k in field_keys}
        fi = {k: [0.0] * len(years) for k in field_keys}
        g = (frame.dropna(subset=["year_int"]).groupby(["field_key", "year_int"])["influence_score"]
             .agg(["size", "mean"]))
        for (key, y), row in g.iterrows():
            fy[key][year_pos[int(y)]] = int(row["size"])
            fi[key][year_pos[int(y)]] = round(float(row["mean"]), 4)
        return fy, fi

    # ---- Per-cluster overview stats ----
    cluster_rows = []
    for cid, g in clustered.groupby("cluster_label"):
        cats = g["category_code"].value_counts()
        top = g.loc[g["influence_score"].idxmax()]
        counts, infl = per_year(g)
        fy, fi = per_field_year(g)
        cluster_rows.append({
            "id": int(cid),
            "title": g["cluster_title"].iloc[0],
            "size": int(len(g)),
            "_x": float(g[xcol].mean()), "_y": float(g[ycol].mean()),
            "influence": round(float(g["influence_score"].mean()), 4),
            "category": cats.index[0],
            "categoryName": cat_name.get(cats.index[0], cats.index[0]),
            "field": cats.index[0] if cats.index[0] in top_fields else "other",
            "categoryShare": round(float(cats.iloc[0] / len(g)), 3),
            "topPaper": " ".join(str(top["title"]).split()),
            "count": counts,
            "inf": infl,
            "fy": fy,
            "fi": fi,
        })
    cx_ = pd.Series([c["_x"] for c in cluster_rows])
    cy_ = pd.Series([c["_y"] for c in cluster_rows])
    nx_, ny_ = normalize_xy(cx_, cy_)
    for i, c in enumerate(cluster_rows):
        c["x"], c["y"] = round(float(nx_.iloc[i]), 5), round(float(ny_.iloc[i]), 5)
        del c["_x"], c["_y"]
    cluster_rows.sort(key=lambda c: -c["size"])

    # ---- Top papers of each cluster, in influence order (a paper's index in its file is its rank) ----
    ranked = clustered.sort_values(["influence_score", "citations_count"], ascending=False)
    top = ranked.groupby("cluster_label").head(args.per_cluster)
    selected = dict(zip(top["s2_paper_id"], top["cluster_label"].astype(int)))
    rank_of = {pid: i for _, g in top.groupby("cluster_label") for i, pid in enumerate(g["s2_paper_id"])}
    cluster_of = dict(zip(clustered["s2_paper_id"], clustered["cluster_label"].astype(int)))

    adj = load_citations(args.graph)
    pair_weights = scan_citations(adj, cluster_of)
    strongest = pair_weights.most_common(args.max_links)
    links = [{"source": int(a), "target": int(b), "weight": int(w)} for (a, b), w in strongest]
    print(f"Citations between clusters: {len(pair_weights)} linked pairs, keeping the strongest {len(links)}.")

    inner_links, cross_links = scan_links(adj, selected)
    del adj  # large; no longer needed

    # ---- Write everything ----
    cdir = os.path.join(args.out, "clusters")
    ddir = os.path.join(args.out, "details")
    os.makedirs(cdir, exist_ok=True)
    for stale in glob.glob(os.path.join(cdir, "*.json")):  # clusters from older runs
        os.remove(stale)
    shutil.rmtree(ddir, ignore_errors=True)

    by_id = clustered.set_index("s2_paper_id", drop=False)
    total_bytes = detail_bytes = n_files = n_nodes = n_external = n_cross = n_inner = 0
    for cid, g in top.groupby("cluster_label"):
        cid = int(cid)
        mid_x, mid_y = (g[xcol].max() + g[xcol].min()) / 2, (g[ycol].max() + g[ycol].min()) / 2
        half = max(g[xcol].max() - g[xcol].min(), g[ycol].max() - g[ycol].min()) / 2 or 1.0
        nodes, details = [], {}
        for rank, row in enumerate(g.to_dict("records")):
            rec = paper_record(row)
            rec["rank"] = rank
            rec["x"], rec["y"] = round(float((row[xcol] - mid_x) / half), 5), round(float((row[ycol] - mid_y) / half), 5)
            nodes.append(rec)
            details.setdefault(rank // DETAIL_BLOCK, {})[row["s2_paper_id"]] = detail_record(row)

        # Papers of other clusters that have a link to this cluster's papers: the website draws them when
        # the other cluster is open, and otherwise only needs their title for the line's tooltip.
        cross = cross_links.get(cid, [])
        external_ids = {}
        for src, dst in cross:
            other = dst if selected[src] == cid else src
            external_ids[other] = external_ids.get(other, 0) + 1
        external, per_neighbor = [], defaultdict(lambda: [0, []])
        for pid in sorted(external_ids, key=lambda i: (selected[i], rank_of[i])):
            rec = paper_record(by_id.loc[pid].to_dict())
            rec["cluster"] = selected[pid]
            rec["rank"] = rank_of[pid]
            rec["inTop"] = True  # every outside paper is one of its own cluster's top papers
            external.append(rec)
            per_neighbor[selected[pid]][0] += external_ids[pid]
            per_neighbor[selected[pid]][1].append(pid)
        neighbors = sorted(({"cluster": int(c), "weight": w, "ids": ids} for c, (w, ids) in per_neighbor.items()),
                           key=lambda n: -n["weight"])

        inner = [(s, t, False) for s, t in inner_links.get(cid, [])]
        cross = [(s, t, True) for s, t in cross]
        payload = {
            "id": cid,
            "nodes": nodes,
            "external": external,
            "neighbors": neighbors,
            "links": [({"source": s, "target": t, "cross": 1} if x else {"source": s, "target": t}) for s, t, x in inner + cross],
        }
        path = os.path.join(cdir, f"{cid}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"))
        total_bytes += os.path.getsize(path)

        os.makedirs(os.path.join(ddir, str(cid)), exist_ok=True)
        for k, block in details.items():
            dpath = os.path.join(ddir, str(cid), f"{k}.json")
            with open(dpath, "w", encoding="utf-8") as f:
                json.dump(block, f, separators=(",", ":"))
            detail_bytes += os.path.getsize(dpath)
            n_files += 1
        n_nodes += len(nodes)
        n_external += len(external)
        n_cross += len(cross)
        n_inner += len(inner)

    all_counts, all_infl = per_year(clustered)
    overview = {
        "meta": {
            "totalPapers": total_papers,
            "clusteredPapers": int(len(clustered)),
            "noisePapers": int(noise_papers),
            "clusterCount": len(cluster_rows),
            "perCluster": args.per_cluster,
            "yearMin": round(float(clustered["year"].min()), 3),
            "yearMax": round(float(clustered["year"].max()), 3),
            "source": os.path.basename(args.csv),
        },
        "years": years,
        "timeline": {"count": all_counts, "inf": all_infl},
        "fields": fields,
        "clusters": cluster_rows,
        "links": links,
    }
    with open(os.path.join(args.out, "overview.json"), "w", encoding="utf-8") as f:
        json.dump(overview, f, separators=(",", ":"))

    n = len(cluster_rows)
    print(f"Wrote overview.json ({os.path.getsize(os.path.join(args.out, 'overview.json')) / 1024:.0f} KB) and "
          f"{n} cluster files ({total_bytes / 1048576:.1f} MB total, ~{total_bytes / n / 1024:.0f} KB each) to {args.out}/")
    print(f"Details: {n_files} files, {detail_bytes / 1048576:.1f} MB total (~{detail_bytes / max(n_files, 1) / 1024:.0f} KB each).")
    print(f"Per cluster: ~{n_nodes / n:.0f} papers, ~{n_inner / n:.0f} links inside it, ~{n_cross / n:.0f} links to "
          f"~{n_external / n:.0f} papers of other clusters.")


if __name__ == "__main__":
    main()
