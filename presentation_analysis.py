"""
Presentation analyses on the full clustered arXiv dataset (~3.2M papers).

1. Cluster dynamics: each cluster's share of all papers per publication year,
   with a log-linear growth rate over the last WINDOW years, separating
   expanding subfields from mature ones.
2. Influence efficiency: age-normalized citation density per cluster, compared
   between large/expansive and small/specialized clusters.
3. Cross-disciplinary breakthroughs: high-influence papers that sit on the
   boundary between clusters, scored by how mixed their embedding neighborhood
   is and how many different clusters cite them.

Inputs
    full CSV (or a Parquet copy of it) from the hierarchical pipeline, with
        id, title, categories, primary_cat, x_2d, y_2d, global_cluster_name, cluster_title
    enriched_arXiv_scientific_dataset.csv: Semantic Scholar citation counts (AI/ML subset)
    citation_graph_adjacency.json: citing paper -> referenced papers (AI/ML subset)

Citation data only exists for the AI/ML subset, so analyses 2 and 3 are limited
to papers that have a citation count.

Usage:
    python presentation_analysis.py enriched_arXiv_scientific_full_dataset.csv \
        --out visualization/presentation_data.json
"""
import argparse
import json
import re

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

WINDOW = 5                 # years in the growth fit
MIN_CLUSTER_PAPERS = 300   # clusters smaller than this are left out of analysis 1
EXPANDING_RATE = 0.10
CONTRACTING_RATE = -0.10
MIN_CITED_PAPERS = 50      # papers with citation data a cluster needs for analysis 2
MIN_CITERS = 10            # in-corpus citers a paper needs for the citer-diversity score
NEIGHBORS = 50

COLUMNS = ["id", "title", "categories", "primary_cat", "x_2d", "y_2d", "global_cluster_name", "cluster_title"]


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------
def load_papers(path):
    if path.endswith(".parquet"):
        df = pd.read_parquet(path, columns=COLUMNS)
    else:
        import pyarrow.csv as pc
        table = pc.read_csv(
            path,
            parse_options=pc.ParseOptions(newlines_in_values=True),
            convert_options=pc.ConvertOptions(include_columns=COLUMNS),
        )
        df = table.to_pandas()
    df["pub_year"] = df["id"].map(year_from_arxiv_id)
    df = df.dropna(subset=["pub_year"]).astype({"pub_year": int})
    df["s2_id"] = "ARXIV:" + df["id"]
    df["is_noise"] = df["global_cluster_name"].str.endswith("_-1")
    df["archive"] = df["primary_cat"].str.split(".").str[0]
    return df


NEW_ID = re.compile(r"^(\d{2})(\d{2})\.\d{4,5}")
OLD_ID = re.compile(r"/(\d{2})(\d{2})\d{3}")


def year_from_arxiv_id(arxiv_id):
    """First-submission year from the arXiv identifier (updated_date is the last revision)."""
    m = NEW_ID.match(arxiv_id) or OLD_ID.search(arxiv_id)
    if not m:
        return None
    yy = int(m.group(1))
    return 1900 + yy if yy >= 91 else 2000 + yy


def attach_citations(df, enriched_csv):
    cites = pd.read_csv(enriched_csv, usecols=["s2_paper_id", "citations_count"])
    cites = cites.dropna().drop_duplicates("s2_paper_id").set_index("s2_paper_id")["citations_count"]
    df["citations"] = df["s2_id"].map(cites)
    return df


# -----------------------------------------------------------------------------
# 1. Cluster dynamics
# -----------------------------------------------------------------------------
def growth_rate(shares, years, window, floor):
    """
    exp(slope) - 1 of a least-squares fit of log(share) on year over the last `window` years.
    Shares are floored at one paper's worth of that year's total so brand-new topics stay finite.
    """
    y = np.log(np.maximum(shares[-window:], floor[-window:]))
    return float(np.exp(np.polyfit(years[-window:], y, 1)[0]) - 1)


def classify(rate):
    if rate >= EXPANDING_RATE:
        return "expanding"
    if rate <= CONTRACTING_RATE:
        return "contracting"
    return "stable"


def cluster_dynamics(df, min_year):
    d = df[df["pub_year"] >= min_year]
    year_totals = d.groupby("pub_year").size()
    years = year_totals.index.to_numpy()
    clustered = d[~d["is_noise"]]
    sizes = clustered.groupby("global_cluster_name").size()
    keep = sizes[sizes >= MIN_CLUSTER_PAPERS].index
    counts = (pd.crosstab(clustered["global_cluster_name"], clustered["pub_year"])
              .reindex(index=keep, columns=years, fill_value=0))
    shares = counts / year_totals.to_numpy()
    floor = 1.0 / year_totals.to_numpy()

    meta = clustered.groupby("global_cluster_name").agg(
        title=("cluster_title", "first"), primary_cat=("primary_cat", "first"),
        archive=("archive", "first"), mean_year=("pub_year", "mean"))
    recent = d["pub_year"] >= years[-1] - WINDOW + 1

    rows = []
    for name in keep:
        s = shares.loc[name].to_numpy()
        rate = growth_rate(s, years, WINDOW, floor)
        m = meta.loc[name]
        rows.append({
            "id": name, "title": m["title"], "primary_cat": m["primary_cat"], "archive": m["archive"],
            "size": int(sizes[name]), "mean_year": round(float(m["mean_year"]), 2),
            "growth_rate": round(rate, 4), "status": classify(rate),
            "share_recent": float(s[-WINDOW:].mean()),
            "recent_frac": round(float(counts.loc[name].to_numpy()[-WINDOW:].sum() / sizes[name]), 3),
            "counts": [int(c) for c in counts.loc[name]],
        })
    clusters = pd.DataFrame(rows)

    # Field-level view: each archive's share of the year's papers.
    arch = pd.crosstab(d["archive"], d["pub_year"]).reindex(columns=years, fill_value=0)
    top_arch = arch.sum(axis=1).sort_values(ascending=False).index[:10]
    arch_share = arch / year_totals.to_numpy()
    other = arch_share.drop(top_arch).sum(axis=0)
    archives = [{"archive": a, "shares": [round(float(v), 5) for v in arch_share.loc[a]]} for a in top_arch]
    archives.append({"archive": "other", "shares": [round(float(v), 5) for v in other]})

    return clusters, {"years": [int(y) for y in years], "year_totals": [int(t) for t in year_totals],
                      "archives": archives, "n_recent": int(recent.sum())}


# -----------------------------------------------------------------------------
# 2. Influence efficiency
# -----------------------------------------------------------------------------
def normalized_citations(df):
    """Citations divided by the mean for papers published the same year (1.0 = average)."""
    c = df.dropna(subset=["citations"])
    year_mean = c.groupby("pub_year")["citations"].transform("mean")
    df.loc[c.index, "norm_cites"] = c["citations"] / year_mean.replace(0, np.nan)
    top10 = c.groupby("pub_year")["citations"].transform(lambda s: s.quantile(0.9))
    df.loc[c.index, "top10"] = (c["citations"] >= top10) & (c["citations"] > 0)
    return df


def influence_efficiency(df, dyn):
    c = df[df["norm_cites"].notna() & ~df["is_noise"]]
    g = c.groupby("global_cluster_name")
    eff = pd.DataFrame({
        "cited_papers": g.size(),
        "density": g["norm_cites"].mean(),
        "median_density": g["norm_cites"].median(),
        "top10_rate": g["top10"].mean(),
        "total_citations": g["citations"].sum(),
    })
    eff = eff[eff["cited_papers"] >= MIN_CITED_PAPERS]
    sizes = df[~df["is_noise"]].groupby("global_cluster_name").size()
    eff["size"] = sizes.reindex(eff.index)
    meta = df.groupby("global_cluster_name").agg(title=("cluster_title", "first"), primary_cat=("primary_cat", "first"))
    eff = eff.join(meta)
    eff = eff.join(dyn.set_index("id")[["growth_rate", "status"]], how="left")

    # Expansive = top quartile by size, specialized = bottom quartile, among these clusters.
    q1, q3 = eff["size"].quantile([0.25, 0.75])
    eff["scope"] = np.where(eff["size"] >= q3, "expansive", np.where(eff["size"] <= q1, "specialized", "mid"))

    def summary(sub):
        return {"clusters": int(len(sub)), "papers": int(sub["size"].sum()),
                "median_density": round(float(sub["density"].median()), 3),
                "median_top10_rate": round(float(sub["top10_rate"].median()), 3)}

    groups = {k: summary(eff[eff["scope"] == k]) for k in ["expansive", "mid", "specialized"]}
    groups["size_cutoffs"] = [int(q1), int(q3)]
    for status in ["expanding", "stable", "contracting"]:
        sub = eff[eff["status"] == status]
        if len(sub):
            groups[f"status_{status}"] = summary(sub)
    rho = eff[["size", "density"]].corr(method="spearman").iloc[0, 1]
    groups["spearman_size_density"] = round(float(rho), 3)
    return eff.reset_index().rename(columns={"global_cluster_name": "id"}), groups


# -----------------------------------------------------------------------------
# 3. Boundary papers
# -----------------------------------------------------------------------------
def entropy_and_count(labels):
    vals, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum()), len(vals)


def boundary_papers(df, adjacency_path, top_n=40):
    cluster_codes, cluster_names = pd.factorize(df["global_cluster_name"])
    df["cluster_code"] = cluster_codes
    cand = df[df["norm_cites"].notna()].copy()

    # (a) Embedding neighborhood: how many clusters/fields surround the paper in the global map.
    tree = KDTree(df[["x_2d", "y_2d"]].to_numpy(dtype=np.float64))
    _, idx = tree.query(cand[["x_2d", "y_2d"]].to_numpy(dtype=np.float64), k=NEIGHBORS + 1)
    pos = df.index.get_indexer(cand.index)
    neigh_entropy, foreign_frac, neigh_archives = [], [], []
    codes = df["cluster_code"].to_numpy()
    archives = df["archive"].to_numpy()
    for p, row in zip(pos, idx):
        nb = row[row != p][:NEIGHBORS]
        e, _ = entropy_and_count(codes[nb])
        neigh_entropy.append(e)
        foreign_frac.append(float((codes[nb] != codes[p]).mean()))
        neigh_archives.append(len(set(archives[nb])))
    cand["neighbor_entropy"] = neigh_entropy
    cand["foreign_neighbor_frac"] = foreign_frac
    cand["neighbor_archives"] = neigh_archives

    # (b) Citer diversity: how many different clusters cite the paper (in-corpus citations).
    with open(adjacency_path) as f:
        adj = json.load(f)
    # Noise citers count toward the citer total and category mix, not the cluster mix.
    cat_codes, cat_names = pd.factorize(df["primary_cat"])
    lookup = pd.DataFrame({"cluster": codes, "cat": cat_codes, "noise": df["is_noise"].to_numpy()}, index=df["s2_id"])
    lookup = lookup[~lookup.index.duplicated()]
    citer_clusters, citer_cats = {}, {}
    for citer, refs in adj.items():
        if citer not in lookup.index:
            continue
        cl, ct, noise = lookup.loc[citer]
        for r in refs:
            citer_cats.setdefault(r, []).append(ct)
            if not noise:
                citer_clusters.setdefault(r, []).append(cl)
    stats = {}
    for k, cats in citer_cats.items():
        if len(cats) < MIN_CITERS:
            continue
        cl = citer_clusters.get(k, [])
        ce, cn = entropy_and_count(np.array(cl)) if cl else (0.0, 0)
        ke, kn = entropy_and_count(np.array(cats))
        stats[k] = (len(cats), ce, cn, ke, kn)
    st = pd.DataFrame.from_dict(stats, orient="index",
                                columns=["corpus_citers", "citer_entropy", "citer_clusters", "citer_cat_entropy", "citer_cats"])
    cand = cand.join(st, on="s2_id")

    # (c) Cross-listing across archives.
    cand["cross_archives"] = cand["categories"].str.split().map(lambda cs: len({c.split(".")[0] for c in cs}))

    cand["influence_pct"] = cand["norm_cites"].rank(pct=True)
    cand["neighbor_pct"] = cand["neighbor_entropy"].rank(pct=True)
    cand["citer_pct"] = cand["citer_entropy"].rank(pct=True)
    cand["citer_cat_pct"] = cand["citer_cat_entropy"].rank(pct=True)
    # Papers without enough in-corpus citers get the embedding score only.
    cand["boundary_score"] = cand[["neighbor_pct", "citer_pct", "citer_cat_pct"]].mean(axis=1, skipna=True)

    breakthroughs = cand[(cand["influence_pct"] >= 0.98) & cand["citer_entropy"].notna()]
    breakthroughs = breakthroughs.sort_values("boundary_score", ascending=False).head(top_n)

    def top_counts(v, names, n=4):
        u, c = np.unique(v, return_counts=True)
        order = np.argsort(-c)[:n]
        return [[str(names[u[i]]), int(c[i])] for i in order]

    titles = df.drop_duplicates("global_cluster_name").set_index("global_cluster_name")["cluster_title"]
    out = []
    for _, r in breakthroughs.iterrows():
        bridged = top_counts(citer_clusters.get(r["s2_id"], []), cluster_names)
        cats = top_counts(citer_cats.get(r["s2_id"], []), cat_names, n=5)
        out.append({
            "id": r["id"], "title": " ".join(str(r["title"]).split()), "year": int(r["pub_year"]),
            "categories": r["categories"], "cluster": r["global_cluster_name"], "cluster_title": r["cluster_title"],
            "citations": int(r["citations"]), "norm_cites": round(float(r["norm_cites"]), 2),
            "corpus_citers": int(r["corpus_citers"]), "citer_clusters": int(r["citer_clusters"]),
            "citer_entropy": round(float(r["citer_entropy"]), 2),
            "citer_cats": int(r["citer_cats"]), "citer_cat_entropy": round(float(r["citer_cat_entropy"]), 2),
            "citing_categories": cats,
            "neighbor_archives": int(r["neighbor_archives"]),
            "foreign_neighbor_frac": round(float(r["foreign_neighbor_frac"]), 2),
            "boundary_score": round(float(r["boundary_score"]), 3),
            "x": round(float(r["x_2d"]), 3), "y": round(float(r["y_2d"]), 3),
            "bridged": [[n, titles.get(n, n), c] for n, c in bridged],
        })

    # Scatter: year-normalized citations vs boundary score for a sample of cited papers
    # (uncited papers all tie at zero, so they are counted but not plotted).
    scored = cand.dropna(subset=["boundary_score"])
    cited = scored[scored["norm_cites"] > 0]
    sample = cited.sample(n=min(6000, len(cited)), random_state=42)
    scatter = [[round(float(a), 4), round(float(b), 3)] for a, b in sample[["norm_cites", "boundary_score"]].itertuples(index=False)]
    corr = scored[["influence_pct", "boundary_score"]].corr(method="spearman").iloc[0, 1]
    meta = {"candidates": int(len(cand)), "with_citers": int(cand["citer_entropy"].notna().sum()),
            "uncited_frac": round(float((scored["norm_cites"] == 0).mean()), 3),
            "top2_cutoff": round(float(cand.loc[cand["influence_pct"] >= 0.98, "norm_cites"].min()), 3),
            "spearman_influence_boundary": round(float(corr), 3)}
    return out, scatter, meta


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("papers", help="Full clustered CSV, or a Parquet copy of it")
    ap.add_argument("--citations", default="enriched_arXiv_scientific_dataset.csv")
    ap.add_argument("--adjacency", default="citation_graph_adjacency.json")
    ap.add_argument("--min-year", type=int, default=1995)
    ap.add_argument("--out", default="visualization/presentation_data.json")
    args = ap.parse_args()

    df = load_papers(args.papers)
    print(f"{len(df):,} papers, years {df.pub_year.min()}-{df.pub_year.max()}, noise {df.is_noise.mean():.1%}")
    df = attach_citations(df, args.citations)
    df = normalized_citations(df)
    print(f"{df.citations.notna().sum():,} papers with citation counts")

    dyn, dyn_meta = cluster_dynamics(df, args.min_year)
    eff, eff_groups = influence_efficiency(df, dyn)
    breakthroughs, scatter, b_meta = boundary_papers(df, args.adjacency)

    dyn_sorted = dyn.sort_values("growth_rate", ascending=False)
    result = {
        "meta": {
            "n_papers": int(len(df)), "n_clusters": int(df.loc[~df.is_noise, "global_cluster_name"].nunique()),
            "n_categories": int(df.primary_cat.nunique()), "noise_frac": round(float(df.is_noise.mean()), 4),
            "n_cited": int(df.citations.notna().sum()), "window": WINDOW,
            "min_cluster_papers": MIN_CLUSTER_PAPERS, "expanding_rate": EXPANDING_RATE,
            "contracting_rate": CONTRACTING_RATE, "min_cited_papers": MIN_CITED_PAPERS,
            "last_year": int(df.pub_year.max()),
            "last_id_month": str(df["id"][df["id"].str.match(NEW_ID)].max()[:4]),
        },
        "dynamics": {**dyn_meta,
                     "status_counts": dyn["status"].value_counts().to_dict(),
                     "clusters": json.loads(dyn_sorted.to_json(orient="records"))},
        "efficiency": {"groups": eff_groups,
                       "clusters": json.loads(eff.drop(columns=["total_citations"]).round(4).to_json(orient="records"))},
        "boundary": {**b_meta, "papers": breakthroughs, "scatter": scatter},
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, separators=(",", ":"), ensure_ascii=False)

    print(f"\n[1] {len(dyn)} clusters >= {MIN_CLUSTER_PAPERS} papers: {result['dynamics']['status_counts']}")
    for _, r in pd.concat([dyn_sorted.head(12), dyn_sorted.tail(8)]).iterrows():
        print(f"   {r['growth_rate']:+7.1%}  {r['size']:6d}  {r['mean_year']:.0f}  {r['id']:<22} {r['title']}")
    print(f"\n[2] {len(eff)} clusters with >= {MIN_CITED_PAPERS} cited papers")
    print(json.dumps(eff_groups, indent=1))
    for _, r in eff.sort_values("density", ascending=False).head(10).iterrows():
        print(f"   {r.density:5.2f}x  n={r.cited_papers:5d}  {r.id:<18} {r.title}")
    print(f"\n[3] {b_meta}")
    for p in breakthroughs[:15]:
        print(f"   {p['boundary_score']:.3f} {p['citations']:6d}c  {p['year']}  {p['title'][:70]}  <- {[b[0] for b in p['citing_categories']]}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
