"""
Presentation analyses on the full clustered arXiv dataset (~3.2M papers).

1. Cluster dynamics: each cluster's share of all papers per publication year,
   with a log-linear growth rate over the last WINDOW years, separating
   expanding subfields from mature ones.
2. Influence efficiency: field- and age-normalized citation density per cluster,
   compared between large/expansive and small/specialized clusters.
3. Cross-disciplinary breakthroughs: high-influence papers that sit on the
   boundary between clusters, scored by how mixed their embedding neighborhood
   is and how many different clusters and arXiv fields cite them.

Inputs
    full CSV (or a Parquet copy of it) from the hierarchical pipeline, with
        id, title, categories, primary_cat, x_2d, y_2d, global_cluster_name, cluster_title
    citation_graph_adjacency_full.json: citing paper -> referenced papers, for the whole corpus

Citations are in-corpus citations: how many papers in the dataset reference a paper.
Citation habits differ a lot between fields (astro-ph papers list ~29 arXiv references,
cs papers ~5), so counts are normalized within each primary category and publication year.

Usage:
    python presentation_analysis.py enriched_arXiv_scientific_full_dataset.csv \
        --out visualization/presentation_data.json [--edge-cache edges.npz]
"""
import argparse
import json
import re

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

from citation_graph import load_edges_cached

WINDOW = 5                 # years in the growth fit
MIN_CLUSTER_PAPERS = 300   # clusters smaller than this are left out of analyses 1 and 2
EXPANDING_RATE = 0.10
CONTRACTING_RATE = -0.10
MIN_CITERS = 10            # in-corpus citers a paper needs to be scored in analysis 3
NEIGHBORS = 50
TOP_INFLUENCE = 0.98       # breakthrough candidates come from this citation percentile and up

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
    df = df.dropna(subset=["pub_year"]).astype({"pub_year": int}).reset_index(drop=True)
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
def normalized_citations(df, cited):
    """
    In-corpus citation count per paper, divided by the mean for papers in the same
    primary category and publication year (1.0 = typical for that field and year).
    """
    df["citations"] = np.bincount(cited, minlength=len(df))
    grp = df.groupby(["primary_cat", "pub_year"])["citations"]
    mean = grp.transform("mean")
    df["norm_cites"] = (df["citations"] / mean.replace(0, np.nan)).fillna(0.0)
    q90 = grp.transform(lambda s: s.quantile(0.9))
    df["top10"] = (df["citations"] >= q90) & (df["citations"] > 0)
    return df


def influence_efficiency(df, dyn):
    c = df[~df["is_noise"]].assign(cited=lambda d: d["citations"] > 0)
    g = c.groupby("global_cluster_name")
    eff = pd.DataFrame({
        "size": g.size(),
        "cited_frac": g["cited"].mean(),
        "density": g["norm_cites"].mean(),
        "median_density": g["norm_cites"].median(),
        "top10_rate": g["top10"].mean(),
    })
    eff = eff[eff["size"] >= MIN_CLUSTER_PAPERS]
    meta = c.groupby("global_cluster_name").agg(title=("cluster_title", "first"), primary_cat=("primary_cat", "first"))
    eff = eff.join(meta).join(dyn.set_index("id")[["growth_rate", "status"]], how="left")

    # Expansive = top quartile by size, specialized = bottom quartile.
    q1, q3 = eff["size"].quantile([0.25, 0.75])
    eff["scope"] = np.where(eff["size"] >= q3, "expansive", np.where(eff["size"] <= q1, "specialized", "mid"))

    def summary(sub):
        return {"clusters": int(len(sub)), "papers": int(sub["size"].sum()),
                "median_density": round(float(sub["density"].median()), 3),
                "median_top10_rate": round(float(sub["top10_rate"].median()), 3)}

    groups = {k: summary(eff[eff["scope"] == k]) for k in ["expansive", "mid", "specialized"]}
    groups["size_cutoffs"] = [int(q1), int(q3)]
    for status in ["expanding", "stable", "contracting"]:
        groups[f"status_{status}"] = summary(eff[eff["status"] == status])
    rho = eff[["size", "density"]].corr(method="spearman").iloc[0, 1]
    groups["spearman_size_density"] = round(float(rho), 3)
    return eff.reset_index().rename(columns={"global_cluster_name": "id"}), groups


# -----------------------------------------------------------------------------
# 3. Boundary papers
# -----------------------------------------------------------------------------
def row_entropy(labels):
    """Shannon entropy (bits) of the label distribution in each row of an int matrix."""
    s = np.sort(labels, axis=1)
    n, k = s.shape
    new_run = np.ones_like(s, dtype=bool)
    new_run[:, 1:] = s[:, 1:] != s[:, :-1]
    run_id = np.cumsum(new_run, axis=1) - 1
    counts = np.zeros((n, k), dtype=np.int32)
    np.add.at(counts, (np.repeat(np.arange(n), k), run_id.ravel()), 1)
    p = counts / k
    with np.errstate(divide="ignore", invalid="ignore"):
        return -(np.where(p > 0, p * np.log2(p), 0.0)).sum(axis=1)


def grouped_entropy(group, label, n_groups):
    """Entropy (bits) and distinct-label count of `label` within each `group` id."""
    base = int(label.max()) + 1
    uniq, cnt = np.unique(group.astype(np.int64) * base + label, return_counts=True)
    g = uniq // base
    tot = np.bincount(g, weights=cnt, minlength=n_groups)
    p = cnt / tot[g]
    ent = np.zeros(n_groups)
    np.add.at(ent, g, -p * np.log2(p))
    return ent, np.bincount(g, minlength=n_groups)


def top_counts(values, names, n):
    u, c = np.unique(values, return_counts=True)
    order = np.argsort(-c)[:n]
    return [[str(names[u[i]]), int(c[i])] for i in order]


def boundary_papers(df, citing, cited, top_n=40):
    n = len(df)
    cluster_codes, cluster_names = pd.factorize(df["global_cluster_name"])
    archive_codes, archive_names = pd.factorize(df["archive"])
    noise = df["is_noise"].to_numpy()

    # (a) Citer diversity across clusters (noise citers left out) and across arXiv fields.
    ok = ~noise[citing]
    citer_cl_ent, citer_cl_n = grouped_entropy(cited[ok], cluster_codes[citing[ok]], n)
    citer_ar_ent, citer_ar_n = grouped_entropy(cited, archive_codes[citing], n)

    cand = df[df["citations"] >= MIN_CITERS].copy()
    pos = cand.index.to_numpy()
    cand["citer_entropy"] = citer_cl_ent[pos]
    cand["citer_clusters"] = citer_cl_n[pos]
    cand["citer_field_entropy"] = citer_ar_ent[pos]
    cand["citer_fields"] = citer_ar_n[pos]

    # (b) Embedding neighborhood: how many clusters surround the paper in the global map.
    tree = KDTree(df[["x_2d", "y_2d"]].to_numpy(dtype=np.float64))
    xy = cand[["x_2d", "y_2d"]].to_numpy(dtype=np.float64)
    neigh_ent = np.empty(len(cand))
    foreign = np.empty(len(cand))
    step = 100_000
    for s in range(0, len(cand), step):
        _, idx = tree.query(xy[s:s + step], k=NEIGHBORS + 1)
        codes = cluster_codes[idx[:, 1:]]               # drop the nearest point (normally the paper itself)
        neigh_ent[s:s + step] = row_entropy(codes)
        foreign[s:s + step] = (codes != cluster_codes[pos[s:s + step], None]).mean(axis=1)
    cand["neighbor_entropy"] = neigh_ent
    cand["foreign_neighbor_frac"] = foreign

    # Influence percentile is over all papers, so "top 2%" means the top 2% of the corpus.
    all_pct = df["norm_cites"].rank(pct=True)
    cand["influence_pct"] = all_pct.to_numpy()[pos]
    parts = []
    for col in ["neighbor_entropy", "citer_entropy", "citer_field_entropy"]:
        cand[col + "_pct"] = cand[col].rank(pct=True)
        parts.append(col + "_pct")
    cand["boundary_score"] = cand[parts].mean(axis=1)

    pool = cand[cand["influence_pct"] >= TOP_INFLUENCE]
    top = pool.sort_values("boundary_score", ascending=False).head(top_n)

    # Citer breakdowns for the listed papers only.
    order = np.argsort(cited, kind="stable")
    sorted_cited = cited[order]
    titles = df.drop_duplicates("global_cluster_name").set_index("global_cluster_name")["cluster_title"]
    out = []
    for i, r in top.iterrows():
        lo, hi = np.searchsorted(sorted_cited, [i, i + 1])
        citers = citing[order[lo:hi]]
        cl = citers[~noise[citers]]
        out.append({
            "id": r["id"], "title": " ".join(str(r["title"]).split()), "year": int(r["pub_year"]),
            "categories": r["categories"], "primary_cat": r["primary_cat"], "archive": r["archive"],
            "cluster": r["global_cluster_name"], "cluster_title": r["cluster_title"],
            "citations": int(r["citations"]), "norm_cites": round(float(r["norm_cites"]), 1),
            "citer_clusters": int(r["citer_clusters"]), "citer_fields": int(r["citer_fields"]),
            "citer_entropy": round(float(r["citer_entropy"]), 2),
            "citer_field_entropy": round(float(r["citer_field_entropy"]), 2),
            "foreign_neighbor_frac": round(float(r["foreign_neighbor_frac"]), 2),
            "boundary_score": round(float(r["boundary_score"]), 3),
            "citing_fields": top_counts(archive_codes[citers], archive_names, 6),
            "bridged": [[nm, titles.get(nm, nm), c] for nm, c in top_counts(cluster_codes[cl], cluster_names, 4)],
        })

    sample = cand.sample(n=min(6000, len(cand)), random_state=42)
    scatter = [[round(float(a), 3), round(float(b), 3)]
               for a, b in sample[["norm_cites", "boundary_score"]].itertuples(index=False)]
    corr = cand[["norm_cites", "boundary_score"]].corr(method="spearman").iloc[0, 1]
    cutoff = float(df["norm_cites"][all_pct >= TOP_INFLUENCE].min())
    meta = {"candidates": int(len(cand)), "top_pool": int(len(pool)),
            "top2_cutoff": round(cutoff, 3), "min_citers": MIN_CITERS,
            "pool_median_fields": float(pool["citer_fields"].median()),
            "spearman_influence_boundary": round(float(corr), 3)}
    return out, scatter, meta


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("papers", help="Full clustered CSV, or a Parquet copy of it")
    ap.add_argument("--adjacency", default="citation_graph_adjacency_full.json")
    ap.add_argument("--edge-cache", default=None, help="Optional .npz cache for the parsed citation graph")
    ap.add_argument("--min-year", type=int, default=1995)
    ap.add_argument("--out", default="visualization/presentation_data.json")
    args = ap.parse_args()

    df = load_papers(args.papers)
    print(f"{len(df):,} papers, years {df.pub_year.min()}-{df.pub_year.max()}, noise {df.is_noise.mean():.1%}")
    citing, cited, gstats = load_edges_cached(args.adjacency, df["s2_id"].tolist(), args.edge_cache)
    print(f"{len(citing):,} in-corpus citations from {gstats['citing_keys']:,} citing papers")
    df = normalized_citations(df, cited)

    dyn, dyn_meta = cluster_dynamics(df, args.min_year)
    eff, eff_groups = influence_efficiency(df, dyn)
    breakthroughs, scatter, b_meta = boundary_papers(df, citing, cited)

    has_refs = np.bincount(citing, minlength=len(df)) > 0
    ref_cov = pd.Series(has_refs).groupby(df["archive"]).mean().sort_values(ascending=False)
    dyn_sorted = dyn.sort_values("growth_rate", ascending=False)
    result = {
        "meta": {
            "n_papers": int(len(df)), "n_clusters": int(df.loc[~df.is_noise, "global_cluster_name"].nunique()),
            "n_categories": int(df.primary_cat.nunique()), "noise_frac": round(float(df.is_noise.mean()), 4),
            "n_edges": int(len(citing)), "n_citing": int(has_refs.sum()),
            "cited_frac": round(float((df.citations > 0).mean()), 3),
            "ref_coverage": {k: round(float(v), 3) for k, v in ref_cov.items()},
            "window": WINDOW, "min_cluster_papers": MIN_CLUSTER_PAPERS, "expanding_rate": EXPANDING_RATE,
            "contracting_rate": CONTRACTING_RATE, "last_year": int(df.pub_year.max()),
            "last_id_month": str(df["id"][df["id"].str.match(NEW_ID)].max()[:4]),
        },
        "dynamics": {**dyn_meta,
                     "status_counts": dyn["status"].value_counts().to_dict(),
                     "clusters": json.loads(dyn_sorted.to_json(orient="records"))},
        "efficiency": {"groups": eff_groups, "clusters": json.loads(eff.round(4).to_json(orient="records"))},
        "boundary": {**b_meta, "papers": breakthroughs, "scatter": scatter},
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, separators=(",", ":"), ensure_ascii=False)

    print(f"\n[1] {len(dyn)} clusters >= {MIN_CLUSTER_PAPERS} papers: {result['dynamics']['status_counts']}")
    print(f"\n[2] {len(eff)} clusters")
    print(json.dumps(eff_groups))
    for _, r in eff.sort_values("density", ascending=False).head(12).iterrows():
        print(f"   {r['density']:5.2f}x  {r['size']:6d}  {r['id']:<22} {r['title']}")
    print(f"\n[3] {b_meta}")
    for p in breakthroughs[:25]:
        print(f"   {p['boundary_score']:.3f} {p['citations']:6d}c {p['year']} {p['primary_cat']:<12} "
              f"{p['title'][:60]}  <- {p['citing_fields'][:4]}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
