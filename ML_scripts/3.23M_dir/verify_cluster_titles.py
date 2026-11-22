import os
import json
import math
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

# ==============================================================================
# CONFIGURATION — Update column names if yours differ
# ==============================================================================
CLUSTERED_CSV = "./outputs/full_dataset_clustering/enriched_arXiv_scientific_full_dataset.csv"
AUDIT_TXT_OUT = "./outputs/full_dataset_clustering/cluster_title_audit_cards.txt"
AUDIT_JSON_OUT = "./outputs/full_dataset_clustering/cluster_title_audit_cards.json"

COL_ID = "s2_paper_id"
COL_TITLE = "title"
COL_CLUSTER_ID = "cluster_label"
COL_CLUSTER_TITLE = "cluster_title"
COL_IS_CORE = None          # Boolean (True = core, False = halo); set to None if unavailable
COL_PAGERANK = "influence_score"        # Float PageRank score; falls back to 'citations_count' if missing
COL_CATEGORIES = "categories"    # Space-separated arXiv categories e.g. "cs.CL cs.AI"
COL_YEAR = "year"                # Optional publication year column; set to None if unavailable

NOISE_CLUSTER_IDS = {-1, "-1", None}
RANDOM_SEED = 42


def shannon_entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts if c > 0)


def compute_ctfidf_terms(
    df: pd.DataFrame,
    valid_cluster_ids: list,
    top_k: int = 6,
    max_papers_per_cluster: int = 1500,
) -> dict:
    """
    Computes class-based TF-IDF (c-TF-IDF) across cluster documents using core
    papers (or top-PageRank papers) so every cluster gets distinctive keywords.
    """
    docs = []
    ordered_cids = []

    sub = df[df[COL_CLUSTER_ID].isin(valid_cluster_ids)]
    for cid, grp in sub.groupby(COL_CLUSTER_ID, sort=False):
        if COL_IS_CORE and COL_IS_CORE in grp.columns:
            core_grp = grp[grp[COL_IS_CORE] == True]
            if len(core_grp) > 0:
                grp = core_grp
        if len(grp) > max_papers_per_cluster:
            grp = grp.sample(n=max_papers_per_cluster, random_state=RANDOM_SEED)

        text_blob = " ".join(grp[COL_TITLE].dropna().astype(str).tolist())
        docs.append(text_blob)
        ordered_cids.append(cid)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        max_features=50_000,
        min_df=2,
        max_df=0.75,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b",
    )
    tfidf_matrix = vectorizer.fit_transform(docs)
    feature_names = np.array(vectorizer.get_feature_names_out())

    top_terms_map = {}
    for row_idx, cid in enumerate(ordered_cids):
        row = tfidf_matrix.getrow(row_idx)
        if row.nnz == 0:
            top_terms_map[cid] = []
            continue
        top_indices = row.indices[np.argsort(row.data)[::-1][:top_k]]
        top_terms_map[cid] = feature_names[top_indices].tolist()

    return top_terms_map


def build_audit_cards(df: pd.DataFrame) -> list[dict]:
    # Filter out noise cluster (-1)
    valid_df = df[~df[COL_CLUSTER_ID].isin(NOISE_CLUSTER_IDS) & df[COL_CLUSTER_ID].notna()].copy()

    # Determine ranking metric (prefer PageRank, fallback to citations_count)
    rank_col = COL_PAGERANK if COL_PAGERANK in valid_df.columns else "citations_count"
    if rank_col not in valid_df.columns:
        valid_df["_dummy_rank"] = 0
        rank_col = "_dummy_rank"

    # Compute per-cluster summary stats
    cluster_stats = []
    title_counts = Counter(
        valid_df.drop_duplicates(COL_CLUSTER_ID)[COL_CLUSTER_TITLE].fillna("").astype(str)
    )

    for cid, grp in valid_df.groupby(COL_CLUSTER_ID, sort=False):
        n_total = len(grp)
        n_core = int((grp[COL_IS_CORE] == True).sum()) if (COL_IS_CORE and COL_IS_CORE in grp.columns) else n_total
        n_halo = n_total - n_core
        ctitle = str(grp[COL_CLUSTER_TITLE].iloc[0]) if COL_CLUSTER_TITLE in grp.columns else "N/A"

        # Category breakdown & Shannon entropy
        cat_counter = Counter()
        if COL_CATEGORIES in grp.columns:
            for raw_cats in grp[COL_CATEGORIES].dropna().astype(str):
                for c in raw_cats.split():
                    cat_counter[c] += 1

        top_cats = [
            (cat, cnt, round(100.0 * cnt / max(1, sum(cat_counter.values())), 1))
            for cat, cnt in cat_counter.most_common(3)
        ]
        entropy = shannon_entropy(list(cat_counter.values()))

        cluster_stats.append({
            "cluster_id": cid,
            "cluster_title": ctitle,
            "size_total": n_total,
            "size_core": n_core,
            "size_halo": n_halo,
            "title_collision_count": title_counts[ctitle],
            "category_entropy": round(entropy, 3),
            "top_categories": top_cats,
        })

    stats_df = pd.DataFrame(cluster_stats)
    print(f"Loaded {len(valid_df):,} clustered papers across {len(stats_df):,} clusters.")

    # --------------------------------------------------------------------------
    # Stratified Sampling Across 4 Diagnostic Buckets
    # --------------------------------------------------------------------------
    selected_buckets: dict[str, list[str]] = {}

    def register_bucket(cids, reason_label: str):
        for cid in cids:
            selected_buckets.setdefault(cid, []).append(reason_label)

    # 1. Top 15 largest clusters
    largest_cids = stats_df.sort_values("size_total", ascending=False).head(15)["cluster_id"].tolist()
    register_bucket(largest_cids, "TOP_15_LARGEST")

    # 2. 15 random mid-sized clusters (25th to 75th percentile by size)
    q25, q75 = stats_df["size_total"].quantile([0.25, 0.75])
    mid_pool = stats_df[(stats_df["size_total"] >= q25) & (stats_df["size_total"] <= q75)]
    mid_cids = mid_pool.sample(n=min(15, len(mid_pool)), random_state=RANDOM_SEED)["cluster_id"].tolist()
    register_bucket(mid_cids, "RANDOM_MID_SIZED")

    # 3. Top 5 duplicate/colliding cluster titles (2 sample clusters per colliding title)
    colliding = stats_df[stats_df["title_collision_count"] > 1].sort_values(
        ["title_collision_count", "size_total"], ascending=[False, False]
    )
    if len(colliding) > 0:
        dup_cids = colliding.groupby("cluster_title").head(2).head(10)["cluster_id"].tolist()
        register_bucket(dup_cids, "DUPLICATE_TITLE_FLAG")

    # 4. Top 10 highest-entropy clusters (among clusters with >= 50 papers)
    non_tiny = stats_df[stats_df["size_total"] >= 50]
    high_ent_cids = non_tiny.sort_values("category_entropy", ascending=False).head(10)["cluster_id"].tolist()
    register_bucket(high_ent_cids, "HIGH_ENTROPY_FLAG")

    sampled_cids = list(selected_buckets.keys())
    print(f"Selected {len(sampled_cids)} unique clusters for stratified audit. Computing c-TF-IDF terms...")

    # Compute c-TF-IDF terms across all clusters (gives true inverse document frequency)
    ctfidf_map = compute_ctfidf_terms(valid_df, stats_df["cluster_id"].tolist(), top_k=6)

    # Assemble detailed audit cards
    stats_lookup = stats_df.set_index("cluster_id").to_dict(orient="index")
    sampled_sub = valid_df[valid_df[COL_CLUSTER_ID].isin(sampled_cids)]
    grouped_sampled = {cid: grp for cid, grp in sampled_sub.groupby(COL_CLUSTER_ID, sort=False)}

    cards = []
    for cid in sampled_cids:
        meta = stats_lookup[cid]
        grp = grouped_sampled[cid]

        # Prioritize CORE papers sorted by PageRank/citations for representative examples
        if COL_IS_CORE and COL_IS_CORE in grp.columns:
            core_only = grp[grp[COL_IS_CORE] == True]
            rep_pool = core_only if len(core_only) >= 5 else grp
        else:
            rep_pool = grp

        top_papers_df = rep_pool.sort_values(rank_col, ascending=False).head(5)
        top_papers = []
        for _, row in top_papers_df.iterrows():
            yr = str(row[COL_YEAR]) if (COL_YEAR and COL_YEAR in row and pd.notna(row[COL_YEAR])) else "?"
            clean_t = " ".join(str(row[COL_TITLE]).split())
            top_papers.append({
                "id": str(row.get(COL_ID, "")),
                "year": yr,
                "score": float(row[rank_col]) if pd.notna(row[rank_col]) else 0.0,
                "is_core": bool(row[COL_IS_CORE]) if (COL_IS_CORE and COL_IS_CORE in row) else True,
                "title": clean_t,
            })

        cards.append({
            "cluster_id": cid,
            "audit_buckets": selected_buckets[cid],
            "cluster_title": meta["cluster_title"],
            "title_collision_count": meta["title_collision_count"],
            "size_total": meta["size_total"],
            "size_core": meta["size_core"],
            "size_halo": meta["size_halo"],
            "category_entropy": meta["category_entropy"],
            "top_categories": meta["top_categories"],
            "ctfidf_terms": ctfidf_map.get(cid, []),
            "top_core_papers": top_papers,
        })

    return cards


def format_and_save_cards(cards: list[dict]):
    os.makedirs(os.path.dirname(AUDIT_TXT_OUT), exist_ok=True)
    lines = []

    for idx, c in enumerate(cards, 1):
        cats_str = ", ".join(f"{cat} ({pct}%)" for cat, _, pct in c["top_categories"]) or "N/A"
        terms_str = ", ".join(c["ctfidf_terms"]) or "N/A"
        buckets_str = " | ".join(c["audit_buckets"])

        lines.append("=" * 84)
        lines.append(f"[{idx}/{len(cards)}] CLUSTER {c['cluster_id']}  |  Buckets: {buckets_str}")
        lines.append(f"GENERATED TITLE : {c['cluster_title']}  (Collisions: {c['title_collision_count']})")
        lines.append(
            f"SIZE            : {c['size_total']:,} total "
            f"({c['size_core']:,} core / {c['size_halo']:,} halo)  |  Cat Entropy: {c['category_entropy']}"
        )
        lines.append(f"TOP CATEGORIES  : {cats_str}")
        lines.append(f"c-TF-IDF TERMS  : {terms_str}")
        lines.append("TOP 5 CORE PAPERS (by PageRank / Citations):")
        for p_i, p in enumerate(c["top_core_papers"], 1):
            core_tag = "CORE" if p["is_core"] else "HALO"
            lines.append(f"   {p_i}. [{p['year']}] ({core_tag}, score={p['score']:.2e}) {p['title']}")
        lines.append("")

    report_text = "\n".join(lines)
    print(report_text)

    with open(AUDIT_TXT_OUT, "w", encoding="utf-8") as f:
        f.write(report_text)

    with open(AUDIT_JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(cards, f, indent=2)

    print(f"\nSaved {len(cards)} audit cards to '{AUDIT_TXT_OUT}' and '{AUDIT_JSON_OUT}'.")


if __name__ == "__main__":
    df_clustered = pd.read_csv(CLUSTERED_CSV, low_memory=False)
    audit_cards = build_audit_cards(df_clustered)
    format_and_save_cards(audit_cards)