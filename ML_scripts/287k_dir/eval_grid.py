import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_OUTPUT_DIR = "./outputs/global_grid_outputs"
SUMMARY_CSV = os.path.join(BASE_OUTPUT_DIR, "master_grid_summary.csv")
METADATA_CSV = "data/arXiv_scientific_dataset_with_citations.csv"          # Must contain 'id', 'title', 'summary'
EMBEDDINGS_NPY = "scibert_embeddings.npy"
S2_ADJ_JSON = "data/citation_graph_adjacency.json"         # Optional: omit or set None if not ready yet

# Weights for composite ranking (all metrics min-max normalized across candidates)
WEIGHTS = {
    "sil_768d": 0.30,
    "tfidf_distinctness": 0.30,
    "citation_lift": 0.25,
    "size_entropy": 0.15,
}

print("Loading metadata, embeddings (mmap), and optional citation graph...")
df_meta = pd.read_csv(METADATA_CSV, dtype={"id": str}, usecols=["id", "title", "summary"])
df_meta["id"] = df_meta["id"].str.strip()
df_meta["text"] = (df_meta["title"].fillna("") + " " + df_meta["summary"].fillna("")).str.strip()
id_to_row = pd.Series(df_meta.index.values, index=df_meta["id"]).to_dict()

embeddings_mmap = np.load(EMBEDDINGS_NPY, mmap_mode="r")

citation_adj = {}
if S2_ADJ_JSON and os.path.exists(S2_ADJ_JSON):
    with open(S2_ADJ_JSON, "r", encoding="utf-8") as f:
        citation_adj = json.load(f)
    print(f"Loaded citation adjacency for {len(citation_adj):,} papers.")


def compute_tfidf_distinctness(texts: list[str], labels: np.ndarray, top_k: int = 10) -> float:
    """Measures 1 - mean pairwise Jaccard overlap of top-k TF-IDF keywords across clusters."""
    valid_mask = labels != -1
    if valid_mask.sum() < 50:
        return 0.0
    df_tmp = pd.DataFrame({"text": np.array(texts)[valid_mask], "label": labels[valid_mask]})
    grouped_docs = df_tmp.groupby("label")["text"].apply(lambda s: " ".join(s.iloc[:2000]))  # Cap per cluster for speed
    if len(grouped_docs) < 2:
        return 0.0

    vec = TfidfVectorizer(max_features=15000, stop_words="english", ngram_range=(1, 2))
    tfidf_mat = vec.fit_transform(grouped_docs.tolist())
    vocab = np.array(vec.get_feature_names_out())

    top_sets = []
    for row_idx in range(tfidf_mat.shape[0]):
        row_arr = tfidf_mat[row_idx].toarray().ravel()
        top_idx = np.argpartition(row_arr, -top_k)[-top_k:]
        top_sets.append(set(vocab[top_idx]))

    overlaps = []
    for i in range(len(top_sets)):
        for j in range(i + 1, len(top_sets)):
            union = len(top_sets[i] | top_sets[j])
            overlaps.append(len(top_sets[i] & top_sets[j]) / union if union else 0.0)
    return float(1.0 - np.mean(overlaps)) if overlaps else 0.0


def compute_citation_lift(paper_ids: np.ndarray, labels: np.ndarray) -> float:
    """Computes observed intra-cluster citation rate divided by random-assignment expectation."""
    if not citation_adj:
        return 0.0
    valid = labels != -1
    pid_to_cluster = dict(zip(paper_ids[valid], labels[valid]))
    if len(pid_to_cluster) < 50:
        return 0.0

    # Expected random intra-cluster hit rate = sum(p_c^2)
    counts = pd.Series(labels[valid]).value_counts(normalize=True)
    expected_random = float((counts ** 2).sum())
    if expected_random <= 0:
        return 0.0

    same_cluster_edges = 0
    total_internal_edges = 0
    for pid, c_src in pid_to_cluster.items():
        clean_key = f"ARXIV:{pid}" if not pid.startswith("ARXIV:") else pid
        refs = citation_adj.get(clean_key, [])
        for ref in refs:
            raw_ref = ref.replace("ARXIV:", "")
            c_dst = pid_to_cluster.get(raw_ref)
            if c_dst is not None:
                total_internal_edges += 1
                if c_src == c_dst:
                    same_cluster_edges += 1

    if total_internal_edges < 25:
        return 0.0
    observed_rate = same_cluster_edges / total_internal_edges
    return float(observed_rate / expected_random)


def compute_size_entropy(labels: np.ndarray) -> float:
    valid = labels[labels != -1]
    probs = pd.Series(valid).value_counts(normalize=True).to_numpy()
    if len(probs) <= 1:
        return 0.0
    return float(-np.sum(probs * np.log(probs)) / np.log(len(probs)))


# ==============================================================================
# EVALUATE ALL GRID RUNS DIRECTLY
# ==============================================================================
summary_df = pd.read_csv(SUMMARY_CSV)
print(f"\nEvaluating {len(summary_df)} global grid candidates...")
eval_results = []

for _, row in summary_df.iterrows():
    csv_path = os.path.join(BASE_OUTPUT_DIR, row["file"])
    df_run = pd.read_csv(csv_path, dtype={"id": str})
    labels = df_run["cluster_label"].to_numpy()
    pids = df_run["id"].str.strip().to_numpy()

    valid_mask = labels != -1
    n_clusters = len(set(labels[valid_mask]))
    noise_pct = (~valid_mask).mean() * 100.0

    # Stage 1 Guardrails
    max_share = pd.Series(labels[valid_mask]).value_counts(normalize=True).max() if n_clusters > 0 else 1.0
    passed_guardrails = (
        (15 <= n_clusters <= 500)
        and (5.0 <= noise_pct <= 25.0)
        and (max_share <= 0.25)
    )

    # Sample up to 10,000 assigned papers for fast 768D Cosine Silhouette
    valid_idx = np.where(valid_mask)[0]
    if n_clusters >= 2 and len(valid_idx) >= 100:
        rng = np.random.default_rng(42)
        sample_local = rng.choice(valid_idx, size=min(10_000, len(valid_idx)), replace=False)
        global_rows = [id_to_row[pids[i]] for i in sample_local]
        emb_sample = normalize(np.array(embeddings_mmap[global_rows], dtype=np.float32))
        sil_768d = float(silhouette_score(emb_sample, labels[sample_local], metric="cosine"))
    else:
        sil_768d = -1.0

    texts = [df_meta["text"].iat[id_to_row[pid]] for pid in pids]
    tfidf_dist = compute_tfidf_distinctness(texts, labels)
    cit_lift = compute_citation_lift(pids, labels)
    ent = compute_size_entropy(labels)

    res = row.to_dict()
    res.update({
        "passed_guardrails": passed_guardrails,
        "max_cluster_share": round(float(max_share), 4),
        "sil_768d": round(sil_768d, 4),
        "tfidf_distinctness": round(tfidf_dist, 4),
        "citation_lift": round(cit_lift, 4),
        "size_entropy": round(ent, 4),
    })
    eval_results.append(res)

full_eval_df = pd.DataFrame(eval_results)

# Min-max normalize metrics across candidates to compute composite score
pool = full_eval_df[full_eval_df["passed_guardrails"]].copy()
if pool.empty:
    pool = full_eval_df.copy()  # Fallback if all candidates slightly missed guardrails

score = np.zeros(len(pool), dtype=float)
for col, w in WEIGHTS.items():
    vals = pool[col].to_numpy(dtype=float)
    span = vals.max() - vals.min()
    norm_vals = (vals - vals.min()) / span if span > 1e-9 else np.ones_like(vals) * 0.5
    score += w * norm_vals

pool["composite_score"] = np.round(score, 4)
full_eval_df = full_eval_df.merge(pool[["file", "composite_score"]], on="file", how="left")
full_eval_df["composite_score"] = full_eval_df["composite_score"].fillna(-1.0)
full_eval_df = full_eval_df.sort_values("composite_score", ascending=False).reset_index(drop=True)

full_eval_df.to_csv(os.path.join(BASE_OUTPUT_DIR, "evaluated_grid_summary.csv"), index=False)
full_eval_df.head(1).to_csv(os.path.join(BASE_OUTPUT_DIR, "best_global_config.csv"), index=False)

print("\n=== RANKED GLOBAL GRID CANDIDATES ===")
display_cols = [
    "file", "n_clusters", "final_noise_pct", "max_cluster_share",
    "sil_768d", "tfidf_distinctness", "size_entropy", "passed_guardrails", "composite_score"
]
print(full_eval_df[display_cols].to_string(index=False))
print("\nSaved full evaluation to 'evaluated_grid_summary.csv' and top winner to 'best_global_config.csv'.")