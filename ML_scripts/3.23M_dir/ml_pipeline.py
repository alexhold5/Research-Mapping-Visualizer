import os
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
import umap.umap_ as umap
import hdbscan
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import gc

# -----------------------------------------------------------------------------
# CONFIGURATION & HARDWARE SETUP (CPU QUICK-TEST FRIENDLY)
# -----------------------------------------------------------------------------
MODEL_NAME = "allenai/scibert_scivocab_uncased"
SAMPLE_SIZE = None      # Set to None or len(df) to process full dataset
BATCH_SIZE = 256         # Memory-friendly batch size for CPU
MAX_LENGTH = 256        # Truncate sequence length for ~4x faster execution
CPU_CORES = os.cpu_count() or 4

# Optimize PyTorch CPU threading
torch.set_num_threads(CPU_CORES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Running on device: {DEVICE} (CPU Threads: {CPU_CORES})")

# -----------------------------------------------------------------------------
# 1. REPRESENTATION LEARNING & EMBEDDING GENERATION
# -----------------------------------------------------------------------------
def load_scibert():
    """Load SciBERT model and tokenizer from Hugging Face."""
    print("Loading SciBERT model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    return tokenizer, model

def mean_pooling(model_output, attention_mask):
    """Mean Pooling - Take attention mask into account for correct averaging."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

@torch.inference_mode()
def extract_scibert_embeddings(texts, tokenizer, model, batch_size=BATCH_SIZE):
    """Generates dense contextual vector representations using SciBERT."""
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating SciBERT Embeddings"):
        batch_texts = texts[i : i + batch_size]
        
        encoded_input = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(DEVICE)
        
        model_output = model(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings.append(sentence_embeddings.cpu().numpy())
        
    return np.vstack(embeddings)

# -----------------------------------------------------------------------------
# 2. DIMENSION REDUCTION & SPATIAL MAPPING
# -----------------------------------------------------------------------------
def reduce_dimensions_umap(embeddings, n_neighbors, min_dist):
    """Compress high-dimensional vectors (768D) to 2D manifold coordinates."""
    print("Performing UMAP dimensionality reduction...")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric='cosine',
        n_jobs=-1,
        low_memory=True
    )
    return reducer.fit_transform(embeddings)

# -----------------------------------------------------------------------------
# 3. CLUSTERING, INFLUENCE SCORING, AND METADATA EXTRACTION
# -----------------------------------------------------------------------------
def compute_hdbscan_clusters(coords_2d, min_cluster_size, min_samples, method):    
    """Density-based spatial clustering for unstructured semantic groupings."""
    print("Computing HDBSCAN density-based clusters...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method=method,
        core_dist_n_jobs=14
    )
    return clusterer.fit_predict(coords_2d)

_CLEAN_RE = re.compile(r"https?://\S+|www\.\S+|\\[a-zA-Z]+|_[a-zA-Z0-9]+|\b\d+\b")

_BANNED_PHRASES = {
    "achieves state", "state art", "achieves state art", "experimental results",
    "real world", "future work", "open source", "large scale", "end end",
    "proposed method", "proposed approach", "novel approach", "novel method",
    "extensive experiments", "benchmark datasets", "empirical study",
    "significant improvements", "outperforms existing", "tuned perform",
    "scientific research", "dans les", "et al",
}


def deduplicate_phrases(phrases, top_n=3, max_overlap_ratio=0.5):
    """
    Filters out phrase subsets, supersets, plurals, boilerplate fragments,
    corrupted short tokens, and sliding-window overlapping bigrams.
    """
    selected, selected_stem_sets = [], []

    for phrase in phrases:
        p_clean = phrase.strip().lower()
        if len(p_clean) < 4 or p_clean.isdigit() or p_clean in _BANNED_PHRASES:
            continue

        tokens = p_clean.split()
        # Reject single-letter artifacts (e.g., 'u', 's', 'x') from stripped LaTeX
        if any(len(t) < 2 for t in tokens):
            continue

        stemmed_tokens = {re.sub(r"(ing|ed|es|s)$", "", t) for t in tokens}

        is_redundant = False
        for existing in selected_stem_sets:
            overlap = len(stemmed_tokens & existing)
            min_len = min(len(stemmed_tokens), len(existing))
            if min_len > 0 and (overlap / min_len) >= max_overlap_ratio:
                is_redundant = True
                break

        if not is_redundant:
            selected.append(phrase.title())
            selected_stem_sets.append(stemmed_tokens)
            if len(selected) == top_n:
                break

    return selected


def generate_cluster_titles(
    df,
    cluster_col="cluster_label",
    text_col="summary",
    score_col="influence_score",
    top_n=3,
    cat_col="primary_cat",
    max_cluster_coverage=0.20,
    max_iter=5,
    max_papers_per_cluster=500,
):
    cluster_title_map = {-1: "Noise / Unclustered"}

    base_stops = list(
        set(TfidfVectorizer(stop_words="english").get_stop_words()).union(
            {
                "paper", "papers", "method", "methods", "models", "model", "results",
                "proposed", "propose", "proposes", "using", "used", "based", "approach",
                "show", "shows", "shown", "data", "dataset", "datasets", "experiments",
                "extensive", "code", "available", "github", "achieve", "achieves",
                "outperform", "outperforms", "performance", "perform", "problem",
                "task", "tasks", "framework", "study", "analysis", "novel", "new",
                "dans", "les", "des", "pour", "avec", "sur", "par", "une", "que",
            }
        )
    )

    # Filter noise and pre-sort ONCE by influence_score descending (avoids per-group sorting)
    valid_df = df[df[cluster_col] != -1].dropna(subset=[text_col])
    if score_col in valid_df.columns:
        valid_df = valid_df.sort_values(score_col, ascending=False, kind="mergesort")
    elif "citations_count" in valid_df.columns:
        valid_df = valid_df.sort_values("citations_count", ascending=False, kind="mergesort")

    for cat_name, cat_df in valid_df.groupby(cat_col, sort=False):
        if cat_df.empty:
            continue

        # Already sorted by influence_score descending, so .head() takes the top landmark papers
        sampled = cat_df.groupby(cluster_col, as_index=False, sort=False).head(max_papers_per_cluster)
        cleaned_texts = sampled[text_col].str.replace(_CLEAN_RE, " ", regex=True)

        grouped = (
            cleaned_texts.groupby(sampled[cluster_col], sort=False)
            .apply(" ".join)
            .reset_index(name=text_col)
        )
        num_clusters = len(grouped)

        if num_clusters == 1:
            c_id = int(grouped[cluster_col].iloc[0])
            cluster_title_map[c_id] = f"Cluster {c_id} ({cat_name}): Atomic Category"
            continue

        vec = TfidfVectorizer(
            stop_words=base_stops,
            ngram_range=(2, 3),
            min_df=2,
            max_features=100_000,
        )
        tfidf_matrix = vec.fit_transform(grouped[text_col])
        features = vec.get_feature_names_out()
        disabled_indices = set()

        for _ in range(max_iter):
            candidate_counts = {}
            for idx in range(num_clusters):
                row = tfidf_matrix[idx]
                if row.nnz == 0:
                    continue
                top_idx = row.indices[row.data.argsort()[-20:][::-1]]
                for i in top_idx:
                    if i not in disabled_indices:
                        candidate_counts[i] = candidate_counts.get(i, 0) + 1

            new_disabled = {
                i for i, count in candidate_counts.items()
                if (count / num_clusters) > max_cluster_coverage
            }
            if not new_disabled:
                break
            disabled_indices.update(new_disabled)

        for idx, c_id in enumerate(grouped[cluster_col]):
            c_id = int(c_id)
            row = tfidf_matrix[idx]
            sorted_indices = row.indices[row.data.argsort()[::-1]]
            valid_candidates = [
                features[i] for i in sorted_indices if i not in disabled_indices
            ][:20]

            keywords = deduplicate_phrases(valid_candidates, top_n=top_n)
            cluster_title_map[c_id] = f"Cluster {c_id} ({cat_name}): " + " / ".join(keywords)

    return cluster_title_map


def load_graph_from_json(json_path: str, df) -> nx.DiGraph:
    """
    Loads citation_graph_adjacency.json into a NetworkX DiGraph.
    Includes all dataset papers (including dangling sink nodes) and adds
    in-dataset edges in bulk.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        adj_dict = json.load(f)

    # 1. Collect all valid dataset IDs (includes dangling out-degree=0 nodes)
    dataset_ids = set(df["s2_paper_id"].dropna().astype(str))

    G = nx.DiGraph()
    G.add_nodes_from(dataset_ids)

    # 2. Stream valid edges in bulk via generator (avoids millions of method-call overheads)
    edges = (
        (source_id, target_id)
        for source_id, target_ids in adj_dict.items()
        if source_id in dataset_ids
        for target_id in target_ids
        if target_id in dataset_ids and target_id != source_id
    )
    G.add_edges_from(edges)

    print(
        f"Graph constructed: {G.number_of_nodes():,} nodes, "
        f"{G.number_of_edges():,} in-dataset edges."
    )
    return G

def calculate_influence_scores(df, graph=None, w_citations: float = 0.5, w_pagerank: float = 0.5):
    """
    Computes landmark influence scores combining log-scaled global OpenAlex
    citation volume and log-scaled internal arXiv PageRank centrality.
    """
    n_papers = len(df)
    if n_papers == 0:
        return np.array([])

    # 1. Global Citation Volume Signal (log-scaled + 99.9th percentile cap)
    citations = df["citations_count"].fillna(0).clip(lower=0).values
    log_citations = np.log1p(citations)
    cit_cap = np.percentile(log_citations, 99.9) or 1.0
    norm_citations = np.clip(log_citations / cit_cap, 0.0, 1.0)

    # 2. Internal Network Centrality Signal (log-scaled relative to baseline PR floor)
    if graph is not None and graph.number_of_edges() > 0:
        pr_dict = nx.pagerank(graph, alpha=0.85)
        raw_pr = np.array([pr_dict.get(pid, 0.0) for pid in df["s2_paper_id"]], dtype=np.float64)
        # Multiply by N so baseline dangling/unconnected node PR is ~0.15, then log1p-scale
        log_pr = np.log1p(raw_pr * max(1, graph.number_of_nodes()))
        pr_cap = np.percentile(log_pr, 99.9) or 1.0
        norm_pageranks = np.clip(log_pr / pr_cap, 0.0, 1.0)
    else:
        norm_pageranks = np.zeros(n_papers, dtype=np.float64)

    # 3. Balanced Blend & Final [0.1, 1.0] Scaling
    raw_score = (w_citations * norm_citations) + (w_pagerank * norm_pageranks)
    scaler = MinMaxScaler(feature_range=(0.1, 1.0))
    return scaler.fit_transform(raw_score.reshape(-1, 1)).flatten()

def get_embeddings(df):
    title = df['title'].fillna('')
    abstract = df['summary'].fillna('')

    # Concatenate title + abstract for rich contextual representations
    df['text_for_embedding'] = (title + ". " + abstract).str.strip()

    t1 = time.perf_counter()
    # 1. Generate Embeddings
    tokenizer, model = load_scibert()
    
    scibert_embeddings = extract_scibert_embeddings(
        df['text_for_embedding'].tolist(),
        tokenizer,
        model
    )

    np.save("scibert_embeddings.npy", scibert_embeddings)
    return scibert_embeddings

# -----------------------------------------------------------------------------
# MAIN PIPELINE EXECUTION (YOUR MODIFIED VERSION)
# -----------------------------------------------------------------------------
def run_pipeline(csv_path="arXiv_scientific_dataset.csv", out_file="enriched_arXiv_scientific_dataset.csv", embeddings=None, n_neighbors=15, min_dist=0.05, min_cluster_size=350, min_samples=3, method='eom'):
    t0 = time.perf_counter()
    # Load dataset
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Subsample if SAMPLE_SIZE is set for fast CPU testing
    if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
        print(f"Subsampling dataset to {SAMPLE_SIZE} rows for quick testing...")
        df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

    t1 = time.perf_counter()
    # Handle missing text gracefully
    if embeddings is None:
        scibert_embeddings = get_embeddings(df)
    else:
        scibert_embeddings = np.load("scibert_embeddings.npy")

    t2 = time.perf_counter()
    print(f"SciBERT Embedding Generation Time: {t2-t1:.2f}s")
    # 2. Dimensionality Reduction (768D -> 2D)
    spatial_coords_2d = reduce_dimensions_umap(scibert_embeddings, n_neighbors, min_dist)
    df['x_2d'] = spatial_coords_2d[:, 0]
    df['y_2d'] = spatial_coords_2d[:, 1]
    t3 = time.perf_counter()
    print(f"Dimension Reduction with UMAP Time: {t3-t2:.2f}s")

    # 3. Dynamic Clustering & Metainfo
    df['cluster_label'] = compute_hdbscan_clusters(spatial_coords_2d, min_cluster_size, min_samples, method)
    cluster_title_map = generate_cluster_titles(df, text_column='summary', top_n_keywords=3)
    df['cluster_title'] = df['cluster_label'].map(cluster_title_map)

    t4 = time.perf_counter()
    print(f"Clustering with HDBSCAN Time: {t4-t3:.2f}s")
    
    # Optional Influence Score computation
    # G = nx.erdos_renyi_graph(n=len(df), p=0.001, directed=True)
    G = load_graph_from_json("citation_graph_adjacency.json", df)
    df['influence_score'] = calculate_influence_scores(df, graph=G)

    t5 = time.perf_counter()
    print(f"Metadata Extraction & Total Pipeline Overhead: {t5-t0:.2f}s")

    # Save enriched output for visualization layers
    # output_filename = f"enriched_{csv_path}"
    df.to_csv(out_file, index=False)
    print(f"\nPipeline Complete! Output saved to: {out_file}")
    return df


def run_from_grid_result(grid_result, csv_path="arXiv_scientific_dataset_with_citations.csv", out_file="enriched_arXiv_scientific_dataset.csv"):
    
    # Load dataset
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, dtype={"id": str})
    df_grid = pd.read_csv(grid_result, dtype={"id": str})

    df = df.drop_duplicates(subset=["id"], keep="first")
    df_grid = df_grid.drop_duplicates(subset=["id"], keep="first")

    df = pd.merge(
        df,
        df_grid,
        on="id",
        how="inner",
        validate="one_to_one",  # Use "many_to_one" if df_grid has multi-category duplicates per paper
    )
    del df_grid
    gc.collect()
    print("data written")

    G = load_graph_from_json("data/citation_graph_adjacency_full.json", df)
    df['influence_score'] = calculate_influence_scores(df, graph=G)

    # 3. Dynamic Clustering & Metainfo
    t0 = time.perf_counter()
    cluster_title_map = generate_cluster_titles(df, cluster_col='cluster_label', text_col='summary', top_n=3)
    print(f"Cluster Titling Time: {time.perf_counter() - t0}")
    df['cluster_title'] = df['cluster_label'].map(cluster_title_map)

    # Optional Influence Score computation
    # G = nx.erdos_renyi_graph(n=len(df), p=0.001, directed=True)

    # Save enriched output for visualization layers
    # output_filename = f"enriched_{csv_path}"
    df.to_csv(out_file, index=False)
    print(f"\nPipeline Complete! Output saved to: {out_file}")
    return df


def create_snapshot(
    df,
    snapshot_year=None,
    max_titles_to_label=15,
    max_legend_clusters=25,
    output_image="arxiv_snapshot.png",
    max_plot_points=250_000,
):
    snapshot_df = df.copy()
    if snapshot_df.empty:
        return

    # 1. Select top influential papers BEFORE any subsampling so landmark papers are never dropped
    sort_col = (
        "influence_score"
        if "influence_score" in snapshot_df.columns
        else ("citations_count" if "citations_count" in snapshot_df.columns else None)
    )
    top_papers_df = pd.DataFrame()
    if sort_col and "title" in snapshot_df.columns and max_titles_to_label > 0:
        top_papers_df = (
            snapshot_df.dropna(subset=["title", "x_2d", "y_2d"])
            .sort_values(sort_col, ascending=False)
            .head(max_titles_to_label)
        )

    # 2. Subsample background points for fast rendering & clean PNG file size
    if len(snapshot_df) > max_plot_points:
        snapshot_df = snapshot_df.sample(n=max_plot_points, random_state=42)

    title_str = f"arXiv Landscape Snapshot ({len(df):,} Papers)"

    # 3. Bound legend entries by cluster size so PNG height never exceeds viewer limits
    clustered_df = snapshot_df[snapshot_df["cluster_label"] != -1]
    cluster_counts = clustered_df["cluster_label"].value_counts()
    sorted_labels = sorted(cluster_counts.index.tolist())
    legend_labels = set(cluster_counts.head(max_legend_clusters).index.tolist())

    fig, ax = plt.subplots(figsize=(16, 11), dpi=150)
    color_map = plt.colormaps["tab20"].resampled(20)
    color_lookup = {lbl: color_map(i % 20) for i, lbl in enumerate(sorted_labels)}

    # 4. Plot Noise Points First
    noise_df = snapshot_df[snapshot_df["cluster_label"] == -1]
    if not noise_df.empty:
        ax.scatter(
            noise_df["x_2d"],
            noise_df["y_2d"],
            c="lightgray",
            alpha=0.15,
            s=3,
            label="Noise / Unclustered",
            rasterized=True,
        )

    # 5. Plot Cluster Continents
    for c_id in sorted_labels:
        c_df = clustered_df[clustered_df["cluster_label"] == c_id]

        legend_label = None
        if c_id in legend_labels:
            raw_title = str(c_df["cluster_title"].iloc[0])
            if raw_title.startswith(f"Cluster {c_id}:"):
                raw_title = raw_title.replace(f"Cluster {c_id}:", "").strip()
            if len(raw_title) > 55:
                raw_title = raw_title[:52] + "..."
            legend_label = f"[{c_id}] {raw_title}"

        sizes = (
            np.clip(c_df["influence_score"] * 80, 10, 300)
            if "influence_score" in c_df.columns
            else 6
        )

        ax.scatter(
            c_df["x_2d"],
            c_df["y_2d"],
            color=color_lookup[c_id],
            alpha=0.5,
            s=sizes,
            label=legend_label,
            edgecolors="none",
            rasterized=True,
        )

    # 6. Direct Centroid Badge Placement (for legend_labels so badges stay readable)
    badge_df = clustered_df[clustered_df["cluster_label"].isin(legend_labels)]
    centroids = badge_df.groupby("cluster_label")[["x_2d", "y_2d"]].median().reset_index()
    for _, row in centroids.iterrows():
        c_num = int(row["cluster_label"])
        ax.annotate(
            str(c_num),
            (row["x_2d"], row["y_2d"]),
            fontsize=9.5,
            fontweight="bold",
            color="black",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="circle,pad=0.3",
                facecolor="white",
                edgecolor="black",
                linewidth=1.2,
                alpha=0.85,
            ),
            zorder=10,
        )

    # 7. Annotate Top Influential Paper Titles (controlled by max_titles_to_label)
    for _, prow in top_papers_df.iterrows():
        paper_title = str(prow["title"]).strip().replace("\n", " ")
        if len(paper_title) > 45:
            paper_title = paper_title[:42] + "..."
        ax.scatter(
            [prow["x_2d"]],
            [prow["y_2d"]],
            c="gold",
            edgecolors="black",
            s=90,
            marker="*",
            zorder=12,
        )
        ax.annotate(
            paper_title,
            (prow["x_2d"], prow["y_2d"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            fontweight="semibold",
            color="#111111",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="lightyellow",
                edgecolor="goldenrod",
                alpha=0.9,
            ),
            zorder=13,
        )

    # 8. Formatting & Safe Export
    ax.set_title(title_str, fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("UMAP Dimension 1", fontsize=11)
    ax.set_ylabel("UMAP Dimension 2", fontsize=11)

    ax.legend(
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        fontsize=8.5,
        frameon=True,
        ncol=1,
        title=f"Top {min(max_legend_clusters, len(sorted_labels))} Clusters",
    )

    fig.savefig(output_image, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":

    # full pipeline from scratch
    # df = run_pipeline("arXiv_scientific_dataset_with_citations.csv", embeddings="scibert_embeddings.npy", n_neighbors=15, min_dist=0.01, min_cluster_size=350, min_samples=3, method='eom')

    # pipeline from grid search results
    OUTPUT_DIR = "full_dataset_clustering"
    os.makedirs(f"outputs/{OUTPUT_DIR}", exist_ok=True)
    df = run_from_grid_result(f"outputs/primary_cat_grid_outputs/master_taxonomy_clusters.csv", csv_path=f"outputs/{OUTPUT_DIR}/enriched_arXiv_scientific_full_dataset.csv", out_file=f"outputs/{OUTPUT_DIR}/enriched_arXiv_scientific_full_dataset.csv")
    
    # snapshot only post pipeline
    # df = pd.read_csv(f"outputs/{OUTPUT_DIR}/enriched_arXiv_scientific_dataset.csv")

    graph_snapshot = create_snapshot(
        df,
        max_titles_to_label=15,
        output_image=f"outputs/{OUTPUT_DIR}/snapshot.png"
    )