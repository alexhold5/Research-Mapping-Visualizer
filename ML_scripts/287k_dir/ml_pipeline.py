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


# Single-word stop list (passed directly to TfidfVectorizer)
SINGLE_STOP_WORDS = list(set(TfidfVectorizer(stop_words='english').get_stop_words()).union({
    'paper', 'method', 'methods', 'models', 'model', 'results', 
    'proposed', 'using', 'based', 'approach',
    # Hard-code root noise tokens as single-word safeguards
    'github', 'com', 'cifar', '10', '95', 'ci', 'et', 'al'
}))

PHRASE_STOP_WORDS = [
    # General Domain / Noise Phrases
    'state art', 'state of the art', 'real world', 'training data',
    'deep learning', 'neural network', 'neural networks', 'machine learning', 
    'artificial intelligence', 'time series', 'et al', 'extensive experiments', 
    '95 ci',
    
    # Flexible Regex matches for URLs, domain artifacts & datasets
    r'github(?:\.|\s+)com',    # Matches "github.com", "github com", "github/com"
    r'cifar(?:\-|\s*)10',       # Matches "cifar-10", "cifar 10", "cifar10"
    r'http[s]?://\S+',         # Wipes raw URLs entirely
]

# Compile pattern with re.IGNORECASE
PHRASE_REGEX = re.compile(r'\b(' + '|'.join(PHRASE_STOP_WORDS) + r')\b', flags=re.IGNORECASE)

def deduplicate_phrases(phrases, top_n=3):
    """Filters out phrase subsets, plurals, and corrupted short tokens."""
    selected, selected_stem_sets = [], []
    
    for phrase in phrases:
        if len(phrase) < 4 or phrase.isdigit():
            continue
            
        tokens = phrase.lower().split()
        stemmed_tokens = {re.sub(r'(ing|es|s)$', '', t) for t in tokens}
        
        is_subset = any(stemmed_tokens.issubset(existing) for existing in selected_stem_sets)
        is_superset = any(existing.issubset(stemmed_tokens) for existing in selected_stem_sets)
        
        if not is_subset and not is_superset:
            selected.append(phrase.title())
            selected_stem_sets.append(stemmed_tokens)
            if len(selected) == top_n:
                break
                
    return selected

# def generate_cluster_titles(df, cluster_col='cluster_label', text_col='abstract', top_n=3):
#     # 3. Filter out noise (-1) and missing abstracts
#     clean_df = df[df[cluster_col] != -1].dropna(subset=[text_col])
#     grouped = clean_df.groupby(cluster_col)[text_col].apply(" ".join).reset_index()
    
#     # 4. Strip multi-word phrase stop words from raw grouped strings
#     cleaned_text = grouped[text_col].apply(lambda txt: PHRASE_REGEX.sub('', txt))
    
#     # 5. Single-pass c-TF-IDF (No warnings!)
#     vectorizer = TfidfVectorizer(
#         stop_words=SINGLE_STOP_WORDS,
#         ngram_range=(2, 3),
#         min_df=1
#     )
    
#     tfidf_matrix = vectorizer.fit_transform(cleaned_text)
#     feature_names = vectorizer.get_feature_names_out()
    
#     # 6. Extract clean cluster titles
#     cluster_titles = {-1: "Noise / Unclustered"}
#     for idx, c_id in enumerate(grouped[cluster_col]):
#         top_indices = tfidf_matrix[idx].toarray()[0].argsort()[-15:][::-1]
#         candidates = [feature_names[i] for i in top_indices]
        
#         keywords = deduplicate_phrases(candidates, top_n=top_n)
#         cluster_titles[c_id] = f"Cluster {c_id}: " + " / ".join(keywords)
        
#     return cluster_titles

def generate_cluster_titles(df, cluster_col='cluster_label', text_col='summary', 
                           top_n=3, max_cluster_coverage=0.15, max_iter=5):
    
    # 1. Filter out noise (-1) and missing text
    clean_df = df[df[cluster_col] != -1].dropna(subset=[text_col])
    grouped = clean_df.groupby(cluster_col)[text_col].apply(" ".join).reset_index()
    num_clusters = len(grouped)

    # 2. Base single-word stop list
    single_stop_words = list(set(TfidfVectorizer(stop_words='english').get_stop_words()).union({
        'paper', 'method', 'models', 'results', 'proposed', 'using', 'based', 'approach',
        "publicly", "available", "future", "research", "existing", "methods", "code", "available", "artificial", "intelligence", "ai", "systems", "high", "quality", "short", "term", "data", "driven"
    }))
    
    # 3. Fit vectorizer ONCE (Fast: 1 single pass)
    vec = TfidfVectorizer(stop_words=single_stop_words, ngram_range=(2, 3), min_df=1)
    tfidf_matrix = vec.fit_transform(grouped[text_col])
    features = vec.get_feature_names_out()
    
    # Track disabled feature indices dynamically (No Regex needed)
    disabled_indices = set()
    
    # 4. Fast In-Memory Index Sweeping Loop
    for iteration in range(max_iter):
        candidate_counts = {}
        
        for idx in range(num_clusters):
            # Get top 20 candidate feature indices for this cluster
            row = tfidf_matrix[idx].tocsr()
            top_idx = row.indices[row.data.argsort()[-20:][::-1]]
            
            for i in top_idx:
                if i not in disabled_indices:
                    candidate_counts[i] = candidate_counts.get(i, 0) + 1
                    
        # Identify indices appearing in > max_cluster_coverage % of clusters
        new_disabled = {
            i for i, count in candidate_counts.items() 
            if (count / num_clusters) > max_cluster_coverage
        }
        
        if not new_disabled:
            print(f"Fast solver converged at iteration {iteration + 1}!")
            break
            
        print(f"Iteration {iteration + 1}: Discovered {len(new_disabled)} redundant phrases -> {[features[i] for i in new_disabled]}")
        disabled_indices.update(new_disabled)
        
    # 5. Final Title Extraction (Filtering out disabled indices)
    cluster_titles = {-1: "Noise / Unclustered"}
    for idx, c_id in enumerate(grouped[cluster_col]):
        row = tfidf_matrix[idx].tocsr()
        # Sort top indices and exclude disabled ones
        sorted_indices = row.indices[row.data.argsort()[::-1]]
        valid_candidates = [features[i] for i in sorted_indices if i not in disabled_indices][:15]
        
        keywords = deduplicate_phrases(valid_candidates, top_n=top_n)
        cluster_titles[c_id] = f"Cluster {c_id}: " + " / ".join(keywords)
        
    return cluster_titles


def load_graph_from_json(json_path, df):
    """
    Loads citation_graph_adjacency.json into a NetworkX DiGraph.
    Matches arXiv-to-arXiv edges directly against dataset papers.
    """
    with open(json_path, 'r') as f:
        adj_dict = json.load(f)

    G = nx.DiGraph()
    
    # 1. Collect all valid dataset IDs
    dataset_ids = set(df['s2_paper_id'].dropna().astype(str))
    
    # Add all dataset papers as nodes
    G.add_nodes_from(dataset_ids)

    # 2. Build edges matching source and target arXiv IDs
    edge_count = 0
    for source_id, target_ids in adj_dict.items():
        if source_id in dataset_ids:
            for target_id in target_ids:
                if target_id in dataset_ids:
                    G.add_edge(source_id, target_id)
                    edge_count += 1

    print(f"Graph constructed: {G.number_of_nodes()} nodes, {edge_count} in-dataset edges.")
    return G


def calculate_influence_scores(df, graph=None):
    """
    Computes final landmark influence scores combining 
    log-scaled citation counts and network PageRank centrality.
    """
    # 1. Citation Volume Signal
    citations = df['citations_count'].fillna(0).values
    log_citations = np.log1p(citations)
    
    # 2. Graph Centrality Signal (PageRank)
    if len(graph.edges) > 0:
        pr_dict = nx.pagerank(graph, alpha=0.85)
        pageranks = np.array([pr_dict.get(pid, 0.0) for pid in df['s2_paper_id']])
    else:
        pageranks = np.zeros(len(df))
        
    # 3. Combine Signals (50% Citation volume, 50% PageRank)
    scaler = MinMaxScaler(feature_range=(0.1, 1.0))
    
    # Normalize components individually before combining
    norm_citations = MinMaxScaler().fit_transform(log_citations.reshape(-1, 1)).flatten()
    norm_pageranks = MinMaxScaler().fit_transform(pageranks.reshape(-1, 1)).flatten() if len(graph.edges) > 0 else np.zeros(len(df))
    
    raw_score = (0.5 * norm_citations) + (0.5 * norm_pageranks)
    
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
    # Load dataset
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Subsample if SAMPLE_SIZE is set for fast CPU testing
    if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
        print(f"Subsampling dataset to {SAMPLE_SIZE} rows for quick testing...")
        df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

    t0 = time.perf_counter()
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
    print("data written")

    # 3. Dynamic Clustering & Metainfo
    t0 = time.perf_counter()
    cluster_title_map = generate_cluster_titles(df, cluster_col='cluster_label', text_col='summary', top_n=3)
    print(f"Cluster Titling Time: {time.perf_counter() - t0}")
    df['cluster_title'] = df['cluster_label'].map(cluster_title_map)

    # Optional Influence Score computation
    # G = nx.erdos_renyi_graph(n=len(df), p=0.001, directed=True)
    # G = load_graph_from_json("citation_graph_adjacency.json", df)
    # df['influence_score'] = calculate_influence_scores(df, graph=G)


    # Save enriched output for visualization layers
    # output_filename = f"enriched_{csv_path}"
    df.to_csv(out_file, index=False)
    print(f"\nPipeline Complete! Output saved to: {out_file}")
    return df

def create_snapshot(df, snapshot_year=None, max_titles_to_label=15, output_image="arxiv_snapshot.png"):
    snapshot_df = df.copy()
    title_str = "arXiv Landscape Snapshot (Full Dataset - 287k Papers)"

    if snapshot_df.empty:
        return

    # 2. Extract and Sort Unique Clustered Labels
    clustered_df = snapshot_df[snapshot_df['cluster_label'] != -1]
    sorted_labels = sorted(clustered_df['cluster_label'].unique())

    # 3. Setup Canvas & Color Map
    plt.figure(figsize=(16, 12))
    color_map = plt.colormaps['tab20'].resampled(max(len(sorted_labels), 1))
    color_lookup = {lbl: color_map(i % 20) for i, lbl in enumerate(sorted_labels)}

    # 4. Plot Noise Points First
    noise_df = snapshot_df[snapshot_df['cluster_label'] == -1]
    if not noise_df.empty:
        plt.scatter(
            noise_df['x_2d'], noise_df['y_2d'],
            c='lightgray', alpha=0.15, s=3,
            label="Noise / Unclustered", rasterized=True
        )

    # 5. Plot Cluster Density Continents using EXACT dataframe titles
    for c_id in sorted_labels:
        c_df = snapshot_df[snapshot_df['cluster_label'] == c_id]
        
        # Pull raw title directly without regex mutation
        raw_title = str(c_df['cluster_title'].iloc[0])
        
        # Safely ensure label format is clean: "[c_id] Title"
        if raw_title.startswith(f"Cluster {c_id}:"):
            raw_title = raw_title.replace(f"Cluster {c_id}:", "").strip()
            
        legend_label = f"[{c_id}] {raw_title}"
        sizes = np.clip(c_df['influence_score'] * 80, 10, 300) if 'influence_score' in c_df.columns else 8
        
        plt.scatter(
            c_df['x_2d'], c_df['y_2d'],
            color=color_lookup[c_id],
            alpha=0.5,
            s=sizes,
            label=legend_label,
            edgecolors='none',
            rasterized=True
        )

    # 6. Direct Centroid Badge Placement
    centroids = clustered_df.groupby('cluster_label')[['x_2d', 'y_2d']].median().reset_index()
    for _, row in centroids.iterrows():
        c_num = int(row['cluster_label'])
        plt.annotate(
            str(c_num),
            (row['x_2d'], row['y_2d']),
            fontsize=9.5,
            fontweight='bold',
            color='black',
            ha='center',
            va='center',
            bbox=dict(
                boxstyle='circle,pad=0.3',
                facecolor='white',
                edgecolor='black',
                linewidth=1.2,
                alpha=0.85
            ),
            zorder=10
        )

    # 7. Formatting & Export
    plt.title(title_str, fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("UMAP Dimension 1", fontsize=11)
    plt.ylabel("UMAP Dimension 2", fontsize=11)
    
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8.5, frameon=True, ncol=1)
    plt.tight_layout()
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":

    # full pipeline from scratch
    # df = run_pipeline("arXiv_scientific_dataset_with_citations.csv", embeddings="scibert_embeddings.npy", n_neighbors=15, min_dist=0.01, min_cluster_size=350, min_samples=3, method='eom')

    # pipeline from grid search results
    OUTPUT_DIR = "cs_dataset_clustering2"
    os.makedirs(f"outputs/{OUTPUT_DIR}", exist_ok=True)
    df = run_from_grid_result(f"outputs/global_grid_outputs/TightIslands_r0.0030_mcs862_ms5_leaf_p0.90.csv", csv_path="data/arXiv_scientific_dataset_with_citations.csv", out_file=f"outputs/{OUTPUT_DIR}/enriched_arXiv_scientific_dataset.csv")
    
    # snapshot only post pipeline
    # df = pd.read_csv(f"outputs/{OUTPUT_DIR}/enriched_arXiv_scientific_dataset.csv")

    graph_snapshot = create_snapshot(
        df,
        max_titles_to_label=15,
        output_image=f"outputs/{OUTPUT_DIR}/snapshot.png"
    )