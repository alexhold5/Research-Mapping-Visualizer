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
BATCH_SIZE = 128         # Memory-friendly batch size for CPU
MAX_LENGTH = 128        # Truncate sequence length for ~4x faster execution
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
def reduce_dimensions_umap(embeddings, n_neighbors=15, min_dist=0.05, n_components=2):
    """Compress high-dimensional vectors (768D) to 2D manifold coordinates."""
    print("Performing UMAP dimensionality reduction...")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric='cosine',
        n_jobs=-1,
    )
    return reducer.fit_transform(embeddings)

# -----------------------------------------------------------------------------
# 3. CLUSTERING, INFLUENCE SCORING, AND METADATA EXTRACTION
# -----------------------------------------------------------------------------
def compute_hdbscan_clusters(coords_2d, min_cluster_size=300):
    # HELLO LOGAN: RUN FOR EACH METHOD (DON'T SAVE OVER ANYTHING) THANK YOU
    
    """Density-based spatial clustering for unstructured semantic groupings."""
    print("Computing HDBSCAN density-based clusters...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='euclidean',
        cluster_selection_method='leaf',
        core_dist_n_jobs=-1,
        min_samples=5
    )
    return clusterer.fit_predict(coords_2d)

    #clusterer = hdbscan.HDBSCAN(
    #    min_cluster_size=1000,
    #    min_samples=15,
    #    cluster_selection_method='leaf',
    #    prediction_data=True,      # Required for soft clustering / membership queries
    #    metric='euclidean'
    #)
    #labels = clusterer.fit_predict(coords_2d)
    ## 2. Calculate membership probabilities for all points
    #soft_clusters = hdbscan.all_points_membership_vectors(clusterer)
    ## 3. Assign noise points (-1) to their highest-probability cluster
    ## (Optional threshold: e.g., only assign if probability > 0.1)
    #reassigned_labels = labels.copy()
    #for i in range(len(labels)):
    #    if labels[i] == -1:
    #        probs = soft_clusters[i]
    #        if np.max(probs) > 0.05:  # Soft threshold to avoid completely lost outliers
    #            reassigned_labels[i] = np.argmax(probs)
    #return reassigned_labels




def deduplicate_phrases(phrases):
    """
    Strips out singular/plural duplicates (e.g., Decision Tree vs Decision Trees)
    and removes smaller substring fragments (e.g., 'Markov Decision' if 'Decision Process' or 'Markov Decision Process' is present).
    """
    clean_phrases = []
    seen_stems = set()

    for phrase in phrases:
        phrase_lower = phrase.lower().strip()
        # Create a simple normalized stem key (strip trailing 's', 'ing', 'es')
        stem_key = re.sub(r'(ing|es|s)$', '', phrase_lower)
        
        # Check if this phrase is a subset or superset of something already selected
        is_duplicate = False
        for seen in seen_stems:
            if stem_key in seen or seen in stem_key:
                is_duplicate = True
                break
                
        if not is_duplicate:
            seen_stems.add(stem_key)
            clean_phrases.append(phrase.title())
            
    return clean_phrases

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_STOP_WORDS = list(TfidfVectorizer(stop_words='english').get_stop_words())

SINGLE_WORD_STOPS = set(BASE_STOP_WORDS + [
    'paper', 'article', 'based', 'using', 'proposed', 'approach', 'method', 
    'methods', 'results', 'problem', 'problems', 'model', 'models', 'data', 
    'algorithm', 'algorithms', 'study', 'analysis', 'show', 'present', 
    'application', 'propose', 'new', 'important', 'science', 'computer'
])

STOP_PHRASES = {
    'computer science', 'artificial intelligence', 'cognitive science',
    'machine learning', 'information computer', 'state art', 'state of the art',
    'real world', 'domain independent', 'international planning',
    'planning competition', 'propose new', 'symmetry important',
    'simplicity generality', 'objects relations'
}

def generate_cluster_titles(df, text_column='abstract', top_n_keywords=3):
    cluster_titles = {}
    unique_clusters = [c for c in df['cluster_label'].unique() if c != -1]
    
    for cluster_id in unique_clusters:
        cluster_docs = df[df['cluster_label'] == cluster_id][text_column].dropna().tolist()
        
        if not cluster_docs:
            cluster_titles[cluster_id] = f"Cluster {cluster_id}"
            continue
            
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=list(SINGLE_WORD_STOPS),  # Only single words passed here
            ngram_range=(2, 3), 
            min_df=2,
            max_features=200
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(cluster_docs)
            scores = tfidf_matrix.sum(axis=0).A1
            words = vectorizer.get_feature_names_out()
            
            # Sort top candidate phrases
            top_candidate_indices = scores.argsort()[::-1]
            
            # Filter out unwanted multi-word phrases post-vectorization
            raw_keywords = []
            for i in top_candidate_indices:
                phrase = words[i].lower().strip()
                if phrase not in STOP_PHRASES and not any(sp in phrase for sp in STOP_PHRASES):
                    raw_keywords.append(words[i])
                if len(raw_keywords) >= 10:
                    break
            
            final_keywords = deduplicate_phrases(raw_keywords)[:top_n_keywords]
            cluster_titles[cluster_id] = f"Cluster {cluster_id}: " + " / ".join(final_keywords)
            
        except ValueError:
            cluster_titles[cluster_id] = f"Cluster {cluster_id}"
            
    cluster_titles[-1] = "Noise / Unclustered"
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


def calculate_influence_scores(df, citation_col='citations', graph=None):
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

# -----------------------------------------------------------------------------
# MAIN PIPELINE EXECUTION (YOUR MODIFIED VERSION)
# -----------------------------------------------------------------------------
def run_pipeline(csv_path="arXiv_scientific_dataset.csv"):
    # Load dataset
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Subsample if SAMPLE_SIZE is set for fast CPU testing
    if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
        print(f"Subsampling dataset to {SAMPLE_SIZE} rows for quick testing...")
        df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

    t0 = time.perf_counter()
    # Handle missing text gracefully
    title = df['title'].fillna('')
    abstract = df['summary'].fillna('')

    # Concatenate title + abstract for rich contextual representations
    df['text_for_embedding'] = (title + ". " + abstract).str.strip()
    
    # 1. Generate Embeddings
    tokenizer, model = load_scibert()
    t1 = time.perf_counter()
    
    scibert_embeddings = extract_scibert_embeddings(
        df['text_for_embedding'].tolist(),
        tokenizer,
        model
    )
    t2 = time.perf_counter()
    print(f"SciBERT Embedding Generation Time: {t2-t1:.2f}s")

    # 2. Dimensionality Reduction (768D -> 2D)
    spatial_coords_2d = reduce_dimensions_umap(scibert_embeddings)
    df['x_2d'] = spatial_coords_2d[:, 0]
    df['y_2d'] = spatial_coords_2d[:, 1]
    t3 = time.perf_counter()
    print(f"Dimension Reduction with UMAP Time: {t3-t2:.2f}s")

    # 3. Dynamic Clustering & Metainfo
    df['cluster_label'] = compute_hdbscan_clusters(spatial_coords_2d)
    cluster_title_map = generate_cluster_titles(df, text_column='summary', top_n_keywords=3)
    df['cluster_title'] = df['cluster_label'].map(cluster_title_map)

    t4 = time.perf_counter()
    print(f"Clustering with HDBSCAN Time: {t4-t3:.2f}s")
    
    # Optional Influence Score computation
    # G = nx.erdos_renyi_graph(n=len(df), p=0.001, directed=True)
    G = load_graph_from_json("citation_graph_adjacency.json", df)
    df['influence_score'] = calculate_influence_scores(df, citation_col='citations', graph=G)

    # Extract Orthogonal Temporal Metadata for 3D timelines
    if 'updated_date' in df.columns:
        df['parsed_date'] = pd.to_datetime(df['updated_date'], errors='coerce', format='mixed')
        # Fill missing dates gracefully before converting to epoch integers
        valid_dates = df['parsed_date'].dropna()
        df['timestamp'] = (df['parsed_date'].astype('int64') // 10**9).where(df['parsed_date'].notna(), 0)

    t5 = time.perf_counter()
    print(f"Metadata Extraction & Total Pipeline Overhead: {t5-t0:.2f}s")

    # Save enriched output for visualization layers
    output_filename = f"enriched_{csv_path}"
    df.to_csv(output_filename, index=False)
    print(f"\nPipeline Complete! Output saved to: {output_filename}")
    return df


def create_snapshot(df, snapshot_year=None, max_titles_to_label=15, output_image="arxiv_snapshot.png"):
    df_copy = df.copy()
    
    # 1. Clean Title Formatting
    if snapshot_year is not None and str(snapshot_year).lower() != 'none':
        df_copy['parsed_date'] = pd.to_datetime(df_copy['updated_date'], errors='coerce', format='mixed')
        snapshot_df = df_copy[df_copy['parsed_date'].dt.year <= snapshot_year].copy()
        title_str = f"arXiv Landscape Snapshot (Up to {snapshot_year})"
    else:
        snapshot_df = df_copy
        title_str = "arXiv Landscape Snapshot (Full Dataset - 287k Papers)"

    if snapshot_df.empty:
        return

    # 2. Setup Canvas
    plt.figure(figsize=(16, 12))
    
    # 3. Categorical Palette for Clusters
    unique_titles = [t for t in snapshot_df['cluster_title'].unique() if "Noise" not in t]
    color_map = plt.colormaps['tab20'].resampled(max(len(unique_titles), 1))
    color_lookup = {title: color_map(i) for i, title in enumerate(unique_titles)}

    # 4. Plot Clusters using groupby
    for title, c_df in snapshot_df.groupby('cluster_title'):
        if "Noise" in title:
            plt.scatter(
                c_df['x_2d'], c_df['y_2d'],
                c='gray', alpha=0.15, s=5,  # Smaller dots and lower alpha for 287k points
                label=title, rasterized=True
            )
        else:
            sizes = np.clip(c_df['influence_score'] * 80, 10, 300) if 'influence_score' in c_df.columns else 10
            plt.scatter(
                c_df['x_2d'], c_df['y_2d'],
                color=color_lookup.get(title),
                alpha=0.6,
                s=sizes,
                label=title,
                edgecolors='none',          # Removed edgecolors to speed up rendering 287k points
                rasterized=True
            )

    # 5. Clean Landmark Text Annotations (Prevents Label Overlap)
    if 'influence_score' in snapshot_df.columns and max_titles_to_label > 0:
        top_landmarks = snapshot_df.sort_values(by='influence_score', ascending=False).head(max_titles_to_label)
        
        for _, row in top_landmarks.iterrows():
            clean_title = str(row['title']).replace('\n', ' ').strip()
            truncated_title = (clean_title[:28] + '...') if len(clean_title) > 30 else clean_title
            
            plt.annotate(
                truncated_title,
                (row['x_2d'], row['y_2d']),
                fontsize=8,
                fontweight='bold',
                alpha=0.9,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'), # White backdrop box
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=0.5, color='black'),
                xytext=(10, 10),
                textcoords='offset points'
            )

    # 6. Formatting & Export
    plt.title(title_str, fontsize=16, fontweight='bold')
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, frameon=True)
    plt.tight_layout()
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    df = run_pipeline("arXiv_scientific_dataset_with_citations.csv")

    graph_snapshot = create_snapshot(
        df,
        snapshot_year=None,
        max_titles_to_label=30,
        output_image="snapshot.png"
    )