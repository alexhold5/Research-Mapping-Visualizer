import os
import gc
import itertools
import numpy as np
import pandas as pd
import umap.umap_ as umap
import hdbscan

# 1. Output Directory Setup
OUTPUT_DIR = "./outputs/cluster_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Load Pre-computed SciBERT Embeddings & Metadata
print("Loading embeddings and metadata...")
# Replace with your actual file paths
embeddings = np.load("scibert_embeddings.npy").astype(np.float32) 
# df_meta = pd.read_csv("arxiv_cleaned_metadata.csv")

# ==============================================================================
# FULL TARGET GRID SEARCH SPACE
# ==============================================================================
# Target: Resolve 70% noise vs. 3-blob collapse by tuning density strictness
UMAP_GRID = [
    {"n_neighbors": 15, "min_dist": 0.01, "label": "TightIslands"},
    {"n_neighbors": 25, "min_dist": 0.04, "label": "BalancedLocal"},
    {"n_neighbors": 40, "min_dist": 0.08, "label": "MacroContinuum"}
]

HDBSCAN_GRID = {
    "min_cluster_size": [150, 250, 350, 500],
    "min_samples": [3, 5, 8],                  # Crucial drop from 10-15 to fix high noise
    "cluster_selection_method": ["leaf", "eom"]  # Testing EOM to prevent blob collapse
}

# Generate all HDBSCAN parameter combinations (4 x 3 x 2 = 24 configs per UMAP space)
hdb_combos = [
    dict(zip(HDBSCAN_GRID.keys(), v)) 
    for v in itertools.product(*HDBSCAN_GRID.values())
]

total_runs = len(UMAP_GRID) * len(hdb_combos)
print(f"Starting Full Grid Search: {total_runs} Total Configurations across {len(UMAP_GRID)} UMAP spaces.")

# ==============================================================================
# EXECUTION LOOP
# ==============================================================================
run_counter = 1

for u_idx, u_cfg in enumerate(UMAP_GRID, 1):
    print(f"\n" + "="*80)
    print(f" UMAP SPACE [{u_idx}/{len(UMAP_GRID)}]: {u_cfg['label']} (n_neighbors={u_cfg['n_neighbors']}, min_dist={u_cfg['min_dist']})")
    print("="*80)
    
    # Run UMAP once per spatial configuration
    reducer = umap.UMAP(
        n_neighbors=u_cfg['n_neighbors'],
        min_dist=u_cfg['min_dist'],
        n_components=2,
        metric='cosine',
        n_jobs=-1,
        low_memory=True,
    )
    umap_coords = reducer.fit_transform(embeddings).astype(np.float32)
    
    # Iterate over all HDBSCAN parameter targets for this UMAP projection
    for h_cfg in hdb_combos:
        mcs = h_cfg['min_cluster_size']
        ms = h_cfg['min_samples']
        method = h_cfg['cluster_selection_method']
        
        filename = f"{u_cfg['label']}_mcs{mcs}_ms{ms}_{method}.csv"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        print(f"[{run_counter}/{total_runs}] Fitting HDBSCAN (mcs={mcs}, ms={ms}, method='{method}')...", end="", flush=True)
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            cluster_selection_method=method,
            metric='euclidean',
            core_dist_n_jobs=14
        )
        labels = clusterer.fit_predict(umap_coords)
        
        # Build lightweight result DataFrame for Stage 1 / Stage 2 batch evaluation
        df_out = pd.DataFrame({
            'x_2d': umap_coords[:, 0],
            'y_2d': umap_coords[:, 1],
            'cluster_label': labels
        })
        
        # Save to CSV
        df_out.to_csv(filepath, index=False)
        
        # Print run summary inline
        n_clusters = len(set(labels) - {-1})
        noise_pct = (labels == -1).sum() / len(labels) * 100
        print(f" Done! -> Clusters: {n_clusters:3d} | Noise: {noise_pct:5.1f}% | Saved: {filename}")
        
        run_counter += 1
        
    # Free UMAP memory before moving to the next projection space
    del reducer, umap_coords
    gc.collect()

print("\n" + "="*80)
print(f" GRID SEARCH COMPLETE! All {total_runs} outputs saved to '{OUTPUT_DIR}'.")
print(" Run your batch evaluation harness on './cluster_outputs' to inspect results.")
print("="*80)