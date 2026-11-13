import os
import gc
import itertools
import numpy as np
import pandas as pd
import umap.umap_ as umap
import fast_hdbscan

# 1. Output Directory & Scale Setup
# Switch DATASET_SCALE between "287k" (primer) and "3.23M" (full STEM corpus)
DATASET_SCALE = "287k"
OUTPUT_DIR = f"./outputs/grid_{DATASET_SCALE}_macro_micro"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Load Pre-computed SciBERT Embeddings
print(f"Loading embeddings for {DATASET_SCALE} scale...")
EMBEDDINGS_PATH = "scibert_embeddings.npy" if DATASET_SCALE == "287k" else "scibert_embeddings_3.23M.npy"
embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)
n_samples = len(embeddings)

# ==============================================================================
# FULL TARGET MACRO / MICRO GRID SEARCH SPACE
# ==============================================================================
if DATASET_SCALE == "287k":
    UMAP_GRID = [
        {"n_neighbors": 15, "min_dist": 0.01, "label": "TightIslands"},
        {"n_neighbors": 25, "min_dist": 0.03, "label": "BalancedLocal"},
    ]

    # Centered on the 35-46 domain sweet spot with halo noise reclamation
    MACRO_GRID = {
        "min_cluster_size": [650, 950],
        "min_samples": [2, 5],
        "cluster_selection_method": ["leaf"],
        "cluster_selection_epsilon": [0.15, 0.25],
    }

    # Shifted down from 300 to keep micro-topics inside the 150-300 target window
    MICRO_GRID = {
        "min_cluster_size": [130, 180],
        "min_samples": [2, 4],
        "cluster_selection_method": ["leaf"],
        "cluster_selection_epsilon": [0.06, 0.12],
    }
else:
    # ~11.2x scaled parameters for the 3.23M full arXiv STEM corpus
    UMAP_GRID = [
        {"n_neighbors": 30, "min_dist": 0.01, "label": "TightIslands3M"},
        {"n_neighbors": 50, "min_dist": 0.03, "label": "BalancedLocal3M"},
    ]

    MACRO_GRID = {
        "min_cluster_size": [7200, 10500],
        "min_samples": [10, 20],
        "cluster_selection_method": ["leaf"],
        "cluster_selection_epsilon": [0.15, 0.25],
    }

    MICRO_GRID = {
        "min_cluster_size": [1450, 2000],
        "min_samples": [5, 10],
        "cluster_selection_method": ["leaf"],
        "cluster_selection_epsilon": [0.06, 0.12],
    }

macro_combos = [dict(zip(MACRO_GRID.keys(), v)) for v in itertools.product(*MACRO_GRID.values())]
micro_combos = [dict(zip(MICRO_GRID.keys(), v)) for v in itertools.product(*MICRO_GRID.values())]

total_pairs = len(UMAP_GRID) * len(macro_combos) * len(micro_combos)
print(f"Starting Hierarchical Grid Search: {total_pairs} Paired Configurations across {len(UMAP_GRID)} UMAP spaces.")

# ==============================================================================
# EXECUTION LOOP
# ==============================================================================
run_counter = 1
summary_records = []

for u_idx, u_cfg in enumerate(UMAP_GRID, 1):
    print(f"\n" + "=" * 80)
    print(f" UMAP SPACE [{u_idx}/{len(UMAP_GRID)}]: {u_cfg['label']} (n_neighbors={u_cfg['n_neighbors']}, min_dist={u_cfg['min_dist']})")
    print("=" * 80)

    # Cache 2D coordinates so UMAP never needs to be recomputed
    coords_cache = os.path.join(OUTPUT_DIR, f"{u_cfg['label']}_coords2d.npy")
    if os.path.exists(coords_cache):
        print(f"Loading cached 2D coordinates from '{coords_cache}'...")
        umap_coords = np.load(coords_cache)
    else:
        reducer = umap.UMAP(
            n_neighbors=u_cfg['n_neighbors'],
            min_dist=u_cfg['min_dist'],
            n_components=2,
            metric='cosine',
            n_jobs=-1,
            low_memory=True,
        )
        umap_coords = reducer.fit_transform(embeddings).astype(np.float32)
        np.save(coords_cache, umap_coords)
        del reducer
        gc.collect()

    # 1. Fit all Macro configurations globally on this UMAP space
    macro_results = []
    for m_cfg in macro_combos:
        labels = fast_hdbscan.HDBSCAN(
            min_cluster_size=m_cfg['min_cluster_size'],
            min_samples=m_cfg['min_samples'],
            cluster_selection_method=m_cfg['cluster_selection_method'],
            cluster_selection_epsilon=m_cfg['cluster_selection_epsilon']
        ).fit_predict(umap_coords)

        tag = f"mcs{m_cfg['min_cluster_size']}_ms{m_cfg['min_samples']}_eps{m_cfg['cluster_selection_epsilon']}_{m_cfg['cluster_selection_method']}"
        macro_results.append((tag, labels))

    # 2. Fit all Micro configurations globally on this UMAP space
    micro_results = []
    for mi_cfg in micro_combos:
        labels = fast_hdbscan.HDBSCAN(
            min_cluster_size=mi_cfg['min_cluster_size'],
            min_samples=mi_cfg['min_samples'],
            cluster_selection_method=mi_cfg['cluster_selection_method'],
            cluster_selection_epsilon=mi_cfg['cluster_selection_epsilon']
        ).fit_predict(umap_coords)

        tag = f"mcs{mi_cfg['min_cluster_size']}_ms{mi_cfg['min_samples']}_eps{mi_cfg['cluster_selection_epsilon']}_{mi_cfg['cluster_selection_method']}"
        micro_results.append((tag, labels))

    # 3. Pair Macro & Micro labels, enforce parent-child hierarchy, and export CSVs
    for (mac_tag, mac_labels), (mic_tag, mic_labels) in itertools.product(macro_results, micro_results):
        filename = f"{u_cfg['label']}_MACRO_{mac_tag}_MICRO_{mic_tag}.csv"
        filepath = os.path.join(OUTPUT_DIR, filename)

        print(f"[{run_counter}/{total_pairs}] Building {filename}...", end="", flush=True)

        df_out = pd.DataFrame({
            'x_2d': umap_coords[:, 0],
            'y_2d': umap_coords[:, 1],
            'macro_cluster_label': mac_labels,
            'micro_cluster_label': mic_labels
        })

        # Enforce strict Macro -> Micro parent hierarchy via majority vote
        valid = df_out[(df_out['micro_cluster_label'] != -1) & (df_out['macro_cluster_label'] != -1)]
        micro_to_macro = (
            valid.groupby('micro_cluster_label')['macro_cluster_label']
            .agg(lambda s: s.value_counts().index[0])
            .to_dict()
        )
        df_out['assigned_parent_macro'] = (
            df_out['micro_cluster_label'].map(micro_to_macro).fillna(-1).astype(int)
        )

        df_out.to_csv(filepath, index=False)

        n_mac = len(set(mac_labels) - {-1})
        mac_noise = (mac_labels == -1).sum() / n_samples * 100
        n_mic = len(set(mic_labels) - {-1})
        mic_noise = (mic_labels == -1).sum() / n_samples * 100

        summary_records.append({
            "umap_space": u_cfg['label'],
            "macro_config": mac_tag,
            "macro_clusters": n_mac,
            "macro_noise_pct": round(mac_noise, 2),
            "micro_config": mic_tag,
            "micro_clusters": n_mic,
            "micro_noise_pct": round(mic_noise, 2),
            "output_file": filename
        })

        print(f" Done! -> Macro: {n_mac:3d} ({mac_noise:4.1f}% noise) | Micro: {n_mic:3d} ({mic_noise:4.1f}% noise)")
        run_counter += 1

    # Free UMAP and label arrays before moving to the next projection space
    del umap_coords, macro_results, micro_results
    gc.collect()

# Save summary manifest
summary_path = os.path.join(OUTPUT_DIR, f"grid_summary_{DATASET_SCALE}.csv")
pd.DataFrame(summary_records).to_csv(summary_path, index=False)

print("\n" + "=" * 80)
print(f" GRID SEARCH COMPLETE! All {total_pairs} outputs saved to '{OUTPUT_DIR}'.")
print(f" Summary manifest saved to '{summary_path}'.")
print("=" * 80)