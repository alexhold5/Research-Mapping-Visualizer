import os
import gc
import itertools
import numpy as np
import pandas as pd
import umap.umap_ as umap
import hdbscan
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier

# ==============================================================================
# 1. CONFIGURATION & FILE PATHS
# ==============================================================================
EMBEDDINGS_PATH = "scibert_embeddings.npy"
METADATA_PATH = "data/arXiv_scientific_dataset_with_citations.csv"
BASE_OUTPUT_DIR = "./outputs/global_grid_outputs"

# Target noise & granularity window
TARGET_NOISE_PCT = 15.0
NOISE_BOUNDS = (10.0, 20.0)
MIN_CONFIDENCE_FLOOR = 0.40  # Never assign a noise point if top class prob < 0.40

os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)


def reclaim_halo_noise(umap_coords: np.ndarray, core_labels: np.ndarray, k: int = 25):
    """
    Reassigns high-confidence -1 boundary points to nearest core clusters using
    distance-weighted k-NN probabilities, tuning the probability threshold to hit
    ~15% final noise (within 10-20%).
    """
    core_mask = core_labels != -1
    noise_mask = ~core_mask
    n_total = len(core_labels)
    initial_noise_pct = noise_mask.mean() * 100.0

    # If already within/below target noise or fewer than 2 cores, return as-is
    unique_cores = np.unique(core_labels[core_mask])
    if len(unique_cores) < 2 or initial_noise_pct <= NOISE_BOUNDS[1]:
        return core_labels.copy(), 1.0, initial_noise_pct

    eff_k = min(k, core_mask.sum())
    knn = KNeighborsClassifier(n_neighbors=eff_k, weights="distance", n_jobs=-1)
    knn.fit(umap_coords[core_mask], core_labels[core_mask])

    probs = knn.predict_proba(umap_coords[noise_mask])
    best_class_idx = np.argmax(probs, axis=1)
    best_probs = probs[np.arange(len(probs)), best_class_idx]
    predicted_classes = knn.classes_[best_class_idx]
    noise_indices = np.where(noise_mask)[0]

    # Sweep candidate probability thresholds to hit closest to TARGET_NOISE_PCT (15%)
    candidate_thresholds = np.linspace(MIN_CONFIDENCE_FLOOR, 0.90, 51)
    best_thresh = MIN_CONFIDENCE_FLOOR
    best_diff = float("inf")

    for thresh in candidate_thresholds:
        remaining_noise = (best_probs < thresh).sum()
        resulting_noise_pct = (remaining_noise / n_total) * 100.0
        diff = abs(resulting_noise_pct - TARGET_NOISE_PCT)
        if diff < best_diff:
            best_diff = diff
            best_thresh = float(thresh)

    final_labels = core_labels.copy()
    confident_mask = best_probs >= best_thresh
    final_labels[noise_indices[confident_mask]] = predicted_classes[confident_mask]
    final_noise_pct = (final_labels == -1).mean() * 100.0

    return final_labels, round(best_thresh, 3), final_noise_pct


# ==============================================================================
# 2. LOAD EMBEDDINGS & IDS DIRECTLY
# ==============================================================================
print("Loading embeddings (mmap) and IDs...")
embeddings_mmap = np.load(EMBEDDINGS_PATH, mmap_mode="r")
df_meta = pd.read_csv(METADATA_PATH, dtype={"id": str}, usecols=["id"])

assert len(embeddings_mmap) == len(df_meta), (
    f"Row mismatch: embeddings ({len(embeddings_mmap):,}) != metadata ({len(df_meta):,})"
)

n_samples = len(df_meta)
target_ids = df_meta["id"].str.strip().to_numpy()
target_embeddings = normalize(np.array(embeddings_mmap, dtype=np.float32))

# ==============================================================================
# 3. PRUNED GRID SEARCH SPACE (LEAF-ONLY)
# ==============================================================================
UMAP_GRID = [
    {"n_neighbors": 15, "min_dist": 0.01, "label": "TightIslands"},
    {"n_neighbors": 25, "min_dist": 0.04, "label": "BalancedLocal"},
]

HDBSCAN_GRID = {
    "mcs_ratio": [0.0022, 0.0030, 0.0038],
    "min_samples": [3, 5],
    "cluster_selection_method": ["leaf"],
}

hdb_combos = [
    dict(zip(HDBSCAN_GRID.keys(), v))
    for v in itertools.product(*HDBSCAN_GRID.values())
]

print(f"Running {len(UMAP_GRID) * len(hdb_combos)} configs across {n_samples:,} records...")

# ==============================================================================
# 4. GRID SEARCH LOOP
# ==============================================================================
summary_records = []

print("\n" + "#" * 80)
print(f" GRID SEARCH: {n_samples:,} records")
print("#" * 80)

for u_cfg in UMAP_GRID:
    eff_neighbors = min(u_cfg["n_neighbors"], max(5, n_samples - 1))
    coords_cache_path = os.path.join(BASE_OUTPUT_DIR, f"{u_cfg['label']}_coords2d.npy")

    if os.path.exists(coords_cache_path):
        umap_coords = np.load(coords_cache_path)
    else:
        print(f"  Computing UMAP {u_cfg['label']} (n_neighbors={eff_neighbors})...", flush=True)
        reducer = umap.UMAP(
            n_neighbors=eff_neighbors,
            min_dist=u_cfg["min_dist"],
            n_components=2,
            metric="cosine",
            low_memory=True,  # Omitting random_state enables full multi-core UMAP speed
        )
        umap_coords = reducer.fit_transform(target_embeddings).astype(np.float32)
        np.save(coords_cache_path, umap_coords)
        del reducer
        gc.collect()

    for h_cfg in hdb_combos:
        ratio = h_cfg["mcs_ratio"]
        mcs = max(25, int(round(n_samples * ratio)))
        ms = min(h_cfg["min_samples"], mcs)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            cluster_selection_method="leaf",
            metric="euclidean",
            core_dist_n_jobs=14,
        )
        core_labels = clusterer.fit_predict(umap_coords)
        core_noise_pct = (core_labels == -1).mean() * 100.0

        # Reclaim boundary noise using distance-weighted k-NN probabilities
        final_labels, used_thresh, final_noise_pct = reclaim_halo_noise(
            umap_coords, core_labels, k=25
        )

        filename = f"{u_cfg['label']}_r{ratio:.4f}_mcs{mcs}_ms{ms}_leaf_p{used_thresh:.2f}.csv"
        filepath = os.path.join(BASE_OUTPUT_DIR, filename)

        pd.DataFrame({
            "id": target_ids,
            "x_2d": umap_coords[:, 0],
            "y_2d": umap_coords[:, 1],
            "core_label": core_labels,
            "cluster_label": final_labels,
        }).to_csv(filepath, index=False)

        n_clusters = len(set(final_labels) - {-1})
        summary_records.append({
            "n_papers": n_samples,
            "umap_label": u_cfg["label"],
            "mcs_ratio": ratio,
            "min_cluster_size": mcs,
            "min_samples": ms,
            "knn_prob_thresh": used_thresh,
            "n_clusters": n_clusters,
            "core_noise_pct": round(core_noise_pct, 2),
            "final_noise_pct": round(final_noise_pct, 2),
            "file": filename,
        })
        print(
            f"    {filename:<54} -> Clusters: {n_clusters:3d} | "
            f"Core Noise: {core_noise_pct:5.1f}% -> Final Noise: {final_noise_pct:5.1f}% (p>={used_thresh:.2f})"
        )

    del umap_coords
    gc.collect()

pd.DataFrame(summary_records).to_csv(
    os.path.join(BASE_OUTPUT_DIR, "master_grid_summary.csv"), index=False
)

del target_embeddings, target_ids
gc.collect()