import gc
import os
import time
import numpy as np
import pandas as pd
import umap
from ml_pipeline import create_snapshot

# ==========================================
# CONFIGURATION — Update paths if needed
# ==========================================
ENRICHED_CSV = "./outputs/full_dataset_clustering/enriched_arXiv_scientific_dataset.csv"
EMBEDDINGS_NPY = "./scibert_embeddings_full.npy"
EMBEDDINGS_IDS_NPY = None  # Optional: string ID array if embeddings aren't pre-aligned

FIT_SAMPLE_SIZE = 200_000       # Stratified sample for fitting global UMAP graph
TRANSFORM_BATCH_SIZE = 250_000  # Chunked transform keeps peak RAM low


def main():
    t0 = time.perf_counter()

    # 1. Load the full enriched dataset so all columns are preserved on write-back
    df = pd.read_csv(ENRICHED_CSV, dtype={"id": str})
    df["id"] = df["id"].str.strip()
    df = df.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)
    print(f"Loaded {len(df):,} unique papers ({len(df.columns)} columns) in {time.perf_counter() - t0:.1f}s.")

    # Guard against re-running on an already-updated CSV so local coords aren't overwritten
    if "local_x" in df.columns and "local_y" in df.columns:
        raise RuntimeError(
            "local_x and local_y already exist in the CSV. Aborting to prevent overwriting local coordinates with global ones."
        )

    RAW_ARXIV_CSV = "./data/arXiv_full_dataset.csv"  # Source CSV used to generate scibert_embeddings_768d.npy

    # 2. Memory-map 768D SciBERT embeddings & align via raw CSV ID order (handles the 93 removed duplicates)
    emb_mmap = np.load(EMBEDDINGS_NPY, mmap_mode="r")

    raw_ids = pd.read_csv(RAW_ARXIV_CSV, usecols=["id"], dtype={"id": str})["id"].str.strip()
    assert len(emb_mmap) == len(raw_ids), (
        f"Embeddings ({len(emb_mmap)}) and raw CSV ({len(raw_ids)}) row counts do not match!"
    )

    # Map each unique paper ID to its first row index in emb_mmap
    id_to_row = pd.Series(np.arange(len(raw_ids), dtype=np.int64), index=raw_ids)
    id_to_row = id_to_row[~id_to_row.index.duplicated(keep="first")]

    row_indices = df["id"].map(id_to_row).to_numpy()
    if np.isnan(row_indices).any():
        raise ValueError(f"Missing embeddings for {int(np.isnan(row_indices).sum())} paper IDs!")
    row_indices = row_indices.astype(np.int64)
    del raw_ids, id_to_row
    gc.collect()

    # 3. Strictly proportional stratified sample WITH a floor of >= 1 point per (primary_cat, cluster_label)
    strata = df["primary_cat"].astype(str) + "_" + df["cluster_label"].astype(str)
    sample_frac = min(1.0, FIT_SAMPLE_SIZE / len(df))

    # Shuffle once, assign sequential rank (0, 1, 2...) within each stratum, and keep ceil(N * sample_frac)
    shuffled_idx = df.sample(frac=1.0, random_state=42).index
    rank_in_stratum = strata.loc[shuffled_idx].groupby(strata.loc[shuffled_idx]).cumcount()
    stratum_sizes = strata.loc[shuffled_idx].map(strata.value_counts())

    keep_mask = rank_in_stratum < np.ceil(stratum_sizes * sample_frac)
    fit_pos = np.sort(shuffled_idx[keep_mask].to_numpy())
    print(f"Fitting global UMAP on {len(fit_pos):,} stratified papers...")
    fit_embeddings = np.asarray(emb_mmap[row_indices[fit_pos]], dtype=np.float32)

    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.15,
        n_components=2,
        metric="cosine",
        low_memory=True,
        random_state=42,
        verbose=False,
    )
    reducer.fit(fit_embeddings)
    del fit_embeddings
    gc.collect()

    # 4. Transform all 3.23M embeddings in batches into float32 arrays
    n_total = len(df)
    global_coords = np.empty((n_total, 2), dtype=np.float32)

    for start in range(0, n_total, TRANSFORM_BATCH_SIZE):
        end = min(start + TRANSFORM_BATCH_SIZE, n_total)
        batch_emb = np.asarray(emb_mmap[row_indices[start:end]], dtype=np.float32)
        global_coords[start:end] = reducer.transform(batch_emb)
        del batch_emb
        gc.collect()
        print(f"Transformed {end:,} / {n_total:,} papers ({end / n_total:.1%})")

    # 5. Shift existing local coords to (local_x, local_y) and overwrite (x_2d, y_2d) with global coords
    df["local_x"] = df["x_2d"].astype(np.float32)
    df["local_y"] = df["y_2d"].astype(np.float32)
    df["x_2d"] = global_coords[:, 0]
    df["y_2d"] = global_coords[:, 1]

    # 6. Write safely to a temporary file first, then atomically replace ENRICHED_CSV
    tmp_csv = f"{ENRICHED_CSV}.tmp"
    df.to_csv(tmp_csv, index=False)
    os.replace(tmp_csv, ENRICHED_CSV)
    print(f"Updated {ENRICHED_CSV} in-place ({len(df):,} rows) in {time.perf_counter() - t0:.1f}s.")
    create_snapshot(df, output_image="snapshot3.png")


if __name__ == "__main__":
    main()