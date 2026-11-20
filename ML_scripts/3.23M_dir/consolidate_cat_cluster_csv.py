import os
import glob
import pandas as pd

BASE_OUTPUT_DIR = "./outputs/primary_cat_grid_outputs"
WINNERS_CSV = os.path.join(BASE_OUTPUT_DIR, "best_configs_per_category.csv")
OUT_CSV = "./outputs/master_taxonomy_clusters.csv"

# 1. Load winning CSV paths for every sub-clustered primary category
winners_df = pd.read_csv(WINNERS_CSV)
winning_files = [os.path.join(BASE_OUTPUT_DIR, f) for f in winners_df["file"]]

# 2. Grab all atomic single-cluster CSVs (< 500 papers) for 100% coverage
atomic_files = glob.glob(os.path.join(BASE_OUTPUT_DIR, "*", "atomic_single_cluster.csv"))

# 3. Read, build string key, and concatenate
dfs = []
for path in winning_files + atomic_files:
    df = pd.read_csv(path, dtype={"id": str})
    df["global_cluster_name"] = df["primary_cat"] + "_" + df["cluster_label"].astype(str)
    dfs.append(df)

master_df = pd.concat(dfs, ignore_index=True)

# 4. Map non-noise clusters to globally unique sequential integers (0..N-1), keeping -1 as -1
valid_mask = master_df["cluster_label"] != -1
master_df["cluster_label"] = -1
master_df.loc[valid_mask, "cluster_label"] = pd.factorize(
    master_df.loc[valid_mask, "global_cluster_name"], sort=True
)[0]

master_df.to_csv(OUT_CSV, index=False)

n_valid_clusters = master_df.loc[valid_mask, "cluster_label"].nunique()
print(f"Saved {len(master_df):,} papers across {n_valid_clusters:,} globally unique integer clusters (+ -1 noise) to '{OUT_CSV}'.")
assert len(master_df) == 3233620  # Should equal 3,233,620