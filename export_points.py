"""
Export the global view: every paper from the full dataset as one dot, in a single compact binary file.

The global view only shows how arXiv clusters, so it needs no per-paper text: no titles, authors, abstracts or
citation lines. Finding individual papers is the job of the drill-down maps (export_maps.py).

Outputs (in web/public/data/points/):
  meta.json   paper count, column layout of points.bin, year range, fields, clusters
  points.bin  little-endian columns, N values each, in this order:
                x Float32, y Float32, influence Float32, cluster Int16 (-1 = noise), year Uint16, field Uint8
              (field = index into meta.fields: the biggest arXiv groups, then "other")

Usage:
  python export_points.py
  python export_points.py --csv path/to/enriched_arXiv_scientific_full_dataset.csv
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from arxiv_categories import group_name
from export_maps import CSV, clean, load
from export_web_data import normalize_xy

OUT_DIR = os.path.join("web", "public", "data", "points")
FIELD_COUNT = 19  # groups that get their own colour; the rest share "other"

# Column order in points.bin. Wider types come first so every column starts aligned for its typed array.
COLUMNS = [("x", "<f4"), ("y", "<f4"), ("influence", "<f4"), ("cluster", "<i2"), ("year", "<u2"), ("field", "u1")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=CSV)
    parser.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args()
    started = time.time()

    df, _ = load(args.csv)
    n = len(df)

    counts = df["group"].value_counts()
    top = list(counts.index[:FIELD_COUNT])
    index = {g: i for i, g in enumerate(top)}
    fields = [{"code": g, "name": group_name(g), "papers": int(counts[g])} for g in top]
    fields.append({"code": "other", "name": "Other fields", "papers": int(n - counts[top].sum())})
    field = df["group"].map(index).fillna(len(top)).astype("uint8").to_numpy()

    x, y = normalize_xy(df["x_2d"].astype(float), df["y_2d"].astype(float))
    cluster = df["cluster_label"].to_numpy()
    values = {
        "x": x.to_numpy(), "y": y.to_numpy(), "influence": df["influence_score"].to_numpy(),
        "cluster": cluster, "year": df["year_int"].to_numpy(), "field": field,
    }

    os.makedirs(args.out, exist_ok=True)
    layout, offset = [], 0
    with open(os.path.join(args.out, "points.bin"), "wb") as f:
        for name, dtype in COLUMNS:
            data = np.ascontiguousarray(values[name], dtype=dtype)
            f.write(data.tobytes())
            layout.append({"name": name, "type": np.dtype(dtype).name, "offset": offset})
            offset += data.nbytes

    # Clusters (noise is reported separately).
    clustered = df[cluster != -1]
    clusters = []
    for cid, g in clustered.groupby("cluster_label"):
        clusters.append({
            "id": int(cid),
            "title": clean(g["cluster_title"].iloc[0]),
            "size": int(len(g)),
            "x": round(float(x[g.index].mean()), 5),
            "y": round(float(y[g.index].mean()), 5),
            "field": int(pd.Series(field[g.index]).mode().iloc[0]),
        })
    clusters.sort(key=lambda c: -c["size"])
    noise = int(n - len(clustered))

    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "count": n, "columns": layout, "noisePapers": noise,
            "yearMin": int(values["year"].min()), "yearMax": int(values["year"].max()),
            "fields": fields, "clusters": clusters, "source": os.path.basename(args.csv),
        }, f, separators=(",", ":"), ensure_ascii=False)

    mb = 1048576
    print(f"{len(clusters):,} clusters, {noise:,} noise papers ({100 * noise / n:.0f}%)")
    print(f"Wrote {args.out}/: points.bin {offset / mb:.1f} MB, meta.json "
          f"{os.path.getsize(os.path.join(args.out, 'meta.json')) / mb:.1f} MB, in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
