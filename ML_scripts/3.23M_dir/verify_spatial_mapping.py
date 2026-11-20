import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

ENRICHED_CSV = "./outputs/full_dataset_clustering/enriched_arXiv_scientific_dataset.csv"
OUT_PLOT = "./outputs/full_dataset_clustering/spatial_hierarchy_full_map.png"

TOP_N_CATEGORIES = 150        # Categories highlighted in left panel & terminal printout
TOP_N_CLUSTERS_PER_CAT = 8   # Clusters printed per category in terminal
TOP_N_CLUSTERS_PLOT = 105     # Top global clusters labeled in right panel

_PREFIX_RE = re.compile(r"^Cluster\s+\d+\s*(?:\([^)]+\))?\s*:\s*", flags=re.IGNORECASE)


def format_range(series: pd.Series) -> str:
    q25, med, q75 = np.percentile(series, [25, 50, 75])
    return f"med={med:6.2f} (IQR: [{q25:6.2f}, {q75:6.2f}], min/max: [{series.min():6.2f}, {series.max():6.2f}])"


def nudge_labels(coords: np.ndarray, min_dist: float = 0.55, iterations: int = 80) -> np.ndarray:
    """Simple repulsive force layout so text callouts don't pile on top of each other."""
    pos = coords.copy().astype(float)
    n = len(pos)
    for _ in range(iterations):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                # Aspect-adjusted distance (text is wider than tall)
                dist = np.hypot(dx * 0.55, dy)
                if dist < min_dist:
                    if dist == 0:
                        dx, dy, dist = 0.01, 0.01, 0.014
                    push = (min_dist - dist) * 0.25
                    ux, uy = dx / dist, dy / dist
                    pos[i, 0] += ux * push * 1.3
                    pos[i, 1] += uy * push * 0.9
                    pos[j, 0] -= ux * push * 1.3
                    pos[j, 1] -= uy * push * 0.9
                    moved = True
        if not moved:
            break
    return pos


def main():
    # 1. Read-only load of required columns
    header = pd.read_csv(ENRICHED_CSV, nrows=0).columns.tolist()
    x_col = "global_x" if "global_x" in header else "x_2d"
    y_col = "global_y" if "global_y" in header else "y_2d"
    title_col = "cluster_title" if "cluster_title" in header else "global_cluster_name"

    use_cols = ["primary_cat", "cluster_label", title_col, x_col, y_col]
    df = pd.read_csv(ENRICHED_CSV, usecols=use_cols)
    df[title_col] = df[title_col].astype(str).str.replace(_PREFIX_RE, "", regex=True).str.strip()

    print(f"Loaded {len(df):,} rows using global columns ({x_col}, {y_col})\n" + "=" * 95)

    # 2. Terminal Verification Printout
    cat_counts = df["primary_cat"].value_counts()
    cats_to_show = cat_counts.index[:TOP_N_CATEGORIES] if TOP_N_CATEGORIES else cat_counts.index

    for cat in cats_to_show:
        cat_df = df[df["primary_cat"] == cat]
        cx_med, cy_med = cat_df[x_col].median(), cat_df[y_col].median()

        print(f"\nCATEGORY: {cat}  ({len(cat_df):,} papers)")
        print(f"  Global X: {format_range(cat_df[x_col])}")
        print(f"  Global Y: {format_range(cat_df[y_col])}")

        clustered = cat_df[cat_df["cluster_label"] != -1]
        if clustered.empty:
            print("  (No non-noise clusters)")
            continue

        top_clusters = (
            clustered.groupby(["cluster_label", title_col], sort=False)
            .agg(
                count=(x_col, "size"),
                x_med=(x_col, "median"),
                x_q25=(x_col, lambda s: np.percentile(s, 25)),
                x_q75=(x_col, lambda s: np.percentile(s, 75)),
                y_med=(y_col, "median"),
                y_q25=(y_col, lambda s: np.percentile(s, 25)),
                y_q75=(y_col, lambda s: np.percentile(s, 75)),
            )
            .sort_values("count", ascending=False)
            .head(TOP_N_CLUSTERS_PER_CAT)
        )

        max_title_len = max(len(str(t)) for _, t in top_clusters.index)
        for (c_id, c_title), row in top_clusters.iterrows():
            dist = np.hypot(row["x_med"] - cx_med, row["y_med"] - cy_med)
            flag = "  <-- [BRIDGE / OUTLIER]" if dist > 3.5 else ""
            print(
                f"    [{int(c_id):4d}] ({int(row['count']):6,d} papers) {str(c_title):<{max_title_len}} | "
                f"X: {row['x_med']:5.2f} [{row['x_q25']:5.2f}..{row['x_q75']:5.2f}] | "
                f"Y: {row['y_med']:5.2f} [{row['y_q25']:5.2f}..{row['y_q75']:5.2f}]{flag}"
            )

    # 3. Two-Panel Spatial Hierarchy Visualization
    fig, (ax_cat, ax_clu) = plt.subplots(1, 2, figsize=(24, 11), dpi=180, sharex=True, sharey=True)
    fig.patch.set_facecolor("#0f1117")
    for ax in (ax_cat, ax_clu):
        ax.set_facecolor("#0f1117")
        ax.tick_params(colors="#9ca3af", labelsize=9)
        ax.grid(True, color="#1f2430", linestyle="--", linewidth=0.5, alpha=0.7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2d3343")

    # Background 300k point sample so rendering is fast & crisp
    bg = df.sample(n=min(300_000, len(df)), random_state=42)
    top_cats = list(cat_counts.index[:TOP_N_CATEGORIES])
    cmap = plt.get_cmap("tab20")
    cat_color_map = {cat: cmap(i % 20) for i, cat in enumerate(top_cats)}

    # --- LEFT PANEL: Macro Categories + IQR Ellipses ---
    other_bg = bg[~bg["primary_cat"].isin(top_cats)]
    ax_cat.scatter(other_bg[x_col], other_bg[y_col], s=0.3, c="#374151", alpha=0.25, rasterized=True)

    cat_anchors = []
    for cat in top_cats:
        sub_bg = bg[bg["primary_cat"] == cat]
        color = cat_color_map[cat]
        ax_cat.scatter(sub_bg[x_col], sub_bg[y_col], s=0.5, c=[color], alpha=0.35, rasterized=True)

        full_cat = df[df["primary_cat"] == cat]
        x25, x50, x75 = np.percentile(full_cat[x_col], [25, 50, 75])
        y25, y50, y75 = np.percentile(full_cat[y_col], [25, 50, 75])

        # Draw IQR bounding ellipse representing core 50% mass of the category
        ell = Ellipse(
            xy=(x50, y50),
            width=(x75 - x25),
            height=(y75 - y25),
            edgecolor=color,
            facecolor=(*color[:3], 0.12),
            linewidth=1.6,
            linestyle="-",
        )
        ax_cat.add_patch(ell)
        cat_anchors.append((cat, len(full_cat), x50, y50, color))

    cat_pos = nudge_labels(np.array([[c[2], c[3]] for c in cat_anchors]), min_dist=0.75)
    for (cat, cnt, x50, y50, color), (lx, ly) in zip(cat_anchors, cat_pos):
        ax_cat.annotate(
            f"{cat}\n({cnt // 1000}k)",
            xy=(x50, y50),
            xytext=(lx, ly),
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.25", fc="#111827", ec=color, lw=1.3, alpha=0.9),
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8, alpha=0.8),
        )

    ax_cat.set_title("Macro Category Continents (Median + 25–75% IQR Ellipses)", color="white", fontsize=13, fontweight="bold", pad=12)
    ax_cat.set_xlabel("Global UMAP X", color="#d1d5db", fontsize=10)
    ax_cat.set_ylabel("Global UMAP Y", color="#d1d5db", fontsize=10)

    # --- RIGHT PANEL: Top Global Clusters + IQR Crosshairs ---
    ax_clu.scatter(bg[x_col], bg[y_col], s=0.3, c="#4b5563", alpha=0.22, rasterized=True)

    clustered_all = df[df["cluster_label"] != -1]
    global_top_clusters = (
        clustered_all.groupby(["cluster_label", "primary_cat", title_col], sort=False)
        .agg(
            count=(x_col, "size"),
            x_med=(x_col, "median"),
            x_q25=(x_col, lambda s: np.percentile(s, 25)),
            x_q75=(x_col, lambda s: np.percentile(s, 75)),
            y_med=(y_col, "median"),
            y_q25=(y_col, lambda s: np.percentile(s, 25)),
            y_q75=(y_col, lambda s: np.percentile(s, 75)),
        )
        .sort_values("count", ascending=False)
        .head(TOP_N_CLUSTERS_PLOT)
    )

    clu_anchors = []
    for (c_id, p_cat, c_title), row in global_top_clusters.iterrows():
        color = cat_color_map.get(p_cat, "#38bdf8")
        # Plot IQR crosshairs showing horizontal and vertical core spread
        ax_clu.plot([row["x_q25"], row["x_q75"]], [row["y_med"], row["y_med"]], color=color, lw=1.5, alpha=0.85)
        ax_clu.plot([row["x_med"], row["x_med"]], [row["y_q25"], row["y_q75"]], color=color, lw=1.5, alpha=0.85)
        ax_clu.scatter([row["x_med"]], [row["y_med"]], s=36, color=color, edgecolors="white", linewidths=0.7, zorder=5)

        short_title = c_title if len(c_title) <= 34 else c_title[:32] + "…"
        clu_anchors.append((f"[{int(c_id)}] {p_cat}: {short_title}", row["x_med"], row["y_med"], color))

    clu_pos = nudge_labels(np.array([[c[1], c[2]] for c in clu_anchors]), min_dist=0.68, iterations=120)
    for (label_txt, x50, y50, color), (lx, ly) in zip(clu_anchors, clu_pos):
        ax_clu.annotate(
            label_txt,
            xy=(x50, y50),
            xytext=(lx, ly),
            ha="center",
            va="center",
            fontsize=6.8,
            color="#f3f4f6",
            bbox=dict(boxstyle="round,pad=0.2", fc="#111827", ec=color, lw=0.9, alpha=0.88),
            arrowprops=dict(arrowstyle="-", color=color, lw=0.7, alpha=0.75),
            zorder=6,
        )

    ax_clu.set_title(f"Top {TOP_N_CLUSTERS_PLOT} Subfield Clusters (Centroids + IQR Spread)", color="white", fontsize=13, fontweight="bold", pad=12)
    ax_clu.set_xlabel("Global UMAP X", color="#d1d5db", fontsize=10)

    plt.suptitle("arXiv 3.23M Spatial Hierarchy Verification", color="white", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(OUT_PLOT), exist_ok=True)
    plt.savefig(OUT_PLOT, facecolor=fig.get_facecolor())
    print(f"\nSaved spatial hierarchy visualization to {OUT_PLOT}")


if __name__ == "__main__":
    main()