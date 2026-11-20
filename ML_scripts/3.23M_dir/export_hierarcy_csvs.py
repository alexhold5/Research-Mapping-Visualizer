import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION — Update paths if needed
# ==========================================
ENRICHED_CSV = "./outputs/full_dataset_clustering/enriched_arXiv_scientific_dataset.csv"
OUT_MACRO_CSV = "./outputs/full_dataset_clustering/macro_topics.csv"
OUT_SUBMACRO_CSV = "./outputs/full_dataset_clustering/submacro_categories.csv"
OUT_COMBINED_PLOT = "./outputs/full_dataset_clustering/hierarchy_bubble_map.png"


def nudge_labels(coords: np.ndarray, min_dist: float = 0.55, iterations: int = 120) -> np.ndarray:
    """Repulsive force layout to prevent text labels from overlapping."""
    pos = coords.copy().astype(float)
    n = len(pos)
    for _ in range(iterations):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
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


def format_vol(v: int) -> str:
    if v >= 1_000_000:
        return f"{v / 1_000_000:.2f}M"
    if v >= 1_000:
        return f"{v / 1_000:.0f}k"
    return str(v)


def draw_bubble_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    title: str,
    color_col: str,
    group_colors: dict,
    max_bubble_size: float = 2600.0,
    min_bubble_size: float = 55.0,
    label_top_n: int | None = None,
    min_label_dist: float = 0.65,
    show_ylabel: bool = True,
):
    ax.set_facecolor("#0f1117")
    ax.tick_params(colors="#9ca3af", labelsize=9)
    ax.grid(True, color="#1f2430", linestyle="--", linewidth=0.6, alpha=0.75)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d3343")

    # Scale bubble area linearly with paper_volume
    vols = df["paper_volume"].to_numpy(dtype=float)
    sizes = min_bubble_size + (vols / vols.max()) * (max_bubble_size - min_bubble_size)
    point_colors = [group_colors[g] for g in df[color_col]]

    # Draw larger bubbles first so smaller nested subfields remain visible on top
    order = np.argsort(-vols)
    df_ord = df.iloc[order].reset_index(drop=True)
    sizes_ord = sizes[order]
    colors_ord = [point_colors[i] for i in order]

    ax.scatter(
        df_ord["x"],
        df_ord["y"],
        s=sizes_ord,
        c=colors_ord,
        alpha=0.75,
        edgecolors="white",
        linewidths=1.1,
        zorder=3,
    )

    label_df = df_ord if label_top_n is None else df_ord.head(label_top_n)
    raw_coords = label_df[["x", "y"]].to_numpy()
    nudged = nudge_labels(raw_coords, min_dist=min_label_dist, iterations=140)

    for idx, row in label_df.iterrows():
        lx, ly = nudged[idx]
        c = group_colors[row[color_col]]
        ax.annotate(
            f"{row['name']} ({format_vol(int(row['paper_volume']))})",
            xy=(row["x"], row["y"]),
            xytext=(lx, ly),
            ha="center",
            va="center",
            fontsize=8.2 if label_top_n is None else 7.0,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.22", fc="#111827", ec=c, lw=1.1, alpha=0.9),
            arrowprops=dict(arrowstyle="-", color=c, lw=0.8, alpha=0.8),
            zorder=5,
        )

    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Global UMAP X (Median Centroid)", color="#d1d5db", fontsize=10)
    if show_ylabel:
        ax.set_ylabel("Global UMAP Y (Median Centroid)", color="#d1d5db", fontsize=10)


def main():
    header = pd.read_csv(ENRICHED_CSV, nrows=0).columns.tolist()
    x_col = "global_x" if "global_x" in header else "x_2d"
    y_col = "global_y" if "global_y" in header else "y_2d"

    df = pd.read_csv(ENRICHED_CSV, usecols=["primary_cat", x_col, y_col])
    df["primary_cat"] = df["primary_cat"].astype(str).str.strip()
    df["macro_category"] = df["primary_cat"].str.split(".").str[0]

    os.makedirs(os.path.dirname(OUT_MACRO_CSV), exist_ok=True)

    # 1. Macro Topics CSV: name, x, y, paper_volume
    macro_df = (
        df.groupby("macro_category", as_index=False)
        .agg(
            x=(x_col, "median"),
            y=(y_col, "median"),
            paper_volume=(x_col, "size"),
        )
        .rename(columns={"macro_category": "name"})
        .sort_values("paper_volume", ascending=False)
        .reset_index(drop=True)
    )
    macro_df["x"] = macro_df["x"].round(4)
    macro_df["y"] = macro_df["y"].round(4)
    macro_df.to_csv(OUT_MACRO_CSV, index=False)

    # 2. Submacro Categories CSV: name, x, y, paper_volume, macro_category
    submacro_df = (
        df.groupby(["primary_cat", "macro_category"], as_index=False)
        .agg(
            x=(x_col, "median"),
            y=(y_col, "median"),
            paper_volume=(x_col, "size"),
        )
        .rename(columns={"primary_cat": "name"})
        [["name", "x", "y", "paper_volume", "macro_category"]]
        .sort_values("paper_volume", ascending=False)
        .reset_index(drop=True)
    )
    submacro_df["x"] = submacro_df["x"].round(4)
    submacro_df["y"] = submacro_df["y"].round(4)
    submacro_df.to_csv(OUT_SUBMACRO_CSV, index=False)

    print(f"Saved {len(macro_df)} macro topics to {OUT_MACRO_CSV}")
    print(f"Saved {len(submacro_df)} submacro categories to {OUT_SUBMACRO_CSV}")

    # 3. Shared color palette across both panels (keyed by macro topic)
    cmap = plt.get_cmap("tab20")
    group_colors = {name: cmap(i % 20) for i, name in enumerate(macro_df["name"])}

    # 4. Combined Two-Panel Figure
    fig, (ax_macro, ax_sub) = plt.subplots(1, 2, figsize=(24, 10.5), dpi=180, sharex=True, sharey=True)
    fig.patch.set_facecolor("#0f1117")

    draw_bubble_panel(
        ax=ax_macro,
        df=macro_df,
        title=f"Macro Topics ({len(macro_df)} Disciplines — Area = Paper Volume)",
        color_col="name",
        group_colors=group_colors,
        max_bubble_size=3200.0,
        min_bubble_size=120.0,
        label_top_n=None,
        min_label_dist=0.72,
        show_ylabel=True,
    )

    draw_bubble_panel(
        ax=ax_sub,
        df=submacro_df,
        title=f"Submacro Categories ({len(submacro_df)} Categories — Colored by Macro Topic)",
        color_col="macro_category",
        group_colors=group_colors,
        max_bubble_size=2200.0,
        min_bubble_size=45.0,
        label_top_n=45,
        min_label_dist=0.58,
        show_ylabel=False,
    )

    # Lock shared coordinate bounds across both panels
    x_pad, y_pad = 1.5, 1.2
    ax_macro.set_xlim(submacro_df["x"].min() - x_pad, submacro_df["x"].max() + x_pad)
    ax_macro.set_ylim(submacro_df["y"].min() - y_pad, submacro_df["y"].max() + y_pad)

    plt.suptitle("arXiv Topic Hierarchy — Macro vs. Submacro Centroids", color="white", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUT_COMBINED_PLOT, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved combined bubble map to {OUT_COMBINED_PLOT}")


if __name__ == "__main__":
    main()