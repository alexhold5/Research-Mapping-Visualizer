"""
Export the drill-down maps from the full 3.23M-paper dataset, in the slim format web/src/lib/slim.ts reads.

Three levels, every map drawn by the same website code (one folder per map under web/public/data/maps/):
  arxiv             one bubble per arXiv group (cs, math, physics, ...). Zooming into a bubble shows the group's
                    categories as nodes (or, for a group that is a single category like hep-th, its clusters).
  <group>, e.g. cs  one bubble per category (cs.AI, cs.LG, ...). Zooming in shows the category's clusters as nodes.
  <category>        one bubble per cluster (the per-category clustering in the CSV). Zooming in shows the cluster's
                    top papers: the PER_BUBBLE highest influence_score, or a random sample until the CSV has that column.
Clicking a bubble, or a node standing for a category or cluster, opens that map (at that cluster).

Files (all compact JSON):
  maps/index.json                    every map: id, title, level, parent
  maps/<map>/overview.json           "slim": 1, meta, years, timeline, fields, bubbles, links
      bubble: {id, title, size, x, y, field, share, top, count, fy, [inf], [map], inner}
        field = index into fields (len(fields) = "other"); fy = {field index: papers per year} (non-empty fields only);
        map = the map clicking it opens; inner = what zooming into it shows ("categories" / "clusters" / "papers").
  maps/<map>/clusters/<bubble id>.json   what zooming into that bubble shows, one of:
      {"papers": [[arxiv id, title, x, y, year, influence, field], ...], "links": [[row, row], ...]}
      {"opens":  [[map, cluster or null, title, x, y, papers, year], ...]}
Authors, abstracts and full citation lists are not stored: the website fetches them for the one selected paper.

The CSV is ~4 GB, so it is read twice to keep memory low: first everything except titles, then only the titles of
papers that are actually shown.

Usage:
  python export_maps.py
  python export_maps.py --csv path/to/enriched_arXiv_scientific_full_dataset.csv
"""
import argparse
import json
import os
import shutil
import time

import numpy as np
import pandas as pd
from pandas.api.types import union_categoricals

from arxiv_categories import category_name, group_name
from export_web_data import normalize_xy

CSV = os.path.join("web", "public", "data", "enriched_arXiv_scientific_full_dataset.csv")
OUT_DIR = os.path.join("web", "public", "data", "maps")
ROOT = "arxiv"     # id of the top map
PER_BUBBLE = 50    # papers shown when a cluster is zoomed into
FIELD_COUNT = 12   # fields that get their own color in a map; the rest share "other"
SEED = 0           # the random sample (used until there are influence scores) is repeatable
CHUNK = 400_000

READ = ["id", "categories", "primary_cat", "x_2d", "y_2d", "cluster_label", "cluster_title", "local_x", "local_y"]
FLOATS = ["x_2d", "y_2d", "local_x", "local_y", "influence_score"]


def clean(text):
    return " ".join(str(text).split())


# ---------------------------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------------------------

def submission_years(ids):
    """arXiv ids encode the submission month: '0704.0001' -> April 2007, 'hep-ex/0111022' -> November 2001."""
    s = ids.str.rsplit("/", n=1).str[-1]
    yy = s.str[:2].astype(int)
    mm = s.str[2:4].astype(int)
    year = np.where(yy >= 91, 1900, 2000) + yy
    return year.astype("int16"), (year + (mm - 1) / 12).astype("float32")


def load(csv_path):
    """Every paper, without titles. Text columns become categoricals and numbers float32 to keep memory low."""
    started = time.time()
    has_influence = "influence_score" in pd.read_csv(csv_path, nrows=0).columns
    cols = READ + (["influence_score"] if has_influence else [])
    dtypes = {"id": str, "primary_cat": "category", "cluster_title": "category", "cluster_label": "int32",
              **{c: "float32" for c in FLOATS if c in cols}}
    parts = []
    for chunk in pd.read_csv(csv_path, usecols=cols, dtype=dtypes, chunksize=CHUNK):
        # The category each paper is most cross-listed with (its first other category), or its own if none.
        chunk["secondary"] = pd.Categorical([
            next((c for c in cats.split() if c != p), p) for cats, p in zip(chunk["categories"], chunk["primary_cat"])
        ])
        chunk["year_int"], chunk["year"] = submission_years(chunk["id"])
        parts.append(chunk.drop(columns="categories"))
        print(f"  read {sum(len(p) for p in parts):,} rows", end="\r")
    df = pd.concat([p.drop(columns=["primary_cat", "secondary", "cluster_title"]) for p in parts], ignore_index=True)
    for c in ("primary_cat", "secondary", "cluster_title"):
        df[c] = union_categoricals([p[c] for p in parts], ignore_order=True)
    del parts
    df["group"] = df["primary_cat"].map(lambda c: c.split(".")[0]).astype("category")
    if not has_influence:
        df["influence_score"] = np.float32(0)
    print(f"Loaded {len(df):,} papers in {time.time() - started:.0f}s "
          f"({'ranking by influence_score' if has_influence else 'no influence_score yet: papers are a random sample'})")
    return df, has_influence


def read_titles(csv_path, wanted):
    """Second pass: titles of the papers in `wanted` only."""
    started = time.time()
    titles = {}
    for chunk in pd.read_csv(csv_path, usecols=["id", "title"], dtype=str, chunksize=CHUNK):
        hit = chunk[chunk["id"].isin(wanted)]
        titles.update(zip(hit["id"], hit["title"].map(clean)))
    print(f"Read {len(titles):,} titles in {time.time() - started:.0f}s")
    return titles


# ---------------------------------------------------------------------------------------------------------------
# Building maps
# ---------------------------------------------------------------------------------------------------------------

class Exporter:
    def __init__(self, df, has_influence):
        self.df = df
        self.has_influence = has_influence
        self.rng = np.random.RandomState(SEED)
        self.y0 = int(df["year_int"].min())
        self.years = list(range(self.y0, int(df["year_int"].max()) + 1))
        self.maps = {}         # map id -> {"overview": ..., "files": {bubble id: contents}}
        self.wanted = set()    # ids of papers shown somewhere (their titles are read later)
        groups = df.groupby("group", observed=True)["primary_cat"].nunique()
        self.multi = {g for g, n in groups.items() if n > 1}  # groups with their own map of categories

    # Map ids. A single-category group (hep-th) has no group map: its bubble opens the category map directly.
    # "astro-ph" and "cond-mat" are both a group and a (general) category, so that category gets ".general".
    def group_map(self, group):
        return group if group in self.multi else self.category_map(group)

    def category_map(self, cat):
        return f"{cat}.general" if cat in self.multi else cat

    def per_year(self, years_int):
        return np.bincount(years_int - self.y0, minlength=len(self.years))

    def build(self, map_id, title, level, parent, frame, key, x_col, y_col, field_col, inner, own_field=None,
              opens=None, contents=None, bubble_title=None, noise=0):
        """
        One map: a bubble per value of frame[key] at its papers' mean (x_col, y_col), colored by its most common
        field_col (not counting own_field when it has others). opens(key) is the map clicking the bubble opens;
        contents(key, frame) what zooming into it shows (default: its top papers).
        """
        fields_count = frame[field_col].value_counts()
        top = [c for c in fields_count.index[:FIELD_COUNT] if fields_count[c] > 0]
        index = {c: i for i, c in enumerate(top)}
        other = len(top)
        cats = frame[field_col].cat
        lut = np.array([index.get(c, other) for c in cats.categories], dtype=np.int16)
        field_of = pd.Series(lut[cats.codes], index=frame.index)
        own = index.get(own_field, -1)

        bubbles, files, xs, ys = [], {}, [], []
        for k, (value, g) in enumerate(frame.groupby(key, observed=True, sort=True)):
            bid = int(value) if key == "cluster_label" else k
            f_g = field_of[g.index].to_numpy()
            y_g = (g["year_int"].to_numpy() - self.y0).astype(np.int64)
            fy = np.zeros((other + 1, len(self.years)), dtype=np.int64)
            np.add.at(fy, (f_g, y_g), 1)
            totals = fy.sum(axis=1)
            ranked = [f for f in np.argsort(-totals, kind="stable") if totals[f] > 0]
            main = next((f for f in ranked if f != own), ranked[0])  # color by what it is cross-listed with
            bubble = {
                "id": bid,
                "title": bubble_title(value, g) if bubble_title else clean(value),
                "size": int(len(g)),
                "field": int(main),
                "share": round(float(totals[main] / len(g)), 3),
                "count": self.per_year(g["year_int"].to_numpy()).tolist(),
                "fy": {str(f): fy[f].tolist() for f in ranked},
                "inner": inner,
            }
            if self.has_influence:
                sums = np.bincount(y_g, weights=g["influence_score"].to_numpy(), minlength=len(self.years))
                n = np.maximum(bubble["count"], 1)
                bubble["inf"] = np.round(sums / n, 3).tolist()
            if opens:
                bubble["map"] = opens(value)
            body = contents(value, g) if contents else self.top_papers(g, field_of, x_col, y_col)
            bubble["top"] = body.pop("_top")
            files[bid] = body
            bubbles.append(bubble)
            xs.append(float(g[x_col].mean()))
            ys.append(float(g[y_col].mean()))

        nx_, ny_ = normalize_xy(pd.Series(xs), pd.Series(ys))
        for b, x, y in zip(bubbles, nx_, ny_):
            b["x"], b["y"] = round(float(x), 4), round(float(y), 4)
        bubbles.sort(key=lambda b: -b["size"])

        names = category_name if field_col in ("primary_cat", "secondary") else group_name
        self.maps[map_id] = {
            "overview": {
                "slim": 1,
                "meta": {
                    "mapId": map_id, "title": title, "level": level, "parent": parent,
                    "parentTitle": self.maps[parent]["overview"]["meta"]["title"] if parent else None,
                    "totalPapers": int(len(frame)) + noise, "clusteredPapers": int(len(frame)), "noisePapers": noise,
                    "clusterCount": len(bubbles), "perCluster": PER_BUBBLE,
                    "yearMin": round(float(frame["year"].min()), 3), "yearMax": round(float(frame["year"].max()), 3),
                    "source": "enriched_arXiv_scientific_full_dataset.csv",
                    "ranked": "influence" if self.has_influence else "random",
                },
                "years": self.years,
                "timeline": {"count": self.per_year(frame["year_int"].to_numpy()).tolist()},
                "fields": [{"code": c, "name": names(c), "papers": int(fields_count[c])} for c in top],
                "bubbles": bubbles,
                "links": [],
            },
            "files": files,
        }

    def top_papers(self, g, field_of, x_col, y_col):
        """A cluster's top papers (by influence, or a random sample without it). Titles are filled in later."""
        n = min(PER_BUBBLE, len(g))
        top = g.nlargest(n, "influence_score") if self.has_influence else g.sample(n=n, random_state=self.rng)
        self.wanted.update(top["id"])
        rows = [[pid, None, round(float(x), 3), round(float(y), 3), round(float(yr), 2), round(float(inf), 3),
                 int(f)]
                for pid, x, y, yr, inf, f in zip(top["id"], top[x_col], top[y_col], top["year"],
                                                 top["influence_score"], field_of[top.index])]
        return {"papers": rows, "links": [], "_top": rows[0][0]}  # "_top": the id whose title names the top paper

    def cluster_nodes(self, cat, g):
        """Nodes standing for each cluster of a category (in the category's own layout); clicking one opens it."""
        rows = []
        for label, c in g[g["cluster_label"] != -1].groupby("cluster_label", observed=True):
            rows.append([self.category_map(cat), int(label), clean(c["cluster_title"].iloc[0]),
                         round(float(c["local_x"].mean()), 3), round(float(c["local_y"].mean()), 3), int(len(c)),
                         round(float(c["year"].median()), 2)])
        rows.sort(key=lambda r: -r[5])
        return {"opens": rows, "_top": f"Largest cluster: {rows[0][2]}" if rows else ""}

    def category_nodes(self, g):
        """Nodes standing for each category of a group (in the global layout); clicking one opens its map."""
        rows = []
        for cat, c in g.groupby("primary_cat", observed=True):
            rows.append([self.category_map(cat), None, f"{category_name(cat)} ({cat})",
                         round(float(c["x_2d"].mean()), 3), round(float(c["y_2d"].mean()), 3), int(len(c)),
                         round(float(c["year"].median()), 2)])
        rows.sort(key=lambda r: -r[5])
        return {"opens": rows, "_top": f"Largest category: {rows[0][2]}" if rows else ""}

    def run(self):
        df = self.df
        print("Building maps:")
        self.build(
            ROOT, "All of arXiv", "root", None, df, key="group", x_col="x_2d", y_col="y_2d", field_col="group",
            inner="categories", opens=self.group_map,
            contents=lambda grp, g: self.category_nodes(g) if grp in self.multi else self.cluster_nodes(grp, g),
            bubble_title=lambda grp, g: f"{group_name(grp)} ({grp})",
        )
        for grp in sorted(self.multi):
            g = df[df["group"] == grp]
            self.build(
                grp, f"{group_name(grp)} ({grp})", "group", ROOT, g, key="primary_cat", x_col="x_2d", y_col="y_2d",
                field_col="primary_cat", inner="clusters", opens=self.category_map,
                contents=lambda cat, c: self.cluster_nodes(cat, c),
                bubble_title=lambda cat, c: f"{category_name(cat)} ({cat})",
            )
        for cat in sorted(df["primary_cat"].cat.categories):
            c = df[df["primary_cat"] == cat]
            clustered = c[c["cluster_label"] != -1]
            if clustered.empty:
                print(f"  (skipping {cat}: no clusters)")
                continue
            grp = cat.split(".")[0]
            self.build(
                self.category_map(cat), f"{category_name(cat)} ({cat})", "category",
                grp if grp in self.multi else ROOT, clustered, key="cluster_label", x_col="local_x",
                y_col="local_y", field_col="secondary", own_field=cat, inner="papers",
                bubble_title=lambda label, g: clean(g["cluster_title"].iloc[0]), noise=int(len(c) - len(clustered)),
            )
        print(f"  {len(self.maps)} maps, {len(self.wanted):,} papers shown")

    def write(self, out_dir, titles):
        # Remove only map folders an earlier run wrote (each holds an overview.json).
        if os.path.isdir(out_dir):
            for name in os.listdir(out_dir):
                path = os.path.join(out_dir, name)
                if os.path.isdir(path) and os.path.exists(os.path.join(path, "overview.json")):
                    shutil.rmtree(path)
        os.makedirs(out_dir, exist_ok=True)

        def dump(path, obj):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, separators=(",", ":"), ensure_ascii=False)
            return os.path.getsize(path)

        sizes = {"overview": 0, "papers": 0, "opens": 0}
        for map_id, m in self.maps.items():
            folder = os.path.join(out_dir, map_id)
            os.makedirs(os.path.join(folder, "clusters"))
            for b in m["overview"]["bubbles"]:
                if "papers" in m["files"][b["id"]]:  # name the top paper (its id was a placeholder)
                    b["top"] = titles.get(b["top"], "")
            sizes["overview"] += dump(os.path.join(folder, "overview.json"), m["overview"])
            for bid, body in m["files"].items():
                for row in body.get("papers", []):
                    row[1] = titles.get(row[0], "")
                sizes["papers" if "papers" in body else "opens"] += dump(
                    os.path.join(folder, "clusters", f"{bid}.json"), body)
        dump(os.path.join(out_dir, "index.json"), [
            {k: m["overview"]["meta"][k] for k in ("mapId", "title", "level", "parent")} for m in self.maps.values()
        ])
        mb = 1e6
        total = sum(sizes.values())
        print(f"Wrote {len(self.maps)} maps to {out_dir}/: {total / mb:.1f} MB "
              f"(overviews {sizes['overview'] / mb:.1f} MB, paper files {sizes['papers'] / mb:.1f} MB, "
              f"category/cluster node files {sizes['opens'] / mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=CSV)
    parser.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args()
    started = time.time()

    df, has_influence = load(args.csv)
    exporter = Exporter(df, has_influence)
    exporter.run()
    titles = read_titles(args.csv, exporter.wanted)
    exporter.write(args.out, titles)
    print(f"Done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
