import pandas as pd

# 1. Load metadata (works for CSV, Parquet, or JSONL — uncomment your format)
df = pd.read_csv("data/arXiv_full_dataset.csv", dtype={"id": str}, usecols=["id", "categories"])
# df = pd.read_parquet("arxiv_cleaned_metadata.parquet")
# df = pd.read_json("arxiv-metadata-oai-snapshot.json", lines=True)


# # 3. Parse all individual arXiv categories (handles space-separated multi-labels like 'cs.LG stat.ML')
all_cats_series = (
    df["categories"]
    .dropna()
    .astype(str)
    .str.split()
    .explode()
)

cat_counts = all_cats_series.value_counts()

print("\n" + "=" * 80)
print(f" ALL UNIQUE ARXIV CATEGORIES ({len(cat_counts)} total)")
print("=" * 80)
for cat, count in cat_counts.items():
    print(f"{cat:<22} | {count:,} papers")

# # Optional: Print top-level Macro prefixes (e.g., cs, physics, math, stat, cond-mat)
# macro_counts = all_cats_series.str.split(".").str[0].value_counts()
# print("\n" + "=" * 80)
# print(f" TOP-LEVEL ARCHIVE PREFIXES ({len(macro_counts)} total)")
# print("=" * 80)
# for prefix, count in macro_counts.items():
#     print(f"{prefix:<22} | {count:,} tag occurrences")


CATEGORY_MAP = {
    "math.MP": "math-ph", "math.IT": "cs.IT", "cs.NA": "math.NA",
    "cs.SY": "eess.SY", "stat.TH": "math.ST", "cmp-lg": "cs.CL",
    "alg-geom": "math.AG", "q-alg": "math.QA", "dg-ga": "math.DG",
    "funct-an": "math.FA", "solv-int": "nlin.SI", "patt-sol": "nlin.PS",
    "adap-org": "nlin.AO", "chao-dyn": "nlin.CD", "comp-gas": "nlin.CG",
    "mtrl-th": "cond-mat.mtrl-sci", "supr-con": "cond-mat.supr-con",
    "chem-ph": "physics.chem-ph", "atom-ph": "physics.atom-ph",
    "acc-phys": "physics.acc-ph", "plasm-ph": "physics.plasm-ph",
    "ao-sci": "physics.ao-ph", "bayes-an": "stat.ME",
}

def get_primary_canonical_category(cat_str: str) -> str:
    raw_primary = str(cat_str).strip().split()[0]
    return CATEGORY_MAP.get(raw_primary, raw_primary)

df["primary_cat"] = df["categories"].apply(get_primary_canonical_category)

THRESHOLD = 10_000
cat_counts = df["primary_cat"].value_counts()

mega_cats = cat_counts[cat_counts >= THRESHOLD].index.tolist()
leaf_cats = cat_counts[cat_counts < THRESHOLD].index.tolist()

print(f"Mega-categories to sub-cluster (>= {THRESHOLD:,}): {len(mega_cats)} ({cat_counts[mega_cats].sum():,} papers)")
print(f"Atomic leaf categories (< {THRESHOLD:,}):          {len(leaf_cats)} ({cat_counts[leaf_cats].sum():,} papers)")
print(f"Total clustered:                          {cat_counts[mega_cats].sum() / len(df)}")