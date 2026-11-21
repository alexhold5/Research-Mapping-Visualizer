"""
Normalize the two enriched-CSV layouts this project produces onto one set of column names.

Two pipelines feed the exporters:
  "scientific"  287k papers  -- s2_paper_id, authors, category, category_code, published_date,
                               citations_count, influence_score, x_2d, y_2d, cluster_label
  "full"        3.2M papers  -- id, author, categories, primary_cat, updated_date, x_2d, y_2d,
                               cluster_label, global_cluster_name, local_x, local_y

The full layout is missing four columns the website needs. Three are derivable and one isn't:
  s2_paper_id     derived from `id` (every id in the full dataset maps cleanly)
  published_date  derived from the arXiv id itself -- its first four digits are the YYMM the
                  paper was announced, which is what the timeline wants. `updated_date` is the
                  last revision, often years later, so it is only a fallback.
  category        looked up from category_code via CATEGORY_NAMES
  citations_count NOT derivable -- it comes from Semantic Scholar, and that scrape only covers
                  128,606 papers. Unscraped papers get 0 and lean on PageRank alone.
"""
import re

import numpy as np
import pandas as pd

MODERN = re.compile(r"(\d{2})(\d{2})\.\d{4,5}$")
LEGACY = re.compile(r"[a-zA-Z.\-]+[/-](\d{2})(\d{2})\d{3}$")


def to_s2_id(raw):
    """'0704.0001' / 'acc-phys/9411001' / 'cs-9308101v1' -> 'ARXIV:...' (Semantic Scholar's form)."""
    if not isinstance(raw, str) or not raw:
        return None
    s = re.sub(r"v\d+$", "", raw.strip())
    m = re.fullmatch(r"([a-zA-Z\-]+(?:\.[a-zA-Z\-]+)?)/(\d{7})", s)
    if m:
        return f"ARXIV:{m.group(1)}/{m.group(2)}"
    m = re.fullmatch(r"([a-zA-Z]+)-(\d{7})", s)
    if m:
        return f"ARXIV:{m.group(1)}/{m.group(2)}"
    m = re.fullmatch(r"\d{4}\.\d{4,5}", s)
    return f"ARXIV:{m.group(0)}" if m else None


def announced_date(raw):
    """
    The YYMM an arXiv id encodes, as 'YYYY-MM-01'.

    Modern ids (0704.0001 onward) started in April 2007, so YY is always 20YY. Legacy ids run
    from 1991 to 2007, so YY >= 91 is 19YY and anything lower is 20YY.
    """
    if not isinstance(raw, str):
        return None
    raw = re.sub(r"v\d+$", "", raw.strip())  # 'cs-9308101v1' -> 'cs-9308101'
    m = MODERN.search(raw) or LEGACY.search(raw)
    if not m:
        return None
    yy, mm = int(m.group(1)), int(m.group(2))
    if not 1 <= mm <= 12:
        return None
    return f"{1900 + yy if yy >= 91 else 2000 + yy}-{mm:02d}-01"


# arXiv's own display names for the codes that actually appear in these datasets. Anything not
# listed falls back to the code itself, which still reads fine in the UI.
CATEGORY_NAMES = {
    "cs.LG": "Machine Learning", "cs.CV": "Computer Vision and Pattern Recognition",
    "cs.CL": "Computation and Language", "cs.AI": "Artificial Intelligence",
    "cs.NE": "Neural and Evolutionary Computing", "cs.RO": "Robotics",
    "cs.CR": "Cryptography and Security", "cs.SE": "Software Engineering",
    "cs.HC": "Human-Computer Interaction", "cs.CY": "Computers and Society",
    "cs.IR": "Information Retrieval", "cs.DS": "Data Structures and Algorithms",
    "cs.SI": "Social and Information Networks", "cs.DC": "Distributed Computing",
    "cs.IT": "Information Theory", "cs.SY": "Systems and Control",
    "stat.ML": "Machine Learning (Statistics)", "stat.ME": "Methodology",
    "stat.AP": "Applications", "stat.TH": "Statistics Theory",
    "eess.IV": "Image and Video Processing", "eess.SP": "Signal Processing",
    "eess.AS": "Audio and Speech Processing", "eess.SY": "Systems and Control",
    "math.OC": "Optimization and Control", "math.ST": "Statistics Theory",
    "math.PR": "Probability", "math.NA": "Numerical Analysis", "math.CO": "Combinatorics",
    "math.AP": "Analysis of PDEs", "math.DG": "Differential Geometry", "math.AG": "Algebraic Geometry",
    "hep-ph": "High Energy Physics - Phenomenology", "hep-th": "High Energy Physics - Theory",
    "hep-ex": "High Energy Physics - Experiment", "hep-lat": "High Energy Physics - Lattice",
    "astro-ph": "Astrophysics", "astro-ph.GA": "Astrophysics of Galaxies",
    "astro-ph.CO": "Cosmology and Nongalactic Astrophysics", "astro-ph.SR": "Solar and Stellar Astrophysics",
    "astro-ph.HE": "High Energy Astrophysical Phenomena", "astro-ph.EP": "Earth and Planetary Astrophysics",
    "astro-ph.IM": "Instrumentation and Methods for Astrophysics",
    "cond-mat": "Condensed Matter", "cond-mat.mtrl-sci": "Materials Science",
    "cond-mat.mes-hall": "Mesoscale and Nanoscale Physics", "cond-mat.str-el": "Strongly Correlated Electrons",
    "cond-mat.stat-mech": "Statistical Mechanics", "cond-mat.supr-con": "Superconductivity",
    "cond-mat.soft": "Soft Condensed Matter", "cond-mat.dis-nn": "Disordered Systems and Neural Networks",
    "quant-ph": "Quantum Physics", "gr-qc": "General Relativity and Quantum Cosmology",
    "nucl-th": "Nuclear Theory", "nucl-ex": "Nuclear Experiment", "nlin.CD": "Chaotic Dynamics",
    "physics.optics": "Optics", "physics.flu-dyn": "Fluid Dynamics",
    "physics.comp-ph": "Computational Physics", "physics.chem-ph": "Chemical Physics",
    "physics.atom-ph": "Atomic Physics", "physics.ins-det": "Instrumentation and Detectors",
    "physics.soc-ph": "Physics and Society", "physics.plasm-ph": "Plasma Physics",
    "physics.bio-ph": "Biological Physics", "physics.data-an": "Data Analysis, Statistics and Probability",
    "q-bio.NC": "Neurons and Cognition", "q-bio.QM": "Quantitative Methods",
    "q-bio.PE": "Populations and Evolution", "q-bio.BM": "Biomolecules",
    "q-fin.ST": "Statistical Finance", "q-fin.CP": "Computational Finance",
    "econ.EM": "Econometrics",
}

#: arXiv's top-level archives, used when the per-category codes are too fragmented to colour.
ARCHIVE_NAMES = {
    "cs": "Computer Science", "math": "Mathematics", "physics": "Physics",
    "astro-ph": "Astrophysics", "cond-mat": "Condensed Matter", "hep-ph": "HEP - Phenomenology",
    "hep-th": "HEP - Theory", "hep-ex": "HEP - Experiment", "hep-lat": "HEP - Lattice",
    "quant-ph": "Quantum Physics", "gr-qc": "General Relativity and Quantum Cosmology",
    "nucl-th": "Nuclear Theory", "nucl-ex": "Nuclear Experiment", "math-ph": "Mathematical Physics",
    "nlin": "Nonlinear Sciences", "q-bio": "Quantitative Biology", "q-fin": "Quantitative Finance",
    "stat": "Statistics", "eess": "Electrical Engineering and Systems Science",
    "econ": "Economics", "cmp-lg": "Computation and Language", "adap-org": "Adaptive Organization",
    "chao-dyn": "Chaotic Dynamics", "solv-int": "Solvable and Integrable Systems",
    "patt-sol": "Pattern Formation and Solitons", "acc-phys": "Accelerator Physics",
    "supr-con": "Superconductivity", "mtrl-th": "Materials Theory", "alg-geom": "Algebraic Geometry",
    "dg-ga": "Differential Geometry", "funct-an": "Functional Analysis", "q-alg": "Quantum Algebra",
    "atom-ph": "Atomic Physics", "plasm-ph": "Plasma Physics", "bayes-an": "Bayesian Analysis",
    "ao-sci": "Atmospheric-Oceanic Sciences", "chem-ph": "Chemical Physics",
}


def archive_of(code):
    """'cond-mat.mtrl-sci' -> 'cond-mat', 'cs.CV' -> 'cs', 'hep-ph' -> 'hep-ph'."""
    return str(code).split(".", 1)[0] if isinstance(code, str) and code else ""


#: Columns the exporters read, after normalization.
CANONICAL = [
    "s2_paper_id", "title", "authors", "summary", "category", "category_code",
    "published_date", "citations_count", "influence_score", "x_2d", "y_2d",
    "cluster_label", "cluster_title",
]


def source_columns(wanted):
    """Which raw CSV columns to read in order to produce `wanted` under either layout."""
    extra = {
        "s2_paper_id": ["id"],
        "authors": ["author"],
        "category_code": ["primary_cat", "categories"],
        "category": ["primary_cat", "categories"],
        "published_date": ["id", "updated_date"],
        "x_2d": ["umap_x"], "y_2d": ["umap_y"],
    }
    cols = set(wanted)
    for w in wanted:
        cols.update(extra.get(w, ()))
    return cols


def normalize(df, verbose=True):
    """
    Fill in the canonical columns this DataFrame is missing, in place. Columns that are already
    present are never overwritten, so the 287k "scientific" layout passes through untouched.
    """
    derived = []

    if "s2_paper_id" not in df.columns:
        df["s2_paper_id"] = [to_s2_id(v) for v in df["id"]]
        derived.append("s2_paper_id (from id)")

    if "published_date" not in df.columns:
        src = df["id"] if "id" in df.columns else df["s2_paper_id"]
        dates = [announced_date(v) for v in src]
        if "updated_date" in df.columns:  # fall back only where the id didn't carry a date
            upd = df["updated_date"].tolist()
            dates = [d if d else upd[i] for i, d in enumerate(dates)]
        df["published_date"] = dates
        derived.append("published_date (from the arXiv id's YYMM)")

    if "category_code" not in df.columns:
        if "primary_cat" in df.columns:
            df["category_code"] = df["primary_cat"].fillna("")
        else:
            df["category_code"] = df["categories"].fillna("").astype(str).str.split().str[0]
        derived.append("category_code (from primary_cat)")

    if "category" not in df.columns:
        df["category"] = df["category_code"].map(lambda c: CATEGORY_NAMES.get(c, c))
        derived.append("category (display name)")

    if "authors" not in df.columns:
        df["authors"] = df["author"].fillna("")
        derived.append("authors (from author)")

    if "citations_count" not in df.columns:
        df["citations_count"] = 0
        derived.append("citations_count (absent -> 0; only PageRank drives influence)")

    if "influence_score" not in df.columns:
        df["influence_score"] = np.nan
        derived.append("influence_score (absent -> supply one with --influence)")

    if verbose and derived:
        print("Derived columns: " + "; ".join(derived))
    return df


# ---------------------------------------------------------------------------
# Shared column maths. These lived in export_web_data.py, the cluster view's exporter, which was
# retired with that view; export_points.py is the only remaining consumer.
# ---------------------------------------------------------------------------


def parse_dates(series):
    """
    Parse a date column, trying the formats these CSVs actually use before the slow fallback.

    "%m/%d/%y" is what the 287k dataset ships; "%Y-%m-%d" is what schema.announced_date derives
    for the 3.2M one. Hitting an explicit format is vectorized; `format="mixed"` parses value by
    value, which on millions of rows takes minutes, so it only ever sees the leftovers.
    """
    dates = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    todo = series.notna()
    for fmt in ("%m/%d/%y", "%Y-%m-%d"):
        if not todo.any():
            return dates
        parsed = pd.to_datetime(series[todo], format=fmt, errors="coerce")
        dates[todo] = parsed
        todo = todo & dates.isna()
    if todo.any():
        dates[todo] = pd.to_datetime(series[todo], errors="coerce", format="mixed")
    return dates


def to_decimal_year(dates):
    """2021-07-01 -> 2021.5 (roughly). Used for the timeline."""
    start = pd.to_datetime(dates.dt.year.astype("Int64").astype(str) + "-01-01", errors="coerce")
    return dates.dt.year + (dates - start).dt.days / 365.25


def normalize_xy(x, y):
    """Center coordinates and scale them to fit [-1, 1], keeping the aspect ratio."""
    cx, cy = (x.max() + x.min()) / 2, (y.max() + y.min()) / 2
    half = max(x.max() - x.min(), y.max() - y.min()) / 2 or 1.0
    return (x - cx) / half, (y - cy) / half


def parse_author_field(value):
    """
    The two layouts store authors differently: the 287k one as the text of a Python list,
    the 3.2M one as a single comma-separated string. Return a list either way.
    """
    if not isinstance(value, str) or not value.strip():
        return []
    s = value.strip()
    if s.startswith("[") and s.endswith("]"):
        import ast
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(a).strip() for a in parsed if str(a).strip()]
        except (ValueError, SyntaxError):
            pass
    return [a.strip() for a in s.split(",") if a.strip()]
