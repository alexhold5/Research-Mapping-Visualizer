"""
FAKE data for trying out the global view (?view=global) without the real CSV: random Gaussian blobs in the same
format export_points.py writes (web/public/data/points/points.bin + meta.json). Not real papers.
Replace it by running export_points.py; do not commit or deploy it.

  python make_fake_points.py            # 3.23M papers, 8,490 clusters (the real scale)
  python make_fake_points.py --n 200000 --clusters 300
"""
import argparse
import json
import os

import numpy as np

COLUMNS = [("x", "<f4"), ("y", "<f4"), ("influence", "<f4"), ("cluster", "<i2"), ("year", "<u2"), ("field", "u1")]
GROUPS = ["cs", "math", "cond-mat", "astro-ph", "hep-ph", "hep-th", "quant-ph", "physics", "stat", "gr-qc",
          "math-ph", "eess", "nlin", "q-bio", "hep-ex", "nucl-th", "q-fin", "econ", "hep-lat"]

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=3_233_527)
p.add_argument("--clusters", type=int, default=8490)
p.add_argument("--out", default=os.path.join("web", "public", "data", "points"))
a = p.parse_args()

r = np.random.RandomState(0)
k = a.clusters
# Cluster centres in 20 field regions; sizes are heavy-tailed like the real ones.
region = r.uniform(-0.8, 0.8, (len(GROUPS) + 1, 2))
cfield = r.randint(0, len(GROUPS) + 1, k)
centre = region[cfield] + r.normal(0, 0.12, (k, 2))
weights = r.pareto(1.2, k) + 1
weights /= weights.sum()
cl = r.choice(k, a.n, p=weights).astype(np.int16)
noise = r.rand(a.n) < 0.13
spread = 0.01 + 0.03 * r.rand(k)
xy = centre[cl] + r.normal(0, 1, (a.n, 2)) * spread[cl, None]
xy[noise] = r.uniform(-1, 1, (int(noise.sum()), 2))
xy = np.clip(xy, -1, 1)
cl[noise] = -1
values = {
    "x": xy[:, 0], "y": xy[:, 1], "influence": r.pareto(2, a.n), "cluster": cl,
    "year": np.clip(r.normal(2016, 7, a.n), 1991, 2026).astype(int), "field": cfield[np.maximum(cl, 0)],
}

os.makedirs(a.out, exist_ok=True)
layout, off = [], 0
with open(os.path.join(a.out, "points.bin"), "wb") as f:
    for name, dt in COLUMNS:
        d = np.ascontiguousarray(values[name], dtype=dt)
        f.write(d.tobytes())
        layout.append({"name": name, "type": np.dtype(dt).name, "offset": off})
        off += d.nbytes
sizes = np.bincount(cl[cl >= 0], minlength=k)
fields = [{"code": g, "name": g, "papers": int((values["field"] == i).sum())} for i, g in enumerate(GROUPS)]
fields.append({"code": "other", "name": "Other fields", "papers": int((values["field"] >= len(GROUPS)).sum())})
clusters = sorted(
    ({"id": i, "title": f"Cluster {i}: Fake topic {i}", "size": int(sizes[i]), "x": round(float(centre[i, 0]), 5),
      "y": round(float(centre[i, 1]), 5), "field": int(cfield[i])} for i in range(k) if sizes[i]),
    key=lambda c: -c["size"])
with open(os.path.join(a.out, "meta.json"), "w") as f:
    json.dump({"count": a.n, "columns": layout, "noisePapers": int(noise.sum()), "yearMin": 1991, "yearMax": 2026,
               "fields": fields, "clusters": clusters, "source": "FAKE"}, f, separators=(",", ":"))
print(f"Wrote fake {a.n:,} papers / {len(clusters)} clusters to {a.out}/ ({off / 1048576:.1f} MB)")
