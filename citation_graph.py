"""
Stream a citation adjacency JSON ({"ARXIV:citing": ["ARXIV:cited", ...], ...})
into integer edge arrays indexed by paper row, without loading the JSON into memory.

The adjacency file can be close to 1 GB; json.load() on it needs many times that
in RAM. This parser reads fixed-size chunks and only cuts them after a closing
"]" (the end of one paper's reference list), so no key or id is ever split.

Edges whose citing or cited paper is not in `ids` are dropped and counted.
"""
import re
import time

import numpy as np

TOKEN = re.compile(r'"([^"]+)"(\s*:)?')


def load_edges(path, ids, chunk_bytes=64 << 20, verbose=True):
    """
    Returns (citing, cited, stats): int32 arrays of row positions in `ids`,
    where `ids` is a sequence of keys as they appear in the file (e.g. "ARXIV:2103.12345").
    """
    index = {k: i for i, k in enumerate(ids)}
    citing_parts, cited_parts = [], []
    stats = {"citing_keys": 0, "citing_in_corpus": 0, "edges": 0, "edges_kept": 0}
    buf = ""
    current = -1
    t0 = time.time()
    with open(path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_bytes)
            buf += chunk
            cut = buf.rfind("]") + 1 if chunk else len(buf)
            if cut == 0:
                if not chunk:
                    break
                continue
            part, buf = buf[:cut], buf[cut:]
            src, dst = [], []
            for m in TOKEN.finditer(part):
                tok, is_key = m.group(1), m.group(2)
                if is_key:
                    stats["citing_keys"] += 1
                    current = index.get(tok, -1)
                    stats["citing_in_corpus"] += current >= 0
                else:
                    stats["edges"] += 1
                    j = index.get(tok, -1)
                    if current >= 0 and j >= 0 and j != current:
                        src.append(current)
                        dst.append(j)
            citing_parts.append(np.asarray(src, dtype=np.int32))
            cited_parts.append(np.asarray(dst, dtype=np.int32))
            stats["edges_kept"] += len(src)
            if verbose:
                print(f"  {stats['citing_keys']:,} citing papers, {stats['edges_kept']:,} edges kept "
                      f"({time.time() - t0:.0f}s)", flush=True)
            if not chunk:
                break
    citing = np.concatenate(citing_parts)
    cited = np.concatenate(cited_parts)
    # Drop duplicate edges (a reference listed twice).
    key = citing.astype(np.int64) * len(ids) + cited
    _, first = np.unique(key, return_index=True)
    stats["duplicates_dropped"] = int(len(key) - len(first))
    first.sort()
    return citing[first], cited[first], stats


def load_edges_cached(path, ids, cache_path=None, verbose=True):
    """load_edges with an .npz cache keyed on the adjacency file's size and the paper count."""
    import os
    sig = np.array([os.path.getsize(path), len(ids)], dtype=np.int64)
    if cache_path and os.path.exists(cache_path):
        z = np.load(cache_path)
        if np.array_equal(z["sig"], sig):
            if verbose:
                print(f"  using cached edges from {cache_path}")
            return z["citing"], z["cited"], dict(zip(z["stat_keys"].tolist(), z["stat_vals"].tolist()))
    citing, cited, stats = load_edges(path, ids, verbose=verbose)
    if cache_path:
        np.savez(cache_path, citing=citing, cited=cited, sig=sig,
                 stat_keys=np.array(list(stats)), stat_vals=np.array(list(stats.values()), dtype=np.int64))
    return citing, cited, stats
