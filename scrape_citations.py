import os
import re
import json
import time
import random
import requests
import pandas as pd

S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
INPUT_CSV = "data/arXiv_full_dataset.csv"
CHECKPOINT_JSONL = "s2_checkpoint.jsonl"
FINAL_CSV = "arXiv_full_dataset_with_citations.csv"
FINAL_ADJ_JSON = "citation_graph_adjacency_full.json"

# Steady token-bucket spacing for 500-ID batches requesting nested references.externalIds
MIN_INTERVAL = 5.5
_LAST_REQ_TS = 0.0
_SESSION = requests.Session()


def clean_arxiv_id(raw_id):
    """Sanitizes modern and pre-2007 legacy arXiv IDs without mangling hyphenated prefixes."""
    if pd.isna(raw_id) or not raw_id:
        return None
    raw_str = re.sub(r"v\d+$", "", str(raw_id).strip())

    # Legacy format with slash already present: hep-th/9901001, math.AG/0601001
    legacy_slash = re.fullmatch(r"([a-zA-Z\-]+(?:\.[a-zA-Z\-]+)?)/(\d{7})", raw_str)
    if legacy_slash:
        return f"ARXIV:{legacy_slash.group(1)}/{legacy_slash.group(2)}"

    # Legacy format stored with hyphen instead of slash: cs-9308101 -> cs/9308101
    legacy_hyphen = re.fullmatch(r"([a-zA-Z]+)-(\d{7})", raw_str)
    if legacy_hyphen:
        return f"ARXIV:{legacy_hyphen.group(1)}/{legacy_hyphen.group(2)}"

    # Modern format: 2103.12345 or 2103.1234
    modern_match = re.fullmatch(r"\d{4}\.\d{4,5}", raw_str)
    if modern_match:
        return f"ARXIV:{modern_match.group(0)}"

    return None


def _paced_post(url, params, json_payload, headers, timeout=45):
    global _LAST_REQ_TS
    elapsed = time.monotonic() - _LAST_REQ_TS
    wait = (MIN_INTERVAL - elapsed) + random.uniform(0.10, 0.40)
    if wait > 0:
        time.sleep(wait)
    _LAST_REQ_TS = time.monotonic()
    return _SESSION.post(url, params=params, json=json_payload, headers=headers, timeout=timeout)


def fetch_s2_batch_bisect(paper_identifiers, api_key=None):
    valid_ids = [pid for pid in paper_identifiers if pid]
    if not valid_ids:
        return {}

    headers = {"x-api-key": api_key} if api_key else {}
    params = {"fields": "paperId,externalIds,citationCount,references.externalIds"}
    backoff = 5.0  # Starts at 5s instead of 10s so transient bursts recover fast

    while True:
        try:
            response = _paced_post(S2_BATCH_URL, params, {"ids": valid_ids}, headers, timeout=45)

            if response.status_code == 200:
                return {
                    qid: pdata
                    for qid, pdata in zip(valid_ids, response.json())
                    if isinstance(pdata, dict)
                }

            elif response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", backoff))
                sleep_for = max(retry_after, backoff) + random.uniform(0.5, 1.5)
                print(f"\n[429 Rate Limit] Sleeping {sleep_for:.1f}s...", flush=True)
                time.sleep(sleep_for)
                backoff = min(backoff * 2, 120.0)

            elif response.status_code == 400:
                if len(valid_ids) == 1:
                    return {}
                mid = len(valid_ids) // 2
                left = fetch_s2_batch_bisect(valid_ids[:mid], api_key=api_key)
                right = fetch_s2_batch_bisect(valid_ids[mid:], api_key=api_key)
                left.update(right)
                return left

            else:
                print(f"\n[HTTP {response.status_code}] Retrying in 5s...", flush=True)
                time.sleep(5.0)

        except Exception as e:
            print(f"\nRequest exception: {e} — retrying in 5s...", flush=True)
            time.sleep(5.0)


def process_citation_enrichment(df, api_key=None, batch_size=500):
    df = df.copy()
    df["s2_paper_id"] = df["id"].apply(clean_arxiv_id)

    # 1. Load existing JSONL checkpoint if resuming
    cached_records = {}
    if os.path.exists(CHECKPOINT_JSONL):
        with open(CHECKPOINT_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                cached_records[rec["s2_paper_id"]] = rec
        print(f"Resumed {len(cached_records):,} papers from '{CHECKPOINT_JSONL}'.")

    # 2. Identify uncached valid IDs
    all_valid_ids = [pid for pid in df["s2_paper_id"].dropna().unique() if pid not in cached_records]
    total_batches = (len(all_valid_ids) + batch_size - 1) // batch_size
    print(f"Fetching {len(all_valid_ids):,} remaining papers across {total_batches:,} batches (batch_size={batch_size})...")

    start_ts = time.monotonic()

    with open(CHECKPOINT_JSONL, "a", encoding="utf-8") as ckpt_f:
        for b_idx in range(total_batches):
            batch_ids = all_valid_ids[b_idx * batch_size : (b_idx + 1) * batch_size]
            batch_map = fetch_s2_batch_bisect(batch_ids, api_key=api_key)

            for qid in batch_ids:
                pdata = batch_map.get(qid)
                if pdata:
                    c_count = pdata.get("citationCount", 0) or 0
                    refs = pdata.get("references", []) or []
                    target_arxiv_ids = []
                    for ref in refs:
                        ext_ids = (ref or {}).get("externalIds") or {}
                        arxiv_ref = ext_ids.get("ArXiv") or ext_ids.get("arxiv")
                        if arxiv_ref:
                            clean_ref = clean_arxiv_id(arxiv_ref)
                            if clean_ref:
                                target_arxiv_ids.append(clean_ref)
                    record = {
                        "s2_paper_id": qid,
                        "citations_count": c_count,
                        "referenced_paper_ids": target_arxiv_ids,
                    }
                else:
                    record = {
                        "s2_paper_id": qid,
                        "citations_count": 0,
                        "referenced_paper_ids": [],
                    }

                cached_records[qid] = record
                ckpt_f.write(json.dumps(record) + "\n")

            if (b_idx + 1) % 10 == 0 or (b_idx + 1) == total_batches:
                ckpt_f.flush()
                elapsed_h = (time.monotonic() - start_ts) / 3600.0
                done_batches = b_idx + 1
                eta_h = (elapsed_h / done_batches) * (total_batches - done_batches)
                print(
                    f"Batch {done_batches:,}/{total_batches:,} "
                    f"({done_batches / total_batches * 100:.1f}%) | "
                    f"Elapsed: {elapsed_h:.2f}h | ETA: {eta_h:.2f}h",
                    end="\r",
                    flush=True,
                )

    # 3. Map checkpointed results back to full DataFrame and export
    df["citations_count"] = df["s2_paper_id"].map(
        lambda x: cached_records[x]["citations_count"] if x in cached_records else 0
    )
    df["referenced_paper_ids"] = df["s2_paper_id"].map(
        lambda x: json.dumps(cached_records[x]["referenced_paper_ids"]) if x in cached_records else "[]"
    )

    citation_graph_adj = {
        qid: rec["referenced_paper_ids"]
        for qid, rec in cached_records.items()
        if rec["referenced_paper_ids"]
    }
    with open(FINAL_ADJ_JSON, "w", encoding="utf-8") as f:
        json.dump(citation_graph_adj, f)

    df.to_csv(FINAL_CSV, index=False)
    print(f"\nDone! Saved '{FINAL_CSV}' and '{FINAL_ADJ_JSON}'.")
    return df


if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV, dtype={"id": str})
    api_key = "s2k-"
    process_citation_enrichment(df, api_key=api_key, batch_size=500)