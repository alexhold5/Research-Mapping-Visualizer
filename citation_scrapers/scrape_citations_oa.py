import os
import re
import gzip
import json
import time
import shutil
import requests
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==============================================================================
# CONFIGURATION
# ==============================================================================
INPUT_CSV = "data/arXiv_full_dataset.csv"
PRIOR_S2_CHECKPOINT_JSONL = "s2_citations_checkpoint.jsonl"

OA_WORK_CACHE_JSONL = "./outputs/oa_s3_arxiv_works.jsonl"
OA_DONE_PARTS_JSON = "./outputs/oa_s3_completed_parts.json"
TMP_SHARD_DIR = "./oa_tmp_shard"

FINAL_CSV = "./outputs/arXiv_with_citations.csv"
FINAL_ADJ_JSON = "./outputs/s2_citation_adjacency.json"

OPENALEX_MANIFEST_URL = "https://openalex.s3.amazonaws.com/data/jsonl/works/manifest.json"
MAX_WORKERS = os.cpu_count() # Adjust based on CPU cores & network bandwidth (4-8 recommended)

# Global worker state initialized once per worker process
_WORKER_TARGET_SET = None


def clean_arxiv_id(raw_id) -> str | None:
    """Normalizes modern, legacy, and URL-prefixed arXiv IDs to 'ARXIV:<id>'."""
    if raw_id is None or (isinstance(raw_id, float) and pd.isna(raw_id)) or not raw_id:
        return None
    raw_str = str(raw_id).strip()
    raw_str = re.sub(r"^https?://arxiv\.org/(?:abs|pdf)/", "", raw_str, flags=re.IGNORECASE)
    raw_str = re.sub(r"^arXiv:", "", raw_str, flags=re.IGNORECASE)
    raw_str = re.sub(r"\.pdf$", "", raw_str, flags=re.IGNORECASE)
    raw_str = re.sub(r"v\d+$", "", raw_str)

    legacy_slash = re.fullmatch(r"([a-zA-Z\-]+(?:\.[a-zA-Z\-]+)?)/(\d{7})", raw_str)
    if legacy_slash:
        return f"ARXIV:{legacy_slash.group(1)}/{legacy_slash.group(2)}"

    legacy_hyphen = re.fullmatch(r"([a-zA-Z]+)-(\d{7})", raw_str)
    if legacy_hyphen:
        return f"ARXIV:{legacy_hyphen.group(1)}/{legacy_hyphen.group(2)}"

    modern_match = re.fullmatch(r"\d{4}\.\d{4,5}", raw_str)
    if modern_match:
        return f"ARXIV:{modern_match.group(0)}"

    return None


def extract_arxiv_from_oa_record(rec: dict) -> str | None:
    """Extracts canonical ARXIV:<id> from OpenAlex 2026 works (doi + locations[])."""
    doi_val = rec.get("doi") or ""
    if "10.48550/arxiv." in doi_val.lower():
        raw_part = doi_val.lower().split("10.48550/arxiv.", 1)[1]
        cid = clean_arxiv_id(raw_part)
        if cid:
            return cid

    locations = rec.get("locations") or []
    prim = rec.get("primary_location")
    if isinstance(prim, dict):
        locations = [prim] + locations

    for loc in locations:
        if not isinstance(loc, dict):
            continue

        loc_id = loc.get("id") or ""
        if "arxiv.org:" in loc_id.lower():
            raw_part = loc_id.lower().split("arxiv.org:", 1)[1]
            cid = clean_arxiv_id(raw_part)
            if cid:
                return cid

        loc_doi = loc.get("doi") or ""
        if "10.48550/arxiv." in loc_doi.lower():
            raw_part = loc_doi.lower().split("10.48550/arxiv.", 1)[1]
            cid = clean_arxiv_id(raw_part)
            if cid:
                return cid

        for url_key in ("landing_page_url", "pdf_url"):
            u = loc.get(url_key) or ""
            if "arxiv.org/" in u.lower():
                cid = clean_arxiv_id(u)
                if cid:
                    return cid

    return None


def s3_uri_to_https(s3_url: str) -> str:
    if s3_url.startswith("s3://openalex/"):
        return "https://openalex.s3.amazonaws.com/" + s3_url[len("s3://openalex/"):]
    return s3_url


def _init_worker(target_set: set[str]):
    global _WORKER_TARGET_SET
    _WORKER_TARGET_SET = target_set


def _process_single_partition(task: tuple[int, str]) -> tuple[str, list[dict]]:
    """Worker: downloads one partition, extracts target arXiv matches, deletes shard."""
    idx, s3_url = task
    https_url = s3_uri_to_https(s3_url)
    local_gz = os.path.join(TMP_SHARD_DIR, f"part_{idx:05d}_{os.getpid()}.gz")
    part_path = local_gz + ".part"

    for attempt in range(1, 6):
        try:
            with requests.get(https_url, stream=True, timeout=180) as r:
                r.raise_for_status()
                with open(part_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            os.replace(part_path, local_gz)
            break
        except Exception as e:
            if os.path.exists(part_path):
                os.remove(part_path)
            if attempt == 5:
                raise RuntimeError(f"Failed downloading {https_url}: {e}")
            time.sleep(2.0 * attempt)

    matches = []
    try:
        with gzip.open(local_gz, "rt", encoding="utf-8") as gz_f:
            for line in gz_f:
                line_low = line.lower()
                if "arxiv.org" not in line_low and "10.48550" not in line_low:
                    continue
                rec = json.loads(line)
                aid = extract_arxiv_from_oa_record(rec)
                if aid and aid in _WORKER_TARGET_SET:
                    oa_short_id = rec["id"].rsplit("/", 1)[-1]
                    ref_short_ids = [
                        r.rsplit("/", 1)[-1]
                        for r in (rec.get("referenced_works") or [])
                        if isinstance(r, str)
                    ]
                    matches.append({
                        "arxiv_id": aid,
                        "oa_id": oa_short_id,
                        "citations_count": int(rec.get("cited_by_count") or 0),
                        "ref_oa_ids": ref_short_ids,
                    })
    finally:
        if os.path.exists(local_gz):
            os.remove(local_gz)

    return s3_url, matches


def main():
    os.makedirs("./outputs", exist_ok=True)
    # Clean any leftover half-downloaded shard from a prior Ctrl+C
    if os.path.exists(TMP_SHARD_DIR):
        shutil.rmtree(TMP_SHARD_DIR, ignore_errors=True)
    os.makedirs(TMP_SHARD_DIR, exist_ok=True)

    # 1. Load target arXiv dataset
    df = pd.read_csv(INPUT_CSV, dtype={"id": str})
    df["s2_paper_id"] = df["id"].apply(clean_arxiv_id)
    target_arxiv_set = set(df["s2_paper_id"].dropna())
    print(f"Target dataset: {len(target_arxiv_set):,} unique canonical arXiv IDs.")

    # 2. Import prior S2 checkpoint if present
    prior_s2_records = {}
    if os.path.exists(PRIOR_S2_CHECKPOINT_JSONL):
        with open(PRIOR_S2_CHECKPOINT_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    prior_s2_records[rec["s2_paper_id"]] = rec
        print(f"Loaded {len(prior_s2_records):,} prior records from '{PRIOR_S2_CHECKPOINT_JSONL}'.")

    # 3. Load resume state
    completed_parts = set()
    if os.path.exists(OA_DONE_PARTS_JSON):
        with open(OA_DONE_PARTS_JSON, "r", encoding="utf-8") as f:
            completed_parts = set(json.load(f))

    oa_id_to_arxiv = {}
    arxiv_to_oa_meta = {}
    if os.path.exists(OA_WORK_CACHE_JSONL):
        with open(OA_WORK_CACHE_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    aid = rec["arxiv_id"]
                    oa_id_to_arxiv[rec["oa_id"]] = aid
                    arxiv_to_oa_meta[aid] = rec
        print(f"Resumed {len(arxiv_to_oa_meta):,} OpenAlex matches across {len(completed_parts):,} completed partitions.")

    # 4. Fetch manifest & sort remaining partitions largest-first for immediate bulk hits
    print("Fetching OpenAlex public works manifest...")
    manifest_resp = requests.get(OPENALEX_MANIFEST_URL, timeout=60)
    manifest_resp.raise_for_status()
    entries = manifest_resp.json().get("files") or manifest_resp.json().get("entries") or []

    remaining_entries = [e for e in entries if e["url"] not in completed_parts]
    remaining_entries.sort(
        key=lambda e: (e.get("meta") or {}).get("content_length", 0),
        reverse=True,
    )
    print(
        f"Manifest total: {len(entries):,} | Already done: {len(completed_parts):,} | "
        f"Remaining: {len(remaining_entries):,} (sorted largest-first, {MAX_WORKERS} workers)"
    )

    # 5. Parallel scan with safe main-process checkpointing
    t0 = time.monotonic()
    done_count = len(completed_parts)
    tasks = [(i, e["url"]) for i, e in enumerate(remaining_entries, 1)]

    with open(OA_WORK_CACHE_JSONL, "a", encoding="utf-8") as cache_f:
        with ProcessPoolExecutor(
            max_workers=MAX_WORKERS,
            initializer=_init_worker,
            initargs=(target_arxiv_set,),
        ) as pool:
            futures = {pool.submit(_process_single_partition, t): t for t in tasks}
            try:
                for fut in as_completed(futures):
                    s3_url, matches = fut.result()
                    new_hits = 0
                    for item in matches:
                        aid = item["arxiv_id"]
                        oa_id_to_arxiv[item["oa_id"]] = aid
                        arxiv_to_oa_meta[aid] = item
                        cache_f.write(json.dumps(item) + "\n")
                        new_hits += 1

                    cache_f.flush()
                    completed_parts.add(s3_url)
                    done_count += 1
                    with open(OA_DONE_PARTS_JSON, "w", encoding="utf-8") as p_f:
                        json.dump(list(completed_parts), p_f)

                    elapsed_m = (time.monotonic() - t0) / 60.0
                    print(
                        f"[{done_count}/{len(entries)}] Matched total: {len(arxiv_to_oa_meta):,} "
                        f"(+{new_hits} in part) | Elapsed: {elapsed_m:.1f}m",
                        end="\r",
                        flush=True,
                    )
            except KeyboardInterrupt:
                print("\nInterrupted safely! Terminating workers and keeping completed checkpoints...")
                pool.shutdown(wait=False, cancel_futures=True)
                shutil.rmtree(TMP_SHARD_DIR, ignore_errors=True)
                return

    print(f"\nScan complete: {len(arxiv_to_oa_meta):,} arXiv papers matched in OpenAlex.")

    # 6. Resolve internal arXiv -> arXiv citation edges & merge with prior S2 records
    final_records = {}
    citation_graph_adj = {}

    for aid in target_arxiv_set:
        if aid in arxiv_to_oa_meta:
            oa_rec = arxiv_to_oa_meta[aid]
            resolved_refs = sorted({
                oa_id_to_arxiv[ref_w]
                for ref_w in oa_rec["ref_oa_ids"]
                if ref_w in oa_id_to_arxiv and oa_id_to_arxiv[ref_w] != aid
            })
            if aid in prior_s2_records:
                resolved_refs = sorted(set(resolved_refs) | set(prior_s2_records[aid].get("referenced_paper_ids", [])))
            final_records[aid] = {
                "citations_count": oa_rec["citations_count"],
                "referenced_paper_ids": resolved_refs,
            }
            if resolved_refs:
                citation_graph_adj[aid] = resolved_refs
        elif aid in prior_s2_records:
            s2_rec = prior_s2_records[aid]
            refs = sorted(set(s2_rec.get("referenced_paper_ids", [])))
            final_records[aid] = {
                "citations_count": int(s2_rec.get("citations_count") or 0),
                "referenced_paper_ids": refs,
            }
            if refs:
                citation_graph_adj[aid] = refs

    # 7. Export drop-in CSV + adjacency JSON
    df["citations_count"] = df["s2_paper_id"].map(
        lambda x: final_records[x]["citations_count"] if x in final_records else 0
    )
    df["referenced_paper_ids"] = df["s2_paper_id"].map(
        lambda x: json.dumps(final_records[x]["referenced_paper_ids"]) if x in final_records else "[]"
    )

    with open(FINAL_ADJ_JSON, "w", encoding="utf-8") as f:
        json.dump(citation_graph_adj, f)

    df.to_csv(FINAL_CSV, index=False)
    print(f"Done! Exported '{FINAL_CSV}' ({len(df):,} rows) and '{FINAL_ADJ_JSON}' ({len(citation_graph_adj):,} nodes).")


if __name__ == "__main__":
    main()