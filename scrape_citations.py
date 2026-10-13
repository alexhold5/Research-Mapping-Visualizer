import os
import re
import json
import time
import requests
import pandas as pd

S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"

def clean_arxiv_id(raw_id):
    """
    Sanitizes raw arXiv IDs into S2-compliant ARXIV:... format.
    """
    if pd.isna(raw_id) or not raw_id:
        return None
    raw_str = str(raw_id).strip()
    
    # Legacy format: cs-9308101 or cs/9308101 -> ARXIV:cs/9308101
    legacy_match = re.search(r'([a-zA-Z\-]+(?:\.[a-zA-Z\-]+)?)[/\-]([0-9]{7})(?:v\d+)?', raw_str)
    if legacy_match:
        cat, num = legacy_match.group(1), legacy_match.group(2)
        cat_normalized = cat.replace('-', '/')
        return f"ARXIV:{cat_normalized}/{num}"

    # Modern format: 2103.12345 -> ARXIV:2103.12345
    modern_match = re.search(r'(\d{4}\.\d{4,5})(?:v\d+)?', raw_str)
    if modern_match:
        return f"ARXIV:{modern_match.group(1)}"

    return None

def fetch_s2_batch(paper_identifiers, api_key=None):
    """
    Queries S2 Batch API requesting citation counts and reference external IDs.
    """
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    params = {"fields": "paperId,title,citationCount,references.externalIds"}
    
    valid_payload_ids = [pid for pid in paper_identifiers if pid and isinstance(pid, str)]
    if not valid_payload_ids:
        return [None] * len(paper_identifiers)

    payload = {"ids": valid_payload_ids}
    
    try:
        response = requests.post(
            S2_BATCH_URL,
            params=params,
            json=payload,
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
            
        elif response.status_code == 400:
            print("\n[400 Bad Request] Malformed ID detected in batch. Retrying batch items individually...")
            results = []
            for pid in valid_payload_ids:
                time.sleep(1.05)
                single_res = requests.post(
                    S2_BATCH_URL, 
                    params=params, 
                    json={"ids": [pid]}, 
                    headers=headers,
                    timeout=15
                )
                if single_res.status_code == 200 and single_res.json():
                    results.append(single_res.json()[0])
                else:
                    results.append(None)
            return results

        elif response.status_code == 429:
            print("\n[429 Rate Limit Hit] Backing off for 10s...")
            time.sleep(10)
            
    except Exception as e:
        print(f"\nRequest exception: {e}")
        
    return [None] * len(paper_identifiers)

def process_citation_enrichment(df, api_key=None, batch_size=100, sample_size=None):
    """
    Processes DataFrame, enriches citation counts and maps arXiv-to-arXiv edges.
    
    Parameters:
        df (pd.DataFrame): Input dataset containing raw arXiv IDs in 'id' column.
        api_key (str): Semantic Scholar API key (optional).
        batch_size (int): Number of IDs per S2 batch request.
        sample_size (int): Optional integer to subsample the first N rows for testing.
    """
    # Slice sample if requested
    if sample_size and isinstance(sample_size, int) and sample_size < len(df):
        print(f"Subsampling dataset to first {sample_size} rows for testing...")
        df = df.iloc[:sample_size].copy()
    else:
        df = df.copy()

    df['s2_paper_id'] = df['id'].apply(clean_arxiv_id)
    
    s2_ids = df['s2_paper_id'].tolist()
    total_papers = len(df)
    
    citation_counts = [0] * total_papers
    referenced_paper_ids = [[] for _ in range(total_papers)]
    citation_graph_adj = {}

    print(f"Starting Semantic Scholar ingestion for {total_papers} papers...")

    for i in range(0, total_papers, batch_size):
        batch_ids = s2_ids[i:i + batch_size]
        batch_indices = list(range(i, min(i + batch_size, total_papers)))
        
        print(f"Processing batch {i // batch_size + 1}/{(total_papers + batch_size - 1) // batch_size}...", end="\r")
        
        api_results = fetch_s2_batch(batch_ids, api_key=api_key)
        
        for idx, paper_data in zip(batch_indices, api_results):
            current_arxiv_id = s2_ids[idx]
            if not current_arxiv_id or not paper_data or not isinstance(paper_data, dict):
                continue

            # 1. Inbound Citation Count
            c_count = paper_data.get('citationCount', 0) or 0
            citation_counts[idx] = c_count

            # 2. Extract arXiv IDs for references
            refs = paper_data.get('references', []) or []
            target_arxiv_ids = []
            
            for ref in refs:
                if not ref or not ref.get('externalIds'):
                    continue
                
                ext_ids = ref.get('externalIds', {})
                arxiv_ref = ext_ids.get('ArXiv') or ext_ids.get('arxiv')
                
                if arxiv_ref:
                    clean_ref = clean_arxiv_id(arxiv_ref)
                    if clean_ref:
                        target_arxiv_ids.append(clean_ref)

            referenced_paper_ids[idx] = target_arxiv_ids
            
            if current_arxiv_id and target_arxiv_ids:
                citation_graph_adj[current_arxiv_id] = target_arxiv_ids

        time.sleep(1.05)

    df['citations_count'] = citation_counts
    df['referenced_paper_ids'] = [json.dumps(refs) for refs in referenced_paper_ids]

    with open("citation_graph_adjacency_1000.json", "w") as f:
        json.dump(citation_graph_adj, f, indent=2)

    print("\nCitation enrichment complete! Saved 'citation_graph_adjacency.json'.")
    return df

if __name__ == "__main__":
    df = pd.read_csv("arXiv_scientific_dataset.csv")
    api_key = "s2k-QtvNlkziQkFUwVVrswv263g1qB4Y1tRx7UdgCKIs"
    
    # Pass sample_size=1000 for quick testing, or sample_size=None for full run
    enriched_df = process_citation_enrichment(df, api_key, sample_size=1000)
    enriched_df.to_csv("arXiv_scientific_dataset_with_citations_1000.csv", index=False)