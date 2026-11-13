import kagglehub
import pandas as pd
import os
import json

# # Download latest version
path = kagglehub.dataset_download("Cornell-University/arxiv")

# print("Path to dataset files:", path)

json_path = os.path.join(path, "arxiv-metadata-oai-snapshot.json")
output_csv = "arXiv_full_dataset.csv"

records = []
print(f"Streaming ALL papers from {json_path}...")

# 2. Line-by-line streaming without category restrictions
count = 0
with open(json_path, 'r', encoding='utf-8') as f:
    for line in f:
        paper = json.loads(line)
        
        records.append({
            'id': paper.get('id'),
            'title': paper.get('title', '').replace('\n', ' ').strip(),
            'summary': paper.get('abstract', '').replace('\n', ' ').strip(),
            'categories': paper.get('categories', ''),
            'updated_date': paper.get('update_date')
        })
        
        count += 1
        if count % 500000 == 0:
            print(f"Processed {count:,} papers...")

print(f"Total papers extracted: {len(records):,}. Converting to CSV...")

# 3. Export full dataset to CSV
df = pd.DataFrame(records)
df.to_csv(output_csv, index=False)

print(f"Done! Full dataset saved to {output_csv}")