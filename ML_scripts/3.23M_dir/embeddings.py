import os
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

CSV_PATH = "arXiv_full_data.csv"
OUT_PATH = "scibert_embeddings_full.npy"

MODEL_NAME = "allenai/scibert_scivocab_uncased"
SAMPLE_SIZE = None      # Set to None or len(df) to process full dataset
BATCH_SIZE = 256         # Memory-friendly batch size for CPU
MAX_LENGTH = 256        # Truncate sequence length for ~4x faster execution
CPU_CORES = os.cpu_count() or 4

# Optimize PyTorch CPU threading
torch.set_num_threads(CPU_CORES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Running on device: {DEVICE} (CPU Threads: {CPU_CORES})")

# -----------------------------------------------------------------------------
# 1. REPRESENTATION LEARNING & EMBEDDING GENERATION
# -----------------------------------------------------------------------------
def load_scibert():
    """Load SciBERT model and tokenizer from Hugging Face."""
    print("Loading SciBERT model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    return tokenizer, model

def mean_pooling(model_output, attention_mask):
    """Mean Pooling - Take attention mask into account for correct averaging."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

@torch.inference_mode()
def extract_scibert_embeddings(texts, tokenizer, model, batch_size=BATCH_SIZE):
    """Generates dense contextual vector representations using SciBERT."""
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating SciBERT Embeddings"):
        batch_texts = texts[i : i + batch_size]
        
        encoded_input = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(DEVICE)
        
        model_output = model(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings.append(sentence_embeddings.cpu().numpy())
        
    return np.vstack(embeddings)


def get_embeddings(csv_path, out_path):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    title = df['title'].fillna('')
    abstract = df['summary'].fillna('')

    # Concatenate title + abstract for rich contextual representations
    df['text_for_embedding'] = (title + ". " + abstract).str.strip()

    t1 = time.perf_counter()
    # 1. Generate Embeddings
    tokenizer, model = load_scibert()
    
    scibert_embeddings = extract_scibert_embeddings(
        df['text_for_embedding'].tolist(),
        tokenizer,
        model
    )

    np.save(out_path, scibert_embeddings)
    return scibert_embeddings


if __name__ == "__main__":

    get_embeddings(CSV_PATH, OUT_PATH)