from typing import Dict, List
import modal
import logging
import time
from config import (
    CACHE_DIR, volume, image, get_secrets, BATCH_SIZE, EMBEDDING_BATCH_SIZE,
    CHUNK_SIZE, NUM_CHUNK_CONTAINERS, NUM_EMBEDDING_CONTAINERS, DEFAULT_TABLE_NAME,
    MODEL_NAME, VECTOR_DIM, LANCEDB_REGION, DATASET_PATH
)
from lancedb_ops import get_or_create_table, add_batch_data
from datasets import load_from_disk, disable_caching

logging.basicConfig(level=logging.INFO)

# Add local sources to the image. Auto-mounting is deprecated 
image = image.add_local_python_source("config", "lancedb_ops") 

app = modal.App("wikipedia-processor",
                image=image,
                secrets=get_secrets())

@app.cls(
    cpu=4,
    memory=64000,
    volumes={CACHE_DIR: volume},
    max_containers=NUM_CHUNK_CONTAINERS,
    timeout=60*60*12,
    #region="us-east" # Using east-east further speeds up the ingestion rate but provisioning takes more time
)
class ChunkProcessor:
    def __init__(self, table_name) -> None:
        disable_caching()
        self.ds = load_from_disk(DATASET_PATH)["train"]
        self.embedder = WikipediaProcessor(table_name=table_name)

    @modal.method()
    def process_batch(self, batch_indices: list):
        subset = self.ds.select(batch_indices)
        chunks = []
        total_chars = 0
        processed = 0
        
        # Generate all chunks first
        for text, doc_id, url, title in zip(subset["text"], subset["id"], 
                                          subset["url"], subset["title"]):
            total_chars += len(text)
            for idx, i in enumerate(range(0, len(text), CHUNK_SIZE)):
                chunks.append({
                    "content": text[i:i + CHUNK_SIZE],
                    "identifier": doc_id,
                    "url": url,
                    "title": title,
                    "chunk_index": idx
                })
        
        # Process in optimized embedding batches of EMBEDDING_BATCH_SIZE
        for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            batch_chunks = chunks[i:i + EMBEDDING_BATCH_SIZE]
            processed += self.embedder.process_chunks.remote(batch_chunks)
            
        return {"processed": processed, "chars": total_chars}

@app.cls(
    gpu="H100",
    timeout=86400,
    memory=64000,
    cpu=24,
    max_containers=NUM_EMBEDDING_CONTAINERS,
    #region="us-east", # Using east-east further speeds up the ingestion rate but provisioning takes more time
)
class WikipediaProcessor:

    def __init__(self, table_name):
        from sentence_transformers import SentenceTransformer
        
        self.model = SentenceTransformer(MODEL_NAME, device='cuda')
        self.model.eval()
        self.table = get_or_create_table(table_name)

    @modal.method()
    def process_chunks(self, chunks: List[Dict]) -> int:
        import torch
        
        texts = [c["content"] for c in chunks]
        
        # Generate embeddings in Batches of EMBEDDING_BATCH_SIZE
        with torch.no_grad():
            embeddings = self.model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
            embeddings = embeddings.cpu().numpy()
        
        return add_batch_data(self.table, chunks, embeddings)

@app.function(
    volumes={CACHE_DIR: volume},
    timeout=86400,
    memory=32000,
    #region="us-east"
)
def start(down_scale: float = 1.0, table_name: str = DEFAULT_TABLE_NAME):
    disable_caching()

    ds = load_from_disk(DATASET_PATH)
    total_size = len(ds["train"])
    sample_size = int(total_size * down_scale)
    
    # Calculate batch size based on number of containers. This is useful to divide the batchaes across 
    # containers well so that most of them get utilized till the end of ingestion jobs
    batch_size = (sample_size + NUM_CHUNK_CONTAINERS - 1) // NUM_CHUNK_CONTAINERS
    
    indices_batches = [
        list(range(i, min(i + batch_size, sample_size))) 
        for i in range(0, sample_size, batch_size)
    ]
    
    # Process all batches in parallel. "map" is a modal util that auto-scales a function
    start = time.perf_counter()
    results = list(ChunkProcessor(table_name=table_name).process_batch.map(indices_batches, return_exceptions=True))
    
    total_processed = sum(r["processed"] for r in results if not isinstance(r, Exception))
    duration = time.perf_counter() - start
    
    print(f"Processed {total_processed:,} chunks")
    print(f"Overall duration: {duration/60:.1f} minutes")

    table = get_or_create_table(table_name)
    table.create_index(metric="dot", vector_column_name="emb", index_type="IVF_PQ")
    table.create_fts_index("title", remove_stop_words=True)
    table.create_fts_index("content", remove_stop_words=True)

@app.local_entrypoint()
def main(down_scale: float = 1.0, table_name: str = DEFAULT_TABLE_NAME):
    start.remote(down_scale, table_name)
