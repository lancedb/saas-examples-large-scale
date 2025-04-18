import modal
import logging
import time
import numpy as np
from pathlib import Path
import os
from typing import List, Dict

logging.basicConfig(level=logging.INFO)

CACHE_DIR = "/data"

# Best perf with h100 - 11 mins
#BATCH_SIZE = 100_000  # Main processing batch size
#EMBEDDING_BATCH_SIZE = 100_000  # Batch size for embedding
#CHUNK_SIZE = 512
#NUM_CHUNK_CONTAINERS = 300
#NUM_EMBEDDING_CONTAINERS = 50

def get_secrets():
    LANCEDB_URI = os.environ.get("LANCEDB_URI")
    LANCEDB_API_KEY = os.environ.get("LANCEDB_API_KEY")
    if not LANCEDB_URI or not LANCEDB_API_KEY:
        raise ValueError("LANCEDB_URI and LANCEDB_API_KEY must be set")
    return [
        modal.Secret.from_dict({"LANCEDB_URI": LANCEDB_URI}),
        modal.Secret.from_dict({"LANCEDB_API_KEY": LANCEDB_API_KEY})
    ]

volume = modal.Volume.from_name("embedding-wikipedia-lancedb")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "lancedb", "fastapi==0.110.1", "numpy==1.26.4", "datasets", "transformers",
    "sentence-transformers"
)


app = modal.App("wikipedia-processor",
                image=image,
                secrets=get_secrets())




BATCH_SIZE = 25_000  # Main processing batch size
EMBEDDING_BATCH_SIZE = 25_000  # Batch size for embedding
CHUNK_SIZE = 512
NUM_CHUNK_CONTAINERS = 600
NUM_EMBEDDING_CONTAINERS = 50


@app.cls(
    cpu=4,
    memory=64000,
    volumes={CACHE_DIR: volume},
    max_containers=NUM_CHUNK_CONTAINERS,
    timeout=60*60*12,
    #region="us-east"
)
class ChunkProcessor:
    def __init__(self, table_name: str):
        from datasets import load_from_disk, disable_caching
        import lancedb
        from lancedb.pydantic import Vector, LanceModel
        
        disable_caching()
        self.ds = load_from_disk(f"{CACHE_DIR}/wikipedia")["train"]
        self.embedder = WikipediaProcessor(table_name)
        
        class Schema(LanceModel):
            vector: Vector(384)
            identifier: int
            chunk_index: int
            content: str
            url: str
            title: str


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
                    "text": text[i:i + CHUNK_SIZE],
                    "metadata": {
                        "id": doc_id,
                        "url": url,
                        "title": title,
                        "chunk_idx": idx
                    }
                })
        
        # Process in optimized embedding batches
        for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            batch_chunks = chunks[i:i + EMBEDDING_BATCH_SIZE]
            processed += self.embedder.process_chunks.remote(batch_chunks)
            
        return {"processed": processed, "chars": total_chars}

@app.cls(
    gpu="A10",
    timeout=86400,
    memory=64000,
    cpu=24,
    max_containers=NUM_EMBEDDING_CONTAINERS,
    #region="us-east",
)
class WikipediaProcessor:
    def __init__(self, table_name: str = None):
        from sentence_transformers import SentenceTransformer
        import lancedb
        from lancedb.pydantic import Vector, LanceModel
        
        self.model = SentenceTransformer('BAAI/bge-small-en-v1.5', device='cuda')
        self.model.eval()
        
        db = lancedb.connect(
            uri=os.environ["LANCEDB_URI"],
            api_key=os.environ["LANCEDB_API_KEY"],
            region="us-east-1"
        )
        
        table_name = table_name or "Wikipedia-ayush-250GPU-A10"
        try:
            self.table = db.open_table(table_name)
        except Exception:
            class Schema(LanceModel):
                vector: Vector(384)
                identifier: int
                chunk_index: int
                content: str
                url: str
                title: str
            self.table = db.create_table(table_name, schema=Schema)

    @modal.method()
    def process_chunks(self, chunks: List[Dict]) -> int:
        import torch
        import time
        
        texts = [c["text"] for c in chunks]
        
        # Generate embeddings
        t1 = time.time()
        with torch.no_grad():
            embeddings = self.model.encode(texts, convert_to_tensor=True)
            embeddings = embeddings.cpu().numpy()
        t2 = time.time()
        print(f"Embedding took {t2-t1:.1f}s")
        
        # Prepare and insert data
        batch_data = [
            {
                "vector": embedding,
                "identifier": chunk["metadata"]["id"],
                "chunk_index": chunk["metadata"]["chunk_idx"],
                "content": chunk["text"],
                "url": chunk["metadata"]["url"],
                "title": chunk["metadata"]["title"]
            }
            for chunk, embedding in zip(chunks, embeddings)
        ]
        
        t1 = time.time()
        self.table.add(batch_data)
        t2 = time.time()
        print(f"Insertion took {t2-t1:.1f}s")
        
        return len(batch_data)

@app.function(
    volumes={CACHE_DIR: volume},
    timeout=86400,
    memory=32000,
    #region="us-east"
)
def start(down_scale: float = 1.0, table_name: str = "wikipedia"):
    from datasets import load_from_disk, disable_caching
    disable_caching()

    ds = load_from_disk(f"{CACHE_DIR}/wikipedia")
    total_size = len(ds["train"])
    sample_size = int(total_size * down_scale)
    
    # Calculate batch size based on number of containers
    batch_size = (sample_size + NUM_CHUNK_CONTAINERS - 1) // NUM_CHUNK_CONTAINERS
    
    # Create optimized batches for chunking
    indices_batches = [
        list(range(i, min(i + batch_size, sample_size))) 
        for i in range(0, sample_size, batch_size)
    ]
    
    # Process all batches in parallel
    start = time.perf_counter()
    results = list(ChunkProcessor(table_name).process_batch.map(indices_batches, return_exceptions=True))
    
    # Aggregate metrics
    total_processed = sum(r["processed"] for r in results if not isinstance(r, Exception))
    duration = time.perf_counter() - start
    
    print(f"Processed {total_processed:,} chunks")
    print(f"Overall duration: {duration/60:.1f} minutes")

@app.local_entrypoint()
def main(down_scale: float = 1.0, table_name: str = "wikipedia"):
    start.remote(down_scale, table_name)
