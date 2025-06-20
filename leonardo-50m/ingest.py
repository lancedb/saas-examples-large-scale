import modal
import logging
import io
import requests
import time
import os

logging.basicConfig(level=logging.INFO)

def get_secrets():
    LANCEDB_URI = os.environ.get("LANCEDB_URI")
    LANCEDB_API_KEY = os.environ.get("LANCEDB_API_KEY")
    if not LANCEDB_URI or not LANCEDB_API_KEY:
        raise ValueError("LANCEDB_URI and LANCEDB_API_KEY must be set")
    return [
        modal.Secret.from_dict({"LANCEDB_URI": LANCEDB_URI}),
        modal.Secret.from_dict({"LANCEDB_API_KEY": LANCEDB_API_KEY}),
    ]

CACHE_DIR = "/data"
ARTSY_DIR = "/artsy"
MODEL_DIR = "/cache"
BATCH_SIZE = 50000
EMBEDDING_BATCH_SIZE = 10000  # avoid GPU OOM

HF_DATASET_NAME = "bigdata-pw/leonardo"
MAX_IMG_DOWNLOAD_RETRIES = 1 # You can change this to a higher number, but it will take longer
MAX_CONCURRENT_GPUS = 50

data_vol = modal.Volume.from_name("leonardo-shards")

stub = modal.App("leonardo-embed-ingest", secrets=get_secrets())

# Add to imports at the top
import aiohttp
import asyncio

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install(
            "torch",
            "datasets",
            "Pillow",
            "lancedb",
            "transformers",
            "tqdm",
            "protobuf==5.29.3",
            "sentencepiece",
            "open-clip-torch",
            "requests",
            "huggingface_hub"
         )).env({"HF_HUB_CACHE": CACHE_DIR,
       "HF_HOME": CACHE_DIR,
       })

SHARD_SIZE = 1_000_000 
TOTAL_ROWS = 958_000_000
NUM_SHARDS = TOTAL_ROWS // SHARD_SIZE + (1 if TOTAL_ROWS % SHARD_SIZE != 0 else 0)


@stub.cls(
    image=image,
    gpu="H100",
    timeout=86400,
    memory=100000,
    cpu=32,
    max_containers=MAX_CONCURRENT_GPUS,
    volumes={
        CACHE_DIR: data_vol,
        }
)
class EmbeddingProcessor:
    def __init__(self, table_name: str):
        import os
        from datasets import load_from_disk, load_dataset
        import open_clip
        import torch
        import lancedb
        from lancedb.pydantic import Vector, LanceModel
        from huggingface_hub import login


        self.model_clip, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')


        self.model_clip = self.model_clip.cuda()
        self.model_clip.eval()

        class SCHEMA(LanceModel):
            description: str
            url: str
            image: bytes
            vector_clip: Vector(512)

        db = lancedb.connect(
            uri=os.environ["LANCEDB_URI"],
            api_key=os.environ["LANCEDB_API_KEY"],
            region="us-east-1"
            )

        tbl_name = table_name
        try:
            self.table = db.open_table(tbl_name)
        except Exception as e:
            try:
                logging.info(f"Creating table {tbl_name}")
                self.table = db.create_table(tbl_name, schema=SCHEMA)
            except ValueError as e:
                if "already exists" in str(e):
                    self.table = db.open_table(tbl_name)
                else:
                    raise e


    @modal.method()
    def process_batch(self, batch_args: dict):
        import os
        import torch
        import time
        import numpy as np
        import pandas as pd
        from PIL import Image
        from concurrent.futures import ThreadPoolExecutor
        from datasets import load_from_disk

        Image.MAX_IMAGE_PIXELS = None
        
        shard_idx = batch_args["shard_idx"]
        shard_path = os.path.join(CACHE_DIR, "leonardo_shards", f"shard_{shard_idx:05d}")

        t1 = time.time()
        try:
            dataset = load_from_disk(shard_path)
            processed_records = dataset.to_pandas().to_dict(orient='records')
            t2 = time.time()
            logging.info(f"Loaded shard {shard_idx} with {len(processed_records)} records in {t2 - t1} seconds")
        except Exception as e:
            logging.warning(f"Failed to load shard {shard_idx}: {e}")
            return 0

        if not processed_records:
            logging.warning(f"No records in shard {shard_idx}")
            return 0

        def chunk_records(records, chunk_size):
            for i in range(0, len(records), chunk_size):
                yield records[i:i + chunk_size]

        total_inserted = 0

        def process_single_record(record):
            if not record.get('url'):
                return None, None

            retries = MAX_IMG_DOWNLOAD_RETRIES
            while retries > 0:
                try:
                    response = requests.get(record['url'], timeout=30)
                    if response.status_code == 200:
                        content_type = response.headers.get('content-type', '')
                        if 'image' in content_type.lower():
                            img_bytes = io.BytesIO(response.content)
                            img_pil = Image.open(img_bytes).convert('RGB').resize((224, 224), Image.Resampling.LANCZOS)
                            
                            img_array = np.asarray(img_pil)
                            resized_bytes = io.BytesIO()
                            img_pil.save(resized_bytes, format='JPEG', quality=95, optimize=True)
                            
                            rec = {
                                'description': record.get('prompt', ''),
                                'url': record['url'],
                                'image': resized_bytes.getvalue()
                            }
                            
                            return img_array, rec
                    elif response.status_code in [404, 429]:
                        if retries == 1:
                            logging.info(f"Image not found or rate limited ({response.status_code}): {record['url']}")
                        return None, None
                except Exception as e:
                    if retries == 1:
                        logging.warning(f"Error downloading {record['url']}: {str(e)}")
                
                retries -= 1

            logging.warning(f"Failed to download after retries: {record['url']}")
            return None, None

        for embed_chunk in chunk_records(processed_records, EMBEDDING_BATCH_SIZE):
            try:
                t1 = time.time()
                
                with ThreadPoolExecutor(max_workers=100) as executor: #this is too high, but let it be becoz it works
                    results = list(executor.map(process_single_record, embed_chunk))
                
                images = []
                valid_records = []
                for img_array, record in results:
                    if img_array is not None and record is not None:
                        images.append(img_array)
                        valid_records.append(record)

                embed_chunk = valid_records

                if not images:
                    continue
                
                image_batch = np.stack(images)
                image_batch = image_batch.transpose((0, 3, 1, 2))
                image_batch = torch.from_numpy(image_batch).float().cuda()
                image_batch = image_batch / 255.0 
                
                mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device='cuda').view(1, 3, 1, 1)
                std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device='cuda').view(1, 3, 1, 1)
                image_batch = (image_batch - mean) / std
                
                # Apply preprocessing in batch
                #image_batch = self.preprocess(image_batch)
                    
                t2 = time.time()
                logging.info(f"Gathering for embedding {len(embed_chunk)} records in {t2 - t1} seconds")
                t1 = time.time()
                with torch.no_grad(), torch.autocast("cuda"):
                    embeddings_clip = self.model_clip.encode_image(image_batch)
                    embeddings_clip = embeddings_clip / embeddings_clip.norm(dim=-1, keepdim=True)
                    embeddings_clip = embeddings_clip.cpu().numpy()

                embeddings = embeddings_clip
                embeddings = [vector.tolist() for vector in embeddings]
                # Add embeddings directly to embed_chunk records
                for record, embedding in zip(embed_chunk, embeddings):
                    record["vector_clip"] = embedding

                t2= time.time()
                logging.info(f"Embedding {len(embed_chunk)} records in {t2 - t1} seconds")
                for insert_chunk in chunk_records(embed_chunk, EMBEDDING_BATCH_SIZE):
                    try:
                        t1 = time.time()
                        self.table.add(insert_chunk)
                        t2 = time.time()
                        chunk_size = len(insert_chunk)
                        total_inserted += chunk_size
                        logging.info(f"call table.add() on chunk of size {chunk_size} ({total_inserted}/{len(processed_records)}) in {t2 - t1} seconds")
                    except Exception as e:
                        logging.warning(f"Insert failed, trying smaller batches: {str(e)}")
                        for sub_chunk in chunk_records(insert_chunk, len(insert_chunk)//3):
                            try:
                                self.table.add(sub_chunk)
                                sub_size = len(sub_chunk)
                                total_inserted += sub_size
                                logging.info(f"Ingested sub-chunk of size {sub_size}")
                            except Exception as sub_e:
                                logging.warning(f"Failed to insert sub-chunk: {str(sub_e)}")

            except Exception as e:
                logging.warning(f"Error processing embedding chunk: {e}")

        return total_inserted

@stub.local_entrypoint()
def main(table_name: str = "leonardo-default"):
    total_processed = 0
    processor = EmbeddingProcessor(table_name=table_name)

    start_shard = 0
    print(f"Starting processing from shard {start_shard} to {NUM_SHARDS} shards")
    print(f"Using table: {table_name}")

    try:
        batch_args = [
            {"shard_idx": idx}
            for idx in range(start_shard, NUM_SHARDS)
        ]
            
        print(f"\nScheduling {len(batch_args)} shards for parallel processing")
        t1 = time.time()
        results = processor.process_batch.map(batch_args, return_exceptions=True)
        
        for i, result in enumerate(results, start=start_shard):
            if isinstance(result, Exception):
                print(f"Error in shard {i}: {result}")
            elif result > 0:
                total_processed += result
                print(f"Processed shard {i}, records: {result:,}")
                remaining_rows = TOTAL_ROWS - (start_shard * SHARD_SIZE)
                progress = (total_processed/(remaining_rows))*100
                print(f"Progress: {total_processed:,}/{remaining_rows:,} remaining records ({progress:.2f}%)")
        t2 = time.time()
        print(f"Processed {total_processed} records in  minutes {((t2-t1)/60):.2f}")
    except Exception as e:
        print(f"Error processing dataset: {e}")

    print(f"Processing completed! Total records ingested: {total_processed:,}")

#if __name__ == "__main__":
#    with app.run():
#        main.call()