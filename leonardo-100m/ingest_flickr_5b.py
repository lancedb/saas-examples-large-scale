import modal
import logging
import io
import requests
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)

CACHE_DIR = "/data"
ARTSY_DIR = "/artsy"
MODEL_DIR = "/cache"
BATCH_SIZE = 50000
EMBEDDING_BATCH_SIZE = 5000  # Smaller batches for GPU memory

HF_DATASET_NAME = "bigdata-pw/leonardo"
MAX_IMG_DOWNLOAD_RETRIES = 1 # You can change this to a higher number, but it will take longer

data_vol = modal.Volume.from_name("leonardo-shards")

stub = modal.App("marqo-gs-embed-ingest-class")

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


# Add near the top of the file with other imports

# Add this constant near other constants
# Add near the top with other constants
SHARD_SIZE = 1_000_000  # 1M rows per shard
TOTAL_ROWS = 958_000_000
NUM_SHARDS = TOTAL_ROWS // SHARD_SIZE + (1 if TOTAL_ROWS % SHARD_SIZE != 0 else 0)

HF_TOKEN = "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"  # Replace with your actual token



@stub.cls(
    image=image,
    gpu="A100",
    timeout=86400,
    memory=100000,
    cpu=32,
    max_containers=6,
    volumes={
        CACHE_DIR: data_vol,
        }
)
class EmbeddingProcessor:
    def __init__(self):
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

        # Connect to database

        db = lancedb.connect(
            uri="db://wikipedia-test-9cusod",
            api_key="sk_CYNOLZVOO5GP3AKZIAX24Q2OGISYW4PJ4NMYYUEZNVXO6OW4T5LA====",
            region="us-east-1"
            )

        # Devrel samp

        #db = lancedb.connect(
        #uri="db://devrel-samp-9a5467",
        #api_key="sk_43CPSRAJXRELJFTZIGSBUHEW6LZYIZO4MWKMLVCUE7JZJS3C3X7A====",
        #region="us-east-1"
        #)
        tbl_name = "flickr-test"
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
        login(token=HF_TOKEN)

    @modal.method()
    def process_batch(self, batch_args: dict):
        import os
        import torch
        import time
        import numpy as np
        import pandas as pd
        from PIL import Image
        from concurrent.futures import ThreadPoolExecutor
        from datasets import load_dataset, load_from_disk
        from requests.adapters import HTTPAdapter
        from urllib3.util import Retry
        from PIL import UnidentifiedImageError
        import aiohttp
        import asyncio

        Image.MAX_IMAGE_PIXELS = None
        
        # Unpack the batch arguments - now just need shard_idx
        shard_idx = batch_args["shard_idx"]
        shard_path = os.path.join(CACHE_DIR, "leonardo_shards", f"shard_{shard_idx:05d}")

        t1 = time.time()
        try:
            # Load the specific shard
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
                    # Download image
                    response = requests.get(record['url'], timeout=30)
                    if response.status_code == 200:
                        content_type = response.headers.get('content-type', '')
                        if 'image' in content_type.lower():
                            # Process image
                            img_bytes = io.BytesIO(response.content)
                            img_pil = Image.open(img_bytes).convert('RGB').resize((224, 224), Image.Resampling.LANCZOS)
                            
                            # Prepare record and image array
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
                        # Log only the final failure
                        if retries == 1:
                            logging.info(f"Image not found or rate limited ({response.status_code}): {record['url']}")
                        return None, None
                except Exception as e:
                    # Log only the final failure
                    if retries == 1:
                        logging.warning(f"Error downloading {record['url']}: {str(e)}")
                
                retries -= 1

            logging.warning(f"Failed to download after retries: {record['url']}")
            return None, None

        # Process in embedding-friendly chunks
        for embed_chunk in chunk_records(processed_records, EMBEDDING_BATCH_SIZE):
            try:
                t1 = time.time()
                
                # Process records in parallel using ThreadPool
                with ThreadPoolExecutor(max_workers=100) as executor:
                    results = list(executor.map(process_single_record, embed_chunk))
                
                # Separate successful results
                images = []
                valid_records = []
                for img_array, record in results:
                    if img_array is not None and record is not None:
                        images.append(img_array)
                        valid_records.append(record)

                embed_chunk = valid_records

                if not images:
                    continue
                
                # Convert numpy arrays to batch tensor
                image_batch = np.stack(images)  # [B, H, W, C]
                image_batch = image_batch.transpose((0, 3, 1, 2))  # [B, C, H, W]
                image_batch = torch.from_numpy(image_batch).float().cuda()  # Convert to float32
                image_batch = image_batch / 255.0  # Normalize to [0,1] range
                
                # Apply ImageNet normalization
                mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device='cuda').view(1, 3, 1, 1)
                std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device='cuda').view(1, 3, 1, 1)
                image_batch = (image_batch - mean) / std
                
                # Apply preprocessing in batch
                #image_batch = self.preprocess(image_batch)
                    
                t2 = time.time()
                logging.info(f"Gathering for embedding {len(embed_chunk)} records in {t2 - t1} seconds")
                # Generate embeddings and add to records
                t1 = time.time()
                #image_batch = torch.cat(images).cuda()
                with torch.no_grad(), torch.autocast("cuda"):
                    embeddings_clip = self.model_clip.encode_image(image_batch)
                    embeddings_clip = embeddings_clip / embeddings_clip.norm(dim=-1, keepdim=True)
                    embeddings_clip = embeddings_clip.cpu().numpy()

                    #embeddings_siglip = self.model_siglip.encode_image(image_batch)
                    #embeddings_siglip = embeddings_siglip / embeddings_siglip.norm(dim=-1, keepdim=True)
                    #embeddings_siglip = embeddings_siglip.cpu().numpy()

                #embeddings = np.concatenate([embeddings_clip, embeddings_siglip], axis=1)
                embeddings = embeddings_clip
                # Convert to list of vectors
                embeddings = [vector.tolist() for vector in embeddings]
                # Add embeddings directly to embed_chunk records
                for record, embedding in zip(embed_chunk, embeddings):
                    record["vector_clip"] = embedding
                    #record["vector_clip"] = embedding[:512]
                    #record["vector_siglip"] = embedding[512:]

                t2= time.time()
                logging.info(f"Embedding {len(embed_chunk)} records in {t2 - t1} seconds")
                # Insert in chunks
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
                        # Try smaller sub-chunks
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
def main():
    total_processed = 0
    processor = EmbeddingProcessor()
    
    start_shard = 25  # Start from shard 35
    print(f"Starting processing from shard {start_shard} to {NUM_SHARDS} shards")
    
    try:
        # Generate shard arguments starting from shard 35
        batch_args = [
            {"shard_idx": idx}
            for idx in range(start_shard, NUM_SHARDS)
        ]
            
        print(f"\nScheduling {len(batch_args)} shards for parallel processing")
        
        # Process all shards in parallel
        results = processor.process_batch.map(batch_args, return_exceptions=True)
        
        # Track progress (adjusted for starting from shard 35)
        for i, result in enumerate(results, start=start_shard):
            if isinstance(result, Exception):
                print(f"Error in shard {i}: {result}")
            elif result > 0:
                total_processed += result
                print(f"Processed shard {i}, records: {result:,}")
                remaining_rows = TOTAL_ROWS - (start_shard * SHARD_SIZE)
                progress = (total_processed/(remaining_rows))*100
                print(f"Progress: {total_processed:,}/{remaining_rows:,} remaining records ({progress:.2f}%)")
            
    except Exception as e:
        print(f"Error processing dataset: {e}")
    
    print(f"Processing completed! Total records ingested: {total_processed:,}")

#if __name__ == "__main__":
#    with app.run():
#        main.call()