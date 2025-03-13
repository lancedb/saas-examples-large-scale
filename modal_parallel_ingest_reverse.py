import modal
from pathlib import Path
import pandas as pd
import logging
from datetime import datetime
import gc

# Define the Modal stub and image
stub = modal.App("met-art-parallel-ingestion")

# Create image with required dependencies
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install(
            "torch",
            "transformers",
            "Pillow",
            "pandas",
            "lancedb",
            "tqdm",
            "protobuf==5.29.3",
            "sentencepiece"
         ))

# Add at the top level, after imports
log_file = f"modal_ingest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@stub.function(
    image=image,
    gpu="A100-80GB",
    timeout=10000,
    memory=184320,
    cpu=32,
    max_containers=10
)
def process_batch(batch_data):
    # Same processing logic as original process_batch
    import torch
    from transformers import AutoProcessor, AutoModel
    import numpy as np
    from PIL import Image
    import io
    import aiohttp
    import asyncio
    import time
    processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    async def download_images(urls):
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(
            limit=100,
            force_close=True,
            enable_cleanup_closed=True
        )) as session:
            async def fetch_url(url, retries=3):  # Increased retries
                for attempt in range(retries):
                    try:
                        async with session.get(url, timeout=30) as response:  # Increased timeout
                            if response.status == 200:
                                content_type = response.headers.get('content-type', '')
                                if 'image' in content_type.lower():  # Validate content type
                                    return await response.read()
                            await asyncio.sleep(0.1)
                    except Exception as e:
                        if attempt == retries - 1:
                            logging.error(f"Error downloading {url}: {e}")
                        await asyncio.sleep(1 * (attempt + 1))  # Increased backoff
                return None

            tasks = [fetch_url(url) for url in urls]
            return await asyncio.gather(*tasks)

    try:
        image_bytes = asyncio.run(download_images([row['image_url'] for row in batch_data]))
    except Exception as e:
        logging.error(f"Error in batch download: {e}")
        image_bytes = [None] * len(batch_data)
    
    images = []
    valid_indices = []
    
    for i, img_bytes in enumerate(image_bytes):
        if img_bytes:
            try:
                # TODO: relax
                if len(img_bytes) < 100:
                    logging.warning(f"Suspiciously small image data: {len(img_bytes)} bytes")
                    continue
                    
                img = Image.open(io.BytesIO(img_bytes))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                images.append(img)
                valid_indices.append(i)
            except Exception as e:
                logging.error(f"Error processing image {i}: {str(e)}")
                continue

    all_embeddings = []
    if images:
        with torch.no_grad():
            inputs = processor(images=images, return_tensors="pt").to(device)
            valid_embeddings = model.get_image_features(**inputs).detach().cpu().numpy()
            
        valid_idx = 0
        for i in range(len(image_bytes)):
            if i in valid_indices:
                all_embeddings.append(valid_embeddings[valid_idx].tolist())
                valid_idx += 1
            else:
                all_embeddings.append([0.0] * 768) 
    else:
        all_embeddings = [[0.0] * 768] * len(image_bytes)

    records = []
    for record, img_bytes, embedding in zip(batch_data, image_bytes, all_embeddings):
        try:
            records.append({
                "object_id": int(record["id"]),
                "title": str(record["title"]) if record["title"] else "",
                "artist": str(record["artist"]) if record["artist"] else "",
                "date": str(record["date"]) if record["date"] else "",
                "medium": str(record["medium"]) if record["medium"] else "",
                "classification": str(record.get("classification", "")),
                "department": str(record["department"]) if record["department"] else "",
                "culture": str(record["culture"]) if record["culture"] else "",
                "misc": str(record["misc"]) if record["misc"] else "",
                "img_url": str(record["image_url"]) if record["image_url"] else "",
                "img": img_bytes if img_bytes else b"",
                "vector": embedding
            })
        except Exception as e:
            print(f"Error processing record: {e}")
            continue
    
    # Instead of returning records, directly ingest them
    import lancedb
    from lancedb.pydantic import Vector, LanceModel

    class SCHEMA(LanceModel):
        object_id: int
        title: str
        artist: str
        date: str
        medium: str
        classification: str
        department: str
        culture: str
        misc: str
        img_url: str
        img: bytes
        vector: Vector(768)

    try:
        db = lancedb.connect(
            uri="db://devrel-samp-9a5467",
            api_key="sk_THUBNC75R5AYPMMRMUV6SEPWWLSPXY7ZSKVQPUUFVCOOQGYKGUKA====",
            region="us-east-1"
        )
        tbl_name = "artworks-modal-reverse"
        if tbl_name in db.table_names():
            table = db.open_table(tbl_name)
        else:
            table = db.create_table(tbl_name, schema=SCHEMA)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                table.add(records)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    logging.error(f"Failed to ingest after {max_retries} attempts: {str(e)}")
                    raise
                logging.warning(f"Ingestion attempt {attempt + 1} failed: {str(e)}. Retrying...")
                time.sleep(2 * (attempt + 1))  # Exponential backoff
        
        return len(records)
    finally:
        del records
        gc.collect()
        db.close()

@stub.local_entrypoint()
def main():
    # Read dataset
    print("Reading dataset...")
    df = pd.read_csv("data/artworks.csv")
    
    # Prepare batches for parallel processing
    batch_size = 2048
    batches = [df.iloc[i:i + batch_size].to_dict('records') 
               for i in range(0, len(df), batch_size)]
    
    # Reverse the batches order
    batches = batches[::-1]
    
    # Launch parallel processing
    print(f"Processing {len(batches)} batches in reverse order using parallel GPU containers...")
    print("Starting parallel processing...")
    print(f"Monitor progress at: https://modal.com/apps/met-art-parallel-ingestion")
    
    # Process results and ingest to LanceDB
    total_processed = 0
    logging.info("Processing and ingesting batches to LanceDB...")
    
    import time
    for processed_count in process_batch.map(batches, return_exceptions=True):
        if isinstance(processed_count, Exception):
            error_msg = f"Batch processing error: {processed_count}"
            logging.error(error_msg)
            print(error_msg)
        else:
            total_processed += processed_count
            log_msg = f"Processed and ingested {processed_count} records. Total: {total_processed}"
            logging.info(log_msg)
            print(log_msg)
    
    logging.info(f"Completed! Total records ingested: {total_processed}")

if __name__ == "__main__":
    main()