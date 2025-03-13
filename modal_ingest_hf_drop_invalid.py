import modal
from pathlib import Path
import pandas as pd
import logging
from datetime import datetime
import gc

stub = modal.App("met-art-parallel-ingestion")

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

@stub.function(
    image=image,
    gpu="A10",
    timeout=10000,
    memory=100000,
    cpu=16,
    max_containers=8
)
def process_batch(batch_data):
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
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(
                limit=30,  # Reduced from 100 to avoid overwhelming the server
                force_close=True,
                enable_cleanup_closed=True
            ),
        ) as session:
            async def fetch_url(url, retries=5):  # Increased retries
                for attempt in range(retries):
                    try:
                        async with session.get(url, timeout=60) as response:
                            if response.status == 200:
                                content_type = response.headers.get('content-type', '')
                                if 'image' in content_type.lower():
                                    return await response.read()
                            elif response.status == 429:  # Too Many Requests
                                wait_time = 5 * (attempt + 1)
                                logging.warning(f"Rate limited. Waiting {wait_time} seconds...")
                                await asyncio.sleep(wait_time)
                            else:
                                logging.warning(f"HTTP {response.status} for {url}. Will retry...")
                            await asyncio.sleep(0.5)  # Increased delay between attempts
                    except Exception as e:
                        if attempt == retries - 1:
                            logging.error(f"Error downloading {url}: {str(e)}")
                        await asyncio.sleep(2 * (attempt + 1)) 
                return None

            # Process URLs in smaller chunks to avoid overwhelming the server
            chunk_size = 20
            all_results = []
            for i in range(0, len(urls), chunk_size):
                chunk = urls[i:i + chunk_size]
                tasks = [fetch_url(url) for url in chunk]
                results = await asyncio.gather(*tasks)
                all_results.extend(results)
                await asyncio.sleep(1)  # Add delay between chunks
            
            return all_results

    try:
        image_bytes = asyncio.run(download_images([row['image_url'] for row in batch_data]))
    except Exception as e:
        logging.error(f"Error in batch download: {e}")
        image_bytes = [None] * len(batch_data)
    
    images = []
    valid_indices = []
    valid_image_bytes = []
    
    for i, img_bytes in enumerate(image_bytes):
        if img_bytes:
            try:
                if len(img_bytes) < 100:
                    logging.warning(f"Suspiciously small image data: {len(img_bytes)} bytes")
                    continue
                    
                img = Image.open(io.BytesIO(img_bytes))
                
                try:
                    img.verify() 
                    img = Image.open(io.BytesIO(img_bytes))  # Need to reopen after verify
                except Exception as e:
                    logging.error(f"Invalid image data for index {i}: {str(e)}")
                    continue
                
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                images.append(img)
                valid_indices.append(i)
                valid_image_bytes.append(img_bytes)
            except Exception as e:
                logging.error(f"Error processing image {i}: {str(e)}")
                continue

    all_embeddings = []
    if images:
        with torch.no_grad():
            inputs = processor(images=images, return_tensors="pt").to(device)
            valid_embeddings = model.get_image_features(**inputs).detach().cpu().numpy()
            all_embeddings = valid_embeddings.tolist()

    records = []
    for idx, embedding in zip(valid_indices, all_embeddings):
        try:
            record = batch_data[idx]
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
                "img": valid_image_bytes[valid_indices.index(idx)],
                "vector": embedding
            })
        except Exception as e:
            logging.error(f"Error processing record: {e}")
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
        tbl_name = "artworks-modal-reverse-valid-8GPU"
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
    print("Reading dataset...")
    df = pd.read_csv("data/artworks.csv")
    
    # Prepare batches for parallel processing
    batch_size = 1024
    batches = [df.iloc[i:i + batch_size].to_dict('records') 
               for i in range(0, len(df), batch_size)]
    
    # Reverse the batches order
    # batches = batches[::-1]
    
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