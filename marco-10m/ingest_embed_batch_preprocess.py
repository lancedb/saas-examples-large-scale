import modal
import logging
import io


# Configure logging
logging.basicConfig(level=logging.INFO)

CACHE_DIR = "/data"
MODEL_DIR = "/cache"
BATCH_SIZE = 200000
EMBEDDING_BATCH_SIZE = 8000 

volume = modal.Volume.from_name("marco-10M")
model_volume = modal.Volume.from_name("hf-hub-cache")

stub = modal.App("marqo-gs-embed-ingest-class")

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
            "open-clip-torch"
         ))

@stub.cls(
    image=image,
    gpu="A100",
    timeout=86400,
    memory=100000,
    cpu=24,
    max_containers=10,
    volumes={
        CACHE_DIR: volume,
        MODEL_DIR: model_volume}
)
class EmbeddingProcessor:
    def __init__(self):
        import os
        from datasets import load_from_disk
        import open_clip
        import torch
        import lancedb
        from lancedb.pydantic import Vector, LanceModel

        # Debug volume mounting
        logging.info(f"Checking directory contents in __enter__: {os.listdir(CACHE_DIR)}")

        self.ds = load_from_disk(f"{CACHE_DIR}/marco-10m")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            #'ViT-L-16-SigLIP2-256',
            #pretrained=f"{MODEL_DIR}/models--timm--ViT-L-16-SigLIP2-256/snapshots/da9426945c7c5acbd3afd0c158c8bb9cfd4d8cc0/open_clip_pytorch_model.bin",
            "ViT-B-32",
            pretrained="laion2b_s34b_b79k",
        )
        self.model = self.model.cuda()
        self.model.eval()

        class SCHEMA(LanceModel):
            product_id: str
            pair_id: str
            no_score: int
            title: str
            split: str
            query: str
            query_id: str
            position: int
            score_linear: int
            score_reciprocal: float
            image: bytes
            vector: Vector(512)

        # Connect to database
        db = lancedb.connect(
        uri="db://wikipedia-test-9cusod",
        api_key="sk_BJIR2QXJ7JAM7AUMTHHPDLGJRY26GM3LN4ONK5TKTTGCFORC4NEQ====",
        region="us-east-1"
        )

        tbl_name = "test-test-test-10M-marco-clip-class-full"
        try:
            self.table = db.open_table(tbl_name)
        except Exception as e:
            try:
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
        from PIL import Image
        from concurrent.futures import ThreadPoolExecutor
        from functools import partial

        # Unpack the batch arguments
        split_name = batch_args["split_name"]
        start_idx = batch_args["start_idx"]
        end_idx = batch_args["end_idx"]

        batch = self.ds[split_name].select(range(start_idx, end_idx))
        if not batch:
            return 0

        # Process the raw dataset into usable dictionaries
        t1 = time.time()
        processed_records = batch.to_pandas().to_dict(orient='records')
        t2 = time.time()
        logging.info(f"Processed {len(processed_records)} records in {t2 - t1} seconds")
        # Debugging: Check the first few records

        if not processed_records:
            logging.error("No records could be processed from the dataset")
            return 0

        def chunk_records(records, chunk_size):
            for i in range(0, len(records), chunk_size):
                yield records[i:i + chunk_size]

        total_inserted = 0

        def process_single_record(record):
            try:
                # Convert image bytes to PIL Image, resize to model's expected size
                img_bytes = io.BytesIO(record["image"]["bytes"])
                img_pil = Image.open(img_bytes).convert('RGB').resize((224, 224), Image.Resampling.LANCZOS)
                img_array = np.array(img_pil)
                
                # Update record (avoid creating new objects)
                record["image"] = record["image"]["bytes"]
                record["split"] = split_name
                
                return img_array, record
            except Exception as e:
                logging.error(f"Error processing record: {e}")
                return None, None

        # Process in embedding-friendly chunks
        for embed_chunk in chunk_records(processed_records, EMBEDDING_BATCH_SIZE):
            try:
                t1 = time.time()
                
                # Process records in parallel
                with ThreadPoolExecutor(max_workers=100) as executor:
                    results = list(executor.map(process_single_record, embed_chunk))
                
                # Separate successful results
                images = []
                valid_records = []
                for img_array, record in results:
                    if img_array is not None:
                        images.append(img_array)
                        valid_records.append(record)
                
                embed_chunk = valid_records  # Update chunk to only valid records
                
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
                    embeddings = self.model.encode_image(image_batch)
                    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                    embeddings = embeddings.cpu().numpy()

                # Add embeddings directly to embed_chunk records
                for record, embedding in zip(embed_chunk, embeddings):
                    record["vector"] = embedding
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
                                logging.error(f"Failed to insert sub-chunk: {str(sub_e)}")

            except Exception as e:
                logging.error(f"Error processing embedding chunk: {e}")

        return total_inserted

@stub.function(
    image=image,
    volumes={CACHE_DIR: volume},
    cpu=8,
    max_containers=3,
    memory=100000,
    timeout=86400
)
def get_batch_indices(split_name: str):
    from datasets import load_from_disk
    import os
    
    # Debug volume mounting
    logging.info(f"Checking directory contents: {os.listdir(CACHE_DIR)}")
    
    ds = load_from_disk(f"{CACHE_DIR}/marco-10m")[split_name]
    total_size = len(ds)
    return [(split_name, i, min(i + BATCH_SIZE, total_size)) 
            for i in range(0, total_size, BATCH_SIZE)]

@stub.local_entrypoint()
def main():
    splits = [
     'in_domain',
     'novel_document', 
     'novel_query', 
     'zero_shot']
    total_processed = 0

    processor = EmbeddingProcessor()
    batch_args = []
    for split in splits:
        print(f"\nProcessing split: {split}")
        batch_indices = get_batch_indices.remote(split)
        batch_args.extend([
                {"split_name": split, "start_idx": start, "end_idx": end} 
                for split, start, end in batch_indices
            ])
    try:
        results = processor.process_batch.map(batch_args, return_exceptions=True)

        split_total = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Error in batch {i}: {result}")
            elif result > 0:
                split_total += result
                print(f"Processed batch {i}, records: {result}, total: {split_total}")

        total_processed += split_total
        print(f"Records ingested: {split_total}")

    except Exception as e:
        print(f"Error processing split : {e}")

    print(f"All splits completed! Total records ingested: {total_processed}")