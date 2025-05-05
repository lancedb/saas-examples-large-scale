import modal
import logging
import io
import time
import numpy as np
import os


# Configure logging
logging.basicConfig(level=logging.INFO)

def get_secrets():
    LANCEDB_URI = os.environ.get("LANCEDB_URI")
    LANCEDB_API_KEY = os.environ.get("LANCEDB_API_KEY")
    if not LANCEDB_URI or not LANCEDB_API_KEY:
        raise ValueError("LANCEDB_URI and LANCEDB_API_KEY must be set")
    return [
        modal.Secret.from_dict({"LANCEDB_URI": LANCEDB_URI}),
        modal.Secret.from_dict({"LANCEDB_API_KEY": LANCEDB_API_KEY})
    ]

CACHE_DIR = "/data"
MODEL_DIR = "/cache"
BATCH_SIZE = 50_000
GATHER_BATCH_SIZE = 10000
EMBEDDING_BATCH_SIZE = 10000
INGEST_BATCH_SIZE = 10000
MAX_CONTAINERS = 50

volume = modal.Volume.from_name("marco-10M")
model_volume = modal.Volume.from_name("hf-hub-cache")

stub = modal.App("marqo-gs-embed-ingest-class", 
                secrets=get_secrets())

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
         ).env(
             
             {
                 "HF_DATASETS_IN_MEMORY_MAX_SIZE": "100000"
             }
         ))

@stub.cls(
    image=image,
    gpu="H100",
    timeout=86400,
    memory=120000,
    cpu=24,
    max_containers=MAX_CONTAINERS,
    volumes={
        CACHE_DIR: volume,
        MODEL_DIR: model_volume},
 region="us-east"
)
class EmbeddingProcessor:
    def __init__(self, table_name: str):
        import os
        from datasets import load_from_disk, disable_caching
        import open_clip
        import torch
        import lancedb
        from lancedb.pydantic import Vector, LanceModel
        disable_caching()
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
            uri=os.environ["LANCEDB_URI"],
            api_key=os.environ["LANCEDB_API_KEY"],
            region="us-east-1"
        )

        tbl_name = table_name
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
        from PIL import Image
        import torch.nn.functional as F

        throughput_metrics = []
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
        
        # Precompute normalization tensors
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device='cuda').view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device='cuda').view(1, 3, 1, 1)

        def process_single_record(record):
            try:
                img_bytes = io.BytesIO(record["image"]["bytes"])
                img_pil = (Image.open(img_bytes)
                          .convert('RGB')
                          .resize((224, 224), Image.Resampling.BILINEAR))  # BILINEAR is faster than default
                img_array = np.asarray(img_pil, dtype=np.float32) / 255.0 
                
                record["image"] = record["image"]["bytes"]
                record["split"] = split_name
                return img_array, record
            except Exception as e:
                logging.error(f"Error processing record: {e}")
                return None, None

        for gather_chunk in chunk_records(processed_records, GATHER_BATCH_SIZE):
            try:
                t1 = time.time()
                
                with ThreadPoolExecutor(max_workers=min(100, len(gather_chunk))) as executor:
                    results = list(executor.map(process_single_record, gather_chunk))
                
                valid_results = [(img, rec) for img, rec in results if img is not None]
                if not valid_results:
                    continue
                    
                images, records = zip(*valid_results)
                records = list(records)
                
                t2 = time.time()
                logging.info(f"Gathering {len(records)} records in {t2 - t1} seconds")

                # Process embeddings in smaller batches
                embedded_records = []
                for embed_idx in range(0, len(records), EMBEDDING_BATCH_SIZE):
                    batch_images = images[embed_idx:embed_idx + EMBEDDING_BATCH_SIZE]
                    batch_records = records[embed_idx:embed_idx + EMBEDDING_BATCH_SIZE]
                    
                    t1 = time.time()
                    image_batch = (torch.from_numpy(np.stack(batch_images))
                                 .permute(0, 3, 1, 2)
                                 .cuda())
                    
                    image_batch = (image_batch - mean) / std
                    
                    with torch.no_grad(), torch.autocast("cuda"):
                        embeddings = self.model.encode_image(image_batch)
                        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                        embeddings = embeddings.cpu().numpy()

                    # Add embeddings to records
                    for record, embedding in zip(batch_records, embeddings):
                        record["vector"] = embedding
                        embedded_records.append(record)
                    t2 = time.time()
                    logging.info(f"Embedded batch of {len(batch_records)} records in {t2 - t1} seconds")

                for insert_chunk in chunk_records(embedded_records, INGEST_BATCH_SIZE):
                    try:
                        t1 = time.time()
                        self.table.add(insert_chunk)
                        t2 = time.time()
                        chunk_size = len(insert_chunk)
                        chunk_time = t2 - t1
                        throughput = chunk_size / chunk_time
                        throughput_metrics.append(throughput)
                        total_inserted += chunk_size
                        logging.info(f"call table.add() size {chunk_size} ({total_inserted}/{len(processed_records)}) in {chunk_time:.2f} seconds. throughput: {throughput:.2f}")
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

        # Return metrics only if we have data
        avg_throughput = np.mean(throughput_metrics) if throughput_metrics else 0
        return (total_inserted, avg_throughput)

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
def main(table_name: str = "macro-10m-docs-default"):
    splits = [
     'in_domain',
     'novel_document', 
     'novel_query', 
     'zero_shot']
    total_processed = 0

    processor = EmbeddingProcessor(table_name=table_name)
    batch_args = []
    # Hardcode
    split_sizes = {
        'in_domain': 3_930_000,      # 3.93M rows
        'novel_document': 3_920_000,  # 3.92M rows
        'novel_query': 981_000,       # 981k rows
        'zero_shot': 981_000          # 981k rows
    }
    #for split in splits:
    #    print(f"\nProcessing split: {split}")
    #    batch_indices = get_batch_indices.remote(split)
    
    for split, total_size in split_sizes.items():
        print(f"\nProcessing split: {split} with {total_size:,} records")
        batch_args.extend([
            {"split_name": split, "start_idx": i, "end_idx": min(i + BATCH_SIZE, total_size)}
            for i in range(0, total_size, BATCH_SIZE)
        ])
    try:
        t1 = time.time()
        results = processor.process_batch.map(batch_args, return_exceptions=True)

        split_total = 0
        total_throughput = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Error in batch {i}: {result}")
            elif isinstance(result, tuple) and len(result) == 2:
                records, throughput = result
                if records > 0:
                    split_total += records
                    total_throughput.append(throughput)
                    print(f"Processed batch {i}, records: {records}, total: {split_total}, throughput: {throughput:.2f} records/sec")

        total_processed += split_total
        avg_throughput = np.mean(total_throughput) if total_throughput else 0
        print(f"Records ingested: {split_total}, Average throughput: {avg_throughput:.2f} records/sec")
        t2 = time.time()
        print(f"Total time: {(t2 - t1)/60.0} min")

    except Exception as e:
        print(f"Error processing split : {e}")

    print(f"All splits completed! Total records ingested: {total_processed}")
