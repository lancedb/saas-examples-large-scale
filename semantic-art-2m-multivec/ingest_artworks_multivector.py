import modal
import logging
import io
import os
import time
import numpy as np
import pandas as pd
import torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset, load_from_disk
import open_clip
# Removed lancedb and pydantic imports, now handled in config.py

# Import configurations and helper functions from config.py
from config import (
    CACHE_DIR, ARTSY_DIR, MODEL_DIR, BATCH_SIZE, EMBEDDING_BATCH_SIZE, REGION,
    artsy_shard_vol, artsy_vol, model_volume,
    DATA_SPLITS, DATA_PATHS, DATA_COL_MAPPINGS, DATA_VOLUME_MAPPINGS,
    SCHEMA, get_lancedb_table # Import the schema and table function
)

logging.basicConfig(level=logging.INFO)

# --- Constants and Volume definitions moved to config.py ---

stub = modal.App("Artsy-repro")

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install(
            "torch",
            "datasets",
            "Pillow",
            "lancedb", # Keep lancedb here if other parts of the file use it directly, otherwise can remove
            "transformers",
            "tqdm",
            "protobuf==5.29.3",
            "sentencepiece",
            "open-clip-torch"
         )
         # Add local python source for config.py so Modal can find it
         .add_local_python_source("config.py"))

# --- Dataset Configuration dictionaries moved to config.py ---

# --- LanceDB Schema moved to config.py ---

def _chunk_records(records, chunk_size):
    for i in range(0, len(records), chunk_size):
        yield records[i:i + chunk_size]

@stub.cls(
    image=image,
    gpu="A100-80GB",
    timeout=86400,
    memory=100000,
    cpu=24,
    max_containers=4,
    volumes={
        ARTSY_DIR: artsy_shard_vol,
        CACHE_DIR: artsy_vol,
        MODEL_DIR: model_volume
        },
        region=REGION
)
class EmbeddingProcessor:
    def __init__(self):
        Image.MAX_IMAGE_PIXELS = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # --- Model loading remains here ---
        self.model_clip, _, _ = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-B-32-256x256-DataComp-s34B-b86K')
        self.model_siglip, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-16-SigLIP2-256",
            pretrained=f"{MODEL_DIR}/models--timm--ViT-L-16-SigLIP2-256/snapshots/da9426945c7c5acbd3afd0c158c8bb9cfd4d8cc0/open_clip_pytorch_model.bin"
        )
        self.model_clip = self.model_clip.to(self.device)
        self.model_clip.eval()
        self.model_siglip = self.model_siglip.to(self.device)
        self.model_siglip.eval()

        self.table = get_lancedb_table()

    def _load_data_batch(self, ds_name, split_name, start_idx, end_idx):
        import os
        import torch
        import time
        import numpy as np
        import pandas as pd
        from PIL import Image
        from concurrent.futures import ThreadPoolExecutor
        from datasets import load_dataset, load_from_disk

        Image.MAX_IMAGE_PIXELS = None
        
        ds_name = batch_args["ds_name"]
        split_name = batch_args["split_name"]
        start_idx = batch_args["start_idx"]
        end_idx = batch_args["end_idx"]

        if ds_name.startswith("artsy"):
            # Ignore split name for artsy shards
            batch = load_from_disk(f"{ARTSY_DIR}/{DATA_PATHS[ds_name]}", keep_in_memory=False).select(range(start_idx, end_idx))
        else:
            batch = load_from_disk(f"{CACHE_DIR}/{DATA_PATHS[ds_name]}")[split_name].select(range(start_idx, end_idx))
            
        if not batch:
            return 0
        processed_records = batch.to_pandas().to_dict(orient='records')

        if not processed_records:
            logging.error("No records could be processed from the dataset")
            return 0

        def chunk_records(records, chunk_size):
            for i in range(0, len(records), chunk_size):
                yield records[i:i + chunk_size]

        total_inserted = 0
        def process_single_record(record):
            rec = {}        
            img_array = None
            try:        
                map_key = "artsy" if ds_name.startswith("artsy") else ds_name
                for col, mapping in DATA_COL_MAPPINGS[map_key].items():
                    if mapping is None:
                        rec[col] = " "
                    elif col == "image":
                        img_bytes = io.BytesIO(record[DATA_COL_MAPPINGS[map_key]["image"]]["bytes"])
                        img_pil = Image.open(img_bytes).convert('RGB').resize((256, 256), Image.Resampling.LANCZOS)
                        img_array = np.array(img_pil)
                        resized_bytes = io.BytesIO()
                        img_pil.save(resized_bytes, format='JPEG', quality=95)
                        rec["image"] = resized_bytes.getvalue()  
                    elif isinstance(mapping, str):
                        rec[col] = record[mapping] if record[mapping] else " "
                    else:
                        try:
                            rec[col] = record[mapping[0]][mapping[1]] if record[mapping[0]][mapping[1]] else " "
                        except:
                             rec[col] = " "
                rec["ds_name"] = ds_name
                return img_array, rec
            except Exception as e:
                logging.error(f"Error processing record: {e}")
                return None, None

        for embed_chunk in chunk_records(processed_records, EMBEDDING_BATCH_SIZE):
            try:
                t1 = time.time()
                
                with ThreadPoolExecutor(max_workers=100) as executor:
                    results = list(executor.map(process_single_record, embed_chunk))
                
                images = []
                valid_records = []
                for img_array, record in results:
                    if img_array is not None:
                        images.append(img_array)
                        valid_records.append(record)

                embed_chunk = valid_records 
                
                if not images:
                    continue
                
                image_batch = np.stack(images)  # [B, H, W, C]
                image_batch = image_batch.transpose((0, 3, 1, 2))  # [B, C, H, W]
                image_batch = torch.from_numpy(image_batch).float().cuda() 
                image_batch = image_batch / 255.0  # Normalize to [0,1] range
                
                mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device='cuda').view(1, 3, 1, 1)
                std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device='cuda').view(1, 3, 1, 1)
                image_batch = (image_batch - mean) / std


                with torch.no_grad(), torch.autocast("cuda"):
                    embeddings_clip = self.model_clip.encode_image(image_batch)
                    embeddings_clip = embeddings_clip / embeddings_clip.norm(dim=-1, keepdim=True)
                    embeddings_clip = embeddings_clip.cpu().numpy()

                    embeddings_siglip = self.model_siglip.encode_image(image_batch)
                    embeddings_siglip = embeddings_siglip / embeddings_siglip.norm(dim=-1, keepdim=True)
                    embeddings_siglip = embeddings_siglip.cpu().numpy()

                embeddings = np.concatenate([embeddings_clip, embeddings_siglip], axis=1)
                embeddings = [vector.tolist() for vector in embeddings]
                # Add embeddings directly to embed_chunk records
                for record, embedding in zip(embed_chunk, embeddings):
                    record["vector_clip"] = embedding[:512]
                    record["vector_siglip"] = embedding[512:]

                for insert_chunk in chunk_records(embed_chunk, EMBEDDING_BATCH_SIZE):
                    try:
                        self.table.add(insert_chunk)
                        chunk_size = len(insert_chunk)
                        total_inserted += chunk_size
                    except Exception as e:
                        logging.warning(f"Insert failed {str(e)}")


            except Exception as e:
                logging.error(f"Error processing embedding chunk: {e}")

        return total_inserted

@stub.function(
    image=image,
    volumes={
        ARTSY_DIR: artsy_shard_vol,
        CACHE_DIR: artsy_vol,
        MODEL_DIR: model_volume},
    cpu=50,
    max_containers=35,
    memory=100000,
    timeout=86400,
    region=REGION
)
def get_batch_indices(batch_args: dict):
    from datasets import load_from_disk, load_dataset
    import os
    
    split_name = batch_args["split"]
    ds_name = batch_args["data"]
    
    if ds_name.startswith("artsy"):
        ds = load_from_disk(f"{ARTSY_DIR}/{DATA_PATHS[ds_name]}")
    else:
        ds = load_from_disk(f"{CACHE_DIR}/{DATA_PATHS[ds_name]}")[split_name]
    total_size = len(ds)
    return [(split_name, i, min(i + BATCH_SIZE, total_size)) 
            for i in range(0, total_size, BATCH_SIZE)]


@stub.local_entrypoint()
def main():
    total_processed = 0
    processor = EmbeddingProcessor()
    
    # Prepare arguments for parallel batch indices generation
    map_args = [
        {"split": split, "data": data}
        for data in DATA_SPLITS.keys()
        for split in (DATA_SPLITS[data] if not data.startswith("artsy") else [None])
    ]
    
    print(f"Getting batch indices for {len(map_args)} dataset splits")
    try:
        # Run batch indices generation in parallel
        all_indices = get_batch_indices.map(map_args, return_exceptions=True)
        
        batch_args = []
        for i, (args, indices) in enumerate(zip(map_args, all_indices)):
            if isinstance(indices, Exception):
                print(f"Error getting batch indices for {args['data']}: {indices}")
                continue
                
            print(f"\nProcessing split: {args['split']} in dataset: {args['data']}")
            batch_args.extend([
                {"split_name": args["split"], "start_idx": start, "end_idx": end, "ds_name": args["data"]} 
                for split, start, end in indices
            ])
        
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
        print(f"Error processing splits: {e}")

    print(f"All splits completed! Total records ingested: {total_processed}")

#if __name__ == "__main__":
#    with app.run():
#        main.call()