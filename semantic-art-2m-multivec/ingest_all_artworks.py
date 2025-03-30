import modal
import logging
import io


# Configure logging
logging.basicConfig(level=logging.INFO)

CACHE_DIR = "/data"
ARTSY_DIR = "/artsy"
MODEL_DIR = "/cache"
BATCH_SIZE = 50000
EMBEDDING_BATCH_SIZE = 5000  # Smaller batches for GPU memory

REGION = "any"

artsy_shard_vol = modal.Volume.from_name("artsy-final")
artsy_vol = modal.Volume.from_name("artsy")
model_volume = modal.Volume.from_name("hf-hub-cache")

stub = modal.App("Artsy-repro")

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

DATA_SPLITS = {}
for i in range(0,31):
    DATA_SPLITS[f"artsy-{i}"] = None

DATA_SPLITS["met"] = ["train"]
DATA_SPLITS["museums"] = ["train"]
DATA_SPLITS["wikiart"] = ["train"]


DATA_PATHS = {
    "met": "met_museum",
    "museums": "art-museums",
    "wikiart": "wikiart"
}
for i in range(0,31):
    DATA_PATHS[f"artsy-{i}"] = f"artsy-small-shards/artsy-{i}" 

DATA_COL_MAPPINGS = {
    "artsy": {
        "artist": "artist.txt",
        "image": "jpg",
        "title": "title.txt",
        "date": "date.txt",
    },
    "wikiart": {
        "artist": "artist",
        "image": "image",
        "title": "description",
        "date": "date"
    },
    "met": {
        "artist": "artist",
        "image": "image",
        "title": "title",
        "date": "accession_year"
    },
    "museums": {
        "artist": ["json", "Author"],
        "image": "jpg",
        "title": ["json", "Title"],
        "date": None
    }
}

DATA_VOLUME_MAPPINGS = {
    "artsy": artsy_shard_vol,
    "met": artsy_vol,
    "museums": artsy_vol,
    "wikiart": artsy_vol
}

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
        import os
        from datasets import load_from_disk, load_dataset
        import open_clip
        import torch
        import lancedb
        from lancedb.pydantic import Vector, LanceModel

        # Debug volume mounting
        
        #self.ds = {}

        #for data, path in DATA_PATHS.items():
        #    if data == "artsy":
        #        pass
        #        #self.ds[data] = load_dataset(
        #        #                f"{CACHE_DIR}/{DATA_PATHS[data]}",
        #        #                split="train",
        #        #                cache_dir=CACHE_DIR,
        #        #                streaming=True
        #        #            )
        #    else:
        #        self.ds[data] = load_from_disk(f"{CACHE_DIR}/{path}")
        #    logging.info(f"Loaded dataset {data} from {path}")

        self.model_clip, _, _ = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-B-32-256x256-DataComp-s34B-b86K')

        self.model_siglip, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-16-SigLIP2-256",
            pretrained=f"{MODEL_DIR}/models--timm--ViT-L-16-SigLIP2-256/snapshots/da9426945c7c5acbd3afd0c158c8bb9cfd4d8cc0/open_clip_pytorch_model.bin"
        )
        self.model_clip = self.model_clip.cuda()
        self.model_clip.eval()

        self.model_siglip = self.model_siglip.cuda()
        self.model_siglip.eval()

        class SCHEMA(LanceModel):
            title: str
            artist: str
            ds_name: str
            date: str
            image: bytes
            vector_clip: Vector(512)
            vector_siglip: Vector(1024)

        # Connect to database

        db = lancedb.connect(
        uri="db://wikipedia-test-9cusod",
        api_key="sk_BE62FD4WVFDSTHMNGEFRUNPIYQX6NTVNHVZ7ARONZF2BAWT26OJA====",
        region="us-east-1"
        )

        # Devrel samp

        #db = lancedb.connect(
        #uri="db://devrel-samp-9a5467",
        #api_key="sk_43CPSRAJXRELJFTZIGSBUHEW6LZYIZO4MWKMLVCUE7JZJS3C3X7A====",
        #region="us-east-1"
        #)
        tbl_name = "artsy_multi_vector_"
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
        from datasets import load_dataset, load_from_disk

        Image.MAX_IMAGE_PIXELS = None
        
        # Unpack the batch arguments
        ds_name = batch_args["ds_name"]
        split_name = batch_args["split_name"]
        start_idx = batch_args["start_idx"]
        end_idx = batch_args["end_idx"]

        if ds_name.startswith("artsy"):
            # Ignore split name for artsy shards
            batch = load_from_disk(f"{ARTSY_DIR}/{DATA_PATHS[ds_name]}", keep_in_memory=False).select(range(start_idx, end_idx))
        else:
            batch = load_from_disk(f"{CACHE_DIR}/{DATA_PATHS[ds_name]}")[split_name].select(range(start_idx, end_idx))
            #batch = self.ds[ds_name][split_name].select(range(start_idx, end_idx))
            
        if not batch:
            return 0
        
        t1 = time.time()
        processed_records = batch.to_pandas().to_dict(orient='records')
        t2 = time.time()
        logging.info(f"Processed {len(processed_records)} records in {t2 - t1} seconds")
        # Process the raw dataset into usable dictionaries
        # Debugging: Check the first few records

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
                        # Save resized image as bytes
                        resized_bytes = io.BytesIO()
                        img_pil.save(resized_bytes, format='JPEG', quality=95)
                        rec["image"] = resized_bytes.getvalue()  # Store resized image bytes
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
                print(f"Record keys ", record.keys())
                print(f"data types in the record")
                for key, value in record.items():
                    print(f"{key}: {type(value)}")
                exit()
                
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
                    embeddings_clip = self.model_clip.encode_image(image_batch)
                    embeddings_clip = embeddings_clip / embeddings_clip.norm(dim=-1, keepdim=True)
                    embeddings_clip = embeddings_clip.cpu().numpy()

                    embeddings_siglip = self.model_siglip.encode_image(image_batch)
                    embeddings_siglip = embeddings_siglip / embeddings_siglip.norm(dim=-1, keepdim=True)
                    embeddings_siglip = embeddings_siglip.cpu().numpy()

                embeddings = np.concatenate([embeddings_clip, embeddings_siglip], axis=1)
                # Convert to list of vectors
                embeddings = [vector.tolist() for vector in embeddings]
                # Add embeddings directly to embed_chunk records
                for record, embedding in zip(embed_chunk, embeddings):
                    record["vector_clip"] = embedding[:512]
                    record["vector_siglip"] = embedding[512:]

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
    
    # Debug volume mounting
    if ds_name.startswith("artsy"):
        logging.info(f"Checking directory contents: {os.listdir(f'{ARTSY_DIR}/{DATA_PATHS[ds_name]}')}")
        ds = load_from_disk(f"{ARTSY_DIR}/{DATA_PATHS[ds_name]}")
    else:
        logging.info(f"Checking directory contents: {os.listdir(f'{CACHE_DIR}/{DATA_PATHS[ds_name]}')}")
        ds = load_from_disk(f"{CACHE_DIR}/{DATA_PATHS[ds_name]}")[split_name]
    total_size = len(ds)
    return [(split_name, i, min(i + BATCH_SIZE, total_size)) 
            for i in range(0, total_size, BATCH_SIZE)]

'''
@stub.local_entrypoint()
def main():
    total_processed = 0
    processor = EmbeddingProcessor()
    batch_args = []
    for data in DATA_SPLITS.keys():
        for split in DATA_SPLITS[data] if not data.startswith("artsy") else [None]:
            print(f"\nProcessing split: {split} in dataset: {data}")
            try:
                batch_indices = get_batch_indices.remote(split, data)
                batch_args.extend([
                        {"split_name": split, "start_idx": start, "end_idx": end, "ds_name": data} 
                        for split, start, end in batch_indices
                    ])
            except Exception as e:
                print(f"Error getting batch indices: {e}")
                continue
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
'''

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