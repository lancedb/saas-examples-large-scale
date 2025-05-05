import modal
import logging
import io


logging.basicConfig(level=logging.INFO)
logging.getLogger('pyvips').setLevel(logging.WARNING)

CACHE_DIR = "/data"
ARTSY_DIR = "/artsy"
MODEL_DIR = "/cache"
BATCH_SIZE = 50000
EMBEDDING_BATCH_SIZE = 5000  # Smaller batches for GPU memory


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
            "open-clip-torch",
            "accelerate>=0.26.0"
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

MIN_CAPTION_LEN = 100

@stub.cls(
    image=image,
    gpu="H100",
    timeout=86400,
    memory=100000,
    cpu=24,
    max_containers=9,
    volumes={
        ARTSY_DIR: artsy_shard_vol,
        CACHE_DIR: artsy_vol,
        MODEL_DIR: model_volume
        },
)
class EmbeddingProcessor:
    def __init__(self):
        import os
        from datasets import load_from_disk, load_dataset
        import open_clip
        import torch
        import lancedb
        from lancedb.pydantic import Vector, LanceModel
        from transformers import Blip2Processor, Blip2ForConditionalGeneration

        # CLIP & SigLIP models for image embeddings
        self.model_clip, _, _ = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-B-32-256x256-DataComp-s34B-b86K')
        self.tokenize_clip  = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-B-32-256x256-DataComp-s34B-b86K')

        self.model_siglip, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-16-SigLIP2-256",
            pretrained=f"{MODEL_DIR}/models--timm--ViT-L-16-SigLIP2-256/snapshots/da9426945c7c5acbd3afd0c158c8bb9cfd4d8cc0/open_clip_pytorch_model.bin"
        )
        self.tokenize_siglip = open_clip.get_tokenizer('ViT-L-16-SigLIP2-256')
        
        # BLIP-2 for caption generation with batch support
        self.processor_caption = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model_caption = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", 
            torch_dtype=torch.float16,
            device_map="cuda"
        )
        self.model_caption.eval()
        # Move models to GPU
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
            caption: str
            vector_clip: Vector(512)
            vector_siglip: Vector(1024)
            vector_clip_caption: Vector(512)
            vector_siglip_caption: Vector(1024)

        # Connect to database
        db = lancedb.connect(
            uri="db://wikipedia-test-9cusod",
            api_key=os.environ["LANCEDB_API_KEY"],
            region="us-east-1"
        )
        
        tbl_name = "artsy_multi_vector_caption_large"
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
    
    def generate_captions_batch(self, images, batch_size=4):
        """Generate captions for images in batches
        
        This implementation is based on the official BLIP-2 examples from Hugging Face.
        Batch size should be adjusted based on available GPU memory.
        """
        import torch
        import logging
        from PIL import Image
        
        captions = []
        total_images = len(images)
        
        # Define the prompt for artistic image description
        prompt = """
        Question: Describe this artwork in detail, focusing on artistic elements such as composition,
        style, colors, techniques, and visual impact. Answer:"""
        
        for i in range(0, total_images, batch_size):
            batch_end = min(i + batch_size, total_images)
            batch_images = images[i:batch_end]
            current_batch_size = len(batch_images)
            
            # Convert batch of numpy arrays to PIL images
            pil_images = [Image.fromarray(img) for img in batch_images]
            
            # Process images in batch (with appropriate prompt)
            inputs = self.processor_caption(
                images=pil_images,
                #text=[prompt] * current_batch_size,
                return_tensors="pt"
            ).to(self.model_caption.device)
            
            # Generate captions for entire batch at once
            with torch.no_grad(), torch.cuda.amp.autocast():  # Use mixed precision
                generated_ids = self.model_caption.generate(
                    **inputs,
                    max_new_tokens=100,
                    num_beams=3,
                    min_length=MIN_CAPTION_LEN,
                    early_stopping=True
                )
            
            # Decode the generated IDs to text - THIS IS CRITICAL
            batch_captions = self.processor_caption.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )
            
            # Append captions to the list
            captions.extend(batch_captions)
            
            if (batch_end) % 50 == 0 or batch_end == total_images: 
                logging.info(f"Sample caption: {batch_captions[0][:100]}...")
                logging.info(f"Generated captions for {batch_end}/{total_images} images")
                    
        return captions

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
            
        if not batch:
            return 0
        
        t1 = time.time()
        processed_records = batch.to_pandas().to_dict(orient='records')
        t2 = time.time()
        logging.info(f"Processed {len(processed_records)} records in {t2 - t1} seconds")

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
                
                t2 = time.time()
                logging.info(f"Gathering for embedding {len(embed_chunk)} records in {t2 - t1} seconds")
                
                # Generate embeddings and add to records
                t1 = time.time()
                with torch.no_grad(), torch.autocast("cuda"):
                    # Generate image embeddings
                    embeddings_clip = self.model_clip.encode_image(image_batch)
                    embeddings_clip = embeddings_clip / embeddings_clip.norm(dim=-1, keepdim=True)
                    embeddings_clip = embeddings_clip.cpu().numpy()

                    embeddings_siglip = self.model_siglip.encode_image(image_batch)
                    embeddings_siglip = embeddings_siglip / embeddings_siglip.norm(dim=-1, keepdim=True)
                    embeddings_siglip = embeddings_siglip.cpu().numpy()

                    # Generate captions using batch processing
                    # Adjust batch size as needed based on GPU memory
                    caption_batch_size = 64  
                    captions = self.generate_captions_batch(images, batch_size=caption_batch_size)

                    # Batch process captions
                    caption_tokens_clip = self.tokenize_clip(captions)
                    caption_tokens_siglip = self.tokenize_siglip(captions)

                    # Encode all captions at once
                    caption_embeddings_clip = self.model_clip.encode_text(caption_tokens_clip.cuda())
                    caption_embeddings_clip = caption_embeddings_clip / caption_embeddings_clip.norm(dim=-1, keepdim=True)
                    caption_embeddings_clip = caption_embeddings_clip.cpu().numpy()

                    caption_embeddings_siglip = self.model_siglip.encode_text(caption_tokens_siglip.cuda())
                    caption_embeddings_siglip = caption_embeddings_siglip / caption_embeddings_siglip.norm(dim=-1, keepdim=True)
                    caption_embeddings_siglip = caption_embeddings_siglip.cpu().numpy()

                # Add all embeddings and captions to records
                for idx, record in enumerate(embed_chunk):
                    record["vector_clip"] = embeddings_clip[idx].tolist()
                    record["vector_siglip"] = embeddings_siglip[idx].tolist()
                    record["caption"] = captions[idx]
                    record["vector_clip_caption"] = caption_embeddings_clip[idx].tolist()
                    record["vector_siglip_caption"] = caption_embeddings_siglip[idx].tolist()

                t2 = time.time()
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
    cpu=16,
    max_containers=35,
    memory=100000,
    timeout=86400,
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