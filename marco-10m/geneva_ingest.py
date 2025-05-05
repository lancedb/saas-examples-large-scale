import pyarrow as pa
import datasets
from datasets import load_dataset
import lance
import os
import shutil
import time
import io
import geneva
from geneva import udf
from geneva.config import override_config
from geneva.config.loader import from_kv
from geneva.runners.ray.raycluster import RayCluster, _HeadGroupSpec, _WorkerGroupSpec
from typing import Callable
import ray


images_path = "gs://ayush-geneva/"
k8s_name = "ayush-k8"
k8s_namespace = "geneva"

def create_dummy_table():
    shutil.rmtree(images_path + "./db", ignore_errors=True)
    shutil.rmtree(images_path + "./ckp", ignore_errors=True)


    def generate_dummy_data():
        # Generate dummy data for all splits
        splits = {
            'in_domain': 3_930_000,      # 3.93M rows
            'novel_document': 3_920_000,  # 3.92M rows
            'novel_query': 981_000,       # 981k rows
            'zero_shot': 981_000          # 981k rows
        }
        
        batch_size = 10000
        for split_name, total_size in splits.items():
            batch = []
            for idx in range(total_size):
                batch.append({
                    "row_idx": idx,
                    "split": split_name
                })
                
                if len(batch) >= batch_size:
                    yield pa.RecordBatch.from_pylist(batch)
                    batch = []
            
            if batch:
                yield pa.RecordBatch.from_pylist(batch)

    schema = pa.schema([
        pa.field("row_idx", pa.int64()),
        pa.field("split", pa.string())
    ])

    os.makedirs(f"{images_path}/db", exist_ok=True)

    lance.write_dataset(
        generate_dummy_data(), 
        images_path +"./db/images.lance",
        mode="overwrite", 
        schema=schema
    )
    print("Created dummy index table for Marqo dataset")


override_config(from_kv({"uploader.upload_dir": images_path}))

raycluster =  RayCluster(
    name= k8s_name,  
    namespace=k8s_namespace,
    head_group=_HeadGroupSpec(
        num_cpus=8,
        service_account="geneva-integ-test"
    ),
    worker_groups=[
        _WorkerGroupSpec(
            name="cpu",
            num_cpus=60,
            memory="120G",
            service_account="geneva-integ-test"
        ),
        _WorkerGroupSpec(
            name="gpu",
            num_cpus=8,
            memory="32G",
            num_gpus=1,
            service_account="geneva-integ-test"
        ),
    ],
)

time.sleep(5)
raycluster.__enter__()

@ray.remote(num_gpus=1)
def check_cuda():
    import geneva # this is currently required before other imports
    import torch
    return torch.version.cuda, torch.cuda.is_available()

print(ray.get(check_cuda.remote()))

@udf(data_type=pa.list_(pa.float32(), 512), input_columns=["row_idx", "split"], cuda=True)
class ImageEmbedding(Callable):
    def __init__(self):
        import geneva
        import open_clip
        import torch
        from datasets import load_dataset
        
        # Initialize model
        self.model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="laion2b_s34b_b79k",
        )
        self.model = self.model.cuda()
        self.model.eval()
        
        # Keep normalization tensors on CPU initially
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        
        # Initialize dataset streams for each split
        self.datasets = {
            split: load_dataset("Marqo/marqo-GS-10M", streaming=True, split=split)
            for split in ['in_domain', 'novel_document', 'novel_query', 'zero_shot']
        }
        
    def __call__(self, batch: pa.RecordBatch) -> pa.Array:
        import geneva
        import torch
        import numpy as np
        from PIL import Image
        import io
        
        row_indices = batch["row_idx"].to_numpy()
        split_name = batch["split"][0].as_py()
        
        # Calculate start and end indices
        start_idx = min(row_indices)
        end_idx = max(row_indices) + 1
        num_elements = end_idx - start_idx
        
        # Get batch of records directly
        try:
            records = list(self.datasets[split_name].skip(start_idx).take(num_elements))
            
            # Process images in batch
            images = []
            for record in records:
                try:
                    img_bytes = io.BytesIO(record["image"]["bytes"])
                    img_pil = (Image.open(img_bytes)
                              .convert('RGB')
                              .resize((224, 224), Image.Resampling.BILINEAR))
                    img_array = np.asarray(img_pil, dtype=np.float32) / 255.0
                    images.append(img_array)
                except Exception as e:
                    print(f"Error processing image: {e}")
                    return None
            
            if not images:
                return pa.array([])
            
            # Process batch with CUDA
            with torch.no_grad(), torch.autocast("cuda"):
                image_batch = (torch.from_numpy(np.stack(images))
                             .permute(0, 3, 1, 2)
                             .cuda())
                
                mean_cuda = self.mean.cuda()
                std_cuda = self.std.cuda()
                image_batch = (image_batch - mean_cuda) / std_cuda
                
                embeddings = self.model.encode_image(image_batch)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                embeddings = embeddings.cpu().numpy()
                
                del mean_cuda, std_cuda
                torch.cuda.empty_cache()
            
            return pa.array(embeddings.tolist())
            
        except Exception as e:
            print(f"Error processing batch for split {split_name}: {e}")
            return None



# Connect to the table
db = geneva.connect(images_path + "./db")
print(db.table_names())
table = db.open_table("images")

table.add_columns({"vector": ImageEmbedding}, materialize=True)
print("Vector column materialization complete")