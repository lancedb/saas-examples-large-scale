import modal
from datasets import load_dataset, Dataset
import os
import math

cache_dir = "/data"
volume = modal.Volume.from_name("leonardo-shards", create_if_missing=True)
HF_TOKEN = "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"
HF_DATASET_NAME = "bigdata-pw/leonardo"

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "datasets", "numpy==1.26.4", "pillow", "huggingface_hub"
).env({"HF_HUB_CACHE": cache_dir,
       "HF_HOME": cache_dir,
       })

app = modal.App(image=image)

@app.function(volumes={
    cache_dir: volume
    }, timeout=60*60*12, cpu=32, memory=150000)
def download_dataset(cache=False):
    from huggingface_hub import login
    login(token=HF_TOKEN)

    # Configuration
    target_shard_size = 1000000
    output_dir = f"{cache_dir}/leonardo_shards"
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset in streaming mode
    print("Loading dataset in streaming mode...")
    stream_dataset = load_dataset(
        "bigdata-pw/leonardo",
        split="train",
        streaming=True,
        verification_mode='no_checks'
    )

    # Process and save shards
    buffer = []
    shard_count = 0

    print(f"Starting sharding process (target shard size: {target_shard_size})...")
    for i, example in enumerate(stream_dataset):
        buffer.append(example)

        if len(buffer) >= target_shard_size:
            shard_dataset = Dataset.from_list(buffer)
            shard_path = os.path.join(output_dir, f"shard_{shard_count:05d}")
            print(f"Saving shard {shard_count} to {shard_path}...")
            shard_dataset.save_to_disk(shard_path)
            
            buffer = []
            shard_count += 1

        if (i + 1) % 10000 == 0:
            print(f"Processed {i+1} examples...")

    # Save final shard if any remaining examples
    if buffer:
        shard_dataset = Dataset.from_list(buffer)
        shard_path = os.path.join(output_dir, f"shard_{shard_count:05d}")
        print(f"Saving final shard {shard_count} to {shard_path}...")
        shard_dataset.save_to_disk(shard_path)

    print(f"Sharding complete. Total shards saved: {shard_count + 1}")
    volume.commit()