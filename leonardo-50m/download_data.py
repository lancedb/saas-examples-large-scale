import modal 
import os
cache_dir = "/data"

volume = modal.Volume.from_name("leonardo-shards", create_if_missing=True)
HF_TOKEN = os.environ["HF_TOKEN"]
HF_DATASET_NAME = "bigdata-pw/leonardo"

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "datasets", "numpy==1.26.4", "pillow", "huggingface_hub"
).env({"HF_HUB_CACHE": cache_dir,
       "HF_HOME": cache_dir,
       })

app = modal.App(image=image)


@app.function(volumes={
    cache_dir: volume
    }, timeout=60*60*12, cpu=32, memory=150000, max_containers=50)
def download_dataset(args: dict):
    import datasets
    from huggingface_hub import login
    login(token=HF_TOKEN)

    start_idx = args['start_idx']
    end_idx = args['end_idx']
    idx = args['idx']
    try:
        
        print(f"Loading dataset with indices {start_idx} to {end_idx}")
        dataset = datasets.load_dataset(
            "bigdata-pw/leonardo",
            split=f"train[{start_idx}:{end_idx}]",
            cache_dir=cache_dir,
            num_proc=16
        )
        
        print(f"Saving shard {idx} to disk at {cache_dir}/leonardo")
        dataset.save_to_disk(
            f"{cache_dir}/leonardo-shards/leonardo-{idx}",
            num_proc=32
        )
        print("Save completed, committing volume")
        volume.commit()
        return "Success"
        
    except Exception as e:
        print(f"Error processing shard {idx}: {e}")
        raise

@app.local_entrypoint()
def main():
    total_rows = 957000000
    shard_size = 100000
    num_shards = total_rows // shard_size + (1 if total_rows % shard_size != 0 else 0)
    
    # Create list of argument dictionaries
    args_list = [
        {
            'start_idx': idx * shard_size,
            'end_idx': min((idx + 1) * shard_size, total_rows),
            'idx': idx
        }
        for idx in range(num_shards)
    ]
    
    # Process all shards in parallel using map
    results = download_dataset.map(args_list)
    
    # Check results
    for idx, result in enumerate(results):
        if result == "Success":
            print(f"Shard {idx} processed successfully")
        else:
            print(f"Shard {idx} failed to process")