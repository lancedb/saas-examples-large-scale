import modal

cache_dir = "/data"
volume = modal.Volume.from_name("artsy-final", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "datasets", "numpy==1.26.4", "pillow", "huggingface_hub"
).env({"HF_HUB_CACHE": cache_dir,
       "HF_HOME": cache_dir,
       })

app = modal.App(image=image)
with image.imports():
    import datasets
    import modal
    import os
    from huggingface_hub import snapshot_download


class DatasetEndReached(Exception):
    pass

@app.function(volumes={
    cache_dir: volume
    }, timeout=60*60*12, cpu=32, memory=150000, max_containers=20)
def download_artsy_dataset(start_idx: int, end_idx: int, idx: int, cache=False):
    # Set up logging
    datasets.logging.enable_progress_bar()
        
    try:
        print(f"Loading dataset with indices {start_idx} to {end_idx}")
        dataset = datasets.load_dataset(
            "bigdata-pw/Artsy",
            split=f"train[{start_idx}:{end_idx}]",
            cache_dir=cache_dir,
            num_proc=16 )
        
        if len(dataset) == 0:
            raise DatasetEndReached("No more data available")
            
        print(f"Dataset loaded, saving to disk at {cache_dir}/artsy")
        dataset.save_to_disk(
            f"{cache_dir}/artsy-small-shards/artsy-{idx}",
            num_proc=32
        )
        print("Save completed, committing volume")
        volume.commit()
        return "Success"
        
    except (IndexError, ValueError) as e:
        print(f"Dataset end reached: {e}")
        raise DatasetEndReached("Dataset end reached") from e
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

@app.local_entrypoint()
def main():
    shard_size = 50000  # Size of each shard
    start_from = 0 
    
    idx = 0  # Resume from where we left off
    while True:
        try:
            start_idx = start_from + (idx * shard_size)
            end_idx = start_idx + shard_size
            print(f"Processing shard {idx}: {start_idx} to {end_idx}")
            
            result = download_artsy_dataset.remote(start_idx, end_idx, idx)
            if result == "Success":
                idx += 1
                print(f"Shard {idx} processed successfully")
            
        except DatasetEndReached:
            print(f"Reached end of dataset after processing {idx} shards")
            break
        except Exception as e:
            print(f"Error processing shard {idx}: {e}")
            break



@app.function(volumes={cache_dir: volume}, timeout=3000)
def download_wikiart_dataset(cache=False):
    # Download and save the dataset locally on Modal worker
    dataset = datasets.load_dataset(
        "Artificio/WikiArt", num_proc=10, trust_remote_code=True
    )
    dataset.save_to_disk(f"{cache_dir}/wikiart")

    # commit changes so they are visible to other Modal functions
    volume.commit()



@app.function(volumes={cache_dir: volume}, timeout=3000)
def download_met_dataset(cache=False):
    # Download and save the dataset locally on Modal worker
    dataset = datasets.load_dataset(
        "miccull/met_museum", trust_remote_code=True
    )
    dataset.save_to_disk(f"{cache_dir}/met_museum")

    # commit changes so they are visible to other Modal functions
    volume.commit()


@app.function(volumes={cache_dir: volume}, timeout=84000)
def download_museum_dataset(cache=False):
    # Download and save the dataset locally on Modal worker
    dataset = datasets.load_dataset(
        "Mitsua/art-museums-pd-440k", trust_remote_code=True
    )
    dataset.save_to_disk(f"{cache_dir}/art-museums")

    # commit changes so they are visible to other Modal functions
    volume.commit()




#if __name__ == "__main__":
#    with app.run():
#        main.call()
#
