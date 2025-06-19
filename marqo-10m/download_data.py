import modal

cache_dir = "/data"
volume = modal.Volume.from_name("marco-10M", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "datasets", "numpy==1.26.4"
)

app = modal.App(image=image)
with image.imports():
    import datasets


@app.function(volumes={cache_dir: volume}, timeout=3000)
def download_dataset(cache=False):
    # Download and save the dataset locally on Modal worker
    dataset = datasets.load_dataset(
        "Marqo/marqo-GS-10M", num_proc=10, trust_remote_code=True
    )
    dataset.save_to_disk(f"{cache_dir}/marco-10m")

    # commit changes so they are visible to other Modal functions
    volume.commit()