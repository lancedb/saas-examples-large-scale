import modal

cache_dir = "/data"
# Initialize a volume to save dataset to. A drive of this name will appear under your modal storage tab
volume = modal.Volume.from_name("embedding-wikipedia-lancedb", create_if_missing=True)

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
        "wikipedia", "20220301.en", num_proc=16, trust_remote_code=True
    )
    dataset.save_to_disk(f"{cache_dir}/wikipedia")

    # commit changes so they are visible to other Modal functions
    volume.commit()
