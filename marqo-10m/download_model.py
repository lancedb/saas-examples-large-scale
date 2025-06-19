import base64
import io
from pathlib import Path

import modal

MODEL_REPO_ID = "timm/ViT-L-16-SigLIP2-256"
MODEL_DIR = "/cache"


app = modal.App("model-download")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "huggingface-hub==0.27.1",
        "Pillow",
        "timm",
        "transformers",
    )
    .apt_install("fonts-freefont-ttf")
    .env({"HF_HUB_CACHE": MODEL_DIR})
)
cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

with image.imports():
    import torch
    from huggingface_hub import snapshot_download
    from PIL import Image, ImageColor, ImageDraw, ImageFont
    from transformers import DetrForObjectDetection, DetrImageProcessor



@app.function(image=image, volumes={MODEL_DIR: cache_volume})
def download_model():
    loc = snapshot_download(repo_id=MODEL_REPO_ID)
    print(f"Saved model to {loc}")
