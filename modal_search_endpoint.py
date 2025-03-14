import modal
from typing import Union, Any
import base64
from fastapi import FastAPI, Request, HTTPException
import json

# Create modal image with dependencies
image = (modal.Image.debian_slim(python_version="3.10")
         .pip_install(
             "open_clip_torch",
             "torch",
             "pillow",
             "lancedb",
             "pandas",
             "transformers",
         )
         )

app = modal.App("art-search", image=image)

with image.imports():
    from PIL import Image
    import io
    import lancedb
    import open_clip
    import torch
    import numpy as np

# Create a class for the model
MODEL_CONFIGS = {
    "clip": {
        "name": "ViT-B-32",
        "pretrained": "laion2b_s34b_b79k",
        "table": "artworks-clip-base-embed-api"
    },
    "siglip": {
        "name": "ViT-L-16-SigLIP2-256",
        "pretrained": "webli",
        "table": "artworks-siglip2-base-embed-api"
    }
}

@app.cls(
    image=image,
    gpu="A10",
    cpu=8,
    scaledown_window=120,
)
class ClipSearcher:

    def __init__(self):
        # Initialize models and database connection
        self.models = {}
        self.preprocessors = {}
        self.tokenizers = {}
        self.tables = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.db = lancedb.connect(
            uri="db://devrel-samp-9a5467",
            api_key="sk_THUBNC75R5AYPMMRMUV6SEPWWLSPXY7ZSKVQPUUFVCOOQGYKGUKA====",
            region="us-east-1"
        )
        
        for model_type, config in MODEL_CONFIGS.items():
            model, _, preprocess = open_clip.create_model_and_transforms(
                config["name"],
                pretrained=config["pretrained"]
            )
            model = model.to(self.device)
            self.models[model_type] = model
            self.preprocessors[model_type] = preprocess
            self.tokenizers[model_type] = open_clip.get_tokenizer(config["name"])
            self.tables[model_type] = self.db.open_table(config["table"])

    def _process_image(self, image_bytes, model_type="siglip"):
        try:
            image = Image.open(io.BytesIO(image_bytes))
            img_tensor = self.preprocessors[model_type](image).unsqueeze(0)
            img_tensor = img_tensor.to(self.device)
            with torch.no_grad():
                image_features = self.models[model_type].encode_image(img_tensor)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().numpy()[0]
        except Exception as e:
            raise Exception(f"Image processing failed: {str(e)}")

    def _process_text(self, text, model_type="siglip"):
        try:
            with torch.no_grad():
                text_tensor = self.tokenizers[model_type](text)
                text_tensor = text_tensor.to(self.device)
                text_features = self.models[model_type].encode_text(text_tensor)
                text_features /= text_features.norm(dim=-1, keepdim=True)
            return text_features.cpu().numpy()[0]
        except Exception as e:
            raise Exception(f"Text processing failed: {str(e)}")

    @modal.method()
    def search_by_text(self, query_text: str, model_type: str = "siglip", limit: int = 5):
        try:
            features = self._process_text(query_text, model_type)
            results = self.tables[model_type].search(features).limit(limit).to_pandas()
            
            # Remove the img column from results
            if 'img' in results.columns:
                results = results.drop(columns=['img'])
                
            return results.to_dict(orient='records')
        except Exception as e:
            return {"error": f"Text search failed: {str(e)}"}

    @modal.method()
    def search_by_image(self, image_bytes: bytes, model_type: str = "siglip", limit: int = 5):
        try:
            features = self._process_image(image_bytes, model_type)
            results = self.tables[model_type].search(features).limit(limit).to_pandas()
            
            # Remove the img column from results
            if 'img' in results.columns:
                results = results.drop(columns=['img'])
                
            return results.to_dict(orient='records')
        except Exception as e:
            return {"error": f"Image search failed: {str(e)}"}

@app.function(keep_warm=1)
@modal.fastapi_endpoint(method="POST")
async def search(request: dict):
    try:

        
        searcher = ClipSearcher()
        
        query_text = request.get("query")
        image_data = request.get("image")
        model = request.get("model", "siglip")
        limit = request.get("limit", 5)
        
        if not query_text and not image_data:
            return {"status": "error", "message": "No query provided"}
        
        if model not in MODEL_CONFIGS:
            return {"status": "error", "message": f"Invalid model type. Use one of: {', '.join(MODEL_CONFIGS.keys())}"}
        
        if query_text:
            results = searcher.search_by_text.remote(query_text, model, limit)
        else:
            # Handle base64 encoded images
            if isinstance(image_data, str):
                try:
                    # Remove potential data URL prefix
                    if "base64," in image_data:
                        image_data = image_data.split("base64,")[1]
                    image_bytes = base64.b64decode(image_data)
                except Exception as e:
                    return {"status": "error", "message": f"Failed to decode base64 image: {str(e)}"}
            else:
                return {"status": "error", "message": "Image must be provided as base64 encoded string"}
                
            results = searcher.search_by_image.remote(image_bytes, model, limit)
            
        if isinstance(results, dict) and "error" in results:
            return {"status": "error", "message": results["error"]}
            
        return {"status": "success", "data": results}
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in search endpoint: {error_details}")
        return {"status": "error", "message": f"Search failed: {str(e)}"}