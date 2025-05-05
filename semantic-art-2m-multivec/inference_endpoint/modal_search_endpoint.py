from lancedb import vector
import modal
import os
import base64

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
    import time

# Create a class for the model
# Update MODEL_CONFIGS to include caption vectors

SEARCH_TYPES = {
    "clip_text_to_image": {
        "model": "clip",
        "vector_column": "vector_clip",
    },
    "sigplip_text_to_image": {
        "model": "siglip",
        "vector_column": "vector_siglip",
    },
    "clip_image_to_image": {
        "model": "clip",
        "vector_column": "vector_clip",
    },
    "sigplip_image_to_image": {
        "model": "siglip",
        "vector_column": "vector_siglip",
    },
    "clip_caption": {
        "model": "clip",
        "vector_column": "vector_clip_caption",
    },
    "sigplip_caption": {
        "model": "siglip",
        "vector_column": "vector_siglip_caption",
    },
    "full_text": {
        "vector_column": ["caption", "artist", "title"],
    },
    "hybrid_clip": {
        "model": "clip",
        "vector_column": "vector_clip_caption",
    },
    "hybrid_siglip": {
        "model": "siglip",
        "vector_column": "vector_siglip_caption",
    },
}

MODEL_CONFIGS = {
    "clip": {
        #"name": "ViT-B-32",
        "model_name": "hf-hub:laion/CLIP-ViT-B-32-256x256-DataComp-s34B-b86K", # TODO: used cached
    },
    "siglip": {
        "model_name": "ViT-L-16-SigLIP2-256",
        "pretrained": "webli",
    }
    }

TABLE_NAME = "artsy_multi_vector_caption_large"

@app.cls(
    image=image,
    gpu="any",
    cpu=8,
    scaledown_window=1200,
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
            uri="db://wikipedia-test-9cusod",
            api_key=os.environ["LANCEDB_API_KEY"],
            region="us-east-1"
            )
        
        for model_type, config in MODEL_CONFIGS.items():
            model_name = config.get("model_name")
            pretrained = config.get("pretrained")
            if model_name is None:
                raise ValueError(f"Model name is required for {model_type}")

            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name=model_name,
                pretrained=pretrained
            )
            model = model.to(self.device)
            self.models[model_type] = model
            self.preprocessors[model_type] = preprocess
            self.tokenizers[model_type] = open_clip.get_tokenizer(config["model_name"])
        self.table = self.db.open_table(TABLE_NAME)

    def _process_image(self, image_bytes, model_type="siglip"):
        # TODO: align with ingestion code
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
    def search_by_text(self, query_text: str, search_type: str, limit: int = 5):
        try:
            model_type = SEARCH_TYPES[search_type]["model"]
            vector_col = SEARCH_TYPES[search_type].get("vector_column")
            if search_type == "full_text":
                return self.full_text_search(query_text, limit)

            features = self._process_text(query_text, model_type)

            t1 = time.time()
            results = self.table.search(
                features,
                vector_column_name=vector_col,
            ).limit(limit).with_row_id(True).select(['title', 'artist', 'image', 'date', 'ds_name', "_rowid"]).to_pandas()
            t2 = time.time()
            print(f"Search took {t2 - t1} seconds")
            # Convert image bytes to base64
            for idx, row in results.iterrows():
                if 'image' in row and row['image'] is not None:
                    results.at[idx, 'image'] = base64.b64encode(row['image']).decode('utf-8')
            print("text search result ", results)
            results =  results.to_dict(orient='records')
            return results
        except Exception as e:
            print(f"Text search error: {str(e)}")
            return {"error": f"Text search failed: {str(e)}"}

    @modal.method()
    def search_by_image(self, image_bytes: bytes, search_type, limit: int = 5):
        try:
            model_type = SEARCH_TYPES[search_type]["model"]
            vector_col = SEARCH_TYPES[search_type].get("vector_column")
            features = self._process_image(image_bytes, model_type)

            t1 = time.time()
            results = self.table.search(
                features, 
                vector_column_name=vector_col,
                ).limit(limit).with_row_id(True).select(['title', 'artist', 'image', 'date', 'ds_name', "_rowid"]).to_pandas()
            t2 = time.time()
            print(f"Search took {t2 - t1} seconds")


            # Convert image bytes to base64 before dropping the column
            for idx, row in results.iterrows():
                if 'image' in row and row['image'] is not None:
                    results.at[idx, 'image'] = base64.b64encode(row['image']).decode('utf-8')
            

            return results.to_dict(orient='records')
        except Exception as e:
            print(f"Image search error: {str(e)}")
            return {"error": f"Image search failed: {str(e)}"}

    @modal.method()
    def search_random(self, model_type: str = "siglip", limit: int = 5):
        try:
            # random list of dim 512
            dim = 512 if model_type == "clip" else 1024
            feats = np.random.rand(dim).tolist() # Remove harcoding
            results = self.table.search(
                feats,
                vector_column_name="vector_clip",
                ).limit(limit).to_pandas()
            
            # Convert image bytes to base64 before dropping the column
            for idx, row in results.iterrows():
                if 'image' in row and row['image'] is not None:
                    results.at[idx, 'image'] = base64.b64encode(row['image']).decode('utf-8')
            
            # Remove the original columns
            results = results.drop(columns=['vector_clip', 'vector_siglip'])
            

            return results.to_dict(orient='records')
        except Exception as e:
            print(f"Image search error: {str(e)}")
            return {"error": f"Image search failed: {str(e)}"}
        
    @modal.method()
    def full_text_search(self, query_text: str, limit: int = 5):
        try:
            # Use text search on both title and artist fields
            results = self.table.search(
                query_text,
                query_type="fts", 
                fts_columns=["title", "artist", "caption"]
            ).limit(limit).select(
                ["title", "artist", "image", "date", "ds_name", "_rowid"]
            ).to_pandas()
            
            # Convert image bytes to base64
            for idx, row in results.iterrows():
                if 'image' in row and row['image'] is not None:
                    results.at[idx, 'image'] = base64.b64encode(row['image']).decode('utf-8')
            return results.to_dict(orient='records')
        except Exception as e:
            print(f"Full text search error: {str(e)}")
            return {"error": f"Full text search failed: {str(e)}"}
    
    @modal.method()
    def hybrid_search(self, query_text: str, search_type="hybrid_clip", limit: int = 5):
        try:
            model_type = SEARCH_TYPES[search_type]["model"]
            vector_col_name = SEARCH_TYPES[search_type].get("vector_column")
            features = self._process_text(query_text, model_type)
            results = self.table.search(
                query_type="hybrid",
                vector_column_name=vector_col_name
                ).vector(features).text(query_text).limit(limit).select(
                    ["title", "artist", "image", "date", "ds_name", "_rowid"]
                ).to_pandas()

            # Convert image bytes to base64
            for idx, row in results.iterrows():
                if 'image' in row and row['image'] is not None:
                    results.at[idx, 'image'] = base64.b64encode(row['image']).decode('utf-8')

            return results.to_dict(orient='records')
        except Exception as e:
            print(f"Hybrid search error: {str(e)}")
            return {"error": f"Hybrid search failed: {str(e)}"}
 

@app.function()
@modal.fastapi_endpoint(method="POST")
async def search(request: dict):
    try:
        searcher = ClipSearcher()
        query_text = request.get("query")
        image_data = request.get("image")
        limit = request.get("limit", 5)
        search_type = request.get("search_type", "clip_text_to_image")

        image_bytes = None
        if image_data:
            image_data = image_data.split("base64,")[1]
            image_bytes = base64.b64decode(image_data)
        print("search type ", search_type)
        if search_type not in SEARCH_TYPES:
            return {"status": "error", "message": f"Invalid search type. Use one of: {list(SEARCH_TYPES.keys())}"}
        
        if search_type in ["clip_image_to_image", "sigplip_image_to_image"]:
            print("Searching by image")
            results =  searcher.search_by_image.remote(image_bytes, search_type, limit)
        elif search_type in ["clip_text_to_image", "sigplip_text_to_image", "clip_caption", "siglip_caption"]:
            print("Searching by text")
            results =  searcher.search_by_text.remote(query_text, search_type, limit )
            print("Searched by text")
        elif search_type in ["full_text"]:
            print("Searching by full text")
            results =  searcher.full_text_search.remote(query_text, limit)
        elif search_type in ["hybrid_clip", "hybrid_siglip"]:
            print("Searching by hybrid")
            results =  searcher.hybrid_search.remote(query_text, search_type, limit)
        else:
            return {"status": "error", "message": "Invalid search type"}
        

        print("resuslts retured ", results)
        return {"status": "success", "data": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.function()
@modal.fastapi_endpoint(method="POST")
def get_total_rows():
    try:
        searcher = ClipSearcher()
        table = searcher.table
        total_rows = table.count_rows()
        return {"total_rows": total_rows}
    except Exception as e:
        return {"status": "error", "message": f"Failed to get total rows: {str(e)}"}
    