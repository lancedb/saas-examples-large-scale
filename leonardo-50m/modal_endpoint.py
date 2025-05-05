import modal
import time
import os
import base64
from typing import Dict, List
import io


def get_secrets():
    LANCEDB_URI = os.environ.get("LANCEDB_URI")
    LANCEDB_API_KEY = os.environ.get("LANCEDB_API_KEY")
    LANCEDB_TABLE_NAME = os.environ.get("LANCEDB_TABLE_NAME", "flickr-test")
    if not LANCEDB_URI or not LANCEDB_API_KEY:
        raise ValueError("LANCEDB_URI and LANCEDB_API_KEY must be set")
    return [
        modal.Secret.from_dict({"LANCEDB_URI": LANCEDB_URI}),
        modal.Secret.from_dict({"LANCEDB_API_KEY": LANCEDB_API_KEY}),
        modal.Secret.from_dict({"LANCEDB_TABLE_NAME": LANCEDB_TABLE_NAME}),
    ]

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install(
            "torch",
            "Pillow",
            "lancedb",
            "transformers",
            "sentencepiece",
            "open-clip-torch",
            "fastapi",
            "numpy",
            "pandas", 
            "huggingface_hub" 
         ))

app = modal.App("leonardo-search", image=image, secrets=get_secrets())

with image.imports():
    import torch
    import open_clip
    import lancedb
    import numpy as np
    from PIL import Image


SCALEDOWN_WINDOW = 60 * 15 # 15 minutes

@app.cls(
    gpu="L4",
    cpu=4,
    memory=16000, 
    min_containers=1, 
    scaledown_window=SCALEDOWN_WINDOW,
    region="us-east-1"
)
class LeonardoSearcher:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.model_clip, _, self.preprocess_clip = open_clip.create_model_and_transforms(
            'ViT-B-32',
            pretrained='laion2b_s34b_b79k'
        )
        self.model_clip = self.model_clip.to(self.device)
        self.model_clip.eval()
        self.tokenizer_clip = open_clip.get_tokenizer('ViT-B-32')

        self.db = lancedb.connect(
            uri=os.environ["LANCEDB_URI"],
            api_key=os.environ["LANCEDB_API_KEY"],
            region="us-east-1"
        )
        self.table_name = os.environ["LANCEDB_TABLE_NAME"]
        try:
            self.table = self.db.open_table(self.table_name)
            print(f"Opened LanceDB table: {self.table_name}")
        except Exception as e:
            print(f"Error opening table {self.table_name}: {e}")
            raise e

    def _process_text(self, text: str):
        with torch.no_grad(), torch.autocast(self.device):
            text_tokens = self.tokenizer_clip([text]).to(self.device)
            text_features = self.model_clip.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()[0]

    @modal.method()
    def search_by_text(self, query_text: str, limit: int = 10):
        try:
            start_time = time.time()
            query_vector = self._process_text(query_text)
            process_time = time.time() - start_time
            print(f"Text processing time: {process_time:.4f}s")

            start_time = time.time()
            search_query = self.table.search(
                query_vector,
                vector_column_name="vector_clip" 
            ).limit(limit).select([
                "description",
                "url",
                "image" 
            ])

            results_df = search_query.to_pandas()
            search_time = time.time() - start_time
            print(f"LanceDB search time: {search_time:.4f}s for {len(results_df)} results")

            results = results_df.to_dict(orient='records')
            start_time = time.time()
            for record in results:
                if 'image' in record and isinstance(record['image'], bytes):
                    if len(record['image']) > 0:
                         try:
                             record['image'] = base64.b64encode(record['image']).decode('utf-8')
                         except Exception as img_err:
                             print(f"Warning: Could not process/encode image for URL {record.get('url', 'N/A')}: {img_err}")
                             record['image'] = None 
                    else:
                         record['image'] = None 
                else:
                     record['image'] = None

            encode_time = time.time() - start_time
            print(f"Base64 encoding time: {encode_time:.4f}s")

            return {
                "results": results,
                "search_time": search_time + process_time # Total time
            }

        except Exception as e:
            print(f"Error during text search: {str(e)}")
            return {"error": f"Search failed: {str(e)}", "results": [], "search_time": 0}


@app.function(
    region="us-east-1",
    min_containers=1,
    scaledown_window=SCALEDOWN_WINDOW,
    keep_warm=1 
)
@modal.fastapi_endpoint(method="POST")
async def search(request: dict):
    """Endpoint to handle search requests."""
    try:
        searcher = LeonardoSearcher() 
        query_text = request.get("query")
        limit = int(request.get("limit", 12)) 

        if not query_text:
            return {"status": "error", "message": "Query text is required", "data": [], "search_time": 0}

        result_data = searcher.search_by_text.remote(query_text, limit)

        if isinstance(result_data, dict) and "error" in result_data:
             return {
                 "status": "error",
                 "message": result_data["error"],
                 "data": [],
                 "search_time": 0
             }

        return {
            "status": "success",
            "data": result_data.get("results", []),
            "search_time": result_data.get("search_time", 0)
        }
    except Exception as e:
        print(f"Error in search endpoint: {str(e)}")
        return {"status": "error", "message": f"An unexpected error occurred: {str(e)}", "data": [], "search_time": 0}


@app.function(
    min_containers=1,
    region="us-east-1",
    scaledown_window=SCALEDOWN_WINDOW
)
@modal.fastapi_endpoint(method="GET")
def get_total_rows():
    """Endpoint to get the total number of rows in the table."""
    try:
        searcher = LeonardoSearcher()
        total_rows = searcher.table.count_rows()
        return {"total_rows": total_rows}
    except Exception as e:
        print(f"Error getting total rows: {str(e)}")
        return {"status": "error", "message": f"Failed to get total rows: {str(e)}", "total_rows": 0}