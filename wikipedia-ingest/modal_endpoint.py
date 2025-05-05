import modal
import time
import os
import base64
from typing import Dict, List

def get_secrets():
    LANCEDB_URI = os.environ.get("LANCEDB_URI")
    LANCEDB_API_KEY = os.environ.get("LANCEDB_API_KEY")
    LANCEDB_TABLE_NAME = os.environ.get("LANCEDB_TABLE_NAME", "Wikipedia-ayush-250GPU-A10")
    if not LANCEDB_URI or not LANCEDB_API_KEY or not LANCEDB_TABLE_NAME:
        raise ValueError("LANCEDB_URI, LANCEDB_API_KEY & LANCEDB_TABLE_NAME must be set")
    return [
        modal.Secret.from_dict({"LANCEDB_URI": LANCEDB_URI}),
        modal.Secret.from_dict({"LANCEDB_API_KEY": LANCEDB_API_KEY}),
        modal.Secret.from_dict({"LANCEDB_TABLE_NAME": LANCEDB_TABLE_NAME}),
    ]


# Create modal image with dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "sentence-transformers",
    "torch",
    "numpy",
    "fastapi",
    "pandas",
    "rerankers",
).run_commands(
    "pip install --pre --extra-index-url https://pypi.fury.io/lancedb/ lancedb"
)

app = modal.App("wikipedia-search", image=image, secrets=get_secrets())

SEARCH_TYPES = {
    "vector": {
        "model": "bge",
        "vector_column": "vector",
    },
    "full_text": {
        "vector_column": ["content", "title"],
    },
    "hybrid": {
        "model": "bge",
        "vector_column": "vector",
    }
}

SCALEDOWN_WINDOW = 60*15


@app.cls(
    cpu=8,
    memory=32000,
    min_containers=1,
    scaledown_window=SCALEDOWN_WINDOW,
    region="us-east-1"
)
class WikipediaSearcher:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        import torch
        import lancedb
        from lancedb.rerankers import ColbertReranker

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize the model
        self.model = SentenceTransformer('BAAI/bge-small-en-v1.5', device=self.device)
        self.model.eval()
        #self.reranker = ColbertReranker(
        #    "answerdotai/answerai-colbert-small-v1",
        #    device=self.device
        #)
        
        # Connect to LanceDB
        self.db = lancedb.connect(
            uri=os.environ["LANCEDB_URI"],
            api_key=os.environ["LANCEDB_API_KEY"],
            region="us-east-1"
        )
        
        self.table = self.db.open_table(os.environ["LANCEDB_TABLE_NAME"])

    def _process_text(self, text: str):
        import torch
        
        with torch.no_grad():
            embedding = self.model.encode(text, convert_to_tensor=True)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            return embedding.cpu().numpy()

    @modal.method()
    def vector_search(self, query_text: str, limit: int = 5, explain: bool = False):
        try:
            features = self._process_text(query_text)
            
            search_query = self.table.search(
                features,
                vector_column_name="vector"
            ).limit(limit).select([
                "content", 
                "title", 
                "url", 
                "identifier", 
                "chunk_index"
            ])
            
            query_plan = search_query.explain_plan(verbose=True) if explain else None
            
            start_time = time.time()
            results = search_query.to_pandas()
            search_time = time.time() - start_time
            
            return {
                "results": results.to_dict(orient='records'), 
                "search_time": search_time,
                "query_plan": query_plan
            }
        except Exception as e:
            print(f"Vector search error: {str(e)}")
            return {"error": f"Vector search failed: {str(e)}"}

    @modal.method()
    def full_text_search(self, query_text: str, limit: int = 5, explain: bool = False):
        try:
            search_query = self.table.search(
                query_text,
                query_type="fts",
            ).limit(limit).select([
                "content", 
                "title", 
                "url", 
                "identifier", 
                "chunk_index"
            ])
            
            query_plan = search_query.explain_plan(verbose=True) if explain else None
            
            start_time = time.time()
            results = search_query.to_pandas()
            search_time = time.time() - start_time
            
            return {
                "results": results.to_dict(orient='records'), 
                "search_time": search_time,
                "query_plan": query_plan
            }
        except Exception as e:
            print(f"Full text search error: {str(e)}")
            return {"error": f"Full text search failed: {str(e)}"}

    @modal.method()
    def hybrid_search(self, query_text: str, limit: int = 5, explain: bool = False):
        try:
            features = self._process_text(query_text)
            
            search_query = self.table.search(
                query_type="hybrid",
                vector_column_name="vector"
            ).vector(features).text(
                query_text
            ).limit(limit*4).select([
                "content", 
                "title", 
                "url", 
                "identifier", 
                "chunk_index"
            ])
            
            query_plan = None
            
            start_time = time.time()
            results = search_query.to_pandas()
            search_time = time.time() - start_time
            
            return {
                "results": results.to_dict(orient='records')[:limit], 
                "search_time": search_time,
                "query_plan": query_plan
            }
        except Exception as e:
            print(f"Hybrid search error: {str(e)}")
            return {"error": f"Hybrid search failed: {str(e)}"}

@app.function(region="us-east-1",
            min_containers=1,
            scaledown_window=SCALEDOWN_WINDOW,
              )
@modal.fastapi_endpoint(method="POST")
async def search(request: dict):
    try:
        searcher = WikipediaSearcher()
        query_text = request.get("query")
        limit = request.get("limit", 5)
        search_type = request.get("search_type", "vector")
        explain = request.get("explain", False)  # New explain flag

        if search_type not in SEARCH_TYPES:
            return {
                "status": "error", 
                "message": f"Invalid search type. Use one of: {list(SEARCH_TYPES.keys())}",
                "data": [],
                "search_time": 0
            }
        
        if search_type == "vector":
            result = searcher.vector_search.remote(query_text, limit, explain)
        elif search_type == "full_text":
            result = searcher.full_text_search.remote(query_text, limit, explain)
        elif search_type == "hybrid":
            result = searcher.hybrid_search.remote(query_text, limit, explain)
        else:
            return {
                "status": "error", 
                "message": "Invalid search type",
                "data": [],
                "search_time": 0
            }

        if isinstance(result, dict) and "error" in result:
            return {
                "status": "error",
                "message": result["error"],
                "data": [],
                "search_time": 0
            }

        return {
            "status": "success", 
            "data": result["results"],
            "search_time": result["search_time"],
            "query_plan": result.get("query_plan")
        }
    except Exception as e:
        return {
            "status": "error", 
            "message": str(e),
            "data": []  # Add empty array for error case
        }

@app.function(min_containers=1,
              region="us-east-1",
              scaledown_window=SCALEDOWN_WINDOW
              )
@modal.fastapi_endpoint(method="GET")
def get_total_rows():
    try:
        searcher = WikipediaSearcher()
        total_rows = searcher.table.count_rows()
        return {"total_rows": total_rows}
    except Exception as e:
        return {"status": "error", "message": f"Failed to get total rows: {str(e)}"}