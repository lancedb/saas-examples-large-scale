import modal
import time
import os


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

app = modal.App("wikipedia-search-test", image=image, secrets=get_secrets())

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
    cpu=4,
    memory=8000,
    min_containers=1,
    scaledown_window=SCALEDOWN_WINDOW,
    #region="us-east-1"
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
            print(f"search took: {search_time}")

            return {
                "results": results.to_dict(orient='records'), 
                "search_time": search_time,
                "query_plan": query_plan
            }
        except Exception as e:
            print(f"Vector search error: {str(e)}")
            return {"error": f"Vector search failed: {str(e)}"}

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
            print(f"search took: {search_time}")
            
            return {
                "results": results.to_dict(orient='records'), 
                "search_time": search_time,
                "query_plan": query_plan
            }
        except Exception as e:
            print(f"Full text search error: {str(e)}")
            return {"error": f"Full text search failed: {str(e)}"}

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
            print(f"search took: {search_time}")

            return {
                "results": results.to_dict(orient='records')[:limit], 
                "search_time": search_time,
                "query_plan": query_plan
            }
        except Exception as e:
            print(f"Hybrid search error: {str(e)}")
            return {"error": f"Hybrid search failed: {str(e)}"}

    @modal.fastapi_endpoint(method="POST")
    def search_endpoint(self, request: dict):
        try:
            print(f" request ", request)
            query_text = request.get("query")
            limit = request.get("limit", 5)
            search_type = request.get("search_type", "vector")
            explain = request.get("explain", False)

            if search_type not in SEARCH_TYPES:
                return {
                    "status": "error",
                    "message": f"Invalid search type. Use one of: {list(SEARCH_TYPES.keys())}",
                    "data": [],
                    "search_time": 0
                }
            
            if search_type == "vector":
                result = self.vector_search(query_text, limit, explain)
            elif search_type == "full_text":
                result = self.full_text_search(query_text, limit, explain)
            elif search_type == "hybrid":
                result = self.hybrid_search(query_text, limit, explain)
            else:
                # This case should ideally be caught by the SEARCH_TYPES check above
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
                "data": []
            }

    @modal.fastapi_endpoint(method="GET")
    def get_total_rows_endpoint(self):
        try:
            total_rows = self.table.count_rows()
            return {"total_rows": total_rows}
        except Exception as e:
            return {"status": "error", "message": f"Failed to get total rows: {str(e)}"}
        

@app.function(
    schedule=modal.Period(minutes=30),
)
def keep_warm():
    """
    Periodically runs a dummy search to keep the LanceDB table warm.
    """
    import lancedb
    import time
    
    db = lancedb.connect(
            uri=os.environ["LANCEDB_URI"],
            api_key=os.environ["LANCEDB_API_KEY"],
            region="us-east-1"
        )
        
    table = db.open_table(os.environ["LANCEDB_TABLE_NAME"])

    print("Running keep-warm search...")
    try:
        for _ in range(10):
            t1 = time.time() 
            _ = table.search([0]*384).limit(1).to_list()
            _ = table.search("whaaat", query_type="fts").limit(1).to_list()
            t2 = time.time()

            print(f" Time taken vector+fts {t2-t1}")

        print("✅ Keep-warm search successful.")
    except Exception as e:
        print(f"❌ Keep-warm search failed: {e}")