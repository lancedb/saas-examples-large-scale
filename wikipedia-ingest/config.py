import os
import modal

# Cache and volume settings
CACHE_DIR = "/data"
VOLUME_NAME = "embedding-wikipedia-lancedb"
volume = modal.Volume.from_name(VOLUME_NAME)

# Dataset settings
DATASET_PATH = f"{CACHE_DIR}/wikipedia"

# Database settings
LANCEDB_URI = os.environ.get("LANCEDB_URI")
LANCEDB_API_KEY = os.environ.get("LANCEDB_API_KEY")
LANCEDB_REGION = "us-east-1"

# Model settings
MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_DIM = 384

# Processing settings
BATCH_SIZE = 100_000  # Main processing batch size
EMBEDDING_BATCH_SIZE = 100_000  # Batch size for embedding
CHUNK_SIZE = 512
NUM_CHUNK_CONTAINERS = 300
NUM_EMBEDDING_CONTAINERS = 50

# Default table name
DEFAULT_TABLE_NAME = "wikipedia"

# Image configuration
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "lancedb",
    "fastapi==0.110.1",
    "numpy==1.26.4",
    "datasets",
    "transformers",
    "sentence-transformers"
)

def get_secrets():
    """Get Modal secrets for LanceDB connection."""
    if not LANCEDB_URI or not LANCEDB_API_KEY:
        raise ValueError("LANCEDB_URI and LANCEDB_API_KEY must be set")
    return [
        modal.Secret.from_dict({"LANCEDB_URI": LANCEDB_URI}),
        modal.Secret.from_dict({"LANCEDB_API_KEY": LANCEDB_API_KEY})
    ] 