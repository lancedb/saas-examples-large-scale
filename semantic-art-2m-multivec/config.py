import modal
import lancedb
import logging
import os
from lancedb.pydantic import Vector, LanceModel

# --- General Configuration ---
CACHE_DIR = "/data"
ARTSY_DIR = "/artsy"
MODEL_DIR = "/cache"
BATCH_SIZE = 50000
EMBEDDING_BATCH_SIZE = 5000
REGION = "any"

# --- Modal Volumes ---
artsy_shard_vol = modal.Volume.from_name("artsy-final")
artsy_vol = modal.Volume.from_name("artsy")
model_volume = modal.Volume.from_name("hf-hub-cache")

# --- Dataset Configuration ---
DATA_SPLITS = {}
for i in range(0,31):
    DATA_SPLITS[f"artsy-{i}"] = None
DATA_SPLITS["met"] = ["train"]
DATA_SPLITS["museums"] = ["train"]
DATA_SPLITS["wikiart"] = ["train"]

DATA_PATHS = {
    "met": "met_museum",
    "museums": "art-museums",
    "wikiart": "wikiart"
}
for i in range(0,31):
    DATA_PATHS[f"artsy-{i}"] = f"artsy-small-shards/artsy-{i}"

DATA_COL_MAPPINGS = {
    "artsy": {
        "artist": "artist.txt", "image": "jpg", "title": "title.txt", "date": "date.txt",
    },
    "wikiart": {
        "artist": "artist", "image": "image", "title": "description", "date": "date"
    },
    "met": {
        "artist": "artist", "image": "image", "title": "title", "date": "accession_year"
    },
    "museums": {
        "artist": ["json", "Author"], "image": "jpg", "title": ["json", "Title"], "date": None
    }
}

DATA_VOLUME_MAPPINGS = {
    "artsy": artsy_shard_vol,
    "met": artsy_vol,
    "museums": artsy_vol,
    "wikiart": artsy_vol
}

# --- LanceDB Configuration ---
LANCEDB_URI = "db://wikipedia-test-9cusod" # Replace with your actual URI if needed
LANCEDB_API_KEY = os.environ.get("LANCEDB_API_KEY")
LANCEDB_REGION = "us-east-1"
DEFAULT_TABLE_NAME = "artsy_multi_vector_"

if not LANCEDB_API_KEY:
    raise ValueError("LANCEDB_API_KEY is not set. Please set it in your environment or directly in the script.")

# --- LanceDB Schema ---
class SCHEMA(LanceModel):
    title: str
    artist: str
    ds_name: str
    date: str
    image: bytes
    vector_clip: Vector(512)
    vector_siglip: Vector(1024)

# --- LanceDB Connection and Table Handling ---
def get_lancedb_table(table_name: str = DEFAULT_TABLE_NAME):
    """Connects to LanceDB and returns the specified table, creating it if necessary."""
    try:
        db = lancedb.connect(
            uri=LANCEDB_URI,
            api_key=LANCEDB_API_KEY,
            region=LANCEDB_REGION
        )
        logging.info(f"Attempting to open table: {table_name}")
        table = db.open_table(table_name)
        logging.info(f"Successfully opened table: {table_name}")
        return table
    except Exception as open_exception:
        logging.warning(f"Could not open table {table_name}: {open_exception}. Attempting to create.")
        try:
            table = db.create_table(table_name, schema=SCHEMA)
            logging.info(f"Successfully created table: {table_name}")
            return table
        except ValueError as create_exception:
            # Handle potential race condition where table was created between open and create attempts
            if "already exists" in str(create_exception):
                logging.warning(f"Table {table_name} already exists (race condition?). Re-attempting open.")
                try:
                    table = db.open_table(table_name)
                    logging.info(f"Successfully opened table {table_name} on second attempt.")
                    return table
                except Exception as final_open_exception:
                     logging.error(f"Failed to open table {table_name} even after creation attempt: {final_open_exception}")
                     raise final_open_exception
            else:
                logging.error(f"Failed to create table {table_name}: {create_exception}")
                raise create_exception
        except Exception as final_create_exception:
            logging.error(f"An unexpected error occurred during table creation for {table_name}: {final_create_exception}")
            raise final_create_exception

