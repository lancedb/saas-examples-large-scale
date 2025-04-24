import os
import lancedb
from lancedb.pydantic import Vector, LanceModel
from lancedb.embeddings import get_registry
from config import (
    LANCEDB_URI, LANCEDB_API_KEY, LANCEDB_REGION,
    DEFAULT_TABLE_NAME, MODEL_NAME, VECTOR_DIM
)

model = get_registry().get("sentence-transformers").create(name=MODEL_NAME, device="cpu")

class WikipediaSchema(LanceModel):
    emb: Vector(VECTOR_DIM) = model.VectorField()
    identifier: int
    chunk_index: int
    content: str = model.SourceField()
    url: str
    title: str

def get_db_connection():
    """Get a connection to the LanceDB database."""
    return lancedb.connect(
        uri=LANCEDB_URI,
        api_key=LANCEDB_API_KEY,
        region=LANCEDB_REGION
    )

def get_or_create_table(table_name: str = None):
    """Get an existing table or create a new one if it doesn't exist."""
    db = get_db_connection()
    table_name = table_name or DEFAULT_TABLE_NAME
    
    try:
        return db.open_table(table_name)
    except Exception:
        return db.create_table(table_name, schema=WikipediaSchema)

def add_batch_data(table, chunks, embeddings):
    """Add a batch of data to the table."""
    batch_data = [
        {"emb": embedding} | chunk
        for chunk, embedding in zip(chunks, embeddings)
    ]
    table.add(batch_data)
    return len(batch_data) 