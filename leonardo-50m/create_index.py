import os
import lancedb

LANCEDB_URI = os.environ.get("LANCEDB_URI")
LANCEDB_API_KEY = os.environ.get("LANCEDB_API_KEY")

db = lancedb.connect(
    uri = LANCEDB_URI,
    api_key = LANCEDB_API_KEY,
    region="us-east-1"
)
print(db.table_names())
tbl = db.open_table("flickr-test")

tbl.create_index(metric="cosine", vector_column_name="vector_clip")