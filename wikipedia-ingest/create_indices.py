import lancedb

db = lancedb.connect(
  uri="db://wikipedia-test-9cusod",
  api_key="sk_BJIR2QXJ7JAM7AUMTHHPDLGJRY26GM3LN4ONK5TKTTGCFORC4NEQ====",
  region="us-east-1"
)

table = db["Wikipedia-ayush-250GPU-A10"]

table.create_index(metric="cosine", vector_column_name="vector", index_type="IVF_HNSW_SQ")
table.create_fts_index("content")
table.create_fts_index("title")