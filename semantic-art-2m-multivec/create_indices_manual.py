import lancedb
import os

db = lancedb.connect(
    uri="db://wikipedia-test-9cusod",
    api_key=os.environ["LANCEDB_API_KEY"],
    region="us-east-1"
)

table = db.open_table("artsy_multi_vector_caption_large")
#table.create_index(vector_column_name="vector_clip_caption")
#table.create_index(vector_column_name="vector_siglip_caption")
#table.create_index(vector_column_name="vector_clip")
#table.create_index(vector_column_name="vector_clip")

#table.create_fts_index("caption")
table.create_fts_index("artist")
table.create_fts_index("title")