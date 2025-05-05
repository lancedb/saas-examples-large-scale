from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import lancedb
from pydantic import BaseModel
from typing import List
import numpy as np
from PIL import Image
import io
import os
from transformers import AutoProcessor, AutoModel
import torch
import base64

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LanceDB connection
def get_lancedb_table():
    db = lancedb.connect(
        uri="db://devrel-samp-9a5467",
        api_key=os.environ["LANCEDB_API_KEY"],
        region="us-east-1"
    )
    return db.open_table("artworks-modal-reverse-valid-2")

# Initialize the SIGLIP model
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

class SearchQuery(BaseModel):
    query: str

def handle_mixed_data(df):
    processed_df = df.copy()
    # drop vectors column
    processed_df = processed_df.drop(columns=['vector'])
    # base64 encode img bytes data
    processed_df['img'] = processed_df['img'].apply(
        lambda x: base64.b64encode(x)
    )
    '''
    for column in processed_df.columns:
        if isinstance(processed_df[column].iloc[0], np.ndarray):
            # Convert numpy arrays to lists
            processed_df[column] = processed_df[column].apply(lambda x: x.tolist())
        elif processed_df[column].dtype == 'object':
            # Handle image URLs
            if column == 'img':
                
                img_sample = processed_df[column].iloc[0]
                pil_img = Image.open(io.BytesIO(img_sample))
                import pdb;pdb.set_trace()
                processed_df[column] = processed_df[column].apply(
                    lambda x: base64.b64encode(x).decode('utf-8') if isinstance(x, bytes) else x
                )
            # Handle other text data
            else:
                processed_df[column] = processed_df[column].apply(
                    lambda x: x.decode('utf-8', errors='ignore') if isinstance(x, bytes) else x
                )
    '''

    return processed_df

@app.get("/random")
async def get_random_artworks():
    table = get_lancedb_table()
    results = table.search().limit(30).to_pandas()
    results = handle_mixed_data(results)
    return results.to_dict(orient="records")

@app.post("/search")
async def search_artworks(query: SearchQuery):
    # Tokenize the text query first
    inputs = processor(
        text=query.query,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)
    
    # Get text features using the tokenized input
    text_feats = model.get_text_features(**inputs).detach().cpu().numpy()[0]
    table = get_lancedb_table()
    print(" searching for query ", query.query)
    results = table.search(text_feats).limit(30).to_pandas()
    print("search done")
    results = handle_mixed_data(results)
    return results.to_dict(orient="records")

@app.post("/search/image")
async def search_by_image(file: UploadFile = File(...)):
    # Read and process the uploaded image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Generate image embedding
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        embedding = model.get_image_features(**inputs).detach().cpu().numpy()[0]
    
    # Search similar images
    table = get_lancedb_table()
    print(" searching for image ")
    results = table.search(embedding).limit(30).to_pandas()
    print("search done")
    results = handle_mixed_data(results)
    return results.to_dict(orient="records")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)