import requests
import time
import json
import pandas as pd
from pathlib import Path
import asyncio
import aiohttp
from tqdm import tqdm

class MetArtFetcher:
    def __init__(self):
        self.base_url = "https://collectionapi.metmuseum.org/public/collection/v1"
        self.output_dir = Path("data")
        self.output_dir.mkdir(exist_ok=True)
        self.batch_size = 50  # Number of concurrent requests
        self.rate_limit = 0.05  # 50ms between requests

    async def get_object_ids(self):
        """Fetch all object IDs from the Met API"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/objects") as response:
                if response.status == 200:
                    data = await response.json()
                    return data["objectIDs"]
        return []

    async def get_object_details(self, session, object_id):
        """Fetch details for a specific object"""
        await asyncio.sleep(self.rate_limit)  # Rate limiting
        async with session.get(f"{self.base_url}/objects/{object_id}") as response:
            if response.status == 200:
                return await response.json()
        return None

    async def process_batch(self, session, batch_ids):
        """Process a batch of object IDs concurrently"""
        tasks = [self.get_object_details(session, obj_id) for obj_id in batch_ids]
        return await asyncio.gather(*tasks)

    async def fetch_objects(self, limit=None):
        """Fetch objects using async requests"""
        object_ids = await self.get_object_ids()
        total_objects = len(object_ids)
        if limit:
            object_ids = object_ids[:limit]

        data_list = []
        successful_fetches = 0
        objects_with_images = 0
        
        async with aiohttp.ClientSession() as session:
            # Process objects in batches
            for i in tqdm(range(0, len(object_ids), self.batch_size)):
                batch = object_ids[i:i + self.batch_size]
                results = await self.process_batch(session, batch)
                
                for obj_data in results:
                    if obj_data:
                        successful_fetches += 1
                        if obj_data.get("primaryImage"):
                            objects_with_images += 1
                            main_fields = {
                                "id", "objectID", "title", "artistDisplayName", 
                                "objectDate", "medium", "department", "primaryImage",
                                "culture", "period", "dimensions"
                            }
                            misc_data = {k: v for k, v in obj_data.items() if k not in main_fields}
                            
                            filtered_data = {
                                "id": obj_data["objectID"],
                                "title": obj_data["title"],
                                "artist": obj_data["artistDisplayName"],
                                "date": obj_data["objectDate"],
                                "medium": obj_data["medium"],
                                "department": obj_data["department"],
                                "image_url": obj_data["primaryImage"],
                                "culture": obj_data["culture"],
                                "period": obj_data["period"],
                                "dimensions": obj_data["dimensions"],
                                "misc": json.dumps(misc_data)
                            }
                            data_list.append(filtered_data)

        return data_list, total_objects, successful_fetches, objects_with_images

    def fetch_and_save_objects(self, limit=None):
        """Main method to fetch and save objects"""
        print(f"Starting fetch {'with limit '+str(limit) if limit else 'for all objects'}")
        
        data_list, total_objects, successful_fetches, objects_with_images = asyncio.run(
            self.fetch_objects(limit)
        )
        
        # Convert to DataFrame and save as CSV
        df = pd.DataFrame(data_list)
        df.to_csv(self.output_dir / "artworks.csv", index=False)
        
        print("\nFetch Summary:")
        print(f"Total objects in API: {total_objects}")
        print(f"Objects processed: {len(data_list)}")
        print(f"Successful fetches: {successful_fetches}")
        print(f"Objects with images: {objects_with_images}")
        print(f"Data saved to: {self.output_dir / 'artworks.csv'}")

if __name__ == "__main__":
    fetcher = MetArtFetcher()
    fetcher.fetch_and_save_objects()