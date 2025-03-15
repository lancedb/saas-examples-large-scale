import requests
import json
import base64
import sys
import time

# Replace with your actual Modal endpoint URL
# It should look like: https://username--art-search-search.modal.run
endpoint_url = "https://lancedb--art-search-search.modal.run"

def test_text_search():
    """Test searching by text query"""
    payload = {
        "query": "a painting of mountains",
        "model": "siglip",
        "limit": 5
    }
    
    print(f"Sending text search request to: {endpoint_url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    # Add retries for transient errors
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(1, max_retries + 1):
        try:
            # Send the request with debug information
            print(f"\nAttempt {attempt}/{max_retries}:")
            
            # Start with a simple GET request to check if the endpoint is alive
            if attempt > 1:
                try:
                    health_check = requests.get(endpoint_url.split("/search")[0], timeout=5)
                    print(f"Health check status: {health_check.status_code}")
                except Exception as e:
                    print(f"Health check failed: {e}")
            
            # Make the actual POST request
            start_time = time.time()
            print(f"Starting request at: {start_time}")
            
            # Add debug headers
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'User-Agent': 'TextSearchTest/1.0'
            }
            
            response = requests.post(
                endpoint_url, 
                json=payload, 
                headers=headers,
                timeout=1000  # Longer timeout for first request
            )
            
            end_time = time.time()
            print(f"Request completed in {end_time - start_time:.2f} seconds")
            
            # Print response details
            print(f"Status code: {response.status_code}")
            print(f"Response headers: {dict(response.headers)}")
            
            # Check if we got content
            if response.text:
                print(f"Response length: {len(response.text)} bytes")
                print(f"Response content preview: {response.text[:200]}...")
                
                try:
                    result = response.json()
                    print("✅ JSON response successfully parsed")
                    print(f"Response data: {json.dumps(result, indent=2)}")
                    return True
                except json.JSONDecodeError as e:
                    print(f"❌ JSON decoding failed: {e}")
                    print(f"Raw response first 500 chars: {response.text[:500]}")
                    
                    # Try to check if we're dealing with HTML error page
                    if "<html" in response.text.lower():
                        print("Response appears to be HTML, likely an error page")
                    elif response.text.strip() == "Internal Server Error":
                        print("Got basic 'Internal Server Error' message")
                    
                    # If this is not the last attempt, try again
                    if attempt < max_retries:
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    return False
            else:
                print("❌ Empty response received")
                
                # If this is not the last attempt, try again
                if attempt < max_retries:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}", file=sys.stderr)
            
            # If this is not the last attempt, try again
            if attempt < max_retries:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
                continue
            return False
    
    # If we get here, all retries failed
    return False

def test_image_search(image_path="test_image.jpg"):
    """Test searching by image"""
    try:
        # Read and encode the image
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        
        payload = {
            "image": image_base64,
            "model": "siglip",
            "limit": 5
        }
        
        print(f"Sending image search request to: {endpoint_url}")
        print(f"Image size: {len(image_bytes)} bytes")
        
        # Send the request
        response = requests.post(endpoint_url, json=payload, timeout=1000)  # Longer timeout for image processing
        
        # Print response details
        print(f"Status code: {response.status_code}")
        
        # Check if we got content
        if response.text:
            try:
                result = response.json()
                print("JSON response successfully parsed")
                print(f"Response data: {json.dumps(result, indent=2)}")
                return True
            except json.JSONDecodeError as e:
                print(f"❌ JSON decoding failed: {e}")
                print(f"Raw response: {response.text[:500]}")
                return False
        else:
            print("❌ Empty response received")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}", file=sys.stderr)
        return False

def search_random():
    """Test random search"""
    payload = {
        "model": "clip",
        "limit": 5
    }

    print(f"Sending random search request to: {endpoint_url}")
    # Send the request
    response = requests.post(endpoint_url, json=payload, timeout=1000)  # Longer timeout for random search
    # Print response details
    print(f"Status code: {response.status_code}")
    # Check if we got content
    if response.text:
        try:
            result = response.json()
            print("JSON response successfully parsed")
            print(f"Response data: {json.dumps(result, indent=2)}")
            return True
        except json.JSONDecodeError as e:
            print(f"❌ JSON decoding failed: {e}")

if __name__ == "__main__":
    import lancedb

    db = lancedb.connect(
    uri="db://devrel-samp-9a5467",
    api_key="sk_THUBNC75R5AYPMMRMUV6SEPWWLSPXY7ZSKVQPUUFVCOOQGYKGUKA====",
    region="us-east-1"
    )

    tbl = db.open_table("artworks-clip-base-embed-api")
    #print(len(tbl))
    import pdb; pdb.set_trace()
    '''
    # Update this URL with your actual Modal deployment URL
    if endpoint_url == "https://your-username--art-search-search.modal.run":
        print("⚠️ Please update the endpoint_url in the script with your actual Modal deployment URL")
        sys.exit(1)
        
    print("=== Testing Text Search ===")
    text_success = test_text_search()
    
    print("\n=== Testing Image Search ===")
    # Uncomment and provide image path to test image search
    # image_success = test_image_search("path/to/test_image.jpg")
    image_success = "Skipped"
    
    print("\n=== Testing Random Search ===")
    random_success = search_random()

    print("\n=== Test Results ===")
    print(f"Text Search: {'✅ Success' if text_success else '❌ Failed'}")
    print(f"Image Search: {image_success}")
    '''