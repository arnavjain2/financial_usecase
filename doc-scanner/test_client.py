# test_client.py

import requests
import sys
import os
import json

# --- CONFIGURATION ---
BASE_URL = "http://127.0.0.1:8000"
ENDPOINT = "/v1/extract"
API_URL = BASE_URL + ENDPOINT

def test_kyc_extraction(file_path: str):
    """
    Sends an image file to the KYC Agent API and prints the response.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return

    print(f"--- Testing with image: {os.path.basename(file_path)} ---")
    
    # Prepare the file for multipart/form-data upload
    with open(file_path, 'rb') as f:
        files = {'image': (os.path.basename(file_path), f, 'image/jpeg')} # You can adjust mime-type if needed
        

        payload = {'external_transaction_id': 'test-client-001'}
        
        try:
            print(f"Sending POST request to {API_URL}...")
            response = requests.post(API_URL, files=files, data=payload, timeout=90) 

            print(f"\nStatus Code: {response.status_code}")
            print("Response JSON:")
            
            print(json.dumps(response.json(), indent=2))

        except requests.exceptions.RequestException as e:
            print(f"\nAn error occurred while making the request: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_client.py <path_to_image_file>")
        sys.exit(1)
        
    image_to_test = sys.argv[1]
    test_kyc_extraction(image_to_test)