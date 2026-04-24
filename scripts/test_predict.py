import requests
import sys
import os

def test_prediction(image_path, url="http://localhost:8000/api/v1/predict"):
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} not found")
        return

    print(f"Testing prediction for: {image_path}")
    
    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
        response = requests.post(url, files=files)

    if response.status_code == 200:
        print("Success!")
        print(response.json())
    else:
        print(f"Failed with status code {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_predict.py <path_to_image>")
        sys.exit(1)
    
    test_prediction(sys.argv[1])
