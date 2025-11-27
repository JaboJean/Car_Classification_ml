import requests
import time

def test_api():
    url = "http://127.0.0.1:8001/predict-file"
    img_path = r"C:/Users/Latitude/.gemini/antigravity/brain/105daa86-763a-4739-9204-2b5db24b60f2/uploaded_image_0_1764249413438.png"
    
    print(f"Sending request to {url}...")
    try:
        with open(img_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files)
        
        if response.status_code == 200:
            data = response.json()
            print("Success!")
            print(f"Prediction: {data.get('pred_class')}")
            print(f"Confidence: {data.get('confidence')}")
            print("Top 5 keys present:", 'top_5_predictions' in data)
            if 'top_5_predictions' in data:
                print(data['top_5_predictions'])
        else:
            print(f"Failed: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Wait a bit for server to start
    time.sleep(5)
    test_api()
