import torch
from src.prediction import predict_image, load_idx_to_class
from src.model import load_model
import os

def reproduce():
    # Paths
    model_path = 'models/car_model.pth'
    idx_path = 'models/idx_to_class.json'
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    # Load resources
    idx_to_class = load_idx_to_class(idx_path)
    model = load_model(model_path, device='cpu')
    
    # Test images (using the paths provided in metadata)
    test_images = [
        r"C:/Users/Latitude/.gemini/antigravity/brain/105daa86-763a-4739-9204-2b5db24b60f2/uploaded_image_0_1764249413438.png",
        r"C:/Users/Latitude/.gemini/antigravity/brain/105daa86-763a-4739-9204-2b5db24b60f2/uploaded_image_1_1764249413438.png",
        r"C:/Users/Latitude/.gemini/antigravity/brain/105daa86-763a-4739-9204-2b5db24b60f2/uploaded_image_2_1764249413438.png"
    ]
    
    print(f"Testing {len(test_images)} images...")
    
    for i, img_path in enumerate(test_images):
        if not os.path.exists(img_path):
            print(f"Image {img_path} not found.")
            continue
            
        print(f"\nImage {i+1}: {img_path}")
        try:
            result = predict_image(img_path, model=model, idx_to_class=idx_to_class)
            print(f"Prediction: {result['pred_class']}")
            print(f"Confidence: {result['confidence']}")
            print("Top 5:")
            for p in result['top_5_predictions']:
                print(f"  {p['class']}: {p['confidence']}")
        except Exception as e:
            print(f"Error predicting image: {e}")

if __name__ == "__main__":
    reproduce()
