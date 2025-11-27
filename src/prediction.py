import json
import torch
from PIL import Image
from torchvision import transforms
from src.model import load_model
from io import BytesIO

def load_idx_to_class(path='models/idx_to_class.json'):
    """Load the mapping from index to class label."""
    with open(path, 'r') as f:
        return json.load(f)

def get_transform(img_size=224):
    """Return the preprocessing transform for the model."""
    return transforms.Compose([
        transforms.Resize(int(img_size*1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])

def predict_image(image_input, model=None, idx_to_class=None, device='cpu'):
    """
    Predict class and probabilities for an image.

    Args:
        image_input: str path or BytesIO object
        model: preloaded PyTorch model
        idx_to_class: preloaded idx_to_class dict
        device: 'cpu' or 'cuda'

    Returns:
        dict with pred_idx, pred_class, confidence, and top_5_predictions
    """
    if idx_to_class is None:
        idx_to_class = load_idx_to_class()
    num_classes = len(idx_to_class)

    if model is None:
        # Load model if not provided
        model = load_model('models/car_model.pth', num_classes, device=device)
    model.eval()

    # Open image (supports path or BytesIO)
    if isinstance(image_input, str):
        img = Image.open(image_input).convert('RGB')
    else:
        img = Image.open(image_input).convert('RGB')

    # Preprocess
    transform = get_transform()
    x = transform(img).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        outputs = model(x)
        probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = int(outputs.argmax(dim=1).cpu().numpy()[0])
        
        # Get predicted class name
        pred_class = idx_to_class.get(str(pred_idx), "unknown")
        
        # Get confidence (probability of predicted class)
        confidence = float(probs[pred_idx])
        
        # Get top 5 predictions with enhanced formatting
        top5_indices = probs.argsort()[-5:][::-1]
        top5_predictions = [
            {
                "rank": i + 1,
                "class": idx_to_class.get(str(idx), f"class_{idx}"),
                "confidence": f"{float(probs[idx]) * 100:.2f}%",
                "probability": round(float(probs[idx]), 4),
                "confidence_score": round(float(probs[idx]), 4)
            }
            for i, idx in enumerate(top5_indices)
        ]

    return {
        "pred_idx": pred_idx,
        "pred_class": pred_class,
        "confidence": f"{confidence * 100:.2f}%",
        "confidence_score": round(confidence, 4),
        "probability": round(confidence, 4),
        "accuracy_percent": f"{confidence * 100:.2f}%",
        "precision": round(confidence, 4),  # For single prediction context
        "top_5_predictions": top5_predictions,
        "all_probabilities": {
            idx_to_class.get(str(i), f"class_{i}"): round(float(p), 6)
            for i, p in enumerate(probs)
        },
        "total_classes": num_classes,
        "model_info": {
            "num_classes": num_classes,
            "device": str(device)
        }
    }