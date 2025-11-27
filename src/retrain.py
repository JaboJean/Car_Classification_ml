import os
import json
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms

from src.model import build_model, save_model

# -----------------------------
# CONFIG
# -----------------------------
UPLOAD_DIR = "uploads/retrain"
MODEL_PATH = "models/car_model.pth"
IDX_PATH = "models/idx_to_class.json"
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# UTILS
# -----------------------------
def get_num_classes(data_dir):
    dataset = ImageFolder(data_dir)
    return len(dataset.classes), dataset.classes

def get_transforms(img_size=224):
    return transforms.Compose([
        transforms.Resize(int(img_size*1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

# -----------------------------
# MAIN RETRAIN FUNCTION
# -----------------------------
def retrain_entrypoint(upload_dir=UPLOAD_DIR, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR):
    train_dir = os.path.join(upload_dir, "train")
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    # Detect classes dynamically
    num_classes, class_names = get_num_classes(train_dir)
    print(f"Found {num_classes} classes: {class_names}")

    # Save idx_to_class.json
    os.makedirs(os.path.dirname(IDX_PATH), exist_ok=True)
    idx_to_class = {i: name for i, name in enumerate(class_names)}
    with open(IDX_PATH, "w") as f:
        json.dump(idx_to_class, f)
    print(f"idx_to_class.json saved to {IDX_PATH}")

    # Load dataset
    transform = get_transforms(IMG_SIZE)
    dataset = ImageFolder(train_dir, transform=transform)

    # Handle class imbalance with WeightedRandomSampler
    class_counts = Counter([label for _, label in dataset])
    weights = [1.0 / class_counts[label] for _, label in dataset]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    # Build model
    model = build_model(num_classes=num_classes, pretrained=True)
    model.to(DEVICE)

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

    # Save trained model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    save_model(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print("Retraining complete!")

# -----------------------------
# RUN DIRECTLY
# -----------------------------
if __name__ == "__main__":
    retrain_entrypoint()
