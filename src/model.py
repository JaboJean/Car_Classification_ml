# src/model.py
import torch
import torch.nn as nn
from torchvision import models

def build_model(num_classes):
    """
    Build a ResNet18 model with `num_classes` output.
    """
    model = models.resnet18(weights=None)  # do not load ImageNet weights
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(path, num_classes=None, device='cpu'):
    """
    Load a model from a .pth file.
    If num_classes is provided, use it.
    Otherwise, infer num_classes from checkpoint.
    """
    checkpoint = torch.load(path, map_location=device)

    # Infer num_classes if not provided
    if num_classes is None:
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        # fc.weight shape is [num_classes, in_features]
        num_classes = state_dict['fc.weight'].shape[0]

    model = build_model(num_classes)
    # load state dict
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
