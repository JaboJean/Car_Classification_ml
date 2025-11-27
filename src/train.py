# src/train.py
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.preprocessing import create_dataloaders
from src.model import build_model, save_model
import numpy as np

def train(train_dir='data/train', val_dir='data/test', epochs=10, batch_size=32, lr=1e-3, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, num_classes = create_dataloaders(train_dir, val_dir, batch_size=batch_size)
    model = build_model(num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0.0

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []
        start = time.time()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = accuracy_score(all_labels, all_preds)

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1).detach().cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(val_labels, val_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='weighted')
        elapsed = time.time() - start

        print(f"Epoch {epoch}/{epochs}  time={elapsed:.1f}s")
        print(f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}")
        print(f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('models', exist_ok=True)
            save_model(model, 'models/best_model.pth')
            print("  Saved best_model.pth")

    print("Training complete. Best val acc:", best_val_acc)

if __name__=='__main__':
    train()
