"""
DIAGNOSTIC SCRIPT: Run this to identify the problem with your predictions
This will show you exactly what's wrong with your model/mapping
"""

import json
import torch
from pathlib import Path
import numpy as np

print("=" * 70)
print("CAR CLASSIFICATION MODEL DIAGNOSTICS")
print("=" * 70)

# -----------------------------
# 1. CHECK CLASS MAPPING FILE
# -----------------------------
print("\n1️⃣  CHECKING CLASS MAPPING...")
print("-" * 70)

try:
    with open('models/idx_to_class.json', 'r') as f:
        idx_to_class = json.load(f)
    
    print(f"✅ Mapping file found: {len(idx_to_class)} classes")
    print("\nCurrent mapping:")
    for idx, name in sorted(idx_to_class.items(), key=lambda x: int(x[0])):
        print(f"  {idx:>3} -> {name}")
    
    # Check class 14 specifically
    if "14" in idx_to_class:
        print(f"\n🔍 Class 14 maps to: {idx_to_class['14']}")
        if idx_to_class['14'] == 'Ferrari':
            print("⚠️  This might be wrong! Check your training data order.")
    
except FileNotFoundError:
    print("❌ ERROR: idx_to_class.json not found!")
    print("   Run 'generate_idx_to_class.py' first")
    idx_to_class = None

# -----------------------------
# 2. CHECK MODEL FILE
# -----------------------------
print("\n2️⃣  CHECKING MODEL FILE...")
print("-" * 70)

try:
    checkpoint = torch.load('models/car_model.pth', map_location='cpu')
    
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("✅ Model checkpoint found (with metadata)")
            if 'epoch' in checkpoint:
                print(f"   Trained epochs: {checkpoint['epoch']}")
            if 'accuracy' in checkpoint:
                print(f"   Final accuracy: {checkpoint['accuracy']:.2%}")
        else:
            state_dict = checkpoint
            print("✅ Model state_dict found")
    else:
        state_dict = checkpoint
        print("✅ Model weights found")
    
    # Get number of classes from model
    fc_weight_shape = state_dict['fc.weight'].shape
    model_num_classes = fc_weight_shape[0]
    
    print(f"\n📊 Model output classes: {model_num_classes}")
    
    if idx_to_class:
        mapping_classes = len(idx_to_class)
        if model_num_classes != mapping_classes:
            print(f"❌ MISMATCH DETECTED!")
            print(f"   Model expects: {model_num_classes} classes")
            print(f"   Mapping has: {mapping_classes} classes")
            print("\n   🔧 FIX: Regenerate idx_to_class.json with correct number of classes")
        else:
            print(f"✅ Model and mapping match: {model_num_classes} classes")

except FileNotFoundError:
    print("❌ ERROR: car_model.pth not found!")
    print("   Train your model first")

except Exception as e:
    print(f"❌ ERROR loading model: {e}")

# -----------------------------
# 3. CHECK TRAINING DATA STRUCTURE
# -----------------------------
print("\n3️⃣  CHECKING TRAINING DATA...")
print("-" * 70)

train_dir = Path("data/train")
if train_dir.exists():
    class_folders = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    print(f"✅ Found {len(class_folders)} class folders:")
    
    for i, folder in enumerate(class_folders):
        img_count = len(list((train_dir / folder).glob('*.[jp][pn][g]')))
        print(f"  {i:>3} -> {folder:<20} ({img_count} images)")
    
    # Check if order matches mapping
    if idx_to_class:
        print("\n🔍 Comparing with current mapping...")
        mismatches = []
        for i, folder in enumerate(class_folders):
            if str(i) in idx_to_class:
                if idx_to_class[str(i)] != folder:
                    mismatches.append((i, folder, idx_to_class[str(i)]))
        
        if mismatches:
            print("❌ MISMATCHES FOUND:")
            for idx, folder, mapped in mismatches:
                print(f"   Index {idx}: Folder='{folder}' but Mapping='{mapped}'")
            print("\n   🔧 FIX: Regenerate idx_to_class.json")
        else:
            print("✅ Mapping matches training folder order")
else:
    print(f"❌ Training directory not found: {train_dir}")

# -----------------------------
# 4. RECOMMENDATION
# -----------------------------
print("\n4️⃣  RECOMMENDATIONS")
print("=" * 70)

recommendations = []

if idx_to_class and '14' in idx_to_class and idx_to_class['14'] == 'Ferrari':
    recommendations.append(
        "⚠️  Class 14 is 'Ferrari' but your image looks like an Audi TT\n"
        "   This suggests your mapping is incorrect."
    )

if idx_to_class and model_num_classes != len(idx_to_class):
    recommendations.append(
        "❌ Model and mapping have different number of classes\n"
        "   You MUST regenerate the mapping or retrain the model."
    )

if not recommendations:
    recommendations.append(
        "✅ No obvious issues detected.\n"
        "   If predictions are still wrong, try:\n"
        "   1. Regenerating idx_to_class.json\n"
        "   2. Retraining the model\n"
        "   3. Testing with different images"
    )

for rec in recommendations:
    print(f"\n{rec}")

print("\n" + "=" * 70)
print("🔧 NEXT STEPS:")
print("=" * 70)
print("1. Run: python generate_idx_to_class.py")
print("2. Restart your API: python src/api.py")
print("3. Test with the same image again")
print("4. If still wrong, retrain your model")
print("=" * 70)