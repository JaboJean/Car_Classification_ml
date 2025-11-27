"""
CRITICAL: Run this script to regenerate your idx_to_class.json mapping
This ensures the class indices match your model's training order
"""

import json
from pathlib import Path
import os

# -----------------------------
# STEP 1: Set your training directory path
# -----------------------------
train_dir = "data/train"  # Your training data folder

# -----------------------------
# STEP 2: Generate mapping from folder names
# -----------------------------
def generate_class_mapping(train_dir):
    """
    Generate idx_to_class mapping from training directory structure.
    IMPORTANT: Classes are sorted alphabetically - this MUST match training!
    """
    train_path = Path(train_dir)
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    
    # Get all subdirectories (each is a class)
    class_dirs = [d for d in train_path.iterdir() if d.is_dir()]
    
    if not class_dirs:
        raise ValueError(f"No class directories found in {train_dir}")
    
    # Sort alphabetically (CRITICAL: must match training order)
    class_names = sorted([d.name for d in class_dirs])
    
    # Create mapping: index -> class name
    idx_to_class = {str(i): name for i, name in enumerate(class_names)}
    
    return idx_to_class, class_names

# -----------------------------
# STEP 3: Generate and save mapping
# -----------------------------
try:
    idx_to_class, class_names = generate_class_mapping(train_dir)
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Save to JSON
    json_path = "models/idx_to_class.json"
    with open(json_path, "w") as f:
        json.dump(idx_to_class, f, indent=4)
    
    print("✅ SUCCESS: idx_to_class.json generated!")
    print(f"📁 Saved to: {json_path}")
    print(f"📊 Number of classes: {len(class_names)}")
    print("\n🏷️  Class Mapping:")
    print("-" * 50)
    for idx, name in idx_to_class.items():
        print(f"  {idx:>3} -> {name}")
    print("-" * 50)
    
    # -----------------------------
    # STEP 4: VERIFY the mapping
    # -----------------------------
    print("\n🔍 VERIFICATION:")
    print("=" * 50)
    
    # Check if class 14 exists
    if "14" in idx_to_class:
        print(f"Class 14 is: {idx_to_class['14']}")
        print("⚠️  If this should NOT be 'Ferrari', your model was trained")
        print("   with a different class order than your current folders!")
    else:
        print("⚠️  Class 14 does not exist in your mapping")
        print(f"   You only have {len(class_names)} classes (0-{len(class_names)-1})")
    
    print("\n💡 IMPORTANT NOTES:")
    print("=" * 50)
    print("1. This mapping is generated from your CURRENT folder structure")
    print("2. It MUST match the order used during model TRAINING")
    print("3. If predictions are still wrong, your model might need retraining")
    print("4. Classes are sorted ALPHABETICALLY by default")
    
except FileNotFoundError as e:
    print(f"❌ ERROR: {e}")
    print(f"\n💡 Make sure '{train_dir}' exists and contains class folders")
    print("\nExpected structure:")
    print("  data/train/")
    print("    ├── Audi/")
    print("    ├── BMW/")
    print("    ├── Ferrari/")
    print("    └── ...")

except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()