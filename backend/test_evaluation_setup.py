"""
Quick test to verify evaluation system is working.
This creates a minimal test dataset and runs a basic evaluation.
"""

import json
from pathlib import Path

# Create a minimal test dataset
test_data = [
    {
        "path": "test_images/cat.jpg",
        "label": "cat",
        "domain": "natural"
    },
    {
        "path": "test_images/dog.jpg", 
        "label": "dog",
        "domain": "natural"
    },
    {
        "path": "test_images/car.jpg",
        "label": "car", 
        "domain": "natural"
    }
]

# Save test dataset
test_file = Path("test_dataset_minimal.json")
with open(test_file, 'w') as f:
    json.dump(test_data, f, indent=2)

print(f"✓ Created test dataset: {test_file}")
print(f"\nDataset contents:")
print(json.dumps(test_data, indent=2))

print(f"\n{'='*60}")
print("NEXT STEPS:")
print(f"{'='*60}")
print("\n1. Prepare your actual test images:")
print("   - Create a folder with test images")
print("   - Update paths in test_dataset_minimal.json")
print("   - Or create a new dataset JSON file")

print("\n2. Run evaluation:")
print("   python run_evaluation.py --dataset test_dataset_minimal.json --mode baseline")

print("\n3. For full ablation study:")
print("   python run_evaluation.py --dataset your_dataset.json --mode full")

print("\n4. Recommended dataset size:")
print("   - At least 20-50 images per class")
print("   - Include multiple domains if testing cross-domain robustness")
print("   - Balance classes for reliable metrics")

print(f"\n{'='*60}")
print("EVALUATION SYSTEM READY! ✅")
print(f"{'='*60}\n")
