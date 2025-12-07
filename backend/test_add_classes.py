"""
Quick script to add initial classes to the CLIP backend
Run this after starting the server to populate some default classes
"""
import requests

# Backend URL
BASE_URL = "http://127.0.0.1:8000"

# Example classes to add (customize these based on your needs)
CLASSES = [
    {"label": "cat", "domain": "natural"},
    {"label": "dog", "domain": "natural"},
    {"label": "car", "domain": "natural"},
    {"label": "airplane", "domain": "natural"},
    {"label": "bird", "domain": "natural"},
    {"label": "bicycle", "domain": "natural"},
    {"label": "person", "domain": "natural"},
    {"label": "horse", "domain": "natural"},
    {"label": "tree", "domain": "natural"},
    {"label": "building", "domain": "natural"},
]

def add_class(label: str, domain: str = "natural"):
    """Add a single class to the backend"""
    try:
        response = requests.post(
            f"{BASE_URL}/api/add-class",
            data={"label": label, "domain": domain}
        )
        if response.status_code == 200:
            print(f"✅ Added: {label} (domain: {domain})")
            return True
        else:
            print(f"❌ Failed to add {label}: {response.json()}")
            return False
    except Exception as e:
        print(f"❌ Error adding {label}: {e}")
        return False

def list_classes():
    """List all classes currently in the backend"""
    try:
        response = requests.get(f"{BASE_URL}/api/classes")
        if response.status_code == 200:
            classes = response.json()["classes"]
            print(f"\n📋 Current classes ({len(classes)}): {', '.join(classes)}")
        else:
            print(f"❌ Failed to get classes: {response.json()}")
    except Exception as e:
        print(f"❌ Error getting classes: {e}")

if __name__ == "__main__":
    print("🚀 Adding classes to CLIP backend...\n")
    
    # Add all classes
    for cls in CLASSES:
        add_class(cls["label"], cls["domain"])
    
    # List final classes
    list_classes()
    print("\n✅ Done! You can now classify images.")
