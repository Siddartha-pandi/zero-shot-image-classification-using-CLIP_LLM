#!/usr/bin/env python3
"""
Quick test script for hybrid classification system
"""
import requests
import sys
from pathlib import Path

API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"✓ Health check: {response.json()}")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False

def test_models_status():
    """Test models status endpoint"""
    print("\nTesting models status...")
    try:
        response = requests.get(f"{API_URL}/api/models/status")
        data = response.json()
        print(f"✓ Models status:")
        print(f"  - ViT-H/14: {'✓' if data['models']['vit_h14']['loaded'] else '✗'} ({data['models']['vit_h14']['device']})")
        print(f"  - MedCLIP: {'✓' if data['models']['medclip']['loaded'] else '✗'} ({data['models']['medclip']['device']})")
        return True
    except Exception as e:
        print(f"✗ Models status failed: {e}")
        return False

def test_domains():
    """Test domains endpoint"""
    print("\nTesting domains endpoint...")
    try:
        response = requests.get(f"{API_URL}/api/domains")
        data = response.json()
        print(f"✓ Supported domains: {', '.join(data['domains'])}")
        return True
    except Exception as e:
        print(f"✗ Domains test failed: {e}")
        return False

def test_classification(image_path: str):
    """Test classification with image"""
    print(f"\nTesting classification with {image_path}...")
    try:
        path = Path(image_path)
        if not path.exists():
            print(f"✗ Image file not found: {image_path}")
            return False
        
        with open(path, 'rb') as f:
            files = {'file': (path.name, f, 'image/jpeg')}
            response = requests.post(
                f"{API_URL}/api/classify-hybrid",
                files=files,
                data={'top_k': 5}
            )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Classification successful:")
            print(f"  Domain: {data['domain']}")
            print(f"  Model: {data['model_used']}")
            print(f"  Prediction: {data['prediction']}")
            print(f"  Confidence: {data['confidence_score']:.2%}")
            print(f"  Inference time: {data['inference_time_seconds']:.2f}s")
            print(f"\n  Top matches:")
            for match in data['top_matches'][:3]:
                print(f"    - {match['label']}: {match['score']:.2%}")
            return True
        else:
            print(f"✗ Classification failed: {response.status_code}")
            print(f"  {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Classification test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Hybrid Classification System - Test Suite")
    print("=" * 60)
    
    tests_passed = 0
    tests_total = 3
    
    if test_health():
        tests_passed += 1
    
    if test_models_status():
        tests_passed += 1
    
    if test_domains():
        tests_passed += 1
    
    # Optional: test with image if provided
    if len(sys.argv) > 1:
        tests_total += 1
        if test_classification(sys.argv[1]):
            tests_passed += 1
    
    print("\n" + "=" * 60)
    print(f"Tests passed: {tests_passed}/{tests_total}")
    print("=" * 60)
    
    return tests_passed == tests_total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
