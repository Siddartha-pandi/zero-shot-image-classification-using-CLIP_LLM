#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced explanation generator
with detailed narrative descriptions
"""
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from PIL import Image
from services.explanation_generator import generate_explanation

def demo_explanation():
    """Demonstrate the enhanced explanation generator"""
    
    print("\n" + "="*80)
    print("ENHANCED EXPLANATION GENERATOR DEMO")
    print("="*80 + "\n")
    
    # Example scenarios
    scenarios = [
        {
            "title": "📋 Student ID Card",
            "domain": "identification",
            "prediction": "Student ID Card",
            "confidence": 0.92,
            "caption": "A photo ID card with student information, showing a portrait photo, name, student ID number, and institutional details",
            "top_matches": [
                {"label": "Student ID Card", "score": 0.92},
                {"label": "Identity Card", "score": 0.78},
                {"label": "Document", "score": 0.65}
            ]
        },
        {
            "title": "👕 Clothing Item (T-Shirt)",
            "domain": "clothing",
            "prediction": "Blue T-Shirt",
            "confidence": 0.88,
            "caption": "A casual blue cotton t-shirt with crew neckline, short sleeves, and relaxed fit",
            "top_matches": [
                {"label": "Blue T-Shirt", "score": 0.88},
                {"label": "Casual Shirt", "score": 0.75},
                {"label": "Apparel", "score": 0.68}
            ]
        },
        {
            "title": "🥬 Vegetable (Cauliflower)",
            "domain": "vegetable",
            "prediction": "Cauliflower",
            "confidence": 0.95,
            "caption": "Fresh white cauliflower head with tightly clustered florets surrounded by green leafy bracts",
            "top_matches": [
                {"label": "Cauliflower", "score": 0.95},
                {"label": "Broccoli", "score": 0.72},
                {"label": "Vegetable", "score": 0.88}
            ]
        },
        {
            "title": "🍝 Food Dish (Pasta)",
            "domain": "food",
            "prediction": "Pasta Dish",
            "confidence": 0.91,
            "caption": "A plate of creamy white pasta with herbs and garnish, served on a ceramic plate",
            "top_matches": [
                {"label": "Pasta Dish", "score": 0.91},
                {"label": "Italian Food", "score": 0.84},
                {"label": "Prepared Meal", "score": 0.79}
            ]
        }
    ]
    
    # Test each scenario
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{scenario['title']}")
        print("-" * 80)
        print(f"Domain: {scenario['domain']}")
        print(f"Prediction: {scenario['prediction']}")
        print(f"Confidence: {scenario['confidence']*100:.0f}%")
        print(f"Caption: {scenario['caption']}")
        
        print("\n📝 ENHANCED EXPLANATION (100-150 words):")
        print("-" * 80)
        
        try:
            explanation = generate_explanation(
                domain=scenario['domain'],
                model_used="ViT-H/14 CLIP + Gemini",
                prediction=scenario['prediction'],
                confidence=scenario['confidence'],
                caption=scenario['caption'],
                top_matches=scenario['top_matches'],
                image=None
            )
            
            word_count = len(explanation.split())
            print(f"\n{explanation}\n")
            print(f"✅ Word Count: {word_count} words")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n")

if __name__ == "__main__":
    demo_explanation()
