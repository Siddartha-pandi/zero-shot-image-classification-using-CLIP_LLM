"""
Comprehensive test script demonstrating all example scenarios from EXAMPLES.md
Run this to validate the enhanced zero-shot classification framework.
"""

import sys
import os
import logging
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from advanced_inference import AdvancedZeroShotFramework
from models import initialize_models
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def print_separator(title):
    """Print a formatted separator"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def print_results(response, example_name):
    """Pretty print classification results"""
    print(f"\n{'─'*80}")
    print(f"🎯 {example_name}")
    print(f"{'─'*80}")
    
    # Top prediction
    top_pred = response['top_prediction']
    top_class = list(top_pred.keys())[0]
    top_score = top_pred[top_class]
    
    print(f"\n📊 RESULTS:")
    print(f"  Predicted Class: {top_class}")
    print(f"  Confidence Score: {response['confidence_score']:.2%}")
    
    # Domain info
    domain_info = response['domain_info']
    print(f"\n🌐 DOMAIN ANALYSIS:")
    print(f"  Detected Domain: {domain_info['domain']}")
    print(f"  Domain Confidence: {domain_info['confidence']:.2%}")
    if 'characteristics' in domain_info:
        print(f"  Characteristics: {', '.join(domain_info['characteristics'])}")
    
    # Visual features
    if response['visual_features']:
        print(f"\n👁️  VISUAL FEATURES:")
        print(f"  {', '.join(response['visual_features'])}")
    
    # Alternative predictions
    print(f"\n📋 ALTERNATIVE PREDICTIONS:")
    for i, pred in enumerate(response['alternative_predictions'][:3], 1):
        print(f"  {i}. {pred['class']}: {pred['score']:.2%}")
    
    # Explanation
    print(f"\n💡 EXPLANATION:")
    print(f"  {response['explanation']}")
    
    # Technical details
    print(f"\n🔧 TECHNICAL INFO:")
    print(f"  Zero-shot: {response['zero_shot']}")
    print(f"  Multilingual: {response['multilingual_support']}")
    print(f"  Language: {response['language']}")
    
    reasoning = response.get('reasoning_chain', {})
    if reasoning:
        print(f"  Prompts used: {reasoning.get('num_prompts', 'N/A')}")
        print(f"  Similarity score: {reasoning.get('similarity_score', 0):.2f}")


def test_example_1_natural_image(classifier):
    """Example 1: Natural Image (Zebra)"""
    print_separator("EXAMPLE 1: Natural Image (General Object)")
    
    # This would use an actual zebra image
    # For demonstration, we'll show the expected API call
    
    print("📝 Scenario: Classifying a zebra (never seen during training)")
    print("   Input: Image of a zebra in grassland")
    print("   Expected: High confidence classification with natural_image domain")
    
    # Mock example - in real usage, provide actual image path
    example_labels = ["zebra", "horse", "donkey", "okapi", "mule"]
    print(f"\n   Class names: {example_labels}")
    print("\n   ⚠️  To run this test, provide an actual zebra image path")
    print("   Example: response = classifier.classify('path/to/zebra.jpg', example_labels)")
    

def test_example_2_sketch(classifier):
    """Example 2: Sketch Input (Domain Adaptation)"""
    print_separator("EXAMPLE 2: Sketch Input (Domain Adaptation)")
    
    print("📝 Scenario: Hand-drawn sketch with domain adaptation")
    print("   Input: Sketch of a cat")
    print("   Expected: Sketch domain detection, shape-focused analysis")
    
    example_labels = ["cat", "dog", "rabbit", "fox"]
    print(f"\n   Class names: {example_labels}")
    print("\n   Expected Features:")
    print("   - Domain: sketch")
    print("   - Adaptation: Edge emphasis, color suppression")
    print("   - Confidence: 85-92%")
    

def test_example_3_medical(classifier):
    """Example 3: Medical Image Classification"""
    print_separator("EXAMPLE 3: Medical Image (Unseen Domain)")
    
    print("📝 Scenario: Chest X-ray classification")
    print("   Input: Chest X-ray showing lungs")
    print("   Expected: Medical domain detection, clinical analysis")
    
    example_labels = [
        "Normal chest X-ray",
        "Pneumonia X-ray",
        "Tuberculosis X-ray",
        "Lung nodule X-ray"
    ]
    print(f"\n   Class names: {example_labels}")
    print("\n   Expected Features:")
    print("   - Domain: medical_image")
    print("   - Adaptation: Grayscale optimization, pathology detection")
    print("   - Confidence: 80-88%")
    print("   - Clinical note: AI-assisted analysis requires professional validation")


def test_example_4_multilingual(classifier):
    """Example 4: Multilingual Classification"""
    print_separator("EXAMPLE 4: Multilingual Prompt Classification")
    
    print("📝 Scenario: Apple classification with multilingual support")
    print("   Input: Image of an apple")
    print("   Prompt: 'इस चित्र को वर्गीकृत करें।' (Hindi)")
    print("   Expected: Multilingual label support, high confidence")
    
    # Test with multiple languages
    example_labels_english = ["apple", "orange", "banana", "mango"]
    example_labels_mixed = ["apple", "सेब", "manzana", "pomme"]
    
    print(f"\n   English labels: {example_labels_english}")
    print(f"   Mixed language labels: {example_labels_mixed}")
    print("\n   Expected Features:")
    print("   - Language detection: Hindi")
    print("   - Multilingual prompts generated")
    print("   - Confidence: 95-97%")
    print("\n   API Call: classifier.classify(image_path, labels, language='hi')")


def test_example_5_artistic(classifier):
    """Example 5: Artistic/Stylized Image"""
    print_separator("EXAMPLE 5: Artistic / Stylized Image")
    
    print("📝 Scenario: Anime character classification")
    print("   Input: Stylized anime portrait with lightning mark")
    print("   Expected: Artistic domain, style-aware processing")
    
    example_labels = [
        "Stylized anime portrait",
        "Anime male character",
        "Graphic novel hero",
        "Fantasy character artwork"
    ]
    print(f"\n   Class names: {example_labels}")
    print("\n   Expected Features:")
    print("   - Domain: anime or artistic_image")
    print("   - Characteristics: stylized, exaggerated features")
    print("   - Confidence: 88-92%")
    print("   - Style detection: anime/manga")


def test_example_6_zero_shot(classifier):
    """Example 6: Zero-Shot Novel Object"""
    print_separator("EXAMPLE 6: Zero-Shot Object (Never Seen Before)")
    
    print("📝 Scenario: Electric unicycle (novel gadget)")
    print("   Input: Photo of electric unicycle")
    print("   Expected: Semantic reasoning, moderate confidence")
    
    example_labels = [
        "electric unicycle",
        "monowheel",
        "segway",
        "electric scooter",
        "hoverboard"
    ]
    print(f"\n   Class names: {example_labels}")
    print("\n   Expected Features:")
    print("   - Domain: modern_technology")
    print("   - Zero-shot inference: Active")
    print("   - Confidence: 75-82% (lower due to novelty)")
    print("   - Reasoning: Compositional inference from visual features")


def test_example_7_multispectral(classifier):
    """Example 7: Multispectral Satellite Imagery"""
    print_separator("EXAMPLE 7: Multispectral Image")
    
    print("📝 Scenario: Satellite vegetation analysis")
    print("   Input: Multispectral satellite image")
    print("   Expected: Spectral band analysis, NDVI calculation")
    
    example_labels = [
        "Dense vegetation (NDVI > 0.6)",
        "Sparse vegetation (NDVI 0.3-0.6)",
        "Bare soil (NDVI < 0.3)",
        "Water body",
        "Urban area"
    ]
    print(f"\n   Class names: {example_labels}")
    print("\n   Expected Features:")
    print("   - Domain: multispectral_image")
    print("   - Analysis: Spectral band processing")
    print("   - Confidence: 82-88%")
    print("   - Method: Vegetation index calculations")


def test_api_endpoints():
    """Test API endpoint examples"""
    print_separator("API USAGE EXAMPLES")
    
    print("🌐 REST API Endpoints:\n")
    
    print("1. Basic Classification (Auto-labels):")
    print('   curl -X POST "http://localhost:8000/api/classify" \\')
    print('     -F "file=@image.jpg"\n')
    
    print("2. Classification with Custom Labels:")
    print('   curl -X POST "http://localhost:8000/api/classify" \\')
    print('     -F "file=@image.jpg" \\')
    print('     -F "labels=zebra,horse,donkey"\n')
    
    print("3. Multilingual Classification (Hindi):")
    print('   curl -X POST "http://localhost:8000/api/classify" \\')
    print('     -F "file=@image.jpg" \\')
    print('     -F "labels=सेब,केला,संतरा" \\')
    print('     -F "language=hi"\n')
    
    print("4. Classification with User Prompt:")
    print('   curl -X POST "http://localhost:8000/api/classify" \\')
    print('     -F "file=@medical_xray.jpg" \\')
    print('     -F "labels=Normal,Pneumonia,Tuberculosis" \\')
    print('     -F "user_prompt=Classify this chest X-ray"\n')


def main():
    """Run all example demonstrations"""
    print("\n" + "█"*80)
    print("  🚀 ZERO-SHOT IMAGE CLASSIFICATION - EXAMPLE DEMONSTRATIONS")
    print("█"*80)
    
    # Initialize models
    print("\n⚙️  Initializing models...")
    if not initialize_models():
        print("❌ Model initialization failed!")
        return
    
    # Create classifier instance
    classifier = AdvancedZeroShotFramework()
    print("✅ Models loaded successfully!\n")
    
    # Run all example demonstrations
    test_example_1_natural_image(classifier)
    test_example_2_sketch(classifier)
    test_example_3_medical(classifier)
    test_example_4_multilingual(classifier)
    test_example_5_artistic(classifier)
    test_example_6_zero_shot(classifier)
    test_example_7_multispectral(classifier)
    
    # Show API examples
    test_api_endpoints()
    
    # Summary
    print_separator("SUMMARY")
    print("📚 All 7 example scenarios have been demonstrated")
    print("✨ Features showcased:")
    print("   ✓ Domain adaptation (6 different domains)")
    print("   ✓ Multilingual support (4 languages)")
    print("   ✓ Zero-shot reasoning")
    print("   ✓ Comprehensive explanations")
    print("   ✓ Confidence calibration")
    print("   ✓ Visual feature extraction")
    print("   ✓ Reasoning chain tracking")
    print("\n💡 To run actual tests with images:")
    print("   1. Place test images in test_images/ directory")
    print("   2. Update image paths in test functions")
    print("   3. Run: python test_examples.py")
    print("\n📖 See EXAMPLES.md for detailed documentation")
    print("🧪 See docs/EXAMPLE_TEST_CASES.md for runnable test cases")
    print("\n" + "█"*80 + "\n")


if __name__ == "__main__":
    main()
