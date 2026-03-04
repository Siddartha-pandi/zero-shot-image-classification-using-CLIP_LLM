"""
Demo Script - Reference Predictions Showcase
Tests domain routing and classification using reference prediction examples
"""
import json
from pathlib import Path


def load_reference_predictions():
    """Load reference predictions"""
    file_path = Path(__file__).parent / "reference_predictions.json"
    with open(file_path, 'r') as f:
        return json.load(f)


def test_domain_inference():
    """Test domain inference from captions and keywords"""
    from app.domain_service import infer_domain_from_caption
    
    data = load_reference_predictions()
    examples = data['examples']
    
    print("\n" + "="*80)
    print(" DOMAIN INFERENCE TEST")
    print("="*80)
    
    correct = 0
    total = 0
    
    for ex in examples:
        caption = ex['caption']
        expected_domain = "medical" if ex['domain'] == "Medical Imaging" else "industrial"
        
        inferred = infer_domain_from_caption(caption)
        match = "✓" if inferred == expected_domain else "✗"
        
        total += 1
        if inferred == expected_domain:
            correct += 1
        
        print(f"\n{match} Caption: {caption}")
        print(f"  Expected: {expected_domain} | Inferred: {inferred}")
    
    accuracy = (correct / total) * 100
    print(f"\n{'='*80}")
    print(f"Domain Inference Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    print(f"{'='*80}\n")


def test_label_availability():
    """Test if predicted labels exist in DEFAULT_LABELS"""
    from app.routes.classify import DEFAULT_LABELS
    
    data = load_reference_predictions()
    examples = data['examples']
    
    print("\n" + "="*80)
    print(" LABEL AVAILABILITY TEST")
    print("="*80)
    
    medical_labels = set([l.lower() for l in DEFAULT_LABELS['medical']])
    industrial_labels = set([l.lower() for l in DEFAULT_LABELS['industrial']])
    
    for ex in examples:
        domain = "medical" if ex['domain'] == "Medical Imaging" else "industrial"
        prediction = ex['prediction'].lower()
        
        label_set = medical_labels if domain == "medical" else industrial_labels
        
        # Check if prediction or similar variant exists
        found = prediction in label_set
        
        if not found:
            # Check for partial matches
            partial_matches = [l for l in label_set if prediction in l or l in prediction]
            if partial_matches:
                found = True
                match_str = f"(partial: {partial_matches[0]})"
            else:
                match_str = "(NOT FOUND - may need to add to DEFAULT_LABELS)"
        else:
            match_str = "(exact match)"
        
        status = "✓" if found else "✗"
        print(f"{status} {domain.upper():12} | {prediction:30} {match_str}")
    
    print("="*80 + "\n")


def show_prediction_summary():
    """Show summary of all reference predictions"""
    data = load_reference_predictions()
    examples = data['examples']
    
    print("\n" + "="*80)
    print(" REFERENCE PREDICTIONS SUMMARY")
    print("="*80)
    
    for idx, ex in enumerate(examples, 1):
        print(f"\n{'─'*80}")
        print(f"Example {idx}: {ex['prediction']}")
        print(f"{'─'*80}")
        print(f"Domain:     {ex['domain']}")
        print(f"Model:      {ex['model_used']}")
        print(f"Confidence: {ex['confidence']:.2%}")
        print(f"\nCaption:")
        print(f"  {ex['caption']}")
        print(f"\nExplanation:")
        print(f"  {ex['explanation']}")
        print(f"\nTop-3 Predictions:")
        for i, pred in enumerate(ex['top_predictions'], 1):
            print(f"  {i}. {pred['label']:40} {pred['score']:.2%}")
    
    print(f"\n{'='*80}\n")


def show_analysis():
    """Display analysis from the dataset"""
    data = load_reference_predictions()
    analysis = data['analysis']
    
    print("\n" + "="*80)
    print(" PREDICTION QUALITY ANALYSIS")
    print("="*80)
    
    print(f"\n📊 Average Confidence:")
    for domain, conf in analysis['average_confidence'].items():
        print(f"   • {domain:25}: {conf:.3f} ({conf*100:.1f}%)")
    
    print(f"\n📈 Confidence Range:")
    print(f"   • Minimum: {analysis['confidence_range']['min']:.2f}")
    print(f"   • Maximum: {analysis['confidence_range']['max']:.2f}")
    
    print(f"\n🔍 Key Observations:")
    for i, obs in enumerate(analysis['key_observations'], 1):
        print(f"   {i}. {obs}")
    
    print(f"\n{'='*80}\n")


def main():
    """Run all demo tests"""
    print("\n" + "="*80)
    print(" 🚀 REFERENCE PREDICTIONS DEMO")
    print("="*80)
    
    # Show summary
    show_prediction_summary()
    
    # Run tests
    test_domain_inference()
    test_label_availability()
    
    # Show analysis
    show_analysis()
    
    print("\n" + "="*80)
    print(" ✅ Demo Complete!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Run visualization: python visualize_reference_predictions.py")
    print("  2. Read analysis report: ANALYSIS_REPORT.md")
    print("  3. Test with real images using the API")
    print("\n")


if __name__ == "__main__":
    main()
