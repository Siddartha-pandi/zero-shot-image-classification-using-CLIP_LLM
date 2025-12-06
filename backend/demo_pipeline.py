"""
Demo script for the reasoning pipeline.

This shows how to use the reasoning_pipeline module with minimal setup.
Run this to test the pipeline with a sample image.

Usage:
    python demo_pipeline.py
"""

import torch
import numpy as np
from PIL import Image
from reasoning_pipeline import run_reasoning_pipeline


# ========== Mock Components for Testing ==========

class MockCLIPModel:
    """Mock CLIP model for testing without loading actual weights"""
    
    def __init__(self, dim=512):
        self.dim = dim
    
    def encode_image(self, image_tensor):
        """Return random embeddings"""
        batch_size = image_tensor.shape[0]
        return torch.randn(batch_size, self.dim)
    
    def parameters(self):
        """For device detection"""
        return [torch.tensor([0.0])]


class MockCLIPPreprocess:
    """Mock CLIP preprocessor"""
    
    def __call__(self, image):
        """Convert PIL image to tensor"""
        return torch.randn(3, 224, 224)


class MockLLMClient:
    """Mock LLM client for testing without API calls"""
    
    def generate(self, prompt: str) -> str:
        """Generate mock responses based on prompt type"""
        
        # Caption generation
        if "caption" in prompt.lower():
            return "A photorealistic image showing a golden retriever dog sitting outdoors"
        
        # Re-ranking
        elif "re-rank" in prompt.lower():
            return '''[
                {"label": "dog", "prob": 0.82},
                {"label": "golden retriever", "prob": 0.15},
                {"label": "animal", "prob": 0.03}
            ]'''
        
        # Reasoning generation
        else:
            return '''{
                "summary": "Classified as 'dog' with 82% confidence using CLIP and LLM enhancement",
                "attributes": [
                    "Domain: Natural Image (95%)",
                    "Confidence: 82%",
                    "Key Feature: Photorealistic",
                    "Enhanced: LLM Re-ranking"
                ],
                "detailed_reasoning": "The image shows a photorealistic animal in natural lighting conditions. Based on visual embedding analysis and semantic understanding, the model identified characteristic features of a dog, including fur texture, facial structure, and posture. The LLM re-ranking confirmed this classification with high confidence, distinguishing it from similar animal categories."
            }'''


# ========== Demo Function ==========

def demo_basic_pipeline():
    """Demonstrate basic pipeline without LLM"""
    
    print("=" * 60)
    print("DEMO 1: Basic Pipeline (CLIP only, no LLM)")
    print("=" * 60)
    
    # Create sample image
    image = Image.new('RGB', (224, 224), color=(100, 150, 200))
    
    # Define labels
    labels = ["dog", "cat", "bird", "car", "person", "tree", "building"]
    
    # Create mock components
    clip_model = MockCLIPModel()
    clip_preprocess = MockCLIPPreprocess()
    
    # Create text embeddings (random for demo)
    text_embs = torch.randn(len(labels), 512)
    text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
    
    # Run pipeline WITHOUT LLM
    result = run_reasoning_pipeline(
        image=image,
        label_names=labels,
        text_embs=text_embs,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        adaptive_module=None,
        llm_client=None,  # No LLM
        temperature=0.01,
        top_k=5,
        use_llm_reranking=False
    )
    
    # Display results
    print(f"\nTop Prediction: {result['top_prediction']['label']}")
    print(f"Confidence: {result['top_prediction']['score']:.2f}%")
    print(f"\nDomain: {result['domain_info']['domain']}")
    print(f"Domain Confidence: {result['domain_info']['confidence']:.1%}")
    print(f"\nAll Predictions:")
    for label, score in result['predictions'].items():
        print(f"  {label}: {score:.2f}%")
    
    print(f"\nReasoning Summary:")
    print(f"  {result['reasoning']['summary']}")
    
    print(f"\nMetadata:")
    print(f"  Temperature: {result['temperature']}")
    print(f"  LLM Reranking: {result['llm_reranking_used']}")
    print(f"  Adaptive Module: {result['adaptive_module_used']}")
    
    return result


def demo_llm_enhanced_pipeline():
    """Demonstrate pipeline with LLM enhancement"""
    
    print("\n" + "=" * 60)
    print("DEMO 2: LLM-Enhanced Pipeline")
    print("=" * 60)
    
    # Create sample image
    image = Image.new('RGB', (224, 224), color=(200, 180, 100))
    
    # Define labels
    labels = ["dog", "golden retriever", "cat", "animal", "pet"]
    
    # Create mock components
    clip_model = MockCLIPModel()
    clip_preprocess = MockCLIPPreprocess()
    llm_client = MockLLMClient()
    
    # Create text embeddings
    text_embs = torch.randn(len(labels), 512)
    text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
    
    # Run pipeline WITH LLM
    result = run_reasoning_pipeline(
        image=image,
        label_names=labels,
        text_embs=text_embs,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        adaptive_module=None,
        llm_client=llm_client,  # With LLM
        temperature=0.01,
        top_k=5,
        use_llm_reranking=True
    )
    
    # Display results
    print(f"\nTop Prediction: {result['top_prediction']['label']}")
    print(f"Confidence: {result['top_prediction']['score']:.2f}%")
    
    print(f"\nReasoning:")
    print(f"  Summary: {result['reasoning']['summary']}")
    print(f"\n  Key Attributes:")
    for attr in result['reasoning']['attributes']:
        print(f"    • {attr}")
    print(f"\n  Detailed Explanation:")
    print(f"    {result['reasoning']['detailed_reasoning']}")
    
    print(f"\nVisual Features:")
    for feat in result['visual_features']:
        print(f"  • {feat}")
    
    print(f"\nAlternative Predictions:")
    for pred in result['alternative_predictions'][:3]:
        print(f"  {pred['label']}: {pred['score']:.2f}%")
    
    print(f"\nMetadata:")
    print(f"  LLM Reranking: {result['llm_reranking_used']}")
    print(f"  Domain: {result['domain_info']['domain']}")
    
    return result


def demo_temperature_comparison():
    """Show effect of temperature on probabilities"""
    
    print("\n" + "=" * 60)
    print("DEMO 3: Temperature Scaling Comparison")
    print("=" * 60)
    
    image = Image.new('RGB', (224, 224))
    labels = ["dog", "cat", "bird"]
    
    clip_model = MockCLIPModel()
    clip_preprocess = MockCLIPPreprocess()
    text_embs = torch.randn(len(labels), 512)
    text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
    
    temperatures = [0.01, 0.1, 1.0]
    
    for temp in temperatures:
        result = run_reasoning_pipeline(
            image=image,
            label_names=labels,
            text_embs=text_embs,
            clip_model=clip_model,
            clip_preprocess=clip_preprocess,
            temperature=temp,
            use_llm_reranking=False
        )
        
        print(f"\nTemperature = {temp}")
        print(f"  Top: {result['top_prediction']['label']} ({result['top_prediction']['score']:.2f}%)")
        print(f"  Distribution: {', '.join([f'{s:.1f}%' for s in result['predictions'].values()])}")


def compare_with_without_llm():
    """Compare results with and without LLM"""
    
    print("\n" + "=" * 60)
    print("DEMO 4: CLIP vs CLIP+LLM Comparison")
    print("=" * 60)
    
    image = Image.new('RGB', (224, 224))
    labels = ["dog", "cat", "wolf", "fox", "coyote"]
    
    clip_model = MockCLIPModel()
    clip_preprocess = MockCLIPPreprocess()
    llm_client = MockLLMClient()
    text_embs = torch.randn(len(labels), 512)
    text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
    
    # Without LLM
    result_clip = run_reasoning_pipeline(
        image=image,
        label_names=labels,
        text_embs=text_embs,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        llm_client=None,
        use_llm_reranking=False
    )
    
    # With LLM
    result_llm = run_reasoning_pipeline(
        image=image,
        label_names=labels,
        text_embs=text_embs,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        llm_client=llm_client,
        use_llm_reranking=True
    )
    
    print("\nCLIP Only:")
    print(f"  Top: {result_clip['top_prediction']['label']} ({result_clip['top_prediction']['score']:.2f}%)")
    print(f"  Reasoning: {result_clip['reasoning']['summary']}")
    
    print("\nCLIP + LLM:")
    print(f"  Top: {result_llm['top_prediction']['label']} ({result_llm['top_prediction']['score']:.2f}%)")
    print(f"  Reasoning: {result_llm['reasoning']['summary']}")
    
    print("\nDifference:")
    print(f"  LLM Re-ranking Applied: {result_llm['llm_reranking_used']}")
    print(f"  Enhanced Reasoning: {len(result_llm['reasoning']['detailed_reasoning'])} chars")


# ========== Main ==========

if __name__ == "__main__":
    print("\n🚀 Reasoning Pipeline Demo\n")
    
    try:
        # Run all demos
        demo_basic_pipeline()
        demo_llm_enhanced_pipeline()
        demo_temperature_comparison()
        compare_with_without_llm()
        
        print("\n" + "=" * 60)
        print("✅ All demos completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Replace MockCLIPModel with actual CLIP model")
        print("2. Replace MockLLMClient with real LLM (OpenAI, local, etc.)")
        print("3. Load real images for classification")
        print("4. Integrate into your FastAPI backend")
        print("5. Test with various domains (sketches, medical, anime, etc.)")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
