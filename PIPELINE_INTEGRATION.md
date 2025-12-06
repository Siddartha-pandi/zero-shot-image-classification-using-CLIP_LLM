# Reasoning Pipeline Integration Example

## Overview

The `reasoning_pipeline.py` module provides a clean, production-ready pipeline for zero-shot image classification with the following stages:

1. **CLIP Stage**: Image encoding with optional adaptive modules
2. **Domain Stage**: Automatic domain detection
3. **Calibration Stage**: Temperature-scaled softmax
4. **Caption + Features**: Image description generation
5. **LLM Re-ranking**: Semantic probability refinement
6. **LLM Explanation**: UI-friendly reasoning generation

---

## Quick Integration

### Option 1: Use in Existing API (Minimal Changes)

```python
# backend/main.py
from reasoning_pipeline import run_reasoning_pipeline

@app.post("/api/classify")
async def classify_image_endpoint(
    file: UploadFile = File(...),
    labels: str = None,
    temperature: float = 0.01,
    use_llm_reranking: bool = False
):
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name
    
    try:
        # Load image
        image = Image.open(temp_path).convert("RGB")
        
        # Parse labels
        if labels:
            class_names = [label.strip() for label in labels.split(',')]
        else:
            class_names = ["dog", "cat", "bird", "car", "person"]  # Default
        
        # Get text embeddings for labels
        text_inputs = clip_processor(text=class_names, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_embs = clip_model.get_text_features(**text_inputs)
            text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
        
        # Run pipeline
        result = run_reasoning_pipeline(
            image=image,
            label_names=class_names,
            text_embs=text_embs,
            clip_model=clip_model,
            clip_preprocess=clip_processor,
            adaptive_module=None,  # Or your adaptive module
            llm_client=llm_client,  # Or None for no LLM
            temperature=temperature,
            top_k=5,
            use_llm_reranking=use_llm_reranking,
            device=str(device)
        )
        
        return result
    
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
```

---

### Option 2: Standalone Service

```python
# backend/services/classification_service.py
from reasoning_pipeline import run_reasoning_pipeline
from models import get_model_manager
from PIL import Image

class ClassificationService:
    def __init__(self):
        self.model_manager = get_model_manager()
        self.clip_model = self.model_manager.clip_model
        self.clip_processor = self.model_manager.clip_processor
        self.device = self.model_manager.device
        self.llm_client = None  # Set to your LLM client
    
    def classify_image(
        self,
        image_path: str,
        labels: list,
        temperature: float = 0.01,
        use_llm: bool = False
    ):
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Encode text labels
        text_inputs = self.clip_processor(
            text=labels,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            text_embs = self.clip_model.get_text_features(**text_inputs)
            text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
        
        # Run pipeline
        return run_reasoning_pipeline(
            image=image,
            label_names=labels,
            text_embs=text_embs,
            clip_model=self.clip_model,
            clip_preprocess=self.clip_processor,
            adaptive_module=None,
            llm_client=self.llm_client if use_llm else None,
            temperature=temperature,
            use_llm_reranking=use_llm,
            device=str(self.device)
        )

# Usage
service = ClassificationService()
result = service.classify_image(
    "image.jpg",
    ["dog", "cat", "bird"],
    use_llm=True
)
```

---

## LLM Client Interface

The pipeline expects an LLM client with a `generate()` method:

```python
class LLMClient:
    """Simple LLM client interface"""
    
    def generate(self, prompt: str) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
        
        Returns:
            Generated text
        """
        raise NotImplementedError

# Example with OpenAI
class OpenAIClient(LLMClient):
    def __init__(self, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
    
    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )
        return response.choices[0].message.content

# Example with local model
class LocalLLMClient(LLMClient):
    def __init__(self, model_path: str):
        from transformers import pipeline
        self.generator = pipeline("text-generation", model=model_path)
    
    def generate(self, prompt: str) -> str:
        result = self.generator(prompt, max_length=200, num_return_sequences=1)
        return result[0]['generated_text']
```

---

## Response Format

The pipeline returns a clean JSON structure:

```json
{
  "predictions": {
    "dog": 85.4,
    "cat": 12.3,
    "bird": 2.3
  },
  "top_prediction": {
    "label": "dog",
    "score": 85.4
  },
  "confidence_score": 0.854,
  "domain_info": {
    "domain": "natural_image",
    "confidence": 0.92,
    "characteristics": ["photorealistic", "natural_lighting"],
    "embedding_stats": {
      "mean": 0.023,
      "std": 0.341,
      "range": 0.682
    }
  },
  "reasoning": {
    "summary": "Classified as 'dog' with 85.4% confidence",
    "attributes": [
      "Dog",
      "Cat",
      "Animal",
      "Photorealistic",
      "Natural Lighting"
    ],
    "detailed_reasoning": "The image shows a photorealistic animal in natural lighting. Based on visual analysis in the natural image domain, it was classified as 'dog' with 85.4% confidence using zero-shot CLIP embeddings and LLM enhancement."
  },
  "visual_features": ["Dog", "Cat", "Animal", "Photorealistic"],
  "alternative_predictions": [
    {"label": "dog", "score": 85.4},
    {"label": "cat", "score": 12.3},
    {"label": "bird", "score": 2.3}
  ],
  "zero_shot": true,
  "multilingual_support": true,
  "language": "en",
  "temperature": 0.01,
  "adaptive_module_used": false,
  "llm_reranking_used": true
}
```

---

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | float | 0.01 | Softmax temperature (0.01-1.0) |
| `top_k` | int | 5 | Number of top predictions |
| `use_llm_reranking` | bool | True | Enable LLM re-ranking |
| `adaptive_module` | Module | None | Domain-adaptive transformation |
| `llm_client` | LLMClient | None | LLM for enhancement |

---

## Performance Tips

### 1. Cache Text Embeddings

```python
# Pre-compute and cache text embeddings
class CachedClassifier:
    def __init__(self):
        self.text_emb_cache = {}
    
    def get_text_embeddings(self, labels):
        key = tuple(sorted(labels))
        if key not in self.text_emb_cache:
            # Compute and cache
            text_inputs = clip_processor(text=labels, ...)
            self.text_emb_cache[key] = clip_model.get_text_features(**text_inputs)
        return self.text_emb_cache[key]
```

### 2. Batch Processing

```python
# Process multiple images
def classify_batch(images: List[Image.Image], labels: List[str]):
    # Encode all images at once
    image_inputs = torch.stack([clip_preprocess(img) for img in images])
    with torch.no_grad():
        image_embs = clip_model.encode_image(image_inputs)
    
    # Process each image
    results = []
    for i, img_emb in enumerate(image_embs):
        result = run_reasoning_pipeline(...)
        results.append(result)
    
    return results
```

### 3. Async LLM Calls

```python
# Use async LLM client for better throughput
import asyncio

class AsyncLLMClient:
    async def generate_async(self, prompt: str) -> str:
        # Async LLM call
        ...

# In pipeline, run LLM calls concurrently
async def run_pipeline_async(...):
    caption_task = llm_client.generate_async(caption_prompt)
    rerank_task = llm_client.generate_async(rerank_prompt)
    reason_task = llm_client.generate_async(reason_prompt)
    
    caption, reranked, reasoning = await asyncio.gather(
        caption_task, rerank_task, reason_task
    )
```

---

## Testing

```python
# test_pipeline.py
from reasoning_pipeline import run_reasoning_pipeline
from PIL import Image
import torch

# Mock CLIP model
class MockCLIP:
    def encode_image(self, x):
        return torch.randn(1, 512)

# Test basic pipeline
def test_basic_classification():
    image = Image.new('RGB', (224, 224))
    labels = ["dog", "cat", "bird"]
    text_embs = torch.randn(3, 512)
    
    result = run_reasoning_pipeline(
        image=image,
        label_names=labels,
        text_embs=text_embs,
        clip_model=MockCLIP(),
        clip_preprocess=lambda x: torch.randn(3, 224, 224),
        temperature=0.01
    )
    
    assert "top_prediction" in result
    assert "reasoning" in result
    assert result["llm_reranking_used"] == False  # No LLM client

# Test with LLM
def test_with_llm():
    # Use mock LLM client
    class MockLLM:
        def generate(self, prompt):
            if "caption" in prompt:
                return "A photo of a dog"
            elif "re-rank" in prompt:
                return '[{"label":"dog","prob":0.9}]'
            else:
                return '{"summary":"test","attributes":[],"detailed_reasoning":"test"}'
    
    result = run_reasoning_pipeline(..., llm_client=MockLLM())
    assert result["llm_reranking_used"] == True
```

---

## Migration from Existing Code

If you're using `advanced_inference.py`, you can gradually migrate:

```python
# Step 1: Add pipeline as alternative
from reasoning_pipeline import run_reasoning_pipeline

def classify(image_path, labels, use_new_pipeline=False):
    if use_new_pipeline:
        return run_reasoning_pipeline(...)
    else:
        return old_classifier.classify(...)

# Step 2: Test with both
result_old = classify(..., use_new_pipeline=False)
result_new = classify(..., use_new_pipeline=True)

# Step 3: Switch to new pipeline when ready
```

---

## Troubleshooting

### Issue: LLM returns invalid JSON

**Solution:** The pipeline has fallback handling. Check logs for warnings.

```python
# Add better JSON extraction
def extract_json(text):
    # Try to find JSON in text
    start = text.find('{')
    end = text.rfind('}') + 1
    return text[start:end] if start >= 0 else text
```

### Issue: Probabilities don't sum to 1

**Solution:** Pipeline auto-normalizes. Verify with:

```python
assert abs(sum(result["predictions"].values()) - 100.0) < 0.01
```

### Issue: Domain detection is inaccurate

**Solution:** Replace heuristic with trained classifier:

```python
def detect_domain(image_emb):
    # Use trained domain classifier
    logits = domain_classifier(image_emb)
    domain_idx = logits.argmax()
    return DOMAIN_NAMES[domain_idx]
```

---

**Date:** December 6, 2025  
**Pipeline Version:** 1.0.0  
**Status:** Production Ready ✅
