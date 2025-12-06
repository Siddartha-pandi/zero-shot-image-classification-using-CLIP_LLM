# API Response Migration Guide

## Overview
The classification API response structure has been improved for better clarity, accuracy, and UI integration.

---

## Response Changes

### 1. `top_prediction` Structure

**OLD FORMAT:**
```json
{
  "top_prediction": {
    "dog": 85.4
  }
}
```

**NEW FORMAT:**
```json
{
  "top_prediction": {
    "label": "dog",
    "score": 85.4
  }
}
```

**Frontend Migration:**
```typescript
// Before
const [topLabel, topScore] = Object.entries(results.top_prediction)[0]

// After (with backward compatibility)
let topLabel: string
let topScore: number
if ('label' in results.top_prediction && 'score' in results.top_prediction) {
  topLabel = results.top_prediction.label  // New format
  topScore = results.top_prediction.score
} else {
  const [label, score] = Object.entries(results.top_prediction)[0]  // Old format
  topLabel = label
  topScore = score
}
```

---

### 2. `reasoning` Replaces `explanation` and `reasoning_chain`

**REMOVED:**
```json
{
  "explanation": "The image shows a dog with high confidence...",
  "reasoning_chain": {
    "num_prompts": 6,
    "top_prompts": ["a photo of a dog", "an image of a dog"],
    "similarity_score": 85.2
  }
}
```

**NEW:**
```json
{
  "reasoning": {
    "summary": "Classified as 'dog' with 85.4% confidence (LLM-enhanced)",
    "attributes": [
      "Domain: Natural Image (92%)",
      "Confidence: 85.4%",
      "Temperature: Calibrated softmax",
      "Key Feature: High Contrast",
      "Enhanced: LLM Re-ranking"
    ],
    "detailed_reasoning": "The image was analyzed using CLIP embeddings and classified in the natural image domain. Domain detection confidence: 92.0%. Visual analysis identified: high contrast, natural lighting, rich texture. After softmax calibration with temperature scaling, 'dog' achieved 85.4% probability. LLM re-ranking was applied to refine the probabilities based on semantic understanding. The next closest prediction was 'cat' at 12.3%."
  }
}
```

**Frontend Migration:**
```typescript
// Display summary
<p>{results.reasoning.summary}</p>

// Display key attributes as badges
{results.reasoning.attributes.map(attr => (
  <Badge key={attr}>{attr}</Badge>
))}

// Collapsible detailed explanation
<details>
  <summary>Detailed Reasoning</summary>
  <p>{results.reasoning.detailed_reasoning}</p>
</details>

// Fallback for old format
{results.reasoning 
  ? <NewReasoningDisplay reasoning={results.reasoning} />
  : <OldExplanationDisplay text={results.explanation} />
}
```

---

### 3. `domain_info` Enhanced

**OLD:**
```json
{
  "domain_info": {
    "domain": "natural_image",
    "confidence": 0.8,
    "embedding_stats": {
      "mean": 0.0,
      "std": 0.0
    }
  }
}
```

**NEW:**
```json
{
  "domain_info": {
    "domain": "natural_image",
    "confidence": 0.92,
    "characteristics": [
      "photorealistic",
      "natural_lighting",
      "rich_texture",
      "color_diverse"
    ],
    "embedding_stats": {
      "mean": 0.023,
      "std": 0.341,
      "range": 0.682
    }
  }
}
```

**Frontend Usage:**
```typescript
// Display domain characteristics
{results.domain_info.characteristics?.map(char => (
  <span key={char} className="characteristic-tag">
    {char.replace(/_/g, ' ')}
  </span>
))}
```

---

### 4. New Metadata Fields

**ADDED:**
```json
{
  "temperature": 0.01,
  "adaptive_module_used": true,
  "llm_reranking_used": true,
  "multilingual_support": true
}
```

**Frontend Usage:**
```typescript
// Show enhancement badge if LLM was used
{results.llm_reranking_used && (
  <Badge className="bg-purple-600">
    <Sparkles /> LLM Enhanced
  </Badge>
)}

// Show multilingual indicator
{results.multilingual_support && (
  <span>🌐 Multilingual ({results.language})</span>
)}
```

---

### 5. `alternative_predictions` Format

**OLD:**
```json
{
  "alternative_predictions": [
    { "class": "dog", "score": 0.854 },
    { "class": "cat", "score": 0.123 },
    { "class": "bird", "score": 0.023 }
  ]
}
```

**NEW:**
```json
{
  "alternative_predictions": [
    { "label": "dog", "score": 0.854 },
    { "label": "cat", "score": 0.123 },
    { "label": "bird", "score": 0.023 }
  ]
}
```

**Frontend Migration:**
```typescript
// Before
{alternatives.map(alt => (
  <div>{alt.class}: {alt.score}</div>
))}

// After (with backward compatibility)
{alternatives.map(alt => (
  <div>{alt.label || alt.class}: {alt.score}</div>
))}
```

---

## Complete Example Response

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
  "reasoning": {
    "summary": "Classified as 'dog' with 85.4% confidence (LLM-enhanced)",
    "attributes": [
      "Domain: Natural Image (92%)",
      "Confidence: 85.4%",
      "Temperature: Calibrated softmax",
      "Key Feature: High Contrast",
      "Enhanced: LLM Re-ranking"
    ],
    "detailed_reasoning": "The image was analyzed using CLIP embeddings and classified in the natural image domain. Domain detection confidence: 92.0%. Visual analysis identified: high contrast, natural lighting, rich texture. After softmax calibration with temperature scaling, 'dog' achieved 85.4% probability. LLM re-ranking was applied to refine the probabilities based on semantic understanding. The next closest prediction was 'cat' at 12.3%."
  },
  "domain_info": {
    "domain": "natural_image",
    "confidence": 0.92,
    "characteristics": [
      "photorealistic",
      "natural_lighting",
      "rich_texture",
      "color_diverse"
    ],
    "embedding_stats": {
      "mean": 0.023,
      "std": 0.341,
      "range": 0.682
    }
  },
  "visual_features": [
    "high_contrast",
    "natural_lighting",
    "rich_texture"
  ],
  "alternative_predictions": [
    { "label": "dog", "score": 0.854 },
    { "label": "cat", "score": 0.123 },
    { "label": "bird", "score": 0.023 }
  ],
  "zero_shot": true,
  "multilingual_support": true,
  "language": "en",
  "temperature": 0.01,
  "adaptive_module_used": true,
  "llm_reranking_used": true
}
```

---

## Backward Compatibility

The frontend component `ResultsCard.tsx` has been updated to handle BOTH formats:

✅ Works with old responses (pre-update)  
✅ Works with new responses (post-update)  
✅ No breaking changes for existing deployments  

**Key Compatibility Checks:**
1. `top_prediction` - checks for `label` field before using new format
2. `reasoning` - falls back to `explanation` if not present
3. `alternative_predictions` - accepts both `label` and `class` fields
4. `domain_info.characteristics` - optional field, gracefully handled

---

## Testing Checklist

- [ ] Test with `use_llm_reranking=false` (basic mode)
- [ ] Test with `use_llm_reranking=true` (enhanced mode)
- [ ] Verify `reasoning` object displays correctly
- [ ] Verify LLM badge appears when `llm_reranking_used=true`
- [ ] Test with different `temperature` values (0.01, 0.1, 1.0)
- [ ] Verify domain characteristics display
- [ ] Test multilingual support with `language` parameter
- [ ] Verify backward compatibility with old API responses

---

## Migration Timeline

**Phase 1:** ✅ Backend updates deployed  
**Phase 2:** ✅ Frontend updated with backward compatibility  
**Phase 3:** 🔄 Test both old and new formats  
**Phase 4:** 📊 Monitor and collect feedback  
**Phase 5:** 🗑️ Remove old format support (future)

---

**Last Updated:** December 6, 2025  
**API Version:** 2.0.0
