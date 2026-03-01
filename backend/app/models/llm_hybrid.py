# app/models/llm_hybrid.py
"""
LLM Service for Hybrid System
Generates explanations and captions for multi-model predictions
"""
import os
import json
from typing import Dict, List, Optional
from dotenv import load_dotenv
import google.generativeai as genai
import logging

logger = logging.getLogger(__name__)

load_dotenv()
_api_key = os.getenv("GEMINI_API_KEY")
if _api_key:
    genai.configure(api_key=_api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    model = None
    logger.warning("GEMINI_API_KEY not found - LLM features will be limited")


def generate_hybrid_explanation(
    domain: str,
    model_used: str,
    prediction: str,
    confidence: float,
    caption: str,
    top_matches: List[Dict],
    domain_scores: Dict[str, float]
) -> Dict[str, str]:
    """
    Generate comprehensive explanation for hybrid model prediction
    
    Args:
        domain: Detected domain
        model_used: Model name (ViT-H/14 or MedCLIP)
        prediction: Top prediction label
        confidence: Confidence score
        caption: Image caption
        top_matches: Top prediction matches
        domain_scores: Domain similarity scores
        
    Returns:
        Dict with explanation, caption, and risk_notes
    """
    if model is None:
        return {
            "explanation": f"Classified as '{prediction}' with {confidence:.1%} confidence using {model_used}.",
            "caption": caption,
            "risk_notes": "LLM unavailable for detailed analysis"
        }
    
    # Build structured prompt
    prompt = f"""
You are an advanced AI image analysis system using a hybrid ViT-H/14 + MedCLIP architecture.

🔍 ANALYSIS CONTEXT:
Domain: {domain}
Model Used: {model_used}
Domain Confidence Scores: {json.dumps(domain_scores, indent=2)}

📊 CLASSIFICATION RESULTS:
Top Prediction: {prediction}
Confidence Score: {confidence:.2%}
Image Caption: "{caption}"

Top Matches:
{json.dumps(top_matches[:5], indent=2)}

🎯 TASK:
Generate a comprehensive analysis with:

1. **Short Caption** (1-2 sentences):
   - Describe what the image shows
   - Include key visual elements

2. **Classification Explanation** (2-3 sentences):
   - Why this classification was made
   - What visual features support it
   - How the model made this decision

3. **Risk Notes** (medical images only):
   - Clinical significance
   - Recommended follow-up actions
   - Disclaimer about AI limitations

For non-medical images, set risk_notes to "N/A - not a medical image"

Respond ONLY as JSON:
{{
  "caption": "...",
  "explanation": "...",
  "risk_notes": "..."
}}
"""
    
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=400,
            )
        )
        
        # Clean and parse response
        response_text = response.text.strip()
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join([line for line in lines if not line.startswith("```")])
            response_text = response_text.strip()
        
        parsed = json.loads(response_text)
        
        return {
            "caption": parsed.get("caption", caption),
            "explanation": parsed.get("explanation", ""),
            "risk_notes": parsed.get("risk_notes", "")
        }
        
    except Exception as e:
        logger.error(f"LLM explanation error: {e}")
        return {
            "caption": caption,
            "explanation": f"Classified as '{prediction}' using {model_used} with {confidence:.1%} confidence in {domain} domain.",
            "risk_notes": "Error generating detailed analysis"
        }


def generate_detailed_narrative(
    domain: str,
    model_used: str,
    prediction: str,
    caption: str,
    top_matches: List[Dict]
) -> str:
    """
    Generate detailed narrative description
    
    Args:
        domain: Image domain
        model_used: Model name
        prediction: Top prediction
        caption: Image caption
        top_matches: Top predictions
        
    Returns:
        Detailed narrative text
    """
    if model is None:
        return caption
    
    prompt = f"""
You are an expert image analyst. Write a detailed, vivid narrative description.

Domain: {domain}
Classification: {prediction}
Caption: "{caption}"
Model: {model_used}

Alternative interpretations: {[m['label'] for m in top_matches[:3]]}

Write a flowing 6-8 sentence narrative that:
- Describes the image comprehensively
- Explains visual characteristics
- Discusses composition and key elements
- Mentions lighting, colors, textures (where applicable)
- Contextualizes the classification
- For medical images: describes anatomical structures and findings
- For other domains: describes scene, objects, and context

Return ONLY the narrative text as a flowing paragraph.
"""
    
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.4,
                max_output_tokens=350,
            )
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Narrative generation error: {e}")
        return caption


def extract_objects_hybrid(
    domain: str,
    caption: str,
    top_matches: List[Dict]
) -> List[Dict]:
    """
    Extract objects/entities from image analysis
    
    Args:
        domain: Image domain
        caption: Image caption
        top_matches: Top classification matches
        
    Returns:
        List of objects with confidence scores
    """
    if model is None:
        # Fallback: use top matches
        return [
            {"name": m["label"], "score": round(m["score"], 2)}
            for m in top_matches[:5]
        ]
    
    prompt = f"""
Extract key objects/entities from this image analysis.

Domain: {domain}
Caption: "{caption}"
Detected classes: {[m['label'] for m in top_matches]}

List 3-8 main objects/entities with confidence scores (0.0-1.0).
For medical images: anatomical structures and abnormalities
For other images: main objects and scene elements

Respond ONLY as JSON array:
[
  {{"name": "object1", "score": 0.95}},
  {{"name": "object2", "score": 0.88}}
]
"""
    
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=200,
            )
        )
        
        response_text = response.text.strip()
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join([line for line in lines if not line.startswith("```")])
            response_text = response_text.strip()
        
        parsed = json.loads(response_text)
        
        if isinstance(parsed, list):
            return [
                {"name": str(obj.get("name", "")), "score": round(float(obj.get("score", 0)), 2)}
                for obj in parsed[:8]
                if "name" in obj and "score" in obj
            ]
    except Exception as e:
        logger.error(f"Object extraction error: {e}")
    
    # Fallback
    return [
        {"name": m["label"], "score": round(m["score"], 2)}
        for m in top_matches[:5]
    ]
