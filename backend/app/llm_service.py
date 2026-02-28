# app/llm_service.py
import os
import json
from typing import Dict, List
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
_api_key = os.getenv("GEMINI_API_KEY")
if _api_key:
    genai.configure(api_key=_api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    model = None


def expand_prompts_with_llm(class_names: List[str], domain: str) -> Dict[str, List[str]]:
    """Optional: use LLM to generate extra prompts. Returns {label: [prompt,...]}"""
    if model is None:
        return {}

    prompt = f"""
You help build prompts for CLIP in a domain-aware way.

Domain: {domain}
Class names: {class_names}

For each class, generate 3 very short descriptive English prompts suitable as image captions.
Respond as JSON: {{"class_name": ["prompt1", "prompt2", "prompt3"], ...}}
"""

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
            )
        )
        return json.loads(response.text)
    except Exception:
        return {}


def llm_reason_and_label(
    caption: str,
    candidates: List[Dict],
    user_hint: str,
    domain: str,
) -> Dict[str, str]:
    """Choose final label + explanation using LLM."""
    if model is None:
        # fallback – just take top candidate
        top = candidates[0]
        return {
            "label": top["label"],
            "reason": "Selected highest-similarity CLIP class (LLM disabled).",
        }

    prompt = f"""
You are an expert visual reasoning assistant.

Image caption: "{caption}"
Domain: {domain}
Top candidate classes with cosine scores: {candidates}
User hint: "{user_hint}"

Choose the most likely label and explain briefly.
Respond ONLY as JSON:
{{
  "label": "<final_label>",
  "reason": "<short explanation>"
}}
"""

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=100,
            )
        )
        
        # Clean response text - remove markdown code blocks if present
        response_text = response.text.strip()
        if response_text.startswith("```"):
            # Remove markdown code blocks
            lines = response_text.split("\n")
            response_text = "\n".join([line for line in lines if not line.startswith("```")])
            response_text = response_text.strip()
        
        # Try to parse JSON
        parsed = json.loads(response_text)
        return parsed
        
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Response text: {response.text}")
        top = candidates[0]
        return {
            "label": top["label"],
            "reason": "Fallback to top CLIP candidate due to JSON parse error.",
        }
    except Exception as e:
        print(f"LLM error: {e}")
        top = candidates[0]
        return {
            "label": top["label"],
            "reason": "Fallback to top CLIP candidate due to LLM error.",
        }


def llm_narrative(
    caption: str,
    candidates: List[Dict],
    user_hint: str,
    domain: str,
) -> str:
    """Generate detailed narrative description."""
    if model is None:
        return caption  # fallback: just caption

    prompt = f"""
You are an expert image analyst writing detailed, descriptive narratives about images.

Caption: "{caption}"
Domain: {domain}
Detected classes with confidence: {candidates}
User hint: "{user_hint}"

Write a comprehensive, detailed 8-12 sentence narrative describing the image. Include:
1. Main subjects or objects present and their characteristics
2. Visual details: colors, textures, patterns, shapes, and materials
3. The setting, environment, and background elements
4. Spatial relationships and composition (foreground, middle, background)
5. Lighting, atmosphere, and mood
6. Actions, poses, or states of subjects
7. Notable features, unique aspects, or interesting details
8. Overall impression and context

Be highly descriptive, vivid, and engaging while remaining factual. Use rich, varied vocabulary to paint a complete picture. Do NOT invent details that aren't supported by the detected classes and caption.
Return ONLY the narrative text as a flowing paragraph.
"""

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.5,
                max_output_tokens=500,
            )
        )
        return response.text.strip()
    except Exception:
        return caption  # fallback on error


def extract_objects(
    caption: str,
    candidates: List[Dict],
    domain: str,
) -> List[Dict]:
    """Extract a list of objects/entities present in the image with confidence scores."""
    if model is None:
        # Fallback: use top candidates as objects
        return [
            {"name": c["label"], "score": round(c.get("score", 0), 2)}
            for c in candidates[:5] if c.get("score", 0) > 0.1
        ]
    
    prompt = f"""
You are an expert visual object detection specialist.

Image caption: "{caption}"
Domain: {domain}
Detected classes with scores: {candidates}

Identify and list the main objects, entities, and significant visual elements present in the image.
For each object, estimate a confidence score between 0.0 and 1.0 based on how clearly identifiable it is in the image.

Respond ONLY as a JSON array of objects with "name" and "score" fields:
[
  {{"name": "object1", "score": 0.95}},
  {{"name": "object2", "score": 0.88}},
  {{"name": "object3", "score": 0.82}}
]

Rules:
- Use 3-8 objects
- Use simple noun phrases for object names
- Scores should reflect confidence (higher for clear/prominent objects)
- For medical images, focus on anatomical structures and abnormalities
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
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join([line for line in lines if not line.startswith("```")])
            response_text = response_text.strip()
        
        parsed = json.loads(response_text)
        if isinstance(parsed, list) and len(parsed) > 0:
            # Validate and clean objects
            objects = []
            for obj in parsed[:10]:  # Max 10 objects
                if isinstance(obj, dict) and "name" in obj and "score" in obj:
                    objects.append({
                        "name": str(obj["name"]).strip(),
                        "score": round(float(obj["score"]), 2)
                    })
            return objects if objects else [
                {"name": c["label"], "score": round(c.get("score", 0), 2)}
                for c in candidates[:5] if c.get("score", 0) > 0.1
            ]
        else:
            # Fallback to candidates
            return [
                {"name": c["label"], "score": round(c.get("score", 0), 2)}
                for c in candidates[:5] if c.get("score", 0) > 0.1
            ]
    except (json.JSONDecodeError, Exception) as e:
        print(f"Error extracting objects: {e}")
        # Fallback to candidates
        return [
            {"name": c["label"], "score": round(c.get("score", 0), 2)}
            for c in candidates[:5] if c.get("score", 0) > 0.1
        ]
