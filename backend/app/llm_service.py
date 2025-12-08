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
