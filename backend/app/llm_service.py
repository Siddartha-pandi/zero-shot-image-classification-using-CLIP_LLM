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
                temperature=0.2,
            )
        )
        return json.loads(response.text)
    except Exception:
        top = candidates[0]
        return {
            "label": top["label"],
            "reason": "Fallback to top CLIP candidate due to JSON parse error.",
        }


def llm_narrative(
    caption: str,
    candidates: List[Dict],
    user_hint: str,
    domain: str,
) -> str:
    """Generate 3–5 sentence narrative description."""
    if model is None:
        return caption  # fallback: just caption

    prompt = f"""
You write descriptive but factual narratives about images.

Caption: "{caption}"
Domain: {domain}
Likely classes: {candidates}
User hint: "{user_hint}"

Write a 3–5 sentence narrative describing the image.
Do NOT invent impossible details.
Return ONLY the narrative text.
"""

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.5,
            )
        )
        return response.text.strip()
    except Exception:
        return caption  # fallback on error
