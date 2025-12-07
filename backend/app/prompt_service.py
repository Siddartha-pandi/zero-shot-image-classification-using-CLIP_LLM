# app/prompt_service.py
from typing import List
from .llm_service import expand_prompts_with_llm

TEMPLATES = {
    "natural": [
        "a photo of a {}",
        "an image of a {}",
        "a realistic picture of a {}",
    ],
    "medical": [
        "a medical X-ray image showing {}",
        "a radiology scan of {}",
        "a grayscale chest X-ray depicting {}",
    ],
    "anime": [
        "an anime-style illustration of a {}",
        "a colorful cartoon drawing of a {}",
    ],
    "sketch": [
        "a black and white line drawing of a {}",
        "a pencil sketch of a {}",
    ],
    "satellite": [
        "a top-down satellite image of {}",
        "an aerial photograph of {}",
    ],
    "unknown": [
        "an image of a {}",
    ],
}

def build_prompts_for_label(label: str, domain: str) -> List[str]:
    base = [t.format(label) for t in TEMPLATES.get(domain, TEMPLATES["unknown"])]
    # optional: LLM expansion (if no API key, this returns {})
    extra = expand_prompts_with_llm([label], domain).get(label, [])
    # de-duplicate while preserving order
    seen = set()
    all_prompts: List[str] = []
    for p in base + extra:
        if p not in seen:
            seen.add(p)
            all_prompts.append(p)
    return all_prompts
