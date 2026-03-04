# backend/services/explanation_generator.py
import logging
from typing import List, Dict
from models.llm_model import get_llm_model

logger = logging.getLogger(__name__)

def generate_explanation(
    domain: str,
    model_used: str,
    prediction: str,
    confidence: float,
    caption: str,
    top_matches: List[Dict]
) -> str:
    """Generate comprehensive explanation for hybrid model prediction using LLM"""
    confidence_pct = int(confidence * 100)
    
    prompt = f"""You are an expert image analyst explaining a model's classification.

DATA:
- Domain: {domain}
- Prediction: {prediction}
- Confidence Score: {confidence:.2f} ({confidence_pct}%)
- Image Caption: "{caption}"
- Model Used: {model_used}

TASK:
Write a single, cohesive paragraph explaining this classification.
It MUST be at least 70 words long and read professionally.
It MUST follow this narrative structure:
1. Describe what the image shows and key visual characteristics observed (use caption as primary reference).
2. Detail specific features, physical traits, or distinctive markers that identify the subject.
3. Explain the context or behavior/habitat where this subject is typically found.
4. Conclude by connecting these visual features to the classification, stating how confidently the system classifies it.

EXAMPLE FORMAT TO MIMIC (adapt this tone and structure to the actual image):
"The image shows a tall bird with distinctive pink and orange feathers, extremely long slender legs, and a curved neck. The beak has a characteristic downward bend with a dark tip, which is typical of flamingos. The bird is standing in shallow water, a common habitat for flamingos where they feed on small aquatic organisms. These visual features match well-known characteristics of flamingo species, allowing the system to confidently classify the image as a flamingo within the animal domain."

Respond ONLY with the explanation text, nothing else.
"""
    try:
        llm = get_llm_model()
        return llm.generate(prompt=prompt, temperature=0.3, max_tokens=350)
    except Exception as e:
        logger.error(f"LLM explanation error: {e}")
        # Use caption as basis for fallback explanation
        return f"The image shows {caption.lower()} The defining characteristics including distinctive coloring, physical structure, and proportions are typical of {prediction.lower()} species. These visual features and markers directly align with well-known characteristics of {prediction.lower()} within the {domain.lower()} domain. The subject exhibits behaviors and is positioned in contextual environments where {prediction.lower()} are commonly found. These visual patterns and structural elements match established {domain.lower()} categories, allowing the {model_used} model to confidently classify the image as a {prediction.lower()} with {confidence_pct}% confidence."
