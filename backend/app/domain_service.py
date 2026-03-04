# app/domain_service.py
from typing import Literal, Optional

Domain = Literal["natural", "medical", "industrial", "anime", "sketch", "satellite", "unknown"]

KEYWORDS = {
    "medical": [
        # Imaging modalities
        "xray", "x-ray", "ct", "mri", "scan", "radiograph", "ultrasound",
        # Anatomical regions
        "chest", "lung", "lungs", "brain", "heart", "abdomen", "retina", "retinal",
        "rib", "ribs", "spine", "vertebra", "organ", "tissue",
        # Medical findings
        "opacity", "pulmonary", "cardiac", "bone", "fracture", "lesion", "tumor",
        "pneumonia", "nodule", "mass", "edema",
        # Medical terminology
        "medical", "clinical", "diagnostic", "anatomy", "radiology", "imaging",
        "patient", "thorax", "skull", "fundus", "optic disc",
        # Specific image types
        "chest x-ray", "brain mri", "ct scan", "skin lesion", "dermatological",
        "fundus photograph", "melanoma", "retinopathy"
    ],
    "industrial": [
        # Defect types
        "crack", "fracture", "corrosion", "rust", "scratch", "wear", "defect",
        # Materials
        "metal", "steel", "surface", "metallic", "industrial",
        # Inspection terms
        "inspection", "quality control", "damage", "flaw", "deterioration",
        # Specific defects
        "surface crack", "oxidation", "abrasion", "structural failure",
        "surface wear", "manufacturing defect"
    ],
    "anime": ["anime", "manga", "cartoon", "illustration", "character", "animated"],
    "satellite": ["satellite", "aerial", "top view", "remote sensing", "overhead", "bird's eye"],
    "sketch": ["sketch", "line art", "line-art", "drawing", "pencil", "hand-drawn"],
}

def infer_domain_from_hint(hint: Optional[str]) -> Domain:
    """Infer domain from user hint text."""
    if not hint:
        return "unknown"
    h = hint.lower()
    for dom, words in KEYWORDS.items():
        if any(w in h for w in words):
            return dom  # type: ignore
    return "natural"

def infer_domain_from_caption(caption: str) -> Domain:
    """Infer domain from image caption."""
    if not caption:
        return "unknown"
    c = caption.lower()
    
    # Check for medical imaging indicators
    medical_indicators = [
        "chest", "lung", "lungs", "xray", "x-ray", "rib", "ribs",
        "medical", "radiograph", "scan", "ct", "mri", "opacity",
        "pulmonary", "cardiac", "bone", "anatomy", "radiology",
        "brain", "retina", "retinal", "fundus", "lesion", "tumor"
    ]
    if any(word in c for word in medical_indicators):
        return "medical"
    
    # Check for industrial inspection indicators
    industrial_indicators = [
        "crack", "fracture", "corrosion", "rust", "scratch", "defect",
        "metal", "surface", "industrial", "damage", "wear", "oxidation",
        "abrasion", "structural", "manufacturing"
    ]
    if any(word in c for word in industrial_indicators):
        return "industrial"
    
    # Check for anime/cartoon indicators  
    anime_indicators = ["anime", "manga", "cartoon", "animated", "character"]
    if any(word in c for word in anime_indicators):
        return "anime"
    
    # Check for sketch/drawing indicators
    sketch_indicators = ["sketch", "drawing", "line art", "pencil", "hand-drawn"]
    if any(word in c for word in sketch_indicators):
        return "sketch"
    
    # Check for satellite/aerial indicators
    satellite_indicators = ["satellite", "aerial", "overhead", "bird's eye", "top view"]
    if any(word in c for word in satellite_indicators):
        return "satellite"
    
    return "natural"

def infer_domain(caption: str = "", user_hint: str = "") -> Domain:
    """
    Infer domain from both caption and user hint.
    User hint takes precedence, then caption analysis.
    """
    # First try user hint
    if user_hint:
        hint_domain = infer_domain_from_hint(user_hint)
        if hint_domain != "unknown" and hint_domain != "natural":
            return hint_domain
    
    # Then try caption
    caption_domain = infer_domain_from_caption(caption)
    if caption_domain != "unknown":
        return caption_domain
    
    # Default to natural
    return "natural"
