# app/domain_service.py
from typing import Literal, Optional

Domain = Literal["natural", "medical", "anime", "sketch", "satellite", "unknown"]

KEYWORDS = {
    "medical": ["xray", "x-ray", "ct", "mri", "scan", "radiograph"],
    "anime": ["anime", "manga", "cartoon", "illustration"],
    "satellite": ["satellite", "aerial", "top view", "remote sensing"],
    "sketch": ["sketch", "line art", "line-art", "drawing"],
}

def infer_domain_from_hint(hint: Optional[str]) -> Domain:
    if not hint:
        return "unknown"
    h = hint.lower()
    for dom, words in KEYWORDS.items():
        if any(w in h for w in words):
            return dom  # type: ignore
    return "natural"
