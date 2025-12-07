# app/clip_service.py
import numpy as np
import torch
import open_clip
from PIL import Image
from typing import Dict, List, Optional, Tuple

from .prompt_service import build_prompts_for_label

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# load CLIP once
_model, _, _preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14",
    pretrained="openai",
)
_tokenizer = open_clip.get_tokenizer("ViT-L-14")
_model = _model.to(DEVICE)
_model.eval()

# adaptive state (in-memory)
CLASS_PROTOTYPES: Dict[str, np.ndarray] = {}
CLASS_COUNTS: Dict[str, int] = {}   # how many times we've updated
CONFIDENCE_THRESHOLD = 0.28         # min cosine sim to trust prediction


def encode_texts(texts: List[str]) -> torch.Tensor:
    with torch.no_grad():
        tokens = _tokenizer(texts).to(DEVICE)
        feats = _model.encode_text(tokens)
        feats /= feats.norm(dim=-1, keepdim=True)
    return feats  # [N, D]


def encode_image(img: Image.Image) -> np.ndarray:
    tensor = _preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feats = _model.encode_image(tensor)
        feats /= feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()[0]  # [D]


def create_class_prototype(
    label: str,
    domain: str = "natural",
    images: Optional[List[Image.Image]] = None,
    w_text: float = 0.5,
    w_image: float = 0.5,
) -> Dict:
    """Create/replace prototype for a label using prompts (+ optional images)."""
    prompts = build_prompts_for_label(label, domain)
    text_embs = encode_texts(prompts).cpu().numpy()  # [n, D]
    t_proto = text_embs.mean(axis=0)
    t_proto /= np.linalg.norm(t_proto) + 1e-8

    if images:
        img_vecs = [encode_image(im) for im in images]
        i_proto = np.stack(img_vecs, axis=0).mean(axis=0)
        i_proto /= np.linalg.norm(i_proto) + 1e-8
        vec = w_text * t_proto + w_image * i_proto
        num_images = len(images)
    else:
        vec = t_proto
        num_images = 0

    norm = np.linalg.norm(vec) + 1e-8
    vec /= norm

    CLASS_PROTOTYPES[label] = vec
    # initialize count with at least 1 to keep alpha small
    CLASS_COUNTS[label] = CLASS_COUNTS.get(label, 1) + max(1, num_images)

    return {"num_images": num_images, "norm": norm}


def _update_prototype_after_prediction(label: str, img_vec: np.ndarray) -> None:
    """Automatic online learning via EMA update."""
    if label not in CLASS_PROTOTYPES:
        return
    old = CLASS_PROTOTYPES[label]
    n = CLASS_COUNTS.get(label, 1)
    # decaying learning rate – more data → smaller updates
    alpha = 1.0 / (n + 1)   # or fixed like 0.1
    new = (1 - alpha) * old + alpha * img_vec
    new /= np.linalg.norm(new) + 1e-8
    CLASS_PROTOTYPES[label] = new
    CLASS_COUNTS[label] = n + 1


def classify_image(
    img: Image.Image,
    top_k: int = 5,
) -> Dict:
    if not CLASS_PROTOTYPES:
        raise RuntimeError("No classes defined. Add classes first using /api/add-class.")

    img_vec = encode_image(img)
    labels = list(CLASS_PROTOTYPES.keys())
    mat = np.stack([CLASS_PROTOTYPES[l] for l in labels])  # [N, D]

    sims = mat @ img_vec
    idx = np.argsort(-sims)
    idx_top = idx[:top_k]

    candidates = [
        {"label": labels[i], "score": float(sims[i])}
        for i in idx_top
    ]
    best = candidates[0]
    best_label, best_score = best["label"], best["score"]

    if best_score >= CONFIDENCE_THRESHOLD:
        _update_prototype_after_prediction(best_label, img_vec)

    return {
        "label": best_label,
        "confidence": best_score,
        "candidates": candidates,
    }


def list_classes() -> List[str]:
    return sorted(CLASS_PROTOTYPES.keys())
