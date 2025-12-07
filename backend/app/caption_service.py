# app/caption_service.py
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_MODEL_NAME = "Salesforce/blip-image-captioning-base"
_processor = BlipProcessor.from_pretrained(_MODEL_NAME)
_model = BlipForConditionalGeneration.from_pretrained(_MODEL_NAME).to(DEVICE)
_model.eval()

def generate_caption(img: Image.Image, max_new_tokens: int = 30) -> str:
    with torch.no_grad():
        inputs = _processor(images=img, return_tensors="pt").to(DEVICE)
        out = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=5,
            early_stopping=True,
        )
        caption = _processor.decode(out[0], skip_special_tokens=True)
    return caption
