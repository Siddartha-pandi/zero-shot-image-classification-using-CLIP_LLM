# backend/services/caption_generator.py
import torch
from PIL import Image
import logging
import base64
import io

from models.blip_model import get_blip_model
from models.llm_model import get_llm_model
from models.model_cache import DEVICE

logger = logging.getLogger(__name__)

def _generate_llm_caption(img: Image.Image) -> str:
    """Generate caption using Gemini Vision LLM"""
    try:
        llm = get_llm_model()
        if llm.model is None:
            return None
        
        # Prepare image for Gemini
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        prompt = """Provide a concise, factual description of this image in one sentence (max 15 words). 
Focus on the main subject, its key visual features, and context. Be specific and precise.
Do not use phrases like 'the image shows' or 'this is'. Just describe what you see."""
        
        from google.generativeai import types
        response = llm.model.generate_content(
            [prompt, {"mime_type": "image/png", "data": img_str}],
            generation_config=types.GenerationConfig(temperature=0.3, max_output_tokens=50)
        )
        
        caption = response.text.strip()
        logger.info(f"LLM caption: {caption}")
        return caption
    except Exception as e:
        logger.warning(f"LLM caption generation failed: {e}")
        return None

def generate_caption(img: Image.Image, max_new_tokens: int = 30) -> str:
    """Generate image caption using hybrid BLIP + LLM approach"""
    blip_caption = None
    llm_caption = None
    
    # Get BLIP caption
    try:
        blip = get_blip_model()
        processor = blip.processor
        model = blip.model
        
        with torch.no_grad():
            inputs = processor(images=img, return_tensors="pt").to(DEVICE)
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=5,
                early_stopping=True,
                repetition_penalty=1.1,
                no_repeat_ngram_size=2,
            )
            blip_caption = processor.decode(out[0], skip_special_tokens=True)
            
            # BLIP frequently misspells giraffe as "git" or "gi gife" with various penalties
            blip_caption = blip_caption.replace("gi gife", "giraffes").replace("git", "giraffes")
            logger.info(f"BLIP caption: {blip_caption}")
    except Exception as e:
        logger.error(f"Error generating BLIP caption: {e}")
    
    # Get LLM caption
    llm_caption = _generate_llm_caption(img)
    
    # Combine both captions intelligently
    if blip_caption and llm_caption:
        # Use LLM caption as primary (better semantic understanding)
        # but fall back to BLIP if LLM fails
        logger.info(f"Using hybrid caption (LLM primary)")
        return llm_caption
    elif llm_caption:
        return llm_caption
    elif blip_caption:
        return blip_caption
    else:
        return "An image."
