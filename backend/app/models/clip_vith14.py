# app/models/clip_vith14.py
"""
OpenCLIP ViT-H/14 Model
Universal zero-shot image understanding
"""
import torch
import open_clip
from PIL import Image
from typing import List, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ViTH14Model:
    """ViT-H/14 model for universal image classification"""
    
    def __init__(self):
        self.device = DEVICE
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load ViT-H/14 model from OpenCLIP"""
        try:
            logger.info("=" * 60)
            logger.info("Loading ViT-H/14 model (this may take 1-2 minutes)...")
            logger.info("=" * 60)
            
            # Use ViT-L-14 by default (faster, smaller, already downloaded)
            try:
                logger.info("Attempting to load ViT-L-14 (lightweight)...")
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    "ViT-L-14",
                    pretrained="openai"
                )
                logger.info("✓ ViT-L-14 loaded successfully")
            except Exception as e1:
                logger.warning(f"ViT-L-14 failed ({e1}), trying ViT-B-32...")
                # Fallback to even smaller model
                try:
                    self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                        "ViT-B-32",
                        pretrained="openai"
                    )
                    logger.info("✓ ViT-B-32 loaded successfully (fallback)")
                except Exception as e2:
                    logger.error(f"All model loading attempts failed: {e2}")
                    raise
            
            self.tokenizer = open_clip.get_tokenizer("ViT-L-14")
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info(f"✓ Model ready on {self.device}")
            logger.info("=" * 60)
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
    
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """
        Encode image to embedding vector
        
        Args:
            image: PIL Image
            
        Returns:
            Normalized embedding vector [D]
        """
        with torch.no_grad():
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()[0]
    
    def encode_text(self, texts: List[str]) -> np.ndarray:
        """
        Encode text prompts to embedding vectors
        
        Args:
            texts: List of text prompts
            
        Returns:
            Normalized embedding matrix [N, D]
        """
        with torch.no_grad():
            text_tokens = self.tokenizer(texts).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()
    
    def compute_similarity(self, image_emb: np.ndarray, text_embs: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between image and text embeddings
        
        Args:
            image_emb: Image embedding [D]
            text_embs: Text embeddings [N, D]
            
        Returns:
            Similarity scores [N]
        """
        # Both are already normalized, so dot product = cosine similarity
        similarities = text_embs @ image_emb
        return similarities
    
    def classify(
        self, 
        image: Image.Image, 
        labels: List[str], 
        top_k: int = 5
    ) -> Tuple[List[dict], np.ndarray]:
        """
        Classify image with given labels
        
        Args:
            image: PIL Image
            labels: List of class labels
            top_k: Number of top predictions to return
            
        Returns:
            Tuple of (top predictions, image embedding)
        """
        # Encode image
        image_emb = self.encode_image(image)
        
        # Encode labels
        text_embs = self.encode_text(labels)
        
        # Compute similarities
        similarities = self.compute_similarity(image_emb, text_embs)
        
        # Get top-k predictions
        top_indices = np.argsort(-similarities)[:top_k]
        
        predictions = [
            {
                "label": labels[idx],
                "score": float(similarities[idx])
            }
            for idx in top_indices
        ]
        
        return predictions, image_emb


# Global instance
_vith14_model = None

def get_vith14_model() -> ViTH14Model:
    """Get or create global ViT-H/14 model instance"""
    global _vith14_model
    if _vith14_model is None:
        _vith14_model = ViTH14Model()
    return _vith14_model
