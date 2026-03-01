# app/models/medclip_model.py
"""
MedCLIP Model
Specialized medical image understanding
"""
import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class MedCLIPModel:
    """MedCLIP model for medical image classification"""
    
    def __init__(self):
        self.device = DEVICE
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load MedCLIP model"""
        try:
            logger.info("=" * 60)
            logger.info("Loading Medical CLIP model...")
            logger.info("=" * 60)
            
            # Try to import and load MedCLIP
            try:
                from medclip import MedCLIPModel as MedCLIPBase, MedCLIPVisionModelViT
                from medclip import MedCLIPProcessor
                
                logger.info("MedCLIP library found, loading pretrained model...")
                # Load pre-trained MedCLIP
                self.processor = MedCLIPProcessor()
                self.model = MedCLIPBase(vision_cls=MedCLIPVisionModelViT)
                self.model.from_pretrained()
                self.model = self.model.to(self.device)
                self.model.eval()
                
                logger.info(f"✓ MedCLIP loaded successfully on {self.device}")
                
            except ImportError:
                logger.warning("MedCLIP not installed, using OpenCLIP fallback for medical")
                # Fallback to regular OpenCLIP with medical prompts
                import open_clip
                logger.info("Loading OpenCLIP ViT-B-16 as medical fallback...")
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    "ViT-B-16",
                    pretrained="openai"
                )
                self.tokenizer = open_clip.get_tokenizer("ViT-B-16")
                self.model = self.model.to(self.device)
                self.model.eval()
                self.processor = None
                logger.info(f"✓ OpenCLIP fallback loaded on {self.device}")
            
            logger.info("=" * 60)
                
        except Exception as e:
            logger.error(f"Failed to load Medical CLIP model: {e}")
            logger.error("Models may need to be downloaded on first request")
            raise
    
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """
        Encode medical image to embedding vector
        
        Args:
            image: PIL Image (medical image)
            
        Returns:
            Normalized embedding vector [D]
        """
        with torch.no_grad():
            if self.processor is not None:
                # Using actual MedCLIP
                inputs = self.processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                image_features = self.model.encode_image(**inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            else:
                # Using OpenCLIP fallback
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                image_features = self.model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()[0]
    
    def encode_text(self, texts: List[str]) -> np.ndarray:
        """
        Encode medical text prompts to embedding vectors
        
        Args:
            texts: List of medical term prompts
            
        Returns:
            Normalized embedding matrix [N, D]
        """
        # Add medical context to prompts
        medical_prompts = [self._format_medical_prompt(text) for text in texts]
        
        with torch.no_grad():
            if self.processor is not None:
                # Using actual MedCLIP
                inputs = self.processor(text=medical_prompts, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                text_features = self.model.encode_text(**inputs)
                text_features /= text_features.norm(dim=-1, keepdim=True)
            else:
                # Using OpenCLIP fallback
                text_tokens = self.tokenizer(medical_prompts).to(self.device)
                text_features = self.model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy()
    
    def _format_medical_prompt(self, text: str) -> str:
        """Format text as medical prompt"""
        # Check if text already contains medical context
        medical_keywords = ["xray", "x-ray", "ct", "mri", "scan", "medical", "chest", "radiograph"]
        text_lower = text.lower()
        
        if any(kw in text_lower for kw in medical_keywords):
            return text
        
        # Add medical context
        return f"a medical x-ray showing {text}"
    
    def compute_similarity(self, image_emb: np.ndarray, text_embs: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between image and text embeddings
        
        Args:
            image_emb: Image embedding [D]
            text_embs: Text embeddings [N, D]
            
        Returns:
            Similarity scores [N]
        """
        similarities = text_embs @ image_emb
        return similarities
    
    def classify(
        self, 
        image: Image.Image, 
        labels: List[str], 
        top_k: int = 5
    ) -> Tuple[List[dict], np.ndarray]:
        """
        Classify medical image with given labels
        
        Args:
            image: PIL Image (medical image)
            labels: List of medical condition labels
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
_medclip_model = None

def get_medclip_model() -> MedCLIPModel:
    """Get or create global MedCLIP model instance"""
    global _medclip_model
    if _medclip_model is None:
        _medclip_model = MedCLIPModel()
    return _medclip_model
