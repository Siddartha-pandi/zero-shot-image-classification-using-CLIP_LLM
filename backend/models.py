"""
Model Management for the Adaptive CLIP-LLM Framework.

Handles loading, initialization, and access to all models.
"""

import torch
import logging
from transformers import CLIPModel, CLIPProcessor, AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ModelManager:
    """
    A singleton class to manage all models for the framework.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ModelManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.clip_model = None
            self.clip_processor = None
            self.llm_model = None
            self.llm_tokenizer = None
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.initialized = False
            logger.info(f"Using device: {self.device}")

    def load_models(self, clip_model_name="openai/clip-vit-base-patch32", llm_model_name="gpt2"):
        """
        Loads all necessary models from Hugging Face.
        """
        if self.initialized and self.clip_model is not None:
            logger.info("Models already loaded")
            return True

        try:
            logger.info(f"Loading models on device: {self.device}")

            # Load CLIP model
            logger.info("Loading CLIP model...")
            self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
            logger.info("CLIP model loaded successfully")

            # Load Language Model for narrative generation
            logger.info("Loading LLM for reasoning...")
            self.llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name).to(self.device)
            self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
            
            # Set pad token if not exists
            if self.llm_tokenizer.pad_token is None:
                self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
            
            logger.info("LLM loaded successfully")

            logger.info("All models loaded successfully")
            self.initialized = True
            return True

        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            self.initialized = False
            return False

    def generate_narrative(self, text_prompt, max_length=50):
        """
        Generates a rich narrative using the LLM.
        """
        if not self.llm_model or not self.llm_tokenizer:
            return "LLM not available for narrative generation."

        try:
            inputs = self.llm_tokenizer(
                text_prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.llm_tokenizer.pad_token_id
                )
            
            return self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        except Exception as e:
            logger.error(f"Error generating narrative: {e}")
            return f"Error generating narrative: {str(e)}"

    def get_models(self):
        """Return loaded models"""
        return {
            'clip_model': self.clip_model,
            'clip_processor': self.clip_processor,
            'llm_model': self.llm_model,
            'llm_tokenizer': self.llm_tokenizer,
            'device': self.device
        }


# Global model manager instance
model_manager = ModelManager()


def initialize_models():
    """Initialize all models"""
    return model_manager.load_models()


def get_model_manager():
    """Get the global model manager instance"""
    return model_manager
