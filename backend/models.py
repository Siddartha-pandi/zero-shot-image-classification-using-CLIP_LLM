"""""""""

Model Management for the Adaptive CLIP-LLM Framework.

Handles loading, initialization, and access to all models.Model Management for the Adaptive CLIP-LLM Framework.Model loading and initialization for the Adaptive CLIP-LLM Framework

This version is updated to support the advanced framework.

"""Handles loading, initialization, and access to all models."""

from transformers import CLIPModel, CLIPProcessor, AutoModelForCausalLM, AutoTokenizer

import torch"""import torch

import logging

from transformers import CLIPModel, CLIPProcessor, AutoModelForCausalLM, AutoTokenizerimport clip

logger = logging.getLogger(__name__)

import torchfrom transformers import pipeline

class ModelManager:

    """import loggingimport numpy as np

    A singleton class to manage all models for the advanced framework.

    """from PIL import Image

    _instance = None

logger = logging.getLogger(__name__)import logging

    def __new__(cls, *args, **kwargs):

        if not cls._instance:

            cls._instance = super(ModelManager, cls).__new__(cls, *args, **kwargs)

        return cls._instanceclass ModelManager:logging.basicConfig(level=logging.INFO)



    def __init__(self):    """logger = logging.getLogger(__name__)

        if not hasattr(self, 'initialized'):

            self.clip_model = None    A singleton class to manage all models.

            self.clip_processor = None

            self.llm_model = None    """class ModelManager:

            self.llm_tokenizer = None

            self.device = "cuda" if torch.cuda.is_available() else "cpu"    _instance = None    def __init__(self):

            self.initialized = False

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_models(self, clip_model_name="openai/clip-vit-base-patch32", llm_model_name="gpt2"):

        """    def __new__(cls, *args, **kwargs):        logger.info(f"Using device: {self.device}")

        Loads all necessary models from Hugging Face.

        """        if not cls._instance:        

        if self.initialized and self.clip_model is not None:

            return True            cls._instance = super(ModelManager, cls).__new__(cls, *args, **kwargs)        # Initialize models

            

        try:        return cls._instance        self.clip_model = None

            logger.info(f"Loading models on device: {self.device}")

                    self.clip_preprocess = None

            # Load CLIP model

            self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)    def __init__(self):        self.llm_generator = None

            self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

                    if not hasattr(self, 'initialized'):        self.domain_classifier = None

            # Load Language Model for narrative generation

            self.llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name).to(self.device)            self.clip_model = None        

            self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)

                        self.clip_processor = None    def load_models(self):

            # Set pad token if not exists

            if self.llm_tokenizer.pad_token is None:            self.llm_model = None        """Load CLIP model, LLM, and domain classifier"""

                self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

                        self.llm_tokenizer = None        try:

            logger.info("All models for the advanced framework loaded successfully.")

            self.initialized = True            self.device = "cuda" if torch.cuda.is_available() else "cpu"            # Load CLIP model

            return True

        except Exception as e:            self.initialized = True            logger.info("Loading CLIP model...")

            logger.error(f"Error loading models: {e}")

            self.initialized = False            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

            return False

    def load_models(self, clip_model_name="openai/clip-vit-base-patch32", llm_model_name="gpt2"):            logger.info("CLIP model loaded successfully")

    def generate_narrative(self, text_prompt, max_length=50):

        """        """            

        Generates a rich narrative using the LLM.

        """        Loads all necessary models from Hugging Face.            # Load LLM for reasoning (using a lightweight model) - make it optional

        if not self.llm_model or not self.llm_tokenizer:

            return "LLM not available for narrative generation."        """            try:

            

        try:        try:                logger.info("Loading LLM for reasoning...")

            inputs = self.llm_tokenizer(text_prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)

            with torch.no_grad():            logger.info(f"Loading models on device: {self.device}")                self.llm_generator = pipeline(

                outputs = self.llm_model.generate(

                    **inputs,                                 "text-generation", 

                    max_length=max_length, 

                    num_return_sequences=1,            # Load CLIP model                    model="distilgpt2",

                    do_sample=True,

                    temperature=0.7,            self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)                    device=0 if self.device == "cuda" else -1

                    pad_token_id=self.llm_tokenizer.pad_token_id

                )            self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)                )

            return self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

        except Exception as e:                            logger.info("LLM loaded successfully")

            logger.error(f"Error generating narrative: {e}")

            return f"Error generating narrative: {str(e)}"            # Load Language Model            except Exception as llm_error:



# Global instance of the model manager            self.llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name).to(self.device)                logger.warning(f"Failed to load LLM: {llm_error}")

_model_manager = ModelManager()

            self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)                logger.warning("Continuing without LLM - reasoning will use fallback methods")

def initialize_models():

    """Initializes all models, returns True on success."""                            self.llm_generator = None

    return _model_manager.load_models()

            logger.info("All models loaded successfully.")            

def get_model_manager():

    """Returns the global instance of the ModelManager."""            return True            # Mock domain classifier (in practice, this would be a trained model)

    return _model_manager
        except Exception as e:            logger.info("Domain classifier initialized")

            logger.error(f"Error loading models: {e}")            

            return False            return True

            

# Global instance of the model manager        except Exception as e:

_model_manager = ModelManager()            logger.error(f"Error loading models: {str(e)}")

            return False

def initialize_models():    

    """Initializes all models, returns True on success."""    def get_models(self):

    return _model_manager.load_models()        """Return loaded models"""

        return {

def get_model_manager():            'clip_model': self.clip_model,

    """Returns the global instance of the ModelManager."""            'clip_preprocess': self.clip_preprocess,

    return _model_manager            'llm_generator': self.llm_generator,

            'domain_classifier': self.domain_classifier,
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
