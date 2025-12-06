"""
Model Management for the Adaptive CLIP-LLM Framework.

Handles loading, initialization, and access to all models.
"""

import torch
import logging
from transformers import CLIPModel, CLIPProcessor, AutoModelForCausalLM, AutoTokenizer
import open_clip

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ModelManager:
    """
    A singleton class to manage all models for the framework.
    Supports multiple CLIP models for ensemble inference.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ModelManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            # Support for multiple CLIP models (ensemble)
            self.clip_models = []
            self.clip_processors = []
            self.model_names = []
            
            # Legacy support for single model access
            self.clip_model = None
            self.clip_processor = None
            
            self.llm_model = None
            self.llm_tokenizer = None
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.initialized = False
            logger.info(f"Using device: {self.device}")

    def load_models(self, clip_model_names=None, llm_model_name="gpt2"):
        """
        Loads all necessary models from Hugging Face and OpenCLIP.
        
        Args:
            clip_model_names: List of CLIP model configurations. Each can be:
                - String for HuggingFace models (e.g., "openai/clip-vit-large-patch14")
                - Tuple for OpenCLIP models (e.g., ("ViT-H-14", "laion2b_s32b_b79k"))
            llm_model_name: Name of the language model to use
        """
        if self.initialized and len(self.clip_models) > 0:
            logger.info("Models already loaded")
            return True

        # Default to ViT-L/14 and OpenCLIP ViT-H/14
        if clip_model_names is None:
            clip_model_names = [
                "openai/clip-vit-large-patch14",  # ViT-L/14
                ("ViT-H-14", "laion2b_s32b_b79k")  # OpenCLIP ViT-H/14
            ]

        try:
            logger.info(f"Loading models on device: {self.device}")

            # Load CLIP models
            for model_config in clip_model_names:
                if isinstance(model_config, tuple):
                    # OpenCLIP model
                    model_name, pretrained = model_config
                    logger.info(f"Loading OpenCLIP model: {model_name} with {pretrained}...")
                    
                    model, _, preprocess = open_clip.create_model_and_transforms(
                        model_name, 
                        pretrained=pretrained,
                        device=self.device
                    )
                    tokenizer = open_clip.get_tokenizer(model_name)
                    
                    # Wrap in a dict to maintain consistent interface
                    self.clip_models.append({
                        'model': model,
                        'preprocess': preprocess,
                        'tokenizer': tokenizer,
                        'type': 'openclip',
                        'name': f"{model_name}_{pretrained}"
                    })
                    self.model_names.append(f"OpenCLIP-{model_name}")
                    logger.info(f"OpenCLIP model {model_name} loaded successfully")
                    
                else:
                    # HuggingFace CLIP model
                    model_name = model_config
                    logger.info(f"Loading HuggingFace CLIP model: {model_name}...")
                    
                    model = CLIPModel.from_pretrained(model_name).to(self.device)
                    processor = CLIPProcessor.from_pretrained(model_name)
                    
                    self.clip_models.append({
                        'model': model,
                        'processor': processor,
                        'type': 'huggingface',
                        'name': model_name
                    })
                    self.model_names.append(model_name)
                    logger.info(f"HuggingFace CLIP model {model_name} loaded successfully")

            # Set the first model as default for backward compatibility
            if len(self.clip_models) > 0:
                first_model = self.clip_models[0]
                if first_model['type'] == 'huggingface':
                    self.clip_model = first_model['model']
                    self.clip_processor = first_model['processor']
                else:
                    # For OpenCLIP, create a compatibility wrapper
                    self.clip_model = first_model['model']
                    self.clip_processor = first_model

            logger.info(f"Loaded {len(self.clip_models)} CLIP model(s): {self.model_names}")

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
            import traceback
            traceback.print_exc()
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
            'clip_models': self.clip_models,  # All models for ensemble
            'model_names': self.model_names,
            'llm_model': self.llm_model,
            'llm_tokenizer': self.llm_tokenizer,
            'device': self.device
        }
    
    def get_ensemble_features(self, image=None, text=None, normalize=True):
        """
        Get ensemble features from all loaded CLIP models.
        
        Args:
            image: PIL Image or image tensor
            text: String or list of strings
            normalize: Whether to normalize the features
            
        Returns:
            Averaged features from all models
        """
        all_features = []
        
        for model_dict in self.clip_models:
            model_type = model_dict['type']
            model = model_dict['model']
            
            if model_type == 'huggingface':
                processor = model_dict['processor']
                
                if image is not None:
                    inputs = processor(images=image, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        features = model.get_image_features(**inputs)
                elif text is not None:
                    inputs = processor(text=text, return_tensors="pt", padding=True).to(self.device)
                    with torch.no_grad():
                        features = model.get_text_features(**inputs)
                else:
                    continue
                    
            else:  # openclip
                tokenizer = model_dict['tokenizer']
                preprocess = model_dict['preprocess']
                
                if image is not None:
                    if not isinstance(image, torch.Tensor):
                        image_tensor = preprocess(image).unsqueeze(0).to(self.device)
                    else:
                        image_tensor = image
                    
                    with torch.no_grad():
                        features = model.encode_image(image_tensor)
                elif text is not None:
                    if isinstance(text, str):
                        text = [text]
                    text_tokens = tokenizer(text).to(self.device)
                    
                    with torch.no_grad():
                        features = model.encode_text(text_tokens)
                else:
                    continue
            
            if normalize:
                features = features / features.norm(dim=-1, keepdim=True)
            
            all_features.append(features)
        
        # Handle different embedding dimensions by concatenating instead of stacking
        if len(all_features) > 0:
            # Check if all features have the same dimension
            dims = [f.shape[-1] for f in all_features]
            if len(set(dims)) == 1:
                # Same dimensions - can stack and average
                ensemble_features = torch.stack(all_features).mean(dim=0)
            else:
                # Different dimensions - concatenate and normalize
                ensemble_features = torch.cat(all_features, dim=-1)
                if normalize:
                    ensemble_features = ensemble_features / ensemble_features.norm(dim=-1, keepdim=True)
            if normalize:
                ensemble_features = ensemble_features / ensemble_features.norm(dim=-1, keepdim=True)
            return ensemble_features
        else:
            return None


# Global model manager instance
model_manager = ModelManager()


def initialize_models():
    """Initialize all models"""
    return model_manager.load_models()


def get_model_manager():
    """Get the global model manager instance"""
    return model_manager
