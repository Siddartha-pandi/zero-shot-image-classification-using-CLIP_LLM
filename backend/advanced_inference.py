"""
The core of the Advanced Zero-Shot Classification Framework.
This module integrates the hybrid multimodal reasoning engine, advanced
prompt engineering, auto-tuning, and online continual learning.
"""
import torch
import numpy as np
import logging
from PIL import Image
from models import get_model_manager
from prompt_engineering import PromptEngineer
from online_learning import OnlineTuner
from domain_adaptation import DomainAdaptation

logger = logging.getLogger(__name__)

class AdvancedZeroShotFramework:
    def __init__(self):
        self.model_manager = get_model_manager()
        self.prompt_engineer = PromptEngineer()
        self.online_tuner = OnlineTuner()
        self.domain_adapter = DomainAdaptation()
        self.device = self.model_manager.device

    def classify(self, image_path, class_names):
        """
        The main classification pipeline that incorporates all advanced features.
        """
        clip_model = self.model_manager.clip_model
        clip_processor = self.model_manager.clip_processor

        # 1. Preprocess the image
        image = Image.open(image_path).convert("RGB")
        image_input = clip_processor(images=image, return_tensors="pt").to(self.device)
        image_features = clip_model.get_image_features(**image_input)

        # 2. Automatic Domain Detection
        image_embeddings_np = image_features.detach().cpu().numpy()
        domain_info = self.domain_adapter.detect_domain(image_embeddings_np)
        detected_domain = domain_info['domain']
        domain_confidence = domain_info['confidence']
        print(f"Detected domain: {detected_domain} (confidence: {domain_confidence:.3f})")

        # 3. Hybrid Multimodal Reasoning: Generate narrative for the image
        # This is a simplified example. A more complex system might use the
        # image features to condition the LLM.
        image_narrative = self.model_manager.generate_narrative("An image shows:")
        print(f"LLM Image Narrative: {image_narrative}")

        final_scores = {}
        raw_scores = []
        for class_name in class_names:
            # 4. Generate multiple prompts for the current class
            prompts = self.prompt_engineer.generate_prompts_for_class(class_name)
            text_inputs = clip_processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
            prompt_embeddings = clip_model.get_text_features(**text_inputs)

            # 5. Dynamic Auto-Tuning: Get weights for each prompt
            prompt_weights = self.prompt_engineer.auto_tune_weights(image_features, prompt_embeddings)
            
            # 6. Create a single, style-agnostic embedding for the class
            # by taking a weighted average of the prompt embeddings.
            class_embedding = torch.sum(prompt_weights.T * prompt_embeddings, dim=0)
            class_embedding /= class_embedding.norm()

            # 7. Online Continual Learning: Adapt the embedding
            # This uses the online tuner to adjust the embedding based on past data.
            adapted_embedding = self.online_tuner.get_adapted_embedding(class_name, class_embedding)
            
            # 8. Calculate similarity score
            similarity = (100.0 * image_features @ adapted_embedding.T).item()
            raw_scores.append(similarity)
        
        # 9. Apply Domain-Specific Adaptive Auto-Tuning
        tuned_scores = self.domain_adapter.adaptive_auto_tuning(raw_scores, detected_domain, domain_confidence)
        
        # Create final scores dictionary
        for idx, class_name in enumerate(class_names):
            final_scores[class_name] = tuned_scores[idx]

        # 10. Update the online tuner with the features of the top-scoring class
        if final_scores:
            best_class = max(final_scores, key=final_scores.get)
            self.online_tuner.update_class_embeddings(best_class, image_features.squeeze())
            print(f"Online tuner updated for class: {best_class}")

        return final_scores, domain_info

    def generate_image_labels(self, image_path, num_labels=5):
        """
        Automatically generate relevant class labels for an image.
        Uses a comprehensive set of common object categories.
        """
        # Common categories organized by domain
        label_categories = {
            'people': ['person', 'man', 'woman', 'child', 'baby', 'group of people'],
            'animals': ['dog', 'cat', 'bird', 'horse', 'cow', 'sheep', 'wildlife'],
            'nature': ['tree', 'flower', 'plant', 'grass', 'mountain', 'sky', 'water', 'forest', 'landscape'],
            'weather': ['rain', 'sunny day', 'cloudy sky', 'storm', 'snow'],
            'activities': ['walking', 'running', 'playing', 'exercising', 'working', 'resting'],
            'objects': ['car', 'bicycle', 'building', 'house', 'road', 'bridge', 'fence'],
            'clothing': ['jacket', 'coat', 'raincoat', 'hat', 'shoes', 'umbrella'],
            'indoor': ['room', 'furniture', 'kitchen', 'bedroom', 'office'],
            'outdoor': ['park', 'street', 'garden', 'beach', 'field', 'pathway'],
        }
        
        # Flatten all labels into a comprehensive list
        all_labels = []
        for category_labels in label_categories.values():
            all_labels.extend(category_labels)
        
        try:
            # Use CLIP to score all potential labels
            from PIL import Image
            image = Image.open(image_path).convert("RGB")
            
            clip_model = self.model_manager.clip_model
            clip_processor = self.model_manager.clip_processor
            
            image_input = clip_processor(images=image, return_tensors="pt").to(self.device)
            image_features = clip_model.get_image_features(**image_input)
            
            # Score all labels
            text_inputs = clip_processor(text=all_labels, return_tensors="pt", padding=True).to(self.device)
            text_features = clip_model.get_text_features(**text_inputs)
            
            # Calculate similarities
            similarities = (image_features @ text_features.T).squeeze(0)
            
            # Get top labels
            top_indices = similarities.argsort(descending=True)[:num_labels * 2]
            
            # Filter and deduplicate
            selected_labels = []
            for idx in top_indices:
                label = all_labels[idx.item()]
                # Avoid very similar labels
                if not any(label in selected or selected in label for selected in selected_labels):
                    selected_labels.append(label)
                if len(selected_labels) >= num_labels:
                    break
            
            return selected_labels
            
        except Exception as e:
            logger.error(f"Error generating labels: {e}")
            # Fallback to generic labels
            return ['person', 'animal', 'outdoor scene', 'nature', 'object']
