"""
The core of the Advanced Zero-Shot Classification Framework.
This module integrates the hybrid multimodal reasoning engine, advanced
prompt engineering, auto-tuning, and online continual learning.
"""
import torch
from PIL import Image
from .models import get_model_manager
from .prompt_engineering import PromptEngineer
from .online_learning import OnlineTuner

class AdvancedZeroShotFramework:
    def __init__(self):
        self.model_manager = get_model_manager()
        self.prompt_engineer = PromptEngineer()
        self.online_tuner = OnlineTuner()
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

        # 2. Hybrid Multimodal Reasoning: Generate narrative for the image
        # This is a simplified example. A more complex system might use the
        # image features to condition the LLM.
        image_narrative = self.model_manager.generate_narrative("An image shows:")
        print(f"LLM Image Narrative: {image_narrative}")

        final_scores = {}
        for class_name in class_names:
            # 3. Generate multiple prompts for the current class
            prompts = self.prompt_engineer.generate_prompts_for_class(class_name)
            text_inputs = clip_processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
            prompt_embeddings = clip_model.get_text_features(**text_inputs)

            # 4. Dynamic Auto-Tuning: Get weights for each prompt
            prompt_weights = self.prompt_engineer.auto_tune_weights(image_features, prompt_embeddings)
            
            # 5. Create a single, style-agnostic embedding for the class
            # by taking a weighted average of the prompt embeddings.
            class_embedding = torch.sum(prompt_weights.T * prompt_embeddings, dim=0)
            class_embedding /= class_embedding.norm()

            # 6. Online Continual Learning: Adapt the embedding
            # This uses the online tuner to adjust the embedding based on past data.
            adapted_embedding = self.online_tuner.get_adapted_embedding(class_name, class_embedding)
            
            # 7. Calculate final similarity score
            similarity = (100.0 * image_features @ adapted_embedding.T).item()
            final_scores[class_name] = similarity

        # 8. Update the online tuner with the features of the top-scoring class
        if final_scores:
            best_class = max(final_scores, key=final_scores.get)
            self.online_tuner.update_class_embeddings(best_class, image_features.squeeze())
            print(f"Online tuner updated for class: {best_class}")

        return final_scores
