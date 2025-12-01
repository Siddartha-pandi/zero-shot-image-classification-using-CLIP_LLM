"""
Handles advanced prompt engineering strategies, including generation and
auto-tuning of prompt weights based on image-text similarity.
"""
import torch

class PromptEngineer:
    def __init__(self):
        # A comprehensive list of templates for style-agnostic adaptability
        self.templates = [
            "a photo of a {}.",
            "a sketch of a {}.",
            "a drawing of a {}.",
            "a rendering of a {}.",
            "a cropped photo of the {}.",
            "a close-up photo of a {}.",
            "a low-quality photo of the {}.",
            "a high-quality photo of the {}.",
            "a photo of the small {}.",
            "a photo of the large {}.",
        ]

    def generate_prompts_for_class(self, class_name):
        """Generates a list of prompts for a single class name."""
        return [template.format(class_name) for template in self.templates]

    def auto_tune_weights(self, image_features, text_features_for_prompts):
        """
        Dynamically determines the best prompt template weights for a given image.
        It computes the cosine similarity between the image and each prompt's
        embedding, then uses a softmax function to create a weighted distribution.
        This allows the model to "focus" on the most relevant prompt styles.
        """
        with torch.no_grad():
            # Normalize features to ensure accurate similarity calculation
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features_for_prompts /= text_features_for_prompts.norm(dim=-1, keepdim=True)
            
            # Calculate cosine similarities
            similarities = (100.0 * image_features @ text_features_for_prompts.T)
            
            # Apply softmax to get weights
            weights = torch.nn.functional.softmax(similarities, dim=-1)
        return weights
