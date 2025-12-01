"""
Implements online continual learning through a proxy-based mechanism.
This allows the model to adapt to new data in real-time without
full retraining.
"""
import torch

class OnlineTuner:
    def __init__(self, alpha=0.1):
        """
        Initializes the OnlineTuner.
        :param alpha: The learning rate for the moving average. A smaller
                      alpha means slower adaptation.
        """
        self.proxy_embeddings = {}
        self.alpha = alpha

    def update_class_embeddings(self, class_name, new_image_features):
        """
        Updates the proxy embedding for a class using a moving average.
        This is called after a classification to help the model learn from
        the new example.
        """
        if class_name not in self.proxy_embeddings:
            self.proxy_embeddings[class_name] = new_image_features
        else:
            # Update using exponential moving average
            self.proxy_embeddings[class_name] = (1 - self.alpha) * self.proxy_embeddings[class_name] + self.alpha * new_image_features

    def get_adapted_embedding(self, class_name, original_text_embedding):
        """
        Returns the adapted embedding for a class if it exists, otherwise
        returns the original text-based embedding.
        """
        return self.proxy_embeddings.get(class_name, original_text_embedding)
