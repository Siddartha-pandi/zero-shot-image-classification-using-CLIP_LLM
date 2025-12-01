"""
Utility functions for image processing and tensor operations.
"""
import torch
from PIL import Image

def preprocess_image(image_path, processor):
    """
    Loads and preprocesses an image for CLIP using the provided processor.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    return inputs['pixel_values']

def cosine_similarity(t1, t2):
    """
    Computes cosine similarity between two tensors.
    """
    return torch.nn.functional.cosine_similarity(t1, t2)
