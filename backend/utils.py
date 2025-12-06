"""
Utility functions for image processing, tensor operations, and mathematical computations.
"""
import torch
import numpy as np
from PIL import Image
from typing import List, Tuple


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


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Numerically stable softmax with temperature scaling.
    
    Args:
        logits: Input logits array
        temperature: Temperature parameter for scaling (higher = softer distribution)
    
    Returns:
        Probability distribution that sums to 1.0
    """
    assert temperature > 0, "Temperature must be positive"
    logits = np.array(logits, dtype=np.float64)
    z = logits / temperature
    z_max = np.max(z, axis=-1, keepdims=True)
    e = np.exp(z - z_max)
    return e / np.sum(e, axis=-1, keepdims=True)


def topk_from_probs(probs: np.ndarray, labels: list, k: int = 5) -> List[Tuple[str, float]]:
    """
    Get top-k labels and their probabilities.
    
    Args:
        probs: Probability array
        labels: List of label names
        k: Number of top predictions to return
    
    Returns:
        List of (label, probability) tuples sorted by probability
    """
    idx = np.argsort(probs)[::-1][:k]
    return [(labels[i], float(probs[i])) for i in idx]


def calibrate_temperature(logits: np.ndarray, true_labels: np.ndarray, 
                         temperatures: List[float] = None) -> float:
    """
    Find optimal temperature for calibration using cross-entropy.
    
    Args:
        logits: Model logits (n_samples, n_classes)
        true_labels: True class indices (n_samples,)
        temperatures: List of temperatures to try
    
    Returns:
        Optimal temperature value
    """
    if temperatures is None:
        temperatures = [0.01, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0]
    
    best_temp = 1.0
    best_loss = float('inf')
    
    for temp in temperatures:
        probs = softmax(logits, temperature=temp)
        # Cross-entropy loss
        loss = -np.mean(np.log(probs[np.arange(len(true_labels)), true_labels] + 1e-10))
        if loss < best_loss:
            best_loss = loss
            best_temp = temp
    
    return best_temp


def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """
    L2 normalize embeddings.
    
    Args:
        embeddings: Tensor of shape (batch, dim) or (dim,)
    
    Returns:
        Normalized embeddings
    """
    return embeddings / embeddings.norm(dim=-1, keepdim=True)

