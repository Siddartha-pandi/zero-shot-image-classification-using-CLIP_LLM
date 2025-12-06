"""
Adaptive embedding module for domain-specific CLIP embedding transformations.
Implements FiLM-style conditioning with learnable scale, shift, and MLP components.
"""
import torch
import torch.nn as nn
import os
import json
import logging

logger = logging.getLogger(__name__)


class AdaptiveModule(nn.Module):
    """
    Lightweight adapter that applies scale-and-shift transformation to embeddings.
    Includes optional FiLM-style MLP for more expressive transformations.
    """
    
    def __init__(self, dim: int, hidden: int = 256, use_mlp: bool = True):
        """
        Args:
            dim: Embedding dimension
            hidden: Hidden dimension for MLP
            use_mlp: Whether to use MLP component
        """
        super().__init__()
        self.dim = dim
        self.hidden = hidden
        self.use_mlp = use_mlp
        
        # Scale and shift parameters
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
        
        # Optional MLP for FiLM-like transformation
        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(dim, hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, dim)
            )
        else:
            self.mlp = None
        
        # Gating mechanism to blend transformations
        self.gate = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive transformation to embeddings.
        
        Args:
            x: Input embeddings of shape (batch, dim) or (dim,)
        
        Returns:
            Transformed embeddings
        """
        original_shape = x.shape
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Scale and shift transformation
        x_scaled = x * self.scale + self.shift
        
        # MLP transformation if enabled
        if self.use_mlp and self.mlp is not None:
            x_mlp = x + self.mlp(x)  # Residual connection
            
            # Blend using learned gate
            gate = torch.sigmoid(self.gate)
            x_out = gate * x_mlp + (1 - gate) * x_scaled
        else:
            x_out = x_scaled
        
        # Restore original shape
        if len(original_shape) == 1:
            x_out = x_out.squeeze(0)
        
        return x_out
    
    def save(self, path: str):
        """Save module state to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            'state_dict': self.state_dict(),
            'dim': self.dim,
            'hidden': self.hidden,
            'use_mlp': self.use_mlp
        }
        torch.save(state, path)
        logger.info(f"Saved AdaptiveModule to {path}")
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu'):
        """Load module from disk."""
        state = torch.load(path, map_location=device)
        module = cls(
            dim=state['dim'],
            hidden=state['hidden'],
            use_mlp=state['use_mlp']
        )
        module.load_state_dict(state['state_dict'])
        module.to(device)
        logger.info(f"Loaded AdaptiveModule from {path}")
        return module


class DomainAdaptiveModules:
    """
    Manager for domain-specific adaptive modules.
    Loads and stores different adapters for different domains.
    """
    
    def __init__(self, base_path: str = "./models/adapters"):
        self.base_path = base_path
        self.modules = {}
        self.default_module = None
        os.makedirs(base_path, exist_ok=True)
    
    def get_module(self, domain: str, dim: int, device: str = 'cpu') -> AdaptiveModule:
        """
        Get adaptive module for a specific domain.
        Creates new module if not exists.
        
        Args:
            domain: Domain name (e.g., 'medical_image', 'sketch')
            dim: Embedding dimension
            device: Device to load module on
        
        Returns:
            AdaptiveModule for the domain
        """
        if domain not in self.modules:
            module_path = os.path.join(self.base_path, f"{domain}_adapter.pt")
            
            if os.path.exists(module_path):
                try:
                    self.modules[domain] = AdaptiveModule.load(module_path, device)
                    logger.info(f"Loaded adapter for domain: {domain}")
                except Exception as e:
                    logger.warning(f"Failed to load adapter for {domain}: {e}, creating new one")
                    self.modules[domain] = AdaptiveModule(dim).to(device)
            else:
                logger.info(f"Creating new adapter for domain: {domain}")
                self.modules[domain] = AdaptiveModule(dim).to(device)
        
        return self.modules[domain]
    
    def save_module(self, domain: str):
        """Save domain-specific module to disk."""
        if domain in self.modules:
            module_path = os.path.join(self.base_path, f"{domain}_adapter.pt")
            self.modules[domain].save(module_path)
    
    def save_all(self):
        """Save all loaded modules."""
        for domain in self.modules:
            self.save_module(domain)
