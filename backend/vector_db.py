"""
Vector database wrapper using FAISS for efficient embedding storage and retrieval.
Caches embeddings and LLM outputs to reduce computation time.
"""
import faiss
import numpy as np
import json
import os
import logging
from typing import List, Dict, Tuple, Optional
import pickle

logger = logging.getLogger(__name__)


class FaissVectorDB:
    """
    FAISS-based vector database for storing and retrieving embeddings.
    Uses inner product similarity (cosine similarity with normalized vectors).
    """
    
    def __init__(self, dim: int, index_path: Optional[str] = None):
        """
        Args:
            dim: Embedding dimension
            index_path: Path to save/load index
        """
        self.dim = dim
        self.index_path = index_path
        self.index = faiss.IndexFlatIP(dim)  # Inner product index (cosine with normalized vectors)
        self.metadata = []  # List of dicts aligned with index
        
        if index_path and os.path.exists(index_path):
            self.load()
    
    def add(self, emb: np.ndarray, meta: dict):
        """
        Add embedding with metadata to the index.
        
        Args:
            emb: Embedding vector of shape (dim,)
            meta: Metadata dictionary (e.g., {'label': 'cat', 'prompt': '...', 'timestamp': ...})
        """
        v = emb.reshape(1, -1).astype('float32')
        faiss.normalize_L2(v)  # Normalize for cosine similarity
        self.index.add(v)
        self.metadata.append(meta)
        logger.debug(f"Added embedding with metadata: {meta.get('label', 'unknown')}")
    
    def add_batch(self, embs: np.ndarray, metas: List[dict]):
        """
        Add batch of embeddings with metadata.
        
        Args:
            embs: Embeddings array of shape (n, dim)
            metas: List of metadata dicts
        """
        v = embs.astype('float32')
        faiss.normalize_L2(v)
        self.index.add(v)
        self.metadata.extend(metas)
        logger.info(f"Added batch of {len(metas)} embeddings")
    
    def search(self, emb: np.ndarray, k: int = 5) -> List[Tuple[Optional[dict], float]]:
        """
        Search for k nearest neighbors.
        
        Args:
            emb: Query embedding of shape (dim,)
            k: Number of results to return
        
        Returns:
            List of (metadata, similarity_score) tuples
        """
        v = emb.reshape(1, -1).astype('float32')
        faiss.normalize_L2(v)
        D, I = self.index.search(v, k)
        
        results = []
        for d, i in zip(D[0], I[0]):
            if i != -1 and i < len(self.metadata):
                meta = self.metadata[i]
            else:
                meta = None
            results.append((meta, float(d)))
        
        return results
    
    def save(self, path: Optional[str] = None):
        """Save index and metadata to disk."""
        save_path = path or self.index_path
        if not save_path:
            logger.warning("No save path specified, skipping save")
            return
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, save_path)
        
        # Save metadata
        meta_path = save_path + '.meta'
        with open(meta_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        logger.info(f"Saved FAISS index to {save_path} ({len(self.metadata)} entries)")
    
    def load(self, path: Optional[str] = None):
        """Load index and metadata from disk."""
        load_path = path or self.index_path
        if not load_path or not os.path.exists(load_path):
            logger.warning(f"Index not found at {load_path}")
            return
        
        # Load FAISS index
        self.index = faiss.read_index(load_path)
        
        # Load metadata
        meta_path = load_path + '.meta'
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                self.metadata = pickle.load(f)
        
        logger.info(f"Loaded FAISS index from {load_path} ({len(self.metadata)} entries)")
    
    def clear(self):
        """Clear all entries from the index."""
        self.index = faiss.IndexFlatIP(self.dim)
        self.metadata = []
        logger.info("Cleared FAISS index")
    
    def size(self) -> int:
        """Return number of entries in the index."""
        return self.index.ntotal


class LLMCache:
    """
    Cache for LLM outputs to avoid redundant API calls.
    Uses JSON file storage for simplicity.
    """
    
    def __init__(self, cache_path: str = "./cache/llm_cache.json"):
        self.cache_path = cache_path
        self.cache = {}
        
        if os.path.exists(cache_path):
            self.load()
    
    def get(self, key: str) -> Optional[str]:
        """Get cached LLM output."""
        return self.cache.get(key)
    
    def set(self, key: str, value: str):
        """Cache LLM output."""
        self.cache[key] = value
        self.save()
    
    def has(self, key: str) -> bool:
        """Check if key exists in cache."""
        return key in self.cache
    
    def save(self):
        """Save cache to disk."""
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)
    
    def load(self):
        """Load cache from disk."""
        try:
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                self.cache = json.load(f)
            logger.info(f"Loaded LLM cache with {len(self.cache)} entries")
        except Exception as e:
            logger.warning(f"Failed to load LLM cache: {e}")
            self.cache = {}
    
    def clear(self):
        """Clear all cache entries."""
        self.cache = {}
        self.save()
        logger.info("Cleared LLM cache")
    
    @staticmethod
    def make_key(prompt: str, **kwargs) -> str:
        """Create cache key from prompt and parameters."""
        import hashlib
        params = json.dumps(kwargs, sort_keys=True)
        content = f"{prompt}|{params}"
        return hashlib.md5(content.encode()).hexdigest()
