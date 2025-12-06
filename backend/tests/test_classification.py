"""
Unit tests for the classification pipeline components.
"""
import pytest
import numpy as np
import torch
from utils import softmax, topk_from_probs, calibrate_temperature, normalize_embeddings


class TestMathUtils:
    """Tests for mathematical utility functions."""
    
    def test_softmax_sum_to_one(self):
        """Test that softmax probabilities sum to 1."""
        logits = np.array([2.0, 1.0, 0.1])
        probs = softmax(logits, temperature=1.0)
        assert pytest.approx(probs.sum(), rel=1e-6) == 1.0
    
    def test_softmax_temperature_scaling(self):
        """Test temperature scaling in softmax."""
        logits = np.array([2.0, 1.0, 0.1])
        
        # Higher temperature should give more uniform distribution
        probs_high_temp = softmax(logits, temperature=10.0)
        probs_low_temp = softmax(logits, temperature=0.1)
        
        # High temp should be more uniform (lower max prob)
        assert probs_high_temp.max() < probs_low_temp.max()
    
    def test_softmax_numerical_stability(self):
        """Test softmax with large values."""
        logits = np.array([1000.0, 1001.0, 999.0])
        probs = softmax(logits, temperature=1.0)
        
        assert pytest.approx(probs.sum(), rel=1e-6) == 1.0
        assert not np.isnan(probs).any()
        assert not np.isinf(probs).any()
    
    def test_topk_selection(self):
        """Test top-k selection from probabilities."""
        labels = ["cat", "dog", "bird", "fish"]
        probs = np.array([0.7, 0.1, 0.15, 0.05])
        
        top = topk_from_probs(probs, labels, k=2)
        
        assert len(top) == 2
        assert top[0][0] == "cat"  # Highest probability
        assert top[0][1] == pytest.approx(0.7)
        assert top[1][0] == "bird"  # Second highest
    
    def test_topk_with_ties(self):
        """Test top-k with tied probabilities."""
        labels = ["a", "b", "c", "d"]
        probs = np.array([0.4, 0.4, 0.1, 0.1])
        
        top = topk_from_probs(probs, labels, k=3)
        assert len(top) == 3
        
        # First two should be the tied values
        assert top[0][1] == pytest.approx(0.4)
        assert top[1][1] == pytest.approx(0.4)
    
    def test_normalize_embeddings(self):
        """Test L2 normalization of embeddings."""
        # Test 2D tensor
        emb_2d = torch.randn(4, 512)
        norm_2d = normalize_embeddings(emb_2d)
        
        # Check L2 norm is 1
        norms = torch.norm(norm_2d, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
        
        # Test 1D tensor
        emb_1d = torch.randn(512)
        norm_1d = normalize_embeddings(emb_1d)
        
        assert pytest.approx(torch.norm(norm_1d).item(), rel=1e-6) == 1.0


class TestCalibration:
    """Tests for temperature calibration."""
    
    def test_calibrate_temperature(self):
        """Test temperature calibration finds reasonable value."""
        # Create mock logits and labels
        n_samples = 100
        n_classes = 5
        
        logits = np.random.randn(n_samples, n_classes) * 2
        true_labels = np.random.randint(0, n_classes, n_samples)
        
        optimal_temp = calibrate_temperature(logits, true_labels)
        
        # Should return a positive temperature
        assert optimal_temp > 0
        assert optimal_temp in [0.01, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0]


class TestAdaptiveModule:
    """Tests for adaptive embedding module."""
    
    @pytest.fixture
    def adaptive_module(self):
        """Create a test adaptive module."""
        from adaptive_module import AdaptiveModule
        return AdaptiveModule(dim=512, hidden=128, use_mlp=True)
    
    def test_forward_pass(self, adaptive_module):
        """Test forward pass through adaptive module."""
        # Test with batch
        x_batch = torch.randn(4, 512)
        out_batch = adaptive_module(x_batch)
        assert out_batch.shape == x_batch.shape
        
        # Test with single vector
        x_single = torch.randn(512)
        out_single = adaptive_module(x_single)
        assert out_single.shape == x_single.shape
    
    def test_save_load(self, adaptive_module, tmp_path):
        """Test saving and loading adaptive module."""
        from adaptive_module import AdaptiveModule
        
        save_path = tmp_path / "test_adapter.pt"
        adaptive_module.save(str(save_path))
        
        loaded_module = AdaptiveModule.load(str(save_path))
        
        # Test that loaded module produces same output
        x = torch.randn(512)
        out1 = adaptive_module(x)
        out2 = loaded_module(x)
        
        assert torch.allclose(out1, out2, atol=1e-6)


class TestVectorDB:
    """Tests for FAISS vector database."""
    
    @pytest.fixture
    def vector_db(self):
        """Create a test vector database."""
        from vector_db import FaissVectorDB
        return FaissVectorDB(dim=512)
    
    def test_add_and_search(self, vector_db):
        """Test adding and searching embeddings."""
        # Add some embeddings
        for i in range(10):
            emb = np.random.randn(512).astype('float32')
            meta = {'label': f'class_{i}', 'index': i}
            vector_db.add(emb, meta)
        
        assert vector_db.size() == 10
        
        # Search for similar embedding
        query = np.random.randn(512).astype('float32')
        results = vector_db.search(query, k=3)
        
        assert len(results) == 3
        assert all(isinstance(r[0], dict) for r in results)
        assert all(isinstance(r[1], float) for r in results)
    
    def test_batch_add(self, vector_db):
        """Test batch adding of embeddings."""
        embs = np.random.randn(20, 512).astype('float32')
        metas = [{'label': f'class_{i}'} for i in range(20)]
        
        vector_db.add_batch(embs, metas)
        assert vector_db.size() == 20
    
    def test_save_load(self, vector_db, tmp_path):
        """Test saving and loading vector DB."""
        from vector_db import FaissVectorDB
        
        # Add some data
        for i in range(5):
            emb = np.random.randn(512).astype('float32')
            vector_db.add(emb, {'label': f'class_{i}'})
        
        save_path = tmp_path / "test_index.faiss"
        vector_db.save(str(save_path))
        
        # Load into new instance
        loaded_db = FaissVectorDB(dim=512, index_path=str(save_path))
        
        assert loaded_db.size() == 5
        assert len(loaded_db.metadata) == 5


class TestLLMCache:
    """Tests for LLM output caching."""
    
    @pytest.fixture
    def llm_cache(self, tmp_path):
        """Create a test LLM cache."""
        from vector_db import LLMCache
        cache_path = tmp_path / "test_cache.json"
        return LLMCache(cache_path=str(cache_path))
    
    def test_get_set(self, llm_cache):
        """Test basic get and set operations."""
        key = "test_prompt_123"
        value = "Generated response"
        
        llm_cache.set(key, value)
        assert llm_cache.has(key)
        assert llm_cache.get(key) == value
    
    def test_make_key(self):
        """Test cache key generation."""
        from vector_db import LLMCache
        
        key1 = LLMCache.make_key("prompt", param1="value1", param2="value2")
        key2 = LLMCache.make_key("prompt", param2="value2", param1="value1")
        key3 = LLMCache.make_key("different", param1="value1", param2="value2")
        
        # Same prompt and params should give same key regardless of order
        assert key1 == key2
        # Different prompt should give different key
        assert key1 != key3
    
    def test_persistence(self, tmp_path):
        """Test cache persistence across instances."""
        from vector_db import LLMCache
        
        cache_path = tmp_path / "persist_cache.json"
        
        # Create cache and add data
        cache1 = LLMCache(cache_path=str(cache_path))
        cache1.set("key1", "value1")
        cache1.set("key2", "value2")
        
        # Create new instance and verify data persisted
        cache2 = LLMCache(cache_path=str(cache_path))
        assert cache2.get("key1") == "value1"
        assert cache2.get("key2") == "value2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
