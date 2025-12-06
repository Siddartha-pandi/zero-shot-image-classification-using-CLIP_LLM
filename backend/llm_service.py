"""
LLM-based prompt generation, fusion, and re-ranking services.
"""
import torch
import logging
from typing import List, Dict, Tuple, Optional
import json
import numpy as np
from vector_db import LLMCache

logger = logging.getLogger(__name__)


class PromptService:
    """
    Service for generating prompts using LLM and fusing prompt embeddings.
    """
    
    def __init__(self, llm_client=None, use_cache: bool = True):
        """
        Args:
            llm_client: LLM client (e.g., OpenAI, local model)
            use_cache: Whether to cache LLM outputs
        """
        self.llm_client = llm_client
        self.cache = LLMCache() if use_cache else None
    
    def generate_prompts_with_llm(self, label: str, languages: List[str] = None, 
                                  domain: str = None, num_prompts: int = 6) -> List[str]:
        """
        Use LLM to generate multiple prompt variants for a class label.
        
        Args:
            label: Class label
            languages: List of languages for prompts
            domain: Domain context (e.g., 'medical', 'sketch')
            num_prompts: Number of prompts to generate
        
        Returns:
            List of prompt strings
        """
        # Build prompt for LLM
        base_prompt = f"Generate {num_prompts} concise descriptive prompts for the class '{label}'."
        
        if domain:
            base_prompt += f" Use domain '{domain}' and include synonyms and common descriptors."
        
        if languages:
            base_prompt += f" Provide prompts in these languages: {', '.join(languages)}."
        
        base_prompt += "\nReturn prompts as a JSON array of strings."
        
        # Check cache
        cache_key = None
        if self.cache:
            cache_key = LLMCache.make_key(base_prompt, label=label, domain=domain, languages=languages)
            cached = self.cache.get(cache_key)
            if cached:
                logger.debug(f"Using cached prompts for {label}")
                return json.loads(cached)
        
        # Generate using LLM
        if self.llm_client is None:
            # Fallback: generate simple prompts without LLM
            logger.warning("No LLM client provided, using fallback prompt generation")
            prompts = self._generate_fallback_prompts(label, domain, num_prompts)
        else:
            try:
                resp = self.llm_client.generate(base_prompt)
                # Try to parse JSON response
                prompts = json.loads(resp)
                if not isinstance(prompts, list):
                    prompts = [p.strip() for p in resp.split('\n') if p.strip()]
            except Exception as e:
                logger.error(f"LLM prompt generation failed: {e}, using fallback")
                prompts = self._generate_fallback_prompts(label, domain, num_prompts)
        
        # Cache result
        if self.cache and cache_key:
            self.cache.set(cache_key, json.dumps(prompts))
        
        return prompts[:num_prompts]
    
    def _generate_fallback_prompts(self, label: str, domain: str = None, 
                                   num_prompts: int = 6) -> List[str]:
        """Generate simple prompts without LLM."""
        templates = [
            f"a photo of a {label}",
            f"an image of a {label}",
            f"{label}",
            f"a picture showing {label}",
            f"{label} in the image",
            f"this is a {label}"
        ]
        
        if domain:
            templates.extend([
                f"a {domain} image of {label}",
                f"{label} in {domain} style"
            ])
        
        return templates[:num_prompts]
    
    def fuse_prompt_embeddings(self, prompt_texts: List[str], clip_model, 
                               clip_processor, device: str, 
                               weights: Optional[List[float]] = None) -> torch.Tensor:
        """
        Encode prompts and produce a weighted fused embedding.
        
        Args:
            prompt_texts: List of prompt strings
            clip_model: CLIP model
            clip_processor: CLIP processor
            device: Device to run on
            weights: Optional weights for each prompt (will be normalized)
        
        Returns:
            Fused embedding tensor of shape (dim,)
        """
        if not prompt_texts:
            raise ValueError("prompt_texts cannot be empty")
        
        with torch.no_grad():
            # Tokenize and encode all prompts
            text_inputs = clip_processor(text=prompt_texts, return_tensors="pt", 
                                        padding=True, truncation=True).to(device)
            embs = clip_model.get_text_features(**text_inputs)
            
            # Normalize embeddings
            embs = embs / embs.norm(dim=-1, keepdim=True)
            
            # Apply weights
            if weights is None:
                weights = torch.ones(embs.size(0), device=device) / embs.size(0)
            else:
                w = torch.tensor(weights, device=device, dtype=embs.dtype)
                weights = w / w.sum()
            
            # Weighted fusion
            fused = (weights.unsqueeze(1) * embs).sum(dim=0)
            fused = fused / fused.norm()
            
            return fused


class LLMReRanker:
    """
    LLM-based re-ranking service for classification candidates.
    """
    
    def __init__(self, llm_client=None, use_cache: bool = True):
        self.llm_client = llm_client
        self.cache = LLMCache() if use_cache else None
    
    def rerank_candidates(self, image_caption: str, candidates: List[Tuple[str, float]], 
                         k: int = 5) -> List[Tuple[str, float]]:
        """
        Ask LLM to reorder and provide reasoning for top candidates.
        
        Args:
            image_caption: Description of the image
            candidates: List of (label, score) tuples
            k: Number of top candidates to rerank
        
        Returns:
            Reranked list of (label, probability) tuples
        """
        if not self.llm_client:
            logger.warning("No LLM client provided, returning original candidates")
            return candidates[:k]
        
        # Build prompt
        cand_str = "\n".join([f"{i+1}. {c[0]} (score {c[1]:.4f})" 
                              for i, c in enumerate(candidates[:k])])
        
        prompt = f"""You are a helpful assistant for image classification. Given the image description below and candidate labels,
re-rank the candidates and assign each a probability (sum to 1). Provide a short rationale.

Image description:
{image_caption}

Candidates:
{cand_str}

Return JSON array like: [{{"label":"...", "prob":0.42, "reason":"..."}}, ...]
"""
        
        # Check cache
        cache_key = None
        if self.cache:
            cache_key = LLMCache.make_key(prompt, caption=image_caption, 
                                         candidates=[c[0] for c in candidates[:k]])
            cached = self.cache.get(cache_key)
            if cached:
                logger.debug("Using cached LLM reranking")
                try:
                    parsed = json.loads(cached)
                    return [(p['label'], float(p['prob'])) for p in parsed]
                except:
                    pass
        
        # Call LLM
        try:
            resp = self.llm_client.generate(prompt)
            parsed = json.loads(resp)
            
            # Cache result
            if self.cache and cache_key:
                self.cache.set(cache_key, resp)
            
            return [(p['label'], float(p['prob'])) for p in parsed]
        
        except Exception as e:
            logger.error(f"LLM re-ranking failed: {e}, returning original candidates")
            return candidates[:k]
    
    def generate_explanation(self, top_label: str, probability: float, 
                           image_caption: str, reasoning: str = None) -> str:
        """
        Generate detailed explanation for the classification result.
        
        Args:
            top_label: Top predicted label
            probability: Prediction probability
            image_caption: Image description
            reasoning: Optional reasoning chain
        
        Returns:
            Human-readable explanation
        """
        if not self.llm_client:
            return f"The image is classified as '{top_label}' with {probability:.1%} confidence."
        
        prompt = f"""Provide a concise explanation for why an image was classified as '{top_label}' 
with {probability:.1%} confidence. Image description: {image_caption}"""
        
        if reasoning:
            prompt += f"\nReasoning chain: {reasoning}"
        
        try:
            explanation = self.llm_client.generate(prompt)
            return explanation
        except Exception as e:
            logger.error(f"LLM explanation generation failed: {e}")
            return f"The image is classified as '{top_label}' with {probability:.1%} confidence based on visual similarity."
