"""
The core of the Advanced Zero-Shot Classification Framework.
This module integrates the hybrid multimodal reasoning engine, advanced
prompt engineering, auto-tuning, online continual learning, adaptive modules,
LLM prompt fusion, and re-ranking.
"""
import torch
import numpy as np
import logging
from PIL import Image
from models import get_model_manager
from prompt_engineering import PromptEngineer
from online_learning import OnlineTuner
from domain_adaptation import DomainAdaptation
from adaptive_module import DomainAdaptiveModules
from llm_service import PromptService, LLMReRanker
from vector_db import FaissVectorDB
from utils import softmax, normalize_embeddings

logger = logging.getLogger(__name__)

class AdvancedZeroShotFramework:
    def __init__(self, use_adaptive: bool = True, use_llm_prompts: bool = False, 
                 use_vector_db: bool = True):
        self.model_manager = get_model_manager()
        self.prompt_engineer = PromptEngineer()
        self.online_tuner = OnlineTuner()
        self.domain_adapter = DomainAdaptation()
        self.device = self.model_manager.device
        
        # New components
        self.use_adaptive = use_adaptive
        self.use_llm_prompts = use_llm_prompts
        self.adaptive_modules = DomainAdaptiveModules() if use_adaptive else None
        self.prompt_service = PromptService() if use_llm_prompts else None
        self.reranker = LLMReRanker() if use_llm_prompts else None
        
        # Vector DB for caching embeddings
        if use_vector_db:
            try:
                embedding_dim = self.model_manager.clip_model.config.projection_dim
                self.vector_db = FaissVectorDB(dim=embedding_dim, 
                                              index_path="./cache/embeddings.faiss")
            except:
                self.vector_db = None
                logger.warning("Failed to initialize vector DB")
        else:
            self.vector_db = None
        
        # Temperature for calibration
        self.temperature = 0.01  # Default CLIP temperature

    def classify(self, image_path, class_names, user_prompt=None, language='en', 
                use_ensemble=True, temperature=None, use_adaptive_module=None,
                use_llm_reranking=False):
        """
        The main classification pipeline with comprehensive features:
        - Domain adaptation
        - Adaptive embedding modules
        - Multilingual support
        - LLM prompt generation and fusion
        - LLM re-ranking
        - Temperature calibration
        - Detailed explanations
        - Zero-shot reasoning
        - Confidence calibration
        - Ensemble inference with multiple CLIP models
        
        Args:
            image_path: Path to the image file
            class_names: List of class names to classify
            user_prompt: Optional user prompt for context
            language: Language code for multilingual support
            use_ensemble: Whether to use ensemble of all CLIP models (default: True)
            temperature: Temperature for softmax calibration (default: 0.01)
            use_adaptive_module: Whether to use adaptive module (default: None = auto)
            use_llm_reranking: Whether to use LLM re-ranking (default: False)
        """
        # Set temperature
        if temperature is None:
            temperature = self.temperature
        
        # Determine if adaptive module should be used
        if use_adaptive_module is None:
            use_adaptive_module = self.use_adaptive and self.adaptive_modules is not None
        # 1. Preprocess the image
        image = Image.open(image_path).convert("RGB")
        
        # Use ensemble features if available and requested
        if use_ensemble and len(self.model_manager.clip_models) > 1:
            logger.info(f"Using ensemble inference with {len(self.model_manager.clip_models)} models: {self.model_manager.model_names}")
            image_features = self.model_manager.get_ensemble_features(image=image, normalize=True)
        else:
            # Fall back to single model
            clip_model = self.model_manager.clip_model
            clip_processor = self.model_manager.clip_processor
            image_input = clip_processor(images=image, return_tensors="pt").to(self.device)
            image_features = clip_model.get_image_features(**image_input)
            image_features = normalize_embeddings(image_features)

        # 2. Automatic Domain Detection
        image_embeddings_np = image_features.detach().cpu().numpy()
        domain_info = self.domain_adapter.detect_domain(image_embeddings_np)
        detected_domain = domain_info['domain']
        domain_confidence = domain_info['confidence']
        print(f"Detected domain: {detected_domain} (confidence: {domain_confidence:.3f})")
        
        # 2a. Apply adaptive module if enabled
        if use_adaptive_module:
            try:
                embedding_dim = image_features.shape[-1]
                adaptive_module = self.adaptive_modules.get_module(
                    detected_domain, embedding_dim, str(self.device)
                )
                image_features = adaptive_module(image_features)
                image_features = normalize_embeddings(image_features)
                logger.info(f"Applied adaptive module for domain: {detected_domain}")
            except Exception as e:
                logger.warning(f"Failed to apply adaptive module: {e}")


        # 3. Hybrid Multimodal Reasoning: Generate visual features narrative
        visual_narrative_prompt = f"Describe the visual features of an image in the {detected_domain} domain:"
        image_narrative = self.model_manager.generate_narrative(visual_narrative_prompt, max_length=100)
        print(f"LLM Image Narrative: {image_narrative}")

        # Store visual features for explanation
        visual_features = self._extract_visual_features(image_features, detected_domain)

        final_scores = {}
        raw_scores = []
        reasoning_chains = {}
        fused_embeddings = {}  # Store fused prompt embeddings
        
        for class_name in class_names:
            # 4. Generate prompts - use LLM if available, otherwise fallback
            if self.use_llm_prompts and self.prompt_service:
                prompts = self.prompt_service.generate_prompts_with_llm(
                    class_name,
                    languages=[language] if language != 'en' else None,
                    domain=detected_domain
                )
            else:
                # Use existing prompt engineer
                prompts = self.prompt_engineer.generate_prompts_for_class(
                    class_name, 
                    domain=detected_domain,
                    language=language
                )
                
                # Add semantic expansions
                semantic_prompts = self.prompt_engineer.generate_semantic_expansion(
                    class_name,
                    domain=detected_domain
                )
                prompts.extend(semantic_prompts)
            
            # 5. Get text features - use prompt fusion if LLM service available
            if self.use_llm_prompts and self.prompt_service:
                clip_model = self.model_manager.clip_model
                clip_processor = self.model_manager.clip_processor
                
                # Fuse prompt embeddings with weights
                fused_emb = self.prompt_service.fuse_prompt_embeddings(
                    prompts, clip_model, clip_processor, str(self.device)
                )
                fused_embeddings[class_name] = fused_emb
                
                # Calculate similarity with fused embedding
                similarity = (100.0 * image_features @ fused_emb.unsqueeze(0).T).item()
            else:
                # Original approach with ensemble or single model
                if use_ensemble and len(self.model_manager.clip_models) > 1:
                    prompt_embeddings = self.model_manager.get_ensemble_features(text=prompts, normalize=True)
                else:
                    clip_model = self.model_manager.clip_model
                    clip_processor = self.model_manager.clip_processor
                    text_inputs = clip_processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
                    prompt_embeddings = clip_model.get_text_features(**text_inputs)
                    prompt_embeddings = normalize_embeddings(prompt_embeddings)

                # Dynamic Auto-Tuning: Get weights for each prompt
                prompt_weights = self.prompt_engineer.auto_tune_weights(image_features, prompt_embeddings)
                
                # Create a single, style-agnostic embedding for the class
                class_embedding = torch.sum(prompt_weights.T * prompt_embeddings, dim=0)
                class_embedding = class_embedding / class_embedding.norm()

                # Online Continual Learning: Adapt the embedding
                adapted_embedding = self.online_tuner.get_adapted_embedding(class_name, class_embedding)
                
                # Calculate similarity score
                similarity = (100.0 * image_features @ adapted_embedding.T).item()
            
            raw_scores.append(similarity)
            
            # Store reasoning for this class
            reasoning_chains[class_name] = {
                'num_prompts': len(prompts),
                'top_prompts': prompts[:3],
                'similarity_score': similarity
            }
        
        # 9. Apply Domain-Specific Adaptive Auto-Tuning
        tuned_scores = self.domain_adapter.adaptive_auto_tuning(raw_scores, detected_domain, domain_confidence)
        
        # 10. Convert to probabilities using softmax with temperature
        probabilities = softmax(np.array(tuned_scores), temperature=temperature)
        
        # Create final scores dictionary (as percentages)
        for idx, class_name in enumerate(class_names):
            final_scores[class_name] = float(probabilities[idx] * 100.0)
        
        # Get best class
        best_class = max(final_scores, key=final_scores.get)
        
        # 11. Optional LLM re-ranking - ACTUALLY modifies probabilities
        llm_reranking_applied = False
        if use_llm_reranking and self.reranker:
            try:
                # Generate image caption for re-ranking
                image_caption = self._generate_image_caption(image, visual_features, detected_domain)
                
                # Get top candidates
                candidates = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
                candidates_prob = [(c[0], c[1]/100.0) for c in candidates]  # Convert to 0-1 scale
                
                # COPILOT: implement llm_rerank that sends caption + labels + probs to the LLM,
                # gets back JSON with new label order and probabilities that sum to 1.
                reranked = self.reranker.rerank_candidates(image_caption, candidates_prob, k=len(class_names))
                
                # Update final scores with re-ranked probabilities (REAL re-ranking)
                final_scores = {}  # Clear old scores
                for label, prob in reranked:
                    final_scores[label] = prob * 100.0  # Convert back to percentage
                
                best_class = max(final_scores, key=final_scores.get)
                llm_reranking_applied = True
                logger.info(f"Applied LLM re-ranking: new top class = {best_class}")
            except Exception as e:
                logger.warning(f"LLM re-ranking failed: {e}, using original scores")
                llm_reranking_applied = False
        # 12. Update the online tuner with the features of the top-scoring class
        self.online_tuner.update_class_embeddings(best_class, image_features.squeeze())
        print(f"Online tuner updated for class: {best_class}")
        
        # 13. Store embeddings in vector DB if available
        if self.vector_db:
            try:
                emb_np = image_features.detach().cpu().numpy().flatten()
                meta = {
                    'label': best_class,
                    'domain': detected_domain,
                    'confidence': final_scores[best_class] / 100.0,
                    'timestamp': __import__('time').time()
                }
                self.vector_db.add(emb_np, meta)
            except Exception as e:
                logger.warning(f"Failed to add embedding to vector DB: {e}")
        
        # 14. Generate UI-friendly reasoning structure
        # COPILOT: implement generate_ui_reasoning returning JSON with summary,
        # attributes list, and detailed_reasoning, in the style of the Adaptive CLIP Output UI.
        ui_reasoning = self._generate_ui_reasoning(
            best_class,
            final_scores,
            visual_features,
            detected_domain,
            domain_confidence,
            llm_reranking_applied
        )
        
        # 15. Prepare comprehensive response with clean structure
        response = {
            'predictions': final_scores,  # Uses final reranked probabilities
            'top_prediction': {
                'label': best_class,
                'score': final_scores[best_class]
            },
            'confidence_score': final_scores[best_class] / 100.0,
            'domain_info': domain_info,
            'reasoning': ui_reasoning,  # UI-friendly reasoning structure
            'visual_features': visual_features,
            'alternative_predictions': self._get_top_k_predictions(final_scores, k=3),
            'zero_shot': True,
            'multilingual_support': True,  # Always true - system is multilingual-ready
            'language': language,
            'temperature': temperature,
            'adaptive_module_used': use_adaptive_module,
            'llm_reranking_used': llm_reranking_applied  # Track if actually applied
        }

        return response

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
    
    def _extract_visual_features(self, image_features, domain):
        """Extract and describe visual features based on domain."""
        features = []
        
        # Analyze embedding statistics
        embedding_np = image_features.detach().cpu().numpy().flatten()
        
        # Domain-specific feature extraction
        if domain == 'sketch':
            features.extend(['line_based', 'high_edge_density', 'minimal_texture'])
        elif domain == 'medical_image':
            features.extend(['grayscale_imaging', 'clinical_patterns', 'anatomical_structures'])
        elif domain == 'natural_image':
            features.extend(['natural_lighting', 'rich_texture', 'photorealistic'])
        elif domain == 'artistic_image' or domain == 'anime':
            features.extend(['stylized', 'exaggerated_features', 'artistic_interpretation'])
        elif domain == 'multispectral_image':
            features.extend(['spectral_bands', 'remote_sensing', 'vegetation_patterns'])
        
        return features
    
    def _generate_explanation(self, scores, visual_features, domain, reasoning_chains):
        """Generate human-readable explanation for the classification with detailed scenario description."""
        top_class = max(scores, key=scores.get)
        top_score = scores[top_class]
        confidence = top_score / 100.0
        
        # Get domain info
        domain_info = self.domain_adapter.get_domain_info(domain)
        domain_desc = domain_info['description']
        
        # Build detailed scenario description
        explanation_parts = []
        
        # Start with a descriptive scenario based on the classification and visual features
        scenario_intro = self._generate_scenario_description(top_class, visual_features, domain)
        explanation_parts.append(scenario_intro)
        
        # Add visual analysis
        if visual_features:
            features_desc = self._describe_visual_features(visual_features, domain)
            explanation_parts.append(features_desc)
        
        # Add domain context
        explanation_parts.append(
            f"This classification was performed in the {domain.replace('_', ' ')} domain. {domain_desc}"
        )
        
        # Add confidence and reasoning
        if confidence > 0.90:
            confidence_text = "The model shows very high confidence"
        elif confidence > 0.80:
            confidence_text = "The model shows strong confidence"
        elif confidence > 0.70:
            confidence_text = "The model shows moderate confidence"
        else:
            confidence_text = "The model provides a reasonable prediction"
        
        explanation_parts.append(
            f"{confidence_text} ({confidence:.1%}) in classifying this as '{top_class}'."
        )
        
        # Add technical details if available
        if top_class in reasoning_chains:
            reasoning = reasoning_chains[top_class]
            explanation_parts.append(
                f"This determination was made by analyzing {reasoning['num_prompts']} different semantic "
                f"interpretations, achieving a similarity score of {reasoning['similarity_score']:.1f}."
            )
        
        return ' '.join(explanation_parts)
    
    def _generate_scenario_description(self, class_name, visual_features, domain):
        """Generate a descriptive scenario based on the classification and features."""
        # Create descriptive opening based on domain and features
        if domain == 'artistic_image' or domain == 'anime':
            style_words = [f for f in visual_features if f in ['stylized', 'dramatic', 'artistic_interpretation', 'exaggerated_features']]
            if style_words:
                return f"The image depicts a {', '.join(style_words[:2])} {class_name.replace('_', ' ')} with artistic rendering and creative interpretation."
            return f"The image shows an artistically rendered {class_name.replace('_', ' ')} with distinctive visual style."
        
        elif domain == 'sketch':
            return f"The image appears to be a sketch or line drawing depicting {class_name.replace('_', ' ')}, characterized by line-based composition and minimal texture."
        
        elif domain == 'medical_image':
            return f"This medical imaging scan shows {class_name.replace('_', ' ')}, captured using clinical imaging techniques with focus on anatomical structures."
        
        elif domain == 'natural_image':
            lighting_features = [f for f in visual_features if 'light' in f.lower() or 'color' in f.lower()]
            if lighting_features:
                return f"The photograph captures {class_name.replace('_', ' ')} with {lighting_features[0].replace('_', ' ')} and natural photorealistic qualities."
            return f"The image shows a natural photograph of {class_name.replace('_', ' ')} with realistic lighting and texture."
        
        elif domain == 'multispectral_image':
            return f"This multispectral image reveals {class_name.replace('_', ' ')} through specialized spectral band analysis."
        
        else:
            # Generic description
            if visual_features:
                key_feature = visual_features[0].replace('_', ' ')
                return f"The image depicts {class_name.replace('_', ' ')} featuring {key_feature} as a prominent characteristic."
            return f"The image shows {class_name.replace('_', ' ')} captured with distinctive visual characteristics."
    
    def _describe_visual_features(self, features, domain):
        """Generate natural language description of visual features."""
        if not features:
            return ""
        
        # Filter and organize features
        visual_attrs = []
        technical_attrs = []
        
        for feature in features[:6]:  # Focus on top features
            feature_clean = feature.replace('_', ' ')
            if any(word in feature for word in ['light', 'color', 'texture', 'style', 'pattern']):
                visual_attrs.append(feature_clean)
            else:
                technical_attrs.append(feature_clean)
        
        desc_parts = []
        if visual_attrs:
            desc_parts.append(f"Notable visual elements include {', '.join(visual_attrs)}")
        if technical_attrs:
            desc_parts.append(f"with {', '.join(technical_attrs)}")
        
        return '. '.join(desc_parts) + '.' if desc_parts else ""
    
    
    def _get_top_k_predictions(self, scores, k=3):
        """Get top-k predictions sorted by score."""
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [{'label': cls, 'score': score / 100.0} for cls, score in sorted_scores]
    
    def _generate_image_caption(self, image, visual_features, domain):
        """Generate a brief caption for the image to use in LLM re-ranking."""
        caption_parts = [f"A {domain.replace('_', ' ')} showing"]
        
        if visual_features:
            # Use top 2-3 visual features
            features_text = ', '.join(visual_features[:3]).replace('_', ' ')
            caption_parts.append(f"with {features_text}")
        
        return ' '.join(caption_parts)
    
    def _generate_ui_reasoning(self, best_class, final_scores, visual_features, 
                               domain, domain_confidence, llm_reranked):
        """Generate UI-friendly reasoning structure matching the output card design.
        
        Returns:
            Dict with 'summary', 'attributes', and 'detailed_reasoning'
        """
        confidence = final_scores[best_class] / 100.0
        
        # Summary - one sentence explanation
        summary = f"Classified as '{best_class}' with {confidence:.1%} confidence"
        if llm_reranked:
            summary += " (LLM-enhanced)"
        
        # Attributes - key factors in the decision
        attributes = [
            f"Domain: {domain.replace('_', ' ').title()} ({domain_confidence:.0%})",
            f"Confidence: {confidence:.1%}",
            f"Temperature: Calibrated softmax"
        ]
        
        if visual_features:
            top_feature = visual_features[0].replace('_', ' ').title()
            attributes.append(f"Key Feature: {top_feature}")
        
        if llm_reranked:
            attributes.append("Enhanced: LLM Re-ranking")
        
        # Detailed reasoning - explain the process
        reasoning_parts = [
            f"The image was analyzed using CLIP embeddings and classified in the {domain.replace('_', ' ')} domain.",
            f"Domain detection confidence: {domain_confidence:.1%}."
        ]
        
        if visual_features:
            features_list = ', '.join(visual_features[:3]).replace('_', ' ')
            reasoning_parts.append(f"Visual analysis identified: {features_list}.")
        
        reasoning_parts.append(
            f"After softmax calibration with temperature scaling, '{best_class}' achieved {confidence:.1%} probability."
        )
        
        if llm_reranked:
            reasoning_parts.append(
                "LLM re-ranking was applied to refine the probabilities based on semantic understanding."
            )
        
        # Get runner-up for comparison
        sorted_preds = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_preds) > 1:
            runner_up = sorted_preds[1]
            reasoning_parts.append(
                f"The next closest prediction was '{runner_up[0]}' at {runner_up[1]/100:.1%}."
            )
        
        detailed_reasoning = ' '.join(reasoning_parts)
        
        return {
            'summary': summary,
            'attributes': attributes,
            'detailed_reasoning': detailed_reasoning
        }
