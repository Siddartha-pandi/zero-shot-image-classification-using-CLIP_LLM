# backend/models/llm_model.py
import os
import google.generativeai as genai
import logging
from dotenv import load_dotenv
from typing import List

logger = logging.getLogger(__name__)

class LLMModel:
    def __init__(self):
        self.model = None
        self.model_name = None
        self._configure()

    @staticmethod
    def _candidate_models() -> List[str]:
        preferred = os.getenv("GEMINI_MODEL", "").strip()
        ordered = [
            preferred,
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-flash-latest",
            "gemini-1.5-flash",
        ]
        # Deduplicate while preserving order and dropping empty values
        return [name for i, name in enumerate(ordered) if name and name not in ordered[:i]]

    def _resolve_supported_model(self) -> str:
        candidates = self._candidate_models()

        try:
            available = set()
            for model in genai.list_models():
                methods = getattr(model, "supported_generation_methods", []) or []
                if "generateContent" in methods:
                    # model.name is usually like "models/gemini-2.0-flash"
                    short_name = model.name.replace("models/", "")
                    available.add(short_name)

            for name in candidates:
                if name in available:
                    return name

            for name in sorted(available):
                if name.startswith("gemini"):
                    logger.warning(
                        "No preferred GEMINI_MODEL available. Falling back to available model: %s",
                        name,
                    )
                    return name
        except Exception as e:
            logger.warning("Failed to enumerate Gemini models, using static fallback list: %s", e)

        return candidates[0]

    def _configure(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                genai.configure(api_key=api_key)
                model_name = self._resolve_supported_model()
                self.model = genai.GenerativeModel(model_name)
                self.model_name = model_name
                logger.info("✓ Gemini API configured successfully with model: %s", model_name)
            except Exception as e:
                logger.warning(f"Failed to configure Gemini API: {e}")
        else:
            logger.warning("GEMINI_API_KEY not found - LLM features will use fallbacks")

    def generate(self, prompt: str, temperature: float = 0.3, max_tokens: int = 300) -> str:
        if self.model is None:
            raise ValueError("LLM is not configured")
        
        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )
        )
        return response.text.strip()

_llm_model = None
def get_llm_model() -> LLMModel:
    global _llm_model
    if _llm_model is None:
        _llm_model = LLMModel()
    return _llm_model
