# backend/models/llm_model.py
import os
import google.generativeai as genai
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class LLMModel:
    def __init__(self):
        self.model = None
        self._configure()

    def _configure(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                logger.info("✓ Gemini API configured successfully")
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
