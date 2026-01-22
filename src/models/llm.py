import logging
from typing import Optional
from llama_index.llms.openai import OpenAI
from src.config import settings

# Configure logging
logger = logging.getLogger(__name__)

_llm: Optional[OpenAI] = None

def get_llm() -> OpenAI:
    """
    Returns a singleton instance of the DeepSeek-V3 LLM (OpenAI-compatible).
    """
    global _llm
    if _llm is None:
        try:
            logger.info(f"Initializing LLM Model {settings.LLM_MODEL_NAME}...")
            # DeepSeek V3 is compatible with OpenAI SDK
            _llm = OpenAI(
                model=settings.LLM_MODEL_NAME, # Standard model name for DeepSeek V3 API
                api_key=settings.ALI_API_KEY,
                api_base=settings.ALI_API_URL,
                temperature=0.1, # Low temperature for technical documentation
                max_tokens=4096,
            )
            logger.info("LLM Model initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise e
            
    return _llm
