import logging
from typing import Optional

from llama_index.llms.openai import OpenAI
from llama_index.llms.dashscope import DashScope
from dashscope import MultiModalConversation
import dashscope

from src.config import settings

# Configure logging
logger = logging.getLogger(__name__)

class VLMClient:
    """
    Custom VLM client using DashScope's MultiModalConversation for image analysis.
    """
    def __init__(self):
        dashscope.api_key = settings.ALI_API_KEY

    def analyze_image(self, image_path: str, prompt: str) -> str:
        """
        Analyze an image using the VLM model and return semantic understanding.
        
        Args:
            image_path (str): Path to the image file.
            prompt (str): Text prompt for analysis.
        
        Returns:
            str: The analysis result text.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"image": f"file://{image_path}"}, 
                    {"text": prompt}
                ]
            }
        ]
        
        response = MultiModalConversation.call(
            model=settings.VLM_MODEL_NAME, 
            messages=messages
        )
        
        if response.status_code == 200:
            return response.output.choices[0].message.content[0]['text']
        else:
            raise Exception(f"VLM call failed: {response.message}")

_llm: Optional[DashScope] = None


def get_llm() -> DashScope:
    """
    Returns a singleton instance of the DeepSeek-V3 LLM (OpenAI-compatible).
    """
    global _llm
    if _llm is None:
        try:
            logger.info(f"Initializing LLM Model {settings.LLM_MODEL_NAME}...")
            # DeepSeek V3 is compatible with OpenAI SDK
            _llm = DashScope(
                model=settings.LLM_MODEL_NAME,  # Standard model name for DeepSeek V3 API
                api_key=settings.ALI_API_KEY,
                api_base=settings.ALI_API_URL,
                temperature=0.1,  # Low temperature for technical documentation
                max_tokens=4096,
            )
            logger.info("LLM Model initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise e

    return _llm


_vlm_client: Optional[VLMClient] = None


def get_vlm() -> VLMClient:
    """
    Returns a singleton instance of the Vision Language Model client.
    """
    global _vlm_client
    if _vlm_client is None:
        try:
            logger.info(f"Initializing VLM Model {settings.VLM_MODEL_NAME}...")
            _vlm_client = VLMClient()
            logger.info("VLM Model initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize VLM: {e}")
            raise e

    return _vlm_client


# Reranker (OpenAI-compatible API client)
_rerank: Optional[OpenAI] = None


def get_rerank_model() -> OpenAI:
    """
    Returns a singleton instance of the Reranker model client (OpenAI-compatible).

    说明:
    - 使用配置中的 `RERANKER_MODEL_NAME` 作为模型名，例如 `gte-rerank-v2`
    - 通过阿里 DashScope 的 OpenAI 兼容接口进行调用 (`ALI_API_URL` + `ALI_API_KEY`)
    - 与 `get_llm()/get_vlm()` 相同的初始化与缓存策略
    """
    global _rerank
    if _rerank is None:
        try:
            logger.info(f"Initializing Rerank Model {settings.RERANKER_MODEL_NAME}...")
            _rerank = OpenAI(
                model=settings.RERANKER_MODEL_NAME,
                api_key=settings.ALI_API_KEY,
                api_base=settings.ALI_API_URL,
                temperature=0.0,
                max_tokens=1024,
            )
            logger.info("Rerank Model initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Rerank Model: {e}")
            raise e

    return _rerank
