import logging
import sys
import os
import base64

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models.llm import get_llm, get_vlm, get_rerank_model
from src.models.embedding import get_embed_model
from src.config import settings

# For VLM image test
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.base.llms.types import TextBlock, ImageBlock
from llama_index.core.schema import ImageDocument


def test_llm_connection():
    """Test LLM connection by initializing and making a simple chat."""
    try:
        llm = get_llm()
        print("LLM initialized successfully.")
        
        # Test a simple chat
        messages = [
            ChatMessage(role=MessageRole.USER, content="Hello, how are you?")
        ]
        response = llm.chat(messages)
        print(f"LLM Chat Response: {response.message.content[:100]}...")
        print("LLM connection test passed.")
    except Exception as e:
        print(f"LLM connection test failed: {e}")


def test_vlm_native_dashscope():
    vlm = get_vlm()
    # 待解析的 PDF 提取图路径
    image_path = "/home/john/project/SpecGrok/data/cache_vlm/dft_core/p2_fig_0.png"
    
    if not os.path.exists(image_path):
        print(f"错误：找不到图片文件 {image_path}")
        return

    # 通用型 Prompt：适用于 PDF 解析场景
    # 引导模型进行：类型识别 -> 文字提取 -> 逻辑描述
    general_prompt = (
        "你是一个专业的文档解析助手。请分析这张来自 PDF 文档的图片，并完成以下任务：\n"
        "1. 识别图片类型（如：流程图、数学公式、坐标曲线图、架构图等）；\n"
        "2. 提取图片中的所有关键文字、标题以及数学公式（使用 LaTeX 格式）；\n"
        "3. 简要描述图片所表达的核心逻辑或信息内容。"
    )
    
    try:
        print(f"正在使用 {settings.VLM_MODEL_NAME} 解析图片: {os.path.basename(image_path)}...")
        
        result_text = vlm.analyze_image(image_path, general_prompt)
        print("\n" + "="*50)
        print("VLM 通用解析结果:")
        print("-" * 50)
        print(result_text)
        print("="*50)
    except Exception as e:
        print(f"\n运行时抛出异常: {e}")

def test_rerank_connection():
    """Test Reranker connection by initializing."""
    try:
        rerank = get_rerank_model()
        print("Reranker initialized successfully.")
        print("Reranker connection test passed (initialization only).")
    except Exception as e:
        print(f"Reranker connection test failed: {e}")

def test_embedding_connection():
    """Test Embedding model connection by initializing and encoding a sample text."""
    try:
        embed_model = get_embed_model()
        print("Embedding model initialized successfully.")
        
        # Test encoding a simple text
        test_text = "Hello, this is a test for embedding."
        result = embed_model.encode(test_text)
        print(f"Embedding encoded successfully. Dense vecs shape: {len(result.get('dense_vecs', []))}")
        print("Embedding connection test passed.")
    except Exception as e:
        print(f"Embedding connection test failed: {e}")

if __name__ == "__main__":
    print("Starting LLM connection tests...")
    test_llm_connection()
    test_vlm_native_dashscope()
    test_rerank_connection()
    test_embedding_connection()
    print("All tests completed.")