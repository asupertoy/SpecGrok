from src.models.llm import get_llm
from src.models.embedding import get_embed_model

# 提供按需获取 LLM/Embedding 的接口，避免在模块 import 时初始化外部客户端。
def get_llm_instance():
    return get_llm()

def get_embed_instance():
    return get_embed_model()


