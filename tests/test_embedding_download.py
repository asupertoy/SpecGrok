import sys
import os
import time

# 1. 强制设置国内镜像源 (必须在 import 其他库之前设置)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.embedding import get_embed_model

def test_local_embedding():
    print(f"🚀 开始初始化本地 Embedding 模型 (BGEM3FlagModel)...")
    print(f"🌍 使用镜像源: {os.environ.get('HF_ENDPOINT')}")
    
    start_time = time.time()
    # verify_ssl=False might be needed if there are SSL issues, generally handled by env
    service = get_embed_model()
    load_time = time.time() - start_time
    print(f"✅ 模型加载完成! 耗时: {load_time:.2f}s")

    test_text = "The quick brown fox jumps over the lazy dog."
    print(f"\n🧪 测试生成向量: '{test_text}'")
    
    start_time = time.time()
    output = service.encode(test_text)
    embed_time = time.time() - start_time
    
    dense_vec = output['dense_vecs']
    lexical_weights = output['lexical_weights']

    print(f"✅ 向量生成成功!")
    print(f"   - Dense 维度: {len(dense_vec)} (应为 1024)")
    print(f"   - Sparse (Lexical) 长度: {len(lexical_weights)} (关键词权重)")
    print(f"   - 关键词示例: {list(lexical_weights.keys())[:5]}")
    print(f"   - 耗时: {embed_time:.4f}s")
    
    if len(dense_vec) == 1024 and len(lexical_weights) > 0:
        print("\n🎉 BGEM3FlagModel (Dense + Sparse) 验证通过!")
    else:
        print("\n⚠️ 警告: 维度不对或 Sparse 为空")

if __name__ == "__main__":
    test_local_embedding()
