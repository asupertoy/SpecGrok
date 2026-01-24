#!/usr/bin/env python3
"""
测试 IngestionPipelineWrapper 的脚本。
"""

import os
import sys
import time
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from llama_index.core import Settings, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from src.models.embedding import get_embed_model, BgeM3Service
from src.database.qdrant_manager import qdrant_manager
from src.ingestion.indexmannager import IndexManager
from src.ingestion.pipeline import IngestionPipelineWrapper
from src.config import settings

# 设置全局嵌入模型
Settings.embed_model = get_embed_model()

# --- 适配器开始 ---
def custom_sparse_doc_fn(texts: list[str]):
    """使用全局 BGE-M3 模型生成文档稀疏向量 (Adapter wrapper)"""
    return BgeM3Service.get_sparse_embedding_adapter(texts)

def custom_sparse_query_fn(text: str):
    """使用全局 BGE-M3 模型生成查询稀疏向量"""
    if not text:
        return ([], [])
    indices, values = BgeM3Service.get_sparse_embedding_adapter([text])
    return (indices[0], values[0])
# --- 适配器结束 ---

def verify_pipeline_processing(pipeline, index_manager, file_name, content):
    """
    辅助函数：验证单个文件的完整处理流程 (Load -> Parse -> Chunk -> Embed -> Upsert)
    """
    file_path = project_root / file_name
    print(f"\n{'='*30}\n🔍 正在测试文件: {file_name}\n{'='*30}")
    
    # 写入测试文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    try:
        # 1. Load
        print(f"[1] Loading file...")
        blobs = pipeline.loader.load(str(file_path))
        print(f"    - Loaded blobs: {[b.source for b in blobs]}")
        assert len(blobs) > 0, "Loader 未能加载任何 Blob"

        # 2. Parse
        print(f"[2] Parsing content...")
        parsed_nodes = []
        for b in blobs:
            ns = pipeline._parse_blob(b)
            print(f"    - Parsed {len(ns)} section nodes from {b.source}")
            parsed_nodes.extend(ns)
        assert len(parsed_nodes) > 0, "解析阶段未产出任何节点"

        # 3. Chunking
        print(f"[3] Chunking nodes...")
        chunked_nodes = pipeline.chunking.run_chunking(parsed_nodes)
        print(f"    - Generated {len(chunked_nodes)} chunks")
        assert len(chunked_nodes) > 0, "Chunking 未产出任何 chunk"
        
        # 验证 Chunk 详情
        for i, n in enumerate(chunked_nodes[:3]):
            preview = n.text[:50].replace('\n', ' ')
            print(f"    - Chunk[{i}] (ref_doc_id={n.ref_doc_id}): {preview}...")
            # 验证引用ID继承
            assert n.ref_doc_id is not None, f"Chunk[{i}] 丢失了 ref_doc_id"
            if settings.ENABLE_HYBRID and 'sparse_values' in n.metadata:
                assert isinstance(n.metadata['sparse_values'], dict), "Sparse values 格式错误"

        # 4. Embedding
        print(f"[4] Generating embeddings...")
        index_manager._ensure_embeddings(chunked_nodes)
        post_embedded = sum(1 for n in chunked_nodes if n.embedding is not None)
        print(f"    - Embeddings generated for {post_embedded}/{len(chunked_nodes)} nodes")
        assert post_embedded == len(chunked_nodes), "部分 Chunk 缺失 Embedding"

        # 5. Upsert
        print(f"[5] Upserting to Vector Store...")
        
        # 获取 upsert 前的数量
        client = qdrant_manager.get_client()
        try:
            if client.collection_exists(settings.QDRANT_COLLECTION_NAME):
                before_count = client.get_collection(settings.QDRANT_COLLECTION_NAME).points_count
            else:
                before_count = 0
        except Exception:
            before_count = 0

        index_manager.upsert_nodes(chunked_nodes)
        
        # 等待异步写入
        time.sleep(1.0)

        # 获取 upsert 后的数量
        try:
            after_count = client.get_collection(settings.QDRANT_COLLECTION_NAME).points_count
        except Exception:
            after_count = -1
            
        print(f"    - Qdrant points count: {before_count} -> {after_count}")
        assert after_count >= before_count + len(chunked_nodes) or after_count > 0, "Upsert 后数据量未正常增加"

        print(f"✅ 文件 {file_name} 测试通过")

    finally:
        # 清理测试文件
        if file_path.exists():
            file_path.unlink()
            print(f"🧹 清理文件: {file_name}")

def test_pipeline():
    """测试完整的 ingestion pipeline，覆盖 TXT, MD, HTML。"""
    print("🚀 开始 Pipeline 集成测试 (TXT, MD, HTML)...")

    # 1. 初始化组件
    qdrant_client = qdrant_manager.get_client()
    
    # 清理环境：为了测试准确性，每次运行前清理集合
    if qdrant_client.collection_exists(settings.QDRANT_COLLECTION_NAME):
        print(f"⚠️  清理旧集合: {settings.QDRANT_COLLECTION_NAME}")
        qdrant_client.delete_collection(settings.QDRANT_COLLECTION_NAME)

    # 初始化 Vector Store
    vector_store_kwargs = {
        "client": qdrant_client,
        "collection_name": settings.QDRANT_COLLECTION_NAME,
        "enable_hybrid": settings.ENABLE_HYBRID,
    }
    if settings.ENABLE_HYBRID:
        print("🔧 启用 Hybrid Search Adapter")
        vector_store_kwargs["sparse_doc_fn"] = custom_sparse_doc_fn
        vector_store_kwargs["sparse_query_fn"] = custom_sparse_query_fn

    vector_store = QdrantVectorStore(**vector_store_kwargs)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index_manager = IndexManager(storage_context=storage_context)
    pipeline = IngestionPipelineWrapper(index_manager=index_manager)

    # 2. 定义测试用例 (文件名, 内容)
    test_cases = [
        ("test_sample.txt", 
            r"""
                这是一个复杂的测试文档。
                ### 第一部分
                - 项目1
                - 子项目1.1
                - 子项目1.2
                - 项目2

                #### 第二部分
                数学公式：$$ \int_{0}^{\infty} e^{-x} dx = 1 $$

                代码示例：
                    def hello():
                        print("Hello World")
                        if True:
                            return 42
            """),
                    
        ("test_sample.md", 
            r"""
                # 复杂Markdown文档

                ## 介绍
                这是一个嵌套结构的文档。

                ### 列表部分
                - 顶级项目
                - 子项目A
                    - 深层子项目A1
                - 子项目B
                - 另一个顶级项目

                ## 代码和公式
                ```python
                def complex_function():
                    if condition:
                        for i in range(10):
                            print(f"Item {i}")
                    return result
                ```

                内联公式：$ E = mc^2 $ 和块公式：
                $$ \sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6} $$
            """),

        ("test_sample.html", 
            r"""
                <!DOCTYPE html>
                <html>
                <body>
                    <h1>复杂HTML文档</h1>
                    <div>
                        <h2>嵌套结构</h2>
                        <p>段落内容。</p>
                        <ul>
                            <li>项目1
                                <ul>
                                    <li>子项目1.1</li>
                                    <li>子项目1.2</li>
                                </ul>
                            </li>
                            <li>项目2</li>
                        </ul>
                        <h3>代码部分</h3>
                        <pre><code>
                def example():
                    if True:
                        print("Indented code")
                        for x in list:
                            process(x)
                        </code></pre>
                        <p>数学公式：$$ a^2 + b^2 = c^2 $$</p>
                    </div>
                </body>
                </html>
            """)
    ]

    # 3. 循环执行测试
    try:
        for fname, content in test_cases:
            verify_pipeline_processing(pipeline, index_manager, fname, content)

        # 4. 测试 pipeline.run() 整体流程及统计
        print(f"\n{'='*30}\n📊 Testing pipeline.run() & Stats\n{'='*30}")
        run_file = project_root / "test_run_stats.txt"
        with open(run_file, 'w', encoding='utf-8') as f:
            f.write("Stats test content.")
            
        try:
            pipeline.run(str(run_file))
            stats = pipeline.get_stats()
            print("Pipeline Stats:", stats)
            assert stats["processed_files"] >= 1
            assert stats["generated_chunks"] >= 1
        finally:
            if run_file.exists():
                run_file.unlink()

        # 最终 Index 状态
        idx_stats = index_manager.get_index_stats()
        print("\n📈 Final Index Stats:", idx_stats)
        print("\n🎉 所有测试用例通过！")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    test_pipeline()
