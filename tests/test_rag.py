#!/usr/bin/env python3
"""
测试完整的 RAG 系统：检索 + 重排序 + 问答
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from llama_index.core import Settings, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms.mock import MockLLM
from database.qdrant_manager import qdrant_manager
from ingestion.indexmannager import IndexManager
from engine import create_query_engine
from models.llm import get_llm
from config import settings

# 设置全局嵌入模型 - 使用标准的 HuggingFace embedding
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    trust_remote_code=True,
)
# 使用 MockLLM 避免外部 API 依赖与模型名校验
Settings.llm = MockLLM(max_tokens=256)


def test_rag_system():
    """测试完整的 RAG 系统"""
    print("🚀 开始 RAG 系统集成测试...")

    # 1. 初始化向量存储
    qdrant_client = qdrant_manager.get_client()

    # 清理旧集合（如果存在）
    if qdrant_client.collection_exists(settings.QDRANT_COLLECTION_NAME):
        print(f"⚠️  清理旧集合: {settings.QDRANT_COLLECTION_NAME}")
        qdrant_client.delete_collection(settings.QDRANT_COLLECTION_NAME)

    # 初始化 Vector Store
    vector_store_kwargs = {
        "client": qdrant_client,
        "collection_name": settings.QDRANT_COLLECTION_NAME,
        "enable_hybrid": True,
    }

    # Attach our sparse adapters from BGE so Qdrant hybrid uses the model's lexical outputs
    from models.embedding import BgeM3Service
    vector_store_kwargs["sparse_doc_fn"] = BgeM3Service.sparse_doc_fn
    vector_store_kwargs["sparse_query_fn"] = BgeM3Service.sparse_query_fn

    vector_store = QdrantVectorStore(**vector_store_kwargs)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index_manager = IndexManager(storage_context=storage_context)

    # 2. 创建测试文档
    test_docs = [
        project_root / "test_rag_doc1.txt",
        project_root / "test_rag_doc2.txt",
    ]

    # 文档内容
    doc_contents = [
        """
        Python 编程语言简介

        Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。
        Python 的设计哲学强调代码的可读性和简洁性，使用缩进代替花括号。

        主要特点：
        1. 简单易学：语法简洁明了
        2. 功能强大：支持面向对象、函数式编程
        3. 生态丰富：拥有大量的第三方库
        4. 跨平台：可以在 Windows、Linux、macOS 上运行

        Python 在数据科学、Web 开发、自动化脚本等领域应用广泛。
        """,
        """
        机器学习基础概念

        机器学习是人工智能的一个分支，通过算法让计算机从数据中学习模式。

        监督学习：
        - 线性回归：预测连续值
        - 逻辑回归：分类问题
        - 决策树：基于特征的分类

        无监督学习：
        - K-means 聚类：数据分组
        - 主成分分析：降维
        - 关联规则挖掘：发现数据关联

        深度学习是机器学习的一个子集，使用神经网络进行学习。
        """
    ]

    # 写入测试文件
    for file_path, content in zip(test_docs, doc_contents):
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content.strip())

    try:
        # 3. 索引文档
        print("\n📚 索引测试文档...")
        from ingestion.pipeline import IngestionPipelineWrapper
        pipeline = IngestionPipelineWrapper(index_manager=index_manager)

        for doc_file in test_docs:
            print(f"  索引文档: {doc_file.name}")
            pipeline.run(str(doc_file))

        # 4. 创建查询引擎
        print("\n🔍 创建查询引擎...")
        # 获取索引（假设只有一个索引）
        index = index_manager.index
        if index is None:
            # 如果没有索引，创建一个空的然后添加节点
            from llama_index.core import VectorStoreIndex
            index = VectorStoreIndex.from_vector_store(vector_store)

        # Disable external API reranker in tests to avoid external model issues
        query_engine = create_query_engine(
            index=index,
            similarity_top_k=10,
            rerank_top_n=5,
            alpha=0.5,
            use_reranker=False,
        )

        # 5. 执行测试查询
        test_queries = [
            "Python 的主要特点是什么？",
            "什么是监督学习？",
            "Python 在哪些领域应用广泛？",
            "机器学习和深度学习的关系是什么？",
        ]

        print("\n❓ 执行测试查询...")
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- 查询 {i}: {query} ---")

            # 获取带来源的回答
            result = query_engine.query_with_sources(query)

            print(f"回答: {result['answer'][:200]}...")
            print(f"来源文档数量: {len(result['sources'])}")

            if result['sources']:
                print("Top 来源:")
                for source in result['sources'][:2]:  # 只显示前2个
                    print(f"  - 文档 {source['index']}: 评分 {source.get('score', 'N/A'):.3f}")
                    print(f"    内容预览: {source['content'][:100]}...")

        # 6. 验证系统统计
        print("\n📊 系统统计...")
        index_stats = index_manager.get_index_stats()
        print(f"索引统计: {index_stats}")

        print("\n🎉 RAG 系统测试完成！所有查询都成功处理。")

    finally:
        # 清理测试文件
        for doc_file in test_docs:
            if doc_file.exists():
                doc_file.unlink()
                print(f"🧹 清理文件: {doc_file.name}")


if __name__ == "__main__":
    test_rag_system()