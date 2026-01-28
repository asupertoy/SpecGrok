import logging
from typing import List, Any, Optional, Dict
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.vector_stores import MetadataFilters
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.vector_stores.qdrant import QdrantVectorStore
from config import settings

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    混合检索器：结合语义检索和关键词检索
    支持 Qdrant 的 Hybrid Search 功能
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        similarity_top_k: int = 10,
        alpha: float = 0.5,  # 平衡语义(1.0)和关键词(0.0)的权重
        vector_store_query_mode: str = "hybrid",
        filters: Optional[MetadataFilters] = None,
        node_postprocessors: Optional[List] = None,
    ):
        """
        初始化混合检索器

        Args:
            index: VectorStoreIndex 实例
            similarity_top_k: 返回的相似文档数量
            alpha: 混合检索权重 (0.0=纯关键词, 1.0=纯语义, 0.5=平衡)
            vector_store_query_mode: 查询模式 ("hybrid", "dense", "sparse")
            filters: 元数据过滤器
            node_postprocessors: 后处理节点列表
        """
        self.index = index
        self.similarity_top_k = similarity_top_k
        self.alpha = alpha
        self.vector_store_query_mode = vector_store_query_mode
        self.filters = filters
        self.node_postprocessors = node_postprocessors or []

        # 创建基础检索器
        self.base_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.similarity_top_k,
            vector_store_query_mode=self.vector_store_query_mode,
            filters=self.filters,
            alpha=self.alpha,  # LlamaIndex 支持 alpha 参数用于混合检索
        )

        logger.info(f"HybridRetriever initialized with alpha={self.alpha}, top_k={self.similarity_top_k}")

    def retrieve(self, query: str, **kwargs) -> List[NodeWithScore]:
        """
        执行混合检索

        Args:
            query: 查询字符串
            **kwargs: 额外参数

        Returns:
            List[NodeWithScore]: 检索到的节点及其相似度分数
        """
        try:
            # 创建查询包
            query_bundle = QueryBundle(query_str=query, **kwargs)

            # 执行检索
            nodes_with_scores = self.base_retriever.retrieve(query_bundle)

            # 应用后处理
            for postprocessor in self.node_postprocessors:
                nodes_with_scores = postprocessor.postprocess_nodes(
                    nodes_with_scores, query_bundle=query_bundle
                )

            logger.info(f"Retrieved {len(nodes_with_scores)} nodes for query: '{query[:50]}...'")
            return nodes_with_scores

        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []

    def get_relevant_documents(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        获取相关文档的简化接口

        Args:
            query: 查询字符串

        Returns:
            List[Dict]: 包含文档内容和元数据的字典列表
        """
        nodes_with_scores = self.retrieve(query, **kwargs)

        results = []
        for node_with_score in nodes_with_scores:
            doc_info = {
                "content": node_with_score.node.get_content(),
                "score": node_with_score.score,
                "metadata": node_with_score.node.metadata,
                "node_id": node_with_score.node.node_id,
                "doc_id": node_with_score.node.ref_doc_id,
            }
            results.append(doc_info)

        return results


def create_hybrid_retriever(
    index: VectorStoreIndex,
    similarity_top_k: int = 10,
    alpha: float = 0.5,
    filters: Optional[MetadataFilters] = None,
) -> HybridRetriever:
    """
    创建混合检索器的工厂函数

    Args:
        index: VectorStoreIndex 实例
        similarity_top_k: 返回的相似文档数量
        alpha: 混合检索权重
        filters: 元数据过滤器

    Returns:
        HybridRetriever: 配置好的检索器实例
    """
    return HybridRetriever(
        index=index,
        similarity_top_k=similarity_top_k,
        alpha=alpha,
        filters=filters,
    )