import logging
from typing import Optional, List, Any, Dict
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts import PromptTemplate

from retrieval import HybridRetriever, get_reranker
from config import settings

logger = logging.getLogger(__name__)


# 定制的 Prompt 模板，针对技术文档优化
DEFAULT_QA_PROMPT_TMPL = """基于以下上下文信息，回答用户的问题。

**重要指导原则：**
1. 严格基于提供的上下文信息回答，不要添加外部知识
2. 如果上下文信息不足以回答问题，请明确说明
3. 对于技术细节，请引用具体的文档来源
4. 保持回答的准确性和客观性
5. 如果遇到矛盾信息，请指出并解释

**上下文信息：**
{context_str}

**用户问题：** {query_str}

**回答：**"""

DEFAULT_REFINE_PROMPT_TMPL = """原始问题：{query_str}
原始回答：{existing_answer}

基于新的上下文信息，完善或修正上述回答。

**新上下文信息：**
{context_msg}

**重要指导原则：**
1. 保持回答的一致性和逻辑性
2. 整合新信息，不要重复已有内容
3. 如果新信息与原有回答矛盾，请解释原因
4. 继续引用文档来源
5. 保持技术准确性

**完善后的回答：**"""


class SpecGrokQueryEngine:
    """
    SpecGrok 查询引擎：集成混合检索 + 重排序 + 问答合成的完整 RAG 系统
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        similarity_top_k: int = 10,
        rerank_top_n: int = 5,
        alpha: float = 0.5,
        use_reranker: bool = True,
    ):
        """
        初始化查询引擎

        Args:
            index: VectorStoreIndex 实例
            similarity_top_k: 检索的相似文档数量
            rerank_top_n: 重排序后保留的文档数量
            alpha: 混合检索权重 (0.0=纯关键词, 1.0=纯语义)
            use_reranker: 是否使用重排序器
        """
        self.index = index
        self.similarity_top_k = similarity_top_k
        self.rerank_top_n = rerank_top_n
        self.alpha = alpha
        self.use_reranker = use_reranker

        # 初始化检索器
        self.retriever = HybridRetriever(
            index=self.index,
            similarity_top_k=self.similarity_top_k,
            alpha=self.alpha,
        )

        # 初始化后处理器
        self.node_postprocessors = []
        if self.use_reranker:
            reranker = get_reranker(top_n=self.rerank_top_n)
            if reranker:
                self.node_postprocessors.append(reranker)
                logger.info("Reranker enabled")
            else:
                logger.warning("Reranker initialization failed, proceeding without reranking")

        # 创建查询引擎 - 使用简化的方式
        # Build query engine via RetrieverQueryEngine to avoid multiple-value conflicts.
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever=self.retriever.base_retriever,
            node_postprocessors=self.node_postprocessors,
            text_qa_template=PromptTemplate(DEFAULT_QA_PROMPT_TMPL),
            refine_template=PromptTemplate(DEFAULT_REFINE_PROMPT_TMPL),
            similarity_top_k=self.similarity_top_k,
            llm=getattr(Settings, "llm", None),
        )

        logger.info("SpecGrokQueryEngine initialized successfully")

    def query(self, query_str: str) -> str:
        """
        执行查询并返回回答

        Args:
            query_str: 查询字符串

        Returns:
            str: 生成的回答
        """
        try:
            logger.info(f"Processing query: '{query_str[:100]}...'")

            # 执行查询
            response = self.query_engine.query(query_str)

            answer = str(response)
            logger.info(f"Generated response of length: {len(answer)}")

            return answer

        except Exception as e:
            logger.error(f"Error during query processing: {e}")
            return f"查询处理失败: {str(e)}"

    def query_with_sources(self, query_str: str) -> Dict[str, Any]:
        """
        执行查询并返回回答及来源信息

        Args:
            query_str: 查询字符串

        Returns:
            Dict: 包含回答和来源信息的字典
        """
        try:
            response = self.query_engine.query(query_str)

            result = {
                "answer": str(response),
                "sources": []
            }

            # 提取来源信息
            if hasattr(response, 'source_nodes'):
                for i, node in enumerate(response.source_nodes):
                    source_info = {
                        "index": i + 1,
                        "content": node.node.get_content()[:200] + "...",
                        "score": getattr(node, 'score', None),
                        "metadata": node.node.metadata,
                        "doc_id": node.node.ref_doc_id,
                    }
                    result["sources"].append(source_info)

            return result

        except Exception as e:
            logger.error(f"Error during query with sources: {e}")
            return {
                "answer": f"查询处理失败: {str(e)}",
                "sources": []
            }


def create_query_engine(
    index: VectorStoreIndex,
    similarity_top_k: int = 10,
    rerank_top_n: int = 5,
    alpha: float = 0.5,
    use_reranker: bool = True,
) -> SpecGrokQueryEngine:
    """
    创建查询引擎的工厂函数

    Args:
        index: VectorStoreIndex 实例
        similarity_top_k: 检索的相似文档数量
        rerank_top_n: 重排序后保留的文档数量
        alpha: 混合检索权重
        use_reranker: 是否使用重排序器

    Returns:
        SpecGrokQueryEngine: 配置好的查询引擎实例
    """
    return SpecGrokQueryEngine(
        index=index,
        similarity_top_k=similarity_top_k,
        rerank_top_n=rerank_top_n,
        alpha=alpha,
        use_reranker=use_reranker,
    )