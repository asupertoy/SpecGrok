from typing import Optional
import logging
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

from FlagEmbedding import FlagReranker
from config import settings
from models.llm import get_rerank_model

logger = logging.getLogger(__name__)


class ApiReranker(BaseNodePostprocessor):
    """
    使用 OpenAI 兼容的 Rerank API 进行打分的重排序器。
    逐文档调用 API，适合小规模检索结果；大规模场景建议批量/本地模型。
    """

    model_name: str
    top_n: int = 5
    _client = None

    def __init__(self, model_name: str, top_n: int = 5):
        super().__init__(model_name=model_name, top_n=top_n)
        self._client = get_rerank_model()
        self.model_name = model_name
        self.top_n = top_n

    def _score_pair(self, query: str, doc: str) -> float:
        """调用 Rerank 模型对 (query, doc) 打分，返回 0~1 概率。"""
        try:
            # 使用兼容的 /rerank 接口，如果不支持则回退到简单打分提示
            if hasattr(self._client, "rerank"):
                resp = self._client.rerank(
                    model=self.model_name,
                    query=query,
                    documents=[doc],
                )
                score = resp.results[0].score
                return float(score)

            # 回退：用对比提示让 LLM 给出 0-1 评分
            prompt = (
                "You are a strict evaluator. Given a query and a document, "
                "return only a relevance score between 0 and 1 with no explanation.\n"
                f"Query: {query}\nDocument: {doc[:1500]}\nScore:"
            )
            completion = self._client.complete(prompt=prompt, max_tokens=4)
            text = completion.text.strip()
            score = float(text.split()[0]) if text else 0.0
            # 规整到 0~1
            score = max(0.0, min(1.0, score))
            return score
        except Exception as e:
            logger.error(f"API rerank scoring failed: {e}")
            return 0.0

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> list[NodeWithScore]:
        if not nodes:
            return []
        if query_bundle is None:
            return nodes[: self.top_n]

        query_str = query_bundle.query_str
        for node in nodes:
            doc = node.node.get_content(metadata_mode="embed")
            node.score = self._score_pair(query_str, doc)

        return sorted(nodes, key=lambda x: x.score, reverse=True)[: self.top_n]


class CustomFlagReranker(BaseNodePostprocessor):
    """
    本地 FlagEmbedding 重排序器作为后备方案。
    """
    model_name: str
    top_n: int = 5
    use_fp16: bool = True
    _model: Optional[FlagReranker] = None

    def __init__(self, model_name: str, top_n: int = 5, use_fp16: bool = True):
        super().__init__(model_name=model_name, top_n=top_n, use_fp16=use_fp16)
        self._model = FlagReranker(model_name, use_fp16=use_fp16)

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> list[NodeWithScore]:
        if not nodes:
            return []
        if query_bundle is None:
            return nodes[: self.top_n]

        query_str = query_bundle.query_str
        texts = [node.node.get_content(metadata_mode="embed") for node in nodes]
        pairs = [[query_str, text] for text in texts]

        scores = self._model.compute_score(pairs)

        for node, score in zip(nodes, scores):
            node.score = float(score)

        return sorted(nodes, key=lambda x: x.score, reverse=True)[: self.top_n]


def get_reranker(top_n: int = 5) -> Optional[BaseNodePostprocessor]:
    """Factory to get the configured reranker (prefer API, fallback to local FlagEmbedding)."""
    model_name = settings.RERANKER_MODEL_NAME
    if not model_name:
        return None

    logger.info(f"Initializing Reranker: {model_name}")

    # 优先使用 API reranker（OpenAI 兼容接口）
    try:
        return ApiReranker(model_name=model_name, top_n=top_n)
    except Exception as e:
        logger.error(f"API reranker init failed, fallback to local FlagReranker: {e}")

    # 回退到本地 FlagEmbedding
    try:
        return CustomFlagReranker(
            model_name=model_name,
            top_n=top_n,
            use_fp16=True,
        )
    except Exception as e:
        logger.error(f"Failed to initialize FlagReranker: {e}")
        return None
