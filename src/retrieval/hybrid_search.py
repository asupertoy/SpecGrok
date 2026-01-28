import logging
from typing import List, Optional, Any, Dict

from llama_index.core import QueryBundle, VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import BM25Retriever
from llama_index.core.callbacks import CallbackManager

from config import settings
from .reranker import get_reranker

logger = logging.getLogger(__name__)

class HybridRetriever(BaseRetriever):
    """
    SpecGrok Hybrid Retriever.
    Combines Dense Vector Search (BGE-M3) with Sparse Keyword Search (BM25).
    Applies Reciprocal Rank Fusion (RRF) and Reranking.
    """

    def __init__(
        self,
        vector_index: VectorStoreIndex,
        nodes: Optional[List[Any]] = None,
        similarity_top_k: int = 10,
        mode: str = "rrf",  # "rrf" or "simple"
        rrf_k: int = 60,
        use_reranker: bool = True,
        callback_manager: Optional[CallbackManager] = None,
    ):
        super().__init__(callback_manager=callback_manager)
        self.vector_index = vector_index
        self.mode = mode
        self.rrf_k = rrf_k
        self.use_reranker = use_reranker
        
        # Initial Retrieval Top K (retrieve more candidates before reraking)
        self.initial_top_k = similarity_top_k * 2 if use_reranker else similarity_top_k
        self.final_top_k = similarity_top_k

        # 1. Setup Vector Retriever
        self.vector_retriever = vector_index.as_retriever(
            similarity_top_k=self.initial_top_k
        )

        # 2. Setup BM25 Retriever
        self.bm25_retriever = None
        if settings.ENABLE_HYBRID:
            logger.info("Initializing BM25 Retriever for Hybrid Search...")
            target_nodes = nodes
            
            # If nodes not provided, try to load from docstore (Memory efficient check needed for Prod)
            if not target_nodes:
                try:
                    target_nodes = list(vector_index.docstore.docs.values())
                    logger.info(f"Loaded {len(target_nodes)} nodes from docstore.")
                except Exception as e:
                    logger.warning(f"Could not load nodes from docstore: {e}")

            if target_nodes:
                self.bm25_retriever = BM25Retriever.from_defaults(
                    nodes=target_nodes,
                    similarity_top_k=self.initial_top_k
                )
            else:
                logger.warning("No nodes available for BM25. Hybrid Search will fallback to Vector Search.")

        # 3. Setup Reranker
        self.reranker = get_reranker(top_n=self.final_top_k) if use_reranker else None

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Execute Hybrid Retrieval:
        1. Retrieve from Vector Store.
        2. Retrieve from BM25 (if enabled).
        3. Fuse results (RRF).
        4. Rerank (if enabled).
        """
        # 1. Vector Search
        vector_nodes = self.vector_retriever.retrieve(query_bundle)
        
        # 2. BM25 Search
        bm25_nodes = []
        if self.bm25_retriever:
            bm25_nodes = self.bm25_retriever.retrieve(query_bundle)

        # 3. Fusion
        if not bm25_nodes:
            combined_nodes = vector_nodes
        else:
            combined_nodes = self._apply_fusion(vector_nodes, bm25_nodes)

        # 4. Reranking
        if self.reranker:
            try:
                combined_nodes = self.reranker.postprocess_nodes(
                    combined_nodes, query_bundle
                )
            except Exception as e:
                logger.error(f"Reranking failed: {e}. Returning fusion results.")
                # Fallback to top_k of fusion
                combined_nodes = combined_nodes[:self.final_top_k]

        return combined_nodes

    def _apply_fusion(
        self, 
        vector_nodes: List[NodeWithScore], 
        bm25_nodes: List[NodeWithScore]
    ) -> List[NodeWithScore]:
        """Apply Reciprocal Rank Fusion (RRF)."""
        
        node_map: Dict[str, NodeWithScore] = {}
        processed_ids = set()
        
        # Helper to process list
        def process_list(nodes, weight_factor=1.0):
            for rank, node in enumerate(nodes):
                if node.node_id not in node_map:
                    node_map[node.node_id] = node
                    # Reset score for calculated fusion score
                    node_map[node.node_id].score = 0.0
                
                # RRF Formula: 1 / (k + rank)
                rrf_score = 1.0 / (self.rrf_k + rank + 1)
                node_map[node.node_id].score += rrf_score * weight_factor

        # Process both lists
        process_list(vector_nodes)
        process_list(bm25_nodes)

        # Convert back to list and sort
        fused_nodes = list(node_map.values())
        fused_nodes.sort(key=lambda x: x.score, reverse=True)
        
        return fused_nodes

