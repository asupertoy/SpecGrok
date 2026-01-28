import logging
import os
import hashlib
from typing import List, Dict, Optional, Any, Union, Callable
from llama_index.core import VectorStoreIndex, StorageContext, Document, Settings
from llama_index.core.schema import BaseNode
from models.embedding import get_embed_model
from models.embedding import BgeM3Service

class IndexManager:
    """管理和操作向量存储索引的类。"""
    
    def __init__(
        self,
        storage_context: StorageContext,
        auto_reingest: bool = False,
        reingest_handler: Optional[Callable[[List[str]], None]] = None,
        predelete_before_upsert: bool = False,
    ):
        """
        初始化 IndexManager，使用传入的 StorageContext。
        
        Args:
            storage_context (StorageContext): 存储上下文，包含 vector_store 等。
            auto_reingest (bool): 是否在检测到变更时自动触发 reingest（需要提供 reingest_handler）。
            reingest_handler (Callable): 当 auto_reingest=True 时，自动调用该 handler 执行 reingest（接收 doc_id 列表）。
            predelete_before_upsert (bool): 在执行 upsert 前是否先删除已有相同 node_id 的点；默认 False（关闭）。
        """
        self.storage_context = storage_context
        self.auto_reingest = auto_reingest
        self.reingest_handler = reingest_handler
        self.predelete_before_upsert = predelete_before_upsert

        # 获取嵌入模型；优先使用全局 Settings.embed_model（LlamaIndex BaseEmbedding），否则回退到本地 BGE 服务
        self.embed_model = getattr(Settings, "embed_model", None) or get_embed_model()
        
        self.index: Optional[VectorStoreIndex] = None
        self._logger = logging.getLogger(__name__)

    def create_index(self, nodes: List[BaseNode]) -> VectorStoreIndex:
        """
        核心：根据 Chunks 列表首次生成 Embedding 并构建索引。
        
        Args:
            nodes (List[BaseNode]): 待索引的节点列表。可以直接传入 Chunking 组件输出的 List[TextNode]。
                                    注意：通常**不需要**转换为 IndexNode，除非你在构建特殊的高级递归检索结构。
                                    对于通过 VectorStore 进行语义检索的标准场景，TextNode 是最适合的一等公民。
            
        Returns:
            VectorStoreIndex: 构建完成的索引对象。
        """
        if not nodes:
            self._logger.warning("No nodes provided to create index.")
            return None
            
        self._logger.info(f"Creating index with {len(nodes)} nodes...")
        
        # 1. 生成 Embeddings (使用 BGE-M3)
        self._ensure_embeddings(nodes)
        
        # 2. 构建 Index
        # 我们传入 storage_context，LlamaIndex 会自动使用其中的 vector_store
        self.index = VectorStoreIndex(
            nodes,
            storage_context=self.storage_context
        )
        
        self._logger.info("Index created successfully.")
        return self.index

    def upsert_nodes(self, nodes: List[BaseNode]) -> None:
        """
        增量/幂等更新：如果 Node 已存在则更新，不存在则插入（基于 node_id）。
        
        Args:
            nodes (List[BaseNode]): 待更新的节点列表（通常为 List[TextNode]）。
        """
        if not nodes:
            return

        self._logger.info(f"Upserting {len(nodes)} nodes...")

        # 在写入之前，为每个 node 尝试添加 file_hash（如果有 source），以便后续的 refresh 能做比较
        for idx, n in enumerate(nodes):
            # 移除 I/O: 假设上游 (Pipeline/Parser) 已经正确设置了 file_hash。
            # IndexManager 属于数据库层，不应直接操作文件系统。
            # 如果 node.metadata 里没有 valid 的 file_hash，我们不应该在这里临时去读文件。
            pass

            # 设置 ref_doc_id 为 doc_id（哈希）
            n._ref_doc_id = n.metadata.get('doc_id')

            # 为了实现幂等 upsert：为每个 Node 生成确定性的 node_id（基于 doc_id 和文本内容的 hash）
            try:
                doc_id = n.metadata.get('doc_id', '')
                # 使用文本内容的 hash 来确保稳定性
                content_hash = hashlib.md5((n.text or '').encode('utf-8')).hexdigest()
                node_key = f"{doc_id}|{content_hash}"
                # 始终覆盖 node_id，保证在多次运行中稳定一致（接受显式覆盖的例外可在未来添加）
                n.node_id = hashlib.md5(node_key.encode('utf-8')).hexdigest()
            except Exception:
                # 如果无法设置 node_id，继续（仍然可以插入，但可能导致重复）
                pass
        # 1. 确保 Embedding 存在
        self._ensure_embeddings(nodes)
        
        # 2. 获取 Index 实例
        index = self._get_or_load_index()
        
        # 3. 根据配置决定是否删除已存在的同 id 节点以保证幂等性（在某些 vector stores insert_nodes 可能无法避免冲突）
        if getattr(self, 'predelete_before_upsert', False):
            try:
                node_ids = [getattr(n, 'node_id', None) for n in nodes if getattr(n, 'node_id', None)]
                if node_ids:
                    try:
                        self._logger.debug(f"Deleting existing nodes with ids: {node_ids}")
                        index.delete_nodes(node_ids)
                    except Exception:
                        # 某些索引实现可能不支持 delete_nodes 或出错，这里容错并继续插入以保证可用性
                        self._logger.debug("Index delete_nodes not supported or failed; continuing with insert.")
            except Exception:
                # 忽略任何删除前的准备错误
                pass
        else:
            self._logger.debug("predelete_before_upsert is disabled; skipping pre-delete step.")

        # 4. 插入数据
        index.insert_nodes(nodes)
        
        self._logger.info("Upsert completed.")

    def refresh_index(self, items: List[Union[Document, BaseNode]]) -> List[str]:
        """
        增量同步（支持 Document 或 TextNode）：

        支持两种输入类型：
          - Document: 按 document-level 的 hash/id 做比较（behavior 同之前）。
          - BaseNode (例如 TextNode): 将传入的节点按其 metadata['doc_id'] 聚合为文档级别，基于 doc_id 比较。

        返回：需要重新注入的 doc_id 列表（调用者负责用 Parser/Chunking 对该 doc 重新入库）。
        """
        self._logger.info(f"Refreshing index for {len(items)} items (Document or Node)...")

        reingest_doc_ids: List[str] = []

        try:
            # 准备 docstore
            try:
                ds = self.storage_context.docstore
            except Exception:
                ds = None

            # 将输入按类型分流
            # 文档级输入直接处理；Node 输入需要先聚合（按 doc_id）
            doc_items: List[Document] = []
            node_groups: Dict[str, List[BaseNode]] = {}

            for it in items:
                # Document-like
                if isinstance(it, Document):
                    doc_items.append(it)
                    continue

                # Node-like (BaseNode or duck-typed)
                if isinstance(it, BaseNode) or hasattr(it, 'text'):
                    src = None
                    try:
                        src = it.metadata.get('doc_id')
                    except Exception:
                        src = None
                    if not src:
                        # 尝试用 ref_doc_id 回退
                        try:
                            src = it.metadata.get('ref_doc_id')
                        except Exception:
                            src = None
                    if not src:
                        # 无法确定归属文档，跳过并记录
                        self._logger.debug('Node without doc_id/ref_doc_id; skipping for refresh.')
                        continue
                    node_groups.setdefault(src, []).append(it)
                    continue

                # 其他类型忽略
                self._logger.debug(f'Unsupported item type in refresh_index: {type(it)}')

            # 处理 Document 输入（原先逻辑）
            if doc_items:
                self._logger.debug(f'Processing {len(doc_items)} Document items for refresh...')
                for doc in doc_items:
                    # 重用原先的 Document 检查逻辑
                    doc_id = getattr(doc, 'doc_id', None) or getattr(doc, 'id_', None) or getattr(doc, 'doc_id', None)
                    if doc_id is None:
                        doc_id = getattr(doc, 'extra_info', {}).get('doc_id') if getattr(doc, 'extra_info', None) else None
                    if doc_id is None:
                        self._logger.debug('Document without id_ or doc_id found; skipping.')
                        continue

                    incoming_hash = None
                    try:
                        extra = getattr(doc, 'extra_info', None) or getattr(doc, 'metadata', None) or {}
                        incoming_hash = (extra.get('doc_id') if isinstance(extra, dict) else None) or (extra.get('hash') if isinstance(extra, dict) else None)
                    except Exception:
                        incoming_hash = None

                    if incoming_hash is None:
                        # 移除 I/O: 必须依赖 metadata 或 extra_info 中的 hash。
                        # IndexManager 不再负责读取源文件。
                        self._logger.debug(f'No incoming hash for doc_id={doc_id}; skipping hash check (no-IO policy).')
                        pass

                    stored_hash = None
                    if ds is not None:
                        try:
                            stored = None
                            try:
                                stored = ds.get_document(doc_id)
                            except Exception:
                                try:
                                    stored = ds.get_doc(doc_id)
                                except Exception:
                                    stored = None

                            if stored is not None:
                                s_extra = getattr(stored, 'extra_info', None) or getattr(stored, 'metadata', None) or {}
                                if isinstance(s_extra, dict):
                                    stored_hash = s_extra.get('doc_id') or s_extra.get('hash')
                        except Exception:
                            stored_hash = None

                    if incoming_hash is None:
                        self._logger.debug(f'No incoming hash for doc_id={doc_id}; skipping hash check.')
                        continue

                    if stored_hash != incoming_hash:
                        self._logger.info(f'Detected change for doc_id={doc_id} (stored={stored_hash} incoming={incoming_hash}). Scheduling reingest.')
                        try:
                            self.delete_by_doc_id(doc_id)
                        except Exception as e:
                            self._logger.warning(f'Failed to delete old doc {doc_id}: {e}')
                        reingest_doc_ids.append(doc_id)

            # 处理 Node 分组（按 doc_id）
            if node_groups:
                self._logger.debug(f'Processing {len(node_groups)} node groups for refresh...')
                for src, nodes in node_groups.items():
                    # 使用 doc_id 作为 doc_id
                    doc_id = src

                    # 使用 doc_id 作为 incoming_hash（因为 doc_id 是内容 hash）
                    incoming_hash = doc_id

                    stored_hash = None
                    if ds is not None and doc_id is not None:
                        try:
                            stored = None
                            try:
                                stored = ds.get_document(doc_id)
                            except Exception:
                                try:
                                    stored = ds.get_doc(doc_id)
                                except Exception:
                                    stored = None

                            if stored is not None:
                                s_extra = getattr(stored, 'extra_info', None) or getattr(stored, 'metadata', None) or {}
                                if isinstance(s_extra, dict):
                                    stored_hash = s_extra.get('doc_id') or s_extra.get('hash')
                        except Exception:
                            stored_hash = None

                    if stored_hash != incoming_hash:
                        self._logger.info(f'Detected change for doc_id={doc_id} (stored={stored_hash} incoming={incoming_hash}). Scheduling reingest.')
                        try:
                            self.delete_by_doc_id(doc_id)
                        except Exception as e:
                            self._logger.warning(f'Failed to delete old doc {doc_id}: {e}')
                        reingest_doc_ids.append(doc_id)

        except Exception as e:
            self._logger.error(f'Error during refresh_index flow: {e}')
        # 如果启用了自动 reingest，调用 handler（如果提供），否则使用默认 handler
        if reingest_doc_ids and getattr(self, 'auto_reingest', False):
            handler = getattr(self, 'reingest_handler', None)
            if callable(handler):
                try:
                    self._logger.info(f"Auto reingest enabled. Invoking reingest_handler for {len(reingest_doc_ids)} docs...")
                    handler(reingest_doc_ids)
                except Exception as e:
                    self._logger.error(f"Error when invoking reingest_handler: {e}")
            else:
                # fallback to default reingest handler
                try:
                    self._logger.info("Auto reingest enabled. Using default reingest handler.")
                    self._default_reingest_handler(reingest_doc_ids)
                except Exception as e:
                    self._logger.error(f"Default reingest handler failed: {e}")
        self._logger.info(f'Refresh complete. {len(reingest_doc_ids)} docs need reingestion.')
        return reingest_doc_ids
    def _default_reingest_handler(self, doc_ids: List[str]) -> None:
        """默认 reingest handler：创建 IngestionPipelineWrapper 并对每个 doc_id 调用 run("""
        try:
            from config import settings as _settings
            from ingestion.pipeline import IngestionPipelineWrapper
            import time

            pipeline = IngestionPipelineWrapper(index_manager=self)

            max_batch = getattr(_settings, 'AUTO_REINGEST_MAX_BATCH', 50)
            delay = getattr(_settings, 'AUTO_REINGEST_DELAY_SECONDS', 0.0)

            for d in doc_ids[:max_batch]:
                try:
                    self._logger.info(f"Default reingest: running pipeline for {d}")
                    pipeline.run(d)
                except Exception as e:
                    self._logger.error(f"Default reingest run failed for {d}: {e}")
                if delay and delay > 0:
                    time.sleep(delay)
        except Exception as e:
            self._logger.exception(f"Failed to run default reingest handler: {e}")
    def delete_by_doc_id(self, doc_id: str) -> None:
        """
        清理：级联删除属于某篇 HTML/Document 的所有 Chunks。
        
        Args:
            doc_id (str): 文档 ID (ref_doc_id)。
        """
        self._logger.info(f"Deleting data for doc_id: {doc_id}")
        index = self._get_or_load_index()
        
        # delete_from_docstore=True 会同时清理 metadata store 中的记录
        index.delete_ref_doc(doc_id, delete_from_docstore=True)
        self._logger.info(f"Deletion logic executed for {doc_id}.")

    def persist(self, path: str) -> None:
        """
        持久化：将索引的元数据（非向量数据，如 docstore, index_store）同步到磁盘。
        注意：对于 Qdrant 这种 server-based vectordb，向量数据是 persist 在 server 端的，这里主要存 LlamaIndex 的映射关系。
        
        Args:
            path (str): 持久化目录路径。
        """
        self._logger.info(f"Persisting storage context to {path}...")
        self.storage_context.persist(persist_dir=path)

    def get_index_stats(self) -> Dict[str, Any]:
        """
        状态查询：返回当前索引的统计信息。

        优先来源顺序：
         1. 尝试从 LlamaIndex 的内存/持久化索引中读取节点数（更贴近业务层的 node 数）。
         2. 回退到 Qdrant 的 points_count（向量层面的真实计数）。
        """
        stats: Dict[str, Any] = {
            "total_nodes": "Unknown",
            "backend": "Unknown",
        }

        total_vectors: Optional[int] = None

        # 1) 尝试读取 Qdrant 底层统计（尽早获取以便 fallback）
        try:
            from database.qdrant_manager import qdrant_manager
            from config import settings

            client = qdrant_manager.get_client()
            collection_name = settings.QDRANT_COLLECTION_NAME

            if client.collection_exists(collection_name):
                info = client.get_collection(collection_name)
                total_vectors = info.points_count
                stats["total_vectors"] = total_vectors
                stats["status"] = str(info.status)
                stats["backend"] = "Qdrant"
        except ImportError:
            self._logger.debug("QdrantManager not available for stats.")
        except Exception as e:
            self._logger.warning(f"Could not retrieve Qdrant stats: {e}")

        # 2) 尝试从 LlamaIndex 的索引结构中读取节点数（更能反映 index 层）
        try:
            index = None
            try:
                index = self._get_or_load_index()
            except Exception as _:
                index = self.index

            if index:
                # 多种索引结构的适配：尝试 index.index_struct / index._index_struct
                idx_struct = getattr(index, "index_struct", None) or getattr(index, "_index_struct", None)
                if idx_struct is not None:
                    n = getattr(idx_struct, "num_nodes", None)
                    if n is None:
                        nodes_attr = getattr(idx_struct, "nodes", None)
                        if nodes_attr is not None:
                            try:
                                n = len(nodes_attr)
                            except Exception:
                                try:
                                    n = len(list(nodes_attr))
                                except Exception:
                                    n = None
                    if isinstance(n, int):
                        stats["total_nodes"] = n

                # 备选策略：尝试使用 docstore 的文档计数
                if stats["total_nodes"] == "Unknown":
                    try:
                        ds = index.storage_context.docstore
                        # 有些实现会暴露 get_all_doc_ids
                        try:
                            ids = ds.get_all_doc_ids()
                            stats["total_nodes"] = len(ids)
                        except Exception:
                            try:
                                stats["total_nodes"] = len(ds)
                            except Exception:
                                pass
                    except Exception:
                        pass
        except Exception as e:
            self._logger.debug(f"Could not read LlamaIndex stats: {e}")

        # 3) 回退与一致性检查
        if stats.get("total_nodes") == "Unknown" and total_vectors is not None:
            stats["total_nodes"] = total_vectors

        if isinstance(stats.get("total_nodes"), int) and total_vectors is not None:
            stats["synced_with_vectors"] = (stats["total_nodes"] == total_vectors)

        return stats

    def _get_or_load_index(self) -> VectorStoreIndex:
        """Helper: 如果内存中没有 index 对象，尝试从 vector store 初始化。"""
        if self.index:
            return self.index
        
        self._logger.info("Initializing VectorStoreIndex from existing vector store...")
        # 指定 embed_model 以避免 LlamaIndex 在未配置远程模型时尝试解析默认 embed（比如 OpenAI）
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.storage_context.vector_store,
            embed_model=self.embed_model,
            storage_context=self.storage_context
        )
        return self.index

    def _ensure_embeddings(self, nodes: List[BaseNode]):
        """
        检查并生成节点的 Embeddings。
        使用 LlamaIndex 的嵌入接口。
        
        注意：此前的手动 Sparse Vector 计算已被移除。
        混合检索的稀疏向量生成应当由 IngestionPipeline (sparse_doc_fn) 
        或 VectorStore (enable_hybrid=True) 自动处理，
        不应在 metadata 中手动注入 'lexical_weights'。
        """
        # 筛选出没有 embedding 的节点
        nodes_to_embed = [n for n in nodes if n.embedding is None]
        if not nodes_to_embed:
            return

        self._logger.info(f"Computing embeddings for {len(nodes_to_embed)} nodes using configured embed model...")
        
        # 提取文本
        texts = [n.get_content(metadata_mode="embed") for n in nodes_to_embed]
        
        # 使用 LlamaIndex 嵌入接口
        dense_vecs = self.embed_model.get_text_embedding_batch(texts)
        
        # 填充
        if dense_vecs and len(dense_vecs) == len(nodes_to_embed):
            for i, node in enumerate(nodes_to_embed):
                node.embedding = dense_vecs[i]
        else:
            self._logger.error("Embedding generation failed: count mismatch or empty result.")
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        获取索引统计信息。
        
        Returns:
            Dict[str, Any]: 包含 total_nodes 和 backend 信息的字典
        """
        try:
            if self.index is None:
                return {"total_nodes": "Unknown", "backend": "Unknown"}
            
            # 获取向量存储统计
            vector_store = self.storage_context.vector_store
            if hasattr(vector_store, 'client'):
                # Qdrant 向量存储
                client = vector_store.client
                collection_name = getattr(vector_store, 'collection_name', 'Unknown')
                
                if client.collection_exists(collection_name):
                    collection_info = client.get_collection(collection_name)
                    total_points = collection_info.points_count
                    return {
                        "total_nodes": total_points,
                        "backend": f"Qdrant ({collection_name})"
                    }
                else:
                    return {"total_nodes": 0, "backend": f"Qdrant ({collection_name} - not found)"}
            else:
                return {"total_nodes": "Unknown", "backend": "Unknown"}
                
        except Exception as e:
            self._logger.error(f"Failed to get index stats: {e}")
            return {"total_nodes": "Unknown", "backend": "Unknown"}
    