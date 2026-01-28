
import logging
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline as LlamaIndexPipeline
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.utils import get_tokenizer
# 引入自定义组件
from .loaders import Loader, Blob
from .parsers.parser_txt import TextParser
from .parsers.parser_pdf import PDFParser
from .parsers.parser_html import HTMLParser
from .parsers.parser_md import MarkdownParser 
from .chunking import Chunking
from models.embedding import get_embed_model, BgeM3Service
from .indexmannager import IndexManager
from config import settings

logger = logging.getLogger(__name__)

class IngestionPipelineWrapper:
    """
    IngestionPipelineWrapper: 
    SpecGrok 项目专用的数据注入管道管理类。
    它组合了 Loader, 解析器, Chunking, Embedding 和 IndexManager，
    并利用 LlamaIndex 的 IngestionPipeline 进行转换流的编排。
    """

    def __init__(self, index_manager: IndexManager, config: Dict[str, Any] = None):
        """
        初始化解析器、分块器、Embedding 模型以及向量库存储。
        
        Args:
            index_manager (IndexManager): 已经初始化好的索引管理器（包含 StorageContext）。
            config (Dict[str, Any]): 配置字典。
        """
        self.config = config or {}
        self.index_manager = index_manager
        
        # 初始化组件
        self.loader = Loader(
            extensions=self.config.get('INGESTION_EXTENSIONS', ['.txt', '.md', '.html', '.pdf']),
            recursive=self.config.get('INGESTION_RECURSIVE', True)
        )
        
        # 实例化解析器
        # 注意：LlamaIndex 的 IngestionPipeline 要求 transformations 列表中的对象是特定类型
        # 这里我们的 parser 是 BaseReader，通常是在 pipeline 之外先将文件转为 [Documents]，
        # 或者使用 TransformComponent 包装。
        # 为了最大化利用我们自定义 Parser 的逻辑，我们将采用以下流程：
        # Loader -> [Blob] -> 自定义逻辑转为 -> [BaseNode/TextNode] (解析阶段) -> IngestionPipeline (分块+Embedding+入库)
        
        self.txt_parser = TextParser(config=self.config)
        self.html_parser = HTMLParser(
            remove_images=self.config.get('REMOVE_IMAGES', True),
            remove_links=self.config.get('REMOVE_LINKS', False)
        )
        self.md_parser = MarkdownParser(remove_images=self.config.get('REMOVE_IMAGES', True)) # 假设已有实现
        self.pdf_parser = PDFParser(vlm_enabled=self.config.get('PDF_VLM_ENABLED', False))
        
        # 分块器
        self.chunking = Chunking(
            size=self.config.get('CHUNK_SIZE', 512),
            overlap=self.config.get('CHUNK_OVERLAP', 100)
        )
        
        # 统计信息
        self.stats = {
            "processed_files": 0,
            "generated_chunks": 0,
            "total_tokens": 0,
            "errors": 0
        }

    def run(self, source_path: str) -> List[BaseNode]:
        """
        主入口：驱动整个流程。
        流程：
        1. Loader 读取文件得到 Blobs。
        2. 根据文件类型调用对应的 Parser，得到初步的 TextNodes (Section Level)。
        3. 调用 Chunking 对大节点进行细分。
        4. 调用 IndexManager.upsert_nodes 进行 Embedding 和 入库。
        
        Args:
            source_path (str): 文件或目录路径。
            
        Returns:
            List[BaseNode]: 最终生成的 Chunks 列表。
        """
        logger.info(f"Starting ingestion pipeline for: {source_path}")
        
        try:
            # 1. Load Files -> Blobs
            blobs = self.loader.load(source_path)
            self.stats["processed_files"] += len(blobs)
            logger.info(f"Loaded {len(blobs)} files.")
            
            # 使用内部处理方法处理 blobs
            return self._process_blobs(blobs)

        except Exception as e:
            self._handle_errors(e, context="Pipeline Run")
            return []

    def _process_blobs(self, blobs: List[Blob]) -> List[BaseNode]:
        """具体的处理逻辑拆分"""
        all_nodes: List[TextNode] = []
        
        # 2. Parse Blobs -> Large Nodes (Sections)
        for blob in blobs:
            try:
                nodes = self._parse_blob(blob)
                all_nodes.extend(nodes)
            except Exception as e:
                self._handle_errors(e, context=f"Parsing {blob.source}")
        
        logger.info(f"Parsed into {len(all_nodes)} section nodes.")
        
        # 3. Chunking (Custom Logic) -> Small Nodes
        # 我们自定义的 Chunking 逻辑比较复杂（包含元数据继承、特殊块保护等），
        # 且它的输入输出都是 List[TextNode]，符合 Transform 的概念。
        chunked_nodes = self.chunking.run_chunking(all_nodes)
        self.stats["generated_chunks"] += len(chunked_nodes)
        logger.info(f"Chunked into {len(chunked_nodes)} granular nodes.")

        # 使用默认 tokenizer（通常是 gpt-3.5-turbo 兼容）进行计数
        try:
            tokenizer = get_tokenizer()
            token_count = 0
            for n in chunked_nodes:
                    # 获取用于 Embedding 的内容（通常包含 metadata）
                    content = n.get_content(metadata_mode="embed")
                    token_count += len(tokenizer(content))
            
            self.stats["total_tokens"] += token_count
            logger.info(f"Estimated token usage for this run: {token_count}")
        except Exception as e:
            logger.warning(f"Token counting failed: {e}")
        
        # 4. Upsert to Vector Store (Embedding + Indexing)
        # 这一步交由 IndexManager 处理
        self.index_manager.upsert_nodes(chunked_nodes)
        
        return chunked_nodes


    def sync_with_source(self, directory: str) -> None:
        """
        增量同步：对比本地文件与向量库，仅处理新增或修改的文件（基于 Hash）。
        
        利用 Loader 在读取时生成的 file_hash，结合 IndexManager.refresh_index 进行筛选，
        跳过未变化的文件，极大节省 Parsing 和 Embedding 开销。
        """
        logger.info(f"Syncing source directory: {directory}")
        
        try:
            # 1. Load Files (now with file_hash in metadata)
            blobs = self.loader.load(directory)
            logger.info(f"Scanned {len(blobs)} files from source.")
            
            if not blobs:
                logger.warning("No files found to sync.")
                return

            # 2. Prepare Virtual Documents for Hash Checking
            # 将 Blob 转为仅包含 ID 和 Hash 的轻量级 Document 对象，传给 IndexManager 进行比对
            blob_map = {}
            docs_to_check = []
            
            for blob in blobs:
                # 使用 doc_id 哈希作为 doc_id
                doc_id = blob.metadata['doc_id']
                blob_map[doc_id] = blob
                
                # 构造虚拟 Doc (IndexManager will read metadata['doc_id'])
                doc = Document(text="", id_=doc_id, metadata=blob.metadata)
                docs_to_check.append(doc)
            
            # 3. Identify Changed/New Files
            # refresh_index 会返回需要重新注入的 doc_ids，并自动清理旧版本的 chunks
            ids_to_reingest = self.index_manager.refresh_index(docs_to_check)
            
            if not ids_to_reingest:
                logger.info("No changes detected. All files are up-to-date.")
                return 

            logger.info(f"Detected {len(ids_to_reingest)} files needing update/ingestion.")
            
            # 4. Filter Blobs
            blobs_to_process = []
            for doc_id in ids_to_reingest:
                if doc_id in blob_map:
                    blobs_to_process.append(blob_map[doc_id])
                else:
                    logger.warning(f"Doc ID {doc_id} returned by refresh but not found in current blobs.")

            # 5. Process Filtered Blobs
            if blobs_to_process:
                self._process_blobs(blobs_to_process)
                
        except Exception as e:
            self._handle_errors(e, context="Sync With Source")


    def _parse_blob(self, blob: Blob) -> List[TextNode]:
        """根据扩展名分发到不同的 Parser"""
        ext = blob.metadata.get('extension', '').lower()
        
        if ext == '.html':
            return self.html_parser.parse(blob)
        elif ext == '.md':
            return self.md_parser.parse(blob)
        elif ext == '.txt':
            return self.txt_parser.parse(blob)
        elif ext == '.pdf':
            return self.pdf_parser.parse(blob)
        else:
            logger.warning(f"No parser for extension {ext}, skipping {blob.source}")
            return []

    def _handle_errors(self, exception: Exception, context: str = "") -> None:
        """
        鲁棒性：处理解析失败、网络中断、API 额度超限等情况。
        """
        self.stats["errors"] += 1
        logger.error(f"Error in {context}: {str(exception)}", exc_info=True)
        # 根据错误类型决定是否抛出或继续
        # 例如网络错误可能需要重试，解析错误则记录日志跳过

    def get_stats(self) -> Dict[str, Any]:
        """
        监控：返回处理了多少文件、生成了多少 Chunk、消耗了多少 Token。
        """
        return self.stats