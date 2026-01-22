from typing import List, Dict, Any
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.utils import get_tokenizer


class Chunking:
    """
    Chunking 类：用于对解析器生成的大节点进行进一步切分，支持不同策略和元数据继承。
    """

    def __init__(self, size: int = 512, overlap: int = 100):
        """
        初始化切分参数。针对 BGE-M3，建议 chunk_size=512~800。

        Args:
            size (int): 每个chunk的大小（token数）。
            overlap (int): chunk之间的重叠大小。
        """
        self.size = size
        self.overlap = overlap

        # 默认使用TokenTextSplitter
        self.default_splitter = TokenTextSplitter(
            chunk_size=self.size,
            chunk_overlap=self.overlap,
            tokenizer=get_tokenizer()
        )

    def run_chunking(self, nodes: List[TextNode]) -> List[TextNode]:
        """
        主逻辑：遍历 Parser 生成的大节点，调用底层切分器进行细分。

        Args:
            nodes (List[TextNode]): 解析器生成的大节点列表。

        Returns:
            List[TextNode]: 切分后的小节点列表。
        """
        chunked_nodes = []

        for node in nodes:
            # 获取适合的splitter
            splitter = self._get_splitter(node)

            # 如果是表格节点，使用特殊处理
            if 'table' in node.metadata.get('section_header', '').lower():
                chunks = self._handle_table_splitting(node.text)
                for i, chunk in enumerate(chunks):
                    enriched_meta = self._enrich_metadata(node.metadata, i)
                    sub_node = TextNode(text=chunk, metadata=enriched_meta)
                    chunked_nodes.append(sub_node)
            else:
                # 使用splitter切分
                # 将TextNode转换为Document-like对象
                from llama_index.core.schema import Document
                doc = Document(text=node.text, metadata=node.metadata)
                sub_nodes = splitter.get_nodes_from_documents([doc])

                # 为每个子节点添加元数据
                for i, sub_node in enumerate(sub_nodes):
                    enriched_meta = self._enrich_metadata(node.metadata, i)
                    sub_node.metadata.update(enriched_meta)
                    chunked_nodes.append(sub_node)

        # 建立节点间的关系
        self._link_nodes(chunked_nodes)

        return chunked_nodes

    def _get_splitter(self, node: TextNode):
        """
        策略分发：根据 Node 来源选择不同的切分算法（如：表格节点用特定逻辑，纯文本用语义切分）。

        Args:
            node (TextNode): 节点对象。

        Returns:
            Splitter: 对应的切分器。
        """
        # 目前简化：所有节点使用默认splitter
        # 可以根据metadata扩展，如node.metadata.get('source')或'level'
        return self.default_splitter

    def _handle_table_splitting(self, text: str) -> List[str]:
        """
        特殊处理：如果一个 Chunk 包含表格，确保切分时带上表头，防止语义丢失。

        Args:
            text (str): 包含表格的文本。

        Returns:
            List[str]: 切分后的chunks，确保每个chunk包含表头。
        """
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        table_header = None

        for line in lines:
            if line.strip().startswith('|') and '|' in line:
                # 表格行
                if table_header is None:
                    # 假设第一行是表头
                    table_header = line
                current_chunk.append(line)

                # 如果chunk大小超过限制，切分
                if len('\n'.join(current_chunk)) > self.size * 4:  # 粗略估算
                    if table_header and current_chunk:
                        chunk_text = table_header + '\n' + '\n'.join(current_chunk)
                        chunks.append(chunk_text)
                        current_chunk = [table_header]  # 保留表头
            else:
                # 非表格行
                current_chunk.append(line)

        # 添加最后一个chunk
        if current_chunk:
            if table_header and current_chunk[0] != table_header:
                chunk_text = table_header + '\n' + '\n'.join(current_chunk)
            else:
                chunk_text = '\n'.join(current_chunk)
            chunks.append(chunk_text)

        return chunks

    def _enrich_metadata(self, parent_meta: Dict, index: int) -> Dict:
        """
        元数据继承：将父节点的 section_path、source 等注入子节点，并添加 chunk_id。

        Args:
            parent_meta (Dict): 父节点元数据。
            index (int): chunk索引。

        Returns:
            Dict: 增强的元数据。
        """
        enriched = parent_meta.copy()
        enriched['chunk_id'] = index
        # 确保关键字段存在
        enriched.setdefault('section_path', parent_meta.get('section_header', 'Unknown'))
        enriched.setdefault('source', parent_meta.get('source', 'Unknown'))
        return enriched

    def _link_nodes(self, nodes: List[TextNode]) -> None:
        """
        关系构建：在 Node 之间建立 next_node 和 prev_node 的引用，方便后续 Window Retrieval。

        Args:
            nodes (List[TextNode]): 节点列表。
        """
        for i in range(len(nodes) - 1):
            # 使用relationships设置前后关系
            nodes[i].relationships['next'] = nodes[i + 1]
            nodes[i + 1].relationships['prev'] = nodes[i]
