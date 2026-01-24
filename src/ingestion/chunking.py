from typing import List, Dict, Any, Optional
import re
import hashlib
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo, Document
from llama_index.core.node_parser import SentenceSplitter

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

        # 使用 SentenceSplitter: 相比 TokenTextSplitter，在句子边界处理上效果更好
        self.text_splitter = SentenceSplitter(
            chunk_size=self.size,
            chunk_overlap=self.overlap
        )
        
        # 定义代码块及数学公式的正则模式 (保护模式)
        self.code_block_pattern = re.compile(r"(```[\s\S]*?```)")
        self.math_block_pattern = re.compile(r"(\$\$[\s\S]*?\$\$)")

    def run_chunking(self, nodes: List[TextNode]) -> List[TextNode]:
        """
        主逻辑：遍历 Parser 生成的大节点，调用底层切分器进行细分。
        修正：修复 Splitter 默认将 Parent 指向临时 Document 的问题。

        Args:
            nodes (List[TextNode]): 解析器生成的大节点列表。

        Returns:
            List[TextNode]: 切分后的小节点列表。
        """
        chunked_nodes = []

        for node in nodes:
            # 获取适合的splitter
            splitter = self._get_splitter(node)
            
            # 判断是否需要特殊处理 (表格 或 混合代码/公式)
            is_table = 'table' in node.metadata.get('section_header', '').lower()
            # 简单启发式：检查是否有代码块或公式标记
            has_special_blocks = '```' in node.text or '$$' in node.text

            if is_table:
                chunks = self._handle_table_splitting(node.text)
                self._create_nodes_from_chunks(chunks, node, chunked_nodes)
                
            elif has_special_blocks:
                # 处理混合内容（文本 + 代码/公式）
                chunks = self._handle_mixed_content_splitting(node.text)
                self._create_nodes_from_chunks(chunks, node, chunked_nodes)
                
            else:
                # 使用splitter切分
                # 将TextNode转换为Document-like对象 (LlamaIndex Splitter 需要 Document 或 文本)
                doc = Document(text=node.text, metadata=node.metadata)
                sub_nodes = splitter.get_nodes_from_documents([doc])

                for i, sub_node in enumerate(sub_nodes):
                    # A. 修正 Parent 指向：Splitter 默认指向中间生成的 doc，需改回指向原 node
                    sub_node.relationships[NodeRelationship.PARENT] = node.as_related_node_info()

                    # A2. 继承 Source 指向：如果父节点有 SOURCE (通常指向原始文件)，子节点也应继承
                    if NodeRelationship.SOURCE in node.relationships:
                        sub_node.relationships[NodeRelationship.SOURCE] = node.relationships[NodeRelationship.SOURCE]
                    
                    # B. 补充自定义元数据
                    sub_node.metadata['chunk_id'] = i
                    
                    # C. 生成确定性 ID (Chunk 阶段 Hash)
                    sub_node.node_id = self._generate_node_id(sub_node.text, sub_node.metadata)
                    
                    chunked_nodes.append(sub_node)

        # 建立节点间的关系
        self._link_nodes(chunked_nodes)

        return chunked_nodes
        
    def _create_nodes_from_chunks(self, chunks: List[str], source_node: TextNode, output_list: List[TextNode]):
        """辅助方法：将字符串chunks封装为Nodes并添加到输出列表"""
        for i, chunk in enumerate(chunks):
            # 丰富元数据
            enriched_meta = self._enrich_metadata(source_node.metadata, i)
            sub_node = TextNode(text=chunk, metadata=enriched_meta)
            
            # 生成确定性 ID (Chunk 阶段 Hash)
            sub_node.node_id = self._generate_node_id(chunk, enriched_meta)

            # 显式建立与原 Node 的父子关系
            sub_node.relationships[NodeRelationship.PARENT] = source_node.as_related_node_info()

            # 继承 Source 关系
            if NodeRelationship.SOURCE in source_node.relationships:
                sub_node.relationships[NodeRelationship.SOURCE] = source_node.relationships[NodeRelationship.SOURCE]

            output_list.append(sub_node)

    def _get_splitter(self, node: TextNode):
        """
        策略分发：根据 Node 来源选择不同的切分算法（如：表格节点用特定逻辑，纯文本用语义切分）。

        Args:
            node (TextNode): 节点对象。

        Returns:
            Splitter: 对应的切分器。
        """
        # 纯文本通常使用 SentenceSplitter 效果更好
        return self.text_splitter

    def _handle_mixed_content_splitting(self, text: str) -> List[str]:
        """
        处理混合内容的切分：
        1. 识别代码块和数学公式块，将其临时替换为短占位符。
        2. 使用 SentenceSplitter 对含有占位符的文本进行切分（利用其 keeping short words together 的特性实现 Block 保护）。
        3. 还原占位符。
        4. (可选) 如果还原后的块过大，再进行二次降级切分。
        """
        placeholders = {}
        counter = 0

        # 正则替换回调
        def replace_match(match):
            nonlocal counter
            key = f"__SPECIAL_BLOCK_{counter}__"
            placeholders[key] = match.group(0)
            counter += 1
            return key

        # 先保护代码块，再保护公式块
        masked_text = self.code_block_pattern.sub(replace_match, text)
        masked_text = self.math_block_pattern.sub(replace_match, masked_text)

        # 使用 SentenceSplitter 切分 Masked Text
        # 注意：这里我们期望 Splitter 把 __SPECIAL_BLOCK_N__ 当作一个整体（单词或短句）处理
        raw_chunks = self.text_splitter.split_text(masked_text)

        final_chunks = []
        for chunk in raw_chunks:
            # 还原
            restored_chunk = chunk
            # 仅还原当前 chunk 包含的 keys (避免无效遍历，虽然 placeholders 也不大)
            # 简单起见，按顺序替换即可，或者正则替换回来
            for key, val in placeholders.items():
                if key in restored_chunk:
                    restored_chunk = restored_chunk.replace(key, val)
            
            # 检查还原后的大小
            # 注意：self.size 是 token 数。这里为了性能和无需引入 tokenizer，可以用字符数估算。
            # 假设 1 token ~= 3 chars (中文) 或 4 chars (英文)
            # 宽松一点： self.size * 4
            estimated_limit = self.size * 4
            
            if len(restored_chunk) > estimated_limit:
                # 如果还原后因为 Block 太大导致超限，则必须对这个 chunk 进行再次切分
                # 这种情况下，我们失去了 Block 的完整性保护，因为 Block 本身就太大，必须切分。
                # 再次调用 splitter (fallback)
                sub_chunks = self.text_splitter.split_text(restored_chunk)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(restored_chunk)

        return final_chunks

    def _handle_table_splitting(self, text: str) -> List[str]:
        """
        特殊处理：如果一个 Chunk 包含表格，确保切分时带上表头，防止语义丢失。
        改进：优化切分条件，避免重复表头，处理密集表格。

        Args:
            text (str): 包含表格的文本。

        Returns:
            List[str]: 切分后的chunks，确保每个chunk包含表头。
        """
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        table_header = None
        
        # 优化B：使用字符数估算 (self.size tokens * 3 chars/token) 替代硬编码
        # 或者使用行数计数（这里保留字符数作为一个近似token限制的安全网）
        max_chars = self.size * 3

        for line in lines:
            line = line.rstrip() # 保留左侧可能的缩进
            if not line: continue 

            if table_header is None and line.strip().startswith('|'):
                table_header = line
            
            current_chunk.append(line)
            
            # 计算 buffer 大小
            current_size = sum(len(l) for l in current_chunk)
            
            # 达到阈值进行切分
            if current_size > max_chars:
                # 优化A：避免双表头
                # 如果当前积累的 chunk 第一行已经是表头，就不再 prepend table_header
                # 否则（比如 chunk 是表格中间的几行），则 prepend table_header
                final_lines = []
                if table_header and current_chunk and current_chunk[0] != table_header:
                    final_lines = [table_header] + current_chunk
                else:
                    final_lines = current_chunk
                
                chunks.append('\n'.join(final_lines))
                current_chunk = [] # 清空

        # 处理剩余内容
        if current_chunk:
            final_lines = []
            if table_header and current_chunk and current_chunk[0] != table_header:
                final_lines = [table_header] + current_chunk
            else:
                final_lines = current_chunk
            chunks.append('\n'.join(final_lines))

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
        关系构建：规范化：使用 NodeRelationship 枚举在 Node 之间建立关系。
        
        Args:
            nodes (List[TextNode]): 节点列表。
        """
        for i in range(len(nodes) - 1):
            current_node = nodes[i]
            next_node = nodes[i + 1]
            
            # 使用官方 NodeRelationship 枚举
            current_node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                node_id=next_node.node_id,
                metadata=next_node.metadata
            )
            next_node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                node_id=current_node.node_id, 
                metadata=current_node.metadata
            )

    def _generate_node_id(self, text: str, metadata: Dict) -> str:
        """
        生成确定性的 Node ID (Chunk Hash)。
        
        逻辑：md5(source | md5(content))
        确保只要源文件路径不变且切分出的文本内容不变，生成的 ID 就永远一致。
        这是实现向量库幂等性(Idempotency)的关键步骤。
        """
        source = metadata.get('source', '') or metadata.get('file_name', '')
        
        # 使用文本内容的 MD5 hash 作为唯一性依据
        content_hash = hashlib.md5((text or '').encode('utf-8')).hexdigest()
        
        # 组合 Key
        node_key = f"{source}|{content_hash}"
        
        return hashlib.md5(node_key.encode('utf-8')).hexdigest()

