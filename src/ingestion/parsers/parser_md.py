from typing import List, Dict, Optional, Tuple
import re
from pathlib import Path

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from ingestion.loaders import Blob

class MarkdownParser(BaseReader):
    def __init__(self, remove_images: bool = True):
        """
        Args:
            remove_images: 是否从文本中移除图片链接（但仍会提取到 metadata）。
        """
        super().__init__()
        self.remove_images = remove_images

    def load_data(self, file: str) -> List[TextNode]:
        """兼容入口：读取本地 .md 文件并触发 parse。"""
        blob = Blob.from_path(file)
        return self.parse(blob)

    def parse(self, blob: Blob) -> List[TextNode]:
        """主调度：将 Markdown 文本读入，调用切分逻辑，返回 Node 列表。"""
        print(f"MarkdownParser 正在解析 Blob: {blob.source} ...")
        
        text = str(blob.data.decode("utf-8")) if isinstance(blob.data, bytes) else blob.data
        if not text:
            return []

        # 1. 预清洗
        text = self._clean_placeholders(text)
        
        # 1.1 链接清洗 (建议：保留文本，移除 Markdown 内部跳转链接)
        # 匹配: [Link Text](#anchor) -> Link Text
        text = re.sub(r"\[([^\]]+)\]\(#[^)]+\)", r"\1", text)

        # 2. 结构化切分
        sections = self._split_by_headers(text)
        
        # 3. 构建节点（维护层级栈）
        nodes = []
        # stack item: (level: int, title: str)
        # 初始化一个根节点，以防没有标题的文档，level设为0
        headers_stack: List[Tuple[int, str]] = [] 

        for section_dict in sections:
            level = section_dict["level"]
            header_text = section_dict["header"]
            content = section_dict["content"]

            # 维护栈：弹出所有层级 >= 当前层级的标题，找到父级
            while headers_stack and headers_stack[-1][0] >= level:
                headers_stack.pop()
            
            headers_stack.append((level, header_text))
            
            # 只有当内容不为空时才创建节点
            if content.strip():
                # 生成层级路径
                section_path = self._get_section_path(headers_stack)
                
                # 提取图片并（可选）从原文中移除
                images = self._extract_images(content)
                if self.remove_images:
                     content = re.sub(r"!\[.*?\]\(.*?\)", "", content)

                node = self._build_node(
                    text=content,
                    metadata={
                        "section_header": header_text, # 当前块的直接标题
                        "section_path": section_path,   # 完整路径
                        "images": images,
                        "level": level
                    },
                    blob=blob
                )
                nodes.append(node)

        return nodes

    def _split_by_headers(self, text: str) -> List[Dict]:
        """核心逻辑：基于 # 标题层级切分文档。
        
        改进：参考 deepdoc/parser/markdown_parser.py 逻辑
        增加：表格保护 (Table Protection)、公式块保护 (Math Block Protection)、代码块保护
        """
        lines = text.split('\n')
        sections = []
        
        current_lines = []
        current_header = "Introduction" # 默认首部内容的标题
        current_level = 0               # 默认层级（小于 H1=1）
                
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # --- 块保护逻辑 (Block Protection) ---
            # 策略：一旦进入保护区，贪婪读取直到结束，不进行 Header 检测
            
            # 1. 代码块 (Code Block)
            if stripped.startswith("```"):
                code_block, new_i = self._consume_block(lines, i, end_marker="```")
                current_lines.extend(code_block)
                i = new_i
                continue
                
            # 2. 数学公式块 (Math Block)
            if stripped.startswith("$$"):
                math_block, new_i = self._consume_block(lines, i, end_marker="$$")
                current_lines.extend(math_block)
                i = new_i
                continue

            # 3. 表格 (Table)
            # 简单的表格识别：以 | 开头
            if stripped.startswith("|"):
                table_block, new_i = self._consume_table(lines, i)
                current_lines.extend(table_block)
                i = new_i
                continue

            # --- 标题检测 (Header Detection) ---
            match = re.match(r"^(#+)\s+(.*)", line)
            if match:
                # 只有当 accumlator buffer 有内容时，才封存上一章节
                if current_lines or current_header != "Introduction":
                    sections.append({
                        "header": current_header,
                        "level": current_level,
                        "content": "\n".join(current_lines).strip()
                    })
                
                # 开启新章节
                current_level = len(match.group(1))
                current_header = match.group(2).strip()
                current_lines = []
                # 标题本身也加入内容，保持上下文；增加空行以避免 Embedding 语义粘连
                current_lines.append(line)
                current_lines.append("")
            else:
                # 普通行
                current_lines.append(line)
            
            i += 1
        
        # Flush last section
        if current_lines:
            sections.append({
                "header": current_header,
                "level": current_level,
                "content": "\n".join(current_lines).strip()
            })
            
        return sections

    def _consume_block(self, lines: List[str], start_idx: int, end_marker: str) -> Tuple[List[str], int]:
        """通用块读取：从 start_idx 开始读取，直到行首出现 end_marker 或文件结束。"""
        # buffer = [lines[start_idx]]
        # 修正逻辑：buffer 从 start_idx 开始
        # 循环应从 start_idx + 1 开始检测 end_marker
        
        buffer = [lines[start_idx]]
        # 从下一行开始找结束标记
        for j in range(start_idx + 1, len(lines)):
            buffer.append(lines[j])
            if lines[j].strip().startswith(end_marker):
                return buffer, j + 1 # 返回新索引（指向块后的下一行）
        
        # 未闭合，读到尾
        return buffer, len(lines)

    def _consume_table(self, lines: List[str], start_idx: int) -> Tuple[List[str], int]:
        """表格读取：连续读取以 | 开头的行。"""
        buffer = [lines[start_idx]]
        next_idx = start_idx + 1
        while next_idx < len(lines):
            line = lines[next_idx]
            if line.strip().startswith("|"):
                buffer.append(line)
                next_idx += 1
            else:
                break
        return buffer, next_idx

    def _get_section_path(self, headers_stack: List[Tuple[int, str]]) -> str:
        """溯源：生成当前块的路径（例如 简介 > 硬件规格 > 雷达天线）。"""
        # 过滤掉 level 0 的 "Introduction" 虚拟根节点，除非它是唯一的路劲
        # 如果 headers_stack 只有 [ (0, 'Intro') ]，则保留
        # 否则只取 headers_stack 中 level > 0 的部分
        
        titles = [h[1] for h in headers_stack if h[0] > 0]
        if not titles and headers_stack:
             return headers_stack[0][1] # Fallback to Intro if that's all we have
             
        return " > ".join(titles)

    def _extract_images(self, text: str) -> List[str]:
        """资源识别：提取 ![]() 语法中的图片路径，存入元数据。"""
        # 匹配 ![alt](href "optional title")
        # 简化版正则，匹配括号内的内容
        return re.findall(r"!\[.*?\]\((.*?)(?:\s+\".*?\")?\)", text)

    def _clean_placeholders(self, text: str) -> str:
        """清洗：去除无效的 Markdown 标记、注释、YAML Frontmatter 及 HTML 标签。"""
        # 1. 去除 YAML Frontmatter (位于文件开头的 --- ... ---)
        text = re.sub(r"^\s*---\n.*?\n---\n", "", text, flags=re.DOTALL)
        
        # 2. 去除 HTML 注释 <!-- ... -->
        text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
        
        # 3. 去除 HTML 标签 (保留内容)
        # 例如 <div class="content">text</div> -> text
        text = re.sub(r"<[^>]+>", "", text)
        
        return text

    def _build_node(self, text: str, metadata: Dict, blob: Blob) -> TextNode:
        """组装：将切好的文本和层级元数据封装为 LlamaIndex 节点。"""
        # 构造基础 metadata（遵循 Parser PDF 的最佳实践）
        base_metadata = blob.metadata.copy()
        
        # 处理可能的字段重命名或提取 (类似 PDF Parser 逻辑)
        if "domain" not in base_metadata and "category" in base_metadata:
             base_metadata["domain"] = base_metadata.pop("category")
        elif "domain" not in base_metadata:
             base_metadata["domain"] = None

        # 确保 file_name 存在
        if "file_name" not in base_metadata:
             base_metadata["file_name"] = Path(blob.source).name

        # 合并特定元数据
        base_metadata.update(metadata)
        # 补充 source
        base_metadata["source"] = blob.source

        node = TextNode(text=text, metadata=base_metadata)
        
        # [Fix] 设置 Source 关系指向 doc_id（内容哈希）
        doc_id = base_metadata.get("doc_id") or blob.source
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
            node_id=doc_id,
            metadata={"file_name": base_metadata.get("file_name"), "source": blob.source}
        )
        
        return node
