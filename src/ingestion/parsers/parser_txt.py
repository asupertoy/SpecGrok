import re
import chardet
import os
from typing import List, Dict, Any, Tuple
from pathlib import Path

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from src.ingestion.loaders import Blob


class TextParser(BaseReader):
    """
    TextParser 类：用于解析纯文本文件，支持编码检测、文本清洗、段落识别和分块。
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化清洗规则（如：是否合并短行、编码检测优先级）。

        Args:
            config (Dict[str, Any]): 配置字典，包含选项如 merge_short_lines, encoding_priority 等。
        """
        super().__init__()
        self.config = config or {}
        self.merge_short_lines = self.config.get('merge_short_lines', True)
        self.encoding_priority = self.config.get('encoding_priority', ['utf-8', 'gbk', 'ansi'])

    def parse(self, blob: Blob) -> List[TextNode]:
        """
        主调度：读取 -> 编码修复 -> 逻辑段落识别 -> 切分 -> 返回 Node。

        Args:
            blob (Blob): 包含文本数据的 Blob 对象。

        Returns:
            List[TextNode]: 解析后的文本节点列表。
        """
        print(f"TextParser 正在解析 Blob: {blob.source} ...")
        
        # 设置源信息
        self.source = blob.source

        # 读取字节数据
        data = blob.data

        # 检测编码
        encoding = self._detect_encoding(data)

        # 解码文本
        text = data.decode(encoding, errors='replace')

        # 清洗文本
        text = self._clean_text(text)

        # 切分sections（改为直接切分，无需先识别段落）
        sections = self._split_into_sections(text)

        # 构建节点
        nodes = []
        for section in sections:
            node = self._build_node(section['content'], section, blob)
            nodes.append(node)

        return nodes

    def load_data(self, file: str) -> List[TextNode]:
        """
        兼容入口：读取本地文本文件并调用 parse。

        Args:
            file (str): 文件路径。

        Returns:
            List[TextNode]: 解析后的文本节点列表。
        """
        with open(file, 'rb') as f:
            data = f.read()
        blob = Blob(data=data, source=file)
        return self.parse(blob)

    def _detect_encoding(self, data: bytes) -> str:
        """
        鲁棒性保证：检测文件编码（UTF-8/GBK/ANSI），防止乱码。

        Args:
            data (bytes): 字节数据。

        Returns:
            str: 检测到的编码。
        """
        result = chardet.detect(data)
        detected = result.get('encoding')
        if detected and detected.lower() in [e.lower() for e in self.encoding_priority]:
            return detected
        # 默认回退到 UTF-8
        return 'utf-8'

    def _clean_text(self, text: str) -> str:
        """
        降噪：去除无意义的控制字符、多余的空白行、统一换行符（CRLF to LF）、无效标记等。

        另外：如果配置了 `merge_short_lines`，会合并短行以恢复被硬换行分割的句子。

        Args:
            text (str): 原始文本。

        Returns:
            str: 清洗后的文本。
        """
        # 统一换行符
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # 去除控制字符（保留换行和制表符）
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)

        # 去除多余的空白行（连续3个以上换行符替换为2个）
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

        # 去除无效标记（类似MarkdownParser的_clean_placeholders）
        text = self._clean_placeholders(text)

        # 合并短行（可选，支持自适应阈值或配置阈值）
        if self.merge_short_lines:
            lines = text.split('\n')

            # 阈值优先来自配置，否则根据文本行长度中位数自适应计算
            threshold = self.config.get('merge_short_line_threshold')
            if threshold is None:
                import statistics
                lengths = [len(l.strip()) for l in lines if l.strip()]
                if lengths:
                    median_len = int(statistics.median(lengths))
                    # 使用中位数的 0.7 倍作为短行阈值，限制在 [30, 200] 区间
                    threshold = max(30, min(200, int(median_len * 0.7)))
                else:
                    threshold = 80  # 回退默认值

            merged_lines: List[str] = []
            current = ''
            in_code_block = False # 状态标记：是否在代码块内

            for line in lines:
                # 获取去除首尾空格的版本用于判断，但保留原行 line 用于代码块
                s = line.strip()

                # --- 1. 代码块保护逻辑 ---
                # 检测代码块边界 (```)
                if s.startswith('```'):
                    # 如果有正在累积的普通文本，先结算
                    if current:
                        merged_lines.append(current.strip())
                        current = ''
                    
                    # 切换状态，并按原样保留该行（保留缩进）
                    in_code_block = not in_code_block
                    merged_lines.append(line.rstrip()) # 仅去除右侧换行符，保留左侧缩进
                    continue
                
                # 如果在代码块内，直接按原样保留，不做合并也不做 strip
                if in_code_block:
                    merged_lines.append(line.rstrip())
                    continue
                
                # --- 以下为普通文本处理逻辑 ---

                # 保持段落分隔（空行）
                if not s:
                    if current:
                        merged_lines.append(current.strip())
                        current = ''
                    merged_lines.append('')
                    continue

                # 如果行看起来像列表项或标题，作为新行处理
                # 表格行 (s.startswith('|')) 也在此处被保护，不参与合并
                if re.match(r'^(\d+\.|\-|\*|\+|•)\s', s) or self._is_potential_header(s) or s.startswith('|'):
                    if current:
                        merged_lines.append(current.strip())
                        current = ''
                    merged_lines.append(s)
                    continue

                # 合并逻辑：短行且前一段未以标点结束则合并
                if not current:
                    current = s
                else:
                    # 连字符断行处理（例如 commu-\nnication）
                    if current.endswith('-'):
                        current = current[:-1] + s
                    # 若当前行短且前一行不以结束标点结束，则合并
                    # 不把逗号视作句子终止符，允许以逗号结尾的行继续合并
                    elif len(s) < threshold and not re.search(r'[。\.\!\?！\?：;；:]$', current):
                        current = current + ' ' + s
                    else:
                        merged_lines.append(current.strip())
                        current = s

            if current:
                merged_lines.append(current.strip())

            # 重建文本，保留空行作为段落分隔
            text = '\n'.join(merged_lines)

            # 再次简化多余空白行
            text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

        return text.strip()

    def _clean_placeholders(self, text: str) -> str:
        """清洗：去除无效的标记、注释、YAML Frontmatter 及 HTML 标签。"""
        # 1. 去除 YAML Frontmatter (位于文件开头的 --- ... ---)
        text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL)
        
        # 2. 去除 HTML 注释 <!-- ... -->
        text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
        
        # 3. 去除 HTML 标签 (保留内容)
        text = re.sub(r"<[^>]+>", "", text)
        
        return text


    def _is_potential_header(self, line: str) -> bool:
        """
        语义发现：识别可能是标题的行（如"第X章"、短行带数字、全大写行等）。

        Args:
            line (str): 单行文本。

        Returns:
            bool: 是否可能是标题。
        """
        line = line.strip()
        if not line:
            return False

        # 中文章节标题（如"第X章"）
        if re.match(r'第[一二三四五六七八九十\d]+章', line):
            return True

        # 全大写英文标题
        if line.isupper() and len(line) > 3:
            return True

        # 罗马数字章节（如I. II. III.）
        if re.match(r'^[IVXLCDM]+\.\s', line):
            return True

        # Markdown 风格标题 (兼容在TXT中写MD的情况)
        if re.match(r'^#+\s', line):
            return True

        # [Removed] 短行且包含数字的规则过于宽泛，容易误判包含公式或年份的普通句子
        # 替代方案：匹配以数字开头的标题格式 (如 "1.1 背景")
        if len(line) < 50 and re.match(r'^\d+(\.\d+)*\s', line):
            return True

        # 其他潜在标题：以大写字母开头的短行，或包含特定关键词
        if len(line) < 30 and re.match(r'^[A-Z]', line):
            return True

        # 中文标题：短行，包含中文字符，可能以章节关键词开头
        if len(line) <= 25 and re.search(r'[\u4e00-\u9fff]', line):
            # 检查是否包含章节关键词（作为词干或行首）
            chapter_keywords = ['章', '节', '部分', '小结', '概述', '前言', '后记', '目录', '摘要']
            if any(line.startswith(keyword) or f'{keyword} ' in line for keyword in chapter_keywords):
                return True
            # 或者检查特定标题模式（关键词后跟空格或行结束，且行不包含标点）
            title_patterns = ['主要', '功能', '特性', '方法', '结果', '讨论', '结论']
            for pattern in title_patterns:
                if pattern in line:
                    # 检查pattern后是否跟空格或行结束，且行不包含句子标点
                    idx = line.find(pattern)
                    if idx >= 0:
                        after_pattern = line[idx + len(pattern):]
                        if (not after_pattern or after_pattern.startswith(' ')) and not re.search(r'[。！？，、；：]', line):
                            return True

        return False

    def _split_into_sections(self, text: str) -> List[Dict]:
        """
        分段：利用标题识别结果，将文本模仿 Markdown 划分为带标题的 Section。
        改进：添加块保护逻辑，防止在代码块、数学块、表格中误识别标题。

        Args:
            text (str): 清洗后的文本。

        Returns:
            List[Dict]: 包含文本和元数据的section字典列表。
        """
        lines = text.split('\n')
        sections = []
        
        current_lines = []
        current_header = "Introduction"  # 默认首部内容的标题
        current_level = 1  # 默认层级
        
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
            if stripped.startswith("|"):
                table_block, new_i = self._consume_table(lines, i)
                current_lines.extend(table_block)
                i = new_i
                continue

            # --- 标题检测 (Header Detection) ---
            if self._is_potential_header(stripped):
                # 只有当 accumulator buffer 有内容时，才封存上一章节
                if current_lines or current_header != "Introduction":
                    sections.append({
                        "header": current_header,
                        "level": current_level,
                        "content": "\n".join(current_lines).strip()
                    })
                
                # 开启新章节
                current_level = 1  # 纯文本假设所有标题都是H1
                current_header = stripped
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
        buffer = [lines[start_idx]]
        for j in range(start_idx + 1, len(lines)):
            buffer.append(lines[j])
            if lines[j].strip().startswith(end_marker):
                return buffer, j + 1  # 返回新索引
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

    def _build_node(self, text: str, section: Dict, blob: Blob) -> TextNode:
        """
        组装：封装为 LlamaIndex 节点，注入 source, line_range 等元数据。

        Args:
            text (str): 节点文本。
            section (Dict): section字典，包含header, level等。
            blob (Blob): Blob对象，用于获取基础元数据。

        Returns:
            TextNode: LlamaIndex 文本节点。
        """
        # 构造基础 metadata
        base_metadata = blob.metadata.copy()
        
        # 处理可能的字段重命名或提取
        if "domain" not in base_metadata and "category" in base_metadata:
            base_metadata["domain"] = base_metadata.pop("category")
        elif "domain" not in base_metadata:
            base_metadata["domain"] = None

        # 确保 file_name 存在
        if "file_name" not in base_metadata:
            base_metadata["file_name"] = Path(blob.source).name

        # 生成section_path
        section_path = section.get('header', 'Introduction')

        # 合并元数据
        base_metadata.update({
            "section_header": section.get('header', 'Introduction'),
            "section_path": section_path,
            "level": section.get('level', 1),
            "source": blob.source
        })

        # 计算行范围
        line_count = len(text.split('\n'))
        base_metadata['line_range'] = f"1-{line_count}"

        node = TextNode(text=text, metadata=base_metadata)
        
        # [Fix] 设置 Source 关系指向原始文件 (blob.source)
        # 这确保了后续 Chunking 阶段生成的子节点能通过 ref_doc_id 关联回文件路径，
        # 从而保证 IndexManager.delete_by_doc_id(path) 能正确删除相关向量。
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
            node_id=blob.source,
            metadata={"file_name": base_metadata.get("file_name")}
        )
        
        return node
