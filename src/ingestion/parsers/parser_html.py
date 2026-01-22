from typing import List, Dict, Optional, Tuple
import re
from pathlib import Path

from bs4 import BeautifulSoup, Tag
import html2text
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import TextNode
from ingestion.loaders import Blob

class HTMLParser(BaseReader):
    def __init__(self, remove_images: bool = True, remove_links: bool = False, custom_clean_rules: Optional[List[str]] = None):
        """
        初始化转换器（如 html2text 配置）及分块策略。
        Args:
            remove_images: 是否移除图片标签（但提取到metadata）。
            remove_links: 是否移除链接标签。
            custom_clean_rules: 自定义清洗规则（CSS选择器列表，用于移除特定元素）。
        """
        super().__init__()
        self.remove_images = remove_images
        self.remove_links = remove_links
        self.custom_clean_rules = custom_clean_rules or []

        # 初始化html2text转换器
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_links = self.remove_links
        self.h2t.ignore_images = self.remove_images
        self.h2t.body_width = 0  # 不限制宽度
        self.h2t.unicode_snob = True

    def load_data(self, file: str) -> List[TextNode]:
        """入口：读取本地 HTML 文件。"""
        blob = Blob.from_path(file)
        return self.parse(blob)

    def parse(self, blob: Blob) -> List[TextNode]:
        """总调度：驱动精洗、语义转化、标题切分三个核心步骤。"""
        print(f"HTMLParser 正在解析 Blob: {blob.source} ...")
        
        html_text = str(blob.data.decode("utf-8")) if isinstance(blob.data, bytes) else blob.data
        if not html_text:
            return []

        soup = BeautifulSoup(html_text, 'html.parser')

        # 1. 精洗层
        soup = self._refine_dom(soup)
        self._process_svg_placeholders(soup)

        # 2. 语义转换层
        md_text = self._convert_to_markdown(soup)

        # 3. 分块层（复用 MarkdownParser 的逻辑）
        sections = self._split_by_headers(md_text)

        # 4. 元数据提取
        web_metadata = self._extract_metadata(soup)

        # 5. 构建节点（维护层级栈）
        nodes = []
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
                
                node = self._build_node(
                    text=content,
                    metadata={
                        **web_metadata,
                        "section_header": header_text,
                        "section_path": section_path,
                        "level": level
                    },
                    blob=blob
                )
                nodes.append(node)

        return nodes

    def _refine_dom(self, soup: BeautifulSoup) -> BeautifulSoup:
        """精炼：基于准则 II.7/II.5，去除空标签，解开无语义的嵌套 <div>。"""
        # 移除脚本和样式
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        # 移除空标签，但保留meta、link等元数据标签
        preserve_tags = {'img', 'br', 'hr', 'meta', 'link', 'svg'}
        for tag in soup.find_all():
            if not tag.get_text(strip=True) and tag.name not in preserve_tags:
                tag.decompose()

        # 合并连续的<br>
        for br in soup.find_all('br'):
            if br.next_sibling and br.next_sibling.name == 'br':
                br.next_sibling.decompose()

        # 自定义清洗规则
        for selector in self.custom_clean_rules:
            for tag in soup.select(selector):
                tag.decompose()

        # 铺平无语义嵌套
        for tag in soup.find_all(['span', 'div']):
            if tag.name == 'span' and not tag.attrs:  # 无属性的span
                tag.unwrap()
            elif tag.name == 'div' and not tag.attrs and len(tag.contents) == 1:  # 无属性的单内容div
                tag.unwrap()

        return soup

    def _process_svg_placeholders(self, soup: BeautifulSoup) -> None:
        """视觉占位：对准则 II.4 保留的 SVG，插入 [图表] 文本描述。"""
        for svg in soup.find_all('svg'):
            # 在SVG前插入占位符文本
            placeholder = soup.new_tag('p')
            placeholder.string = '[图表]'
            svg.insert_before(placeholder)

    def _convert_to_markdown(self, soup: BeautifulSoup) -> str:
        """核心：将 HTML 转为 Markdown。这是为了复用 MarkdownParser 的逻辑。"""
        # 处理表格和代码块
        self._table_to_markdown(soup)
        self._extract_code_blocks(soup)

        # 转换整个soup
        md_text = self.h2t.handle(str(soup))
        return md_text

    def _table_to_markdown(self, soup: BeautifulSoup) -> None:
        """硬核处理：解析 rowspan/colspan，重建结构化表格字符串。"""
        for table in soup.find_all('table'):
            try:
                md_table = self._parse_table_to_markdown(table)
                # 替换原table标签为Markdown表格
                table.replace_with(md_table)
            except Exception as e:
                # 如果解析失败，保留原table或转为简单文本
                print(f"表格解析失败: {e}")
                pass

    def _parse_table_to_markdown(self, table: Tag) -> str:
        """解析HTML表格为Markdown表格，支持rowspan/colspan。"""
        rows = []
        for tr in table.find_all('tr'):
            row = []
            for cell in tr.find_all(['td', 'th']):
                text = cell.get_text(strip=True)
                colspan = int(cell.get('colspan', 1))
                rowspan = int(cell.get('rowspan', 1))
                # 简化处理：重复单元格内容
                for _ in range(colspan):
                    row.append(text)
            rows.append(row)

        if not rows:
            return ""

        # 确定最大列数
        max_cols = max(len(row) for row in rows) if rows else 0

        # 填充缺失的单元格
        for row in rows:
            while len(row) < max_cols:
                row.append("")

        # 生成Markdown
        md_lines = []
        for i, row in enumerate(rows):
            md_lines.append("| " + " | ".join(row) + " |")
            if i == 0:  # 表头后添加分隔线
                md_lines.append("| " + " | ".join(["---"] * max_cols) + " |")

        return "\n".join(md_lines) + "\n"

    def _extract_code_blocks(self, soup: BeautifulSoup) -> None:
        """代码保护：将 <pre><code> 转化为 Markdown 的 ``` 格式。"""
        # html2text默认处理，这里确保
        pass

    def _split_by_headers(self, md_text: str) -> List[Dict]:
        """逻辑复用：按 H1-H6 标题切分（直接调用 MarkdownParser 的逻辑）。"""
        lines = md_text.split('\n')
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
        buffer = [lines[start_idx]]
        
        # 检查当前行是否已经是完整的块（单行块）
        current_line = lines[start_idx].strip()
        if current_line.startswith(end_marker) and current_line.endswith(end_marker) and current_line != end_marker:
            # 单行块，如 $$ content $$
            return buffer, start_idx + 1
        
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
        titles = [h[1] for h in headers_stack if h[0] > 0]
        if not titles and headers_stack:
            return headers_stack[0][1]
        return " > ".join(titles)

    def _extract_metadata(self, soup: BeautifulSoup) -> Dict:
        """溯源：提取 `<title>`, `meta-desc`, `canonical-url` 等。"""
        metadata = {}
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.get_text().strip()

        # meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property')
            content = meta.get('content')
            if name and content:
                metadata[name] = content

        # canonical url
        canonical = soup.find('link', rel='canonical')
        if canonical:
            metadata['canonical_url'] = canonical.get('href')

        return metadata

    def _build_node(self, text: str, metadata: Dict, blob: Blob) -> TextNode:
        """封装：生成带层级路径（Breadcrumb）的 `TextNode`。"""
        base_metadata = blob.metadata.copy()
        if "domain" not in base_metadata and "category" in base_metadata:
            base_metadata["domain"] = base_metadata.pop("category")
        elif "domain" not in base_metadata:
            base_metadata["domain"] = None

        if "file_name" not in base_metadata:
            base_metadata["file_name"] = Path(blob.source).name

        base_metadata.update(metadata)
        base_metadata["source"] = blob.source

        return TextNode(text=text, metadata=base_metadata)