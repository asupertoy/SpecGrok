from typing import List, Union, Optional

# 标准库
import logging
import os
import re
from pathlib import Path

# 第三方库
import fitz  # PyMuPDF
from PIL import Image
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document, TextNode, NodeRelationship, RelatedNodeInfo

# 本地模块
from src.config import settings
from src.ingestion.loaders import Blob

class PDFParser(BaseReader):
    def __init__(self, ocr_enabled: bool = False):
        super().__init__()
        self.ocr_enabled = ocr_enabled
        # 延迟初始化 Pix2Text 相关字段
        self._p2t = None
        self._p2t_home: Optional[Path] = None
        self._p2t_init_attempted: bool = False

        if self.ocr_enabled:
            self._init_pix2text()

    def _init_pix2text(self) -> None:
        """延迟初始化 Pix2Text 及相关配置。

        计算模型目录并设置 `PIX2TEXT_HOME`，只有在启用 OCR 时才导入并实例化
        Pix2Text。若已尝试初始化则直接返回以避免重复失败重试。
        """
        if self._p2t_init_attempted:
            return
        self._p2t_init_attempted = True

        base_root = Path(__file__).resolve().parents[2]
        p2t_path = Path(settings.PIX2TEXT_HOME)
        if not p2t_path.is_absolute():
            p2t_path = base_root / p2t_path
        self._p2t_home = p2t_path

        os.environ.setdefault("PIX2TEXT_HOME", str(self._p2t_home))
        try:
            # 延迟导入，避免在不使用 OCR 时引入重依赖
            from pix2text import Pix2Text
        except Exception as e:
            logging.warning("Pix2Text 初始化失败：%s", e)
            # 保持 self._p2t 为 None，后续不会重复尝试
            return

        try:
            self._p2t = Pix2Text()
        except Exception as e:
            logging.exception("Pix2Text 实例化失败：%s", e)
            self._p2t = None

    def _get_pix2text(self):
        """按需返回 Pix2Text 实例；若尚未初始化则尝试初始化。

        若初始化曾经失败（或模块不存在），将返回 None。
        """
        if self._p2t is None and self.ocr_enabled and not self._p2t_init_attempted:
            self._init_pix2text()
        return self._p2t

    def parse(self, blob: Blob) -> List[TextNode]:
        """从 Blob 对象解析 PDF 数据并返回 TextNode 列表。

        Args:
            blob: 包含 PDF 数据和元数据的 Blob 对象。
        Returns:
            包含 PDF 内容的 TextNode 列表。
        """
        print(f"PDFParser正在深度解析 Blob: {blob.source} ...")

        if blob.data:
            doc = fitz.open(stream=blob.data, filetype="pdf")
        else:
            doc = fitz.open(blob.source)
        
        nodes = []
        current_section = "Unknown"
        # buffer 存储元素结构: {"text": str, "page": int, "is_formula": bool}
        buffer = []

        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            for i, block in enumerate(blocks):
                # 1. 识别图片
                if self._is_image_block(block):
                    # 策略：对于图片，采取“记录位置、保留元数据”的策略
                    # 将图片保存并记录引用，而不是尝试 OCR 文本提取
                    # 之后可以考虑加入 VLM 进行语义理解
                    image_path = self._save_image(page, block, blob, page_num, i)
                    if image_path:
                        # 插入图片引用（Markdown 格式），作为未来多模态处理的锚点
                        img_ref = f"\n![Figure]({image_path})\n"
                        buffer.append({"text": img_ref, "page": page_num, "is_formula": False})
                    continue

                # 2. 识别表格（预留接口）
                if self._is_table_block(block):
                    # TODO: 调用专门的表格识别逻辑
                    pass

                # 3. 提取文本
                text = ""
                if block.get("type", 0) == 0:
                    text = "".join(span["text"] for line in block.get("lines", []) for span in line.get("spans", [])).strip()

                # 4. 检测是否需要 OCR (乱码或公式)
                is_formula = False
                if not text or self._detect_formula(block):
                    ocr_text = self._ocr_block_with_pix2text(page, block)
                    if ocr_text:
                        text = ocr_text
                        # 简单的公式判定逻辑覆盖
                        if "$$" in text or "\\" in text:
                            is_formula = True

                # 5. 清洗文本
                text = self._clean_text(text)
                if not text:
                    continue

                # 6.判断是否为章节标题
                if self._is_section_title(text):
                    # flush 上一个section
                    if buffer:
                        nodes.append(self._build_node(buffer, current_section, blob))
                        buffer = []

                    current_section = text
                else:
                    buffer.append({"text": text, "page": page_num, "is_formula": is_formula})

        if buffer:
            nodes.append(self._build_node(buffer, current_section, blob))

        return nodes

    def load_data(self, file: str) -> List[TextNode]:
        """兼容旧接口"""
        blob = Blob.from_path(file)
        return self.parse(blob)

    def _save_image(self, page, block, blob, page_num, block_index) -> Optional[str]:
        """保存图片块到本地文件，并返回路径。
        
        保存路径：在 PDF 文件同级目录下创建 pdf_images/{pdf_current_path_hash_or_name}/ 文件夹
        """
        bbox = block.get("bbox")
        if not bbox:
            return None
            
        try:
            # 确定基础路径
            source_path = Path(blob.source)
            # 创建专门存放该 PDF 图片的子目录
            images_base_dir = source_path.parent / "pdf_images"
            pdf_images_dir = images_base_dir / source_path.stem
            pdf_images_dir.mkdir(parents=True, exist_ok=True)
            
            image_filename = f"p{page_num+1}_img{block_index}.png"
            image_path = pdf_images_dir / image_filename
            
            # 如果图片已存在，可以选择跳过或覆盖
            # 这里简单起见直接保存/覆盖
            rect = fitz.Rect(*bbox)
            # 使用较高的 DPI 以适应后续可能的 VLM 分析
            pix = page.get_pixmap(clip=rect, dpi=300) 
            pix.save(str(image_path))
            
            return str(image_path)
            
        except Exception as e:
            logging.warning(f"保存图片失败: {e}")
            return None

    def _is_image_block(self, block: dict) -> bool:
        """判定逻辑：识别图片对象（type: 1），标记其在页面的位置。"""
        return block.get("type", 0) == 1

    def _is_table_block(self, block: dict) -> bool:
        """判定逻辑：根据 PyMuPDF 的 block["type"] 或布局密度判断是否为表格区。
        
        目前 PyMuPDF 的默认 block 分割对表格支持有限，这里作为预留接口。
        """
        # 可以通过分析 lines 的对齐方式或者 spans 的密度来推断，这里暂时返回 False
        return False

    def _detect_formula(self, block: dict) -> bool:
        """判定逻辑：检测乱码、上下标特征，决定是否触发 OCR。
        
        合并了原 _looks_garbled 的逻辑。
        """
        # 如果 block 类型不对，直接认为需要处理
        if block.get("type", 0) != 0:
            return True
            
        # 提取原始文本进行分析
        text = "".join(span["text"] for line in block.get("lines", []) for span in line.get("spans", [])).strip()
        
        # --- 乱码检测逻辑 ---
        if not text:
            return True
        printable = [ch for ch in text if ch.isprintable() or ch.isspace()]
        # 如果几乎没有可打印字符
        if not printable:
            return True
        # 可打印字符占比过低
        if len(printable) / len(text) < 0.6:
            return True
        # 过多替换字符
        if text.count("") / len(text) > 0.1:
            return True
        
        # --- 公式特征检测 ---
        # 2. 检测潜在的数学公式特征 (LaTeX 常见符号或上下标指示)
        math_indicators = ["∫", "∑", "∂", "√", "∈", "∀", "∃", "\\"]
        if any(char in text for char in math_indicators):
            return True
            
        return False

    def _clean_text(self, text: str) -> str:
        """构造/清洗：处理换行符、去除页码噪音、合并断裂单词。"""
        if not text:
            return ""
        
        # 1. 合并连字符断行 (例如 "commu-\nnication" -> "communication")
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        
        # 2. 替换多余的空白字符
        text = re.sub(r"\s+", " ", text)
        
        return text.strip()


    def _is_section_title(self, text: str) -> bool:
        """
        参考 deepdoc/parser/pdf_parser.py 的 proj_match 方法，
        改进标题识别，支持更多格式的章节标题。
        """
        if len(text) <= 2:
            return False
        # 排除纯数字或符号的行
        if re.match(r"[0-9 ().,%%+/-]+$", text):
            return False
        
        # 匹配各种标题模式（优先级从高到低）
        title_patterns = [
            # 中文章节标题
            r"第[零一二三四五六七八九十百千]+章",
            r"第[零一二三四五六七八九十百千]+[条节]",
            r"[零一二三四五六七八九十百千]+[、 　]",
            r"[\(（][零一二三四五六七八九十百千]+[）\)]",
            # 英文章节标题
            r"^(Chapter|Section|§)\s*\d+(\.\d+)*",
            # 数字编号
            r"[0-9]+(、|\.[　 ]|\.[^0-9])",
            r"[0-9]+\.[0-9]+(、|[. 　]|[^0-9])",
            r"[0-9]+\.[0-9]+\.[0-9]+(、|[ 　]|[^0-9])",
            r"[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+(、|[ 　]|[^0-9])",
            # 冒号结尾的短句
            r".{,48}[：:?？]$",
            # 括号编号
            r"[0-9]+）",
            r"[\(（][0-9]+[）\)]",
            # 其他常见标题特征
            r"[零一二三四五六七八九十百千]+是",
            r"[⚫•➢✓]",
        ]
        
        for pattern in title_patterns:
            if re.match(pattern, text):
                return True
        
        return False


    def _normalize_pix2text_result(self, result: Optional[Union[str, dict, list]]) -> str:
        """将 Pix2Text 结果统一为纯文本/Markdown 字符串。"""
        if result is None:
            return ""
        
        # 1. 结果本身就是字符串（Pix2Text v1.0+ 混合识别通常返回 Markdown）
        if isinstance(result, str):
            return result.strip()
            
        # 2. 结果是字典（单项识别）
        if isinstance(result, dict):
            # 优先使用 text，如果它包含 markdown 结构则更好
            # 如果存在 latex 字段，通常意味着这是一个独立公式，将其包裹为 Markdown 块级公式
            if "latex" in result and result["latex"].strip():
                return f"$$ {result['latex'].strip()} $$"
            return (result.get("text") or "").strip()
            
        # 3. 结果是列表（多项识别结果拼接）
        if isinstance(result, list):
            parts = []
            for item in result:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    if "latex" in item and item["latex"].strip():
                        parts.append(f"$$ {item['latex'].strip()} $$")
                    else:
                        parts.append(item.get("text") or "")
            return "\n".join(p.strip() for p in parts if p.strip())
            
        return ""


    def _ocr_block_with_pix2text(self, page, block) -> str:
        """对指定 block 执行 Pix2Text OCR，优先输出可读文本/LaTeX。"""
        p2t = self._get_pix2text()
        if p2t is None:
            return ""

        bbox = block.get("bbox")
        if not bbox:
            return ""

        rect = fitz.Rect(*bbox)
        try:
            pix = page.get_pixmap(clip=rect, dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        except Exception:
            return ""

        result = None
        # 尝试文本/公式联合识别，回退到通用 recognize
        if hasattr(p2t, "recognize_text_formula"):
            try:
                result = p2t.recognize_text_formula(img)
            except Exception:
                result = None
        if result is None and hasattr(p2t, "recognize"):
            try:
                result = p2t.recognize(img)
            except Exception:
                result = None

        return self._normalize_pix2text_result(result)


    def _build_node(self, buffer: List[dict], section: str, blob: Blob) -> TextNode:
        texts = [item["text"] for item in buffer]
        content = "\n".join(texts)

        # 提取元数据
        pages = sorted(list(set(item["page"] for item in buffer)))
        start_page = pages[0] if pages else 0

        latex_formulas = [item["text"] for item in buffer if item.get("is_formula")]

        # 构造基础 metadata
        metadata = blob.metadata.copy()  # 复制一份，避免修改原对象
        if "domain" not in metadata:
            metadata["domain"] = None

        # 覆盖 parser 特有的元数据
        metadata.update({
            "source": blob.source,
            "file_name": metadata.get("file_name", Path(blob.source).name),
            "section": section,
            "page": start_page,
            "pages": pages,
            "latex_formula": latex_formulas if latex_formulas else None,
        })

        node = TextNode(text=content, metadata=metadata)
        
        # [Fix] 设置 Source 关系指向原始文件
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
            node_id=blob.source,
            metadata={"file_name": metadata.get("file_name")}
        )

        return node