import logging
import re
import hashlib
import os
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import fitz  # PyMuPDF
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo, ImageNode


from src.config import settings
from src.ingestion.loaders import Blob
from src.models.llm import get_vlm, get_llm

logger = logging.getLogger(__name__)

class PDFParser(BaseReader):
    def __init__(
        self,
        vlm_enabled: bool = True,
        vlm_max_workers: int = 4,
        vlm_retry_attempts: int = 3,
        vlm_retry_base_delay: float = 0.5,
        max_pages_per_node: int = 2,
        pix2tex_enabled: bool = True,
        pix2tex_min_score: float = 0.85,
        vlm_markdown_polish: bool = True,
    ):
        """
        初始化 PDFParser 类，用于解析 PDF 文档并提取文本、表格、图片和公式。
        """
        super().__init__()
        self.vlm_enabled = vlm_enabled
        self.cache_dir = Path(settings.CACHE_DIR) if hasattr(settings, 'CACHE_DIR') else Path("data/cache_vlm")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.vlm_max_workers = vlm_max_workers
        self.vlm_retry_attempts = vlm_retry_attempts
        self.vlm_retry_base_delay = vlm_retry_base_delay
        self.max_pages_per_node = max(1, int(max_pages_per_node))
        self.vlm_markdown_polish = bool(vlm_markdown_polish)

        # Pix2Text / pix2tex（本地公式 OCR）
        self.pix2tex_enabled = bool(pix2tex_enabled)
        self.pix2tex_min_score = float(pix2tex_min_score)
        self._latex_ocr = None
        
        if self.pix2tex_enabled:
            self._init_pix2tex_local()

        self._current_doc_domain: Optional[str] = None
        self._current_doc_source: Optional[str] = None

    def load_data(self, file: str, extra_info: Optional[Dict] = None) -> List[TextNode]:
        """
        兼容 BaseReader 接口。
        注意：在生产管道中，通常直接调用 parse(blob)。
        """
        blob = Blob.from_path(file)
        if extra_info:
            blob.metadata.update(extra_info)
        return self.parse(blob)

    def parse(self, blob: Blob) -> List[TextNode]:
        """
        核心解析逻辑。
        Args:
            blob: 包含 PDF 二进制数据和元数据的 Blob 对象。
        Returns:
            List[TextNode]: 解析后的节点列表。
        """
        logger.info(f"正在解析 Blob: {blob.source} (VLM={self.vlm_enabled})...")

        # 1. 打开 PDF 文档 (支持内存流)
        if blob.data:
            doc = fitz.open(stream=blob.data, filetype="pdf")
        else:
            doc = fitz.open(blob.source)

        # 2. 上下文推断
        self._current_doc_domain = self._infer_document_domain(blob)
        self._current_doc_source = blob.source

        nodes: List[TextNode] = []
        current_section = "Introduction"  # 默认起始章节

        # 缓冲区：用于积累未达到切分条件的文本块、图片占位符等
        buffer: List[dict] = []
        block_id_counter = 0

        # 使用线程池并发处理 VLM/OCR 请求
        with ThreadPoolExecutor(max_workers=self.vlm_max_workers) as executor:
            for page_num, page in enumerate(doc):
                # ---------------------------------------------------
                # 切分检查：按页数阈值强制切分，避免 Node 过大
                # ---------------------------------------------------
                if buffer:
                    buffer_first_page = min(item.get("page", page_num) for item in buffer)
                    if page_num - buffer_first_page >= self.max_pages_per_node:
                        self._resolve_pending_texts(buffer)
                        nodes.extend(self._build_node(buffer, current_section, blob))
                        buffer = []

                # ---------------------------------------------------
                # 页面预处理
                # ---------------------------------------------------
                # 寻找表格区域
                tables = page.find_tables()
                table_bboxes = [fitz.Rect(t.bbox) for t in tables]

                # 获取所有 Blocks
                blocks = page.get_text("dict")["blocks"]
                page_height = page.rect.height
                body_font_size = self._estimate_body_font_size(blocks)

                # 公式 Block 预合并 (处理跨 Block 的大型公式)
                merged_formula = self._merge_formula_blocks(blocks, page_height)
                merged_index_map = merged_formula["index_map"]

                # 构建处理队列 (表格 + 文本块)，按垂直位置排序
                page_items = [{"kind": "table", "bbox": t_rect} for t_rect in table_bboxes]
                for i, block in enumerate(blocks):
                    page_items.append({
                        "kind": "block",
                        "index": i,
                        "block": block,
                        "bbox": fitz.Rect(block.get("bbox")),
                    })
                page_items.sort(key=lambda x: (x["bbox"].y0, x["bbox"].x0))

                # ---------------------------------------------------
                # 遍历页面元素
                # ---------------------------------------------------
                for item in page_items:
                    bbox = item["bbox"]

                    # === Case A: 表格区域 (VLM 解析) ===
                    if item["kind"] == "table":
                        img_path = self._save_smart_crop(page, bbox, blob.source, page_num, 0, "table")
                        if not img_path:
                            continue
                        cached_text, future = self._schedule_vlm_task(img_path, "table", executor)
                        block_id_counter += 1
                        placeholder = f"[IMAGE_BLOCK_{block_id_counter}]"
                        
                        # 如果有缓存直接用，否则在 buffer 里存 future
                        table_text_clean = self._clean_vlm_response((cached_text or "").strip()) if cached_text else ""
                        table_text_value = placeholder if not table_text_clean else f"{placeholder}\n{table_text_clean}"
                        
                        buffer.append({
                            "kind": "table",
                            "text": table_text_value,
                            "content": table_text_clean if cached_text else "", # 暂时为空，稍后 resolve
                            "block_id": block_id_counter,
                            "future": future,
                            "page": page_num,
                            "is_formula": False,
                            "img_path": img_path,
                            "raw_text": "",
                        })
                        continue

                    # === Case B/C: 文本/图片 Block ===
                    block = item["block"]
                    i = item["index"]

                    # 如果 Block 被包含在表格内，跳过（避免重复）
                    if self._is_table_block(block, table_bboxes):
                        continue

                    # --- Case B: 图片 (Type=1) ---
                    if self._is_image_block(block):
                        img_path = self._save_smart_crop(page, bbox, blob.source, page_num, i, "fig")
                        if not img_path:
                            continue
                        cached_text, future = self._schedule_vlm_task(img_path, "description", executor)
                        
                        desc = self._clean_vlm_response((cached_text or "").strip()) if cached_text else ""
                        block_id_counter += 1
                        placeholder = f"[IMAGE_BLOCK_{block_id_counter}]"
                        
                        buffer.append({
                            "kind": "image",
                            "text": placeholder if not desc else f"{placeholder}\n{desc}",
                            "content": desc,
                            "block_id": block_id_counter,
                            "future": future,
                            "page": page_num,
                            "is_formula": False,
                            "img_path": img_path,
                            "description": desc,
                            "raw_text": "",
                        })
                        continue

                    # --- Case C: 文本 (Type=0) ---
                    raw_text, span_stats = self._extract_text_with_inline_math(block, page_width=page.rect.width)
                    max_font_size = self._get_block_max_font_size(block)

                    # 检查是否是被合并的公式块的非首部
                    merged_info = merged_index_map.get(i)
                    if merged_info and not merged_info["is_first"]:
                        continue # 跳过，等待首部处理
                    
                    processing_bbox = merged_info["bbox"] if merged_info else bbox

                    # --- Case C-1: 公式块 (OCR) ---
                    if (self.pix2tex_enabled or self.vlm_enabled) and self._should_ocr_formula_block(raw_text, span_stats):
                        img_path = self._save_smart_crop(page, processing_bbox, blob.source, page_num, i, "formula")
                        if not img_path:
                            continue
                        cached_text, future = self._schedule_vlm_task(img_path, "transcription", executor)
                        block_id_counter += 1
                        placeholder = f"[IMAGE_BLOCK_{block_id_counter}]"
                        
                        formula_text = cached_text if cached_text else raw_text
                        formula_text = self._postprocess_formula_text(formula_text)
                        
                        buffer.append({
                            "kind": "formula",
                            "text": placeholder if not formula_text else f"{placeholder}\n{formula_text}",
                            "content": formula_text,
                            "block_id": block_id_counter,
                            "future": future,
                            "page": page_num,
                            "is_formula": True,
                            "img_path": img_path,
                            "raw_text": raw_text,
                        })
                        continue

                    # --- Case C-2: 普通文本 / 章节标题 ---
                    # 优先保留代码块的原生格式，否则进行清洗
                    block_text = raw_text if span_stats.get("has_code") else self._clean_text(raw_text)
                    if not block_text:
                        continue

                    # 章节标题检测 -> 触发切分
                    if self._is_section_title(block_text, bbox, max_font_size, body_font_size, page_height):
                        if buffer:
                            self._resolve_pending_texts(buffer)
                            nodes.extend(self._build_node(buffer, current_section, blob))
                            buffer = []
                        current_section = block_text
                    else:
                        buffer.append({
                            "kind": "text",
                            "text": block_text,
                            "future": None,
                            "page": page_num,
                            "is_formula": False,
                            "img_path": None,
                            "raw_text": "",
                        })

        # 处理最后一个 buffer
        if buffer:
            self._resolve_pending_texts(buffer)
            nodes.extend(self._build_node(buffer, current_section, blob))

        # 4. 全文语义对齐（可选）：校对拼接后的正文，修复 OCR 空格乱码
        if self.vlm_enabled and self.vlm_markdown_polish:
            nodes = self._maybe_polish_nodes_with_vlm(nodes)

        logger.info(f"解析完成: {blob.source}, 生成 {len(nodes)} 个节点。")
        return nodes

    # =========================================================================
    #  Helper Methods
    # =========================================================================

    def _infer_document_domain(self, blob: Blob) -> Optional[str]:
        """从 metadata 或文件路径推断文档领域。"""
        if blob.metadata:
            domain = blob.metadata.get("domain")
            if domain:
                return domain

        source = blob.source or ""
        path_tokens = [p.lower() for p in Path(source).parts]
        protocol_tokens = {"http", "protocol", "rfc", "tcp", "udp", "ip"}
        dsp_tokens = {"dsp"}

        if any(tok in protocol_tokens for tok in path_tokens):
            return "HTTP"
        if any(tok in dsp_tokens for tok in path_tokens):
            return "DSP"
        return None

    def _is_protocol_document(self) -> bool:
        domain = (self._current_doc_domain or "").lower()
        if any(key in domain for key in ("http", "protocol", "rfc")):
            return True
        source = (self._current_doc_source or "").lower()
        protocol_keywords = ("http", "https", "tcp", "udp", "rfc", "protocol")
        return any(k in source for k in protocol_keywords)

    def _schedule_vlm_task(
        self, image_path: str, mode: str, executor: ThreadPoolExecutor
    ) -> Tuple[Optional[str], Optional[Future]]:
        """准备 VLM 任务（带缓存）。返回 (cached_text, future)。"""
        if mode == "transcription":
            return self._schedule_formula_task(image_path=image_path, executor=executor)

        img_hash = self._get_image_hash(image_path)
        prompt = self._get_prompt_by_mode(mode)
        prompt_hash = hashlib.md5(prompt.encode("utf-8")).hexdigest()[:8]
        cache_file = self.cache_dir / f"{img_hash}_{mode}_{prompt_hash}.txt"

        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                return f.read(), None

        if not self.vlm_enabled:
            return "", None

        future = executor.submit(self._vlm_task_worker, image_path, cache_file, prompt)
        return None, future

    def _schedule_formula_task(
        self, image_path: str, executor: ThreadPoolExecutor
    ) -> Tuple[Optional[str], Optional[Future]]:
        img_hash = self._get_image_hash(image_path)
        cache_file = self.cache_dir / f"{img_hash}_formula_pix2tex_v1.txt"
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                return f.read(), None

        future = executor.submit(self._formula_task_worker, image_path, cache_file)
        return None, future

    def _formula_task_worker(self, image_path: str, cache_file: Path) -> str:
        try:
            latex_code, score = self._latex_model(image_path)
            latex_code = self._fix_dsp_rotating_factor(latex_code)

            needs_refine = self._is_low_confidence_formula(latex_code, score)
            if needs_refine and self.vlm_enabled:
                prompt = self._get_formula_refine_prompt(latex_code)
                refined = self._invoke_vlm_with_retry(image_path, prompt)
                cleaned = self._clean_vlm_response(refined)
                cleaned = self._postprocess_formula_text(cleaned)
            else:
                cleaned = self._postprocess_formula_text(latex_code)

            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(cleaned)
            return cleaned
        except Exception as e:
            logger.error(f"Formula OCR failed for {image_path}: {e}")
            return ""

    def _vlm_task_worker(self, image_path: str, cache_file: Path, prompt: str) -> str:
        try:
            result_text = self._invoke_vlm_with_retry(image_path, prompt)
            cleaned_text = self._clean_vlm_response(result_text)
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(cleaned_text)
            return cleaned_text
        except Exception as e:
            logger.error(f"VLM process failed for {image_path}: {e}")
            return ""

    def _get_prompt_by_mode(self, mode: str) -> str:
        is_protocol = self._is_protocol_document()
        if mode == "table":
            return (
                "Identify the table in this image and convert it into a standard Markdown table.\n"
                "1. Preserve the header structure exactly.\n"
                "2. Ensure all rows and columns are aligned.\n"
                "3. If cells contain formulas, use LaTeX format (e.g., $$x^2$$).\n"
                "4. Output ONLY the markdown table, no other text."
            )
        elif mode == "transcription":
            return (
                r"Transcribe the text in this image strictly.\n"
                r"Return formulas in valid LaTeX format only.\n"
                r"Use \\sum for summation, \\frac{a}{b} for fractions, and ^ / _ for super/subscripts.\n"
                r"Example: X(e^{j\\omega}) = \\sum_{n=-\\infty}^{\\infty} x[n]e^{-j\\omega n}\n"
                r"DO NOT use plain text characters to mimic formulas.\n"
                r"Wrap every mathematical expression in $$...$$ (block math).\n"
                r"Output ONLY the transcribed content, no introductions or explanations."
            )
        else:  # description
            if is_protocol:
                return (
                    "请描述此 HTTP/协议报文或流程图。\n"
                    "- 按出现顺序列出所有 Header 字段或关键帧，并解释其位宽/作用。\n"
                    "- 若是交互流程，按步骤说明节点与消息方向。\n"
                    "- 仅输出结构化描述，不要编造不存在的字段。"
                )
            return (
                "Analyze this image for a technical document.\n"
                "1. Briefly describe the visual content (chart trends, diagram structure).\n"
                "2. Do not transcribe text if it's too dense, just summarize the meaning.\n"
                "3. Output ONLY the description."
            )

    def _save_smart_crop(self, page, bbox, source_name, page_num, index, prefix) -> Optional[str]:
        try:
            rect = fitz.Rect(bbox)
            if rect.is_empty or rect.width <= 0 or rect.height <= 0:
                return None
            
            width_pt = rect.width
            height_pt = rect.height
            
            zoom = 2.0
            if width_pt < 100 or height_pt < 50:
                zoom = 4.0
            elif width_pt > 500 or height_pt > 700:
                zoom = 1.5
            
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(clip=rect, matrix=mat)
            
            safe_name = Path(source_name).stem
            save_dir = self.cache_dir / safe_name
            save_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"p{page_num+1}_{prefix}_{index}.png"
            filepath = save_dir / filename
            
            pix.save(str(filepath))
            return str(filepath)
        except Exception as e:
            logger.warning(f"Screenshot failed: {e}")
            return None

    def _clean_vlm_response(self, text: str) -> str:
        if not text: return ""
        text = re.sub(r"^(Sure|Certainly|Here|Okay).*?[:\n]", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
        lines = [re.sub(r"^\s*>+\s?", "", line) for line in text.splitlines()]
        text = "\n".join(lines)
        text = self._normalize_math_delimiters(text)
        text = self._ensure_latex_wrapped(text)
        return text.strip()

    def _get_image_hash(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def _invoke_vlm(self, image_path: str, prompt: str) -> str:
        """
        Invoke the configured VLM client to analyze an image.
        Supports both the custom DashScope-based `VLMClient` (with `analyze_image`) and
        LlamaIndex/OpenAI-style clients (with `complete` accepting `image_documents`).
        """
        vlm = get_vlm()
        # Preferred path: custom VLM client with analyze_image(image_path, prompt)
        if hasattr(vlm, "analyze_image"):
            try:
                return vlm.analyze_image(image_path, prompt)
            except Exception as e:
                logger.warning(f"VLM analyze_image failed for {image_path}: {e}")
                raise

        # Fallback: LlamaIndex/OpenAI-style client which may accept ImageNode
        if hasattr(vlm, "complete"):
            try:
                image_node = ImageNode(image_path=image_path)
                response = vlm.complete(prompt=prompt, image_documents=[image_node])
                return getattr(response, "text", "") or ""
            except TypeError as e:
                logger.warning(f"VLM client 'complete' signature mismatch: {e}")
                raise
            except Exception as e:
                logger.warning(f"VLM client 'complete' failed for {image_path}: {e}")
                raise

        raise RuntimeError("No supported VLM interface available for image analysis")

    def _init_pix2tex_local(self) -> None:
        try:
            from pix2text.latex_ocr import LatexOCR
        except Exception as e:
            logger.warning(f"pix2text not available: {e}")
            self._latex_ocr = None
            return

        pix2text_root = Path(getattr(settings, "PIX2TEXT_HOME", "models/pix2text"))
        if (pix2text_root / "1.1").exists():
            pix2text_root = pix2text_root / "1.1"

        model_dir = pix2text_root / "mfr-1.5-onnx"
        if not model_dir.exists() and not list(pix2text_root.glob("*.onnx")):
             logger.warning(f"Pix2Text onnx weights not found under {model_dir}")
             # 尝试用 root 初始化
        
        # 简化初始化，假设库能自动处理或用户已正确配置路径
        try:
            self._latex_ocr = LatexOCR(model_backend="onnx", model_dir=model_dir if model_dir.exists() else None)
            logger.info(f"Pix2Text LatexOCR initialized.")
        except Exception as e:
            logger.warning(f"Pix2Text init failed: {e}")
            self._latex_ocr = None

    def _latex_model(self, image_path: str) -> Tuple[str, Optional[float]]:
        if not self.pix2tex_enabled or self._latex_ocr is None:
            return "", None
        try:
            result = self._latex_ocr.recognize(image_path)
            if isinstance(result, list):
                result = result[0] if result else {}
            if isinstance(result, dict):
                text = (result.get("text") or "").strip()
                score = result.get("score")
                try:
                    score = float(score) if score is not None else None
                except Exception:
                    score = None
                return text, score
            if isinstance(result, str):
                return result.strip(), None
        except Exception as e:
            logger.debug(f"LatexOCR processing error: {e}")
        return "", None

    def _is_low_confidence_formula(self, latex_code: str, score: Optional[float]) -> bool:
        if not latex_code or len(latex_code.strip()) < 3:
            return True
        if score is None:
            return False
        return score < self.pix2tex_min_score

    def _get_formula_refine_prompt(self, latex_code: str) -> str:
        latex_code = latex_code.strip() if latex_code else ""
        return (
            "You are correcting OCR for a mathematical formula from an image.\n"
            f"This is the preliminary OCR result: {latex_code!r}\n"
            "Please use the image context to correct any mistakes.\n"
            "Return formulas in valid LaTeX format only.\n"
            "Output ONLY the corrected LaTeX, no explanation."
        )

    def _fix_dsp_rotating_factor(self, latex_code: str) -> str:
        if not latex_code: return ""
        pattern = re.compile(r"W\s*(?:_\s*\{?\s*N\s*\}?|_\s*N)?\s*\^\s*\{?\s*n\s*k\s*\}?", flags=re.IGNORECASE)
        if pattern.search(latex_code):
            latex_code = pattern.sub(r"W_{N}^{nk}", latex_code)
        latex_code = re.sub(r"W_\{N\}\^\{n\s*k\}", r"W_{N}^{nk}", latex_code)
        return latex_code

    def _invoke_vlm_with_retry(self, image_path: str, prompt: str) -> str:
        last_error = None
        for attempt in range(self.vlm_retry_attempts):
            try:
                return self._invoke_vlm(image_path, prompt)
            except Exception as e:
                last_error = e
                time.sleep(self.vlm_retry_base_delay * (2 ** attempt))
        if last_error:
            raise last_error
        return ""

    def _normalize_math_delimiters(self, text: str) -> str:
        if not text: return ""
        text = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", text, flags=re.DOTALL)
        text = re.sub(r"\\\((.*?)\\\)", r"$\1$", text, flags=re.DOTALL)
        text = re.sub(r"(?<!\$)\$([^\n\$]+)\$(?!\$)", r"$$\1$$", text)
        return text

    def _ensure_latex_wrapped(self, text: str) -> str:
        if not text: return ""
        stripped = text.strip()
        if "|" in stripped and "\n" in stripped: return stripped
        if "$$" in stripped: return stripped
        looks_like_latex = ("\\" in stripped or any(tok in stripped for tok in ("_", "^", "\\sum", "\\frac", "\\int")))
        if not looks_like_latex: return stripped
        return f"$$\n{stripped}\n$$"

    def _postprocess_formula_text(self, text: str) -> str:
        if not text: return ""
        cleaned = self._fix_spaced_letters(text.strip())
        cleaned = self._normalize_math_delimiters(cleaned)
        if "$$" in cleaned: return cleaned
        return f"$$\n{cleaned}\n$$"

    def _detect_formula(self, text: str) -> bool:
        if not text: return True
        printable = [ch for ch in text if ch.isprintable() or ch.isspace()]
        if len(printable) / len(text) < 0.8: return True
        if self._is_garbled_text(text): return True
        dsp_patterns = [r"W_N", r"x\[n\]", r"X\(z\)", r"H\(e\^", r"e\^\{?[j-]\w", r"\\[a-zA-Z]+", r"[_\^]\{?[\w\+\-]+\}?"]
        for pattern in dsp_patterns:
            if re.search(pattern, text): return True
        math_indicators = ["∫", "∑", "∏", "∂", "√", "∈", "∀", "∃", "≠", "≈", "≤", "≥"]
        if any(char in text for char in math_indicators): return True
        return False

    def _is_garbled_text(self, text: str) -> bool:
        if not text: return False
        total = len(text)
        printable = sum(1 for ch in text if ch.isprintable() or ch.isspace())
        if total == 0: return False
        if printable / total < 0.75: return True
        return text.count("�") / total > 0.02

    def _is_table_block(self, block: dict, table_bboxes: List[fitz.Rect]) -> bool:
        if not table_bboxes or "bbox" not in block: return False
        b_rect = fitz.Rect(block["bbox"])
        for t_rect in table_bboxes:
            intersect = t_rect & b_rect
            if intersect.get_area() / b_rect.get_area() > 0.6: return True
        return False

    def _is_image_block(self, block: dict) -> bool:
        return block.get("type", 0) == 1

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        text = self._fix_spaced_letters(text)
        text = self._merge_inline_math_fragments(text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _fix_spaced_letters(self, text: str) -> str:
        if not text: return ""
        def _merge(match: re.Match) -> str: return match.group(0).replace(" ", "")
        return re.sub(r"\b(?:[A-Za-z]\s){2,}[A-Za-z]\b", _merge, text)

    def _merge_inline_math_fragments(self, text: str) -> str:
        """合并被拆分的行内数学片段，如 $x$[$n$] -> $x[n]$。"""
        if not text:
            return ""
        text = re.sub(r"\$([A-Za-z])\$\s*\[\s*\$([A-Za-z0-9]+)\$\s*\]", r"$\1[\2]$", text)
        text = re.sub(r"\$W\$\s*\$N\$", r"$W_N$", text)
        return text

    def _extract_text_with_inline_math(self, block: dict, page_width: Optional[float] = None) -> Tuple[str, Dict[str, Any]]:
        wrapped_spans = 0
        total_spans = 0
        math_font_spans = 0
        symbol_font_spans = 0
        italic_spans = 0
        math_line_count = 0
        has_code = False
        line_entries: List[Tuple[str, bool]] = []

        for line in block.get("lines", []):
            span_parts: List[str] = []
            code_span_count = 0
            line_bbox = line.get("bbox") if isinstance(line, dict) else None

            for span in line.get("spans", []):
                total_spans += 1
                span_text = span.get("text") or ""
                font = span.get("font") or ""
                flags = int(span.get("flags") or 0)

                if self._is_code_span(span_text, font):
                    code_span_count += 1
                if self._is_math_font(font):
                    math_font_spans += 1
                if self._is_symbol_font(font):
                    symbol_font_spans += 1
                if self._is_italic_flags(flags):
                    italic_spans += 1
                if span_text and self._is_inline_math_span(span_text, font, flags):
                    span_text = self._wrap_inline_math(span_text)
                    wrapped_spans += 1
                span_parts.append(span_text)

            line_text = "".join(span_parts)
            if not line_text.strip(): continue

            code_like = self._looks_like_code_line(line_text)
            code_span_ratio = (code_span_count / max(1, len(line.get("spans", []))))
            line_is_code = (code_span_ratio >= 0.6 and code_like)

            if not line_is_code and line_bbox and page_width and code_like and code_span_count > 0:
                try:
                    x0 = float(line_bbox[0])
                    if x0 >= page_width * 0.12: line_is_code = True
                except Exception: pass

            if self._looks_like_math_line(line_text):
                math_line_count += 1
            line_entries.append((line_text.rstrip("\n") if line_is_code else line_text.strip(), line_is_code))
            if line_is_code: has_code = True

        if not line_entries:
            return "", {
                "wrapped_spans": wrapped_spans,
                "total_spans": total_spans,
                "has_code": has_code,
                "math_span_ratio": 0.0,
                "symbol_span_ratio": 0.0,
                "italic_span_ratio": 0.0,
                "math_line_ratio": 0.0,
            }

        output_lines: List[str] = []
        in_code = False
        for line_text, is_code in line_entries:
            if is_code and not in_code:
                output_lines.append("```")
                in_code = True
            if not is_code and in_code:
                output_lines.append("```")
                in_code = False
            output_lines.append(line_text)
        if in_code: output_lines.append("```")

        text = "\n".join(output_lines).strip()
        span_denom = max(1, total_spans)
        line_denom = max(1, len(line_entries))
        return text, {
            "wrapped_spans": wrapped_spans,
            "total_spans": total_spans,
            "has_code": has_code,
            "math_span_ratio": math_font_spans / span_denom,
            "symbol_span_ratio": symbol_font_spans / span_denom,
            "italic_span_ratio": italic_spans / span_denom,
            "math_line_ratio": math_line_count / line_denom,
        }

    def _is_code_span(self, span_text: str, font: str) -> bool:
        if not span_text: return False
        return bool(re.search(r"(mono|courier|consolas|menlo|monaco)", font or "", flags=re.IGNORECASE))

    def _is_math_font(self, font: str) -> bool:
        return bool(
            re.search(
                r"(math|symbol|cmmi|cmsy|cmex|cmr|cmbx|stix|tex|mathit|mt|msam|msbm)",
                font or "",
                flags=re.IGNORECASE,
            )
        )

    def _is_symbol_font(self, font: str) -> bool:
        return bool(re.search(r"(symbol|wingdings|zapfdingbats)", font or "", flags=re.IGNORECASE))

    def _is_italic_flags(self, flags: int) -> bool:
        try:
            return bool(flags & 2)
        except Exception:
            return False

    def _looks_like_code_line(self, text: str) -> bool:
        if not text: return False
        code_tokens = ["{", "}", ";", "::", "->", "=>", "==", "!=", "<=", ">=", "#", "/", "\\"]
        if any(tok in text for tok in code_tokens): return True
        if re.search(r"\b(def|class|return|if|else|for|while|import|from|public|private|struct|enum)\b", text): return True
        if re.search(r"(/[^\s]+|\\[^\s]+|^[\$#]\s+)", text.strip()): return True
        return False

    def _looks_like_math_line(self, text: str) -> bool:
        if not text:
            return False
        stripped = text.strip()
        if len(stripped) <= 2:
            return False
        math_syms = sum(1 for ch in stripped if ch in "=+*/^_[](){}<>|∑∫√−×±")
        letters = sum(1 for ch in stripped if ch.isalpha())
        if math_syms >= 2 and letters <= max(8, len(stripped) * 0.4):
            return True
        if re.search(r"\\(sum|frac|int|prod|sqrt|omega|pi)\b", stripped):
            return True
        if re.search(r"\b[a-zA-Z]\s*[=\+\-]\s*[a-zA-Z0-9]", stripped):
            return True
        return False

    def _is_inline_math_span(self, span_text: str, font: str, flags: int = 0) -> bool:
        token = (span_text or "").strip()
        if not token or (token.startswith("$") and token.endswith("$")) or "$$" in token: return False
        if not (self._is_italic_flags(flags) or re.search(r"(italic|math)", font or "", flags=re.IGNORECASE)):
            return False
        if re.search(r"\s", token) or len(token) > 40: return False
        if re.fullmatch(r"[A-Za-z](?:\[[^\]]+\])?", token): return True
        if re.search(r"[\[\]_\^\\]|\d", token): return True
        return False

    def _wrap_inline_math(self, token: str) -> str:
        token = (token or "").strip()
        if not token: return ""
        if token.startswith("$") and token.endswith("$"): return token
        return f"${token}$"

    def _should_ocr_formula_block(self, raw_text: str, span_stats: Dict[str, Any]) -> bool:
        if not raw_text: return True
        if self._is_garbled_text(raw_text): return True
        if not self._detect_formula(raw_text): return False

        wrapped_spans = int(span_stats.get("wrapped_spans") or 0)
        math_span_ratio = float(span_stats.get("math_span_ratio") or 0.0)
        symbol_span_ratio = float(span_stats.get("symbol_span_ratio") or 0.0)
        math_line_ratio = float(span_stats.get("math_line_ratio") or 0.0)

        letters = sum(1 for ch in raw_text if ch.isalpha())
        math_syms = sum(1 for ch in raw_text if ch in "=+*/^_[](){}<>|∑∫√−×±")
        length = len(raw_text)

        if length >= 60 and (letters / max(1, length)) > 0.6 and math_syms <= 1:
            if math_span_ratio < 0.2 and symbol_span_ratio < 0.1 and math_line_ratio < 0.2:
                return False

        if math_line_ratio >= 0.3 or math_span_ratio >= 0.25 or symbol_span_ratio >= 0.2:
            return True

        if any(tok in raw_text for tok in ("=", "∑", "Σ", "\u222b")):
            return True
        if re.search(r"\\(sum|frac|int|prod|infty)\b", raw_text):
            return True
        if re.search(r"[_\^]\{", raw_text):
            return True

        if wrapped_spans <= 1 and len(raw_text) <= 80: return False
        return math_line_ratio >= 0.2

    def _needs_markdown_polish(self, text: str) -> bool:
        if not text: return False
        if re.search(r"\b(?:[A-Za-z]\s){2,}[A-Za-z]\b", text): return True
        if re.search(r"\\mathrm\{(?:[A-Za-z]\s){2,}[A-Za-z]\}", text): return True
        return False

    def _get_markdown_polish_prompt(self, text: str) -> str:
        return (
            "You are a meticulous technical editor.\n"
            "Task: Clean up OCR artifacts and improve Markdown readability WITHOUT changing meaning.\n"
            "Rules:\n"
            "- Fix spaced-letter artifacts like 'T h e' -> 'The'.\n"
            "- Preserve all LaTeX; do not alter math except removing obvious OCR spaces inside words.\n"
            "- Do NOT add new content.\n"
            "Output ONLY the corrected text.\n\n"
            f"INPUT:\n{text}"
        )

    def _get_final_stitch_prompt(self, text: str, formula_blocks: List[Dict], visual_blocks: List[Dict]) -> str:
        formulas = "\n".join([f"- {b.get('placeholder')}: {b.get('latex')}" for b in formula_blocks if b.get("placeholder") and b.get("latex")])
        visuals = "\n".join([f"- {b.get('placeholder')} ({b.get('type')}): {b.get('text')}" for b in visual_blocks if b.get("placeholder") and b.get("text")])
        return (
            "你是一个多领域技术专家。请根据以下内容重新编排文档：\n"
            "1. [Text]：原始文本。\n"
            "2. [Formula]：LaTeX 公式。\n"
            "3. [Visual]：图片/表格描述。\n\n"
            "请将它们根据 [IMAGE_BLOCK_{ID}] 占位符重新编排：\n"
            "- 严禁修改代码、路径。\n"
            "- 修正换行和无效空格。\n"
            "- 最终输出 Markdown。\n\n"
            f"[Text]\n{text}\n\n"
            f"[Formula]\n{formulas if formulas else '(none)'}\n\n"
            f"[Visual]\n{visuals if visuals else '(none)'}"
        )

    def _clean_llm_response(self, text: str) -> str:
        if not text: return ""
        text = re.sub(r"^(Sure|Certainly|Here|Okay).*?[:\n]", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
        lines = [re.sub(r"^\s*>+\s?", "", line) for line in text.splitlines()]
        return "\n".join(lines).strip()

    def _invoke_llm_text(self, prompt: str) -> str:
        llm = get_llm()
        response = llm.complete(prompt=prompt)
        return response.text

    def _invoke_llm_with_retry(self, prompt: str) -> str:
        last_error = None
        for attempt in range(self.vlm_retry_attempts):
            try:
                return self._invoke_llm_text(prompt)
            except Exception as e:
                last_error = e
                time.sleep(self.vlm_retry_base_delay * (2 ** attempt))
        if last_error: raise last_error
        return ""

    def _maybe_polish_nodes_with_vlm(self, nodes: List[TextNode]) -> List[TextNode]:
        polished: List[TextNode] = []
        for node in nodes:
            if node.metadata.get("node_type") == "image":
                polished.append(node)
                continue
            text = getattr(node, "text", "") or ""
            visual_blocks = node.metadata.get("visual_blocks", []) or []
            formula_blocks = node.metadata.get("formula_blocks", []) or []
            has_placeholders = bool(visual_blocks or formula_blocks) and "[IMAGE_BLOCK_" in text
            locally_fixed = self._fix_spaced_letters(text)
            if not has_placeholders and not self._needs_markdown_polish(locally_fixed):
                node.text = locally_fixed
                polished.append(node)
                continue
            
            cache_key = locally_fixed + ("\n" + str(visual_blocks) + "\n" + str(formula_blocks) if has_placeholders else "")
            text_hash = hashlib.md5(cache_key.encode("utf-8")).hexdigest()
            cache_file = self.cache_dir / f"{text_hash}_markdown_polish_v2.txt"
            if cache_file.exists():
                try:
                    node.text = cache_file.read_text(encoding="utf-8")
                except Exception:
                    node.text = locally_fixed
                polished.append(node)
                continue
            try:
                prompt = self._get_final_stitch_prompt(locally_fixed, formula_blocks, visual_blocks) if has_placeholders else self._get_markdown_polish_prompt(locally_fixed)
                corrected = self._clean_llm_response(self._invoke_llm_with_retry(prompt))
                node.text = self._fix_spaced_letters(corrected)
                cache_file.write_text(node.text, encoding="utf-8")
            except Exception as e:
                logger.warning(f"Polish failed: {e}")
                node.text = locally_fixed
            polished.append(node)
        return polished

    def _is_section_title(self, text: str, bbox: fitz.Rect, max_font_size: float, body_font_size: float, page_height: float) -> bool:
        if len(text) > 100 or len(text) < 2: return False
        if re.match(r"^\d+$", text) or re.match(r"^\d+\s*[-/]\s*\d+$", text): return False
        patterns = [r"^Chapter\s+\d+", r"^Section\s+\d+", r"^Lecture\s+\d+", r"^Part\s+[IVX]+", r"^Table\s+of\s+Contents", r"^References", r"^Appendix", r"^\d+(\.\d+)*\s+[\w\s]+", r"^[IVX]+\.\s+[\w\s]+"]
        for p in patterns:
            if re.match(p, text, re.IGNORECASE):
                if bbox and page_height and bbox.y0 > page_height * 0.6: return False
                if body_font_size and max_font_size and max_font_size < body_font_size * 1.2: return False
                return True
        letters = re.sub(r"[^A-Za-z]+", "", text)
        is_all_caps = bool(letters) and letters.isupper() and len(letters) >= 4
        if is_all_caps:
            if bbox and page_height and bbox.y0 > page_height * 0.75: return False
            if body_font_size and max_font_size and max_font_size < body_font_size * 1.1: return False
            return True
        if body_font_size and max_font_size and max_font_size >= body_font_size * 1.6:
            if bbox and page_height and bbox.y0 <= page_height * 0.7: return True
        return False

    def _build_node(self, buffer: List[dict], section: str, blob: Blob) -> List[TextNode]:
        texts = [item["text"] for item in buffer if item.get("text")]
        content = "\n".join(texts)
        visual_blocks = []
        formula_blocks = []
        for item in buffer:
            block_id = item.get("block_id")
            if not block_id: continue
            placeholder = f"[IMAGE_BLOCK_{block_id}]"
            if item.get("kind") in ("image", "table"):
                visual_blocks.append({"id": block_id, "placeholder": placeholder, "type": item.get("kind"), "text": item.get("content", "").strip()})
            elif item.get("kind") == "formula":
                formula_blocks.append({"id": block_id, "placeholder": placeholder, "latex": item.get("content", "").strip()})
        img_paths = list(set([item["img_path"] for item in buffer if item.get("img_path")]))
        pages = sorted(list(set(item["page"] for item in buffer)))
        metadata = blob.metadata.copy() if blob.metadata else {}
        metadata.update({
            "source": blob.source,
            "file_name": metadata.get("file_name", Path(blob.source).name),
            "section": section,
            "page": pages[0] if pages else 0,
            "page_numbers": pages,
            "image_paths": img_paths,
            "has_formula": any(item.get("is_formula") for item in buffer),
            "visual_blocks": visual_blocks,
            "formula_blocks": formula_blocks,
        })
        node = TextNode(text=content, metadata=metadata)
        doc_id = metadata.get("doc_id") or blob.source
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
            node_id=doc_id,
            metadata={"file_name": Path(blob.source).name, "source": blob.source}
        )
        nodes = [node]
        for item in buffer:
            if item.get("kind") == "image" and item.get("img_path"):
                img_node = ImageNode(
                    text=item.get("description", "").strip(),
                    image_path=item["img_path"],
                    metadata={"node_type": "image", "source": blob.source, "section": section, "page": item.get("page", 0), "image_path": item["img_path"], "doc_id": doc_id}
                )
                img_node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
                    node_id=doc_id,
                    metadata={"file_name": Path(blob.source).name, "source": blob.source}
                )
                img_node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(node_id=node.node_id, metadata={"section": section})
                nodes.append(img_node)
        return nodes

    def _resolve_pending_texts(self, buffer: List[dict]) -> None:
        for item in buffer:
            future = item.get("future")
            if not future: continue
            try:
                result_text = future.result()
            except Exception as e:
                logger.error(f"VLM future failed: {e}")
                result_text = ""
            item["future"] = None
            placeholder = item.get("text") or ""
            if item.get("block_id"):
                placeholder = f"[IMAGE_BLOCK_{item['block_id']}]"
            if item.get("kind") == "image":
                desc = self._clean_vlm_response(result_text) if result_text else ""
                if not desc:
                    desc = "Figure (description unavailable)."
                item.update({"description": desc, "content": desc, "text": f"{placeholder}\n{desc}" if desc else placeholder})
            elif item.get("kind") == "table":
                cleaned = self._clean_vlm_response(result_text) if result_text else ""
                item.update({"content": f"\n{cleaned}\n" if cleaned else "", "text": f"{placeholder}\n{cleaned}" if cleaned else placeholder})
            elif item.get("kind") == "formula":
                cleaned = result_text.strip() if result_text else item.get("raw_text", "")
                cleaned = self._postprocess_formula_text(cleaned)
                item.update({"content": cleaned, "text": f"{placeholder}\n{cleaned}" if cleaned else placeholder})

    def _estimate_body_font_size(self, blocks: List[dict]) -> float:
        sizes = []
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span.get("size"): sizes.append(span["size"])
        return float(statistics.median(sizes)) if sizes else 0.0

    def _get_block_max_font_size(self, block: dict) -> float:
        max_size = 0.0
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if span.get("size", 0.0) > max_size: max_size = span["size"]
        return max_size

    def _merge_formula_blocks(self, blocks: List[dict], page_height: float) -> Dict[str, Any]:
        """合并相邻的公式 Block。"""
        candidates = []
        for i, block in enumerate(blocks):
            if block.get("type", 0) != 0: continue
            raw_text, span_stats = self._extract_text_with_inline_math(block)
            if self._should_ocr_formula_block(raw_text, span_stats):
                candidates.append((i, fitz.Rect(block.get("bbox"))))
        
        regions = []
        index_map: Dict[int, Dict[str, Any]] = {}
        if not candidates: return {"regions": regions, "index_map": index_map}

        def _h_overlap(a: fitz.Rect, b: fitz.Rect) -> float:
            inter = min(a.x1, b.x1) - max(a.x0, b.x0)
            return inter / min(a.width, b.width) if inter > 0 else 0.0

        current_indices = [candidates[0][0]]
        current_bbox = candidates[0][1]

        for idx, bbox in candidates[1:]:
            v_gap = max(0, bbox.y0 - current_bbox.y1)
            v_close = v_gap <= min(10, page_height * 0.02)
            h_overlap = _h_overlap(current_bbox, bbox) >= 0.5
            if v_close and h_overlap:
                current_bbox = current_bbox | bbox
                current_indices.append(idx)
            else:
                regions.append((current_indices, current_bbox))
                current_indices = [idx]
                current_bbox = bbox
        regions.append((current_indices, current_bbox))

        for indices, bbox in regions:
            for j, b_idx in enumerate(indices):
                index_map[b_idx] = {"bbox": bbox, "is_first": j == 0}
        return {"regions": regions, "index_map": index_map}