import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock
import fitz

# Ensure we can import the `src` package
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ingestion.parsers.parser_pdf import PDFParser
from ingestion.loaders import Blob, Loader

def test_pdf_parser_integration(tmp_path):
    # 1. 创建一个模拟的 PDF 文件
    pdf_path = tmp_path / "test.pdf"
    doc = fitz.open()
    page = doc.new_page()
    # 使用独立的 insert_text 调用确保生成不同的 blocks
    # Title
    page.insert_text((50, 50), "Chapter 1 Introduction")
    # Content (simulate some distance to ensure new block)
    page.insert_text((50, 80), "This is a test PDF.")
    doc.save(str(pdf_path))
    doc.close()

    # 调试辅助：打印生成的 PDF 结构以确认 PyMuPDF 能读到
    doc_debug = fitz.open(str(pdf_path))
    for p in doc_debug:
        print(f"DEBUG PAGE: {p.get_text('dict')}")
    doc_debug.close()

    # 2. 使用 Loader 加载文件得到 Blob
    loader = Loader(extensions=['.pdf'])
    blobs = loader.load(pdf_path)
    assert len(blobs) == 1
    blob = blobs[0]
    
    # 3. 验证 Blob 属性
    assert blob.metadata['extension'] == '.pdf'
    assert blob.metadata['file_name'] == 'test.pdf'

    # 4. 使用 PDFParser 解析 Blob
    parser = PDFParser()
    documents = parser.parse(blob)

    # 5. 验证解析结果
    assert len(documents) > 0
    doc_res = documents[0]
    
    # 根据 PDFParser 实现：
    # 标题文本被作为 section metadata 提取，并不包含在 content text 中
    # 正文内容应该在 text 中
    assert "This is a test PDF" in doc_res.text
    
    # 验证 metadata 传递和提取
    assert doc_res.metadata['file_name'] == 'test.pdf'
    assert doc_res.metadata['extension'] == '.pdf'
    
    # 验证 nodes 信息
    nodes = doc_res.metadata.get('nodes')
    assert nodes is not None
    assert len(nodes) > 0
    node = nodes[0]
    
    # 验证 section 标题提取到了 metadata 中
    assert node.metadata['section'] == "Chapter 1 Introduction"
    # 验证 Blob 和 Node 的关联
    assert node.metadata['file_name'] == 'test.pdf'
    assert node.metadata['source'] == str(pdf_path)

def test_pdf_parser_load_data_compatibility(tmp_path):
    # 测试旧接口 load_data 是否正常工作 (它内部调用 self.parse)
    pdf_path = tmp_path / "test_legacy.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Legacy Test")
    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    documents = parser.load_data(str(pdf_path))
    
    assert len(documents) > 0
    assert "Legacy Test" in documents[0].text
    assert documents[0].metadata['file_name'] == 'test_legacy.pdf'


def test_parse_real_rfc_pdf():
    # 使用仓库中的真实 RFC PDF（若存在）进行集成测试
    pdf_path = ROOT / "data" / "raw" / "rfc" / "http" / "rfc7230_http1.1.pdf"
    if not pdf_path.exists():
        pytest.skip(f"Test PDF not found: {pdf_path}")

    loader = Loader(extensions=['.pdf'])
    blobs = loader.load(pdf_path)
    assert len(blobs) > 0
    blob = blobs[0]

    parser = PDFParser()
    documents = parser.parse(blob)

    assert len(documents) > 0
    doc_res = documents[0]

    # 基本断言：文件名与扩展名在 metadata 中
    assert doc_res.metadata['file_name'] == pdf_path.name
    assert doc_res.metadata['extension'] == '.pdf'

    # 至少有一些文本内容
    assert doc_res.text is not None
    assert len(doc_res.text) > 0

    # nodes 包含 metadata，且 source 匹配 Blob.source
    nodes = doc_res.metadata.get('nodes')
    assert nodes is not None and len(nodes) > 0
    assert nodes[0].metadata['source'] == str(pdf_path)


# Parameterize across all PDFs found in the data/raw directory
raw_dir = ROOT / "data" / "raw"
pdf_paths = list(raw_dir.rglob('*.pdf'))
PDF_IDS = [p.name for p in pdf_paths]


@pytest.mark.parametrize("pdf_path", pdf_paths, ids=PDF_IDS)
def test_parse_pdf_summary(pdf_path):
    """For each PDF, run Loader+PDFParser and print a summary: first paragraph, page count, node count."""
    parser = PDFParser()

    loader = Loader(extensions=['.pdf'])
    blobs = loader.load(pdf_path)
    assert len(blobs) > 0, f"Loader failed to find blob for {pdf_path}"

    parsed_any = False
    for blob in blobs:
        docs = parser.parse(blob)
        assert len(docs) > 0, f"Parser returned no documents for {pdf_path}"
        for d in docs:
            text = d.text or ""
            # first paragraph heuristics
            first_para = ""
            if text.strip():
                parts = [p for p in text.split('\n\n') if p.strip()]
                if parts:
                    first_para = parts[0].strip()
                else:
                    first_para = text.strip().splitlines()[0].strip()

            nodes = d.metadata.get('nodes') or []
            node_count = len(nodes)

            # page count using fitz for summary
            try:
                doc_fit = fitz.open(str(pdf_path))
                page_count = len(doc_fit)
                doc_fit.close()
            except Exception:
                page_count = None

            # Print summary (pytest captures this output per test)
            print(f"PARSE SUMMARY: {pdf_path.name} | pages={page_count} | nodes={node_count} | first_para={first_para[:200]}")

            if text and len(text.strip()) > 0:
                parsed_any = True
                # basic metadata validation
                assert d.metadata['file_name'] == pdf_path.name
                assert d.metadata['extension'] == '.pdf'
                break
    assert parsed_any, f"No non-empty text parsed from {pdf_path}"

