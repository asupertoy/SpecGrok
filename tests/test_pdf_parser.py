import os
import sys
from pathlib import Path

# Add project root to path (so 'src' package is found)
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest

from src.ingestion.parsers.parser_pdf import PDFParser
from src.ingestion.loaders import Blob


class TestPDFParser:
    """测试 PDFParser 类的功能，包括基本解析和 VLM 功能。"""

    @pytest.fixture
    def pdf_file_path(self):
        """测试用的 PDF 文件路径。"""
        return "/home/john/project/SpecGrok/data/raw_with_others/dsp/pdf/dft_core.pdf"

    def test_parse_without_vlm(self, pdf_file_path):
        """测试禁用 VLM 的 PDF 解析。"""
        blob = Blob.from_path(pdf_file_path)
        parser = PDFParser(vlm_enabled=False)
        nodes = parser.parse(blob)

        # 断言解析成功，返回节点列表
        assert isinstance(nodes, list)
        assert len(nodes) > 0

        # 检查第一个节点的结构
        first_node = nodes[0]
        assert hasattr(first_node, 'text')
        assert hasattr(first_node, 'metadata')
        assert first_node.text.strip() != ""
        assert "source" in first_node.metadata
        assert first_node.metadata["source"] == pdf_file_path
        assert "section" in first_node.metadata
        assert "page" in first_node.metadata
        assert "page_numbers" in first_node.metadata
        assert "image_paths" in first_node.metadata
        assert "has_formula" in first_node.metadata
        assert isinstance(first_node.metadata["has_formula"], bool)

    def test_parse_with_vlm(self, pdf_file_path):
        """测试启用 VLM 的 PDF 解析。"""
        print("=== 开始测试启用 VLM 的 PDF 解析 ===")
        blob = Blob.from_path(pdf_file_path)
        print(f"Blob 创建成功，源文件: {blob.source}")
        
        parser = PDFParser(vlm_enabled=True)
        print("PDFParser 初始化完成，VLM 已启用")
        
        nodes = parser.parse(blob)
        print(f"解析完成，共生成 {len(nodes)} 个节点")

        # 断言解析成功，返回节点列表
        assert isinstance(nodes, list)
        assert len(nodes) > 0

        # 检查节点内容
        first_node = nodes[0]
        print(f"第一个节点的文本预览: {first_node.text[:200]}...")
        print(f"第一个节点的元数据: {first_node.metadata}")
        
        assert hasattr(first_node, 'text')
        assert hasattr(first_node, 'metadata')
        assert first_node.text.strip() != ""
        assert "source" in first_node.metadata
        assert first_node.metadata["source"] == pdf_file_path
        assert "section" in first_node.metadata
        assert "page" in first_node.metadata
        assert "page_numbers" in first_node.metadata
        assert "image_paths" in first_node.metadata
        assert "has_formula" in first_node.metadata
        assert isinstance(first_node.metadata["has_formula"], bool)

        # 检查是否有公式（如果 VLM 检测到）
        has_formula_nodes = any(node.metadata.get("has_formula") for node in nodes)
        if has_formula_nodes:
            print("检测到公式节点！")
            formula_count = sum(1 for node in nodes if node.metadata.get("has_formula"))
            print(f"共有 {formula_count} 个节点包含公式")
        else:
            print("未检测到公式节点")
        
        # 检查图片路径
        image_nodes = [node for node in nodes if node.metadata.get("node_type") == "image"]
        if image_nodes:
            print(f"检测到 {len(image_nodes)} 个图片节点")
            # If VLM is enabled, we expect at least one image to have a non-empty description
            assert any(n.text.strip() for n in image_nodes), "VLM enabled but image descriptions are empty"
        else:
            print("未检测到图片节点")
        
        print("=== VLM 测试完成 ===")

        # 保存测试结果到 txt 文件
        output_file = os.path.join(current_dir, "test_output_files", "test_pdf_parser_vlm_results.txt")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("VLM 测试结果\n")
            f.write("=" * 50 + "\n")
            f.write(f"总节点数: {len(nodes)}\n\n")
            for i, node in enumerate(nodes):
                f.write(f"节点 {i}:\n")
                f.write(f"  文本: {node.text}...\n")  
                f.write(f"  元数据: {node.metadata}\n")
                f.write("\n")
        print(f"测试结果已保存到: {output_file}")
