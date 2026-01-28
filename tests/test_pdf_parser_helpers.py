import pytest

from src.ingestion.parsers.parser_pdf import PDFParser


class TestPDFParserHelpers:
    def test_garbled_text_detection(self):
        parser = PDFParser(vlm_enabled=False, pix2tex_enabled=False)
        assert parser._is_garbled_text("Hello world") is False
        # 混入占位符字符，比例超过阈值应视为乱码
        assert parser._is_garbled_text("� � � �") is True

    def test_protocol_prompt_selected(self):
        parser = PDFParser(vlm_enabled=False, pix2tex_enabled=False)
        parser._current_doc_domain = "HTTP"
        parser._current_doc_source = "/tmp/http_doc.pdf"
        prompt = parser._get_prompt_by_mode("description")
        assert "HTTP/协议" in prompt
