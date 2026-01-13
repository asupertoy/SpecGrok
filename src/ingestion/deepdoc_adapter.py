import sys
import os
import re

# Add libs to path so deepdoc can find 'common' and 'rag'
# This must be done BEFORE importing from deepdoc/common/rag
current_dir = os.path.dirname(os.path.abspath(__file__))
libs_dir = os.path.join(current_dir, 'libs')
if libs_dir not in sys.path:
    sys.path.insert(0, libs_dir)

try:
    from deepdoc.parser.pdf_parser import RAGFlowPdfParser
except ImportError as e:
    print(f"Error importing RAGFlowPdfParser: {e}")
    # This might happen if common/rag are not found in libs
    raise

class DeepDocAdapter(RAGFlowPdfParser):
    """
    Adapter for RAGFlow's DeepDoc PDF Parser to be used in the ingestion pipeline.
    Inherits from RAGFlowPdfParser to access internal methods for layout analysis.
    """
    def __init__(self):
        try:
            super().__init__()
        except Exception as e:
            print(f"Warning: Initialization of DeepDoc Parser failed (likely due to missing models): {e}")
            raise

    def parse_pdf(self, file_path: str, from_page=0, to_page=100000, zoomin=3):
        """
        Parses a PDF file and returns structured sections and tables.
        
        Args:
            file_path: Path to the PDF file.
            from_page: Start page index.
            to_page: End page index.
            zoomin: Zoom factor for OCR/Image processing.
            
        Returns:
            dict: {
                "sections": [(text, layout_label), ...],
                "tables": [table_data...],
                "images": [image_data...] 
            }
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        def callback(prog, msg):
            print(f"[DeepDoc] {msg}")

        # 1. Convert PDF to images
        self.__images__(file_path, zoomin, from_page, to_page, callback)
        
        # 2. Layout Analysis
        self._layouts_rec(zoomin)
        
        # 3. Table Extraction
        self._table_transformer_job(zoomin)
        
        # 4. Text Merge and Organization
        self._text_merge()
        
        # 5. Extract Tables and Figures
        # _extract_table_figure(need_image, ZM, return_html, need_position)
        tbls = self._extract_table_figure(True, zoomin, True, True) 
        
        # 6. Concat columns/pages
        self._concat_downward()
        
        # 7. Filter pages
        self._filter_forpages()
        
        # 8. Clean text
        for b in self.boxes:
            if "text" in b:
                b["text"] = re.sub(r"([\t 　]|\u3000){2,}", " ", b["text"].strip())
            
        # 9. Extract sections
        # Filter boxes that are text or title
        sections = [
            (b["text"] + self._line_tag(b, zoomin), b.get("layoutno", "")) 
            for b in self.boxes 
            if re.match(r"(text|title)", b.get("layoutno", "text"))
        ]
        
        return {
            "sections": sections,
            "tables": tbls
        }

if __name__ == "__main__":
    # Test block
    adapter = DeepDocAdapter()
    print("DeepDocAdapter initialized successfully.")
