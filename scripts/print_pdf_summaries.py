from pathlib import Path
import sys
import fitz

# ensure project src is in PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from src.ingestion.loaders import Loader
from src.ingestion.parsers.parser_pdf import PDFParser

RAW_DIR = ROOT / "data" / "raw"


def first_paragraph(text: str) -> str:
    if not text:
        return ""
    parts = [p for p in text.split('\n\n') if p.strip()]
    if parts:
        return parts[0].strip()
    return text.strip().splitlines()[0].strip()


def summarize_pdf(pdf_path: Path):
    loader = Loader(extensions=['.pdf'])
    parser = PDFParser()

    try:
        blobs = loader.load(pdf_path)
    except Exception as e:
        print(f"ERROR loading {pdf_path}: {e}")
        return

    for blob in blobs:
        try:
            docs = parser.parse(blob)
        except Exception as e:
            print(f"ERROR parsing {pdf_path}: {e}")
            continue

        for d in docs:
            text = d.text or ""
            fp = first_paragraph(text)
            nodes = d.metadata.get('nodes') or []
            node_count = len(nodes)
            try:
                page_count = len(fitz.open(str(pdf_path)))
            except Exception:
                page_count = None

            print(f"{pdf_path.name} | pages={page_count} | nodes={node_count} | first_para={fp[:200]}")


if __name__ == '__main__':
    pdfs = sorted(RAW_DIR.rglob('*.pdf'))
    if not pdfs:
        print(f"No PDFs found under {RAW_DIR}")

    for p in pdfs:
        summarize_pdf(p)