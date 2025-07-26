import fitz  # PyMuPDF

def detect_pdf_type(pdf_path):
    doc = fitz.open(pdf_path)
    sample_page = doc[0]
    text = sample_page.get_text().strip()
    return "scanned" if not text else "layout"
