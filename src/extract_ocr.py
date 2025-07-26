import fitz
import pytesseract
import cv2
import numpy as np
from PIL import Image
import io

def extract_ocr_text(pdf_path):
    doc = fitz.open(pdf_path)
    results = []
    for page_num in range(len(doc)):
        pix = doc[page_num].get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes()))
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(img_cv, 150, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(thresh)
        for line in text.split("\n"):
            line = line.strip()
            if line:
                results.append({
                    "text": line,
                    "features": {
                        "source": "OCR",
                        "page": page_num + 1
                    }
                })
    return results
