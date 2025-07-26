# src/main.py

import os
import json
from pathlib import Path
from loader import detect_pdf_type
from extract_layout import extract_layout_text
from extract_ocr import extract_ocr_text

INPUT_DIR = "inputs"
OUTPUT_DIR = "outputs"
Path(OUTPUT_DIR).mkdir(exist_ok=True)

def process_pdf(pdf_path):
    print(f"🔍 Analyzing: {pdf_path}")
    pdf_type = detect_pdf_type(pdf_path)
    print(f"📘 Detected type: {pdf_type}")

    if pdf_type == "scanned":
        result = extract_ocr_text(pdf_path)
    else:
        result = extract_layout_text(pdf_path)

    out_path = Path(OUTPUT_DIR) / (Path(pdf_path).stem + ".json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"✅ Output saved to: {out_path}\n")

def main():
    print(f"📂 Scanning directory: {INPUT_DIR}")
    pdf_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".pdf")])
    
    if not pdf_files:
        print("❌ No PDFs found.")
        return
    
    for file in pdf_files:
        pdf_path = os.path.join(INPUT_DIR, file)
        process_pdf(pdf_path)

    print("🎉 All PDFs processed!")

if __name__ == "__main__":
    main()
