import os
import json
from pathlib import Path
from loader import detect_pdf_type
from extract_layout import extract_layout_text
from extract_ocr import extract_ocr_text
from labeler import DocumentLabeler

INPUT_DIR = "inputs"
OUTPUT_DIR = "outputs"

# Ensure the outputs directory exists
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# Initialize the labeler once for efficiency
labeler = DocumentLabeler()

def process_pdf(pdf_path):
    """Process a single PDF through extraction and labeling pipeline"""
    print(f"🔍 Analyzing: {pdf_path}")
    pdf_type = detect_pdf_type(pdf_path)
    print(f"📘 Detected type: {pdf_type}")
    # Step 1: Extract content based on PDF type
    if pdf_type == "scanned":
        result = extract_ocr_text(pdf_path)
    else:
        result = extract_layout_text(pdf_path)
    # Step 2: Save initial extraction
    out_path = Path(OUTPUT_DIR) / (Path(pdf_path).stem + ".json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"📄 Extraction saved to: {out_path}")
    # Step 3: Apply hierarchical labeling and overwrite JSON
    print(f"🏷️ Applying hierarchical labeling...")
    try:
        labeling_success = labeler.process_single_file(out_path)
        if labeling_success:
            print(f"✅ Hierarchical labeling completed")
            # Show quick stats for the file in-place
            if out_path.exists():
                with open(out_path, "r", encoding="utf-8") as f:
                    final_data = json.load(f)
                title_status = "✓" if final_data.get('title') else "✗"
                outline_count = len(final_data.get('outline', []))
                print(f"📊 Schema output - Title: {title_status}, Outline items: {outline_count}")
                print(f"📁 File: {out_path}")
        else:
            print(f"⚠️ Hierarchical labeling failed for: {out_path}")
    except Exception as e:
        print(f"❌ Labeling error for {pdf_path}: {str(e)}")
    print()  # Empty line for better readability

def process_all_extracted_files():
    """Apply hierarchical labeling to all existing JSON files in output directory"""
    print(f"🏷️ Applying hierarchical labeling to all files in {OUTPUT_DIR}...")
    json_files = list(Path(OUTPUT_DIR).glob("*.json"))
    if not json_files:
        print("❌ No JSON files found in output directory.")
        return False
    success = labeler.process_output_folder(OUTPUT_DIR)
    return success

def main():
    print(f"📂 Scanning directory: {INPUT_DIR}")
    pdf_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".pdf")])
    if not pdf_files:
        print("❌ No PDFs found.")
        return
    print(f"🚀 Starting processing pipeline for {len(pdf_files)} PDF(s)...\n")
    # Process each PDF through extraction and labeling
    for file in pdf_files:
        pdf_path = os.path.join(INPUT_DIR, file)
        process_pdf(pdf_path)
    print("🎉 All PDFs processed through complete pipeline!")
    # Optional: Show final summary
    show_final_summary()

def show_final_summary():
    """Display final processing summary"""
    output_files = list(Path(OUTPUT_DIR).glob("*.json"))
    print(f"\n📈 FINAL SUMMARY")
    print(f"================")
    print(f"Schema files (outputs): {len(output_files)}")
    if output_files:
        print(f"\n📁 Schema-compliant files in '{OUTPUT_DIR}' folder:")
        title_count = 0
        total_outline_items = 0
        for output_file in output_files[:5]:
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                has_title = bool(data.get('title'))
                outline_count = len(data.get('outline', []))
                if has_title:
                    title_count += 1
                total_outline_items += outline_count
                print(f" 📄 {output_file.name}: Title: {'✓' if has_title else '✗'}, Outline: {outline_count} items")
            except:
                print(f" ❌ {output_file.name}: Error reading file")
        if len(output_files) > 5:
            print(f" ... and {len(output_files) - 5} more files")
        print(f"\n📊 Overall: {title_count}/{len(output_files)} files have titles, {total_outline_items} total outline items")

def batch_label_existing_files():
    """Standalone function to label existing extracted files"""
    print("🏷️ Batch labeling existing files...")
    return process_all_extracted_files()

if __name__ == "__main__":
    import sys
    # Support command line arguments for different modes
    if len(sys.argv) > 1:
        if sys.argv[1] == "--label-only":
            # Only apply labeling to existing JSON files
            success = batch_label_existing_files()
            sys.exit(0 if success else 1)
        elif sys.argv[1] == "--help":
            print("Usage:")
            print(" python main.py # Full pipeline (extract + label)")
            print(" python main.py --label-only # Only apply labeling to existing JSONs")
            print(" python main.py --help # Show this help")
            sys.exit(0)
    # Default: run full pipeline
    main()
