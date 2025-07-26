import fitz  # PyMuPDF
import numpy as np
from collections import defaultdict


def extract_layout_text(pdf_path):
    doc = fitz.open(pdf_path)
    paragraphs = []
    all_font_sizes = []

    page_paragraphs = []

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")['blocks']
        lines_group = []

        for block in blocks:
            if block['type'] != 0:
                continue
            for line in block['lines']:
                spans = line['spans']
                line_text = " ".join([span['text'].strip() for span in spans if span['text'].strip()])

                if not line_text or len(line_text) <= 1:
                    continue

                font_sizes = [span['size'] for span in spans]
                fonts = [span['font'] for span in spans]
                is_bold = any("Bold" in font for font in fonts)
                font_size = round(np.mean(font_sizes), 2)

                all_font_sizes.append(font_size)

                bbox = line['bbox']
                y0 = bbox[1]
                y1 = bbox[3]
                height = y1 - y0
                alignment = get_alignment(spans)
                text_case = get_text_case(line_text)

                line_features = {
                    "text": line_text,
                    "font_size": font_size,
                    "bbox": bbox,
                    "is_bold": is_bold,
                    "alignment": alignment,
                    "text_case": text_case,
                    "font_names": list(set(fonts)),
                    "line_height": height,
                    "y0": y0,
                    "y1": y1
                }
                lines_group.append(line_features)

        # group by vertical proximity (paragraph detection)
        grouped_paragraphs = group_into_paragraphs(lines_group)
        page_paragraphs.extend(grouped_paragraphs)

    # Now calculate relative font size rankings
    font_size_ranks = build_relative_font_ranks(all_font_sizes)

    for para in page_paragraphs:
        fs = para["avg_font_size"]
        para["relative_font_size"] = font_size_ranks.get(fs, None)
        paragraphs.append(para)

    return paragraphs


def build_relative_font_ranks(font_sizes):
    """
    Takes a list of font sizes and returns a dict:
    {font_size: relative_rank}, where 0 is smallest.
    """
    unique_sorted = sorted(set(font_sizes))
    return {fs: idx for idx, fs in enumerate(unique_sorted)}


def group_into_paragraphs(lines):
    if not lines:
        return []

    lines.sort(key=lambda x: x['y0'])
    paragraphs = []
    current_para = [lines[0]]

    for i in range(1, len(lines)):
        prev = current_para[-1]
        curr = lines[i]
        spacing = curr['y0'] - prev['y1']
        curr['line_spacing'] = spacing

        # heuristics: same alignment, small spacing, similar font size
        if spacing < 15 and abs(curr['font_size'] - prev['font_size']) < 2:
            current_para.append(curr)
        else:
            paragraphs.append(aggregate_paragraph(current_para))
            current_para = [curr]

    if current_para:
        paragraphs.append(aggregate_paragraph(current_para))

    return paragraphs


def aggregate_paragraph(lines):
    para_text = " ".join([l['text'] for l in lines])
    font_sizes = [l['font_size'] for l in lines]
    bbox = [
        min(l['bbox'][0] for l in lines),
        min(l['bbox'][1] for l in lines),
        max(l['bbox'][2] for l in lines),
        max(l['bbox'][3] for l in lines)
    ]
    spacing_vals = [l.get("line_spacing", 0) for l in lines if l.get("line_spacing", 0) > 0]

    return {
        "text": para_text,
        "avg_font_size": round(np.mean(font_sizes), 2),
        "bbox": bbox,
        "is_bold": any(l['is_bold'] for l in lines),
        "is_upper": para_text.isupper(),
        "alignment": most_common([l['alignment'] for l in lines]),
        "line_count": len(lines),
        "line_spacing_avg": round(np.mean(spacing_vals), 2) if spacing_vals else 0.0,
        "font_names": list(set(f for l in lines for f in l['font_names'])),
        "text_case": most_common([l['text_case'] for l in lines]),
        "length": len(para_text)
    }


def get_text_case(text):
    if text.isupper():
        return "UPPER"
    elif text.islower():
        return "lower"
    elif text.istitle():
        return "Title"
    return "Mixed"


def get_alignment(spans):
    x_coords = [s['bbox'][0] for s in spans if 'bbox' in s]
    if not x_coords:
        return "unknown"
    x0 = min(x_coords)
    if x0 < 80:
        return "left"
    elif x0 > 300:
        return "right"
    return "center"


def most_common(lst):
    return max(set(lst), key=lst.count) if lst else None
