import fitz  # PyMuPDF
import numpy as np
from collections import defaultdict, Counter
from uuid import uuid4
import logging
import re
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFLayoutExtractor:
    def __init__(self, 
                 vertical_spacing_threshold: float = 15.0,
                 min_font_size: float = 0.01,
                 font_size_tolerance: float = 0.5,
                 alignment_thresholds: Dict[str, float] = None):
        """
        Initialize PDF Layout Extractor with configurable parameters.
        
        Args:
            vertical_spacing_threshold: Maximum spacing to consider lines as same paragraph
            min_font_size: Minimum font size to avoid division by zero
            font_size_tolerance: Maximum difference in font sizes to group together (strict)
            alignment_thresholds: Custom alignment detection thresholds
        """
        self.vertical_spacing_threshold = vertical_spacing_threshold
        self.min_font_size = min_font_size
        self.font_size_tolerance = font_size_tolerance
        self.alignment_thresholds = alignment_thresholds or {
            'left_max': 0.15,    # 15% from left
            'right_min': 0.75,   # 75% from left
            'center_min': 0.25,  # Between 25% and 75%
            'center_max': 0.75
        }
        self.bold_keywords = ['bold', 'heavy', 'black', 'semibold', 'demibold', 'extrabold']

def extract_layout_text(pdf_path: str, extractor: Optional[PDFLayoutExtractor] = None) -> List[Dict[str, Any]]:
    """
    Extract layout text from PDF with improved error handling and strict paragraph grouping.
    
    Args:
        pdf_path: Path to the PDF file
        extractor: Optional PDFLayoutExtractor instance with custom settings
        
    Returns:
        List of paragraph dictionaries with extracted features
    """
    if extractor is None:
        extractor = PDFLayoutExtractor()
    
    # Validate input
    if not pdf_path or not isinstance(pdf_path, str):
        logger.error("Invalid PDF path provided")
        return []
    
    # Try to open document with comprehensive error handling
    try:
        doc = fitz.open(pdf_path)
        if doc.is_encrypted:
            logger.error("PDF is password protected")
            return []
        if doc.page_count == 0:
            logger.warning("PDF has no pages")
            return []
    except Exception as e:
        logger.error(f"Failed to open PDF '{pdf_path}': {e}")
        return []
    
    try:
        return _extract_paragraphs_from_document(doc, extractor)
    except Exception as e:
        logger.error(f"Error during text extraction: {e}")
        return []
    finally:
        # Ensure document is always closed
        try:
            doc.close()
        except:
            pass

def _extract_paragraphs_from_document(doc: fitz.Document, extractor: PDFLayoutExtractor) -> List[Dict[str, Any]]:
    """Extract paragraphs from all pages in the document."""
    paragraphs = []
    all_font_sizes = []
    
    for page_num in range(len(doc)):
        try:
            page = doc[page_num]
            page_paragraphs, page_font_sizes = _extract_page_paragraphs(page, page_num, extractor)
            paragraphs.extend(page_paragraphs)
            all_font_sizes.extend(page_font_sizes)
        except Exception as e:
            logger.warning(f"Error processing page {page_num}: {e}")
            continue
    
    if not all_font_sizes:
        logger.warning("No font sizes found in document")
        return paragraphs
    
    # Calculate relative font size rankings
    font_size_ranks = build_relative_font_ranks(all_font_sizes)
    
    # Add relative font size to paragraphs
    for para in paragraphs:
        fs = para.get("avg_font_size", 0)
        para["relative_font_size"] = font_size_ranks.get(fs, 0)
    
    return paragraphs

def _extract_page_paragraphs(page: fitz.Page, page_num: int, extractor: PDFLayoutExtractor) -> Tuple[List[Dict], List[float]]:
    """Extract paragraphs from a single page."""
    try:
        page_dict = page.get_text("dict")
        if not page_dict or 'blocks' not in page_dict:
            return [], []
    except Exception as e:
        logger.warning(f"Failed to extract text from page {page_num}: {e}")
        return [], []
    
    blocks = page_dict.get('blocks', [])
    if not blocks:
        return [], []
    
    lines_group = []
    page_font_sizes = []
    page_rect = page.rect
    
    for block in blocks:
        if not isinstance(block, dict) or block.get('type') != 0:
            continue
            
        lines = block.get('lines', [])
        for line in lines:
            try:
                line_features, line_font_sizes = _process_line(line, page_num, page_rect, extractor)
                if line_features:
                    lines_group.append(line_features)
                    page_font_sizes.extend(line_font_sizes)
            except Exception as e:
                logger.debug(f"Error processing line on page {page_num}: {e}")
                continue
    
    # Group lines into paragraphs with strict matching
    try:
        paragraphs = group_into_paragraphs_strict(lines_group, extractor)
    except Exception as e:
        logger.warning(f"Error grouping paragraphs on page {page_num}: {e}")
        paragraphs = []
    
    return paragraphs, page_font_sizes

def _process_line(line: Dict, page_num: int, page_rect: fitz.Rect, extractor: PDFLayoutExtractor) -> Tuple[Optional[Dict], List[float]]:
    """Process a single line and extract its features."""
    spans = line.get('spans', [])
    if not spans:
        return None, []
    
    # Filter and clean spans
    valid_spans = []
    for span in spans:
        if not isinstance(span, dict):
            continue
        text = span.get('text', '').strip()
        if text and len(text) > 1:  # Filter out single characters and empty text
            valid_spans.append(span)
    
    if not valid_spans:
        return None, []
    
    # Extract text
    line_text = " ".join([span['text'].strip() for span in valid_spans])
    if not line_text or len(line_text) <= 1:
        return None, []
    
    # Extract font information with error handling
    font_sizes = []
    fonts = []
    
    for span in valid_spans:
        try:
            size = span.get('size', 0)
            if isinstance(size, (int, float)) and size > 0:
                font_sizes.append(float(size))
            
            font = span.get('font', '')
            if isinstance(font, str) and font:
                fonts.append(font)
        except (TypeError, ValueError):
            continue
    
    if not font_sizes:
        return None, []
    
    # Calculate font metrics
    avg_font_size = round(float(np.mean(font_sizes)), 2)
    is_bold = detect_bold_text(fonts)
    
    # Extract bounding box with validation
    bbox = line.get('bbox')
    if not bbox or len(bbox) < 4:
        return None, []
    
    try:
        bbox = [float(x) for x in bbox[:4]]
        y0, y1 = bbox[1], bbox[3]
        height = max(y1 - y0, 0.1)  # Ensure positive height
    except (TypeError, ValueError, IndexError):
        return None, []
    
    # Calculate height to font ratio safely
    height_to_font_ratio = round(height / avg_font_size, 2) if avg_font_size >= extractor.min_font_size else 0.0
    
    # Get alignment and text case
    alignment = get_alignment(valid_spans, page_rect, extractor.alignment_thresholds)
    text_case = get_text_case(line_text)
    
    line_features = {
        "text": line_text,
        "font_size": avg_font_size,
        "normalized_font_size": normalize_font_size(avg_font_size),
        "bbox": bbox,
        "is_bold": is_bold,
        "alignment": alignment,
        "text_case": text_case,
        "font_names": list(set(fonts)),
        "line_height": round(height, 2),
        "height_to_font_ratio": height_to_font_ratio,
        "y0": y0,
        "y1": y1,
        "page_num": page_num
    }
    
    return line_features, font_sizes

def normalize_font_size(font_size: float) -> float:
    """Normalize font sizes to standard classes for consistent grouping."""
    # Round to nearest 0.5 to create consistent font size classes
    return round(font_size * 2) / 2

def detect_bold_text(fonts: List[str]) -> bool:
    """Detect if text is bold using improved font name analysis."""
    if not fonts:
        return False
    
    bold_keywords = ['bold', 'heavy', 'black', 'semibold', 'demibold', 'extrabold']
    
    for font in fonts:
        if not isinstance(font, str):
            continue
        font_lower = font.lower()
        if any(keyword in font_lower for keyword in bold_keywords):
            return True
    
    return False

def build_relative_font_ranks(font_sizes: List[float]) -> Dict[float, int]:
    """Build relative font size rankings with error handling."""
    if not font_sizes:
        return {}
    
    try:
        # Remove duplicates and sort
        unique_sizes = sorted(set(font_sizes))
        if not unique_sizes:
            return {}
        
        # Handle floating point precision issues
        rounded_sizes = [round(fs, 2) for fs in unique_sizes]
        unique_rounded = sorted(set(rounded_sizes))
        
        return {fs: idx for idx, fs in enumerate(unique_rounded)}
    except Exception as e:
        logger.warning(f"Error building font ranks: {e}")
        return {}

def group_into_paragraphs_strict(lines: List[Dict], extractor: PDFLayoutExtractor) -> List[Dict]:
    """Group lines into paragraphs with strict font size matching."""
    if not lines:
        return []
    
    # Sort lines by vertical position
    try:
        lines.sort(key=lambda x: (x.get('page_num', 0), x.get('y0', 0)))
    except Exception as e:
        logger.warning(f"Error sorting lines: {e}")
        return []
    
    paragraphs = []
    current_para = [lines[0]]
    
    for i in range(1, len(lines)):
        try:
            prev = current_para[-1]
            curr = lines[i]
            
            # Calculate spacing (handle negative spacing for overlapping text)
            spacing = max(0, curr.get('y0', 0) - prev.get('y1', 0))
            curr['line_spacing'] = spacing
            
            # STRICT MATCHING CONDITIONS
            same_page = curr.get('page_num') == prev.get('page_num')
            
            # Use normalized font sizes for exact matching
            same_font_class = curr.get('normalized_font_size') == prev.get('normalized_font_size')
            
            # Alternative: strict tolerance-based matching
            # same_font_size = abs(curr.get('font_size', 0) - prev.get('font_size', 0)) <= extractor.font_size_tolerance
            
            same_font_weight = curr.get('is_bold') == prev.get('is_bold')
            same_font_name = bool(set(curr.get('font_names', [])).intersection(set(prev.get('font_names', []))))
            close_vertically = spacing < extractor.vertical_spacing_threshold
            
            # All conditions must match for grouping
            if (same_page and close_vertically and same_font_class and 
                same_font_weight and same_font_name):
                current_para.append(curr)
            else:
                # Finalize current paragraph and start new one
                para = aggregate_paragraph(current_para)
                if para:
                    paragraphs.append(para)
                current_para = [curr]
                
        except Exception as e:
            logger.debug(f"Error processing line {i}: {e}")
            continue
    
    # Don't forget the last paragraph
    if current_para:
        para = aggregate_paragraph(current_para)
        if para:
            paragraphs.append(para)
    
    return paragraphs

def aggregate_paragraph(lines: List[Dict]) -> Optional[Dict]:
    """Aggregate lines into a paragraph with comprehensive error handling."""
    if not lines:
        return None
    
    try:
        # Combine text
        texts = [l.get('text', '') for l in lines if l.get('text')]
        if not texts:
            return None
        
        para_text = " ".join(texts)
        if not para_text.strip():
            return None
        
        # Calculate numeric aggregates safely
        font_sizes = [l.get('font_size', 0) for l in lines if isinstance(l.get('font_size'), (int, float))]
        normalized_font_sizes = [l.get('normalized_font_size', 0) for l in lines if isinstance(l.get('normalized_font_size'), (int, float))]
        line_heights = [l.get('line_height', 0) for l in lines if isinstance(l.get('line_height'), (int, float))]
        ratios = [l.get('height_to_font_ratio', 0) for l in lines if isinstance(l.get('height_to_font_ratio'), (int, float))]
        spacing_vals = [l.get('line_spacing', 0) for l in lines if isinstance(l.get('line_spacing'), (int, float)) and l.get('line_spacing', 0) > 0]
        
        # Calculate averages with fallbacks
        avg_font_size = round(float(np.mean(font_sizes)), 2) if font_sizes else 0.0
        avg_normalized_font_size = round(float(np.mean(normalized_font_sizes)), 2) if normalized_font_sizes else 0.0
        avg_line_height = round(float(np.mean(line_heights)), 2) if line_heights else 0.0
        avg_ratio = round(float(np.mean(ratios)), 2) if ratios else 0.0
        avg_spacing = round(float(np.mean(spacing_vals)), 2) if spacing_vals else 0.0
        
        # Aggregate font information
        font_weights = [l.get('is_bold', False) for l in lines]
        font_names_lists = [l.get('font_names', []) for l in lines]
        all_font_names = [name for names in font_names_lists for name in names if isinstance(name, str)]
        
        # Get most common values
        alignments = [l.get('alignment', 'unknown') for l in lines]
        text_cases = [l.get('text_case', 'Mixed') for l in lines]
        
        most_common_alignment = most_common(alignments)
        most_common_text_case = most_common(text_cases)
        most_common_font_names = most_common([tuple(sorted(names)) for names in font_names_lists if names])
        
        # Calculate bounding box
        bboxes = [l.get('bbox', []) for l in lines if l.get('bbox') and len(l.get('bbox', [])) >= 4]
        if bboxes:
            try:
                bbox = [
                    min(bbox[0] for bbox in bboxes),
                    min(bbox[1] for bbox in bboxes),
                    max(bbox[2] for bbox in bboxes),
                    max(bbox[3] for bbox in bboxes)
                ]
            except (TypeError, ValueError, IndexError):
                bbox = [0, 0, 0, 0]
        else:
            bbox = [0, 0, 0, 0]
        
        # Calculate bold ratio
        bold_ratio = round(sum(font_weights) / len(font_weights), 2) if font_weights else 0.0
        
        # Check for consistent formatting within paragraph
        font_size_variance = round(float(np.var(font_sizes)), 2) if len(font_sizes) > 1 else 0.0
        
        return {
            "paragraph_id": f"p_{uuid4().hex[:8]}",
            "page_num": lines[0].get("page_num", -1),
            "text": para_text,
            "avg_font_size": avg_font_size,
            "normalized_font_size": avg_normalized_font_size,
            "font_size_variance": font_size_variance,
            "avg_line_height": avg_line_height,
            "avg_height_to_font_ratio": avg_ratio,
            "line_spacing_avg": avg_spacing,
            "bbox": bbox,
            "is_bold": all(font_weights) and len(font_weights) > 0,
            "bold_ratio": bold_ratio,
            "is_upper": para_text.isupper(),
            "alignment": most_common_alignment or "unknown",
            "line_count": len(lines),
            "font_names": list(set(all_font_names)),
            "primary_font_family": list(most_common_font_names) if most_common_font_names else [],
            "text_case": most_common_text_case or "Mixed",
            "length": len(para_text),
            "is_homogeneous": font_size_variance < 0.1  # Flag for consistent formatting
        }
        
    except Exception as e:
        logger.warning(f"Error aggregating paragraph: {e}")
        return None

def get_text_case(text: str) -> str:
    """Determine text case with error handling."""
    if not isinstance(text, str) or not text:
        return "Mixed"
    
    try:
        if text.isupper():
            return "UPPER"
        elif text.islower():
            return "lower"
        elif text.istitle():
            return "Title"
        return "Mixed"
    except Exception:
        return "Mixed"

def get_alignment(spans: List[Dict], page_rect: fitz.Rect, thresholds: Dict[str, float]) -> str:
    """Determine text alignment based on position relative to page width."""
    if not spans or not page_rect:
        return "unknown"
    
    try:
        x_coords = []
        for span in spans:
            bbox = span.get('bbox')
            if bbox and len(bbox) >= 4:
                try:
                    x_coords.append(float(bbox[0]))
                except (TypeError, ValueError):
                    continue
        
        if not x_coords:
            return "unknown"
        
        x0 = min(x_coords)
        page_width = page_rect.width
        
        if page_width <= 0:
            return "unknown"
        
        # Calculate relative position (0.0 = left edge, 1.0 = right edge)
        relative_pos = x0 / page_width
        
        if relative_pos <= thresholds['left_max']:
            return "left"
        elif relative_pos >= thresholds['right_min']:
            return "right"
        elif thresholds['center_min'] <= relative_pos <= thresholds['center_max']:
            return "center"
        else:
            return "unknown"
            
    except Exception as e:
        logger.debug(f"Error determining alignment: {e}")
        return "unknown"

def most_common(lst: List) -> Any:
    """Find most common element in list with error handling."""
    if not lst:
        return None
    
    try:
        # Filter out None values
        filtered_lst = [x for x in lst if x is not None]
        if not filtered_lst:
            return None
        
        counter = Counter(filtered_lst)
        return counter.most_common(1)[0][0]
    except Exception:
        return None

# Convenience function for backward compatibility
def group_into_paragraphs(lines: List[Dict], extractor: PDFLayoutExtractor) -> List[Dict]:
    """Wrapper function that calls the strict grouping method."""
    return group_into_paragraphs_strict(lines, extractor)

# Example usage and testing
def main():
    """Example usage of the improved PDF layout extractor with strict paragraph grouping."""
    # Create custom extractor with specific settings
    extractor = PDFLayoutExtractor(
        vertical_spacing_threshold=12.0,  # Tighter paragraph grouping
        min_font_size=0.1,  # Higher minimum font size
        font_size_tolerance=0.5,  # Very strict font size matching
        alignment_thresholds={
            'left_max': 0.2,
            'right_min': 0.7,
            'center_min': 0.3,
            'center_max': 0.7
        }
    )
    
    # Extract paragraphs
    paragraphs = extract_layout_text("example.pdf", extractor)
    
    # Process results
    for para in paragraphs:
        print(f"Page {para['page_num']}: {para['text'][:100]}...")
        print(f"Font size: {para['avg_font_size']} (normalized: {para['normalized_font_size']})")
        print(f"Bold: {para['is_bold']}, Homogeneous: {para['is_homogeneous']}")
        print(f"Relative font rank: {para.get('relative_font_size', 'N/A')}")
        print("---")

def analyze_document_structure(paragraphs: List[Dict]) -> Dict[str, Any]:
    """Analyze document structure based on extracted paragraphs."""
    if not paragraphs:
        return {}
    
    font_sizes = [p['avg_font_size'] for p in paragraphs]
    relative_ranks = [p.get('relative_font_size', 0) for p in paragraphs]
    
    structure_analysis = {
        'total_paragraphs': len(paragraphs),
        'unique_font_sizes': len(set(font_sizes)),
        'font_size_range': (min(font_sizes), max(font_sizes)),
        'relative_rank_range': (min(relative_ranks), max(relative_ranks)),
        'homogeneous_paragraphs': sum(1 for p in paragraphs if p.get('is_homogeneous', False)),
        'bold_paragraphs': sum(1 for p in paragraphs if p.get('is_bold', False)),
        'pages_covered': len(set(p['page_num'] for p in paragraphs))
    }
    
    return structure_analysis
"""
if __name__ == "__main__":
    main()
"""