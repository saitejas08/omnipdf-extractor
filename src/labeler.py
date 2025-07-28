import json
import os
import numpy as np
from collections import Counter
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentLabeler:
    def __init__(self):
        self.processed_count = 0
        self.failed_count = 0

    def label_document_hierarchy(self, json_data):
        if not json_data:
            return json_data

        all_objects = [obj for obj in json_data if isinstance(obj, dict)]
        if not all_objects:
            return json_data

        page_numbers = [obj.get('page_num', 1) for obj in all_objects if obj.get('page_num') is not None]
        lowest_page_num = min(page_numbers) if page_numbers else 1
        print(f"📄 Lowest page number found: {lowest_page_num}")
        first_page_objects = [obj for obj in all_objects if obj.get('page_num', 1) == lowest_page_num]

        # 1. Title logic
        title_candidate = None
        title_max_font_size = None
        if first_page_objects:
            title_candidates = [
                obj for obj in first_page_objects
                if (obj.get('relative_font_size', 0) > 0 and
                    len(obj.get('text', '').strip()) > 5 and
                    len(obj.get('text', '').strip()) < 200 and
                    not obj.get('text', '').strip().isdigit())
            ]
            if title_candidates:
                title_max_font_size = max(obj.get('relative_font_size', 0) for obj in title_candidates)
                max_font_objs = [obj for obj in title_candidates if obj.get('relative_font_size', 0) == title_max_font_size]
                def title_feats(obj):
                    return (
                        obj.get('relative_font_size', 0),
                        tuple(obj.get('primary_font_family') or []),
                        obj.get('is_bold', False),
                    )
                clusters = {}
                for obj in max_font_objs:
                    k = title_feats(obj)
                    clusters.setdefault(k, []).append(obj)
                largest_cluster = max(clusters.values(), key=len)
                merged_text = " ".join(obj.get('text', '').strip() for obj in largest_cluster if obj.get('text', '').strip())
                merged_title_obj = largest_cluster[0].copy()
                merged_title_obj['text'] = merged_text
                if len(largest_cluster) > 1:
                    bboxes = [obj.get('bbox', [0, 0, 0, 0]) for obj in largest_cluster]
                    merged_title_obj['bbox'] = [
                        float(np.mean([b[0] for b in bboxes])),
                        float(np.mean([b[1] for b in bboxes])),
                        float(np.mean([b[2] for b in bboxes])),
                        float(np.mean([b[3] for b in bboxes]))
                    ]
                title_candidate = merged_title_obj
        if title_candidate:
            title_text = title_candidate.get('text', '')
            short_title_text = title_text[:50] + "..." if len(title_text) > 50 else title_text
            print(f"🏷️ Title detected on page {lowest_page_num}: \"{short_title_text}\"")

        # 2. Thresholds
        font_sizes = [
            obj.get('relative_font_size', 0)
            for obj in all_objects if obj.get('relative_font_size', 0) > 0
            and (title_candidate is None or obj != title_candidate)
        ]
        if not font_sizes:
            for obj in json_data:
                if isinstance(obj, dict):
                    if title_candidate and obj == title_candidate:
                        obj['level'] = 'title'
                    else:
                        obj['level'] = 'p'
            return json_data
        font_sizes_np = np.array(font_sizes)
        if len(font_sizes_np) >= 4:
            h1_threshold = np.percentile(font_sizes_np, 90)
            h2_threshold = np.percentile(font_sizes_np, 75)
            h3_threshold = np.percentile(font_sizes_np, 60)
        else:
            unique_font_sizes = sorted(set(font_sizes), reverse=True)
            if len(unique_font_sizes) >= 3:
                h1_threshold, h2_threshold, h3_threshold = unique_font_sizes[:3]
            elif len(unique_font_sizes) == 2:
                h1_threshold, h2_threshold = unique_font_sizes
                h3_threshold = 0
            elif len(unique_font_sizes) == 1:
                h1_threshold = unique_font_sizes[0]
                h2_threshold = 0
                h3_threshold = 0
            else:
                h1_threshold = h2_threshold = h3_threshold = 0
        print(f"📊 Font size thresholds - H1: {h1_threshold}, H2: {h2_threshold}, H3: {h3_threshold}")

        def calculate_enhanced_features(obj):
            text = obj.get('text', '').strip()
            font_size = obj.get('relative_font_size', 0)
            is_bold = obj.get('is_bold', False)
            bold_ratio = obj.get('bold_ratio', 0)
            avg_line_height = obj.get('avg_line_height', 0)
            text_case = obj.get('text_case', '')
            alignment = obj.get('alignment', '').lower()
            bbox = obj.get('bbox', [0, 0, 0, 0])
            font_family = ''
            primary_font_family = obj.get("primary_font_family", [])
            if isinstance(primary_font_family, list) and primary_font_family:
                font_family = str(primary_font_family[0]) or ""
            elif isinstance(primary_font_family, str) and primary_font_family:
                font_family = primary_font_family
            else:
                font_names = obj.get("font_names", [])
                if isinstance(font_names, list) and font_names:
                    font_family = str(font_names[0]) or ""
                elif isinstance(font_names, str) and font_names:
                    font_family = font_names
            font_family = font_family.lower()
            features = {
                'font_size': font_size,
                'is_bold': is_bold,
                'bold_ratio': bold_ratio,
                'avg_line_height': avg_line_height,
                'has_bold_family': any(k in font_family for k in ['bold', 'heading', 'heavy', 'black']),
                'is_uppercase': text_case == 'UPPER',
                'is_title_case': text_case == 'Title',
                'text_length': len(text),
                'alignment': alignment,
                'bbox': bbox,
                'font_family': font_family
            }
            return features

        def calculate_header_score(obj, h1_threshold, h2_threshold, h3_threshold):
            features = calculate_enhanced_features(obj)
            score = 0
            max_threshold = max(h1_threshold, h2_threshold, h3_threshold, 1)
            font_size_ratio = features['font_size'] / max_threshold
            score += font_size_ratio * 30
            if features['is_bold']:
                score += 25
            score += features['bold_ratio'] * 20
            if features['has_bold_family']:
                score += 15
            elif 'arial' in features['font_family'] or 'helvetica' in features['font_family']:
                score += 5
            if features['avg_line_height'] > 0:
                score += min(features['avg_line_height'] / 20, 1) * 10
            if features['is_uppercase']:
                score += 10
            elif features['is_title_case']:
                score += 8
            if features['alignment'] == 'center':
                score += 10
            elif features['alignment'] == 'left':
                score += 7
            bbox = features['bbox']
            y_top = bbox[1] if bbox and len(bbox) > 1 else 0
            if y_top < 150:
                score += 7
            elif y_top < 400:
                score += 5
            return score

        initial_labels = [None] * len(json_data)
        def assign_enhanced_level(idx, obj):
            if not isinstance(obj, dict):
                return 'p'
            if title_candidate:
                if obj == title_candidate:
                    return 'title'
                if obj.get('page_num', 1) == lowest_page_num:
                    if obj.get('relative_font_size', 0) < title_max_font_size:
                        return 'p'
            features = calculate_enhanced_features(obj)
            font_size = features['font_size']
            text = obj.get('text', '')
            if features['text_length'] <= 3 and (text.isdigit() or not any(c.isalpha() for c in text)):
                return 'p'
            if features['text_length'] > 350:
                return 'p'
            header_score = calculate_header_score(obj, h1_threshold, h2_threshold, h3_threshold)
            if header_score >= 65:
                if font_size >= h1_threshold and h1_threshold > 0:
                    return 'H1'
                elif font_size >= h2_threshold and h2_threshold > 0:
                    return 'H2'
                elif font_size >= h3_threshold and h3_threshold > 0:
                    return 'H3'
                elif header_score >= 75:
                    return 'H3'
                else:
                    return 'p'
            elif 45 <= header_score < 65:
                if font_size >= h1_threshold and h1_threshold > 0:
                    return 'H1'
                elif font_size >= h2_threshold and h2_threshold > 0:
                    return 'H2'
                elif font_size >= h3_threshold and h3_threshold > 0:
                    return 'H3'
                elif header_score >= 55 and features['is_bold']:
                    return 'H3'
                else:
                    return 'p'
            elif 25 <= header_score < 45:
                if font_size >= h1_threshold and h1_threshold > 0 and features['is_bold']:
                    return 'H1'
                elif font_size >= h2_threshold and h2_threshold > 0 and features['is_bold']:
                    return 'H2'
                elif font_size >= h3_threshold and h3_threshold > 0 and features['is_bold']:
                    return 'H3'
                else:
                    return 'p'
            else:
                return 'p'

        labeled_data = []
        for idx, obj in enumerate(json_data):
            if isinstance(obj, dict):
                label = assign_enhanced_level(idx, obj)
                initial_labels[idx] = label
                labeled_obj = obj.copy()
                labeled_obj['level'] = label
                labeled_data.append(labeled_obj)
            else:
                labeled_data.append(obj)

        for idx, (obj, label) in enumerate(zip(labeled_data, initial_labels)):
            if not isinstance(obj, dict) or label in ['H1', 'H2', 'H3', 'title']:
                continue
            features = calculate_enhanced_features(obj)
            next_idx = idx + 1
            if next_idx < len(labeled_data):
                next_obj = labeled_data[next_idx]
                next_label = initial_labels[next_idx]
                if (isinstance(next_obj, dict) and
                    next_label == "p" and
                    features['is_bold'] and features['font_size'] >= h3_threshold and features['font_size'] > 0):
                    labeled_data[idx]['level'] = 'H3'

        labeled_data = self._ensure_logical_hierarchy(labeled_data)
        return labeled_data

    def _ensure_logical_hierarchy(self, labeled_data):
        seen_title = False
        seen_h1 = False
        seen_h2 = False
        for obj in labeled_data:
            if not isinstance(obj, dict):
                continue
            current_label = obj.get('level', 'p')
            if current_label == 'title':
                seen_title = True
            elif current_label == 'H1':
                seen_h1 = True
            elif current_label == 'H2':
                if not seen_h1:
                    obj['level'] = 'H1'
                    seen_h1 = True
                else:
                    seen_h2 = True
            elif current_label == 'H3':
                if not seen_h2:
                    if seen_h1:
                        obj['level'] = 'H2'
                        seen_h2 = True
                    else:
                        obj['level'] = 'H1'
                        seen_h1 = True
        return labeled_data

    def transform_to_schema(self, labeled_data):
        title = ""
        title_obj = next((obj for obj in labeled_data if isinstance(obj, dict) and obj.get('level') == 'title'), None)
        if title_obj:
            title = title_obj.get('text', '').strip()
        outline = []
        for obj in labeled_data:
            if not isinstance(obj, dict):
                continue
            level = obj.get('level', '')
            text = obj.get('text', '').strip()
            page_num = obj.get('page_num', 1)
            if level in ['H1', 'H2', 'H3'] and text:
                outline.append({
                    "level": level,
                    "text": text,
                    "page": int(page_num) if page_num is not None else 1
                })
        result = {
            "title": title,
            "outline": outline
        }
        return result

    def process_single_file(self, json_file_path):
        """Process a single JSON file and overwrite in-place in outputs folder"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as file:
                json_data = json.load(file)
            labeled_data = self.label_document_hierarchy(json_data)
            schema_data = self.transform_to_schema(labeled_data)
            # Overwrite the SAME file in-place
            with open(json_file_path, 'w', encoding='utf-8') as file:
                json.dump(schema_data, file, indent=2, ensure_ascii=False)
            title_found = "✓" if schema_data.get('title') else "✗"
            outline_count = len(schema_data.get('outline', []))
            if schema_data.get('outline'):
                level_counts = Counter(item['level'] for item in schema_data['outline'])
                logger.info(f" Title: {title_found}, Outline items: {outline_count}")
                logger.info(f" Levels: {dict(level_counts)}")
            else:
                logger.info(f" Title: {title_found}, Outline items: {outline_count}")
            self.processed_count += 1
            return True
        except Exception as e:
            logger.error(f"❌ Failed to process {json_file_path}: {str(e)}")
            self.failed_count += 1
            return False

    def process_output_folder(self, output_folder_path, file_pattern="*.json"):
        """Process all JSON files in the output folder and overwrite originals"""
        output_path = Path(output_folder_path)
        if not output_path.exists():
            logger.error(f"Output folder does not exist: {output_folder_path}")
            return False
        json_files = list(output_path.glob(file_pattern))
        if not json_files:
            logger.warning(f"No JSON files found in {output_folder_path}")
            return True
        logger.info(f"Found {len(json_files)} JSON files to process")
        self.processed_count = 0
        self.failed_count = 0
        for json_file in json_files:
            logger.info(f"Processing: {json_file.name}")
            success = self.process_single_file(json_file)
            if success:
                logger.info(f"✅ Successfully processed: {json_file.name}")
            else:
                logger.error(f"❌ Failed to process: {json_file.name}")
        total_files = len(json_files)
        logger.info(f"\n=== PROCESSING SUMMARY ===")
        logger.info(f"Total files: {total_files}")
        logger.info(f"Successfully processed: {self.processed_count}")
        logger.info(f"Failed: {self.failed_count}")
        logger.info(f"Success rate: {(self.processed_count/total_files)*100:.1f}%")
        logger.info(f"Files overwritten in-place in: {output_path}")
        return self.failed_count == 0
