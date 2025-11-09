import json
import os
import re
import ssl
import sys
from typing import List, Tuple
import cv2
import numpy as np
try:
    import easyocr
except ImportError:
    easyocr = None
try:
    import pytesseract
except ImportError:
    pytesseract = None
if easyocr:
    print('✓ EasyOCR available for text detection')
elif pytesseract:
    print('✓ Tesseract available for text detection')
else:
    print('⚠ No OCR library available. Install easyocr or pytesseract for text detection.')
OCR_AVAILABLE = bool(easyocr or pytesseract)

class VisionMixin:

    def initialize_ocr(self) -> None:
        try:
            if easyocr and 'easyocr' in sys.modules:
                print('Initializing EasyOCR...')
                try:
                    ssl._create_default_https_context = ssl._create_unverified_context
                    self.ocr_reader = easyocr.Reader(['en'], gpu=False, download_enabled=True)
                    print('✓ EasyOCR initialized successfully!')
                    return
                except Exception as exc:
                    print(f'EasyOCR failed to initialize: {exc}')
                    print('Falling back to basic text detection...')
                    self.use_fallback = True
                    self.ocr_reader = 'fallback'
                    return
            if pytesseract and 'pytesseract' in sys.modules:
                print('✓ Tesseract OCR available')
                self.ocr_reader = 'tesseract'
        except Exception as exc:
            print(f'Error initializing OCR: {exc}')
            print('Using fallback text detection method...')
            self.use_fallback = True
            self.ocr_reader = 'fallback'

    def initialize_google_vision(self) -> None:
        try:
            from google.cloud import vision
            creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '').strip()
            if not creds_path or not os.path.isfile(creds_path):
                try:
                    for fname in os.listdir('.'):
                        if not fname.lower().endswith('.json'):
                            continue
                        if not os.path.isfile(fname):
                            continue
                        try:
                            with open(fname, 'r', encoding='utf-8') as handle:
                                data = json.load(handle)
                        except Exception:
                            continue
                        if isinstance(data, dict) and data.get('type') == 'service_account':
                            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.abspath(fname)
                            print(f'✓ Using Google Vision credentials: {fname}')
                            break
                except Exception:
                    pass
            self.vision_client = vision.ImageAnnotatorClient()
            print('✓ Google Cloud Vision initialized')
        except Exception as exc:
            self.vision_client = None
            creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '(not set)')
            print(f'Google Vision init failed: {exc}\n  GOOGLE_APPLICATION_CREDENTIALS={creds_path}\nUsing EasyOCR fallback.')

    def google_vision_ocr(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        if self.vision_client is None:
            return (frame_bgr, [])
        try:
            from google.cloud import vision
        except Exception:
            return (frame_bgr, [])
        try:
            processed = self.preprocess_for_vision(frame_bgr)
            success, buffer = cv2.imencode('.png', processed)
            if not success:
                return (frame_bgr, [])
            image = vision.Image(content=buffer.tobytes())
            features = [vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION, model='builtin/latest')]
            params = vision.TextDetectionParams(enable_text_detection_confidence_score=True)
            context = vision.ImageContext(language_hints=['en'], text_detection_params=params)
            request = vision.AnnotateImageRequest(image=image, features=features, image_context=context)
            batch_response = self.vision_client.batch_annotate_images(requests=[request])
            response = batch_response.responses[0]
            if response.error.message:
                print(f'Google Vision error: {response.error.message}')
                return (frame_bgr, [])
            height, width = frame_bgr.shape[:2]
            if not response.full_text_annotation or not response.full_text_annotation.pages:
                return (frame_bgr, [])
            regions: List[dict] = []

            def bbox(vertices):
                xs = [v.x for v in vertices]
                ys = [v.y for v in vertices]
                x0, y0 = (max(0, min(xs)), max(0, min(ys)))
                x1, y1 = (min(width, max(xs)), min(height, max(ys)))
                return (int(x0), int(y0), int(max(0, x1 - x0)), int(max(0, y1 - y0)))
            full_text_parts: List[str] = []
            for page in response.full_text_annotation.pages:
                for block in page.blocks:
                    for para in block.paragraphs:
                        if self.granularity == 'paragraph':
                            xs = [v.x for v in para.bounding_box.vertices]
                            ys = [v.y for v in para.bounding_box.vertices]
                            x0, y0 = (max(0, min(xs)), max(0, min(ys)))
                            x1, y1 = (min(width, max(xs)), min(height, max(ys)))
                            w_val = int(max(0, x1 - x0))
                            h_val = int(max(0, y1 - y0))
                            words_txt: List[str] = []
                            for word in para.words:
                                token = ''.join((sym.text for sym in word.symbols))
                                if token:
                                    words_txt.append(token)
                            text_para = self.normalize_math_text(' '.join(words_txt))
                            if w_val > 0 and h_val > 0 and text_para:
                                pad_x = int(max(4, w_val * 0.06))
                                pad_y = int(max(4, h_val * 0.06))
                                x_pos = max(0, x0 - pad_x)
                                y_pos = max(0, y0 - pad_y)
                                w_box = min(width - x_pos, w_val + 2 * pad_x)
                                h_box = min(height - y_pos, h_val + 2 * pad_y)
                                if self.is_valid_text_region(w_box, h_box) and self.is_in_center_region(x_pos, y_pos, w_box, h_box, width, height):
                                    regions.append({'x': x_pos, 'y': y_pos, 'w': w_box, 'h': h_box, 'score': 0.94, 'label': text_para})
                        if self.granularity == 'word':
                            word_regions: List[dict] = []
                            for word in para.words:
                                token = ''.join((sym.text for sym in word.symbols))
                                if not token:
                                    continue
                                normalized = self.normalize_math_text(token)
                                x_val, y_val, w_val, h_val = bbox(word.bounding_box.vertices)
                                if w_val <= 0 or h_val <= 0 or (not normalized):
                                    continue
                                pad_x = int(max(1, w_val * 0.05))
                                pad_y = int(max(1, h_val * 0.06))
                                x_pos = max(0, x_val - pad_x)
                                y_pos = max(0, y_val - pad_y)
                                w_box = min(width - x_pos, w_val + 2 * pad_x)
                                h_box = min(height - y_pos, h_val + 2 * pad_y)
                                if self.is_valid_word_region(w_box, h_box) and self.is_in_center_region(x_pos, y_pos, w_box, h_box, width, height):
                                    word_regions.append({'x': x_pos, 'y': y_pos, 'w': w_box, 'h': h_box, 'score': 0.9, 'label': normalized})
                            word_regions = self.merge_subscript_word_boxes(word_regions)
                            regions.extend(word_regions)
                        current_line_text: List[str] = []
                        current_line_vertices = []

                        def flush_line():
                            if not current_line_text or not current_line_vertices:
                                return
                            txt = ''.join(current_line_text).strip()
                            norm_txt = self.normalize_math_text(txt)
                            if norm_txt:
                                xs = [v.x for v in current_line_vertices]
                                ys = [v.y for v in current_line_vertices]
                                x0, y0 = (max(0, min(xs)), max(0, min(ys)))
                                x1, y1 = (min(width, max(xs)), min(height, max(ys)))
                                w_val = int(max(0, x1 - x0))
                                h_val = int(max(0, y1 - y0))
                                pad_x = int(max(4, w_val * 0.07))
                                pad_y = int(max(4, h_val * 0.09))
                                x_pos = max(0, int(x0 - pad_x))
                                y_pos = max(0, int(y0 - pad_y))
                                w_box = min(width - x_pos, int(w_val + 2 * pad_x))
                                h_box = min(height - y_pos, int(h_val + 2 * pad_y))
                                if self.is_in_center_region(x_pos, y_pos, w_box, h_box, width, height):
                                    full_text_parts.append(norm_txt)
                                if self.granularity == 'line' and self.is_valid_text_region(w_box, h_box):
                                    regions.append({'x': x_pos, 'y': y_pos, 'w': w_box, 'h': h_box, 'score': 0.95, 'label': norm_txt})
                            current_line_text.clear()
                            current_line_vertices.clear()
                        for word in para.words:
                            for sym in word.symbols:
                                token = sym.text or ''
                                current_line_text.append(token)
                                current_line_vertices.extend(sym.bounding_box.vertices)
                                br = getattr(sym, 'property', None)
                                brk = getattr(br, 'detected_break', None)
                                brk_type = getattr(brk, 'type_', None)
                                if brk_type in (vision.TextAnnotation.DetectedBreak.BreakType.EOL_SURE_SPACE, vision.TextAnnotation.DetectedBreak.BreakType.LINE_BREAK):
                                    flush_line()
                                elif brk_type in (vision.TextAnnotation.DetectedBreak.BreakType.SPACE, vision.TextAnnotation.DetectedBreak.BreakType.SURE_SPACE):
                                    current_line_text.append(' ')
                            current_line_text.append(' ')
                        flush_line()
            self.last_full_text = '\n'.join([p for p in full_text_parts if p])
            for region in regions:
                cv2.rectangle(frame_bgr, (region['x'], region['y']), (region['x'] + region['w'], region['y'] + region['h']), (0, 128, 255), 2)
            return (frame_bgr, regions)
        except Exception as exc:
            print(f'Error in Google Vision OCR: {exc}')
            return (frame_bgr, [])

    def detect_text_fallback(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        try:
            frame_h, frame_w = frame.shape[:2]
            scale = 0.5 if max(frame_w, frame_h) > 700 else 1.0
            if scale != 1.0:
                small = cv2.resize(frame, (int(frame_w * scale), int(frame_h * scale)))
            else:
                small = frame
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
            grad = cv2.convertScaleAbs(cv2.absdiff(grad_x, grad_y))
            _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
            closed = cv2.dilate(closed, kernel, iterations=1)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            regions = []
            for contour in contours:
                x_pos, y_pos, w_box, h_box = cv2.boundingRect(contour)
                if scale != 1.0:
                    x_pos = int(x_pos / scale)
                    y_pos = int(y_pos / scale)
                    w_box = int(w_box / scale)
                    h_box = int(h_box / scale)
                if self.is_valid_text_region(w_box, h_box) and self.is_in_center_region(x_pos, y_pos, w_box, h_box, frame_w, frame_h):
                    regions.append((x_pos, y_pos, w_box, h_box))
            regions = self.remove_overlapping_regions(regions, overlap_threshold=0.5)
            self.last_text_regions = [{'x': x_pos, 'y': y_pos, 'w': w_box, 'h': h_box, 'score': float(w_box * h_box), 'label': ''} for x_pos, y_pos, w_box, h_box in regions]
            for x_pos, y_pos, w_box, h_box in regions:
                cv2.rectangle(frame, (x_pos, y_pos), (x_pos + w_box, y_pos + h_box), (255, 0, 0), 2)
            return (frame, len(regions))
        except Exception as exc:
            print(f'Error in fallback detection: {exc}')
            return (frame, 0)

    def detect_text_easyocr(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        if not easyocr or not isinstance(self.ocr_reader, easyocr.Reader):
            return (frame, 0)
        result = frame.copy()
        detections = self.ocr_reader.readtext(frame)
        regions: List[dict] = []
        for box, text, confidence in detections:
            text = self.normalize_math_text(text)
            if not self.is_valid_text_content(text):
                continue
            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            x_min, x_max = (int(min(xs)), int(max(xs)))
            y_min, y_max = (int(min(ys)), int(max(ys)))
            w_box = x_max - x_min
            h_box = y_max - y_min
            if not self.is_valid_text_region(w_box, h_box) or not self.is_in_center_region(x_min, y_min, w_box, h_box, frame.shape[1], frame.shape[0]):
                continue
            regions.append({'x': x_min, 'y': y_min, 'w': w_box, 'h': h_box, 'score': float(confidence), 'label': text})
        regions = self.remove_overlapping_text_regions(regions, overlap_threshold=0.5)
        self.last_text_regions = regions
        self._draw_regions(result, regions)
        return (result, len(regions))

    def detect_text_tesseract(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        if not pytesseract or self.ocr_reader != 'tesseract':
            return (frame, 0)
        result = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 5)
        config = '--oem 3 --psm 6'
        data = pytesseract.image_to_data(gray, config=config, output_type=pytesseract.Output.DICT)
        regions: List[dict] = []
        for i in range(len(data['text'])):
            text = data['text'][i]
            conf = float(data['conf'][i]) if data['conf'][i] != '-1' else 0.0
            text = self.normalize_math_text(text)
            if not text or conf < 40:
                continue
            x_pos, y_pos, w_box, h_box = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            if not self.is_valid_text_region(w_box, h_box) or not self.is_in_center_region(x_pos, y_pos, w_box, h_box, frame.shape[1], frame.shape[0]):
                continue
            regions.append({'x': x_pos, 'y': y_pos, 'w': w_box, 'h': h_box, 'score': conf / 100.0, 'label': text})
        regions = self.remove_overlapping_text_regions(regions, overlap_threshold=0.5)
        self.last_text_regions = regions
        self._draw_regions(result, regions)
        return (result, len(regions))

    def detect_text_accurate(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        if not easyocr or not isinstance(self.ocr_reader, easyocr.Reader):
            return self.detect_text_easyocr(frame)
        result = frame.copy()
        detections = self.ocr_reader.readtext(frame, detail=1, paragraph=False)
        word_candidates: List[dict] = []
        for box, text, confidence in detections:
            normalized = self.normalize_math_text(text)
            if not normalized:
                continue
            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            x_min, x_max = (int(min(xs)), int(max(xs)))
            y_min, y_max = (int(min(ys)), int(max(ys)))
            w_box = x_max - x_min
            h_box = y_max - y_min
            if not self.is_valid_word_region(w_box, h_box):
                continue
            word_candidates.append({'x': x_min, 'y': y_min, 'w': w_box, 'h': h_box, 'score': float(confidence), 'label': normalized})
        regions = self.merge_subscript_word_boxes(word_candidates)
        regions = [r for r in regions if self.is_valid_word_region(r['w'], r['h']) and self.is_in_center_region(r['x'], r['y'], r['w'], r['h'], frame.shape[1], frame.shape[0])]
        regions = self.remove_overlapping_text_regions(regions, overlap_threshold=0.45)
        self.last_text_regions = regions
        self._draw_regions(result, regions)
        return (result, len(regions))

    def preprocess_for_vision(self, frame_bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a_channel, b_channel))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)
        return sharpened

    def preprocess_crop_for_ocr(self, crop_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 3)
        return gray

    def normalize_math_text(self, text: str) -> str:
        if not text:
            return ''
        text = text.strip()
        replacements = {'×': '*', '÷': '/', '−': '-', '–': '-', '—': '-', '•': '.', '·': '.', '°': 'deg', '“': '"', '”': '"', '‘': "'", '’': "'"}
        for src, dst in replacements.items():
            text = text.replace(src, dst)
        text = re.sub('\\s+', ' ', text)
        return text.strip()

    def merge_subscript_word_boxes(self, boxes: List[dict]) -> List[dict]:
        if not boxes:
            return boxes
        boxes = sorted(boxes, key=lambda b: (b['x'], b['y']))
        merged: List[dict] = []
        i = 0
        while i < len(boxes):
            current = boxes[i]
            j = i + 1
            while j < len(boxes):
                candidate = boxes[j]
                if candidate['x'] > current['x'] + current['w'] * 1.2:
                    break
                same_base = abs(candidate['y'] - (current['y'] + current['h'])) < max(6, current['h'] * 0.4)
                close_horizontal = abs(candidate['x'] - (current['x'] + current['w'])) < max(8, current['w'] * 0.45)
                is_subscript = same_base and close_horizontal and (len(candidate['label']) <= 2)
                if is_subscript:
                    new_x = min(current['x'], candidate['x'])
                    new_y = min(current['y'], candidate['y'])
                    new_w = max(current['x'] + current['w'], candidate['x'] + candidate['w']) - new_x
                    new_h = max(current['y'] + current['h'], candidate['y'] + candidate['h']) - new_y
                    current = {'x': new_x, 'y': new_y, 'w': new_w, 'h': new_h, 'score': max(current['score'], candidate['score']), 'label': f'{current['label']}{candidate['label']}'}
                    j += 1
                else:
                    break
            merged.append(current)
            i = j
        return merged

    def ocr_extract_text(self, crop_bgr: np.ndarray) -> str:
        try:
            if easyocr and isinstance(self.ocr_reader, easyocr.Reader):
                result = self.ocr_reader.readtext(crop_bgr, detail=0, paragraph=True)
                text = ' '.join(result)
                text = self.normalize_math_text(text)
                if self.is_valid_text_content(text):
                    return text
            if pytesseract and self.ocr_reader == 'tesseract':
                gray = self.preprocess_crop_for_ocr(crop_bgr)
                raw_text = pytesseract.image_to_string(gray, config='--oem 3 --psm 6')
                text = self.normalize_math_text(raw_text)
                if self.is_valid_text_content(text):
                    return text
        except Exception as exc:
            print(f'OCR extraction failed: {exc}')
        return ''

    def detect_text(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        if not self.detection_enabled:
            return (frame, 0)
        if self.ocr_reader is None:
            return (frame, 0)
        if self.vision_client is not None:
            annotated, regions = self.google_vision_ocr(frame.copy())
            if regions:
                self.last_text_regions = regions
                return (annotated, len(regions))
        if easyocr and isinstance(self.ocr_reader, easyocr.Reader):
            if self.accurate_mode:
                return self.detect_text_accurate(frame)
            return self.detect_text_easyocr(frame)
        if self.ocr_reader == 'tesseract':
            return self.detect_text_tesseract(frame)
        return self.detect_text_fallback(frame)
