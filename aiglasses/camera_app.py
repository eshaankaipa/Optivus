import os
import sys
import time
import queue
import threading
from datetime import datetime
from typing import List, Optional
import cv2
import numpy as np
from .vision_pipeline import VisionMixin, OCR_AVAILABLE, easyocr
from .autosolve_pipeline import AutosolveMixin

class TextDetectionCamera(VisionMixin, AutosolveMixin):

    def __init__(self):
        self.camera = None
        self.captured_images = []
        self.output_dir = 'captured_images'
        self.running = True
        self.ocr_reader = None
        self.detection_enabled = True
        self.use_fallback = False
        self.fast_mode = False
        self.accurate_mode = True
        self.last_text_regions: List[dict] = []
        self._det_queue: 'queue.Queue[np.ndarray]' = queue.Queue(maxsize=1)
        self._det_thread: Optional[threading.Thread] = None
        self._det_stop = threading.Event()
        self._det_last_time: float = 0.0
        self._det_interval_sec: float = 0.15
        self.vision_client = None
        self.granularity = 'word'
        self.center_fraction: float = 0.62
        self._hold_signature: Optional[str] = None
        self._hold_start_time: float = 0.0
        self._autosolve_in_progress: bool = False
        self.autosolve_hold_seconds: float = 3.0
        self.autosolve_change_debounce_seconds: float = 0.6
        self._pending_signature: Optional[str] = None
        self._pending_since: float = 0.0
        self.last_full_text: str = ''
        self.last_llm_solution_text: Optional[str] = None
        self.last_llm_solution_time: float = 0.0
        self.solution_cooldown_seconds: float = 5.0
        self.mathpix_app_id = os.getenv('MATHPIX_APP_ID')
        self.mathpix_app_key = os.getenv('MATHPIX_APP_KEY')
        self.mathpix_enabled = bool(self.mathpix_app_id and self.mathpix_app_key)
        if self.mathpix_enabled:
            print('✓ Mathpix OCR enabled for math symbols')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if OCR_AVAILABLE:
            self.initialize_ocr()
        self.initialize_google_vision()
        try:
            cv2.setUseOptimized(True)
        except Exception:
            pass

    def start_camera(self):
        try:
            camera_index = 0
            best_camera = None
            best_fps = 0
            print('Scanning for available cameras...')
            for i in range(3):
                test_camera = cv2.VideoCapture(i)
                if test_camera.isOpened():
                    test_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                    test_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                    test_camera.set(cv2.CAP_PROP_FPS, 60)
                    actual_fps = test_camera.get(cv2.CAP_PROP_FPS)
                    actual_width = test_camera.get(cv2.CAP_PROP_FRAME_WIDTH)
                    actual_height = test_camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    print(f'Camera {i}: {actual_width:.0f}x{actual_height:.0f} @ {actual_fps:.1f} FPS')
                    if actual_fps > best_fps:
                        best_fps = actual_fps
                        best_camera = i
                        if actual_fps >= 30:
                            break
                    test_camera.release()
            if best_camera is not None:
                camera_index = best_camera
                print(f'Selected camera {camera_index} with {best_fps:.1f} FPS')
            else:
                print('No suitable camera found, using default camera 0')
                camera_index = 0
            self.camera = cv2.VideoCapture(camera_index)
            if not self.camera.isOpened():
                print(f'Error: Could not open camera {camera_index}. Please check if your camera is connected.')
                return False
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.camera.set(cv2.CAP_PROP_FPS, 60)
            try:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                self.camera.set(cv2.CAP_PROP_FOURCC, fourcc)
            except Exception:
                pass
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, test_frame = self.camera.read()
            if not ret:
                print('Error: Camera opened but cannot read frames.')
                return False
            actual_w = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            print(f'Camera initialized: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS')
            return True
        except Exception as e:
            print(f'Error initializing camera: {e}')
            return False

    def is_in_center_region(self, x, y, w, h, frame_width, frame_height):
        mid_x = frame_width // 2
        text_left = x
        text_right = x + w
        text_top = y
        text_bottom = y + h
        left_overlap_x = max(0, min(text_right, mid_x) - max(text_left, 0))
        left_overlap_y = max(0, min(text_bottom, frame_height) - max(text_top, 0))
        text_area = w * h
        left_overlap_area = left_overlap_x * left_overlap_y
        return left_overlap_area >= text_area * 0.5

    def is_valid_text_region(self, w, h):
        if w < 12 or h < 8 or w > 1600 or (h > 500):
            return False
        aspect_ratio = w / float(h)
        if aspect_ratio < 0.15 or aspect_ratio > 30:
            return False
        area = w * h
        if area < 80 or area > 600000:
            return False
        return True

    def is_valid_word_region(self, w: int, h: int) -> bool:
        if w < 6 or h < 6 or w > 2000 or (h > 600):
            return False
        aspect_ratio = w / float(max(1, h))
        if aspect_ratio < 0.08 or aspect_ratio > 50:
            return False
        area = w * h
        if area < 30 or area > 800000:
            return False
        return True

    def is_valid_text_content(self, text):
        text = text.strip()
        if len(text) < 1:
            return False
        readable_chars = sum((1 for c in text if c.isalnum()))
        if readable_chars < 1:
            return False
        if len(set(text)) < 2 and len(text) > 3:
            return False
        if all((not c.isalnum() for c in text)) and len(text) > 2:
            return False
        if len(text) >= 1 and readable_chars >= 1:
            return True
        return False

    def remove_overlapping_regions(self, regions, overlap_threshold=0.5):
        if not regions:
            return regions
        regions = sorted(regions, key=lambda x: x[2] * x[3], reverse=True)
        filtered_regions = []
        for region in regions:
            x1, y1, w1, h1 = region
            is_overlapping = False
            for existing in filtered_regions:
                x2, y2, w2, h2 = existing
                x_left = max(x1, x2)
                y_top = max(y1, y2)
                x_right = min(x1 + w1, x2 + w2)
                y_bottom = min(y1 + h1, y2 + h2)
                if x_right > x_left and y_bottom > y_top:
                    intersection = (x_right - x_left) * (y_bottom - y_top)
                    area1 = w1 * h1
                    area2 = w2 * h2
                    smaller_area = min(area1, area2)
                    if intersection / smaller_area > overlap_threshold:
                        is_overlapping = True
                        break
            if not is_overlapping:
                filtered_regions.append(region)
        return filtered_regions

    def remove_overlapping_text_regions(self, regions, overlap_threshold=0.5):
        if not regions:
            return regions
        regions = sorted(regions, key=lambda x: x[5], reverse=True)
        filtered_regions = []
        for region in regions:
            x1, y1, w1, h1, text1, conf1 = region
            is_overlapping = False
            for existing in filtered_regions:
                x2, y2, w2, h2, text2, conf2 = existing
                x_left = max(x1, x2)
                y_top = max(y1, y2)
                x_right = min(x1 + w1, x2 + w2)
                y_bottom = min(y1 + h1, y2 + h2)
                if x_right > x_left and y_bottom > y_top:
                    intersection = (x_right - x_left) * (y_bottom - y_top)
                    area1 = w1 * h1
                    area2 = w2 * h2
                    smaller_area = min(area1, area2)
                    if intersection / smaller_area > overlap_threshold:
                        is_overlapping = True
                        break
            if not is_overlapping:
                filtered_regions.append(region)
        return filtered_regions

    def _detection_worker(self):
        while not self._det_stop.is_set():
            try:
                frame = self._det_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                now = time.time()
                if now - self._det_last_time < self._det_interval_sec:
                    continue
                self._det_last_time = now
                _disp, _cnt = self.detect_text(frame)
            except Exception as e:
                print(f'Detection worker error: {e}')
            finally:
                pass

    def _start_detection_thread(self):
        if self._det_thread is None or not self._det_thread.is_alive():
            self._det_stop.clear()
            self._det_thread = threading.Thread(target=self._detection_worker, daemon=True)
            self._det_thread.start()

    def _stop_detection_thread(self):
        try:
            self._det_stop.set()
            if self._det_thread is not None:
                self._det_thread.join(timeout=0.5)
        except Exception:
            pass

    def _offer_frame_to_detector(self, frame: np.ndarray):
        try:
            if self._det_queue.full():
                try:
                    _ = self._det_queue.get_nowait()
                except queue.Empty:
                    pass
            self._det_queue.put_nowait(frame)
        except Exception:
            pass

    def _draw_regions(self, frame: np.ndarray, regions: List[dict]):
        for r in regions:
            x, y, w, h = (r['x'], r['y'], r['w'], r['h'])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            label = r.get('label')
            if label:
                txt = str(label)[:20]
                cv2.putText(frame, txt, (x, max(10, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    def select_best_region(self) -> Optional[dict]:
        if not self.last_text_regions:
            return None
        areas = [r['w'] * r['h'] for r in self.last_text_regions]
        max_area = max(areas) if areas else 1.0

        def composite(r):
            area_component = r['w'] * r['h'] / max_area
            score_component = float(r.get('score', 0.0))
            return 0.5 * area_component + 0.5 * score_component
        return max(self.last_text_regions, key=composite)

    def capture_image_with_frame(self, processed_frame):
        if self.camera is None:
            print('Error: Camera not initialized')
            return None
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'text_detection_{timestamp}.jpg'
            filepath = os.path.join(self.output_dir, filename)
            success = cv2.imwrite(filepath, processed_frame)
            if not success:
                print('Error: Could not save image')
                return None
            self.captured_images.append(filepath)
            print(f'✓ Image captured and saved: {os.path.basename(filepath)}')
            return (filepath, processed_frame)
        except Exception as e:
            print(f'Error capturing image: {e}')
            return None

    def capture_image(self):
        if self.camera is None:
            print('Error: Camera not initialized')
            return None
        try:
            ret, frame = self.camera.read()
            if not ret:
                print('Error: Could not capture frame')
                return None
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'text_detection_{timestamp}.jpg'
            filepath = os.path.join(self.output_dir, filename)
            success = cv2.imwrite(filepath, frame)
            if not success:
                print('Error: Could not save image')
                return None
            self.captured_images.append(filepath)
            print(f'✓ Image captured and saved: {os.path.basename(filepath)}')
            return (filepath, frame)
        except Exception as e:
            print(f'Error capturing image: {e}')
            return None

    def run_camera_feed(self):
        if not self.start_camera():
            return
        print('\n=== Text Detection Camera Controls ===')
        print("• Press 't' to toggle text detection on/off")
        print("• Press 'a' to toggle accurate mode (slower but more accurate)")
        print("• Press 'g' to cycle granularity (paragraph/line/word)")
        print("• Press 'f' to show/save full recognized text (Vision)")
        print("• Press 'q' to quit")
        print('• Close the camera window to quit')
        print('=====================================\n')
        if self.accurate_mode:
            if 'easyocr' in sys.modules and isinstance(self.ocr_reader, easyocr.Reader):
                print('✓ Accurate mode: EasyOCR (reads text)')
            elif self.ocr_reader == 'tesseract':
                print('✓ Accurate mode: Tesseract (reads text)')
            else:
                print('⚠ Accurate mode requested but OCR not available; using fast contour detection')
        else:
            print('✓ Fast mode: contour-based detection (no OCR) for higher FPS')
        print()
        cv2.namedWindow('Text Detection Camera', cv2.WINDOW_NORMAL)
        frame_count = 0
        last_capture_time = 0
        text_count = 0
        last_detection_time = time.time()
        detection_results = None
        detection_frame_count = 0
        self._start_detection_thread()
        while self.running:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    print('Error: Could not read frame from camera')
                    break
                display_frame = frame
                if self.detection_enabled and self.ocr_reader is not None:
                    self._offer_frame_to_detector(frame.copy())
                    if self.last_text_regions:
                        self._draw_regions(display_frame, self.last_text_regions)
                        text_count = len(self.last_text_regions)
                    else:
                        text_count = 0
                else:
                    text_count = 0
                    if self.last_text_regions:
                        self.last_text_regions = []
                h, w = display_frame.shape[:2]
                cv2.putText(display_frame, "Press 't' to toggle, 'a' for accurate mode", (w - 350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_frame, f'Captured: {len(self.captured_images)}', (w - 200, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                status = 'ON' if self.detection_enabled else 'OFF'
                cv2.putText(display_frame, f'Detection: {status}', (w - 200, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                mode = 'ACCURATE' if self.accurate_mode else 'FAST'
                cv2.putText(display_frame, f'Mode: {mode}', (w - 200, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                if text_count > 0:
                    cv2.putText(display_frame, f'Text regions: {text_count}', (w - 200, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                else:
                    cv2.putText(display_frame, 'No text detected', (w - 200, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
                try:
                    frame_h, frame_w = frame.shape[:2]
                    if frame_count % 10 == 0:
                        current_signature = self._compute_regions_signature(self.last_text_regions, frame_w, frame_h)
                    else:
                        current_signature = getattr(self, '_last_computed_signature', '')
                    self._last_computed_signature = current_signature
                    now = time.time()
                    time_since_last_solution = now - self.last_llm_solution_time
                    in_cooldown = time_since_last_solution < self.solution_cooldown_seconds
                    if not current_signature:
                        if self.last_text_regions:
                            self.last_text_regions = []
                            print('Text no longer visible - cleared cached regions')
                        self._hold_signature = None
                        self._hold_start_time = 0.0
                        self._pending_signature = None
                        self._pending_since = 0.0
                    if current_signature and (not in_cooldown):
                        if self._hold_signature is None:
                            self._hold_signature = current_signature
                            self._hold_start_time = now
                            self._pending_signature = None
                            self._pending_since = 0.0
                        elif current_signature != self._hold_signature:
                            if self._pending_signature is None:
                                self._pending_signature = current_signature
                                self._pending_since = now
                            elif current_signature == self._pending_signature:
                                if now - self._pending_since > 2.5:
                                    self._hold_signature = current_signature
                                    self._hold_start_time = now
                                    self._pending_signature = None
                                    self._pending_since = 0.0
                                    print('Text changed - resetting timer')
                            else:
                                self._pending_signature = current_signature
                                self._pending_since = now
                        hold_secs = now - self._hold_start_time
                        if not self._autosolve_in_progress and hold_secs >= self.autosolve_hold_seconds:
                            self.last_llm_solution_text = None
                            self._start_autosolve(self.last_text_regions, frame.copy())
                            cv2.putText(display_frame, 'Capturing complete equation...', (w - 250, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)
                        elif self._autosolve_in_progress:
                            cv2.putText(display_frame, 'Solution Processing...', (w - 200, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)
                        else:
                            remaining_time = max(0.0, self.autosolve_hold_seconds - hold_secs)
                            cv2.putText(display_frame, f'Hold steady: {remaining_time:.1f}s', (w - 200, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)
                    elif in_cooldown:
                        remaining_cooldown = max(0.0, self.solution_cooldown_seconds - time_since_last_solution)
                        cv2.putText(display_frame, f'Cooldown: {remaining_cooldown:0.1f}s', (w - 200, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                        self._hold_signature = None
                        self._hold_start_time = 0.0
                        self._pending_signature = None
                        self._pending_since = 0.0
                    else:
                        self._hold_signature = None
                        self._hold_start_time = 0.0
                        self._pending_signature = None
                        self._pending_since = 0.0
                except Exception as e:
                    print(f'Error in autosolve logic: {e}')
                    pass
                left_w = int(w * 0.6)
                mid_x = left_w + 10
                cv2.line(display_frame, (mid_x, 0), (mid_x, h), (100, 100, 100), 2)
                left_h = h - 80
                left_x = 10
                left_y = 40
                cv2.rectangle(display_frame, (left_x, left_y), (left_x + left_w, left_y + left_h), (0, 255, 255), 2)
                if self._autosolve_in_progress and (not self.last_llm_solution_text):
                    try:
                        right_w = w - mid_x - 20
                        box_w = right_w - 20
                        box_h = 120
                        x0 = mid_x + 10
                        y0 = (h - box_h) // 2
                        x1 = x0 + box_w
                        y1 = y0 + box_h
                        overlay = display_frame.copy()
                        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)
                        cv2.rectangle(display_frame, (x0, y0), (x1, y1), (255, 0, 255), 2)
                        processing_text = 'Solution Processing...'
                        text_size = cv2.getTextSize(processing_text, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)[0]
                        text_width = text_size[0]
                        text_x = x0 + (box_w - text_width) // 2
                        text_y = y0 + (box_h + text_size[1]) // 2
                        cv2.putText(display_frame, processing_text, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
                    except Exception:
                        pass
                elif self.last_llm_solution_text:
                    try:
                        clean_text = self._clean_solution_text(self.last_llm_solution_text)
                        right_w = w - mid_x - 20
                        max_chars = max(25, int(right_w / 16))
                        lines = []
                        for paragraph in clean_text.splitlines():
                            p = paragraph.strip()
                            if not p:
                                lines.append('')
                                continue
                            while len(p) > max_chars:
                                brk = p.rfind(' ', 0, max_chars)
                                if brk <= 0:
                                    brk = max_chars
                                lines.append(p[:brk].strip())
                                p = p[brk:].strip()
                            if p:
                                lines.append(p)
                        line_h = 35
                        box_w = right_w - 20
                        box_h = min(h - 40, len(lines) * line_h + 80)
                        x0 = mid_x + 10
                        y0 = (h - box_h) // 2
                        x1 = x0 + box_w
                        y1 = y0 + box_h
                        overlay = display_frame.copy()
                        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)
                        cv2.rectangle(display_frame, (x0, y0), (x1, y1), (255, 0, 255), 2)
                        header_y = y0 + 35
                        header_text = 'AI SOLUTION'
                        header_size = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)[0]
                        header_width = header_size[0]
                        header_x = x0 + (box_w - header_width) // 2
                        cv2.putText(display_frame, header_text, (header_x, header_y), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 0), 2)
                        cv2.putText(display_frame, header_text, (header_x, header_y), cv2.FONT_HERSHEY_DUPLEX, 0.8, (100, 200, 255), 1)
                        y_txt = y0 + 75
                        for i, ln in enumerate(lines):
                            color = (255, 255, 255)
                            text_size = cv2.getTextSize(ln, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)[0]
                            text_width = text_size[0]
                            text_x = x0 + (box_w - text_width) // 2
                            cv2.putText(display_frame, ln, (text_x + 1, y_txt + 1), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)
                            cv2.putText(display_frame, ln, (text_x, y_txt), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 1)
                            y_txt += line_h
                    except Exception:
                        pass
                cv2.imshow('Text Detection Camera', display_frame)
                key = cv2.waitKey(1) & 255
                try:
                    if cv2.getWindowProperty('Text Detection Camera', cv2.WND_PROP_VISIBLE) < 1:
                        print('Camera window closed. Quitting...')
                        break
                except:
                    break
                if key == ord('q'):
                    print('Quitting camera feed...')
                    break
                elif key == ord('t'):
                    self.detection_enabled = not self.detection_enabled
                    status = 'enabled' if self.detection_enabled else 'disabled'
                    print(f'Text detection {status}')
                elif key == ord('a'):
                    self.accurate_mode = not self.accurate_mode
                    mode = 'accurate' if self.accurate_mode else 'fast'
                    print(f'Switched to {mode} mode')
                elif key == ord('g'):
                    self.granularity = 'line' if self.granularity == 'paragraph' else 'word' if self.granularity == 'line' else 'paragraph'
                    print(f'Granularity set to: {self.granularity}')
                elif key == ord('f'):
                    if self.last_full_text:
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        txt_path = os.path.join(self.output_dir, f'full_text_{timestamp}.txt')
                        try:
                            with open(txt_path, 'w') as f:
                                f.write(self.last_full_text)
                            print(f'✓ Saved full text to {os.path.basename(txt_path)}')
                        except Exception as e:
                            print(f'Could not save full text: {e}')
                        print('\n=== Recognized Text (Vision) ===\n' + self.last_full_text + '\n===============================\n')
                    else:
                        print('No full text available yet. Ensure Vision OCR is active and in view.')
                frame_count += 1
            except Exception as e:
                print(f'Error in camera loop: {e}')
                break
        self.cleanup()

    def cleanup(self):
        try:
            self._stop_detection_thread()
            if self.camera is not None:
                self.camera.release()
                self.camera = None
        except Exception as e:
            print(f'Error releasing camera: {e}')
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            print(f'Error destroying windows: {e}')

    def get_captured_images(self):
        return self.captured_images

def main():
    print('=== Real-time Text Detection Camera ===')
    print('This application detects text in real-time and draws blue boxes around it.')
    print('Perfect for capturing math problems and other text content.')
    print()
    if not OCR_AVAILABLE:
        print('⚠ WARNING: No OCR library found!')
        print('To enable text detection, install one of the following:')
        print('  pip install easyocr')
        print('  pip install pytesseract')
        print('The camera will still work, but without text detection.')
        print()
    app = TextDetectionCamera()
    try:
        app.run_camera_feed()
        if app.captured_images:
            print(f'\nTotal images captured: {len(app.captured_images)}')
            print('Images saved in:', app.output_dir)
        print('\nApplication completed successfully!')
    except KeyboardInterrupt:
        print('\nApplication interrupted by user')
        app.cleanup()
    except Exception as e:
        print(f'Unexpected error: {e}')
        app.cleanup()
    finally:
        app.cleanup()
if __name__ == '__main__':
    main()
