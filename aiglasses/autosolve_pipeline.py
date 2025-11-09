import base64
import json
import os
import re
import threading
import time
from datetime import datetime
from typing import List, Optional
import cv2
import numpy as np
import requests
from .vision_pipeline import easyocr

class AutosolveMixin:

    def _compute_regions_signature(self, regions: List[dict], frame_w: int, frame_h: int) -> str:
        if not regions:
            return ''
        sig_items = []
        for r in regions:
            x, y, w, h = (r.get('x', 0), r.get('y', 0), r.get('w', 0), r.get('h', 0))
            if not self.is_in_center_region(x, y, w, h, frame_w, frame_h):
                continue
            label = str(r.get('label', '')).strip()
            q = 50
            qx, qy, qw, qh = (x // q, y // q, max(1, w // q), max(1, h // q))
            sig_items.append(f'{label}|{qx},{qy},{qw},{qh}')
        if not sig_items:
            return ''
        sig_items.sort()
        return '#'.join(sig_items)

    def _start_autosolve(self, regions: List[dict], frame_for_crop: np.ndarray):
        if self._autosolve_in_progress:
            return
        complete_regions = self._ensure_complete_equation(regions, frame_for_crop)
        parts: List[dict] = []
        for r in complete_regions:
            x, y, w, h = (r['x'], r['y'], r['w'], r['h'])
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(frame_for_crop.shape[1], x + w)
            y1 = min(frame_for_crop.shape[0], y + h)
            if x1 <= x0 or y1 <= y0:
                continue
            crop = frame_for_crop[y0:y1, x0:x1]
            crop = self.preprocess_crop_for_ocr(crop)
            parts.append({'img': crop, 'text': r.get('label', '')})
        if not parts:
            return

        def worker():
            try:
                self._autosolve_in_progress = True
                answer = self.send_multiple_images_to_llm(parts)
                if answer:
                    concise = self._extract_concise_answer(answer)
                    print('\n=== LLM Solution (autosolve) ===\n' + answer + '\n===============================\n')
                    if concise:
                        print(f'Answer: {concise}')
                    self.last_llm_solution_text = answer.strip()
                    self.last_llm_solution_time = time.time()
                else:
                    print('Autosolve: LLM did not return a solution.')
            finally:
                self._autosolve_in_progress = False
                self._hold_signature = None
                self._hold_start_time = 0.0
        threading.Thread(target=worker, daemon=True).start()

    def _cluster_blocks(self, blocks: List[dict], frame_w: int, frame_h: int) -> List[List[dict]]:
        if not blocks:
            return []
        clusters: List[List[dict]] = []

        def iou(a, b):
            ax0, ay0, ax1, ay1 = (a['x'], a['y'], a['x'] + a['w'], a['y'] + a['h'])
            bx0, by0, bx1, by1 = (b['x'], b['y'], b['x'] + b['w'], b['y'] + b['h'])
            ix0, iy0 = (max(ax0, bx0), max(ay0, by0))
            ix1, iy1 = (min(ax1, bx1), min(ay1, by1))
            iw, ih = (max(0, ix1 - ix0), max(0, iy1 - iy0))
            inter = iw * ih
            area_a = (ax1 - ax0) * (ay1 - ay0)
            area_b = (bx1 - bx0) * (by1 - by0)
            union = area_a + area_b - inter + 1e-06
            return inter / union

        def are_neighbors(a, b):
            if iou(a, b) > 0.02:
                return True
            avg_h = (a['h'] + b['h']) / 2.0
            avg_w = (a['w'] + b['w']) / 2.0
            gap_x = max(0, max(a['x'], b['x']) - min(a['x'] + a['w'], b['x'] + b['w']))
            gap_y = max(0, max(a['y'], b['y']) - min(a['y'] + a['h'], b['y'] + b['h']))
            return gap_x < avg_w * 0.3 or gap_y < avg_h * 0.6
        for block in blocks:
            placed = False
            for cluster in clusters:
                if any((are_neighbors(block, other) for other in cluster)):
                    cluster.append(block)
                    placed = True
                    break
            if not placed:
                clusters.append([block])
        merged = True
        while merged and len(clusters) > 1:
            merged = False
            out = []
            while clusters:
                c = clusters.pop()
                merged_into_existing = False
                for d in out:
                    if any((are_neighbors(a, b) for a in c for b in d)):
                        d.extend(c)
                        merged_into_existing = True
                        merged = True
                        break
                if not merged_into_existing:
                    out.append(c)
            clusters = out
        return clusters

    def _ensure_complete_equation(self, regions: List[dict], frame_for_crop: np.ndarray) -> List[dict]:
        if not regions:
            return regions
        clusters = self._cluster_blocks(regions, frame_for_crop.shape[1], frame_for_crop.shape[0])
        complete_regions = []
        for cluster in clusters:
            if not cluster:
                continue
            min_x = min((block['x'] for block in cluster))
            min_y = min((block['y'] for block in cluster))
            max_x = max((block['x'] + block['w'] for block in cluster))
            max_y = max((block['y'] + block['h'] for block in cluster))
            padding_x = int(max(20, (max_x - min_x) * 0.1))
            padding_y = int(max(15, (max_y - min_y) * 0.15))
            x = max(0, min_x - padding_x)
            y = max(0, min_y - padding_y)
            w = min(frame_for_crop.shape[1] - x, max_x - min_x + 2 * padding_x)
            h = min(frame_for_crop.shape[0] - y, max_y - min_y + 2 * padding_y)
            combined_text = ' '.join((block.get('label', '') for block in cluster if block.get('label')))
            complete_region = {'x': x, 'y': y, 'w': w, 'h': h, 'score': max((block.get('score', 0.9) for block in cluster)), 'label': combined_text, 'is_complete_equation': True}
            complete_regions.append(complete_region)
        return complete_regions if complete_regions else regions

    def save_crop(self, crop_bgr: np.ndarray) -> str:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'ocr_crop_{timestamp}.jpg'
        filepath = os.path.join(self.output_dir, filename)
        cv2.imwrite(filepath, crop_bgr)
        return filepath

    def _extract_concise_answer(self, text: str) -> str:
        if not text:
            return ''
        lower = text.lower()
        for key in ['answer:', 'final answer:', 'result:']:
            if key in lower:
                idx = lower.rfind(key)
                return text[idx + len(key):].strip().splitlines()[0].strip()
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        if lines:
            return lines[-1]
        return text.strip()

    def _clean_solution_text(self, text: str) -> str:
        text = re.sub('\\\\\\[(.*?)\\\\\\]', '\\1', text)
        text = re.sub('\\\\\\((.*?)\\\\\\)', '\\1', text)
        replacements = {'\\\\frac\\{([^}]+)\\}\\{([^}]+)\\}': '\\1/\\2', '\\\\sqrt\\{([^}]+)\\}': '√\\1', '\\\\times': '×', '\\\\div': '÷', '\\\\geq': '≥', '\\\\leq': '≤', '\\\\neq': '≠', '\\\\pm': '±', '\\\\infty': '∞', '\\\\sum': 'Σ', '\\\\int': '∫', '\\\\alpha': 'α', '\\\\beta': 'β', '\\\\gamma': 'γ', '\\\\delta': 'δ', '\\\\theta': 'θ', '\\\\pi': 'π', '\\\\sigma': 'σ', '\\\\mu': 'μ', '\\\\lambda': 'λ', '\\\\cdot': '·', '\\\\approx': '≈', '\\\\propto': '∝', '\\\\partial': '∂', '\\\\nabla': '∇', '\\\\forall': '∀', '\\\\exists': '∃', '\\\\in': '∈', '\\\\notin': '∉', '\\\\subset': '⊂', '\\\\supset': '⊃', '\\\\cup': '∪', '\\\\cap': '∩', '\\\\emptyset': '∅', '\\\\rightarrow': '→', '\\\\leftarrow': '←', '\\\\leftrightarrow': '↔', '\\\\Rightarrow': '⇒', '\\\\Leftarrow': '⇐', '\\\\Leftrightarrow': '⇔'}
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        text = re.sub('\\s+', ' ', text)
        text = re.sub('\\*\\*(.*?)\\*\\*', '\\1', text)
        text = re.sub('\\*(.*?)\\*', '\\1', text)
        text = text.replace('\\', '')
        text = text.replace('{', '').replace('}', '')
        text = text.replace('[', '').replace(']', '')
        text = text.replace('???', '')
        text = text.replace('??', '')
        text = text.replace('?', '')
        text = re.sub('(\\d+)\\s*×\\s*(\\d+)', '\\1 × \\2', text)
        text = re.sub('(\\d+)\\s*÷\\s*(\\d+)', '\\1 ÷ \\2', text)
        text = re.sub('(\\d+)\\s*\\+\\s*(\\d+)', '\\1 + \\2', text)
        text = re.sub('(\\d+)\\s*-\\s*(\\d+)', '\\1 - \\2', text)
        text = text.replace('\\boxed{', '').replace('}', '')
        text = text.replace('\\text{', '').replace('}', '')
        text = text.replace('\\mathrm{', '').replace('}', '')
        text = re.sub('\\\\[a-zA-Z]+\\{([^}]*)\\}', '\\1', text)
        lines = text.split('\n')
        structured_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            line = re.sub('^\\d+\\.\\s*', '', line)
            if line.lower().startswith('step') and len(line) < 20:
                continue
            if line.startswith('Calculate') or line.startswith('Then') or line.startswith('Next') or line.startswith('Finally') or line.startswith('Subtract') or line.startswith('Divide') or line.startswith('Add') or line.startswith('Multiply'):
                if structured_lines and structured_lines[-1].strip():
                    structured_lines.append('')
                structured_lines.append(line)
            elif '=' in line and any((op in line for op in ['+', '-', '×', '÷', '/'])):
                if structured_lines and structured_lines[-1].strip():
                    structured_lines.append('')
                structured_lines.append(line)
            elif line.lower().startswith('answer') or line.lower().startswith('solution') or line.lower().startswith('result'):
                if structured_lines and structured_lines[-1].strip():
                    structured_lines.append('')
                structured_lines.append('')
                structured_lines.append(line)
                structured_lines.append('')
            else:
                structured_lines.append(line)
        result = '\n'.join(structured_lines)
        result = re.sub('\\n\\s*\\n\\s*\\n', '\n\n', result)
        return result.strip()

    def send_image_to_llm(self, crop_bgr: np.ndarray) -> Optional[str]:
        try:
            success, buf = cv2.imencode('.jpg', crop_bgr)
            if not success:
                print('Error: Failed to encode crop for LLM')
                return None
            b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
            data_url = f'data:image/jpeg;base64,{b64}'
            recognized_text = self.ocr_extract_text(crop_bgr)
            provider = os.getenv('LLM_PROVIDER', '').strip().lower()
            openai_key = os.getenv('OPENAI_API_KEY')
            anthropic_key = os.getenv('ANTHROPIC_API_KEY')
            had_provider = False
            last_error_message = None
            if (provider == 'openai' or not provider) and openai_key:
                try:
                    from openai import OpenAI
                    org_id = os.getenv('OPENAI_ORG_ID')
                    client = OpenAI(api_key=openai_key, organization=org_id) if org_id else OpenAI(api_key=openai_key)
                    prompt = 'Solve this problem:'
                    resp = client.chat.completions.create(model=os.getenv('OPENAI_VISION_MODEL', 'gpt-4o-mini'), messages=[{'role': 'user', 'content': [{'type': 'text', 'text': prompt}, {'type': 'image_url', 'image_url': {'url': data_url}}, *([{'type': 'text', 'text': f'OCR (may be noisy): {recognized_text}'}] if recognized_text else [])]}], temperature=0.0)
                    text = resp.choices[0].message.content if resp.choices else None
                    if text:
                        return text
                except Exception as e:
                    last_error_message = f'OpenAI call failed: {e}'
                    print(last_error_message)
                finally:
                    had_provider = True
            if anthropic_key:
                try:
                    import anthropic
                    client = anthropic.Anthropic(api_key=anthropic_key)
                    prompt = 'Solve this problem:'
                    msg = client.messages.create(model=os.getenv('ANTHROPIC_VISION_MODEL', 'claude-3-5-sonnet-20240620'), max_tokens=1024, temperature=0.0, messages=[{'role': 'user', 'content': [{'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': b64}}, {'type': 'text', 'text': prompt}, *([{'type': 'text', 'text': f'OCR (may be noisy): {recognized_text}'}] if recognized_text else [])]}])
                    parts = []
                    for block in msg.content:
                        if getattr(block, 'type', None) == 'text':
                            parts.append(getattr(block, 'text', ''))
                    return '\n'.join((p for p in parts if p))
                except Exception as e:
                    last_error_message = f'Anthropic call failed: {e}'
                    print(last_error_message)
                finally:
                    had_provider = True
            if not had_provider:
                print('No LLM provider configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.')
            elif last_error_message:
                print(f'LLM call did not succeed: {last_error_message}')
            return None
        except Exception as e:
            print(f'Error preparing image for LLM: {e}')
            return None

    def send_multiple_images_to_llm(self, parts: List[dict]) -> Optional[str]:
        try:
            content_blocks = []
            content_blocks.append({'type': 'text', 'text': 'Solve this question and output the answer:'})
            for i, p in enumerate(parts, start=1):
                img = p.get('img')
                text_hint = p.get('text', '')
                success, buf = cv2.imencode('.jpg', img)
                if not success:
                    continue
                b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
                data_url = f'data:image/jpeg;base64,{b64}'
                content_blocks.append({'type': 'image_url', 'image_url': {'url': data_url}})
                if text_hint:
                    content_blocks.append({'type': 'text', 'text': f'OCR (may be noisy) part {i}: {text_hint}'})
            provider = os.getenv('LLM_PROVIDER', '').strip().lower()
            openai_key = os.getenv('OPENAI_API_KEY')
            anthropic_key = os.getenv('ANTHROPIC_API_KEY')
            had_provider = False
            last_error_message = None
            if (provider == 'openai' or not provider) and openai_key:
                try:
                    from openai import OpenAI
                    org_id = os.getenv('OPENAI_ORG_ID')
                    client = OpenAI(api_key=openai_key, organization=org_id) if org_id else OpenAI(api_key=openai_key)
                    resp = client.chat.completions.create(model=os.getenv('OPENAI_VISION_MODEL', 'gpt-4o-mini'), messages=[{'role': 'user', 'content': content_blocks}], temperature=0.0)
                    text = resp.choices[0].message.content if resp.choices else None
                    if text:
                        return text
                except Exception as e:
                    last_error_message = f'OpenAI call failed: {e}'
                    print(last_error_message)
                finally:
                    had_provider = True
            if anthropic_key:
                try:
                    import anthropic
                    client = anthropic.Anthropic(api_key=anthropic_key)
                    anthro_content = []
                    for block in content_blocks:
                        if block.get('type') == 'text':
                            anthro_content.append({'type': 'text', 'text': block.get('text', '')})
                        elif block.get('type') == 'image_url':
                            url = block.get('image_url', {}).get('url', '')
                            if url.startswith('data:image/jpeg;base64,'):
                                b64 = url.split(',', 1)[1]
                                anthro_content.append({'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': b64}})
                    msg = client.messages.create(model=os.getenv('ANTHROPIC_VISION_MODEL', 'claude-3-5-sonnet-20240620'), max_tokens=1024, temperature=0.0, messages=[{'role': 'user', 'content': anthro_content}])
                    parts_out = []
                    for block in msg.content:
                        if getattr(block, 'type', None) == 'text':
                            parts_out.append(getattr(block, 'text', ''))
                    return '\n'.join((p for p in parts_out if p))
                except Exception as e:
                    last_error_message = f'Anthropic call failed: {e}'
                    print(last_error_message)
                finally:
                    had_provider = True
            if not had_provider:
                print('No LLM provider configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.')
            elif last_error_message:
                print(f'LLM call did not succeed: {last_error_message}')
            return None
        except Exception as e:
            print(f'Error preparing multi-image request for LLM: {e}')
            return None
