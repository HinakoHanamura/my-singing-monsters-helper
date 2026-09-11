import difflib
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from rapidocr_onnxruntime import RapidOCR
    _HAS_RAPIDOCR = True
except ImportError:
    _HAS_RAPIDOCR = False

from config import PROJECT_ROOT

logger = logging.getLogger(__name__)

DEFAULT_LETTERS_DIR = os.path.join(PROJECT_ROOT, "assets", "letters")


KNOWN_ISLANDS: Tuple[str, ...] = ()

_SHARED_OCR_ENGINE: Optional[Any] = None


def get_shared_ocr_engine() -> Optional[Any]:
    """Return process-wide singleton RapidOCR instance, lazily initialized."""
    global _SHARED_OCR_ENGINE
    if _SHARED_OCR_ENGINE is None and _HAS_RAPIDOCR:
        try:
            _SHARED_OCR_ENGINE = RapidOCR(use_angle_cls=False)
            logger.info("RapidOCR shared singleton initialized (use_angle_cls=False)")
        except Exception as e:
            logger.warning("Failed to initialize RapidOCR shared singleton: %s", e)
    return _SHARED_OCR_ENGINE


class LetterRecognizer:
    """Character-level recognition engine for island card titles.

    Isolates character bounding boxes from the high-contrast white text band
    and matches them against 26*2 single-letter templates.
    """

    def __init__(self, letters_dir: Optional[str] = None) -> None:
        self._dir = letters_dir or DEFAULT_LETTERS_DIR
        self._templates: Dict[str, Tuple[Dict, np.ndarray]] = {}
        self._template_holes: Dict[str, int] = {}
        self._hash_cache: List[Tuple[int, str]] = []
        self._load_templates()

    def _load_templates(self) -> None:
        """Load 26*2 letter templates and metadata from disk."""
        if not os.path.isdir(self._dir):
            logger.warning("letter template directory not found: %s", self._dir)
            return

        meta_path = os.path.join(self._dir, "letters.json")
        metadata: Dict[str, Dict] = {}
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception:
                logger.exception("failed to load letter metadata from %s", meta_path)

        for fname in os.listdir(self._dir):
            if not fname.endswith(".png"):
                continue
            key = fname[:-4]
            path = os.path.join(self._dir, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            h, w = img.shape
            info = metadata.get(
                key,
                {
                    "char": key.split("_")[-1],
                    "is_upper": key.startswith("upper_"),
                    "orig_w": w,
                    "orig_h": h,
                    "ar": round(float(w / h), 3) if h > 0 else 1.0,
                },
            )
            norm = img.astype(np.float32) / 255.0
            self._templates[key] = (info, norm)
            # Precompute enclosed hole count for topological matching
            padded = np.pad((norm > 0.1).astype(np.uint8), 1, mode="constant", constant_values=0)
            n_labels, _ = cv2.connectedComponents((padded == 0).astype(np.uint8))
            self._template_holes[key] = max(0, n_labels - 2)

        logger.info("loaded %d character templates from %s", len(self._templates), self._dir)

    def recognize_text_band(self, band_bgr: np.ndarray) -> str:
        """Extract and recognize characters from a title band crop.

        Args:
            band_bgr: BGR crop containing the island name text.

        Returns:
            Recognized title string (e.g. 'Cold Island').
        """
        if band_bgr is None or band_bgr.size == 0 or not self._templates:
            return ""

        # Extract white text core (RGB > 200)
        white_mask = (
            (band_bgr[:, :, 0] > 200)
            & (band_bgr[:, :, 1] > 200)
            & (band_bgr[:, :, 2] > 200)
        )
        white_u8 = (white_mask * 255).astype(np.uint8)

        contours, _ = cv2.findContours(
            white_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        raw_boxes: List[Tuple[int, int, int, int]] = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # Filter noise and subtitle progress text below main title
            if w >= 2 and h >= 3 and y < 24:
                raw_boxes.append((x, y, w, h))

        if not raw_boxes:
            return ""

        # Merge vertically aligned dots (e.g. 'i', 'j')
        merged = self._merge_dots(raw_boxes)

        chars: List[str] = []
        prev_right: Optional[int] = None
        is_first_in_word = True

        for x, y, w, h in merged:
            # Filter trailing badge icon if x is far right and area is large
            if x > 230 and (w * h) > 200:
                continue

            # Word spacing gap
            if prev_right is not None and (x - prev_right) > 7:
                chars.append(" ")
                is_first_in_word = True
            prev_right = x + w

            chip = white_u8[y : y + h, x : x + w]
            char = self._classify_chip(chip, w, h, is_first_in_word=is_first_in_word)
            chars.append(char)
            is_first_in_word = False

        return "".join(chars).strip()

    @staticmethod
    def clean_title_tokens(text: str) -> str:
        """Normalize title text into clean whitespace-delimited lowercase words."""
        if not text:
            return ""
        s = text.replace("\ufffd", "")
        raw_words = s.strip().split()
        words = [re.sub(r"^[^\w]+|[^\w]+$", "", w) for w in raw_words]
        words = [w for w in words if w]
        return " ".join(words).strip().lower()

    @staticmethod
    def words_match(a: str, b: str) -> bool:
        """Compare two word tokens with 'l'/'1'/'i' interchangeability and prefix guarding."""
        if a == b:
            return True
        na = a.replace("l", "i").replace("1", "i")
        nb = b.replace("l", "i").replace("1", "i")
        if na == nb:
            return True
        if abs(len(na) - len(nb)) >= 2 and min(len(na), len(nb)) <= 4:
            return False
        if len(na) <= 4 and len(nb) <= 4 and na[0] != nb[0]:
            return False
        return difflib.SequenceMatcher(None, na, nb).ratio() >= 0.65

    @classmethod
    def names_fuzzy_match(cls, s1: str, s2: str) -> bool:
        """Fuzzy match two island names, resilient to OCR typos and optional badge noise."""
        if not s1 or not s2:
            return False
        c1 = cls.clean_title_tokens(s1)
        c2 = cls.clean_title_tokens(s2)
        if not c1 or not c2:
            return False
        if c1 == c2:
            return True

        w1 = c1.split()
        w2 = c2.split()

        # Direct token-by-token comparison
        if len(w1) == len(w2) and len(w1) > 0:
            return all(cls.words_match(a, b) for a, b in zip(w1, w2))

        # If word counts differ due to an isolated single-char badge (e.g. 'A'), compare multi-char tokens
        if len(w1) != len(w2):
            w1_multi = [w for w in w1 if len(w) > 1]
            w2_multi = [w for w in w2 if len(w) > 1]
            if len(w1_multi) == len(w2_multi) and len(w1_multi) > 0:
                return all(cls.words_match(a, b) for a, b in zip(w1_multi, w2_multi))

        return False

    def resolve_canonical_name(
        self,
        raw_text: str,
        vocabulary: Optional[Sequence[str]] = None,
    ) -> str:
        """Resolve raw OCR text against an optional reference vocabulary.

        If no vocabulary is provided, returns the stripped raw text directly.
        """
        raw = raw_text.strip()
        if not raw or not vocabulary:
            return raw

        for k in vocabulary:
            if raw.lower() == k.lower():
                return k

        words = raw.lower().split()
        first_word = words[0] if words else ""

        best_match = raw
        best_score = 0.0

        for k in vocabulary:
            k_words = k.lower().split()
            k_first = k_words[0]

            len_ratio = min(len(first_word), len(k_first)) / max(len(first_word), len(k_first))
            same_initial = (first_word[0] == k_first[0]) if (first_word and k_first) else False
            same_second = (len(first_word) > 1 and len(k_first) > 1 and first_word[1] == k_first[1])
            same_last = (first_word[-1] == k_first[-1]) if (first_word and k_first) else False

            # Guard against short word cross-consonant match (e.g. cold vs gold)
            if len(first_word) <= 4 and len(k_first) <= 4 and not same_initial:
                continue

            s_first = difflib.SequenceMatcher(None, first_word, k_first).ratio()
            s_full = difflib.SequenceMatcher(None, raw.lower(), k.lower()).ratio()

            base_score = max(s_first, s_full)
            base_score *= (0.6 + 0.4 * len_ratio)

            if same_initial:
                base_score += 0.12 * len_ratio
            if same_second:
                base_score += 0.08 * len_ratio
            if same_last:
                base_score += 0.06 * len_ratio

            if base_score > best_score:
                best_score = base_score
                best_match = k

        return best_match if best_score >= 0.50 else raw

    @staticmethod
    def _compute_dhash(crop: np.ndarray, hash_size: int = 8) -> int:
        if crop is None or crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            return 0
        resized = cv2.resize(crop, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
        diff = gray[:, 1:] > gray[:, :-1]
        val = 0
        for bit in diff.flatten():
            val = (val << 1) | int(bit)
        return val

    @staticmethod
    def _hash_dist(h1: int, h2: int) -> int:
        return bin((h1 ^ h2) & 0xFFFFFFFFFFFFFFFF).count("1")

    @staticmethod
    def extract_title_crop(card_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Extract the high-contrast 'black outline with white interior' title region from a card.

        Isolates the title text by detecting white-fill connected components with dark
        perimeters, groups them into horizontal text lines, selects strictly the top-most
        line of text (ignoring the second line containing monster capacity counts like 42/69),
        clusters horizontally to exclude isolated artwork/pins, and returns the cropped title.
        """
        if card_bgr is None or card_bgr.size == 0:
            return None

        h, w = card_bgr.shape[:2]
        # Dynamically search the upper region where title and stats live
        roi = card_bgr[0 : int(h * 0.55), 0 : w]
        if roi.size == 0:
            return None

        rh, rw = roi.shape[:2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 1. White interior mask: High luminance (gray >= 170) and low saturation (sat <= 65)
        white_mask = (gray >= 170) & (hsv[:, :, 1] <= 65)

        # 2. Black outline detection: Letters have a distinct dark stroke (gray <= 90)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(white_mask.astype(np.uint8))

        boxes = []
        for lbl in range(1, num_labels):
            bx, by, bw, bh, area = stats[lbl]
            # Noise filter: area < 4 or height not matching font bounds
            if area < 4 or bh < 5 or bh > int(rh * 0.90):
                continue
            # Filter non-letter geometry: allow vertical strokes (I, l) up to 8.0 and wide letters (W, M) up to 3.0
            if bw / float(bh) > 3.0 or bh / float(bw) > 8.0:
                continue
            comp_m = (labels == lbl)
            comp_dil = cv2.dilate(comp_m.astype(np.uint8), kernel, iterations=2)
            outline = (comp_dil == 1) & (~comp_m)
            dark_pixels = np.sum(outline & (gray <= 90))
            # Black outline with white interior: at least 40% of the perimeter must be dark border
            # (Real game font letters empirically have 72%-83% dark outline, while building highlights have <30%)
            if dark_pixels / max(1, np.sum(outline)) >= 0.40:
                boxes.append((bx, by, bw, bh, area))

        if not boxes:
            return roi

        # 3. Group boxes into horizontal text lines (rows) based on vertical overlap
        boxes.sort(key=lambda b: b[1])
        lines: List[List[Tuple[int, int, int, int, int]]] = []
        for b in boxes:
            bx, by, bw, bh, area = b
            placed = False
            for line in lines:
                line_y1 = min(box[1] for box in line)
                line_y2 = max(box[1] + box[3] for box in line)
                bc_y = by + bh / 2
                lc_y = (line_y1 + line_y2) / 2
                if abs(bc_y - lc_y) <= max(bh, 10) * 0.75:
                    line.append(b)
                    placed = True
                    break
            if not placed:
                lines.append([b])

        # Filter lines: only consider lines with at least 2 letter candidates
        text_lines = [l for l in lines if len(l) >= 2]
        text_lines.sort(key=lambda l: min(box[1] for box in l))

        # Select strictly the top-most line of text (the second line with monster count is ignored)
        top_line = text_lines[0] if text_lines else (lines[0] if lines else [])
        if not top_line:
            return roi

        # 4. Horizontal clustering within top line: adaptive distance-based clustering scaled to character height
        top_line.sort(key=lambda b: b[0])
        char_heights = [b[3] for b in top_line]
        p75_h = float(np.percentile(char_heights, 75)) if char_heights else 16.0
        # In MSM font, inter-word space is ~1.0 * char_h; any gap > 1.30 * char_h isolates distant buildings/pins
        max_gap = max(14, int(p75_h * 1.30))
        clusters: List[List[Tuple[int, int, int, int, int]]] = []
        cur: List[Tuple[int, int, int, int, int]] = [top_line[0]]
        for b in top_line[1:]:
            if b[0] - (cur[-1][0] + cur[-1][2]) <= max_gap:
                cur.append(b)
            else:
                clusters.append(cur)
                cur = [b]
        clusters.append(cur)

        # Primary title is the cluster with the largest cumulative letter area
        t_cl = max(clusters, key=lambda cl: sum(b[4] for b in cl))
        min_x = min(b[0] for b in t_cl)
        max_x = max(b[0] + b[2] for b in t_cl)
        min_y = min(b[1] for b in t_cl)
        max_y = max(b[1] + b[3] for b in t_cl)

        # Generous padding to ensure RapidOCR recognizes spaces and edge letters accurately
        pad_x = max(14, int(rh * 0.18))
        pad_y = max(6, int(rh * 0.10))
        c_x1 = max(0, min_x - pad_x)
        c_x2 = min(rw, max_x + pad_x)
        c_y1 = max(0, min_y - pad_y)
        c_y2 = min(rh, max_y + pad_y)
        return roi[c_y1:c_y2, c_x1:c_x2]

    def recognize_card(self, card_bgr: np.ndarray) -> str:
        """Recognize island title from an island card crop using RapidOCR with fallback.

        Args:
            card_bgr: BGR crop of the island card (approx 109x360).

        Returns:
            Recognized island title string preserving symbols and punctuation.
        """
        if card_bgr is None or card_bgr.size == 0:
            return ""

        # 1. Check visual perceptual hash cache (0ms lookup)
        chash = self._compute_dhash(card_bgr)
        if chash != 0:
            for cached_hash, cached_name in self._hash_cache:
                if self._hash_dist(chash, cached_hash) <= 6:
                    return cached_name

        ocr = get_shared_ocr_engine()
        if ocr is not None:
            try:
                title_crop = self.extract_title_crop(card_bgr)
                if title_crop is None or title_crop.size == 0:
                    h, w = card_bgr.shape[:2]
                    title_crop = card_bgr[int(h * 0.04) : int(h * 0.35), int(w * 0.18) : int(w * 0.85)]

                # Direct line text recognition (bypasses full-frame detection, 25x faster)
                if hasattr(ocr, "text_rec") and title_crop.size > 0:
                    rec_res, _ = ocr.text_rec([title_crop])
                    if rec_res and rec_res[0]:
                        raw_str, rec_score = rec_res[0]
                        if raw_str.strip():
                            s = raw_str.replace("\ufffd", "")
                            # Split camelCase / PascalCase word boundaries (e.g. EarthIsland -> Earth Island)
                            s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
                            s = re.sub(r"[ \t]+", " ", s).strip()
                            clean_title = " ".join(w.capitalize() for w in s.split())
                            if clean_title:
                                if chash != 0:
                                    self._hash_cache.append((chash, clean_title))
                                return clean_title

                # Full RapidOCR detection fallback if direct line recognition produces empty
                res, _ = ocr(card_bgr)
                if res:
                    items = []
                    for box, text, score in res:
                        s = text.strip()
                        if not s:
                            continue
                        # Skip subtitle progress counters (e.g. 42/69, 0/32) and standalone digits
                        if re.search(r"^\d+\s*/\s*\d+$", s) or s.isdigit():
                            continue
                        # Filter elements positioned deep in card (subtitle area y > 48)
                        y_top = min(pt[1] for pt in box)
                        if y_top > 48:
                            continue
                        items.append((box, s, score))

                    if items:
                        # Sort horizontally left-to-right
                        items.sort(key=lambda it: min(pt[0] for pt in it[0]))
                        raw_result = " ".join(it[1] for it in items).strip()
                        if raw_result.endswith("."):
                            raw_result = raw_result[:-1].strip()
                        raw_result = re.sub(r"([a-z])([A-Z])", r"\1 \2", raw_result)
                        clean_title = " ".join(w.capitalize() for w in raw_result.split())
                        if clean_title:
                            if chash != 0:
                                self._hash_cache.append((chash, clean_title))
                            return clean_title
            except Exception as e:
                logger.warning("RapidOCR card recognition failed, falling back: %s", e)

        # Fallback to template matching on title band
        h, w = card_bgr.shape[:2]
        y1 = max(0, int(h * 0.04))
        y2 = min(h, int(h * 0.42))
        x1 = max(0, int(w * 0.20))
        x2 = min(w, int(w * 0.96))
        band = card_bgr[y1:y2, x1:x2]
        raw_text = self.recognize_text_band(band)
        result = raw_text.strip()
        if result and chash != 0:
            self._hash_cache.append((chash, result))
        return result

    def is_blacklisted(
        self,
        island_name: str,
        blacklist: Sequence[str],
        threshold: float = 0.55,
    ) -> bool:
        """Check if an island name matches any entry in the blacklist.

        Compares candidate directly against user blacklist items using dynamic
        token similarity without requiring hardcoded alias rules.
        """
        if not island_name or not blacklist:
            return False

        for item in blacklist:
            clean_item = item.strip()
            if not clean_item:
                continue
            if self.names_fuzzy_match(island_name, clean_item):
                return True

        return False

    def _merge_dots(
        self, boxes: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """Merge separate dot components with lower stems (e.g. for 'i')."""
        boxes.sort(key=lambda b: b[0])
        merged: List[Tuple[int, int, int, int]] = []
        skip = set()

        for i in range(len(boxes)):
            if i in skip:
                continue
            x1, y1, w1, h1 = boxes[i]
            for j in range(i + 1, len(boxes)):
                if j in skip:
                    continue
                x2, y2, w2, h2 = boxes[j]
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                if overlap_x >= min(w1, w2) * 0.3 and abs(x1 - x2) <= 5:
                    gap_y = abs(y2 - (y1 + h1)) if y2 > y1 else abs(y1 - (y2 + h2))
                    if gap_y <= 8:
                        nx = min(x1, x2)
                        ny = min(y1, y2)
                        nw = max(x1 + w1, x2 + w2) - nx
                        nh = max(y1 + h1, y2 + h2) - ny
                        x1, y1, w1, h1 = nx, ny, nw, nh
                        skip.add(j)
            if h1 >= 7:
                merged.append((x1, y1, w1, h1))

        merged.sort(key=lambda b: b[0])
        return merged

    def _classify_chip(
        self,
        chip: np.ndarray,
        orig_w: int,
        orig_h: int,
        is_first_in_word: Optional[bool] = None,
    ) -> str:
        """Classify a single character chip against template library."""
        if chip.size == 0 or orig_h <= 0 or orig_w <= 0:
            return "?"

        target_dim = 18
        scale = target_dim / max(orig_h, orig_w)
        nh = max(1, int(round(orig_h * scale)))
        nw = max(1, int(round(orig_w * scale)))
        resized = cv2.resize(chip, (nw, nh), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((24, 24), dtype=np.float32)
        yo = (24 - nh) // 2
        xo = (24 - nw) // 2
        canvas[yo : yo + nh, xo : xo + nw] = (resized > 100).astype(np.float32)

        padded_chip = np.pad((canvas > 0.1).astype(np.uint8), 1, mode="constant", constant_values=0)
        n_labels_chip, _ = cv2.connectedComponents((padded_chip == 0).astype(np.uint8))
        chip_holes = max(0, n_labels_chip - 2)

        chip_ar = float(orig_w / orig_h)
        best_char = "?"
        best_score = -1.0

        candidates = list(self._templates.items())
        if is_first_in_word is not None:
            scoped = [item for item in candidates if item[1][0]["is_upper"] == is_first_in_word]
            if scoped:
                candidates = scoped

        for key, (info, tmpl) in candidates:
            # Aspect ratio factor penalty
            ar_diff = abs(chip_ar - info["ar"])
            ar_factor = max(0.0, 1.0 - ar_diff * 0.5)

            # Max cosine similarity over 1px translation shifts
            best_cos = 0.0
            best_excess = 0.0
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    M = np.float32([[1, 0, dx], [0, 1, dy]])
                    shifted = cv2.warpAffine(canvas, M, (24, 24))
                    dot = (shifted * tmpl).sum()
                    norm = np.linalg.norm(shifted) * np.linalg.norm(tmpl)
                    cos_sim = float(dot / norm) if norm > 0 else 0.0
                    if cos_sim > best_cos:
                        best_cos = cos_sim
                        best_excess = float((tmpl * (1.0 - (shifted > 0.1))).sum() / max(1e-5, tmpl.sum()))

            # Topology match: bonus/penalty for matching number of enclosed holes
            tmpl_holes = self._template_holes.get(key, 0)
            topo_bonus = 0.06 if (chip_holes == tmpl_holes) else -0.06

            score = best_cos * 0.75 + ar_factor * 0.15 - best_excess * 0.05 + topo_bonus
            if score > best_score:
                best_score = score
                best_char = info["char"]

        return best_char
