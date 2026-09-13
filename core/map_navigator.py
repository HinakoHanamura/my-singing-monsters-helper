from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import win32con

from config import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    TARGET_MAP_BUTTON,
    TARGET_MAP_GO,
    AppConfig,
)
from core.action_agent import ActionAgent
from core.adaptive_roi import DynamicROI, MatchResult, _scale_template
from core.game_window import GameWindow
from core.letter_recognizer import LetterRecognizer, get_shared_ocr_engine

logger = logging.getLogger(__name__)


class ScreenState(str, Enum):
    """Major screen states in the game navigation cycle."""

    ISLAND = "island"
    MAP = "map"
    LOADING = "loading"
    MODAL = "modal"
    UNKNOWN = "unknown"


def compute_card_hash(crop: np.ndarray, hash_size: int = 8) -> int:
    """Compute 64-bit difference hash (dHash) for an island card thumbnail.

    Resizes the card image, converts to grayscale, and compares adjacent pixel gradients.
    Highly invariant to lighting, anti-aliasing, and subtle animation noise.
    """
    if crop is None or crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return 0
    resized = cv2.resize(crop, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
    diff = gray[:, 1:] > gray[:, :-1]
    val = 0
    for bit in diff.flatten():
        val = (val << 1) | int(bit)
    return val


def hash_distance(h1: int, h2: int) -> int:
    """Calculate the Hamming distance (number of differing bits) between two 64-bit hashes."""
    return bin((h1 ^ h2) & 0xFFFFFFFFFFFFFFFF).count("1")


def compute_card_hist(crop: np.ndarray) -> np.ndarray:
    """Compute normalized 16x16 Hue-Saturation color histogram for an island card.

    Highly invariant to card selection glow, highlights, and subtle animation noise.
    Empirically verified: same card (selected vs unselected) correl >= 0.95,
    different cards correl <= 0.25.
    """
    if crop is None or crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return np.zeros((16, 16), dtype=np.float32)
    hsv = cv2.cvtColor(cv2.resize(crop, (100, 100)), cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


@dataclass(frozen=True)
class IslandCardInfo:
    """Represents an island entry detected in the left list."""

    index: int
    name: str
    rect: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    click_point: Tuple[int, int]
    is_fully_visible: bool = True
    card_hash: int = 0
    card_hist: Optional[np.ndarray] = None


class MapNavigator:
    """Navigates the map screen, iterates island cards, and transitions between islands.

    Decoupled from specific gathering logic: accepts an optional callback to execute
    on each visited island before proceeding.
    """

    def __init__(
        self,
        action_agent: ActionAgent,
        window: GameWindow,
        config: AppConfig = DEFAULT_CONFIG,
        letter_recognizer: Optional[LetterRecognizer] = None,
    ) -> None:
        self._action = action_agent
        self._window = window
        self._cfg = config
        self._recognizer = letter_recognizer or LetterRecognizer()

        # Pre-load navigation button templates
        template_dir = self._cfg.vision.template_dir
        self._map_btn_tmpl = cv2.imread(f"{template_dir}/map_button.png")
        self._go_btn_tmpl = cv2.imread(f"{template_dir}/map_go_button.png")
        self._go_clean_tmpl = cv2.imread(f"{template_dir}/map_go_clean.png")
        self._here_btn_tmpl = cv2.imread(f"{template_dir}/map_you_are_here.png")
        self._here_clean_tmpl = cv2.imread(f"{template_dir}/map_you_are_here_clean.png")
        self._close_btn_tmpl = cv2.imread(f"{template_dir}/map_close_button.png")
        self._back_btn_tmpl = cv2.imread(f"{template_dir}/map_back_button.png")
        self._card_rail_tmpl = cv2.imread(f"{template_dir}/map_card_rail.png")
        self._modal_cancel_tmpl = cv2.imread(f"{template_dir}/modal_cancel.png")
        self._modal_cancel_round_tmpl = cv2.imread(f"{template_dir}/modal_cancel_round.png")

        # Dynamic ROI trackers (default: no ROI -> global scan fallback)
        self._roi_go = DynamicROI("map_go", margin_x=80, margin_y=80)
        self._roi_here = DynamicROI("map_here", margin_x=80, margin_y=80)
        self._roi_map_btn = DynamicROI("map_button", margin_x=60, margin_y=60)
        self._roi_close_btn = DynamicROI("map_close_or_back", margin_x=60, margin_y=60)
        self._roi_modal_cancel = DynamicROI("modal_cancel", margin_x=60, margin_y=60)

    @property
    def letter_recognizer(self) -> LetterRecognizer:
        """Expose OCR recognizer instance for blacklist checks and title parsing."""
        return self._recognizer

    def _get_scale_steps(self, frame: np.ndarray) -> Tuple[float, ...]:
        """Compute multi-scale factors relative to 1024x768 reference resolution."""
        if frame is None or frame.size == 0:
            return (1.0,)
        h, w = frame.shape[:2]
        base_s = h / 768.0
        return (base_s * 0.90, base_s * 0.95, base_s, base_s * 1.05, base_s * 1.10)

    # -------------------------------------------------------- State Detection

    def detect_state(self, frame: np.ndarray) -> ScreenState:
        """Detect whether the current frame is inside an island, map, or loading.

        Zero fixed coordinates: searches dynamically using DynamicROI
        with multi-scale adaptation and automatic global scan fallback.
        """
        if frame is None or frame.size == 0:
            return ScreenState.UNKNOWN

        h, w = frame.shape[:2]
        ui_scale = h / 768.0

        # Check for close/cancel buttons first (modal popups darken screen corners with a dark scrim)
        modal_cancel_res = self.find_modal_cancel(frame)

        # 1. Fast loading check (< 0.1ms): dark iris transition or low-variance frame
        # (Guarded: a loading screen never has a close button)
        if modal_cancel_res is None:
            corner_margin = max(10, int(40 * ui_scale))
            top_left_dark = float(frame[:corner_margin, :corner_margin].mean()) < 35.0
            top_right_dark = float(frame[:corner_margin, -corner_margin:].mean()) < 35.0
            if (top_left_dark and top_right_dark) or float(frame.std()) < 18.0 or float(frame.mean()) < 25.0:
                return ScreenState.LOADING

        scales = self._get_scale_steps(frame)

        # 2. Fast structural check for Map view: presence of vertical island card rail grooves (< 8ms)
        spacings = []
        if self._card_rail_tmpl is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            th_tmpl = cv2.cvtColor(self._card_rail_tmpl, cv2.COLOR_BGR2GRAY)
            tw = max(10, int(th_tmpl.shape[1] * ui_scale))
            th = max(4, int(th_tmpl.shape[0] * ui_scale))
            if tw < w and th < h:
                scaled_tmpl = cv2.resize(th_tmpl, (tw, th))
                res = cv2.matchTemplate(gray, scaled_tmpl, cv2.TM_CCOEFF_NORMED)
                row_max = res.max(axis=1)
                peaks = [
                    y
                    for y in range(1, len(row_max) - 1)
                    if row_max[y] >= 0.65 and row_max[y] >= row_max[y - 1] and row_max[y] >= row_max[y + 1]
                ]
                filtered = []
                for p in peaks:
                    if not any(abs(p - f) < int(35 * ui_scale) for f in filtered):
                        filtered.append(p)
                min_card_h = int(75 * ui_scale)
                max_card_h = int(150 * ui_scale)
                spacings = [
                    filtered[i + 1] - filtered[i]
                    for i in range(len(filtered) - 1)
                    if min_card_h <= filtered[i + 1] - filtered[i] <= max_card_h
                ]
                if len(spacings) >= 2:
                    return ScreenState.MAP

        # 3. Check for Map view: GO button, 'You are here!' button, or CLOSE button
        if self._go_clean_tmpl is not None or self._go_btn_tmpl is not None:
            go_tmpls = [("clean", self._go_clean_tmpl), ("plant", self._go_btn_tmpl)]
            valid_go = [(name, t) for name, t in go_tmpls if t is not None]
            if valid_go:
                go_res = self._roi_go.match_any(
                    frame, valid_go, threshold=0.55, scales=scales
                )
                if go_res is not None:
                    return ScreenState.MAP

        if self._here_clean_tmpl is not None or self._here_btn_tmpl is not None:
            here_tmpls = [("clean", self._here_clean_tmpl), ("faded", self._here_btn_tmpl)]
            valid_here = [(name, t) for name, t in here_tmpls if t is not None]
            if valid_here:
                here_res = self._roi_here.match_any(
                    frame, valid_here, threshold=0.55, scales=scales
                )
                if here_res is not None:
                    return ScreenState.MAP

        # Check for close/cancel buttons (modal red X, stone close, back)
        modal_cancel_res = self.find_modal_cancel(frame)
        close_res = None
        if modal_cancel_res is None and self._close_btn_tmpl is not None:
            close_res = self._roi_close_btn.match(
                frame, self._close_btn_tmpl, threshold=0.58, scales=scales
            )

        back_res = None
        if modal_cancel_res is None and close_res is None and self._back_btn_tmpl is not None:
            back_res = self._roi_close_btn.match(
                frame, self._back_btn_tmpl, threshold=0.88, scales=scales
            )

        if modal_cancel_res is not None or close_res is not None or back_res is not None:
            # A close/back button alone is common to dialogs (Mailbox, Ads, Settings, etc.).
            # Positively identify MAP only if genuine card rails or multiple visible island cards exist.
            if len(spacings) >= 2 or len(self.get_visible_cards(frame)) >= 2:
                return ScreenState.MAP
            return ScreenState.MODAL

        # 4. Check for Island view: MAP button on screen
        if self._map_btn_tmpl is not None:
            map_res = self._roi_map_btn.match(
                frame, self._map_btn_tmpl, threshold=0.58, scales=scales
            )
            if map_res is not None:
                return ScreenState.ISLAND

        # 5. High-reliability OCR fallback on HUD ROI (strictly restricted to avoid full-frame delay)
        ocr = get_shared_ocr_engine()
        if ocr is not None:
            try:
                # Check bottom HUD bar for Island navigation buttons
                bottom_crop = frame[int(h * 0.70):, :]
                rec_res, _ = ocr(bottom_crop)
                if rec_res:
                    for box, text, score in rec_res:
                        clean_t = text.strip().upper()
                        if score >= 0.70 and clean_t in ("MAP", "COLLECT ALL", "COLLECTALL", "MARKET"):
                            return ScreenState.ISLAND

                # Check left list area for Map markers (e.g. Mirror islands)
                left_crop = frame[:, :int(w * 0.35)]
                rec_left, _ = ocr(left_crop)
                if rec_left:
                    for box, text, score in rec_left:
                        clean_t = text.strip().upper()
                        if score >= 0.70 and clean_t in ("MIRROR",):
                            return ScreenState.MAP
            except Exception:
                pass

        return ScreenState.UNKNOWN

    def wait_for_state(
        self,
        target_state: ScreenState,
        timeout: float = 8.0,
        step_sleep: float = 0.04,
    ) -> bool:
        """Poll the window frame until target_state is reached or timeout expires."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            frame = self._window.capture()
            if frame is not None:
                state = self.detect_state(frame)
                if state == target_state:
                    return True
            time.sleep(step_sleep)
        return False

    # ---------------------------------------------------- High-Level Commands

    def find_modal_cancel(
        self,
        frame: np.ndarray,
        threshold: float = 0.68,
    ) -> Optional[MatchResult]:
        """Perform full-frame multi-scale search for modal cancel/close buttons.

        Uses coarse-to-fine downsampled matching with local native refinement
        to achieve sub-tenth-second detection across high-resolution frames.
        Supports various close button variants (square stone red X, round vine red X,
        or stone CLOSE button) anywhere across the entire screen without spatial restrictions.
        """
        if frame is None or frame.size == 0:
            return None
        h, w = frame.shape[:2]
        ds = 0.5 if (w >= 1200 or h >= 900) else 1.0
        if ds != 1.0:
            small = cv2.resize(frame, (int(w * ds), int(h * ds)), interpolation=cv2.INTER_AREA)
        else:
            small = frame

        scales = self._get_scale_steps(frame)
        candidate_tmpls = [
            self._modal_cancel_tmpl,
            getattr(self, "_modal_cancel_round_tmpl", None),
            self._close_btn_tmpl,
        ]
        valid_tmpls = [t for t in candidate_tmpls if t is not None]

        best_score = -1.0
        best_info = None

        for tmpl in valid_tmpls:
            for s in scales:
                scaled = _scale_template(tmpl, s * ds)
                th, tw = scaled.shape[:2]
                sh, sw = small.shape[:2]
                if sh >= th and sw >= tw:
                    res = cv2.matchTemplate(small, scaled, cv2.TM_CCOEFF_NORMED)
                    _, max_v, _, loc = cv2.minMaxLoc(res)
                    if max_v >= (threshold - 0.05) and max_v > best_score:
                        best_score = max_v
                        best_info = (tmpl, s, loc, tw, th)
                    if max_v >= 0.85:
                        break
            if best_score >= 0.85:
                break

        if best_info is None or best_score < (threshold - 0.05):
            return None

        tmpl, s, loc, tw, th = best_info
        if ds == 1.0:
            if best_score >= threshold:
                cx = loc[0] + tw // 2
                cy = loc[1] + th // 2
                rect = (loc[0], loc[1], loc[0] + tw, loc[1] + th)
                return MatchResult(center=(cx, cy), rect=rect, score=float(best_score))
            return None

        # Refine on native resolution
        scaled_native = _scale_template(tmpl, s)
        nth, ntw = scaled_native.shape[:2]
        est_x = int(loc[0] / ds)
        est_y = int(loc[1] / ds)
        margin = int(24 / ds)
        x0 = max(0, est_x - margin)
        y0 = max(0, est_y - margin)
        x1 = min(w, est_x + ntw + margin)
        y1 = min(h, est_y + nth + margin)

        patch = frame[y0:y1, x0:x1]
        if patch.shape[0] >= nth and patch.shape[1] >= ntw:
            res_fine = cv2.matchTemplate(patch, scaled_native, cv2.TM_CCOEFF_NORMED)
            _, fine_v, _, fine_loc = cv2.minMaxLoc(res_fine)
            if fine_v >= threshold:
                fx = x0 + fine_loc[0] + ntw // 2
                fy = y0 + fine_loc[1] + nth // 2
                rect = (x0 + fine_loc[0], y0 + fine_loc[1], x0 + fine_loc[0] + ntw, y0 + fine_loc[1] + nth)
                return MatchResult(center=(fx, fy), rect=rect, score=float(fine_v))
        return None

    def dismiss_modal(self, frame: Optional[np.ndarray] = None) -> bool:
        """Check if a modal dialog with close button is on screen and dismiss it.

        1. Finds close button anywhere on screen via full-frame matching.
        2. First clicks the detected close button directly.
        3. Verifies that the close button has disappeared; if still present, sends ESC.
        Returns True if a modal was detected and dismissed, False otherwise.
        """
        if frame is None:
            frame = self._window.capture()
            if frame is None or frame.size == 0:
                return False

        match = self.find_modal_cancel(frame)
        if match is None:
            return False

        cx, cy = match.center
        logger.info("detected modal close button at (%d, %d); clicking close button", cx, cy)
        self._action.click(cx, cy)
        time.sleep(0.35)

        # Verification: check if modal close button has disappeared
        fresh = self._window.capture()
        if fresh is not None and fresh.size > 0:
            fresh_match = self.find_modal_cancel(fresh)
            if fresh_match is not None and abs(fresh_match.center[0] - cx) < 35 and abs(fresh_match.center[1] - cy) < 35:
                # Still present at same location; try ESC as fallback
                logger.info("modal close button still present at (%d, %d); sending ESC fallback", cx, cy)
                self._action.send_key(win32con.VK_ESCAPE)
                time.sleep(0.35)

        return True

    def recover_blocked_island_ui(self, frame: Optional[np.ndarray] = None) -> bool:
        """Self-healing: dismiss modal popups or deselect selected apparatus/monsters.

        1. If a modal dialog with a red 'X' close button is detected anywhere on screen:
           Dismisses via dismiss_modal() (ESC key or direct Red X click).
        2. If no modal dialog is present, an apparatus or monster is likely selected:
           Zooms out via multi-notch wheel centered on screen, waits for camera easing,
           then clicks the safe lower-left ocean void (offset to the right of the zoom-out
           minus icon) to safely clear the selection without hitting any buildings or monsters.
        """
        if frame is None or frame.size == 0:
            frame = self._window.capture()
            if frame is None or frame.size == 0:
                return False

        # 1. Check for modal popup with Red X
        if self.dismiss_modal(frame):
            return True

        # 2. Deselect building / monster selection:
        logger.info("executing deselect recovery: zoom out and click lower-left safe ocean")
        h, w = frame.shape[:2]
        center_x = w // 2
        center_y = h // 2

        # Send 3 wheel notches centered on screen to smoothly zoom out
        self._action.wheel(-120, x=center_x, y=center_y, steps=3, step_delay=0.03)
        # Wait for game camera zoom easing animation to complete
        time.sleep(0.45)

        # safe_x is to the right of minus button (x ~ 110-120), safe_y is around 670-690
        safe_x = int(w * 0.11)
        safe_y = int(h * 0.88)
        self._action.click(safe_x, safe_y)
        time.sleep(0.25)
        return True

    def open_map(self, timeout: float = 8.0) -> bool:
        """From island view, click the MAP button and wait for the map interface.

        Active retry mechanism: continuously checks state, locates the MAP button
        with multi-scale adaptation, clicks it, and confirms entry.
        If MAP button is missing, automatically recovers from modal popups or
        accidentally selected apparatus/monsters.
        """
        deadline = time.monotonic() + timeout
        last_click_time = 0.0
        last_recovery_time = 0.0
        recovery_attempts = 0

        while time.monotonic() < deadline:
            frame = self._window.capture()
            if frame is None:
                time.sleep(0.1)
                continue

            # State check: already on map screen? (Only valid if state is MAP)
            cur_state = self.detect_state(frame)
            if cur_state == ScreenState.MAP and len(self.get_visible_cards(frame)) >= 2:
                logger.info("map screen confirmed active")
                self.wait_for_list_stable(timeout=1.2)
                return True

            # Priority: if a modal dialog is on screen, dismiss it first and reset timeout
            modal_cancel = self.find_modal_cancel(frame)
            if modal_cancel is not None:
                logger.info("open_map: modal popup detected on island; dismissing it")
                self.dismiss_modal(frame)
                deadline = time.monotonic() + timeout
                time.sleep(0.35)
                continue

            if self._map_btn_tmpl is None:
                logger.error("MAP button template not available")
                return False

            scales = self._get_scale_steps(frame)
            match_res = self._roi_map_btn.match(
                frame, self._map_btn_tmpl, threshold=0.55, scales=scales
            )

            # OCR fallback for MAP button: strictly restricted to bottom HUD bar
            if match_res is None:
                ocr = get_shared_ocr_engine()
                if ocr is not None:
                    try:
                        h, w = frame.shape[:2]
                        bottom_crop = frame[int(h * 0.70):, :]
                        rec_res, _ = ocr(bottom_crop)
                        if rec_res:
                            for box, text, score in rec_res:
                                if text.strip().upper() == "MAP" and score >= 0.70:
                                    bx1 = min(pt[0] for pt in box)
                                    bx2 = max(pt[0] for pt in box)
                                    by1 = min(pt[1] for pt in box) + int(h * 0.70)
                                    by2 = max(pt[1] for pt in box) + int(h * 0.70)
                                    btn_cx = int((bx1 + bx2) / 2)
                                    btn_cy = int(by1 - (by2 - by1) * 1.2)
                                    match_res = MatchResult(
                                        center=(btn_cx, btn_cy),
                                        rect=(int(bx1), btn_cy - 30, int(bx2), int(by2)),
                                        score=float(score),
                                    )
                                    break
                    except Exception:
                        pass

            now = time.monotonic()
            if match_res is not None:
                click_x, click_y = match_res.center
                if now - last_click_time >= 0.5:
                    logger.info("clicking detected MAP button at (%d, %d)", click_x, click_y)
                    self._action.click(click_x, click_y)
                    last_click_time = now

                    # Wait for map screen to load and list to settle
                    if self.wait_for_state(ScreenState.MAP, timeout=3.0):
                        self.wait_for_list_stable(timeout=1.2)
                        return True
                    fresh = self._window.capture()
                    if fresh is not None and self.detect_state(fresh) == ScreenState.MAP:
                        self.wait_for_list_stable(timeout=1.2)
                        return True
            else:
                if self.detect_state(frame) == ScreenState.MAP:
                    logger.info("already on map screen")
                    self.wait_for_list_stable(timeout=1.2)
                    return True

                # Self-healing: if MAP button is missing, UI may be blocked by a modal popup or selected building
                if (now - last_recovery_time >= 1.2) and (recovery_attempts < 3):
                    logger.warning("MAP button not found on screen, attempting UI recovery...")
                    if self.recover_blocked_island_ui(frame):
                        last_recovery_time = time.monotonic()
                        recovery_attempts += 1
                        time.sleep(0.2)
                        continue

            time.sleep(0.1)

        # Final check if transitioned at deadline
        final_frame = self._window.capture()
        if final_frame is not None and self.detect_state(final_frame) == ScreenState.MAP:
            self.wait_for_list_stable(timeout=1.2)
            return True

        logger.warning("failed to open map within %.1fs timeout", timeout)
        return False

    def close_map(self, timeout: float = 4.0) -> bool:
        """Close map interface using the detected CLOSE or BACK button with active retry."""
        deadline = time.monotonic() + timeout
        last_click_time = 0.0

        while time.monotonic() < deadline:
            frame = self._window.capture()
            if frame is None:
                time.sleep(0.1)
                continue

            # Fast check: already left map view?
            if self.detect_state(frame) == ScreenState.ISLAND:
                return True

            scales = self._get_scale_steps(frame)
            close_match = None
            if self._close_btn_tmpl is not None:
                close_match = self._roi_close_btn.match(
                    frame, self._close_btn_tmpl, threshold=0.55, scales=scales
                )

            back_match = None
            if close_match is None and self._back_btn_tmpl is not None:
                back_match = self._roi_close_btn.match(
                    frame, self._back_btn_tmpl, threshold=0.85, scales=scales
                )

            target_btn = close_match or back_match
            now = time.monotonic()
            if target_btn is not None:
                cx, cy = target_btn.center
                if now - last_click_time >= 0.5:
                    logger.info("clicking map close/back button at (%d, %d)", cx, cy)
                    self._action.click(cx, cy)
                    last_click_time = now

                    if self.wait_for_state(ScreenState.ISLAND, timeout=2.0):
                        return True
            time.sleep(0.1)

        final_frame = self._window.capture()
        if final_frame is not None and self.detect_state(final_frame) == ScreenState.ISLAND:
            return True
        return False

    # ----------------------------------------------------- Island Card Scanning

    def get_visible_cards(self, frame: np.ndarray) -> List[IslandCardInfo]:
        """Scan the left island list and return recognized cards with click points."""
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        ui_scale = h / 768.0
        card_w = max(int(w * 0.36), int(360 * ui_scale))
        center_x = int(card_w / 2)

        # 1. Dynamically lock card positions using the card frame rail template across full frame
        grooves: List[int] = []
        if self._card_rail_tmpl is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            th_tmpl = cv2.cvtColor(self._card_rail_tmpl, cv2.COLOR_BGR2GRAY)
            best_peaks: List[Tuple[int, float]] = []
            for s in (ui_scale * 0.95, ui_scale, ui_scale * 1.05):
                tw = max(10, int(th_tmpl.shape[1] * s))
                th = max(4, int(th_tmpl.shape[0] * s))
                if tw >= w or th >= h:
                    continue
                scaled_tmpl = cv2.resize(th_tmpl, (tw, th))
                res = cv2.matchTemplate(gray, scaled_tmpl, cv2.TM_CCOEFF_NORMED)
                row_max = res.max(axis=1)
                peaks: List[Tuple[int, float]] = []
                for y in range(1, len(row_max) - 1):
                    if row_max[y] >= 0.65 and row_max[y] >= row_max[y - 1] and row_max[y] >= row_max[y + 1]:
                        peaks.append((y, float(row_max[y])))
                filtered: List[Tuple[int, float]] = []
                for p in peaks:
                    if not any(abs(p[0] - f[0]) < int(35 * ui_scale) for f in filtered):
                        filtered.append(p)
                filtered.sort(key=lambda p: p[0])
                if len(filtered) > len(best_peaks):
                    best_peaks = filtered
            grooves = [p[0] for p in best_peaks]

        min_card_h = int(75 * ui_scale)
        max_card_h = int(150 * ui_scale)
        detected_spacings = [
            grooves[i + 1] - grooves[i]
            for i in range(len(grooves) - 1)
            if min_card_h <= (grooves[i + 1] - grooves[i]) <= max_card_h
        ]
        nominal_h = int(np.median(detected_spacings)) if detected_spacings else int(109 * ui_scale)

        cards_tuples: List[Tuple[int, int, str]] = []

        # Card above the first detected groove (e.g. Plant Island at top)
        if grooves:
            first_cy1 = grooves[0]
            if first_cy1 >= min_card_h:
                top_cy1 = max(0, first_cy1 - nominal_h)
                card_crop = frame[top_cy1:first_cy1, 0:card_w]
                name = self._recognizer.recognize_card(card_crop)
                if name:
                    cards_tuples.append((top_cy1, first_cy1, name))

        # Intermediate cards between detected grooves
        for i in range(len(grooves) - 1):
            cy1 = grooves[i]
            cy2 = grooves[i + 1]
            ch = cy2 - cy1
            if min_card_h <= ch <= max_card_h:
                card_crop = frame[cy1:cy2, 0:card_w]
                name = self._recognizer.recognize_card(card_crop)
                if name:
                    cards_tuples.append((cy1, cy2, name))

        # Card below the last detected groove
        if grooves:
            last_cy2 = grooves[-1]
            if h - last_cy2 >= min_card_h:
                bot_cy2 = min(h, last_cy2 + nominal_h)
                card_crop = frame[last_cy2:bot_cy2, 0:card_w]
                name = self._recognizer.recognize_card(card_crop)
                if name:
                    cards_tuples.append((last_cy2, bot_cy2, name))

        cards: List[IslandCardInfo] = []
        for idx, (cy1, cy2, name) in enumerate(cards_tuples):
            is_full = (cy2 <= int(680 * ui_scale)) and (cy2 - cy1 >= int(nominal_h * 0.92))
            card_crop = frame[cy1:cy2, 0:card_w]
            chash = compute_card_hash(card_crop)
            chist = compute_card_hist(card_crop)
            cards.append(
                IslandCardInfo(
                    index=idx,
                    name=name,
                    rect=(0, cy1, card_w, cy2),
                    click_point=(center_x, (cy1 + cy2) // 2),
                    is_fully_visible=is_full,
                    card_hash=chash,
                    card_hist=chist,
                )
            )

        return cards

    # ---------------------------------------------------- Island Transitions

    def select_island(self, card: IslandCardInfo) -> bool:
        """Click an island card in the left list."""
        cx, cy = card.click_point
        logger.info("selecting island card '%s' at (%d, %d)", card.name, cx, cy)
        return self._action.click(cx, cy)

    def wait_for_panel_stable(
        self,
        timeout: float = 0.5,
        poll_interval: float = 0.02,
        motion_threshold: float = 4.0,
        consecutive_required: int = 2,
    ) -> bool:
        """Wait dynamically until the right-hand island preview panel stabilizes.

        Prevents false-starts where a previous island's preview or button
        is clicked before the new island's panel finishes animating in.
        """
        deadline = time.monotonic() + timeout
        last_crop = None
        stable_count = 0

        while time.monotonic() < deadline:
            frame = self._window.capture()
            if frame is None:
                time.sleep(poll_interval)
                continue

            h, w = frame.shape[:2]
            y1, y2 = int(h * 0.10), int(h * 0.70)
            x1, x2 = int(w * 0.40), int(w * 0.90)
            cur_crop = frame[y1:y2, x1:x2]

            if last_crop is not None and last_crop.shape == cur_crop.shape:
                diff = float(np.mean(cv2.absdiff(cur_crop, last_crop)))
                if diff < motion_threshold:
                    stable_count += 1
                    if stable_count >= consecutive_required:
                        logger.debug("island preview panel has settled (diff=%.2f, count=%d)", diff, stable_count)
                        return True
                else:
                    stable_count = 0

            last_crop = cur_crop
            time.sleep(poll_interval)

        return False

    def enter_selected_island(
        self,
        timeout: float = 8.0,
        target_card: Optional[IslandCardInfo] = None,
    ) -> bool:
        """Wait for screen transition and click detected 'GO' or 'Here' button.

        Includes anti-race guards:
        1. Dynamically waits for the right panel to settle when target_card is provided,
           preventing clicking an old island's GO button.
        2. Rapid matching with fast cadence (~20ms).
        3. Active retry mechanism if click was dropped by game (re-click after 350ms).
        4. Positive island confirmation to prevent premature action in island view.
        """
        if target_card is not None:
            self.wait_for_panel_stable(timeout=0.8, poll_interval=0.01)

        deadline = time.monotonic() + timeout
        step_sleep = 0.02
        last_click_time = 0.0
        clicked_transition = False

        go_templates = [
            ("clean", self._go_clean_tmpl),
            ("plant", self._go_btn_tmpl),
        ]
        here_templates = [
            ("clean", self._here_clean_tmpl),
            ("faded", self._here_btn_tmpl),
        ]

        while time.monotonic() < deadline:
            frame = self._window.capture()
            if frame is not None:
                h, w = frame.shape[:2]
                state = self.detect_state(frame)

                # 1. Screen is already in island view -> entry succeeded!
                if state == ScreenState.ISLAND:
                    logger.info("screen confirmed in island view")
                    return True

                # 2. If screen transitioned into loading or left map after click, wait for island
                if state == ScreenState.LOADING or (clicked_transition and state != ScreenState.MAP):
                    if self.wait_for_state(ScreenState.ISLAND, timeout=self._cfg.map.map_timeout):
                        return True

                now = time.monotonic()
                scales = self._get_scale_steps(frame)

                # 3. Check GO button FIRST via DynamicROI across full frame
                go_match = self._roi_go.match_any(frame, go_templates, threshold=0.45, scales=scales)
                if go_match is not None:
                    _, go_res = go_match
                    cx, cy = go_res.center
                    if now - last_click_time >= 0.35:
                        logger.info(
                            "detected GO button (score=%.3f), clicking (%d, %d)",
                            go_res.score,
                            cx,
                            cy,
                        )
                        self._action.click(cx, cy)
                        last_click_time = now
                        clicked_transition = True
                        if self.wait_for_state(ScreenState.ISLAND, timeout=self._cfg.map.map_timeout):
                            return True

                # 4. Check 'You are here!' button via DynamicROI across full frame
                here_match = self._roi_here.match_any(frame, here_templates, threshold=0.50, scales=scales)
                if here_match is not None:
                    _, here_res = here_match
                    cx, cy = here_res.center
                    if now - last_click_time >= 0.35:
                        logger.info(
                            "detected 'You are here!' button (score=%.3f), clicking (%d, %d)",
                            here_res.score,
                            cx,
                            cy,
                        )
                        self._action.click(cx, cy)
                        last_click_time = now
                        clicked_transition = True
                        if self.wait_for_state(ScreenState.ISLAND, timeout=self._cfg.map.map_timeout):
                            return True

            time.sleep(step_sleep)

        # Final check if screen transitioned at the end of timeout
        final_frame = self._window.capture()
        if final_frame is not None:
            if self.detect_state(final_frame) == ScreenState.ISLAND:
                return True
            if clicked_transition:
                st = self.detect_state(final_frame)
                if st != ScreenState.MAP:
                    cards = self.get_visible_cards(final_frame)
                    if len(cards) == 0:
                        logger.info("transition completed (left map screen, cards cleared)")
                        return True

        logger.warning("neither GO nor 'You are here!' entered successfully within timeout")
        return False

    def wait_for_list_stable(
        self,
        timeout: float = 1.8,
        poll_interval: float = 0.05,
        motion_threshold: float = 2.5,
        consecutive_required: int = 2,
    ) -> bool:
        """Wait dynamically until the left island list decelerates to stationary.

        Captures consecutive frames and computes pixel difference in the list region.
        Requires `consecutive_required` consecutive samples below `motion_threshold`
        to guarantee that deformation rebound has fully ended and list velocity is zero.
        """
        deadline = time.monotonic() + timeout
        last_crop = None
        stable_count = 0

        while time.monotonic() < deadline:
            frame = self._window.capture()
            if frame is None:
                time.sleep(poll_interval)
                continue

            h, w = frame.shape[:2]
            sy = h / 768.0
            sx = w / 1024.0
            ui_scale = min(sx, sy)
            y1, y2 = int(80 * sy), int(680 * sy)
            x1, x2 = int(50 * ui_scale), int(330 * ui_scale)
            cur_crop = frame[y1:y2, x1:x2]

            if last_crop is not None and last_crop.shape == cur_crop.shape:
                diff = float(np.mean(cv2.absdiff(cur_crop, last_crop)))
                if diff < motion_threshold:
                    stable_count += 1
                    if stable_count >= consecutive_required:
                        logger.debug("island list has settled to stationary (motion diff=%.2f, count=%d)", diff, stable_count)
                        return True
                else:
                    stable_count = 0

            last_crop = cur_crop
            time.sleep(poll_interval)

        logger.debug("list stabilization reached timeout (motion did not fully drop below %.2f)", motion_threshold)
        return False

    def scroll_down(self) -> bool:
        """Drag the left list upwards to reveal the next cards."""
        w, h = self._window.client_size()
        if w <= 0 or h <= 0:
            w, h = 1024, 768
        sx = w / 1024.0
        sy = h / 768.0
        ui_scale = min(sx, sy)

        drag_x = int(self._cfg.map.drag_x * ui_scale)
        start_y = int(self._cfg.map.drag_start_y * sy)
        end_y = int(self._cfg.map.drag_end_y * sy)

        logger.info("scrolling list downwards: drag (%d, %d) -> (%d, %d)", drag_x, start_y, drag_x, end_y)
        ok = self._action.drag(
            start_x=drag_x,
            start_y=start_y,
            end_x=drag_x,
            end_y=end_y,
            duration=self._cfg.map.drag_duration,
            steps=self._cfg.map.drag_steps,
        )
        self.wait_for_list_stable(timeout=1.5)
        return ok

    def scroll_to_top(self, max_swipes: int = 10) -> bool:
        """Drag the left list downwards dynamically until the top ceiling is reached.

        Pure dynamic overscroll invariance:
        Exploits list physics: dragging downwards at top stretches/bounces the list,
        then rebounds back to its resting position and decelerates to stationary.
        When the resting list region content remains identical to the content
        before the swipe, initialization is complete.
        """
        w, h = self._window.client_size()
        if w <= 0 or h <= 0:
            w, h = 1024, 768
        sx = w / 1024.0
        sy = h / 768.0
        ui_scale = min(sx, sy)

        drag_x = int(self._cfg.map.drag_x * ui_scale)
        start_y = int(220 * sy)
        end_y = int(580 * sy)

        y1, y2 = int(80 * sy), int(680 * sy)
        x1, x2 = int(50 * ui_scale), int(330 * ui_scale)

        brake_mode = getattr(self._cfg.map, "init_brake_mode", "dynamic")
        first_island_target = getattr(self._cfg.map, "first_island_name", "Plant Island").strip()

        # Ensure list is completely stationary before recording pre-drag baseline
        self.wait_for_list_stable(timeout=1.0)
        pre_frame = self._window.capture()
        last_settled_crop: Optional[np.ndarray] = None
        last_cards: List[IslandCardInfo] = []

        if pre_frame is not None:
            cards = self.get_visible_cards(pre_frame)
            if cards:
                if brake_mode == "first_island" and first_island_target:
                    top_name = cards[0].name.strip().lower()
                    tgt_name = first_island_target.lower()
                    if top_name == tgt_name or tgt_name in top_name or self._recognizer.is_blacklisted(cards[0].name, [first_island_target]):
                        logger.info("already at specified first island '%s' at top", first_island_target)
                        self._action.park_cursor()
                        return True
                last_cards = cards
            last_settled_crop = pre_frame[y1:y2, x1:x2].copy()

        for swipe_idx in range(max_swipes):
            logger.info("pulling list downwards to top (swipe %d/%d, mode=%s)", swipe_idx + 1, max_swipes, brake_mode)
            self._action.drag(
                start_x=drag_x,
                start_y=start_y,
                end_x=drag_x,
                end_y=end_y,
                duration=self._cfg.map.drag_duration,
                steps=self._cfg.map.drag_steps,
            )

            # Wait for list to decelerate to complete rest (absorb deformation and rebound damping)
            self.wait_for_list_stable(timeout=1.8, motion_threshold=2.5, consecutive_required=2)

            frame = self._window.capture()
            if frame is None:
                continue

            curr_settled_crop = frame[y1:y2, x1:x2]
            cards = self.get_visible_cards(frame)

            # 1. Target first island check
            if brake_mode == "first_island" and first_island_target and cards:
                top_name = cards[0].name.strip().lower()
                tgt_name = first_island_target.lower()
                if top_name == tgt_name or tgt_name in top_name or self._recognizer.is_blacklisted(cards[0].name, [first_island_target]):
                    logger.info("reached specified first island '%s' at top after %d swipe(s), stopping", first_island_target, swipe_idx + 1)
                    break

            # 2. Physical overscroll deformation and rebound consistency check
            if last_settled_crop is not None and last_settled_crop.shape == curr_settled_crop.shape:
                crop_diff = float(np.mean(cv2.absdiff(curr_settled_crop, last_settled_crop)))
                cards_match = bool(
                    cards and last_cards and
                    cards[0].card_hash != 0 and last_cards[0].card_hash != 0 and
                    hash_distance(cards[0].card_hash, last_cards[0].card_hash) <= 2 and
                    abs(cards[0].rect[1] - last_cards[0].rect[1]) <= 10
                )
                if crop_diff < 4.0 or cards_match:
                    logger.info(
                        "列表上拉形变回弹并减速静止后，区域内容与前次完全一致（差异度 %.2f），物理置顶初始化完毕",
                        crop_diff,
                    )
                    break

            last_settled_crop = curr_settled_crop.copy()
            last_cards = cards

        self._action.park_cursor()
        return True

    def find_and_enter_island(
        self,
        target_name: str,
        on_located: Optional[Callable[[str], None]] = None,
    ) -> Tuple[bool, str]:
        """Search the map list for target_name and enter it.

        Returns (success: bool, status: str)
        status: "success" | "entry_timeout" | "not_found" | "open_map_failed"
        """
        clean_target = target_name.strip()
        if not clean_target:
            return False, "invalid_target"

        frame = self._window.capture()
        if frame is None:
            return False, "capture_failed"

        if self.detect_state(frame) != ScreenState.MAP:
            logger.info("opening map to locate target island '%s'", clean_target)
            if not self.open_map():
                return False, "open_map_failed"

        self.scroll_to_top()

        last_bottom_name = ""
        consecutive_same_bottom = 0

        while True:
            frame = self._window.capture()
            if frame is None:
                break

            cards = self.get_visible_cards(frame)
            if not cards:
                self.scroll_down()
                continue

            for card in cards:
                if card.name == clean_target or self._recognizer.is_blacklisted(card.name, [clean_target]):
                    logger.info("target island '%s' matched card '%s'", clean_target, card.name)
                    if on_located:
                        on_located(card.name)
                    self.select_island(card)
                    entered = self.enter_selected_island(target_card=card)
                    if entered:
                        return True, "success"
                    fresh_frame = self._window.capture()
                    if fresh_frame is not None and self.detect_state(fresh_frame) == ScreenState.ISLAND:
                        return True, "success"
                    return False, "entry_timeout"

            current_bottom_name = cards[-1].name.strip().lower() if cards else ""
            if current_bottom_name == last_bottom_name:
                consecutive_same_bottom += 1
                if consecutive_same_bottom >= 2:
                    logger.warning("reached end of island list without finding '%s'", clean_target)
                    break
            else:
                consecutive_same_bottom = 0
                last_bottom_name = current_bottom_name

            self.scroll_down()
            time.sleep(0.3)

        return False, "not_found"
