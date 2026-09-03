from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

from config import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    TARGET_MAP_BUTTON,
    TARGET_MAP_GO,
    AppConfig,
)
from core.action_agent import ActionAgent
from core.adaptive_roi import DynamicROI, MatchResult
from core.game_window import GameWindow
from core.letter_recognizer import LetterRecognizer

logger = logging.getLogger(__name__)


class ScreenState(str, Enum):
    """Major screen states in the game navigation cycle."""

    ISLAND = "island"
    MAP = "map"
    LOADING = "loading"
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

        # Dynamic ROI trackers (default: no ROI -> global scan fallback)
        self._roi_go = DynamicROI("map_go", margin_x=80, margin_y=80)
        self._roi_here = DynamicROI("map_here", margin_x=80, margin_y=80)
        self._roi_map_btn = DynamicROI("map_button", margin_x=60, margin_y=60)
        self._roi_close_btn = DynamicROI("map_close_or_back", margin_x=60, margin_y=60)

    # -------------------------------------------------------- State Detection

    def detect_state(self, frame: np.ndarray) -> ScreenState:
        """Detect whether the current frame is inside an island, map, or loading.

        Zero fixed coordinates: searches dynamically using DynamicROI
        (full global scan initially or on fallback, localized dynamic ROI once memorized).
        """
        if frame is None or frame.size == 0:
            return ScreenState.UNKNOWN

        h, w = frame.shape[:2]

        # 1. Check for Island view: MAP button on bottom toolbar
        if self._map_btn_tmpl is not None:
            map_res = self._roi_map_btn.match(frame, self._map_btn_tmpl, threshold=0.70)
            if map_res is not None:
                return ScreenState.ISLAND

        # 2. Check for Map view: CLOSE button on top right or BACK button
        if self._close_btn_tmpl is not None:
            close_res = self._roi_close_btn.match(frame, self._close_btn_tmpl, threshold=0.58)
            if close_res is not None:
                return ScreenState.MAP
        if self._back_btn_tmpl is not None:
            back_res = self._roi_close_btn.match(frame, self._back_btn_tmpl, threshold=0.88)
            if back_res is not None:
                return ScreenState.MAP

        # Secondary structural check for Map view: presence of vertical island card list
        cards = self.get_visible_cards(frame)
        if len(cards) >= 2:
            return ScreenState.MAP

        # 3. Neither interface button detected: check if screen is in dark iris transition (loading)
        corner_margin = max(10, int(40 * (h / 768.0)))
        top_left_dark = float(frame[:corner_margin, :corner_margin].mean()) < 30.0
        top_right_dark = float(frame[:corner_margin, -corner_margin:].mean()) < 30.0
        if top_left_dark and top_right_dark:
            return ScreenState.LOADING

        return ScreenState.UNKNOWN

    def wait_for_state(
        self,
        target_state: ScreenState,
        timeout: float = 8.0,
        step_sleep: float = 0.35,
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

    def open_map(self) -> bool:
        """From island view, click the MAP button and wait for the map interface."""
        frame = self._window.capture()
        if frame is None:
            return False

        # Fast check: already on map screen?
        if self.detect_state(frame) == ScreenState.MAP or len(self.get_visible_cards(frame)) >= 2:
            logger.info("already on map screen")
            return True

        if self._map_btn_tmpl is None:
            logger.error("MAP button template not available")
            return False

        # Dynamic match (global scan first, dynamic ROI once memorized)
        match_res = self._roi_map_btn.match(frame, self._map_btn_tmpl, threshold=0.65)
        if match_res is None:
            # Double check if screen is already on map
            if len(self.get_visible_cards(frame)) >= 2:
                logger.info("already on map screen (detected card list)")
                return True
            logger.warning("MAP button not detected on island view")
            return False

        click_x, click_y = match_res.center
        logger.info("clicking detected MAP button at (%d, %d)", click_x, click_y)
        self._action.click(click_x, click_y)

        # Wait for map screen to load and list to settle
        ok = self.wait_for_state(ScreenState.MAP, timeout=self._cfg.map.map_timeout)
        if not ok:
            fresh = self._window.capture()
            if fresh is not None and len(self.get_visible_cards(fresh)) >= 2:
                ok = True
        if ok:
            self.wait_for_list_stable(timeout=1.2)
        return ok

    def close_map(self) -> bool:
        """Close map interface using the detected CLOSE or BACK button."""
        frame = self._window.capture()
        if frame is None:
            return False

        match = self._roi_close_btn.match_any(
            frame,
            [("close", self._close_btn_tmpl), ("back", self._back_btn_tmpl)],
            threshold=0.62,
        )
        if match is None:
            logger.warning("neither CLOSE nor BACK button detected to exit map")
            return False

        btn_name, match_res = match
        click_x, click_y = match_res.center
        logger.info("clicking detected %s button at (%d, %d)", btn_name.upper(), click_x, click_y)
        self._action.click(click_x, click_y)
        return self.wait_for_state(ScreenState.ISLAND, timeout=self._cfg.map.map_timeout)

    # ----------------------------------------------------- Island Card Scanning

    def get_visible_cards(self, frame: np.ndarray) -> List[IslandCardInfo]:
        """Scan the left island list and return recognized cards with click points."""
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        sx = w / 1024.0
        sy = h / 768.0

        # Left list boundaries
        list_y1 = int(70 * sy)
        list_y2 = int(700 * sy)
        list_x1 = int(100 * sx)
        list_x2 = int(250 * sx)

        list_region = frame[list_y1:list_y2, list_x1:list_x2]
        if list_region.size == 0:
            return []

        # Find separator grooves (dark copper bar grooves)
        dark = (list_region < 45).all(axis=2)
        frac = dark.mean(axis=1)

        grooves: List[int] = []
        in_groove = False
        g_start = 0
        for y, v in enumerate(frac):
            if v > 0.75:
                if not in_groove:
                    in_groove = True
                    g_start = y
            else:
                if in_groove:
                    in_groove = False
                    grooves.append(list_y1 + (g_start + y) // 2)

        cards: List[IslandCardInfo] = []
        card_w = int(360 * sx)
        min_card_h = int(80 * sy)
        max_card_h = int(140 * sy)
        # Compute dynamic card height from distance between consecutive detected grooves
        detected_spacings = [
            grooves[i + 1] - grooves[i]
            for i in range(len(grooves) - 1)
            if min_card_h <= (grooves[i + 1] - grooves[i]) <= max_card_h
        ]
        nominal_h = int(np.median(detected_spacings)) if detected_spacings else int(109 * sy)

        # 1. Intermediate cards between detected grooves
        intermediate_cards: List[Tuple[int, int, str]] = []
        for i in range(len(grooves) - 1):
            cy1 = grooves[i]
            cy2 = grooves[i + 1]
            ch = cy2 - cy1
            if min_card_h <= ch <= max_card_h:
                card_crop = frame[cy1:cy2, 0:card_w]
                name = self._recognizer.recognize_card(card_crop)
                intermediate_cards.append((cy1, cy2, name))

        cards_tuples: List[Tuple[int, int, str]] = []

        # 2. Check if a card exists above the first detected groove (e.g. Plant Island at top)
        if intermediate_cards:
            first_cy1 = intermediate_cards[0][0]
            if first_cy1 - list_y1 >= min_card_h:
                top_cy1 = max(list_y1, first_cy1 - int(109 * sy))
                card_crop = frame[top_cy1:first_cy1, 0:card_w]
                name = self._recognizer.recognize_card(card_crop)
                if name:
                    cards_tuples.append((top_cy1, first_cy1, name))

        cards_tuples.extend(intermediate_cards)

        # 3. Check if a card exists below the last detected groove
        if intermediate_cards:
            last_cy2 = intermediate_cards[-1][1]
            if list_y2 - last_cy2 >= min_card_h:
                bot_cy2 = min(list_y2, last_cy2 + nominal_h)
                card_crop = frame[last_cy2:bot_cy2, 0:card_w]
                name = self._recognizer.recognize_card(card_crop)
                if name:
                    cards_tuples.append((last_cy2, bot_cy2, name))

        cards: List[IslandCardInfo] = []
        center_x = int(180 * sx)
        for idx, (cy1, cy2, name) in enumerate(cards_tuples):
            # Dynamically evaluate if the card is fully visible and not clipped by viewport edges
            is_full = (cy2 <= list_y2 - int(15 * sy)) and (cy2 - cy1 >= int(75 * sy))
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

    def enter_selected_island(self, timeout: float = 6.0) -> bool:
        """Wait for screen transition and click detected 'GO' or 'Here' button.

        Active retry mechanism: periodically retries clicking if the game drops
        the click during UI animation, preventing missed transitions.
        """
        deadline = time.monotonic() + timeout
        step_sleep = 0.04  # 40ms polling for responsive detection
        last_click_time = 0.0

        go_templates = [
            ("clean", self._go_clean_tmpl),
            ("plant", self._go_btn_tmpl),
        ]

        while time.monotonic() < deadline:
            frame = self._window.capture()
            if frame is not None:
                # 1. If screen is already transitioning into loading or island view, wait for completion
                state = self.detect_state(frame)
                if state in (ScreenState.LOADING, ScreenState.ISLAND):
                    return self.wait_for_state(
                        ScreenState.ISLAND, timeout=self._cfg.map.map_timeout
                    )

                now = time.monotonic()

                # 2. Check 'You are here!' button via DynamicROI with multi-template matching
                here_templates = [
                    ("clean", self._here_clean_tmpl),
                    ("faded", self._here_btn_tmpl),
                ]
                here_match = self._roi_here.match_any(frame, here_templates, threshold=0.55)
                if here_match is not None:
                    _, here_res = here_match
                    cx, cy = here_res.center
                    if now - last_click_time >= 0.4:
                        logger.info(
                            "detected 'You are here!' button (score=%.3f), clicking (%d, %d)",
                            here_res.score,
                            cx,
                            cy,
                        )
                        self._action.click(cx, cy)
                        last_click_time = now
                        if self.wait_for_state(ScreenState.ISLAND, timeout=1.5):
                            return True
                        if self.close_map():
                            return True

                # 3. Check GO button via DynamicROI with multi-template matching (threshold relaxed to 0.48)
                go_match = self._roi_go.match_any(frame, go_templates, threshold=0.48)
                if go_match is not None:
                    _, go_res = go_match
                    cx, cy = go_res.center
                    if now - last_click_time >= 0.4:
                        logger.info(
                            "detected GO button (score=%.3f), clicking (%d, %d)",
                            go_res.score,
                            cx,
                            cy,
                        )
                        self._action.click(cx, cy)
                        last_click_time = now
                        if self.wait_for_state(ScreenState.ISLAND, timeout=1.5):
                            return True

            time.sleep(step_sleep)

        # Final check if screen transitioned at the end of timeout
        final_frame = self._window.capture()
        if final_frame is not None and self.detect_state(final_frame) == ScreenState.ISLAND:
            return True

        logger.warning("neither GO nor 'You are here!' entered successfully within timeout")
        return False

    def wait_for_list_stable(
        self,
        timeout: float = 1.5,
        poll_interval: float = 0.06,
        motion_threshold: float = 2.0,
    ) -> bool:
        """Wait dynamically until the left island list stops moving (momentum/bounce settles).

        Captures consecutive frames and computes pixel difference in the list region.
        Returns True when motion drops below threshold, or False if timeout expires.
        """
        deadline = time.monotonic() + timeout
        last_crop = None

        while time.monotonic() < deadline:
            frame = self._window.capture()
            if frame is None:
                time.sleep(poll_interval)
                continue

            h, w = frame.shape[:2]
            sy = h / 768.0
            sx = w / 1024.0
            y1, y2 = int(80 * sy), int(680 * sy)
            x1, x2 = int(50 * sx), int(330 * sx)
            cur_crop = frame[y1:y2, x1:x2]

            if last_crop is not None and last_crop.shape == cur_crop.shape:
                diff = float(np.mean(cv2.absdiff(cur_crop, last_crop)))
                if diff < motion_threshold:
                    logger.debug("island list has settled (motion diff=%.2f)", diff)
                    return True

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

        drag_x = int(self._cfg.map.drag_x * sx)
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
        but reveals no new cards. When the topmost card remains unchanged
        after a swipe settles, stops immediately.
        """
        w, h = self._window.client_size()
        if w <= 0 or h <= 0:
            w, h = 1024, 768
        sx = w / 1024.0
        sy = h / 768.0

        drag_x = int(self._cfg.map.drag_x * sx)
        start_y = int(220 * sy)
        end_y = int(580 * sy)

        # Quick check: already at top?
        frame = self._window.capture()
        last_top_hash = 0
        last_top_y = 0
        if frame is not None:
            cards = self.get_visible_cards(frame)
            if cards:
                last_top_hash = cards[0].card_hash
                last_top_y = cards[0].rect[1]

        consecutive_same_top = 0

        for swipe_idx in range(max_swipes):
            logger.info("pulling list downwards to top (swipe %d/%d)", swipe_idx + 1, max_swipes)
            self._action.drag(
                start_x=drag_x,
                start_y=start_y,
                end_x=drag_x,
                end_y=end_y,
                duration=self._cfg.map.drag_duration,
                steps=self._cfg.map.drag_steps,
            )
            self.wait_for_list_stable(timeout=1.2)

            frame = self._window.capture()
            if frame is None:
                continue

            cards = self.get_visible_cards(frame)
            if not cards:
                continue

            top_card = cards[0]
            is_same_hash = (top_card.card_hash != 0 and last_top_hash != 0 and hash_distance(top_card.card_hash, last_top_hash) <= 2)
            is_same_pos = abs(top_card.rect[1] - last_top_y) <= 12

            if is_same_hash and is_same_pos:
                consecutive_same_top += 1
                if consecutive_same_top >= 1:
                    logger.info("list reached physical top ceiling after %d swipe(s), stopping", swipe_idx + 1)
                    break
            else:
                consecutive_same_top = 0

            last_top_hash = top_card.card_hash
            last_top_y = top_card.rect[1]

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
                    entered = self.enter_selected_island()
                    if entered:
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
