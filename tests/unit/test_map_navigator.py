"""Unit tests for the MapNavigator state machine and card detection."""

from __future__ import annotations

import os
from typing import List, Optional, Tuple
import cv2
import numpy as np
import pytest

from config import DEFAULT_CONFIG, PROJECT_ROOT
from core.action_agent import ActionAgent, WindowHandleProvider
from core.map_navigator import IslandCardInfo, MapNavigator, ScreenState


class MockWindow(WindowHandleProvider):
    def __init__(self, frame: Optional[np.ndarray] = None) -> None:
        self._frame = frame
        self._client_size = (1024, 768)

    @property
    def hwnd(self) -> Optional[int]:
        return 12345

    def ensure_attached(self) -> bool:
        return True

    def client_size(self) -> Tuple[int, int]:
        return self._client_size

    def capture(self) -> Optional[np.ndarray]:
        return self._frame


class MockAction(ActionAgent):
    def __init__(self, window: WindowHandleProvider) -> None:
        super().__init__(window=window)
        self.clicks: List[Tuple[int, int]] = []
        self.drags: List[Tuple[int, int, int, int]] = []

    def click(self, x: int, y: int) -> bool:
        self.clicks.append((x, y))
        return True

    def drag(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration: float = 0.25,
        steps: int = 10,
    ) -> bool:
        self.drags.append((start_x, start_y, end_x, end_y))
        return True


def test_detect_state_map_frame() -> None:
    """Map capture frame should detect ScreenState.MAP."""
    fpath = os.path.join(PROJECT_ROOT, "captures", "map", "map_20260902_185615_714.png")
    if not os.path.isfile(fpath):
        pytest.skip("captures/map frame not found")

    frame = cv2.imread(fpath)
    window = MockWindow(frame=frame)
    action = MockAction(window=window)
    nav = MapNavigator(action_agent=action, window=window, config=DEFAULT_CONFIG)

    assert nav.detect_state(frame) == ScreenState.MAP


def test_detect_state_island_frame() -> None:
    """Island capture frame should detect ScreenState.ISLAND."""
    fpath = os.path.join(PROJECT_ROOT, "captures", "map", "map_20260902_185628_230.png")
    if not os.path.isfile(fpath):
        pytest.skip("captures/map frame not found")

    frame = cv2.imread(fpath)
    window = MockWindow(frame=frame)
    action = MockAction(window=window)
    nav = MapNavigator(action_agent=action, window=window, config=DEFAULT_CONFIG)

    assert nav.detect_state(frame) == ScreenState.ISLAND


def test_detect_state_loading_frame() -> None:
    """Transition/loading frame should detect ScreenState.LOADING."""
    fpath = os.path.join(PROJECT_ROOT, "captures", "map", "map_20260902_185635_276.png")
    if not os.path.isfile(fpath):
        pytest.skip("captures/map frame not found")

    frame = cv2.imread(fpath)
    window = MockWindow(frame=frame)
    action = MockAction(window=window)
    nav = MapNavigator(action_agent=action, window=window, config=DEFAULT_CONFIG)

    assert nav.detect_state(frame) == ScreenState.LOADING


def test_get_visible_cards() -> None:
    """Verify island cards are detected with valid names and click targets."""
    fpath = os.path.join(PROJECT_ROOT, "captures", "map", "map_20260902_185615_714.png")
    if not os.path.isfile(fpath):
        pytest.skip("captures/map frame not found")

    frame = cv2.imread(fpath)
    window = MockWindow(frame=frame)
    action = MockAction(window=window)
    nav = MapNavigator(action_agent=action, window=window, config=DEFAULT_CONFIG)

    cards = nav.get_visible_cards(frame)
    assert len(cards) >= 5
    assert cards[0].name == "Plant Island"
    assert cards[1].name == "Cold Island"
    assert cards[0].card_hash != 0

    for card in cards:
        assert isinstance(card.name, str)
        assert len(card.name) > 0
        cx, cy = card.click_point
        assert 0 <= cx < 1024
        assert 0 <= cy < 768


def test_select_island_and_scroll() -> None:
    """Verify select_island and scroll_down invoke action agent."""
    window = MockWindow()
    action = MockAction(window=window)
    nav = MapNavigator(action_agent=action, window=window, config=DEFAULT_CONFIG)

    card = IslandCardInfo(index=0, name="Cold Island", rect=(0, 100, 360, 200), click_point=(180, 150))
    nav.select_island(card)
    assert (180, 150) in action.clicks

    nav.scroll_down()
    assert len(action.drags) == 1
    sx, sy, ex, ey = action.drags[0]
    assert sx == ex == 200
    assert sy > ey  # Drag upwards


def test_scroll_to_top() -> None:
    """Verify scroll_to_top drags downwards to reset list."""
    window = MockWindow()
    action = MockAction(window=window)
    nav = MapNavigator(action_agent=action, window=window, config=DEFAULT_CONFIG)

    nav.scroll_to_top(max_swipes=2)
    assert len(action.drags) == 2
    for sx, sy, ex, ey in action.drags:
        assert sx == ex == 200
        assert sy < ey  # Drag downwards to scroll to top


def test_scroll_to_top_stops_when_already_at_top() -> None:
    """Verify scroll_to_top detects top ceiling dynamically via overscroll invariance."""
    fpath = os.path.join(PROJECT_ROOT, "captures", "map", "map_20260902_185615_714.png")
    if not os.path.isfile(fpath):
        pytest.skip("captures/map frame not found")

    frame = cv2.imread(fpath)
    window = MockWindow(frame=frame)
    action = MockAction(window=window)
    nav = MapNavigator(action_agent=action, window=window, config=DEFAULT_CONFIG)

    nav.scroll_to_top(max_swipes=4)
    # The list is already at the physical top, so it terminates promptly without spinning
    assert len(action.drags) <= 1


def test_enter_selected_island_on_you_are_here_frame() -> None:
    """On frame displaying 'You are here!' button, enter_selected_island detects and clicks it."""
    fpath = os.path.join(PROJECT_ROOT, "captures", "map", "map_20260902_185621_219.png")
    if not os.path.isfile(fpath):
        pytest.skip("YOU ARE HERE frame not found")

    frame = cv2.imread(fpath)
    window = MockWindow(frame=frame)
    action = MockAction(window=window)
    nav = MapNavigator(action_agent=action, window=window, config=DEFAULT_CONFIG)

    nav.wait_for_state = lambda state, timeout=6.0: True  # type: ignore
    ok = nav.enter_selected_island()
    assert ok
    # Verified: it clicked the 'You are here!' button
    assert len(action.clicks) >= 1
    cx, cy = action.clicks[0]
    assert 600 <= cx <= 850
    assert 550 <= cy <= 650


def test_enter_selected_island_on_go_frame() -> None:
    """On frame displaying GO button, enter_selected_island clicks GO."""
    fpath = os.path.join(PROJECT_ROOT, "captures", "map", "map_20260902_185641_307.png")
    if not os.path.isfile(fpath):
        pytest.skip("GO frame not found")

    frame = cv2.imread(fpath)
    window = MockWindow(frame=frame)
    action = MockAction(window=window)
    nav = MapNavigator(action_agent=action, window=window, config=DEFAULT_CONFIG)

    # Fake state transition after click to avoid long timeout
    def fake_wait(state: ScreenState, timeout: float = 6.0) -> bool:
        return True
    nav.wait_for_state = fake_wait  # type: ignore

    entered = nav.enter_selected_island(timeout=1.0)
    assert entered is True
    assert len(action.clicks) >= 1
    cx, cy = action.clicks[0]
    # Center of GO button is around x=730, y=615
    assert 600 <= cx <= 850
    assert 560 <= cy <= 660


def test_enter_selected_island_on_shifted_go_frame() -> None:
    """Verify GO button is recognized even when badges are absent and GO shifts up (e.g. y=557)."""
    fpath = os.path.join(PROJECT_ROOT, "captures", "map", "map_20260902_185722_531.png")
    if not os.path.isfile(fpath):
        pytest.skip("Shifted GO frame not found")

    frame = cv2.imread(fpath)
    window = MockWindow(frame=frame)
    action = MockAction(window=window)
    nav = MapNavigator(action_agent=action, window=window, config=DEFAULT_CONFIG)

    def fake_wait(state: ScreenState, timeout: float = 6.0) -> bool:
        return True
    nav.wait_for_state = fake_wait  # type: ignore

    entered = nav.enter_selected_island(timeout=1.0)
    assert entered is True
    assert len(action.clicks) >= 1
    cx, cy = action.clicks[0]
    # Center of upward-shifted GO button is around y=579
    assert 650 <= cx <= 800
    assert 540 <= cy <= 610


def test_enter_selected_island_on_clean_go_frame() -> None:
    """Verify GO button is recognized on dark-theme frames (clean white GO)."""
    fpath = os.path.join(PROJECT_ROOT, "captures", "map", "map_20260902_185737_937.png")
    if not os.path.isfile(fpath):
        pytest.skip("Clean GO frame not found")

    frame = cv2.imread(fpath)
    window = MockWindow(frame=frame)
    action = MockAction(window=window)
    nav = MapNavigator(action_agent=action, window=window, config=DEFAULT_CONFIG)

    def fake_wait(state: ScreenState, timeout: float = 6.0) -> bool:
        return True
    nav.wait_for_state = fake_wait  # type: ignore

    entered = nav.enter_selected_island(timeout=1.0)
    assert entered is True
    assert len(action.clicks) >= 1
    cx, cy = action.clicks[0]
    assert 650 <= cx <= 800
    assert 560 <= cy <= 660


def test_wait_for_list_stable() -> None:
    """Verify wait_for_list_stable returns True on static frames."""
    fpath = os.path.join(PROJECT_ROOT, "captures", "map", "map_20260902_185615_714.png")
    if not os.path.isfile(fpath):
        pytest.skip("Static frame not found")

    frame = cv2.imread(fpath)
    window = MockWindow(frame=frame)
    action = MockAction(window=window)
    nav = MapNavigator(action_agent=action, window=window, config=DEFAULT_CONFIG)

    # Static frame should stabilize immediately
    assert nav.wait_for_list_stable(timeout=0.5, poll_interval=0.02) is True


def test_card_is_fully_visible_distinguishes_clipped_bottom_cards() -> None:
    """Verify is_fully_visible correctly marks unclipped cards as True and clipped cards as False."""
    fpath = os.path.join(PROJECT_ROOT, "captures", "map", "map_20260902_185615_714.png")
    if not os.path.isfile(fpath):
        pytest.skip("Map frame not found")

    frame = cv2.imread(fpath)
    window = MockWindow(frame=frame)
    action = MockAction(window=window)
    nav = MapNavigator(action_agent=action, window=window, config=DEFAULT_CONFIG)

    cards = nav.get_visible_cards(frame)
    assert len(cards) >= 5
    # First 5 cards (0..4) are fully bounded by grooves
    for c in cards[:5]:
        assert c.is_fully_visible is True
    # The 6th card (index 5) is partially clipped by the bottom border
    if len(cards) >= 6:
        assert cards[5].is_fully_visible is False


def test_1080p_resolution_adaptation_and_open_map() -> None:
    """Verify state detection and open_map function accurately under 1080p scaling."""
    island_path = os.path.join(PROJECT_ROOT, "captures", "map", "map_20260902_185628_230.png")
    map_path = os.path.join(PROJECT_ROOT, "captures", "map", "map_20260902_185615_714.png")
    if not (os.path.isfile(island_path) and os.path.isfile(map_path)):
        pytest.skip("Capture frames not found")

    island_frame = cv2.resize(cv2.imread(island_path), (1920, 1080))
    map_frame = cv2.resize(cv2.imread(map_path), (1920, 1080))

    window = MockWindow(frame=island_frame)
    window._client_size = (1920, 1080)
    action = MockAction(window=window)
    nav = MapNavigator(action_agent=action, window=window, config=DEFAULT_CONFIG)

    # 1. 1080p Island frame should detect ScreenState.ISLAND
    assert nav.detect_state(island_frame) == ScreenState.ISLAND

    # 2. open_map on 1080p should detect the scaled MAP button and click it
    # Mock transition to MAP on click
    def fake_wait(state: ScreenState, timeout: float = 6.0) -> bool:
        return True
    nav.wait_for_state = fake_wait  # type: ignore

    opened = nav.open_map(timeout=1.0)
    assert opened is True
    assert len(action.clicks) >= 1
    click_x, click_y = action.clicks[0]
    # Button center on 1080p is approx (935, 977)
    assert 850 <= click_x <= 1000
    assert 900 <= click_y <= 1050

    # 3. 1080p Map frame should detect ScreenState.MAP directly
    assert nav.detect_state(map_frame) == ScreenState.MAP


def test_scroll_to_top_deceleration_and_stationary_comparison() -> None:
    """Verify scroll_to_top continues when content changes, and stops when settled content is identical."""
    mid_path = os.path.join(PROJECT_ROOT, "captures", "map", "map_20260902_185624_703.png")
    top_path = os.path.join(PROJECT_ROOT, "captures", "map", "map_20260902_185615_714.png")
    if not (os.path.isfile(mid_path) and os.path.isfile(top_path)):
        pytest.skip("Capture frames not found")

    mid_frame = cv2.imread(mid_path)
    top_frame = cv2.imread(top_path)

    # Frame sequence: starts at mid, swipe 1 transitions to top, swipe 2 remains at top
    frames = [mid_frame, top_frame, top_frame]
    current_idx = 0

    class SequenceWindow(MockWindow):
        def capture(self) -> Optional[np.ndarray]:
            return frames[min(current_idx, len(frames) - 1)]

    window = SequenceWindow()
    action = MockAction(window=window)

    def on_drag(start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 0.25, steps: int = 10) -> bool:
        nonlocal current_idx
        action.drags.append((start_x, start_y, end_x, end_y))
        current_idx += 1
        return True

    action.drag = on_drag  # type: ignore
    nav = MapNavigator(action_agent=action, window=window, config=DEFAULT_CONFIG)

    ok = nav.scroll_to_top(max_swipes=5)
    assert ok is True
    # Swipe 1: mid -> top (content changed -> continues)
    # Swipe 2: top -> top (settled content identical -> stops)
    assert len(action.drags) == 2
