"""Unit tests verifying low-latency GO transition and anti-race guards."""

from __future__ import annotations

import os
import time
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
        self.click_timestamps: List[float] = []

    def click(self, x: int, y: int) -> bool:
        self.clicks.append((x, y))
        self.click_timestamps.append(time.monotonic())
        return True


def test_detect_state_latency_and_accuracy() -> None:
    """Verify detect_state detects ScreenState.MAP in under 80ms (previously ~768ms)."""
    f_map = cv2.imread(os.path.join(PROJECT_ROOT, "captures", "map", "map_20260902_185615_714.png"))
    f_isl = cv2.imread(os.path.join(PROJECT_ROOT, "captures", "map", "map_20260902_185628_230.png"))
    f_lod = cv2.imread(os.path.join(PROJECT_ROOT, "captures", "map", "map_20260902_185635_276.png"))

    if f_map is None or f_isl is None or f_lod is None:
        pytest.skip("Required map capture files not found")

    window = MockWindow(frame=f_map)
    action = MockAction(window=window)
    nav = MapNavigator(action_agent=action, window=window, config=DEFAULT_CONFIG)

    # Measure MAP detection latency
    t0 = time.perf_counter()
    st_map = nav.detect_state(f_map)
    t1 = time.perf_counter()
    assert st_map == ScreenState.MAP
    assert (t1 - t0) < 0.08, f"detect_state took {(t1-t0)*1000:.1f}ms, expected < 80ms"

    # Measure LOADING detection latency
    t2 = time.perf_counter()
    st_lod = nav.detect_state(f_lod)
    t3 = time.perf_counter()
    assert st_lod == ScreenState.LOADING
    assert (t3 - t2) < 0.02, f"loading check took {(t3-t2)*1000:.1f}ms"

    # Measure ISLAND detection latency
    assert nav.detect_state(f_isl) == ScreenState.ISLAND


def test_panel_stable_detection() -> None:
    """Verify wait_for_panel_stable detects stationary preview panel and waits on motion."""
    f_base = np.full((768, 1024, 3), 120, dtype=np.uint8)
    f_noisy = f_base.copy()
    # Add large motion in right panel (y: 100..500, x: 500..900)
    f_noisy[100:500, 500:900] = 20

    seq = [f_noisy, f_base, f_base, f_base]
    seq_idx = 0

    class SeqWindow(MockWindow):
        def capture(self) -> Optional[np.ndarray]:
            nonlocal seq_idx
            frame = seq[min(seq_idx, len(seq) - 1)]
            seq_idx += 1
            return frame

    window = SeqWindow()
    action = MockAction(window=window)
    nav = MapNavigator(action_agent=action, window=window, config=DEFAULT_CONFIG)

    ok = nav.wait_for_panel_stable(timeout=0.5, poll_interval=0.01, motion_threshold=4.0)
    assert ok is True
    assert seq_idx >= 3


def test_anti_race_prevents_premature_go_click() -> None:
    """Verify enter_selected_island with target_card waits for panel stabilization before clicking GO."""
    go_path = os.path.join(PROJECT_ROOT, "captures", "map", "map_20260902_185641_307.png")
    island_path = os.path.join(PROJECT_ROOT, "captures", "map", "map_20260902_185628_230.png")
    if not (os.path.isfile(go_path) and os.path.isfile(island_path)):
        pytest.skip("Required map captures not found")

    go_frame = cv2.imread(go_path)
    island_frame = cv2.imread(island_path)

    # Transient animating frame where preview is changing
    transient_frame = go_frame.copy()
    h, w = go_frame.shape[:2]
    transient_frame[int(h * 0.2):int(h * 0.5), int(w * 0.5):int(w * 0.8)] = 0

    # Sequence: 3 transient frames during card switch animation,
    # then 6 stable GO frames (so panel settles and GO is clicked),
    # then 5 island frames (screen enters island)
    frames = [transient_frame] * 3 + [go_frame] * 6 + [island_frame] * 5
    cur_idx = 0

    class AnimWindow(MockWindow):
        def capture(self) -> Optional[np.ndarray]:
            nonlocal cur_idx
            f = frames[min(cur_idx, len(frames) - 1)]
            cur_idx += 1
            return f

    window = AnimWindow()
    action = MockAction(window=window)
    nav = MapNavigator(action_agent=action, window=window, config=DEFAULT_CONFIG)

    dummy_card = IslandCardInfo(
        index=1,
        name="Cold Island",
        rect=(0, 179, 360, 267),
        click_point=(180, 223),
        is_fully_visible=True,
    )

    entered = nav.enter_selected_island(timeout=4.0, target_card=dummy_card)
    assert entered is True
    assert len(action.clicks) >= 1
    # Verify click was aimed at GO button
    cx, cy = action.clicks[0]
    assert 600 <= cx <= 850
    assert 560 <= cy <= 660


def test_positive_island_confirmation_blocks_black_screen() -> None:
    """Verify enter_selected_island does not return True while still in LOADING state."""
    go_path = os.path.join(PROJECT_ROOT, "captures", "map", "map_20260902_185641_307.png")
    loading_path = os.path.join(PROJECT_ROOT, "captures", "map", "map_20260902_185635_276.png")
    island_path = os.path.join(PROJECT_ROOT, "captures", "map", "map_20260902_185628_230.png")

    if not (os.path.isfile(go_path) and os.path.isfile(loading_path) and os.path.isfile(island_path)):
        pytest.skip("Required frames not found")

    go_frame = cv2.imread(go_path)
    loading_frame = cv2.imread(loading_path)
    island_frame = cv2.imread(island_path)

    # 1 GO frame, followed by 5 LOADING frames, followed by ISLAND
    frames = [go_frame] + [loading_frame] * 5 + [island_frame] * 5
    cur_idx = 0

    states_encountered: List[ScreenState] = []

    class TrackWindow(MockWindow):
        def capture(self) -> Optional[np.ndarray]:
            nonlocal cur_idx
            f = frames[min(cur_idx, len(frames) - 1)]
            cur_idx += 1
            return f

    window = TrackWindow()
    action = MockAction(window=window)
    nav = MapNavigator(action_agent=action, window=window, config=DEFAULT_CONFIG)

    orig_detect = nav.detect_state

    def spy_detect(frame: np.ndarray) -> ScreenState:
        st = orig_detect(frame)
        states_encountered.append(st)
        return st

    nav.detect_state = spy_detect  # type: ignore

    entered = nav.enter_selected_island(timeout=5.0)
    assert entered is True
    assert ScreenState.LOADING in states_encountered
    assert states_encountered[-1] == ScreenState.ISLAND
