"""Unit tests for island entry transition, OCR alias cleanup, and navigation guardrails."""

from __future__ import annotations

import os
from typing import List, Optional, Tuple
import cv2
import numpy as np
import pytest

from config import DEFAULT_CONFIG, PROJECT_ROOT
from core.action_agent import ActionAgent, WindowHandleProvider
from core.letter_recognizer import LetterRecognizer
from core.map_navigator import MapNavigator, ScreenState
from tests.unit.test_map_navigator import MockAction, MockWindow


def test_clean_title_tokens_dynamic_normalization() -> None:
    """Verify clean_title_tokens cleanly normalizes tokens without hardcoded alias tuples or 'island' stripping."""
    lr = LetterRecognizer()

    # Standard cases (full island title preserved)
    assert lr.clean_title_tokens("Plant Island") == "plant island"
    assert lr.clean_title_tokens("Cold Island") == "cold island"
    assert lr.clean_title_tokens("Air Island") == "air island"
    assert lr.clean_title_tokens("Water Island") == "water island"
    assert lr.clean_title_tokens("Earth Island") == "earth island"
    assert lr.clean_title_tokens("Fire Haven") == "fire haven"
    assert lr.clean_title_tokens("Fire Oasis") == "fire oasis"
    assert lr.clean_title_tokens("The Colossingum") == "the colossingum"

    # Fuzzy matching handles OCR typo variations dynamically
    assert lr.names_fuzzy_match("Plant lsland", "Plant Island") is True
    assert lr.names_fuzzy_match("Cold lslond", "Cold Island") is True
    assert lr.names_fuzzy_match("Air lsiond", "Air Island") is True
    assert lr.names_fuzzy_match("Water iuland", "Water Island") is True
    assert lr.names_fuzzy_match("Earth 1sland", "Earth Island") is True
    assert lr.names_fuzzy_match("Gold Islan", "Gold Island") is True
    assert lr.names_fuzzy_match("Fire Haven", "Fire Oasis") is False
    assert lr.names_fuzzy_match("Cold Island", "Gold Island") is False



def test_detect_state_1080p_and_widescreen_ocr_fallback() -> None:
    """Verify detect_state detects ScreenState.ISLAND on 1920x1080 widescreen even with theme/transparency variations."""
    fpath = os.path.join(PROJECT_ROOT, "captures", "frame_20260829_124217_421.png")
    if not os.path.isfile(fpath):
        fpath = os.path.join(PROJECT_ROOT, "captures", "map", "map_20260902_185628_230.png")
    if not os.path.isfile(fpath):
        pytest.skip("Island frame not found")

    frame = cv2.imread(fpath)
    # Resize to full 1920x1080
    frame_1080p = cv2.resize(frame, (1920, 1080))
    window = MockWindow(frame=frame_1080p)
    window._client_size = (1920, 1080)
    action = MockAction(window=window)
    nav = MapNavigator(action_agent=action, window=window, config=DEFAULT_CONFIG)

    state = nav.detect_state(frame_1080p)
    assert state == ScreenState.ISLAND


def test_enter_selected_island_handles_extended_loading() -> None:
    """Verify enter_selected_island does not prematurely time out after 1.5s when loading takes 3s."""
    go_path = os.path.join(PROJECT_ROOT, "captures", "map", "map_20260902_185641_307.png")
    loading_path = os.path.join(PROJECT_ROOT, "captures", "map", "map_20260902_185635_276.png")
    island_path = os.path.join(PROJECT_ROOT, "captures", "map", "map_20260902_185628_230.png")

    if not (os.path.isfile(go_path) and os.path.isfile(loading_path) and os.path.isfile(island_path)):
        pytest.skip("Required frames not found")

    go_frame = cv2.imread(go_path)
    loading_frame = cv2.imread(loading_path)
    island_frame = cv2.imread(island_path)

    frame_sequence = [go_frame] + [loading_frame] * 3 + [island_frame] * 10
    seq_idx = 0

    class SequenceWindow(MockWindow):
        def capture(self) -> Optional[np.ndarray]:
            nonlocal seq_idx
            frame = frame_sequence[min(seq_idx, len(frame_sequence) - 1)]
            seq_idx += 1
            return frame

    window = SequenceWindow()
    action = MockAction(window=window)
    nav = MapNavigator(action_agent=action, window=window, config=DEFAULT_CONFIG)

    entered = nav.enter_selected_island(timeout=6.0)
    assert entered is True
    assert len(action.clicks) >= 1
