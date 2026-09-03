"""Unit tests for the DynamicROI adaptive tracker and fallback engine."""

from __future__ import annotations

import numpy as np
import pytest

from core.adaptive_roi import DynamicROI, MatchResult


def test_dynamic_roi_initial_state() -> None:
    """Initial state must have NO memorized ROI (forces global scan)."""
    tracker = DynamicROI("test_target", margin_x=50, margin_y=50)
    assert tracker.has_memorized_roi is False
    assert tracker.center is None
    assert tracker.get_crop_bounds((768, 1024, 3)) is None


def test_dynamic_roi_learns_and_tracks() -> None:
    """Tracker learns location on global scan, then uses dynamic ROI."""
    tracker = DynamicROI("test_box", margin_x=40, margin_y=40)

    # Create distinct textured template
    rng = np.random.RandomState(42)
    template = rng.randint(50, 220, (30, 30, 3), dtype=np.uint8)

    canvas = np.zeros((400, 400, 3), dtype=np.uint8)
    canvas[150:180, 150:180] = template

    # 1. First match: no ROI -> global scan succeeds and memorizes center
    res1 = tracker.match(canvas, template, threshold=0.90)
    assert res1 is not None
    assert res1.center == (165, 165)
    assert tracker.has_memorized_roi is True
    assert tracker.center == (165, 165)

    # 2. Second match: fast dynamic ROI should hit immediately
    res2 = tracker.match(canvas, template, threshold=0.90)
    assert res2 is not None
    assert res2.center == (165, 165)

    # 3. Target shifts outside the previous ROI to (320, 320)
    shifted_canvas = np.zeros((400, 400, 3), dtype=np.uint8)
    shifted_canvas[320:350, 320:350] = template

    # Fast ROI will fail, triggering instant global scan fallback
    res3 = tracker.match(shifted_canvas, template, threshold=0.90)
    assert res3 is not None
    assert res3.center == (335, 335)
    assert tracker.center == (335, 335)


def test_dynamic_roi_invalidates_when_target_missing() -> None:
    """When target is completely missing, tracker invalidates memorized ROI."""
    tracker = DynamicROI("disappearing_target")
    tracker.update(center=(100, 100), size=(30, 30))
    assert tracker.has_memorized_roi is True

    rng = np.random.RandomState(42)
    template = rng.randint(50, 220, (30, 30, 3), dtype=np.uint8)
    blank_canvas = np.zeros((300, 300, 3), dtype=np.uint8)

    res = tracker.match(blank_canvas, template, threshold=0.80)
    assert res is None
    assert tracker.has_memorized_roi is False
    assert tracker.center is None


def test_match_any_multiple_templates() -> None:
    """match_any dynamically picks the correct candidate template."""
    tracker = DynamicROI("multi_target")
    canvas = np.zeros((300, 300, 3), dtype=np.uint8)

    rng1 = np.random.RandomState(10)
    rng2 = np.random.RandomState(20)
    t_red = rng1.randint(50, 220, (20, 20, 3), dtype=np.uint8)
    t_blue = rng2.randint(50, 220, (20, 20, 3), dtype=np.uint8)

    # Place blue pattern on canvas
    canvas[80:100, 80:100] = t_blue

    match = tracker.match_any(
        canvas,
        [("red", t_red), ("blue", t_blue)],
        threshold=0.85,
    )
    assert match is not None
    name, res = match
    assert name == "blue"
    assert res.center == (90, 90)
    assert tracker.center == (90, 90)
