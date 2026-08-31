"""Window-size adaptation of template matching.

cv2.matchTemplate has no scale invariance, so a template cropped at one window
size stops matching at another. The vision layer compensates by scaling the
template by the window ratio before matching.

These tests are the regression guard for that behaviour: one template, several
window sizes, and the detected centre must land on the correctly scaled position
every time. They are what makes "resize the game window freely" a claim rather
than a hope.
"""

from __future__ import annotations

import pytest

from core.geometry import scale_factor
from core.vision_agent import VisionAgent
from tests.synthetic import (
    COIN_SIZE,
    REFERENCE_SIZE,
    make_scene,
    render_at,
    scale_point,
)

COIN_A = (900, 400)
COIN_B = (1300, 620)


def build_agent(template_dir: str) -> VisionAgent:
    return VisionAgent(
        template_dir=template_dir,
        match_threshold=0.80,
        reference_size=REFERENCE_SIZE,
        use_fake_detection=False,
    )


@pytest.fixture
def scene():
    return make_scene(
        REFERENCE_SIZE[0], REFERENCE_SIZE[1], coin_centers=[COIN_A, COIN_B]
    )


class TestNativeResolution:
    def test_finds_every_coin_exactly_once(self, template_dir, scene):
        detections = build_agent(template_dir).detect("coin", scene)
        # Two coins, and de-duplication collapsed the multi-scale hits.
        assert len(detections) == 2

    def test_centres_are_pixel_accurate(self, template_dir, scene):
        detections = build_agent(template_dir).detect("coin", scene)
        found = sorted(d.center for d in detections)
        for got, want in zip(found, sorted([COIN_A, COIN_B])):
            assert abs(got[0] - want[0]) <= 3
            assert abs(got[1] - want[1]) <= 3

    def test_confidence_is_high_for_an_exact_paste(self, template_dir, scene):
        detections = build_agent(template_dir).detect("coin", scene)
        assert all(d.confidence > 0.95 for d in detections)

    def test_box_matches_template_size(self, template_dir, scene):
        detections = build_agent(template_dir).detect("coin", scene)
        assert all(d.width == COIN_SIZE and d.height == COIN_SIZE for d in detections)

    def test_results_are_sorted_by_confidence(self, template_dir, scene):
        detections = build_agent(template_dir).detect("coin", scene)
        scores = [d.confidence for d in detections]
        assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize(
    "window",
    [
        (1280, 720),   # 0.75x, the common smaller window
        (2560, 1440),  # 1.5x, a high-DPI or maximised window
        (1024, 576),   # 0.6x, small window
        (1600, 900),   # non-integer ratio
        (854, 480),    # 0.5x, extreme downscale
    ],
)
class TestResizedWindows:
    """One template, many window sizes. This is the adaptation contract."""

    def test_all_coins_still_found(self, template_dir, scene, window):
        rendered = render_at(scene, window)
        detections = build_agent(template_dir).detect("coin", rendered)
        assert len(detections) == 2

    def test_centres_track_the_window_ratio(self, template_dir, scene, window):
        rendered = render_at(scene, window)
        detections = build_agent(template_dir).detect("coin", rendered)

        found = sorted(d.center for d in detections)
        expected = sorted(
            scale_point(c, REFERENCE_SIZE, window) for c in (COIN_A, COIN_B)
        )
        # Tolerance grows with downscaling: fewer pixels means coarser localisation.
        factor = scale_factor(window, REFERENCE_SIZE)
        tolerance = max(6, int(round(8 * factor)) + 4)

        for got, want in zip(found, expected):
            assert abs(got[0] - want[0]) <= tolerance, f"{found} vs {expected}"
            assert abs(got[1] - want[1]) <= tolerance, f"{found} vs {expected}"

    def test_box_size_scales_with_the_window(self, template_dir, scene, window):
        rendered = render_at(scene, window)
        detections = build_agent(template_dir).detect("coin", rendered)
        expected = int(round(COIN_SIZE * scale_factor(window, REFERENCE_SIZE)))
        assert all(abs(d.width - expected) <= 3 for d in detections)


class TestDegenerateInputs:
    def test_empty_frame_returns_nothing(self, template_dir):
        import numpy as np

        agent = build_agent(template_dir)
        assert agent.detect("coin", np.zeros((0, 0, 3), np.uint8)) == []

    def test_none_frame_returns_nothing(self, template_dir):
        assert build_agent(template_dir).detect("coin", None) == []

    def test_missing_template_returns_nothing_instead_of_raising(self, tmp_path, scene):
        agent = VisionAgent(
            template_dir=str(tmp_path),  # empty directory
            reference_size=REFERENCE_SIZE,
            use_fake_detection=False,
        )
        assert agent.detect("coin", scene) == []

    def test_frame_smaller_than_template_is_handled(self, template_dir):
        import numpy as np

        tiny = np.zeros((10, 10, 3), np.uint8)
        assert build_agent(template_dir).detect("coin", tiny) == []


class TestBackwardCompatibleHelpers:
    """find_target / find_all_targets are the simple wrappers over detect()."""

    def test_find_target_returns_the_best_centre(self, template_dir, scene):
        agent = build_agent(template_dir)
        best = agent.find_target("coin", scene)
        assert best in (COIN_A, COIN_B) or any(
            abs(best[0] - c[0]) <= 3 and abs(best[1] - c[1]) <= 3
            for c in (COIN_A, COIN_B)
        )

    def test_find_target_returns_none_when_absent(self, template_dir):
        empty = make_scene(REFERENCE_SIZE[0], REFERENCE_SIZE[1], coin_centers=[])
        assert build_agent(template_dir).find_target("coin", empty) is None

    def test_find_all_targets_returns_every_centre(self, template_dir, scene):
        points = build_agent(template_dir).find_all_targets("coin", scene)
        assert len(points) == 2
