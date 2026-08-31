"""Template variants and duplicate suppression.

Two behaviours are covered:

* Animated targets need several template variants. Files named ``coin*.png`` are
  all treated as the same logical target, so an animated coin can be described by
  a few cropped animation phases.
* A single coin produces many raw hits (adjacent pixels, several scale steps).
  Greedy suppression must collapse them into one detection, otherwise the guard
  and the click limiter would operate on phantom duplicates.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from core.vision_agent import VisionAgent
from tests.synthetic import (
    REFERENCE_SIZE,
    make_coin_template,
    make_scene,
    write_template,
)


def build_agent(template_dir: str, **kwargs) -> VisionAgent:
    options = dict(
        template_dir=template_dir,
        match_threshold=0.80,
        reference_size=REFERENCE_SIZE,
        use_fake_detection=False,
    )
    options.update(kwargs)
    return VisionAgent(**options)


class TestDuplicateSuppression:
    def test_one_coin_yields_one_detection(self, template_dir):
        scene = make_scene(*REFERENCE_SIZE, coin_centers=[(800, 500)])
        detections = build_agent(template_dir).detect("coin", scene)
        assert len(detections) == 1

    def test_coins_further_apart_than_the_gap_stay_separate(self, template_dir):
        scene = make_scene(*REFERENCE_SIZE, coin_centers=[(800, 500), (900, 500)])
        detections = build_agent(template_dir).detect("coin", scene)
        assert len(detections) == 2

    def test_suppression_keeps_the_strongest_hit(self, template_dir):
        scene = make_scene(*REFERENCE_SIZE, coin_centers=[(800, 500)])
        detections = build_agent(template_dir).detect("coin", scene)
        # An exact paste should win with a near-perfect score.
        assert detections[0].confidence > 0.98

    def test_max_detections_caps_the_output(self, template_dir):
        centers = [(300 + 80 * i, 400) for i in range(10)]
        scene = make_scene(*REFERENCE_SIZE, coin_centers=centers)
        agent = build_agent(template_dir, max_detections=4)
        assert len(agent.detect("coin", scene)) == 4


class TestTemplateVariants:
    def test_all_matching_files_are_loaded_as_one_target(self, tmp_path):
        """coin.png plus coin_phase2.png describe the same logical target."""
        base = make_coin_template()
        # A visibly different second phase: rotated highlight, dimmer body.
        variant = cv2.convertScaleAbs(base, alpha=0.75, beta=10)

        write_template(tmp_path, "coin", base)
        write_template(tmp_path, "coin_phase2", variant)

        agent = build_agent(str(tmp_path))
        scene = make_scene(*REFERENCE_SIZE, coin_centers=[])
        # Paste the *variant* only; the base template alone might miss it.
        from tests.synthetic import paste

        paste(scene, variant, (900, 450))

        detections = agent.detect("coin", scene)
        assert len(detections) == 1
        assert abs(detections[0].center[0] - 900) <= 3
        assert abs(detections[0].center[1] - 450) <= 3

    def test_unrelated_prefix_is_not_picked_up(self, tmp_path):
        write_template(tmp_path, "cloud", make_coin_template())
        agent = build_agent(str(tmp_path))
        scene = make_scene(*REFERENCE_SIZE, coin_centers=[(900, 450)])
        # Asking for "coin" must not silently match cloud.png.
        assert agent.detect("coin", scene) == []


class TestThresholdBehaviour:
    def test_higher_threshold_rejects_a_degraded_target(self, template_dir):
        scene = make_scene(*REFERENCE_SIZE, coin_centers=[(900, 450)])
        # Heavy blur destroys the fine structure the template relies on.
        degraded = cv2.GaussianBlur(scene, (0, 0), 4.0)

        lenient = build_agent(template_dir, match_threshold=0.50)
        strict = build_agent(template_dir, match_threshold=0.99)

        assert len(strict.detect("coin", degraded)) <= len(
            lenient.detect("coin", degraded)
        )

    def test_no_false_positive_on_an_empty_background(self, template_dir):
        empty = make_scene(*REFERENCE_SIZE, coin_centers=[])
        assert build_agent(template_dir).detect("coin", empty) == []


class TestFakeDetectionMode:
    """The placeholder mode used before real templates exist."""

    def test_fake_mode_ignores_templates_entirely(self, tmp_path):
        agent = VisionAgent(
            template_dir=str(tmp_path),  # empty
            reference_size=REFERENCE_SIZE,
            use_fake_detection=True,
            simulate_miss_rate=0.0,
        )
        frame = np.zeros((720, 1280, 3), np.uint8)
        assert agent.detect("coin", frame)

    def test_miss_rate_one_always_returns_nothing(self, tmp_path):
        agent = VisionAgent(
            template_dir=str(tmp_path),
            use_fake_detection=True,
            simulate_miss_rate=1.0,
        )
        frame = np.zeros((720, 1280, 3), np.uint8)
        assert agent.detect("coin", frame) == []

    @pytest.mark.parametrize("size", [(1280, 720), (1707, 960), (2560, 1440)])
    def test_fake_boxes_stay_inside_the_frame(self, tmp_path, size):
        agent = VisionAgent(
            template_dir=str(tmp_path),
            reference_size=REFERENCE_SIZE,
            use_fake_detection=True,
            simulate_miss_rate=0.0,
        )
        width, height = size
        frame = np.zeros((height, width, 3), np.uint8)
        for _ in range(40):
            for detection in agent.detect("coin", frame):
                cx, cy = detection.center
                assert 0 <= cx < width
                assert 0 <= cy < height
