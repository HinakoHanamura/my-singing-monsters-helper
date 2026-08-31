"""Scale conversion arithmetic: the foundation of window-size adaptation.

Everything calibrated in reference-resolution pixels flows through these
functions, so a bug here silently mis-places every click.
"""

from __future__ import annotations

import pytest

from core.geometry import (
    aspect_mismatch,
    denorm_rect,
    distance,
    point_in_rect,
    rects_overlap,
    scale_factor,
    scale_length,
)
from tests.synthetic import REFERENCE_SIZE


class TestScaleFactor:
    def test_identity_at_reference_resolution(self):
        assert scale_factor(REFERENCE_SIZE, REFERENCE_SIZE) == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "client_size, expected",
        [
            ((854, 480), 0.5),
            ((1280, 720), 0.75),
            ((2560, 1440), 1.5),
            ((3414, 1920), 2.0),
        ],
    )
    def test_proportional_windows(self, client_size, expected):
        assert scale_factor(client_size, REFERENCE_SIZE) == pytest.approx(
            expected, abs=0.01
        )

    def test_uses_geometric_mean_so_aspect_drift_is_damped(self):
        # Stretched only horizontally: neither pure width nor pure height ratio.
        factor = scale_factor((3414, 960), REFERENCE_SIZE)
        assert factor == pytest.approx(2.0 ** 0.5, abs=0.01)

    @pytest.mark.parametrize("bad", [(0, 960), (1707, 0), (-5, 100)])
    def test_degenerate_sizes_fall_back_to_one(self, bad):
        # A minimized window reports a zero-sized client area; never divide by it.
        assert scale_factor(bad, REFERENCE_SIZE) == 1.0


class TestAspectMismatch:
    def test_zero_for_same_aspect(self):
        assert aspect_mismatch((1280, 720), REFERENCE_SIZE) == pytest.approx(
            0.0, abs=0.01
        )

    def test_positive_when_aspect_differs(self):
        # 4:3 against 16:9 is a large mismatch and should be reported as such.
        assert aspect_mismatch((1024, 768), REFERENCE_SIZE) > 0.2


class TestScaleLength:
    def test_scales_and_rounds(self):
        assert scale_length(24, 0.75) == 18
        assert scale_length(24, 1.5) == 36

    def test_never_returns_below_the_floor(self):
        # A tolerance of zero pixels would make every comparison fail.
        assert scale_length(2, 0.1, minimum=1) == 1

    def test_floor_can_be_zero_for_optional_jitter(self):
        assert scale_length(0, 1.0, minimum=0) == 0


class TestDenormRect:
    def test_ratios_map_to_pixels(self):
        assert denorm_rect((0.0, 0.0, 0.5, 0.25), 1000, 800) == (0, 0, 500, 200)

    def test_same_ratios_track_window_size(self):
        zone = (0.0, 0.86, 1.0, 1.0)
        big = denorm_rect(zone, 1707, 960)
        small = denorm_rect(zone, 1280, 720)
        # The bottom bar stays a bottom bar in both windows.
        assert big[1] / 960 == pytest.approx(small[1] / 720, abs=0.01)

    def test_reversed_coordinates_are_normalised(self):
        assert denorm_rect((0.5, 0.4, 0.1, 0.2), 100, 100) == (10, 20, 50, 40)


class TestRectHelpers:
    def test_point_in_rect_is_half_open(self):
        rect = (10, 10, 20, 20)
        assert point_in_rect((10, 10), rect)
        assert not point_in_rect((20, 15), rect)
        assert not point_in_rect((9, 15), rect)

    def test_overlap_detection(self):
        assert rects_overlap((0, 0, 10, 10), (5, 5, 15, 15))
        assert not rects_overlap((0, 0, 10, 10), (10, 0, 20, 10))

    def test_distance(self):
        assert distance((0, 0), (3, 4)) == pytest.approx(5.0)
