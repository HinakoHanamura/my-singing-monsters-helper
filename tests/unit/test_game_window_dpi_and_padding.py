"""Capture geometry: DPI context handling and the black-padding safety net.

Regression tests for a live failure that offline tools could not reproduce.

On a display scaled to 150%, the game (a DPI-unaware application) renders
1024x768 while a DPI-aware caller is told the client area is 1536x1152.
Constructing a QApplication makes the process DPI-aware, so the Qt app saw
1536x1152 frames with the game drawn into the top-left corner and black padding
elsewhere, while the plain capture scripts saw a correct 1024x768. Template
matching then scaled every template by 1.5 and detected nothing.

The primary fix performs the capture in a DPI-unaware thread context. These
tests cover the secondary safety net, which is pure and therefore testable
without a game: detect that padding and trim it, while refusing to trim anything
that merely looks dark.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.game_window import (
    GameWindow,
    content_bounds,
    dpi_unaware_thread,
    strip_black_padding,
)


def padded_frame(surface=(1536, 1152), content=(1024, 768), value=200) -> np.ndarray:
    """A capture surface with content anchored top-left and black elsewhere."""
    frame = np.zeros((surface[1], surface[0], 3), np.uint8)
    frame[0 : content[1], 0 : content[0]] = value
    return frame


class TestContentBounds:
    def test_full_bleed_frame_reports_the_whole_frame(self):
        frame = np.full((768, 1024, 3), 120, np.uint8)
        assert content_bounds(frame) == (0, 0, 1024, 768)

    def test_padded_frame_reports_the_content_box(self):
        assert content_bounds(padded_frame()) == (0, 0, 1024, 768)

    def test_all_black_frame_falls_back_to_the_whole_frame(self):
        """Never hand back an empty box; callers would divide by zero."""
        frame = np.zeros((100, 200, 3), np.uint8)
        assert content_bounds(frame) == (0, 0, 200, 100)

    def test_a_single_lit_channel_counts_as_content(self):
        frame = np.zeros((50, 50, 3), np.uint8)
        frame[10:20, 5:15, 2] = 90  # red only
        assert content_bounds(frame) == (5, 10, 15, 20)

    def test_empty_input_is_handled(self):
        assert content_bounds(np.zeros((0, 0, 3), np.uint8)) == (0, 0, 0, 0)
        assert content_bounds(None) == (0, 0, 0, 0)


class TestStripBlackPadding:
    def test_reproduces_the_observed_failure_and_trims_it(self):
        """The exact geometry seen in the field: 150% scaling on 1024x768."""
        frame = padded_frame()
        out, cropped = strip_black_padding(frame)
        assert cropped
        assert out.shape[:2] == (768, 1024)

    @pytest.mark.parametrize(
        "surface, content",
        [
            ((1536, 1152), (1024, 768)),  # 150%
            ((1280, 960), (1024, 768)),   # 125%
            ((2048, 1536), (1024, 768)),  # 200%
        ],
    )
    def test_trims_at_several_scaling_factors(self, surface, content):
        out, cropped = strip_black_padding(padded_frame(surface, content))
        assert cropped
        assert out.shape[:2] == (content[1], content[0])

    def test_a_correct_capture_is_left_alone(self):
        frame = np.full((768, 1024, 3), 120, np.uint8)
        out, cropped = strip_black_padding(frame)
        assert not cropped
        assert out is frame

    def test_a_dark_scene_is_not_mistaken_for_padding(self):
        """A night-time island is dim everywhere but still fills the frame."""
        frame = np.full((768, 1024, 3), 6, np.uint8)
        out, cropped = strip_black_padding(frame)
        assert not cropped
        assert out.shape[:2] == (768, 1024)

    def test_content_not_anchored_at_the_corner_is_never_trimmed(self):
        """Letterboxing on all sides is a different problem; do not guess at it."""
        frame = np.zeros((1152, 1536, 3), np.uint8)
        frame[100:868, 100:1124] = 200
        out, cropped = strip_black_padding(frame)
        assert not cropped
        assert out.shape[:2] == (1152, 1536)

    def test_a_thin_dark_border_stays_below_the_threshold(self):
        """Trimming a few pixels would shift coordinates for no good reason."""
        frame = np.full((768, 1024, 3), 120, np.uint8)
        frame[:, 1020:] = 0
        frame[762:, :] = 0
        out, cropped = strip_black_padding(frame, min_ratio=0.02)
        assert not cropped

    def test_threshold_is_configurable(self):
        frame = np.full((768, 1024, 3), 120, np.uint8)
        frame[:, 1000:] = 0
        assert not strip_black_padding(frame, min_ratio=0.10)[1]
        assert strip_black_padding(frame, min_ratio=0.01)[1]

    def test_returns_a_copy_so_the_source_buffer_can_be_freed(self):
        frame = padded_frame()
        out, cropped = strip_black_padding(frame)
        assert cropped
        out[0, 0] = 7
        assert frame[0, 0, 0] == 200

    def test_empty_input_is_handled(self):
        empty = np.zeros((0, 0, 3), np.uint8)
        out, cropped = strip_black_padding(empty)
        assert not cropped
        assert out is empty


class TestDpiContext:
    def test_context_manager_always_restores_and_never_raises(self):
        with dpi_unaware_thread() as switched:
            assert isinstance(switched, bool)
        # A second entry must still work, proving the first restored cleanly.
        with dpi_unaware_thread() as switched_again:
            assert isinstance(switched_again, bool)

    def test_exceptions_inside_the_block_still_restore(self):
        with pytest.raises(RuntimeError):
            with dpi_unaware_thread():
                raise RuntimeError("boom")
        with dpi_unaware_thread() as switched:
            assert isinstance(switched, bool)


class TestMissingWindow:
    """Every path must degrade quietly when the game is not running."""

    @pytest.fixture
    def absent(self):
        return GameWindow("__msm_helper_window_that_does_not_exist__")

    def test_attach_reports_failure(self, absent):
        assert absent.attach() is False
        assert absent.hwnd is None

    def test_is_alive_is_false(self, absent):
        assert absent.is_alive() is False

    def test_client_size_is_zero(self, absent):
        assert absent.client_size() == (0, 0)

    def test_capture_returns_none(self, absent):
        assert absent.capture() is None

    def test_is_minimized_is_false_rather_than_raising(self, absent):
        assert absent.is_minimized() is False

    def test_detach_is_idempotent(self, absent):
        absent.detach()
        absent.detach()
        assert absent.hwnd is None
