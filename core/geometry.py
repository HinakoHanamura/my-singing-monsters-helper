"""Geometry and scale conversion helpers.

Everything that makes the bot adapt to the window size lives here. The config
file stores either *pixels measured at the reference resolution* or *normalised
ratios*; at run time every one of those values passes through this module to
become actual pixels in the current client area.

The payoff is that calibration happens once. However large the player drags the
window, whatever resolution they pick, thresholds and regions follow along --
there is no per-size parameter set to maintain.
"""

from __future__ import annotations

import math
from typing import Tuple

# Pixel rectangle (x1, y1, x2, y2); the bottom-right corner is exclusive.
PixelRect = Tuple[int, int, int, int]
Point = Tuple[int, int]


def scale_factor(
    client_size: Tuple[int, int], reference_size: Tuple[int, int]
) -> float:
    """Scale of the current window relative to the reference resolution.

    This is the geometric mean of the width and height ratios. Using width alone
    skews as soon as the window's aspect ratio drifts from the reference; the
    geometric mean holds up better under a slight aspect change.
    """
    cw, ch = client_size
    rw, rh = reference_size
    if cw <= 0 or ch <= 0 or rw <= 0 or rh <= 0:
        return 1.0
    return math.sqrt((cw / rw) * (ch / rh))


def aspect_mismatch(
    client_size: Tuple[int, int], reference_size: Tuple[int, int]
) -> float:
    """How far the aspect ratio deviates; 0 means an exact match.

    Template matching degrades noticeably once this grows large.
    """
    cw, ch = client_size
    rw, rh = reference_size
    if min(cw, ch, rw, rh) <= 0:
        return 0.0
    return abs((cw / ch) / (rw / rh) - 1.0)


def scale_length(value: float, factor: float, minimum: int = 1) -> int:
    """Convert a reference-resolution length to the current window.

    The result is never smaller than ``minimum``.
    """
    return max(minimum, int(round(value * factor)))


def denorm_rect(
    rect: Tuple[float, float, float, float], width: int, height: int
) -> PixelRect:
    """Normalised rectangle -> pixel rectangle in the current window."""
    x1, y1, x2, y2 = rect
    px1 = int(round(x1 * width))
    py1 = int(round(y1 * height))
    px2 = int(round(x2 * width))
    py2 = int(round(y2 * height))
    # Tolerate a config that lists the corners the other way round.
    if px1 > px2:
        px1, px2 = px2, px1
    if py1 > py2:
        py1, py2 = py2, py1
    return px1, py1, px2, py2


def point_in_rect(point: Point, rect: PixelRect) -> bool:
    x, y = point
    x1, y1, x2, y2 = rect
    return x1 <= x < x2 and y1 <= y < y2


def rects_overlap(a: PixelRect, b: PixelRect) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
