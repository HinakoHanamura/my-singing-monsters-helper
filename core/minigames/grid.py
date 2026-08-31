"""Recover a stable reading order from an irregular card layout.

The memory game does not hand us a tidy matrix. Cards are laid out in rows that
may hold different numbers of cards, may be nudged a few pixels off a shared
baseline, and shift wholesale when the window is resized. Template matching
returns those cards in score order, which is meaningless to a human and
unstable between frames.

Everything downstream (which card to flip first, which slot a face belongs to)
keys off an *index*, so that index has to mean the same thing every frame. This
module turns a bag of boxes into the order a person would read them:
top-to-bottom by row, left-to-right inside each row.

Two design choices worth calling out:

* Row banding uses a running mean of the row's vertical centres rather than the
  first card's centre. A layout with a slight slant would otherwise accumulate
  error until the row splits in half.
* The default tolerance is derived from the cards themselves (a fraction of the
  median card height) instead of a hard pixel count. Card height already scales
  with the window, so the tolerance scales with it for free and never needs
  re-calibrating per resolution.

The functions return **indices into the caller's sequence**, never copies of the
boxes. That keeps the module free of any opinion about what a card record looks
like.
"""

from __future__ import annotations

from statistics import median
from typing import List, Sequence, Tuple

from core.geometry import PixelRect

#: Fraction of the median card height used as the default row band. Half a card
#: is comfortably larger than the few pixels of layout jitter we expect, yet
#: comfortably smaller than the pitch between two rows, so it separates rows
#: without splitting one.
_ROW_TOLERANCE_RATIO = 0.5

#: Fallback band for degenerate input (zero-height boxes). Only reachable with
#: malformed data; picked so that identical y values still group together.
_MIN_ROW_TOLERANCE = 1.0


def box_center(box: PixelRect) -> Tuple[float, float]:
    """Centre of a box as floats, so ordering never suffers rounding ties."""
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def box_height(box: PixelRect) -> int:
    _, y1, _, y2 = box
    return abs(y2 - y1)


def default_row_tolerance(boxes: Sequence[PixelRect]) -> float:
    """Vertical band, in pixels, within which two cards count as one row.

    Derived from the median card height rather than configured, because the
    cards are uniform in size and already carry the window's scale. Median (not
    mean) so one spurious detection cannot stretch the band.
    """
    if not boxes:
        return _MIN_ROW_TOLERANCE
    heights = [box_height(b) for b in boxes if box_height(b) > 0]
    if not heights:
        return _MIN_ROW_TOLERANCE
    return max(_MIN_ROW_TOLERANCE, median(heights) * _ROW_TOLERANCE_RATIO)


def group_rows(
    boxes: Sequence[PixelRect], row_tolerance: float | None = None
) -> List[List[int]]:
    """Cluster boxes into rows, ordered top-to-bottom then left-to-right.

    Args:
        boxes: card bounding boxes in client-area pixels, any order.
        row_tolerance: vertical band for "same row". Defaults to a fraction of
            the median card height.

    Returns:
        A list of rows, each a list of indices into ``boxes``. Rows run
        top-to-bottom; indices inside a row run left-to-right.
    """
    if not boxes:
        return []

    tolerance = (
        default_row_tolerance(boxes) if row_tolerance is None else float(row_tolerance)
    )

    centers = [box_center(b) for b in boxes]
    # Sort by y first; x breaks ties so the result is deterministic for cards
    # sharing a baseline exactly.
    by_vertical = sorted(range(len(boxes)), key=lambda i: (centers[i][1], centers[i][0]))

    rows: List[List[int]] = []
    current: List[int] = []
    running_sum = 0.0

    for idx in by_vertical:
        cy = centers[idx][1]
        if current and abs(cy - running_sum / len(current)) > tolerance:
            rows.append(current)
            current = []
            running_sum = 0.0
        current.append(idx)
        running_sum += cy

    if current:
        rows.append(current)

    for row in rows:
        row.sort(key=lambda i: centers[i][0])

    return rows


def reading_order(
    boxes: Sequence[PixelRect], row_tolerance: float | None = None
) -> List[int]:
    """Flattened left-to-right, top-to-bottom ordering of ``boxes``.

    ``reading_order(boxes)[0]`` is the top-left card, which is the first card
    the solver will try to flip.
    """
    return [idx for row in group_rows(boxes, row_tolerance) for idx in row]


def describe_grid(
    boxes: Sequence[PixelRect], row_tolerance: float | None = None
) -> str:
    """Human-readable shape summary, e.g. ``"3 行 (4/4/2)，共 10 张"``.

    Used in the log so a wrong row split is obvious at a glance: a board that
    should read 4/4 showing up as 2/2/2/2 is a tolerance problem, and seeing it
    beats guessing.
    """
    rows = group_rows(boxes, row_tolerance)
    if not rows:
        return "未检测到卡牌"
    sizes = "/".join(str(len(r)) for r in rows)
    return f"{len(rows)} 行 ({sizes})，共 {sum(len(r) for r in rows)} 张"
