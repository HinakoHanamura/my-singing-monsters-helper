"""Card-back detection: plumbing, filters and the board gate.

Scope, and what these tests may not be used for
-----------------------------------------------
The images here are drawn in HSV so that "this pixel is inside the colour
window" is true by construction rather than by resemblance to the game. That
makes them suitable for checking the wiring -- one detection per tile, boxes
where the tiles are, rejects shaped like the ones the real screens produce --
and unsuitable for anything else.

**Thresholds must never be calibrated from these images.** Synthetic tiles are
flat colour; real cards are animated sprites with gradients, a drop shadow and a
darker spiral through the middle. Numbers derived from flat fills would be
fiction. The colour window, aspect bounds and fill floor are all measured on the
recordings by ``tools/probe_card_backs.py``, and the end-to-end behaviour over
all nine real levels is checked by ``tools/probe_board_tracking.py``.

Assertions are on the stable ASCII ``code`` of each verdict, never on the
Chinese ``reason``: a test that grepped display text has produced a false
failure before, when a message happened to contain the word being searched for.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from core.minigames.card_vision import (
    CARD_BACK,
    DEFAULT_CARD_BACK_PARAMS,
    CardBackVision,
    board_screen_verdict,
    median_card_size,
    width_spread,
)

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Comfortably inside the measured colour window on all three channels.
CARD_HSV = (135, 120, 190)

# Dark and washed out, like the wood board: fails on saturation and on value, so
# no single threshold change can accidentally make the background match.
BACKGROUND_HSV = (130, 25, 45)


def make_frame(tiles, background=BACKGROUND_HSV, tile_hsv=CARD_HSV) -> np.ndarray:
    """Draw filled tiles on a background and hand back a BGR frame.

    Args:
        tiles: iterable of (x, y, w, h) in pixels.

    Built in HSV and converted once at the end, so the colours the detector sees
    are exactly the ones named above.
    """
    hsv = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)
    hsv[:, :] = background
    for x, y, w, h in tiles:
        hsv[y : y + h, x : x + w] = tile_hsv
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def boxes_of(detections):
    return sorted(d.bbox for d in detections)


@pytest.fixture
def vision() -> CardBackVision:
    return CardBackVision()


# --- the detector ------------------------------------------------------------


def test_each_tile_becomes_one_detection(vision):
    tiles = [(40, 40, 80, 80), (200, 40, 80, 80), (40, 200, 80, 80)]
    frame = make_frame(tiles)

    detections = vision.detect(CARD_BACK, frame)

    assert len(detections) == 3
    assert all(d.name == CARD_BACK for d in detections)
    # Morphological closing can round the outline by a pixel, so boxes are
    # compared with a tolerance rather than exactly.
    for (x, y, w, h), box in zip(sorted(tiles), boxes_of(detections)):
        assert box == pytest.approx((x, y, x + w, y + h), abs=2)


def test_confidence_reports_how_much_of_the_box_is_card_coloured(vision):
    frame = make_frame([(40, 40, 80, 80)])

    detection = vision.detect(CARD_BACK, frame)[0]

    # A solid tile fills its own bounding box, so the score should be near one.
    # It is a real measurement rather than a constant, which is what lets a
    # caller rank shaky candidates.
    assert 0.9 <= detection.confidence <= 1.0


def test_detections_come_back_sorted_by_confidence(vision):
    # A hollow tile fills less of its box than a solid one of the same size.
    frame = make_frame([(40, 40, 80, 80), (200, 40, 80, 80)])
    hsv_hole = np.zeros((30, 30, 3), np.uint8)
    hsv_hole[:, :] = BACKGROUND_HSV
    frame[55:85, 215:245] = cv2.cvtColor(hsv_hole, cv2.COLOR_HSV2BGR)

    detections = vision.detect(CARD_BACK, frame)

    scores = [d.confidence for d in detections]
    assert scores == sorted(scores, reverse=True)


def test_unknown_target_returns_nothing_instead_of_raising(vision):
    """A miss must degrade to "not found" rather than kill the bot thread."""
    frame = make_frame([(40, 40, 80, 80)])

    assert vision.detect("coin", frame) == []


@pytest.mark.parametrize(
    "frame",
    [None, np.zeros((0, 0, 3), np.uint8)],
    ids=["none", "empty"],
)
def test_degenerate_frames_return_nothing(vision, frame):
    """Capture can hand back a collapsed client area while a window closes."""
    assert vision.detect(CARD_BACK, frame) == []


def test_background_alone_yields_no_detections(vision):
    assert vision.detect(CARD_BACK, make_frame([])) == []


def test_portrait_tiles_are_rejected(vision):
    """The results screen draws portrait reward backs in the same violet.

    Measured at 82x117, aspect 0.70. An earlier 0.70 lower bound let three of
    them through, which is why the floor now sits above them.
    """
    aspect = 82 / 117
    assert aspect < DEFAULT_CARD_BACK_PARAMS.min_aspect

    assert vision.detect(CARD_BACK, make_frame([(40, 40, 82, 117)])) == []


def test_square_tiles_at_real_card_aspect_are_kept(vision):
    """Game cards measure about 1.04 wide-to-tall; 70x67 is a real level size."""
    detections = vision.detect(CARD_BACK, make_frame([(40, 40, 70, 67)]))

    assert len(detections) == 1


def test_a_shape_that_only_half_fills_its_box_is_rejected(vision):
    """The regression the second live run exposed.

    A patch of violet *inside* a revealed card matched hue, size and aspect, and
    sat 12 px from the slot centre -- close enough for the slot matcher to read
    the card as still face down, so the runner waited forever on a card that had
    plainly flipped.

    Hue could not separate them: the blob measured 127/151/155 against the card
    back's 131/144/151. What did was how completely each fills its own bounding
    box. Real card backs are nearly solid across the whole size range (0.89 at
    122 px, 0.97 at 87 px, 0.92 at 63 px); the blob was 0.49.

    An L shape stands in for it here: the bounding box is card-sized and square,
    but only part of it is filled.
    """
    frame = make_frame([])
    hsv_tile = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)
    hsv_tile[:, :] = BACKGROUND_HSV
    # Two arms of an L, sharing one corner: a 90x90 bounding box roughly half full.
    hsv_tile[100:190, 100:135] = CARD_HSV
    hsv_tile[155:190, 100:190] = CARD_HSV
    frame = cv2.cvtColor(hsv_tile, cv2.COLOR_HSV2BGR)

    detections = vision.detect(CARD_BACK, frame)

    assert detections == [], "a half-filled box was accepted as a card back"


def test_a_solid_tile_of_the_same_size_is_still_accepted(vision):
    """Guards the other direction, so the fill floor cannot be raised blindly."""
    detections = vision.detect(CARD_BACK, make_frame([(100, 100, 90, 90)]))

    assert len(detections) == 1
    assert detections[0].confidence >= DEFAULT_CARD_BACK_PARAMS.min_fill


@pytest.mark.parametrize(
    "size",
    [(20, 20), (300, 300)],
    ids=["too_small", "too_large"],
)
def test_tiles_outside_the_size_range_are_rejected(vision, size):
    width, height = size
    frame = make_frame([(10, 10, width, height)])

    assert vision.detect(CARD_BACK, frame) == []


def test_the_full_measured_size_range_is_accepted(vision):
    """63 px and 122 px are the narrowest and widest cards the game draws."""
    for side in (63, 122):
        detections = vision.detect(CARD_BACK, make_frame([(40, 40, side, side)]))
        assert len(detections) == 1, f"{side}px card was dropped"


# --- geometry helpers --------------------------------------------------------


def test_width_spread_and_median_size():
    boxes = [(0, 0, 70, 67), (100, 0, 171, 67), (0, 100, 69, 67)]

    assert width_spread(boxes) == 2
    assert median_card_size(boxes) == (70, 67)


@pytest.mark.parametrize("boxes", [[], [(0, 0, 70, 67)]], ids=["empty", "single"])
def test_width_spread_is_zero_without_a_pair_to_compare(boxes):
    assert width_spread(boxes) == 0


# --- the board gate ---------------------------------------------------------
#
# Every rejection below corresponds to a screen that really did produce
# plausible boxes: the results screen, the replay prompt and the island view.


def square_boxes(count, side=70, spread=0):
    """``count`` boxes laid out in a row, the last one widened by ``spread``."""
    boxes = []
    for index in range(count):
        width = side + (spread if index == count - 1 else 0)
        left = index * (side + 20)
        boxes.append((left, 0, left + width, side))
    return boxes


def test_empty_detection_is_not_a_board():
    verdict = board_screen_verdict([])

    assert verdict.ok is False
    assert verdict.code == "NO_CARDS"


def test_too_few_cards_is_not_a_board():
    """The island view produced three violet blobs."""
    verdict = board_screen_verdict(square_boxes(3))

    assert verdict.ok is False
    assert verdict.code in {"TOO_FEW_CARDS", "INCONSISTENT_SIZE"}


def test_odd_count_is_not_a_full_board():
    """Cards come in pairs, so an odd count cannot be a whole layout.

    The replay prompt produced five tiles at the same aspect as real cards,
    which is the case aspect ratio alone cannot catch.
    """
    verdict = board_screen_verdict(square_boxes(5))

    assert verdict.ok is False
    assert verdict.code == "ODD_CARD_COUNT"


def test_inconsistent_card_sizes_are_not_a_board():
    """On a real board the width spread measured at most 2 px on every level."""
    verdict = board_screen_verdict(square_boxes(6, spread=41))

    assert verdict.ok is False
    assert verdict.code == "INCONSISTENT_SIZE"


def test_a_uniform_even_layout_is_accepted():
    verdict = board_screen_verdict(square_boxes(6))

    assert verdict.ok is True
    assert verdict.code == "BOARD_OK"
    assert verdict.card_count == 6
    assert verdict.spread == 0


def test_an_odd_count_is_fine_once_the_total_is_known():
    """Mid-level the face-down set shrinks by ones and twos.

    A card flipped and awaiting comparison leaves an odd number face down, which
    is normal; only a *fresh* board has to be even.
    """
    verdict = board_screen_verdict(square_boxes(5), expected_count=6)

    assert verdict.ok is True
    assert verdict.code == "BOARD_OK"


def test_more_cards_than_the_level_holds_is_rejected():
    """Face-down cards can only ever fall within a level, never rise above the total."""
    verdict = board_screen_verdict(square_boxes(8), expected_count=6)

    assert verdict.ok is False
    assert verdict.code == "MORE_THAN_EXPECTED"


def test_even_count_is_necessary_but_not_sufficient_for_a_fresh_board():
    """Documents a real limitation, so nobody reads BOARD_OK as "untouched".

    Level 8 holds 32 cards. Replaying the recording, a frame with two already
    revealed showed 30 backs -- even, uniform, accepted. Whether the whole layout
    is present cannot be settled from colour alone, so the caller must guarantee
    it by only building a slot map when it knows a level has just started.
    """
    verdict = board_screen_verdict(square_boxes(30))

    assert verdict.ok is True
    assert verdict.card_count == 30
