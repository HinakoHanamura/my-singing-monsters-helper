"""Locate face-down memory-game cards by colour rather than by template.

Why not template matching
-------------------------
The minigame lays out a different number of cards on each of its nine levels and
scales them to fit the board, so the same card-back artwork is about 122 px wide
on level 1 and about 63 px wide on level 7 -- a factor of roughly 1.9 measured
across the recordings in ``captures/memory``. One fixed-scale template cannot
span that, and a multi-scale search would pay for every scale on every frame.

The card back is the one part of this game that never varies: a saturated violet
tile with a darker spiral, drawn on low-saturation dark-purple wood. Selecting on
hue and saturation is therefore a detector that does not care about size at all,
which is both simpler and cheaper than covering the scale range by brute force.

What colour alone cannot do
---------------------------
Violet is not unique to the card back. The revealed artwork uses it too: the
first live run stopped with the runner waiting forever on a card that had plainly
flipped, because a patch *inside* the revealed picture passed the colour window
and sat 12 px from the slot centre -- close enough for the slot matcher to read it
as "still face down".

Measured, that blob and a real card back are indistinguishable by hue (127/151/155
against 131/144/151) and pass the same aspect bound. What separates them is how
completely they fill their own bounding box: a card back is a nearly solid tile,
artwork is not. Hence the fill floor below, which is the load-bearing filter here
rather than a formality.

Why this implements BaseVisionAgent
-----------------------------------
``BaseVisionAgent.detect`` exists so the perception layer can be swapped without
the layers above noticing -- it was written with a learned model in mind. A
colour segmenter is exactly that kind of alternative implementation, so it goes
through the same contract and returns the same ``Detection`` records, box and
score included.

What this deliberately does not do
----------------------------------
It does not decide whether the board is even on screen. The same violet appears
on the results screen and on the island, so a caller that trusts a bare
detection will click on the wrong screen; see ``board_screen_verdict`` and
:mod:`core.minigames.board` for the gate. Keeping the two apart means the
detector stays a pure measurement and the policy lives where it can be tested.

Every constant below was measured, not guessed. ``tools/probe_card_backs.py``
reproduces the counts over the recordings and ``tools/explain_card_backs.py``
prints the per-box measurements for any single frame.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from statistics import median
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from core.geometry import PixelRect
from core.vision_agent import BaseVisionAgent, Detection

logger = logging.getLogger(__name__)

#: Logical target name this detector answers to.
CARD_BACK = "card_back"


@dataclass(frozen=True)
class CardBackParams:
    """Measured thresholds for the violet card back.

    Hue and saturation are in OpenCV's ranges (H 0-179, S/V 0-255). The defaults
    were validated on all nine levels: the detected count matched a manual count
    on every level-start frame.
    """

    #: Violet sits near H=135; real card backs measure hue p10/p50/p90 of
    #: 131/144/151 at every card size. The window is wide enough to cover that
    #: gradient without reaching the pinks of the reward icons.
    hue_low: int = 120
    hue_high: int = 155

    #: The wood background is dark and washed out; these two bounds are what
    #: actually separate card from background.
    saturation_low: int = 45
    value_low: int = 90

    #: The spiral in the middle of the tile is darker than its border and falls
    #: outside the colour window, which would leave a ring of fragments instead
    #: of one blob. Closing the mask first merges them.
    close_kernel: int = 7

    #: Game cards measure about 1.04 wide-to-tall. The reward cards on the
    #: results screen are portrait at about 0.70, and an earlier 0.70 lower
    #: bound let three of them through, so the floor sits above them.
    min_aspect: float = 0.85
    max_aspect: float = 1.40

    #: Smallest and largest observed cards are about 63 px and 122 px across.
    #: The bounds are widened so an unseen level size does not silently vanish.
    min_side: int = 40
    max_side: int = 200

    # Share of the bounding box that must be card-back coloured.
    #
    # This is the filter that separates a card back from violet *inside* revealed
    # artwork, and it earned its keep on the first live run: a patch within a
    # flipped card matched hue, size and aspect, and only fill told it apart.
    #
    # Measured across the whole size range, real card backs are nearly solid:
    #
    #   122x117 cards   0.89 - 0.91
    #    87x84  cards   0.96 - 0.97
    #    63x61  cards   0.92 - 0.93
    #   artwork blob    0.49
    #
    # 0.70 sits about 0.19 below the least-solid real card and about 0.21 above
    # the blob -- near the middle of the gap, with room on both sides. The
    # previous value of 0.45 was permissive enough to admit the blob.
    min_fill: float = 0.70

    def as_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.array([self.hue_low, self.saturation_low, self.value_low], np.uint8)
        upper = np.array([self.hue_high, 255, 255], np.uint8)
        return lower, upper


DEFAULT_CARD_BACK_PARAMS = CardBackParams()


def card_back_mask(frame: np.ndarray, params: CardBackParams) -> np.ndarray:
    """Binary mask of card-back coloured pixels."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower, upper = params.as_bounds()
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (params.close_kernel, params.close_kernel)
    )
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


class CardBackVision(BaseVisionAgent):
    """Colour-segmentation detector for face-down cards.

    Only ``"card_back"`` is answered; any other target returns nothing rather
    than raising, so a caller probing for something this detector does not know
    about degrades to "not found" instead of crashing the bot thread.
    """

    def __init__(self, params: CardBackParams = DEFAULT_CARD_BACK_PARAMS) -> None:
        self.params = params

    def detect(self, target_name: str, screenshot: np.ndarray) -> List[Detection]:
        if target_name != CARD_BACK:
            logger.debug("CardBackVision 不认识目标 %s，返回空", target_name)
            return []
        if screenshot is None or screenshot.size == 0:
            return []

        params = self.params
        mask = card_back_mask(screenshot, params)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: List[Detection] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if not (params.min_side <= w <= params.max_side):
                continue
            if not (params.min_side <= h <= params.max_side):
                continue
            aspect = w / float(h)
            if not (params.min_aspect <= aspect <= params.max_aspect):
                continue
            fill = float(mask[y : y + h, x : x + w].mean()) / 255.0
            if fill < params.min_fill:
                continue
            detections.append(
                Detection(
                    name=CARD_BACK,
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                    # Fill doubles as the score: it says how much of the box is
                    # really card back, which is what a caller ranking shaky
                    # candidates wants to know.
                    confidence=fill,
                )
            )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections


# --- telling the board apart from every other violet screen -------------------
#
# Measured false positives with the detector alone:
#
#   island view          3 boxes, widths 43..84   (scenery that happens to be violet)
#   results screen       3 boxes, 82x117          (portrait reward backs)
#   replay prompt        5 boxes, 77x74           (reward tiles, aspect ~= game cards)
#
# Aspect ratio removes the first two but not the replay prompt, whose tiles are
# as square as real cards. What does separate them is uniformity: on a real board
# every card is drawn at one size, and the widest-minus-narrowest spread measured
# at most 2 px on every level. Scenery and prompt decorations are never that
# consistent.

#: Largest width spread, in pixels, still consistent with one board. The measured
#: spread was 0..2 px, so this leaves room for antialiasing without admitting the
#: 41 px spread seen on the island.
MAX_WIDTH_SPREAD = 6

#: A board always holds an even number of cards, and the smallest level has four.
MIN_BOARD_CARDS = 4


def width_spread(boxes: Sequence[PixelRect]) -> int:
    """Widest minus narrowest box width; 0 for fewer than two boxes."""
    if len(boxes) < 2:
        return 0
    widths = [x2 - x1 for x1, _, x2, _ in boxes]
    return max(widths) - min(widths)


def median_card_size(boxes: Sequence[PixelRect]) -> Tuple[int, int]:
    """Median width and height, the scale everything else is derived from."""
    if not boxes:
        return 0, 0
    widths = [x2 - x1 for x1, _, x2, _ in boxes]
    heights = [y2 - y1 for _, y1, _, y2 in boxes]
    return int(median(widths)), int(median(heights))


@dataclass(frozen=True)
class BoardScreenVerdict:
    """Why a frame was accepted or rejected as a minigame board.

    Carries a stable ASCII ``code`` so tests can assert on it; the Chinese
    ``reason`` is for the log only. Asserting on display text has produced a
    false failure before, when a message happened to contain a word the test was
    grepping for.
    """

    ok: bool
    code: str
    reason: str
    card_count: int
    spread: int


def board_screen_verdict(
    boxes: Sequence[PixelRect], expected_count: int | None = None
) -> BoardScreenVerdict:
    """Decide whether these detections plausibly come from a live board.

    Args:
        boxes: card-back boxes from :class:`CardBackVision`.
        expected_count: total slots in the current level, when known. Face-down
            cards drop as pairs are cleared, so the live count only has to stay
            at or below the total -- never above it.

    A rejection is a reason to look again rather than to click. Clicking on a
    misread screen is the one outcome that can cost the user something real.
    """
    count = len(boxes)
    spread = width_spread(boxes)

    if count == 0:
        return BoardScreenVerdict(False, "NO_CARDS", "未检测到卡牌", count, spread)

    if expected_count is None:
        # Fresh board: the full layout must be present, so the count has to be
        # even and at least as large as the smallest level.
        if count < MIN_BOARD_CARDS:
            return BoardScreenVerdict(
                False, "TOO_FEW_CARDS", f"仅检出 {count} 张，不足一盘", count, spread
            )
        if count % 2 != 0:
            return BoardScreenVerdict(
                False, "ODD_CARD_COUNT", f"检出 {count} 张为奇数，卡牌必成对", count, spread
            )
    elif count > expected_count:
        return BoardScreenVerdict(
            False,
            "MORE_THAN_EXPECTED",
            f"检出 {count} 张，多于本关的 {expected_count} 张",
            count,
            spread,
        )

    if spread > MAX_WIDTH_SPREAD:
        return BoardScreenVerdict(
            False,
            "INCONSISTENT_SIZE",
            f"卡牌宽度极差 {spread}px，超出同盘容差",
            count,
            spread,
        )

    return BoardScreenVerdict(True, "BOARD_OK", f"盘面可信，{count} 张背面", count, spread)


def board_screen_confidence(boxes: Sequence[PixelRect]) -> bool:
    """Convenience wrapper for call sites that only need the yes/no."""
    return board_screen_verdict(boxes).ok
