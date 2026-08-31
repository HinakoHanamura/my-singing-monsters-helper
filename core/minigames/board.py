"""Track which board slot is which, using only what the current frame shows.

The problem this solves
-----------------------
``MemorySolver`` reasons about *slot indices*: it learns that slot 7 shows a
particular face and expects that to still be true several turns later. Those
indices therefore have to mean the same card for the whole level.

Detection alone cannot provide that. The detector finds face-down cards, and
face-down cards disappear as pairs get matched, so re-deriving indices from a
fresh detection renumbers everything underneath the solver. Measured on level 9
of ``captures/memory``:

    frame 194   rows 6/6/6/6/6   30 boxes -> indices 0..29
    frame 195   rows 5/5/6/6/6   28 boxes -> indices 0..27
    frame 197   rows 4/4/6/6/6   26 boxes -> indices 0..25
    frame 200   rows 3/3/6/6/6   24 boxes -> indices 0..23

The same physical card is index 7 in one frame and index 5 in the next. Knowledge
learned about one card would silently transfer to another.

What this module does instead
-----------------------------
Slot *identity* is fixed once, from the opening frame of a level where every card
is still face down. Slot *geometry* is re-measured on every single observation:
each freshly detected box is matched to the nearest known slot and the stored
coordinates are replaced with the new ones.

That split is the point. Nothing about the layout is ever taken on trust from an
earlier frame, so the click coordinates are always from the frame just captured
-- if the window is nudged a few pixels the boxes follow it. At the same time
nothing renumbers, so the solver's memory stays attached to the right cards.

When the geometry moves further than the matching tolerance -- a window resize,
a scene change, a level transition -- matching fails as a whole and says so. The
caller then re-initialises the level rather than clicking at stale coordinates.
Re-initialising throws away learned faces and costs a few mismatches, which is
affordable: the game grants ceil(1.5 * pairs) mismatches while the solver's
proven worst case is pairs, so there is roughly 50% headroom. Clicking a stale
coordinate has no such safety margin, which is why the trade runs this way.

No layout is ever precomputed, stored or reused between levels. Every level
starts from a fresh scan, deliberately, so a future game update that moves the
cards changes nothing here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from core.geometry import PixelRect, Point, distance
from core.minigames.card_vision import median_card_size
from core.minigames.grid import describe_grid, reading_order

logger = logging.getLogger(__name__)

#: Slot matching tolerance as a fraction of the median card width. Half a card
#: is far larger than the jitter between consecutive frames, yet smaller than
#: the gap to the neighbouring slot, so a box can only ever claim its own slot.
_MATCH_TOLERANCE_RATIO = 0.5

#: Floor for degenerate input, so identical centres still match.
_MIN_MATCH_TOLERANCE = 2.0

#: A frame where a few boxes match no slot while the rest match theirs. Reported
#: separately from drift because the two have opposite causes and opposite cures:
#: drift means every stored coordinate is suspect and the level must restart; a
#: stray blob means this one frame is unusable and the next one probably is not.
CODE_STRAY_BOXES = "STRAY_BOXES"

#: How many unmatched boxes can still be read as strays rather than as drift.
#:
#: Genuine drift moves the whole layout, so it shows up as *most* boxes failing
#: to match, not one. A detector false positive shows up as one or two. The
#: measured false-positive counts from card_vision's negative controls are
#: exactly that range: 0 on the rewards page, 0 inside a revealed card face,
#: 2 on the replay dialog, 1 on the island screen. Two is therefore the whole
#: observed range of strays, and more than that is no longer a stray pattern.
_MAX_STRAY_BOXES = 2


def box_center(box: PixelRect) -> Point:
    x1, y1, x2, y2 = box
    return (x1 + x2) // 2, (y1 + y2) // 2


@dataclass(frozen=True)
class BoardObservation:
    """What one frame says about a board whose slots are already identified.

    ``face_down`` is the set the solver may still choose from. ``face_up`` is
    everything else, and deliberately does not distinguish "just flipped by me"
    from "matched and settled" -- those two states are pixel-identical in this
    game, so only the solver's own bookkeeping can tell them apart.

    ``code`` is stable ASCII for tests to assert on; ``reason`` is Chinese for
    the log.
    """

    ok: bool
    code: str
    reason: str
    face_down: FrozenSet[int]
    face_up: FrozenSet[int]
    unmatched_boxes: int


class SlotMap:
    """Stable slot identities over freshly measured geometry.

    Construct from a frame where the whole layout is face down, then call
    :meth:`observe` once per cycle. Coordinates are refreshed by every call;
    identities never change for the lifetime of the object, which is one level.
    """

    def __init__(self, boxes: Sequence[PixelRect]) -> None:
        if not boxes:
            raise ValueError("空盘面无法建立槽位表")

        # Reading order is what makes the flip sequence predictable to someone
        # watching: top-to-bottom by row, left-to-right inside a row.
        order = reading_order(boxes)
        self._boxes: List[PixelRect] = [tuple(boxes[i]) for i in order]  # type: ignore[misc]
        self._card_size = median_card_size(self._boxes)
        self._initial_shape = describe_grid(self._boxes)

    # ---- geometry as of the most recent observation ----

    @property
    def slot_count(self) -> int:
        return len(self._boxes)

    @property
    def card_size(self) -> Tuple[int, int]:
        """Median card width and height from the latest observation."""
        return self._card_size

    @property
    def initial_shape(self) -> str:
        """Row shape recorded at level start, e.g. ``"5 行 (6/6/6/6/6)，共 30 张"``.

        Logged once so a wrong row split is obvious at a glance instead of
        having to be inferred from odd clicking behaviour later.
        """
        return self._initial_shape

    def box(self, slot: int) -> PixelRect:
        return self._boxes[slot]

    def center(self, slot: int) -> Point:
        return box_center(self._boxes[slot])

    def match_tolerance(self) -> float:
        width = self._card_size[0]
        return max(_MIN_MATCH_TOLERANCE, width * _MATCH_TOLERANCE_RATIO)

    # ---- the per-cycle update ----

    def observe(self, boxes: Sequence[PixelRect]) -> BoardObservation:
        """Assign freshly detected boxes to slots and refresh their coordinates.

        Args:
            boxes: card-back boxes from the frame just captured, any order.

        Returns:
            Which slots are face down right now. On failure the geometry is left
            untouched and ``ok`` is False, so a caller that ignores the verdict
            still cannot act on half-updated coordinates.
        """
        if not boxes:
            # A legitimate end state: the last pair has been matched and nothing
            # is face down any more. Not an error, just an empty set.
            return BoardObservation(
                True, "ALL_FACE_UP", "全部已翻开", frozenset(), frozenset(range(self.slot_count)), 0
            )

        if len(boxes) > self.slot_count:
            return BoardObservation(
                False,
                "MORE_THAN_SLOTS",
                f"检出 {len(boxes)} 张，多于 {self.slot_count} 个槽位",
                frozenset(),
                frozenset(),
                len(boxes),
            )

        tolerance = self.match_tolerance()
        # Greedy nearest matching is sound here because the tolerance is below
        # half the slot pitch: a box inside one slot's radius cannot be inside
        # another's, so there is no ordering-dependent outcome to worry about.
        claimed: Dict[int, PixelRect] = {}
        unmatched = 0

        for box in boxes:
            center = box_center(box)
            slot = self._nearest_slot(center, tolerance, exclude=set(claimed))
            if slot is None:
                unmatched += 1
                continue
            claimed[slot] = tuple(box)  # type: ignore[assignment]

        if unmatched:
            # Two different faults land here and they need different answers.
            #
            # Drift -- a resize, a scene change, a level transition -- moves the
            # whole layout at once, so nearly every box misses its slot. Nothing
            # stored can be trusted and the level has to be re-initialised.
            #
            # A detector false positive is one stray blob while every other box
            # sits in its own slot. Treating that as drift threw away a level
            # that was perfectly playable, which is what used to happen: one
            # unmatched box out of twenty-two stopped the run.
            #
            # Reporting them separately does *not* mean believing the frame. The
            # stray case still refuses to write coordinates, because a box that
            # matched nothing could also be a real card that moved, and in that
            # reading some slot is missing its box and would be reported face-up
            # -- which the solver could mistake for "already matched". The caller
            # looks again instead; a blob from an animation frame is gone by the
            # next capture, while real drift is still there.
            if unmatched <= _MAX_STRAY_BOXES and len(claimed) > unmatched:
                return BoardObservation(
                    False,
                    CODE_STRAY_BOXES,
                    f"{unmatched} 个检出框对不上槽位，其余 {len(claimed)} 个都对上了，"
                    f"按杂框处理，重新观察",
                    frozenset(),
                    frozenset(),
                    unmatched,
                )
            return BoardObservation(
                False,
                "GEOMETRY_DRIFT",
                f"{unmatched} 个检出框对不上任何槽位，几何已漂移",
                frozenset(),
                frozenset(),
                unmatched,
            )

        # Only now is it safe to write: a partial update would leave some slots
        # on this frame's coordinates and others on an older frame's.
        for slot, box in claimed.items():
            self._boxes[slot] = box
        self._card_size = median_card_size(list(claimed.values()))

        face_down = frozenset(claimed)
        face_up = frozenset(range(self.slot_count)) - face_down
        return BoardObservation(
            True,
            "OBSERVED",
            f"背面 {len(face_down)} 张，已翻开 {len(face_up)} 张",
            face_down,
            face_up,
            0,
        )

    def _nearest_slot(
        self, center: Point, tolerance: float, exclude: set
    ) -> Optional[int]:
        best_slot: Optional[int] = None
        best_distance = tolerance
        for slot in range(self.slot_count):
            if slot in exclude:
                continue
            gap = distance(center, self.center(slot))
            if gap <= best_distance:
                best_distance = gap
                best_slot = slot
        return best_slot
