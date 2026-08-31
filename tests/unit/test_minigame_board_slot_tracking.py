"""Slot tracking: identities that hold still while geometry is re-measured.

What is being protected
-----------------------
``MemorySolver`` remembers "slot 7 shows this face" across turns, so slot 7 has
to stay the same physical card for the whole level. Detection cannot supply that
on its own, because face-down cards vanish as pairs are matched and a fresh
ordering renumbers everything underneath the solver. Measured on level 9 of the
recordings, the same card is index 7 in one frame and index 5 two turns later.

The other half of the contract is the opposite requirement: coordinates must
*not* be remembered. Clicking is done at the coordinates of the frame just
captured, so a window nudged a few pixels is followed rather than missed.

These two pull against each other, and the tests below pin down both: identities
are asserted to survive a shrinking board, and coordinates are asserted to be
replaced on every observation.

Boxes here are plain tuples, no images involved -- this is the bookkeeping layer.
Whether the boxes themselves are right is a question for
``tools/probe_board_tracking.py``, which replays all nine real levels.
"""

from __future__ import annotations

import pytest

from core.minigames.board import SlotMap, box_center

CARD = 80
GAP = 20


def grid_boxes(rows: int, columns: int, card: int = CARD, gap: int = GAP):
    """Row-major boxes, deliberately returned in reading order already."""
    boxes = []
    for row in range(rows):
        for column in range(columns):
            left = 50 + column * (card + gap)
            top = 60 + row * (card + gap)
            boxes.append((left, top, left + card, top + card))
    return boxes


def shifted(boxes, dx: int, dy: int):
    return [(x1 + dx, y1 + dy, x2 + dx, y2 + dy) for x1, y1, x2, y2 in boxes]


# --- building the map --------------------------------------------------------


def test_slots_are_numbered_in_reading_order():
    """Top-to-bottom by row, left-to-right inside a row.

    Predictable order is a product requirement, not an implementation detail:
    the user asked for cards to be flipped in that priority so the bot's
    behaviour is legible to someone watching it.
    """
    boxes = grid_boxes(2, 3)
    # Hand them over scrambled; the map must not depend on input order.
    scrambled = [boxes[4], boxes[0], boxes[3], boxes[5], boxes[1], boxes[2]]

    slot_map = SlotMap(scrambled)

    centers = [slot_map.center(i) for i in range(slot_map.slot_count)]
    assert centers == [box_center(b) for b in boxes]


def test_card_size_and_tolerance_follow_the_cards():
    """Tolerance is derived, so a different level size needs no re-calibration."""
    small = SlotMap(grid_boxes(2, 2, card=63))
    large = SlotMap(grid_boxes(2, 2, card=122))

    assert small.card_size == (63, 63)
    assert large.card_size == (122, 122)
    assert small.match_tolerance() < large.match_tolerance()


def test_initial_shape_is_recorded_for_the_log():
    slot_map = SlotMap(grid_boxes(3, 4))

    # Logged once per level so a wrong row split is visible immediately rather
    # than being inferred later from odd clicking.
    assert "3" in slot_map.initial_shape
    assert "12" in slot_map.initial_shape


def test_an_empty_board_cannot_produce_a_map():
    with pytest.raises(ValueError):
        SlotMap([])


# --- observing -------------------------------------------------------------


def test_all_slots_are_face_down_on_the_opening_frame():
    boxes = grid_boxes(2, 3)
    slot_map = SlotMap(boxes)

    observation = slot_map.observe(boxes)

    assert observation.ok is True
    assert observation.code == "OBSERVED"
    assert observation.face_down == frozenset(range(6))
    assert observation.face_up == frozenset()


def test_missing_boxes_become_face_up_slots():
    """A matched pair stays on screen showing its picture, so it simply stops
    being detected as a back. That is the only signal available."""
    boxes = grid_boxes(2, 3)
    slot_map = SlotMap(boxes)

    remaining = [boxes[i] for i in (0, 2, 3, 5)]
    observation = slot_map.observe(remaining)

    assert observation.ok is True
    assert observation.face_down == frozenset({0, 2, 3, 5})
    assert observation.face_up == frozenset({1, 4})


def test_identities_survive_a_shrinking_board():
    """The regression this module exists for.

    Slot 5 must still be slot 5 after the board has lost cards, even though a
    fresh ordering of the remaining boxes would call it something else.
    """
    boxes = grid_boxes(2, 3)
    slot_map = SlotMap(boxes)
    slot_five_center = slot_map.center(5)

    slot_map.observe([boxes[i] for i in (0, 1, 2, 5)])
    slot_map.observe([boxes[i] for i in (0, 5)])
    observation = slot_map.observe([boxes[5]])

    assert observation.face_down == frozenset({5})
    assert slot_map.center(5) == slot_five_center


def test_coordinates_are_replaced_on_every_observation():
    """Clicking must use this frame's geometry, never an earlier frame's."""
    boxes = grid_boxes(2, 2)
    slot_map = SlotMap(boxes)

    nudged = shifted(boxes, 5, -4)
    observation = slot_map.observe(nudged)

    assert observation.ok is True
    assert observation.face_down == frozenset(range(4))
    for slot in range(4):
        assert slot_map.box(slot) == nudged[slot]


def test_a_small_shift_still_matches_the_same_slots():
    """Frame-to-frame jitter must not be mistaken for a different layout."""
    boxes = grid_boxes(2, 3)
    slot_map = SlotMap(boxes)
    tolerance = slot_map.match_tolerance()
    within = int(tolerance) - 1

    observation = slot_map.observe(shifted(boxes, within, 0))

    assert observation.ok is True
    assert observation.face_down == frozenset(range(6))


def test_an_empty_frame_means_the_level_is_over():
    """Nothing face down is a legitimate end state, not a failure."""
    boxes = grid_boxes(2, 2)
    slot_map = SlotMap(boxes)

    observation = slot_map.observe([])

    assert observation.ok is True
    assert observation.code == "ALL_FACE_UP"
    assert observation.face_down == frozenset()
    assert observation.face_up == frozenset(range(4))


# --- refusing to guess -------------------------------------------------------


def test_a_large_shift_is_reported_as_drift():
    """A resize or scene change must stop the turn, not move the clicks."""
    boxes = grid_boxes(2, 3)
    slot_map = SlotMap(boxes)

    observation = slot_map.observe(shifted(boxes, 400, 300))

    assert observation.ok is False
    assert observation.code == "GEOMETRY_DRIFT"
    assert observation.unmatched_boxes > 0


def test_one_stray_box_among_matching_ones_is_not_called_drift():
    """A detector false positive and a moved layout are different faults.

    Drift moves the whole layout at once, so nearly every box misses its slot. A
    false positive is one blob while every other box sits in its own slot. Both
    used to report GEOMETRY_DRIFT, which stopped a live level that was perfectly
    playable: one unmatched box out of twenty-two ended the run.

    The two need different answers -- re-initialise the level versus look at
    another frame -- so they get different codes.
    """
    boxes = grid_boxes(2, 3)
    slot_map = SlotMap(boxes)

    observation = slot_map.observe(list(boxes[:5]) + [(2000, 2000, 2080, 2080)])

    assert observation.ok is False
    assert observation.code == "STRAY_BOXES"
    assert observation.unmatched_boxes == 1


def test_a_stray_box_still_refuses_to_publish_geometry():
    """Reporting it separately is not the same as believing the frame.

    A box that matched nothing could also be a real card that moved. Under that
    reading some slot is missing its box and would be reported face-up, which the
    solver could take for "already matched". So the coordinates stay where they
    were and the caller is expected to look again.
    """
    boxes = grid_boxes(2, 3)
    slot_map = SlotMap(boxes)
    before = [slot_map.box(i) for i in range(6)]

    nudged = shifted(boxes[:5], 4, 4)
    observation = slot_map.observe(nudged + [(2000, 2000, 2080, 2080)])

    assert observation.code == "STRAY_BOXES"
    assert observation.face_down == frozenset()
    assert [slot_map.box(i) for i in range(6)] == before


def test_too_many_unmatched_boxes_is_drift_not_strays():
    """The stray reading only holds while the strays are a minority.

    Bound taken from card_vision's measured negative controls, where the whole
    observed range of false positives is 0 to 2 boxes per frame. Above that, the
    pattern is no longer a stray one.
    """
    boxes = grid_boxes(2, 3)
    slot_map = SlotMap(boxes)

    far = [(2000 + 200 * i, 2000, 2080 + 200 * i, 2080) for i in range(3)]
    observation = slot_map.observe(list(boxes[:3]) + far)

    assert observation.ok is False
    assert observation.code == "GEOMETRY_DRIFT"


def test_geometry_is_untouched_when_matching_fails():
    """Half-updated coordinates would be worse than stale ones.

    A caller that ignores ``ok`` still must not be able to click at a mixture of
    two frames' geometry, so the write happens only after every box has matched.
    """
    boxes = grid_boxes(2, 2)
    slot_map = SlotMap(boxes)
    before = [slot_map.box(i) for i in range(4)]

    # First box lands on its slot, second is far away.
    slot_map.observe([boxes[0], (500, 400, 580, 480)])

    assert [slot_map.box(i) for i in range(4)] == before


def test_more_boxes_than_slots_is_rejected():
    """Face-down cards can only fall within a level; a rise means wrong screen."""
    boxes = grid_boxes(2, 2)
    slot_map = SlotMap(boxes)

    observation = slot_map.observe(boxes + [(500, 400, 580, 480)])

    assert observation.ok is False
    assert observation.code == "MORE_THAN_SLOTS"


def test_two_boxes_cannot_claim_the_same_slot():
    """Otherwise one slot would swallow a stray blob and hide the problem."""
    boxes = grid_boxes(2, 2)
    slot_map = SlotMap(boxes)

    x1, y1, x2, y2 = boxes[0]
    duplicate = (x1 + 3, y1 + 3, x2 + 3, y2 + 3)
    observation = slot_map.observe([boxes[0], duplicate])

    # The second box has nowhere to go, so the observation fails rather than
    # silently dropping it.
    assert observation.ok is False
    assert observation.code == "GEOMETRY_DRIFT"
