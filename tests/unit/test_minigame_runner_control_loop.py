"""The memory minigame control loop, driven by a simulated board.

How this is tested without a game
---------------------------------
``FakeBoard`` renders an actual frame -- violet tiles for backs, distinct
coloured pictures for faces -- and applies this game's rules to clicks: two flips
per turn, a matched pair stays face up forever, a mismatched one turns back. The
runner therefore goes through the real detector, the real slot map, the real
fingerprinting and the real solver; only the window, the clock and the mouse are
stand-ins.

That matters because the interesting failures are in the seams. A loop tested
against a mocked perception layer would pass while doing nothing useful, and the
one property most worth proving -- that no click can land outside a card -- is
meaningless unless the clicks travel the same path they do in production.

The faces are flat colours rather than sprites. That is fine for driving the loop,
since correlation between two crops of the same flat colour is 1 and between
different ones is low, but it means **no threshold may be calibrated from this
file**. The real numbers come from ``tools/calibrate_face_similarity.py`` over the
recordings.

Assertions are on the ASCII ``code`` of each ``StepResult``, never on the Chinese
message.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import pytest

from core.minigames.fingerprint import crop_face
from core.minigames.memory_game import MemorySolver
from core.minigames.memory_runner import (
    CODE_CLICK_REFUSED,
    CODE_GEOMETRY_DRIFT,
    CODE_LEVEL_CLEARED,
    CODE_NO_BOARD,
    CODE_NO_WINDOW,
    CODE_OUT_OF_BUDGET,
    CODE_REGROUPED,
    CODE_RESYNCED,
    CODE_RUN_COMPLETE,
    CODE_TURN_MATCHED,
    CODE_TURN_MISSED,
    CODE_UNRECOVERABLE,
    MemoryRunner,
    RunnerParams,
)

FRAME_WIDTH = 720
FRAME_HEIGHT = 560
CARD = 80
GAP = 30

#: Cards per row by default. Four fits inside the frame with room to spare; a
#: sixth column ran off the right edge and produced a clipped card, which the
#: board gate correctly rejected for inconsistent size.
DEFAULT_COLUMNS = 4

# Captures a mismatched pair stays visible before turning back over.
#
# Generous on purpose. The runner needs one capture to click and three more to see
# the card settle (two consecutive stable comparisons), so a hold of four made the
# fake turn the cards back *while they were still being sampled* -- a property of
# the stand-in, not of the game, which shows a mismatched pair for around a second.
# Keeping this well above what sampling costs stops the tests from encoding the
# runner's current poll arithmetic.
REVEAL_HOLD = 10

BACK_HSV = (135, 120, 190)
BACKGROUND_HSV = (130, 25, 45)

# Face colours in HSV, deliberately far apart in hue and outside both the card
# back window and the glow window, so a face is never mistaken for either.
FACE_HSV = [
    (0, 200, 180),
    (60, 200, 180),
    (90, 200, 180),
    (170, 200, 180),
    (15, 200, 120),
    (75, 200, 120),
    (100, 200, 120),
    (160, 200, 120),
]


def face_pattern(face: int) -> np.ndarray:
    """A 3x3 bit pattern unique to ``face``, used to give the picture texture.

    Flat colour will not do. Normalised correlation divides by each patch's
    standard deviation, so between two constant images it is mathematically
    undefined and OpenCV's answer is not meaningful -- with flat faces every card
    compared as identical and the loop "matched" pairs that do not match. Real
    card art always has structure; these patterns stand in for it.
    """
    bits = face + 1
    cells = np.zeros((3, 3), np.uint8)
    for position in range(9):
        cells[position // 3][position % 3] = 1 if (bits >> (position % 4)) & 1 else 0
        bits = bits * 3 + position
    return cells


class FakeBoard:
    """A memory board that renders itself and obeys this game's rules."""

    def __init__(self, pairs: int, layout: Optional[List[int]] = None) -> None:
        self.pairs = pairs
        # Faces laid out so partners are never adjacent, which keeps the tests
        # from passing by accident on a board where any two neighbours match.
        self.faces: List[int] = []
        for index in range(pairs):
            self.faces.append(index)
        for index in range(pairs):
            self.faces.append(index)

        self.boxes: List[Tuple[int, int, int, int]] = []
        total = pairs * 2
        if layout is None:
            layout = []
            remaining = total
            while remaining > 0:
                layout.append(min(DEFAULT_COLUMNS, remaining))
                remaining -= min(DEFAULT_COLUMNS, remaining)
        slot = 0
        top = 40
        for count in layout:
            left = 40
            for _ in range(count):
                if slot >= total:
                    break
                self.boxes.append((left, top, left + CARD, top + CARD))
                left += CARD + GAP
                slot += 1
            top += CARD + GAP

        assert self.boxes[-1][2] <= FRAME_WIDTH, "layout runs off the frame"
        assert self.boxes[-1][3] <= FRAME_HEIGHT, "layout runs off the frame"

        self.cleared: Set[int] = set()
        self.face_up: List[int] = []
        self.clicks: List[Tuple[int, int]] = []
        self.offset = (0, 0)
        self._hold = 0

    # ---- rules ----

    def click(self, x: int, y: int) -> bool:
        self.clicks.append((x, y))
        dx, dy = self.offset
        for slot, (x1, y1, x2, y2) in enumerate(self.boxes):
            if not (x1 + dx <= x < x2 + dx and y1 + dy <= y < y2 + dy):
                continue
            if slot in self.cleared or slot in self.face_up:
                return True
            self.face_up.append(slot)
            if len(self.face_up) == 2:
                a, b = self.face_up
                if self.faces[a] == self.faces[b]:
                    # A matched pair is resolved at once and stays face up for
                    # the rest of the level, as the real game does. Nothing is
                    # removed from the board; the cards simply keep showing their
                    # pictures.
                    self.cleared.update((a, b))
                    self.face_up = []
                else:
                    # Only a mismatch waits, so that both cards are on screen long
                    # enough to be photographed before they turn back.
                    self._hold = REVEAL_HOLD
            return True
        return True

    def _advance(self) -> None:
        """Let time pass by one capture, turning a mismatched pair back over."""
        if len(self.face_up) < 2:
            return
        self._hold -= 1
        if self._hold <= 0:
            self.face_up = []

    def is_back(self, slot: int) -> bool:
        return slot not in self.cleared and slot not in self.face_up

    # ---- rendering ----

    def render(self) -> np.ndarray:
        self._advance()
        hsv = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)
        hsv[:, :] = BACKGROUND_HSV
        dx, dy = self.offset
        for slot, (x1, y1, x2, y2) in enumerate(self.boxes):
            x1, y1, x2, y2 = x1 + dx, y1 + dy, x2 + dx, y2 + dy
            if self.is_back(slot):
                hsv[y1:y2, x1:x2] = BACK_HSV
            else:
                # A revealed card: dark frame with the picture inside, matching
                # the real game where the frame differs from the back.
                hsv[y1:y2, x1:x2] = (0, 0, 40)
                inset = 8
                face = self.faces[slot] % len(FACE_HSV)
                inner_x1, inner_y1 = x1 + inset, y1 + inset
                inner_x2, inner_y2 = x2 - inset, y2 - inset
                hsv[inner_y1:inner_y2, inner_x1:inner_x2] = (0, 0, 20)
                cells = face_pattern(face)
                cell_w = (inner_x2 - inner_x1) // 3
                cell_h = (inner_y2 - inner_y1) // 3
                for row in range(3):
                    for column in range(3):
                        if not cells[row][column]:
                            continue
                        cx1 = inner_x1 + column * cell_w
                        cy1 = inner_y1 + row * cell_h
                        hsv[cy1 : cy1 + cell_h, cx1 : cx1 + cell_w] = FACE_HSV[face]
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


class FakeWindow:
    """Capture source backed by a FakeBoard."""

    def __init__(self, board: Optional[FakeBoard]) -> None:
        self.board = board
        self.fail_next = 0

    def capture(self) -> Optional[np.ndarray]:
        if self.fail_next > 0:
            self.fail_next -= 1
            return None
        if self.board is None:
            return None
        return self.board.render()


class FakeAction:
    """Click sink that forwards to the board and records the scale."""

    def __init__(self, board: Optional[FakeBoard]) -> None:
        self.board = board
        self.scales: List[float] = []
        self.refuse = False

    def set_scale(self, scale: float) -> None:
        self.scales.append(scale)

    def click(self, x: int, y: int) -> bool:
        if self.refuse:
            return False
        if self.board is None:
            return False
        return self.board.click(x, y)


def make_runner(board: Optional[FakeBoard], **overrides):
    window = FakeWindow(board)
    action = FakeAction(board)
    params = RunnerParams(
        poll_interval=0.0,  # no real waiting in tests
        settle_polls=overrides.pop("settle_polls", 10),
        flip_back_polls=overrides.pop("flip_back_polls", 20),
        board_polls=overrides.pop("board_polls", 5),
        max_levels=overrides.pop("max_levels", 1),
        max_recoveries=overrides.pop("max_recoveries", 2),
        scan_first=overrides.pop("scan_first", False),
        stability_gap=0.0,  # no real waiting in tests
    )
    runner = MemoryRunner(
        window=window,
        action=action,
        params=params,
        sleep=lambda _seconds: None,
    )
    return runner, window, action


# --- starting a level -------------------------------------------------------


def test_a_full_board_starts_a_level():
    board = FakeBoard(pairs=4)
    runner, _, _ = make_runner(board)

    result = runner.seek_board()

    assert result.ok is True
    assert runner.session is not None
    assert runner.session.slot_map.slot_count == 8
    assert runner.session.pair_count == 4


def test_no_window_means_no_board():
    runner, _, _ = make_runner(None)

    result = runner.seek_board()

    assert result.ok is False
    assert result.code == CODE_NO_BOARD


def test_capture_failures_are_retried_before_giving_up():
    board = FakeBoard(pairs=4)
    runner, window, _ = make_runner(board, board_polls=5)
    window.fail_next = 3

    result = runner.seek_board()

    assert result.ok is True


# --- playing ---------------------------------------------------------------


def test_a_turn_reports_matched_or_missed():
    board = FakeBoard(pairs=4)
    runner, _, _ = make_runner(board)
    runner.seek_board()

    result = runner.play_turn()

    assert result.ok is True
    assert result.code in {CODE_TURN_MATCHED, CODE_TURN_MISSED}


def test_a_whole_level_is_solved():
    """The end-to-end property: the loop finishes a board through real perception."""
    board = FakeBoard(pairs=4)
    runner, _, _ = make_runner(board)

    result = runner.run()

    assert result.ok is True
    assert runner.levels_cleared == 1
    assert board.cleared == set(range(8))


def test_mismatches_stay_within_the_games_allowance():
    """The solver's proven bound, checked through the full pipeline.

    Every miss reveals two previously unseen cards and nothing learned is
    discarded, so misses cannot exceed the number of pairs. The game grants
    ceil(1.5 * pairs), so clearing a level must never be in doubt.
    """
    board = FakeBoard(pairs=6)
    runner, _, _ = make_runner(board)

    runner.run()

    assert runner.levels_cleared == 1
    # The session is cleared on completion, so the count comes from the board.
    assert board.cleared == set(range(12))


def test_an_irregular_layout_is_handled():
    """Real boards have rows of unequal length with holes in them."""
    board = FakeBoard(pairs=5, layout=[4, 2, 4])
    runner, _, _ = make_runner(board)

    result = runner.run()

    assert result.ok is True
    assert board.cleared == set(range(10))


def test_level_completion_is_reported_once_nothing_is_face_down():
    board = FakeBoard(pairs=2)
    runner, _, _ = make_runner(board)
    runner.seek_board()

    codes = []
    for _ in range(12):
        result = runner.play_turn()
        codes.append(result.code)
        if result.code == CODE_LEVEL_CLEARED or not result.ok:
            break

    assert CODE_LEVEL_CLEARED in codes


# --- refusing to click the wrong thing --------------------------------------


def test_every_click_lands_inside_a_card():
    """The property that keeps the paid REPLAY button unreachable.

    Rather than trusting a coordinate blacklist, this checks the invariant the
    loop is built on: no delivered click is outside a card.
    """
    board = FakeBoard(pairs=4)
    runner, _, _ = make_runner(board)

    runner.run()

    assert board.clicks, "the run delivered no clicks at all"
    for x, y in board.clicks:
        assert any(
            x1 <= x < x2 and y1 <= y < y2 for x1, y1, x2, y2 in board.boxes
        ), f"click ({x}, {y}) landed outside every card"


def test_a_slot_that_is_no_longer_face_down_is_not_clicked():
    board = FakeBoard(pairs=4)
    runner, _, _ = make_runner(board)
    runner.seek_board()
    session = runner.session
    assert session is not None

    # Reveal slot 0 behind the runner's back, then ask it to click that slot.
    board.face_up = [0]
    result = runner._click_slot(session, 0)

    assert result.ok is False
    assert result.code == CODE_CLICK_REFUSED


def test_no_click_is_delivered_when_the_screen_is_not_a_board():
    """On the results screen there are no card boxes, so nothing is clickable."""
    board = FakeBoard(pairs=4)
    runner, window, _ = make_runner(board)
    runner.seek_board()

    window.board = None
    result = runner.play_turn()

    assert result.ok is False
    assert result.code == CODE_NO_WINDOW
    # Nothing was clicked while the board was unavailable.
    assert board.clicks == []


# --- refusing to guess -----------------------------------------------------


def test_a_moved_board_stops_the_turn():
    """A resize or scene change must not be clicked through."""
    board = FakeBoard(pairs=4)
    runner, _, _ = make_runner(board)
    runner.seek_board()

    board.offset = (200, 150)
    result = runner.play_turn()

    assert result.ok is False
    assert result.code == CODE_GEOMETRY_DRIFT
    assert board.clicks == []


def test_a_small_shift_does_not_stop_the_turn():
    """Coordinates are re-measured every capture, so jitter is followed."""
    board = FakeBoard(pairs=4)
    runner, _, _ = make_runner(board)
    runner.seek_board()

    board.offset = (6, -5)
    result = runner.play_turn()

    assert result.ok is True


def flood_registry(session, count: int) -> None:
    """Give the registry more distinct pictures than the level can hold.

    Textured patches, for the same reason the fake faces are textured: flat colour
    has no variance and correlation between two constant images is undefined, so
    flat patches would all land in one group.
    """
    rng = np.random.default_rng(20260829)
    for _ in range(count):
        session.registry.identify(rng.integers(0, 255, size=(48, 48, 3), dtype=np.uint8))


def test_too_many_face_groups_rejoins_the_closest_two():
    """More pictures than the level has pairs means one picture got split.

    The count is not a guess: a level of n pairs shows exactly n pictures, so n+1
    groups can only mean the threshold split one of them, and the two halves are
    the most similar groups present.

    This used to rebuild from the board instead, which discards every learned
    face to fix one wrong grouping. On the live run that motivated the change, 11
    of the 12 groups were correct; the rebuild threw all of them away and then
    spent the level's remaining allowance re-reading cards it had already read.
    Rejoining flips nothing.

    Injected directly, because provoking it through the pipeline would need a
    deliberately mis-set threshold and the point here is the response.
    """
    board = FakeBoard(pairs=2)
    runner, _, _ = make_runner(board)
    runner.seek_board()
    session = runner.session
    assert session is not None
    flood_registry(session, 3)
    assert session.registry.group_count > session.pair_count

    result = runner.play_turn()

    assert result.ok is True
    assert result.code == CODE_REGROUPED
    assert runner.session is not None
    # One group absorbed another, and nothing else was forgotten.
    assert runner.session.registry.group_count == 2
    assert runner.session.regroupings == 1
    assert runner.session.recoveries == 0
    assert board.clicks == []


def test_a_violet_card_face_is_not_counted_as_a_back():
    """The root cause of the stall, not the stall itself.

    Colour segmentation answers "is this violet, square and solid", and a violet
    card face is all three. On a live level 8 board a lavender monster card,
    revealed at slot 13, passed every colour and shape filter -- so that slot read
    as still face down, the settle wait for it and its partner therefore never
    completed, and the scan re-probed the same two cards until the allowance was
    gone.

    Within a level every back is the same picture, which separates them by far
    more than any colour margin: measured on the frame that stalled, the impostor
    scored -0.152 against the other boxes while 24 real backs scored 0.992 to
    1.000.

    Here an extra box carries a picture unrelated to the backs. It must be
    dropped, and the real backs must all survive.

    The backs are textured, and that is not decoration. Correlation between two
    flat patches is undefined, so a flat stand-in makes the gate abstain rather
    than judge -- the real game's spiral pattern is what gives it something to
    measure.
    """
    board = FakeBoard(pairs=3)
    runner, _, _ = make_runner(board)

    rng = np.random.default_rng(31337)
    back = rng.integers(0, 255, size=(CARD, CARD, 3), dtype=np.uint8)
    face = rng.integers(0, 255, size=(CARD, CARD, 3), dtype=np.uint8)

    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)
    boxes = []
    for index in range(3):
        left = 40 + index * (CARD + GAP)
        frame[40:40 + CARD, left:left + CARD] = back
        boxes.append((left, 40, left + CARD, 40 + CARD))

    impostor_left = 40 + 3 * (CARD + GAP)
    frame[40:40 + CARD, impostor_left:impostor_left + CARD] = face
    impostor = (impostor_left, 40, impostor_left + CARD, 40 + CARD)

    class Session:
        back_reference = crop_face(frame, boxes[0], runner._fp)

    kept = runner._only_real_backs(Session(), frame, boxes + [impostor])

    assert impostor not in kept
    # The negative control, in the same test so the two cannot drift apart: a
    # gate that dropped everything would satisfy the line above while blinding
    # the runner, and a rejected real back makes a face-down card look turned
    # over.
    assert sorted(kept) == sorted(boxes)


def test_without_a_reference_no_box_is_dropped():
    """Refusing everything would be far worse than the fault this guards against."""
    board = FakeBoard(pairs=3)
    runner, _, _ = make_runner(board)
    runner.seek_board()
    session = runner.session
    assert session is not None
    session.back_reference = None

    frame, boxes = runner._capture_and_detect()

    assert runner._only_real_backs(session, frame, boxes) == list(boxes)


def test_an_unreadable_card_is_deferred_instead_of_retried_forever():
    """A card whose face will not read must not hold up the whole scan.

    Reproduced from a live 26 card level that repeated the same round until it was
    stopped by hand:

        failed 12 & 13, failed 12 & 13, failed 12 & 13, ...

    The scan always took the two lowest unread slots, and a slot left unread is
    immediately the lowest again, so one unreadable crop cost a mismatch per round
    against a budget of 20. Whatever makes a crop unreadable belongs to the card,
    so waiting does not help.

    Here slot 0 never yields a face. The scan must move past it and read the
    others rather than spending the level on it.
    """
    board = FakeBoard(pairs=4)
    runner, _, _ = make_runner(board, scan_first=True)
    runner.seek_board()
    session = runner.session
    assert session is not None

    # Withhold slot 0's crop, which is what an unreadable card looks like from
    # here: the settle wait returns without it. Patching the settle rather than
    # the registry keeps the signal deterministic -- the registry is asked once
    # per slot with no way to tell which slot it is being asked about.
    real_await = runner._await_settled_faces

    def never_settles_slot_zero(session_, slots, passes):
        crops = real_await(session_, slots, passes)
        crops.pop(0, None)
        return crops

    runner._await_settled_faces = never_settles_slot_zero

    for _ in range(6):
        if runner.session is None or runner.session.solver.is_solved():
            break
        runner.play_turn()

    assert runner.session is not None
    # Slot 0 was tried, gave up its place, and other cards got read anyway.
    assert runner.session.read_failures.get(0, 0) >= 1
    assert runner.session.solver.known_face(0) is None
    assert len(runner.session.solver.unknown_slots()) < 8


def test_a_deferred_card_goes_to_the_back_of_the_scan_order():
    """The ordering rule on its own, without driving a whole level."""
    board = FakeBoard(pairs=4)
    runner, _, _ = make_runner(board, scan_first=True)
    runner.seek_board()
    session = runner.session
    assert session is not None

    before = runner._scan_order(session)
    assert before[0] == 0

    session.read_failures[0] = runner._params.max_read_attempts

    after = runner._scan_order(session)

    assert after[0] != 0
    assert after[-1] == 0
    assert sorted(after) == sorted(before)


def test_a_rejoin_the_game_refuses_is_not_tried_again():
    """Rejoining is a proposal; the game is the authority on pairs.

    Both halves of the user-facing loop are checked here: the rejoined groups make
    the solver offer those slots as a pair, and a refusal must take that pair out
    of circulation. Without the second half an incorrect rejoin is not a one-off
    cost -- the same two slots would be proposed, refused and proposed again until
    the allowance ran out, which is worse than the split grouping it repaired.
    """
    solver = MemorySolver(4)
    solver.learn_face(0, 1)
    solver.learn_face(1, 2)

    # What a rejoin does: two keys become one, so the slots look like a pair.
    solver.rename_face(2, 1)
    assert solver.known_pair() == (0, 1)

    solver.record_pair_result(0, 1, matched=False)

    assert solver.is_rejected(0, 1)
    assert solver.known_pair() is None
    assert solver.partner_for(0) is None


def test_a_finished_card_turning_up_face_down_rebuilds_from_the_board():
    """The case a short match confirmation cannot rule out.

    Confirming a match by watching for a few frames is not proof -- a mismatched
    pair also stays up for over a second -- so the record can end up believing a
    pair was collected when it was not. The board disagreeing is exactly the
    signal, and it arrives for free at the start of the next turn.
    """
    board = FakeBoard(pairs=3)
    runner, _, _ = make_runner(board)
    runner.seek_board()
    session = runner.session
    assert session is not None
    # Claim a pair is done while both cards are in fact still face down.
    session.solver.mark_cleared([0, 1])

    result = runner.play_turn()

    assert result.ok is True
    assert result.code == CODE_RESYNCED
    assert runner.session is not None
    assert 0 in runner.session.solver.active_slots()


def test_a_stray_detection_is_retried_rather_than_believed():
    """One bad frame must not end a level.

    A stray blob belongs to the frame it appeared in, so another frame is the
    whole cure -- and a capture costs no mismatches, which is what makes looking
    again strictly better than either believing it or stopping.

    The first capture here carries one box that matches no slot; every later one
    is clean. The turn must proceed.

    The stray replaces a real box rather than being added to them, because a board
    that already has every slot detected plus one extra trips the earlier
    "more boxes than slots" guard instead. That is also what the live board looked
    like: a pair had just been matched, so there was room under the slot count.
    """
    board = FakeBoard(pairs=3)
    runner, _, _ = make_runner(board)
    runner.seek_board()
    assert runner.session is not None

    real = runner._capture_and_detect
    calls = {"n": 0}

    def with_one_stray():
        calls["n"] += 1
        frame, boxes = real()
        if calls["n"] == 1:
            return frame, list(boxes)[:-1] + [(3000, 3000, 3060, 3060)]
        return frame, boxes

    runner._capture_and_detect = with_one_stray

    result = runner.play_turn()

    assert calls["n"] >= 2, "the stray frame should have been re-captured"
    assert result.code != CODE_GEOMETRY_DRIFT


def test_a_persistent_stray_is_not_ignored_forever():
    """The retries are a tolerance, not a blindfold.

    If the extra box is there on every frame it is not a one-frame artefact, and
    continuing would mean acting on a board that does not match the slot map. The
    run has to end rather than quietly drop the box.
    """
    board = FakeBoard(pairs=3)
    runner, _, _ = make_runner(board)
    runner.seek_board()
    assert runner.session is not None

    real = runner._capture_and_detect
    calls = {"n": 0}

    def always_stray():
        calls["n"] += 1
        frame, boxes = real()
        return frame, list(boxes)[:-1] + [(3000, 3000, 3060, 3060)]

    runner._capture_and_detect = always_stray

    result = runner.play_turn()

    # Retried, then gave up rather than dropping the box.
    assert calls["n"] == runner._params.stray_box_retries + 1
    assert result.ok is False


def test_a_rebuild_the_budget_cannot_pay_for_stops_instead():
    """Rebuilding is only a repair while there are misses left to spend.

    This is a real level loss, reproduced. A rebuild throws away every learned
    face, so every card still in play has to be read again -- and reading costs
    mismatches, which the game rations at ceil(1.5 * pairs). On a live run:

        22 cards / 11 pairs, budget 17. A full scan had spent 11, leaving 6.
        Rebuilding meant re-reading all 22, worst case 11 more. It rebuilt
        anyway, spent the remaining 6, and the level failed.

    Nothing in the code disagreed with that, because the affordability of a
    rebuild was never asked about; only the *number* of rebuilds was capped. The
    margin formula needed for the question already existed one method away, used
    for a smaller decision about verification depth.

    Refusing here is arithmetic rather than caution: past the budget a rebuild
    cannot win the level, while stopping leaves the board playable and spends
    nothing. Modelled at four pairs, where the budget is 6 and a full re-read
    costs 4, so three misses already make it unaffordable.

    The disagreement used to provoke it is a slot recorded as matched that the
    board still shows face down. Too many face groups would not do: that cause is
    repaired by rejoining groups now and never reaches the rebuild.
    """
    board = FakeBoard(pairs=4)
    runner, _, _ = make_runner(board)
    runner.seek_board()
    session = runner.session
    assert session is not None

    session.solver.mismatches = 3
    session.solver.mark_cleared([0, 1])

    result = runner.play_turn()

    assert result.ok is False
    assert result.code == CODE_OUT_OF_BUDGET
    # It stopped without spending anything further, and without rebuilding.
    assert runner.session is not None
    assert runner.session.recoveries == 0
    assert runner.session.solver.mismatches == 3


def test_a_rebuild_the_budget_can_still_pay_for_goes_ahead():
    """The negative control for the check above.

    A guard that never lets anything through would pass the test above while
    breaking recovery entirely, so the affordable case has to be pinned too.
    Same board and the same disagreement, one miss spent instead of three: after
    a rebuild the eight cards cost at most 4 more, and 1 + 4 is inside the budget
    of 6, so the rebuild is allowed.
    """
    board = FakeBoard(pairs=4)
    runner, _, _ = make_runner(board)
    runner.seek_board()
    session = runner.session
    assert session is not None

    session.solver.mismatches = 1
    session.solver.mark_cleared([0, 1])

    result = runner.play_turn()

    assert result.ok is True
    assert result.code == CODE_RESYNCED
    assert runner.session is not None
    assert runner.session.recoveries == 1


def test_repeated_failures_to_agree_stop_the_run():
    """A repair that keeps being needed is a fault, not a hiccup.

    Provoked with a slot recorded as matched that the board still shows face
    down. Too many face groups would not reach the rebuild any more: that cause
    is repaired by rejoining groups, which is the whole point of rejoining.
    """
    board = FakeBoard(pairs=2)
    runner, _, _ = make_runner(board, max_recoveries=1)
    runner.seek_board()
    session = runner.session
    assert session is not None

    session.solver.mark_cleared([0, 1])
    first = runner.play_turn()
    assert first.code == CODE_RESYNCED

    assert runner.session is not None
    runner.session.solver.mark_cleared([0, 1])
    second = runner.play_turn()

    assert second.ok is False
    assert second.code == CODE_UNRECOVERABLE


def test_a_failed_click_is_not_treated_as_a_flip():
    board = FakeBoard(pairs=4)
    runner, _, action = make_runner(board)
    runner.seek_board()

    action.refuse = True
    result = runner.play_turn()

    assert result.ok is False
    # Nothing was recorded about a card that was never turned over.
    assert runner.session is not None
    assert runner.session.solver.face_up == ()


def test_window_scale_is_synced_so_click_jitter_stays_proportional():
    board = FakeBoard(pairs=2)
    runner, _, action = make_runner(board)

    runner.seek_board()

    assert action.scales, "scale was never synced"
    assert all(value > 0 for value in action.scales)


# --- when a run ends --------------------------------------------------------
#
# The run is over when boards stop appearing, not after a fixed count. The results
# screen produces no card-back boxes at all, so "no board within the patience
# window" is how the end announces itself -- and whether that is good news depends
# on whether anything was cleared first.


# Captures a completed board lingers before the next one appears.
#
# Not zero. A first version swapped the board out the instant its last pair
# matched, so the runner was still sampling that final card when the screen went
# empty underneath it -- and read the blank background as "never revealed". The
# real board stays put and fades over roughly a second, which is what this models.
FADE_RENDERS = 6


class Arena:
    """Serves a sequence of boards, then a screen with no cards on it.

    Stands in for a whole run: each board is played out, lingers while it fades,
    the next appears, and once they are exhausted what remains is the results
    screen -- which from the detector's point of view is simply empty.
    """

    def __init__(self, boards: List[FakeBoard], fade: int = FADE_RENDERS) -> None:
        self.boards = boards
        self.index = 0
        self.clicks: List[Tuple[int, int]] = []
        #: Captures served. Every wait in the loop is a run of captures, so this
        #: is a machine-independent clock for comparing how costly a turn is.
        self.captures = 0
        #: Captures a finished board lingers. Adjustable because the live game
        #: swapped boards faster than the loop expected on a four-card level.
        self.fade = fade
        self._fade = 0

    @property
    def current(self) -> Optional[FakeBoard]:
        if self.index >= len(self.boards):
            return None
        return self.boards[self.index]

    def capture(self) -> Optional[np.ndarray]:
        self.captures += 1
        board = self.current
        if board is None:
            # No cards anywhere. Still a real frame, because the results screen is
            # a picture -- just not a board.
            hsv = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)
            hsv[:, :] = BACKGROUND_HSV
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        frame = board.render()
        if len(board.cleared) >= len(board.boxes):
            self._fade += 1
            if self._fade > self.fade:
                self.index += 1
                self._fade = 0
        return frame

    def set_scale(self, scale: float) -> None:
        pass

    def click(self, x: int, y: int) -> bool:
        self.clicks.append((x, y))
        board = self.current
        if board is None:
            return False
        return board.click(x, y)


def make_arena_runner(boards: List[FakeBoard], **overrides):
    arena = Arena(boards, fade=overrides.pop("fade", FADE_RENDERS))
    params = RunnerParams(
        poll_interval=0.0,
        settle_polls=overrides.pop("settle_polls", 10),
        flip_back_polls=overrides.pop("flip_back_polls", 20),
        # Long enough to outlast the fade of the board just finished, so the next
        # one is seen rather than the run being called over prematurely.
        board_polls=overrides.pop("board_polls", 12),
        max_levels=overrides.pop("max_levels", None),
        max_recoveries=overrides.pop("max_recoveries", 2),
        scan_first=overrides.pop("scan_first", False),
        stability_gap=0.0,
    )
    runner = MemoryRunner(
        window=arena,
        action=arena,
        params=params,
        sleep=lambda _seconds: None,
    )
    return runner, arena


class AdjacentPairsBoard(FakeBoard):
    """Laid out AABB..., which is what the live game's four-card level used.

    ``FakeBoard`` deliberately separates partners so tests cannot pass by
    accident. This one deliberately does the opposite, because the order changes
    the *shape* of the level rather than just its difficulty: every scanning turn
    matches, so there is never a flip back to wait out and the level ends almost
    as fast as the clicks can be sent.
    """

    def __init__(self, pairs: int, layout: Optional[List[int]] = None) -> None:
        super().__init__(pairs, layout)
        self.faces = [slot // 2 for slot in range(pairs * 2)]


def test_a_finished_level_is_not_mistaken_for_a_drifted_board():
    """The bug a live four-card AABB board found.

    Both turns matched, so the level was over roughly a second after it started
    and the game put up the next board -- which has a different number of cards.
    A look taken then, against the finished level's slot map, counted six cards in
    four slots and stopped the run with GEOMETRY_DRIFT.

    The record already knew the answer. Every pair had been matched, so there was
    nothing left to photograph and no reason to look at all. Modelled with no
    fade, which is the worst case for the race.
    """
    boards = [AdjacentPairsBoard(pairs=2), FakeBoard(pairs=3)]
    runner, arena = make_arena_runner(
        boards, fade=0, scan_first=True, max_levels=None
    )

    result = runner.run()

    assert result.ok, f"{result.code} - {result.message}"
    assert result.code == CODE_RUN_COMPLETE
    assert runner.levels_cleared == 2
    assert all(len(board.cleared) == len(board.boxes) for board in arena.boards)


def test_an_aabb_board_matches_on_every_scanning_turn():
    """Documents why that layout is the fast one, so the fixture is not tidied
    away as an arbitrary variation."""
    runner, arena = make_arena_runner(
        [AdjacentPairsBoard(pairs=2)], fade=0, scan_first=True
    )

    runner.seek_board()
    codes = []
    for _ in range(6):
        result = runner.step()
        codes.append(result.code)
        if result.code in {CODE_LEVEL_CLEARED, CODE_RUN_COMPLETE} or not result.ok:
            break

    assert CODE_TURN_MISSED not in codes, codes
    assert arena.boards[0].cleared == {0, 1, 2, 3}


def test_the_run_ends_when_boards_stop_appearing():
    """One board, then nothing: a finished run, not an error."""
    runner, _ = make_arena_runner([FakeBoard(pairs=2)])

    result = runner.run()

    assert result.ok is True
    assert result.code == CODE_RUN_COMPLETE
    assert runner.levels_cleared == 1


def test_every_board_offered_is_played():
    """No level count is assumed; the game decides how long a run is."""
    runner, arena = make_arena_runner([FakeBoard(pairs=2), FakeBoard(pairs=3), FakeBoard(pairs=2)])

    result = runner.run()

    assert result.code == CODE_RUN_COMPLETE
    assert runner.levels_cleared == 3
    assert all(len(board.cleared) == len(board.boxes) for board in arena.boards)


def test_no_board_and_nothing_cleared_is_a_failure():
    """Told apart from a finished run by history.

    Nothing cleared means we were never on a board, which is a user-facing
    mistake -- the script has to be started at a level opening.
    """
    runner, arena = make_arena_runner([])

    result = runner.run()

    assert result.ok is False
    assert result.code == CODE_NO_BOARD
    assert arena.clicks == []


def test_an_explicit_limit_is_still_honoured():
    """A short trial run remains possible, for a first cautious live attempt."""
    runner, arena = make_arena_runner(
        [FakeBoard(pairs=2), FakeBoard(pairs=2)], max_levels=1
    )

    result = runner.run()

    assert result.code == CODE_RUN_COMPLETE
    assert runner.levels_cleared == 1
    # The second board was never touched.
    assert arena.boards[1].cleared == set()


def test_turns_are_counted_for_the_ui():
    runner, _ = make_arena_runner([FakeBoard(pairs=3)])

    runner.run()

    assert runner.turns_played >= 3


def test_each_level_is_timed():
    """Timing is what settles which flip order is actually faster.

    Arithmetic can compare misses but not seconds, because the dominant cost per
    miss is a flip-back animation the game imposes regardless of order.
    """
    runner, _ = make_arena_runner([FakeBoard(pairs=2), FakeBoard(pairs=2)])

    result = runner.run()

    assert len(runner.level_seconds) == 2
    assert all(seconds >= 0 for seconds in runner.level_seconds)
    # The summary is what the user reads, so it has to carry the timings and
    # not just the verdict. Checked by finding the formatted decimals: the
    # wording is Chinese, and asserting on display text is both unstable and
    # what put a non-ASCII byte in this file in the first place.
    timings = [word for word in result.message.split() if "." in word]
    assert timings, result.message


# --- the two flip orders ----------------------------------------------------
#
# The fake board pairs slot i with slot i + n, so partners never share a scanning
# turn. That is the worst case for scanning and makes the difference between the
# orders visible rather than incidental.


def test_both_orders_solve_the_board():
    for scan_first in (False, True):
        runner, arena = make_arena_runner([FakeBoard(pairs=4)], scan_first=scan_first)

        runner.run()

        assert arena.boards[0].cleared == set(range(8)), f"scan_first={scan_first}"


def test_scanning_first_costs_more_turns_and_more_misses():
    """The arithmetic, checked through the real loop.

    A miss needs the first card flipped to be the first copy of its picture seen;
    the opening turn burns two such cards and every later miss burns at least one,
    which caps the opportunistic order at n - 1. Scanning declines to complete the
    pairs it walks into, so on this layout all n scanning turns miss.
    """
    opportunistic, _ = make_arena_runner([FakeBoard(pairs=4)])
    opportunistic.run()

    scanning, _ = make_arena_runner([FakeBoard(pairs=4)], scan_first=True)
    scanning.run()

    assert scanning.turns_played > opportunistic.turns_played
    assert opportunistic.turns_played == 6  # 2 probes + 4 collections
    assert scanning.turns_played == 8  # 4 scans + 4 collections


def test_scanning_reads_every_card_before_collecting_any():
    """The defining property of the order, not just its cost.

    Nothing is collected while cards remain unread, so the first half of the turns
    are all misses on this layout and the second half are all collections.
    """
    runner, arena = make_arena_runner([FakeBoard(pairs=3)], scan_first=True)
    board = arena.boards[0]
    cleared_after_each_turn = []

    runner.seek_board()
    for _ in range(12):
        result = runner.step()
        if not result.ok or result.code == CODE_RUN_COMPLETE:
            break
        cleared_after_each_turn.append(len(board.cleared))

    # Nothing was collected until the scan finished, so the first three turns
    # left the board untouched.
    assert cleared_after_each_turn[:3] == [0, 0, 0]
    assert board.cleared == set(range(6))


# --- what the scanning phase can do that the other order cannot -------------
#
# Deciding that a card has stopped moving is the expensive part of a turn, and one
# sequence of frames answers it for every card that is up. A scanning turn is two
# independent reads, so it can use that; an ordinary turn cannot, because its
# second card is chosen from what the first one turned out to be.


def spy_on_settling(runner, monkeypatch) -> List[Tuple[int, ...]]:
    """Record the slots of every settling wait the runner performs."""
    waits: List[Tuple[int, ...]] = []
    original = runner._await_settled_faces

    def recorder(session, slots, passes):
        waits.append(tuple(slots))
        return original(session, slots, passes)

    monkeypatch.setattr(runner, "_await_settled_faces", recorder)
    return waits


def test_a_scanning_turn_settles_once_for_both_cards(monkeypatch):
    runner, _ = make_arena_runner([FakeBoard(pairs=4)], scan_first=True)
    runner.seek_board()
    waits = spy_on_settling(runner, monkeypatch)

    result = runner.step()

    assert result.ok, result.code
    assert len(waits) == 1, waits
    assert len(waits[0]) == 2, waits


def test_an_ordinary_turn_settles_once_per_card(monkeypatch):
    """The contrast, and the reason it is not a missed optimisation there.

    The second card is picked after the first has been read, because that is what
    lets the solver offer a partner instead of just another unseen card.
    """
    runner, _ = make_arena_runner([FakeBoard(pairs=4)])
    runner.seek_board()
    waits = spy_on_settling(runner, monkeypatch)

    result = runner.step()

    assert result.ok, result.code
    assert len(waits) == 2, waits
    assert all(len(slots) == 1 for slots in waits), waits


def test_a_scanning_turn_costs_fewer_captures_than_an_ordinary_one():
    """The same saving seen from outside, on the only clock the fakes have.

    Wall time is unusable here because the injected sleep does nothing, but every
    wait in the loop is a run of captures, so counting them is machine
    independent.
    """
    scanning, scan_arena = make_arena_runner([FakeBoard(pairs=4)], scan_first=True)
    scanning.seek_board()
    scan_looks = count_looks_for_one_turn(scanning, scan_arena)

    ordinary, plain_arena = make_arena_runner([FakeBoard(pairs=4)])
    ordinary.seek_board()
    plain_looks = count_looks_for_one_turn(ordinary, plain_arena)

    assert scan_looks < plain_looks, (scan_looks, plain_looks)


def count_looks_for_one_turn(runner, arena) -> int:
    """Captures consumed by a single turn."""
    before = arena.captures
    result = runner.step()
    assert result.ok, result.code
    return arena.captures - before


def test_a_watched_flip_back_is_not_followed_by_another_wait_for_quiet():
    """Waiting for stillness right after watching a pair go face down measures
    nothing, because that wait is what stillness looks like.

    Asserted through the session flag rather than by timing, so it says what the
    loop believes rather than how long it happened to take.
    """
    board = FakeBoard(pairs=4)
    runner, _, _ = make_runner(board)
    runner.seek_board()
    session = runner.session

    assert session.board_is_quiet is False, "nothing has been established yet"

    # Probe until a turn misses, which is the case that ends in a flip back.
    for _ in range(6):
        result = runner.play_turn()
        assert result.ok, result.code
        if result.code == CODE_TURN_MISSED:
            break
    else:  # pragma: no cover - a four-pair board always misses early
        pytest.fail("no turn missed")

    assert session.board_is_quiet is True


def test_a_match_leaves_the_board_unquiet():
    """The celebration is an animation, so the next sample has to wait it out."""
    board = FakeBoard(pairs=4)
    runner, _, _ = make_runner(board)
    runner.seek_board()
    session = runner.session

    for _ in range(12):
        result = runner.play_turn()
        assert result.ok, result.code
        if result.code == CODE_TURN_MATCHED:
            break
    else:  # pragma: no cover - the board is always solved within 12 turns
        pytest.fail("no turn matched")

    assert session.board_is_quiet is False


@pytest.mark.parametrize("pairs", [2, 3, 6, 8])
def test_levels_of_several_sizes_are_all_solved(pairs):
    board = FakeBoard(pairs=pairs)
    runner, _, _ = make_runner(board)

    runner.run()

    assert board.cleared == set(range(pairs * 2))


def test_a_two_card_board_is_below_the_gate_floor():
    """Documents where the floor comes from, so nobody lowers it casually.

    The game's smallest level holds four cards. The floor is what rejects the
    three violet blobs the island view produces, so admitting a two-card board
    would trade a real protection for a case the game never presents.
    """
    board = FakeBoard(pairs=1)
    runner, _, _ = make_runner(board)

    result = runner.seek_board()

    assert result.ok is False
    assert result.code == CODE_NO_BOARD
    assert board.clicks == []
