"""Memory-game solver logic, exercised without any pixels.

The solver receives opaque face keys, so a plain string stands in for whatever
the fingerprinter will eventually produce. Everything here is deterministic.
"""

from __future__ import annotations

import random

import pytest

from core.minigames.grid import reading_order
from core.minigames.memory_game import MemorySolver


def make_board(pairs, seed):
    """A shuffled board of ``pairs`` pairs, as a list of face keys per slot."""
    faces = [chr(ord("A") + i) for i in range(pairs)] * 2
    random.Random(seed).shuffle(faces)
    return faces


def play(faces, max_turns=500):
    """Drive a full game against a perfect fingerprinter."""
    solver = MemorySolver(len(faces))
    turns = 0
    while not solver.is_solved():
        turns += 1
        assert turns <= max_turns, "求解未收敛，可能陷入死循环"

        first = solver.next_flip()
        assert first is not None, f"无牌可翻但棋盘未清空: {solver.summary()}"
        solver.record_flip(first, faces[first])

        second = solver.next_flip()
        assert second is not None, f"翻开第一张后无第二张可选: {solver.summary()}"
        solver.record_flip(second, faces[second])

        solver.record_pair_result(first, second, faces[first] == faces[second])
    return solver


class TestConstruction:
    def test_negative_slot_count_is_rejected(self):
        with pytest.raises(ValueError):
            MemorySolver(-2)

    def test_odd_slot_count_warns_about_missed_detection(self, caplog):
        with caplog.at_level("WARNING"):
            MemorySolver(9)
        assert "奇数" in caplog.text

    def test_empty_board_is_already_solved(self):
        solver = MemorySolver(0)
        assert solver.is_solved()
        assert solver.next_flip() is None
        assert not solver.is_stuck()

    def test_fresh_board_knows_nothing(self):
        solver = MemorySolver(6)
        assert solver.active_slots() == [0, 1, 2, 3, 4, 5]
        assert solver.unknown_slots() == [0, 1, 2, 3, 4, 5]
        assert solver.face_up == ()
        assert solver.known_pair() is None


class TestFlipPriority:
    def test_first_flip_is_the_top_left_card(self):
        assert MemorySolver(8).next_flip() == 0

    def test_probing_walks_the_board_in_reading_order(self):
        solver = MemorySolver(6)
        solver.record_flip(0, "A")
        # No known partner for A yet, so the probe is the next unseen slot.
        assert solver.next_flip() == 1

    def test_cleared_slots_are_skipped_when_probing(self):
        solver = MemorySolver(6)
        solver.record_flip(0, "A")
        solver.record_flip(1, "A")
        solver.record_pair_result(0, 1, True)
        assert solver.next_flip() == 2

    def test_already_seen_slots_are_not_reprobed(self):
        solver = MemorySolver(6)
        solver.record_flip(0, "A")
        solver.record_flip(1, "B")
        solver.record_pair_result(0, 1, False)
        # 0 and 1 are known but unmatched; the next probe moves on to 2.
        assert solver.next_flip() == 2


class TestKnownPairsArePreferred:
    def test_known_pair_is_taken_before_probing(self):
        solver = MemorySolver(6)
        solver.learn_face(3, "X")
        solver.learn_face(5, "X")
        # Slot 0 is unseen, but clearing a known pair costs no miss.
        assert solver.next_flip() == 3
        assert solver.known_pair() == (3, 5)

    def test_second_flip_completes_a_known_pair(self):
        solver = MemorySolver(6)
        solver.learn_face(4, "A")
        solver.record_flip(0, "A")
        assert solver.next_flip() == 4

    def test_known_pair_with_lowest_index_wins(self):
        solver = MemorySolver(8)
        solver.learn_face(5, "X")
        solver.learn_face(7, "X")
        solver.learn_face(1, "Y")
        solver.learn_face(6, "Y")
        assert solver.known_pair() == (1, 6)
        assert solver.next_flip() == 1

    def test_partner_lookup_ignores_the_card_already_face_up(self):
        solver = MemorySolver(4)
        solver.record_flip(0, "A")
        # The only card known to show "A" is the one already up.
        assert solver.partner_for(0) is None

    def test_partner_lookup_ignores_cleared_cards(self):
        solver = MemorySolver(6)
        solver.record_flip(0, "A")
        solver.record_flip(1, "A")
        solver.record_pair_result(0, 1, True)
        solver.learn_face(2, "A")
        assert solver.partner_for(2) is None


class TestTurnBookkeeping:
    def test_match_clears_both_slots(self):
        solver = MemorySolver(4)
        solver.record_flip(0, "A")
        solver.record_flip(2, "A")
        solver.record_pair_result(0, 2, True)
        assert solver.active_slots() == [1, 3]
        assert solver.matches == 1
        assert solver.mismatches == 0
        assert solver.face_up == ()

    def test_miss_is_counted_and_keeps_both_cards_on_the_board(self):
        solver = MemorySolver(4)
        solver.record_flip(0, "A")
        solver.record_flip(1, "B")
        solver.record_pair_result(0, 1, False)
        assert solver.active_slots() == [0, 1, 2, 3]
        assert solver.mismatches == 1
        assert solver.turns == 1

    def test_faces_survive_a_miss(self):
        # This is the whole game: remembering what a failed turn revealed.
        solver = MemorySolver(4)
        solver.record_flip(0, "A")
        solver.record_flip(1, "B")
        solver.record_pair_result(0, 1, False)
        assert solver.known_face(0) == "A"
        assert solver.known_face(1) == "B"
        assert solver.unknown_slots() == [2, 3]

    def test_learned_face_completes_a_pair_on_a_later_turn(self):
        solver = MemorySolver(4)
        solver.record_flip(0, "A")
        solver.record_flip(1, "B")
        solver.record_pair_result(0, 1, False)
        solver.record_flip(2, "B")
        assert solver.next_flip() == 1

    def test_flipping_the_same_slot_twice_in_a_turn_is_rejected(self):
        solver = MemorySolver(4)
        solver.record_flip(0, "A")
        with pytest.raises(ValueError):
            solver.record_pair_result(0, 0, False)

    def test_third_flip_in_one_turn_is_rejected(self):
        solver = MemorySolver(6)
        solver.record_flip(0, "A")
        solver.record_flip(1, "B")
        with pytest.raises(ValueError):
            solver.record_flip(2, "C")

    def test_no_flip_offered_while_a_turn_is_unresolved(self):
        solver = MemorySolver(6)
        solver.record_flip(0, "A")
        solver.record_flip(1, "B")
        assert solver.next_flip() is None
        assert not solver.is_stuck()  # waiting for a result, not deadlocked

    def test_re_recording_the_same_slot_updates_its_face(self):
        # A retried fingerprint should be allowed to correct an earlier read.
        solver = MemorySolver(4)
        solver.record_flip(0, "A")
        solver.record_flip(0, "B")
        assert solver.known_face(0) == "B"
        assert solver.face_up == (0,)


class TestInvalidSlots:
    def test_out_of_range_slot_is_rejected(self):
        solver = MemorySolver(4)
        with pytest.raises(IndexError):
            solver.record_flip(4, "A")

    def test_cleared_slot_cannot_be_touched_again(self):
        solver = MemorySolver(4)
        solver.record_flip(0, "A")
        solver.record_flip(1, "A")
        solver.record_pair_result(0, 1, True)
        with pytest.raises(ValueError):
            solver.record_flip(0, "A")

    def test_none_is_not_a_valid_face_key(self):
        # A failed fingerprint must be reported, not silently stored.
        solver = MemorySolver(4)
        with pytest.raises(ValueError):
            solver.record_flip(0, None)


class TestPreviewReveal:
    def test_learn_face_does_not_mark_a_card_as_flipped(self):
        solver = MemorySolver(4)
        solver.learn_face(0, "A")
        assert solver.face_up == ()
        assert solver.known_face(0) == "A"

    def test_a_full_preview_lets_every_pair_be_taken_without_a_miss(self):
        faces = make_board(4, seed=7)
        solver = MemorySolver(len(faces))
        for index, key in enumerate(faces):
            solver.learn_face(index, key)

        while not solver.is_solved():
            first = solver.next_flip()
            solver.record_flip(first, faces[first])
            second = solver.next_flip()
            solver.record_flip(second, faces[second])
            solver.record_pair_result(first, second, faces[first] == faces[second])

        assert solver.mismatches == 0
        assert solver.matches == 4


class TestRecovery:
    def test_inconsistent_knowledge_is_reported_as_stuck(self):
        # Every card seen, no two keys equal: fingerprinting split the pairs.
        solver = MemorySolver(4)
        for index, key in enumerate("ABCD"):
            solver.learn_face(index, key)
        assert solver.next_flip() is None
        assert solver.is_stuck()

    def test_forgetting_faces_recovers_from_a_stuck_board(self):
        solver = MemorySolver(4)
        for index, key in enumerate("ABCD"):
            solver.learn_face(index, key)
        solver.forget_faces()
        assert not solver.is_stuck()
        assert solver.next_flip() == 0
        assert solver.unknown_slots() == [0, 1, 2, 3]

    def test_forgetting_faces_keeps_cleared_slots(self):
        solver = MemorySolver(4)
        solver.record_flip(0, "A")
        solver.record_flip(1, "A")
        solver.record_pair_result(0, 1, True)
        solver.forget_faces()
        assert solver.active_slots() == [2, 3]

    def test_mark_cleared_resyncs_vanished_cards(self):
        solver = MemorySolver(6)
        solver.record_flip(0, "A")
        solver.mark_cleared([0, 3])
        assert solver.active_slots() == [1, 2, 4, 5]
        assert solver.face_up == ()

    def test_mark_cleared_validates_indices(self):
        solver = MemorySolver(4)
        with pytest.raises(IndexError):
            solver.mark_cleared([9])


class TestFullGames:
    @pytest.mark.parametrize("pairs", [1, 2, 3, 4, 6, 8, 12])
    def test_boards_of_various_sizes_are_solved(self, pairs):
        solver = play(make_board(pairs, seed=pairs * 13))
        assert solver.is_solved()
        assert solver.matches == pairs

    @pytest.mark.parametrize("seed", range(25))
    def test_random_boards_never_exceed_one_miss_per_pair(self, seed):
        # Each miss reveals exactly two previously unseen cards, and knowledge
        # is never discarded, so misses cannot outnumber the pairs.
        pairs = 8
        solver = play(make_board(pairs, seed=seed))
        assert solver.is_solved()
        assert solver.mismatches <= pairs
        assert solver.turns == solver.matches + solver.mismatches

    def test_a_solved_board_offers_no_further_flips(self):
        solver = play(make_board(3, seed=99))
        assert solver.next_flip() is None
        assert not solver.is_stuck()

    def test_summary_reports_progress(self):
        solver = play(make_board(2, seed=5))
        text = solver.summary()
        assert "共 4 张" in text
        assert "2 对" in text


class TestGridIntegration:
    def test_flip_order_follows_the_recovered_reading_order(self):
        # grid gives the index mapping, the solver gives the flip sequence;
        # together the first flip must land on the visually top-left card.
        boxes = [
            (300, 200, 360, 280),  # top-right
            (100, 200, 160, 280),  # top-left
            (100, 320, 160, 400),  # bottom-left
            (300, 320, 360, 400),  # bottom-right
        ]
        order = reading_order(boxes)
        assert order == [1, 0, 2, 3]

        solver = MemorySolver(len(order))
        first_slot = solver.next_flip()
        assert first_slot == 0
        # Slot 0 maps back to the detection that is visually top-left.
        assert boxes[order[first_slot]] == (100, 200, 160, 280)
