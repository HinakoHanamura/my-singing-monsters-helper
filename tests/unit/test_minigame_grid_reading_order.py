"""Reading-order recovery from irregular card layouts.

What matters here is that an index means the same card every frame, and that
uneven rows (the board layout the user described) do not scramble the order.
"""

from __future__ import annotations

import random

from core.minigames.grid import (
    box_center,
    default_row_tolerance,
    describe_grid,
    group_rows,
    reading_order,
)

CARD_W, CARD_H = 60, 80


def make_box(cx, cy, w=CARD_W, h=CARD_H):
    """Box centred on (cx, cy), matching how a detector reports a card."""
    return (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)


def build_layout(rows):
    """Boxes for a layout given as [(y, [x, x, ...]), ...].

    Returns ``(shuffled_boxes, expected_reading_order)``. Shuffling is the
    point: a detector returns cards in score order, so the module must not rely
    on input order.
    """
    expected = [make_box(x, y) for y, xs in rows for x in xs]
    shuffled = list(expected)
    random.Random(20260828).shuffle(shuffled)
    return shuffled, expected


def order_as_boxes(boxes):
    return [boxes[i] for i in reading_order(boxes)]


class TestDegenerateInput:
    def test_empty_input_yields_no_rows(self):
        assert group_rows([]) == []
        assert reading_order([]) == []

    def test_single_card_is_its_own_row(self):
        boxes = [make_box(100, 100)]
        assert group_rows(boxes) == [[0]]
        assert reading_order(boxes) == [0]

    def test_zero_height_boxes_fall_back_to_minimum_tolerance(self):
        # Malformed input must not crash or divide by zero.
        boxes = [(10, 50, 70, 50), (100, 50, 160, 50)]
        assert default_row_tolerance(boxes) >= 1.0
        assert len(group_rows(boxes)) == 1

    def test_describe_grid_reports_empty_board(self):
        assert describe_grid([]) == "未检测到卡牌"


class TestRegularGrid:
    def test_shuffled_grid_recovers_reading_order(self):
        boxes, expected = build_layout(
            [(100, [100, 200, 300]), (220, [100, 200, 300])]
        )
        assert order_as_boxes(boxes) == expected

    def test_rows_run_top_to_bottom(self):
        boxes, _ = build_layout([(400, [100]), (100, [100]), (250, [100])])
        centers = [box_center(b)[1] for b in order_as_boxes(boxes)]
        assert centers == sorted(centers)

    def test_within_row_runs_left_to_right(self):
        boxes, _ = build_layout([(100, [500, 100, 300, 200])])
        centers = [box_center(b)[0] for b in order_as_boxes(boxes)]
        assert centers == sorted(centers)

    def test_row_count_matches_layout(self):
        boxes, _ = build_layout(
            [(100, [100, 200]), (220, [100, 200]), (340, [100, 200])]
        )
        assert len(group_rows(boxes)) == 3


class TestIrregularLayout:
    def test_rows_of_different_length_keep_their_shape(self):
        # The user's board is not a rectangle; row sizes must survive intact.
        boxes, expected = build_layout(
            [
                (100, [100, 200, 300, 400]),
                (220, [100, 200, 300, 400]),
                (340, [150, 250]),
            ]
        )
        rows = group_rows(boxes)
        assert [len(r) for r in rows] == [4, 4, 2]
        assert order_as_boxes(boxes) == expected

    def test_jitter_within_tolerance_stays_one_row(self):
        # Cards nudged a few pixels off a shared baseline are still one row.
        boxes, _ = build_layout([(100, [100]), (108, [200]), (94, [300])])
        rows = group_rows(boxes)
        assert len(rows) == 1
        assert order_as_boxes(boxes) == [
            make_box(100, 100),
            make_box(200, 108),
            make_box(300, 94),
        ]

    def test_gradual_slant_does_not_split_a_row(self):
        # Running-mean banding tolerates accumulated drift that a fixed
        # first-card reference would eventually break on.
        slanted = [make_box(100 + 100 * i, 100 + 8 * i) for i in range(5)]
        assert len(group_rows(slanted)) == 1

    def test_separated_rows_are_split(self):
        # A full card pitch between rows must read as two rows.
        boxes = [make_box(100, 100), make_box(100, 100 + CARD_H + 20)]
        assert len(group_rows(boxes)) == 2

    def test_identical_centers_are_ordered_deterministically(self):
        boxes = [make_box(100, 100), make_box(100, 100)]
        assert reading_order(boxes) == [0, 1]


class TestTolerance:
    def test_default_tolerance_tracks_card_height(self):
        small = [make_box(100, 100, h=40)]
        large = [make_box(100, 100, h=120)]
        assert default_row_tolerance(large) > default_row_tolerance(small)

    def test_default_tolerance_uses_median_not_mean(self):
        # One oversized false positive must not stretch the band.
        boxes = [make_box(100, 100, h=80) for _ in range(5)]
        boxes.append(make_box(500, 500, h=1000))
        assert default_row_tolerance(boxes) == 40.0

    def test_explicit_tolerance_can_force_a_split(self):
        boxes = [make_box(100, 100), make_box(200, 120)]
        assert len(group_rows(boxes, row_tolerance=100)) == 1
        assert len(group_rows(boxes, row_tolerance=5)) == 2


class TestScaleInvariance:
    def test_order_is_unchanged_when_the_window_is_resized(self):
        # Same board, window scaled 1.5x: card height scales too, so the
        # derived tolerance scales with it and the order must be identical.
        boxes, _ = build_layout([(100, [100, 200, 300]), (220, [100, 200, 300])])
        scaled = [tuple(int(v * 1.5) for v in b) for b in boxes]
        assert reading_order(boxes) == reading_order(scaled)


class TestDiagnostics:
    def test_describe_grid_reports_rows_and_total(self):
        boxes, _ = build_layout([(100, [100, 200, 300]), (220, [100, 200])])
        text = describe_grid(boxes)
        assert "2 行" in text
        assert "3/2" in text
        assert "5 张" in text
