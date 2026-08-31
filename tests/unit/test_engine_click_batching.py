"""Batching: click several targets per round, but only independent ones.

The distinction this enforces
----------------------------
Stacked plaques must be peeled one per round. A front plaque hides the ones
behind it, so their positions are not yet known; firing at them would fire at
stale coordinates or at whatever the occluded match happened to score.

Plaques that do not overlap are independent - clicking one cannot move or reveal
another - so making each wait for its own capture-and-detect cycle wastes the
majority of a round.

`select_independent` is the whole decision, kept as a plain function so it can be
tested without an engine, a window, or a Qt event loop.
"""

from __future__ import annotations

import pytest

from core.bot_engine import inflate, select_independent
from core.vision_agent import Detection

# The measured icon geometry: 50x38 at the reference resolution, with distinct
# plaques never closer than about 39px centre to centre.
BOX_W, BOX_H = 50, 38


def detection_at(center, confidence=0.9, size=(BOX_W, BOX_H)) -> Detection:
    return Detection(
        name="coin",
        x=center[0] - size[0] // 2,
        y=center[1] - size[1] // 2,
        width=size[0],
        height=size[1],
        confidence=confidence,
    )


def by_confidence(*detections):
    return sorted(detections, key=lambda d: d.confidence, reverse=True)


class TestInflate:
    def test_grows_on_every_side(self):
        assert inflate((10, 10, 20, 20), 5) == (5, 5, 25, 25)

    def test_zero_margin_is_identity(self):
        assert inflate((10, 10, 20, 20), 0) == (10, 10, 20, 20)


class TestIndependentTargetsAreBatched:
    def test_widely_separated_targets_all_get_picked(self):
        picks = select_independent(
            by_confidence(
                detection_at((200, 200), 0.95),
                detection_at((600, 200), 0.90),
                detection_at((200, 600), 0.85),
            ),
            limit=3,
            margin=6,
        )
        assert len(picks) == 3

    def test_the_limit_is_respected(self):
        candidates = by_confidence(
            *[detection_at((200 + 200 * i, 300), 0.9 - 0.01 * i) for i in range(6)]
        )
        assert len(select_independent(candidates, limit=3, margin=6)) == 3

    def test_the_highest_confidence_target_is_always_first(self):
        picks = select_independent(
            by_confidence(
                detection_at((600, 200), 0.70),
                detection_at((200, 200), 0.99),
            ),
            limit=3,
            margin=6,
        )
        assert picks[0].confidence == pytest.approx(0.99)


class TestStackedTargetsAreDeferred:
    def test_overlapping_boxes_yield_only_one_pick(self):
        """Two plaques 20px apart overlap heavily; only the clearest may be clicked."""
        picks = select_independent(
            by_confidence(
                detection_at((400, 300), 0.95),
                detection_at((420, 300), 0.80),
            ),
            limit=3,
            margin=6,
        )
        assert len(picks) == 1
        assert picks[0].confidence == pytest.approx(0.95)

    def test_the_measured_minimum_plaque_spacing_still_counts_as_overlap(self):
        """39px apart with 50px-wide boxes: they overlap, so peel them one by one."""
        picks = select_independent(
            by_confidence(
                detection_at((400, 300), 0.95),
                detection_at((439, 300), 0.85),
            ),
            limit=3,
            margin=6,
        )
        assert len(picks) == 1

    def test_boxes_that_merely_touch_are_treated_as_interfering(self):
        """A collection animation on one can bleed over a neighbour, so pad first."""
        touching = BOX_W  # exactly edge to edge
        picks = select_independent(
            by_confidence(
                detection_at((400, 300), 0.95),
                detection_at((400 + touching, 300), 0.85),
            ),
            limit=3,
            margin=6,
        )
        assert len(picks) == 1

    def test_a_zero_margin_allows_edge_to_edge_batching(self):
        """Documents what the margin actually buys, by removing it."""
        picks = select_independent(
            by_confidence(
                detection_at((400, 300), 0.95),
                detection_at((400 + BOX_W, 300), 0.85),
            ),
            limit=3,
            margin=0,
        )
        assert len(picks) == 2

    def test_vertical_stacking_is_detected_too(self):
        picks = select_independent(
            by_confidence(
                detection_at((400, 300), 0.95),
                detection_at((400, 320), 0.85),
            ),
            limit=3,
            margin=6,
        )
        assert len(picks) == 1

    def test_a_mixed_scene_batches_the_free_ones_and_defers_the_stack(self):
        picks = select_independent(
            by_confidence(
                detection_at((400, 300), 0.95),   # front of a stack
                detection_at((415, 310), 0.88),   # behind it -> deferred
                detection_at((800, 300), 0.86),   # independent
                detection_at((810, 305), 0.80),   # behind that one -> deferred
            ),
            limit=3,
            margin=6,
        )
        centers = [p.center for p in picks]
        assert len(picks) == 2
        assert (400, 300) in centers
        assert (800, 300) in centers


class TestDegenerateInputs:
    def test_limit_of_one_returns_a_single_pick(self):
        candidates = by_confidence(
            detection_at((200, 200), 0.95),
            detection_at((900, 500), 0.90),
        )
        assert len(select_independent(candidates, limit=1, margin=6)) == 1

    def test_limit_below_one_is_treated_as_one(self):
        """A nonsensical setting must not stop the bot clicking entirely."""
        candidates = by_confidence(detection_at((200, 200), 0.95))
        assert len(select_independent(candidates, limit=0, margin=6)) == 1

    def test_no_candidates_yields_nothing(self):
        assert select_independent([], limit=3, margin=6) == []

    def test_a_single_candidate_is_returned(self):
        assert len(select_independent([detection_at((300, 300))], limit=3, margin=6)) == 1

    def test_identical_positions_collapse_to_one(self):
        picks = select_independent(
            by_confidence(
                detection_at((400, 300), 0.95),
                detection_at((400, 300), 0.94),
            ),
            limit=3,
            margin=6,
        )
        assert len(picks) == 1


class TestThroughputClaim:
    def test_a_spread_out_island_collects_several_per_round(self):
        """The point of batching: a busy, spread-out scene should not be serialised."""
        candidates = by_confidence(
            *[
                detection_at((150 + 150 * (i % 5), 200 + 120 * (i // 5)), 0.9 - 0.01 * i)
                for i in range(10)
            ]
        )
        picks = select_independent(candidates, limit=4, margin=6)
        assert len(picks) == 4

    def test_a_single_tight_stack_still_collects_one_per_round(self):
        """And the safety property: a pile must not be batched."""
        candidates = by_confidence(
            *[detection_at((400 + 8 * i, 300 + 5 * i), 0.9 - 0.01 * i) for i in range(8)]
        )
        assert len(select_independent(candidates, limit=4, margin=6)) == 1
