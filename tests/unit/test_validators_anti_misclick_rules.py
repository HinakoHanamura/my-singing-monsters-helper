"""Anti-misclick rule chain.

Assertions key off ``Verdict.code`` rather than the displayed message, so
rewording a log string never breaks a test. The chain short-circuits, and the
ordering of rules is part of the contract: cheap and high-yield rules run first.
"""

from __future__ import annotations

import pytest

from config import SafetyConfig
from core.geometry import scale_factor
from core.validators import (
    MinConfidence,
    NeighborBelow,
    NoOverlapWith,
    NotInExclusionZones,
    NotNearEdge,
    RuleChain,
    TextureBelow,
    ValidationContext,
    build_rule_chain,
)
from core.vision_agent import Detection
from tests.synthetic import REFERENCE_SIZE, make_scene, render_at, scale_point

COIN_AT = (900, 400)


def detection_at(center, size=40, confidence=0.97, name="coin") -> Detection:
    return Detection(
        name=name,
        x=center[0] - size // 2,
        y=center[1] - size // 2,
        width=size,
        height=size,
        confidence=confidence,
    )


@pytest.fixture
def scene_with_blob():
    """A coin sitting above a high-contrast blob (stands in for a monster)."""
    return make_scene(*REFERENCE_SIZE, coin_centers=[COIN_AT], monster_under=[COIN_AT])


@pytest.fixture
def scene_flat():
    """A coin on flat, empty ground."""
    return make_scene(*REFERENCE_SIZE, coin_centers=[COIN_AT])


def context_for(frame, size=REFERENCE_SIZE, detections=None) -> ValidationContext:
    return ValidationContext(
        frame=frame,
        client_size=size,
        scale=scale_factor(size, REFERENCE_SIZE),
        detections=detections or {},
    )


class TestExclusionZones:
    """The most reliable defence: never click inside UI furniture."""

    @pytest.fixture
    def rule(self):
        return NotInExclusionZones(SafetyConfig().exclusion_zones)

    def test_playfield_centre_is_allowed(self, rule, scene_with_blob):
        verdict = rule.check(detection_at(COIN_AT), context_for(scene_with_blob))
        assert verdict.ok

    @pytest.mark.parametrize(
        "center, description",
        [
            ((900, 10), "top status bar"),
            ((900, 930), "bottom button bar"),
            ((180, 400), "left island list"),
            ((1690, 20), "top-right window controls"),
        ],
    )
    def test_ui_regions_are_rejected(self, rule, scene_with_blob, center, description):
        verdict = rule.check(detection_at(center), context_for(scene_with_blob))
        assert not verdict.ok, description
        assert verdict.code == "EXCLUSION_ZONE"

    def test_a_box_merely_overlapping_a_zone_is_rejected(self, rule, scene_with_blob):
        """Centre outside the zone is not enough; an overlapping box still loses.

        The left column's right edge is read from the configuration rather than
        hard-coded, so recalibrating the zones cannot silently invalidate this
        test.
        """
        left_column = next(
            zone for zone in SafetyConfig().exclusion_zones
            if zone[0] == 0.0 and zone[1] > 0.0 and zone[2] < 0.5
        )
        edge_x = int(REFERENCE_SIZE[0] * left_column[2])
        inside_y = int(REFERENCE_SIZE[1] * (left_column[1] + left_column[3]) / 2)

        verdict = rule.check(
            detection_at((edge_x + 8, inside_y), size=40),
            context_for(scene_with_blob),
        )
        assert not verdict.ok
        assert verdict.code == "EXCLUSION_ZONE"

    @pytest.mark.parametrize("window", [(1280, 720), (2560, 1440), (854, 480)])
    def test_zones_follow_the_window_size(self, rule, scene_with_blob, window):
        """Normalized zones must land on the same UI furniture at any size."""
        rendered = render_at(scene_with_blob, window)
        ctx = context_for(rendered, size=window)
        ratio = window[0] / REFERENCE_SIZE[0]

        inside_panel = detection_at(
            scale_point((180, 400), REFERENCE_SIZE, window), size=max(8, int(40 * ratio))
        )
        on_playfield = detection_at(
            scale_point(COIN_AT, REFERENCE_SIZE, window), size=max(8, int(40 * ratio))
        )

        assert not rule.check(inside_panel, ctx).ok
        assert rule.check(on_playfield, ctx).ok


class TestNotNearEdge:
    def test_centre_passes(self, scene_flat):
        rule = NotNearEdge(0.012)
        assert rule.check(detection_at((850, 480)), context_for(scene_flat)).ok

    @pytest.mark.parametrize("center", [(3, 480), (1704, 480), (850, 3), (850, 957)])
    def test_edges_are_rejected(self, scene_flat, center):
        rule = NotNearEdge(0.012)
        verdict = rule.check(detection_at(center), context_for(scene_flat))
        assert not verdict.ok
        assert verdict.code == "NEAR_EDGE"


class TestMinConfidence:
    def test_above_threshold_passes(self, scene_flat):
        rule = MinConfidence(0.80)
        assert rule.check(detection_at(COIN_AT, confidence=0.91), context_for(scene_flat)).ok

    def test_below_threshold_is_rejected_with_numbers_in_the_detail(self, scene_flat):
        rule = MinConfidence(0.80)
        verdict = rule.check(
            detection_at(COIN_AT, confidence=0.55), context_for(scene_flat)
        )
        assert not verdict.ok
        assert verdict.code == "LOW_CONFIDENCE"
        assert "0.55" in verdict.detail


class TestTextureBelow:
    """Cheap proxy for "a collectible coin sits above a monster"."""

    @pytest.fixture
    def rule(self):
        return TextureBelow(probe_height_ratio=1.3, min_std=12.0)

    def test_passes_above_a_textured_blob(self, rule, scene_with_blob):
        assert rule.check(detection_at(COIN_AT), context_for(scene_with_blob)).ok

    def test_rejects_on_flat_ground(self, rule, scene_flat):
        verdict = rule.check(detection_at(COIN_AT), context_for(scene_flat))
        assert not verdict.ok
        assert verdict.code == "NO_TEXTURE_BELOW"

    def test_measure_exposes_the_raw_value_for_calibration(self, rule, scene_with_blob, scene_flat):
        """The threshold must be derived from measurements, not guessed.

        This is the hook calibration tooling uses to collect the real
        distribution of the metric over captured frames.
        """
        busy = rule.measure(detection_at(COIN_AT), scene_with_blob)
        flat = rule.measure(detection_at(COIN_AT), scene_flat)
        assert busy > flat
        assert flat >= 0.0

    def test_probe_beyond_the_frame_bottom_is_rejected_not_crashed(self, rule, scene_flat):
        verdict = rule.check(detection_at((900, 958)), context_for(scene_flat))
        assert not verdict.ok


class TestNeighbourRules:
    """Rules needing extra templates; disabled by default for that reason."""

    def test_monster_below_within_range_passes(self, scene_flat):
        rule = NeighborBelow("monster", max_distance_ref=160)
        ctx = context_for(
            scene_flat,
            detections={"monster": [detection_at((900, 470), size=70, name="monster")]},
        )
        assert rule.check(detection_at(COIN_AT), ctx).ok

    def test_monster_too_far_below_is_rejected(self, scene_flat):
        rule = NeighborBelow("monster", max_distance_ref=160)
        ctx = context_for(
            scene_flat,
            detections={"monster": [detection_at((900, 900), size=70, name="monster")]},
        )
        assert not rule.check(detection_at(COIN_AT), ctx).ok

    def test_monster_above_does_not_count(self, scene_flat):
        rule = NeighborBelow("monster", max_distance_ref=160)
        ctx = context_for(
            scene_flat,
            detections={"monster": [detection_at((900, 320), size=70, name="monster")]},
        )
        assert not rule.check(detection_at(COIN_AT), ctx).ok

    def test_missing_category_rejects_everything(self, scene_flat):
        """Without monster.png the category is never detected: fails closed."""
        rule = NeighborBelow("monster", max_distance_ref=160)
        verdict = rule.check(detection_at(COIN_AT), context_for(scene_flat, detections={}))
        assert not verdict.ok
        assert verdict.code == "MISSING_NEIGHBOR"

    def test_overlapping_blocker_is_rejected(self, scene_flat):
        rule = NoOverlapWith("cloud")
        ctx = context_for(
            scene_flat,
            detections={"cloud": [detection_at((905, 405), size=60, name="cloud")]},
        )
        verdict = rule.check(detection_at(COIN_AT), ctx)
        assert not verdict.ok
        assert verdict.code == "BLOCKED_BY_OVERLAP"

    def test_non_overlapping_blocker_is_ignored(self, scene_flat):
        rule = NoOverlapWith("cloud")
        ctx = context_for(
            scene_flat,
            detections={"cloud": [detection_at((1400, 700), size=60, name="cloud")]},
        )
        assert rule.check(detection_at(COIN_AT), ctx).ok

    def test_rules_declare_the_categories_they_need(self):
        assert NeighborBelow("monster", 160).required_targets == {"monster"}
        assert NoOverlapWith("cloud").required_targets == {"cloud"}
        assert MinConfidence(0.8).required_targets == set()


class TestRuleChain:
    def test_short_circuits_on_the_first_rejection(self, scene_flat):
        """A rule after a rejection must not run at all."""

        class Exploding(TextureBelow):
            def check(self, detection, ctx):  # pragma: no cover
                raise AssertionError("must not be reached")

        chain = RuleChain([MinConfidence(0.9), Exploding(1.3, 12.0)])
        verdict = chain.evaluate(detection_at(COIN_AT, confidence=0.1), context_for(scene_flat))
        assert verdict.code == "LOW_CONFIDENCE"

    def test_accepts_when_every_rule_passes(self, scene_with_blob):
        chain = build_rule_chain(SafetyConfig(), min_confidence=0.80)
        assert chain.evaluate(detection_at(COIN_AT), context_for(scene_with_blob)).ok

    def test_required_targets_is_the_union_of_its_rules(self):
        chain = build_rule_chain(
            SafetyConfig(
                require_texture_below=False,
                require_monster_below=True,
                reject_cloud_overlap=True,
            ),
            min_confidence=0.80,
        )
        assert chain.required_targets == {"monster", "cloud"}

    def test_disabled_rules_are_not_in_the_chain_at_all(self):
        """Switched-off rules must cost nothing, not merely always pass."""
        lean = build_rule_chain(SafetyConfig(require_texture_below=False), 0.80)
        full = build_rule_chain(SafetyConfig(require_texture_below=True), 0.80)
        assert len(lean.rules) == len(full.rules) - 1
        assert not any(isinstance(r, TextureBelow) for r in lean.rules)

    def test_exclusion_zone_runs_before_the_expensive_texture_check(self):
        """Ordering is a performance contract, so assert it explicitly.

        The texture rule is disabled in the calibrated default configuration, so
        it is enabled here explicitly: this test is about ordering, not about
        whether the rule ships switched on.
        """
        chain = build_rule_chain(SafetyConfig(require_texture_below=True), 0.80)
        types = [type(rule) for rule in chain.rules]
        assert types.index(NotInExclusionZones) < types.index(TextureBelow)

    def test_calibrated_default_leaves_the_texture_rule_out(self):
        """Guards the calibration finding.

        Measured on 219 real detections the probe value never fell below the
        threshold, so the rule protected nothing while appearing active. If
        someone re-enables it by default, that decision should have to update
        this test and state its evidence.
        """
        chain = build_rule_chain(SafetyConfig(), 0.80)
        assert not any(isinstance(rule, TextureBelow) for rule in chain.rules)

    def test_describe_lists_the_rules(self):
        chain = build_rule_chain(SafetyConfig(), 0.80)
        assert "NotInExclusionZones" in chain.describe()
