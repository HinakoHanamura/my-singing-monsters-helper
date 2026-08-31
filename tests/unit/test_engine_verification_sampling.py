"""Sampled verification: keep the self-correction, drop most of its cost.

Re-reading a frame after every click was the largest fixed cost per collected
item. Removing it would remove the only mechanism that notices a false positive,
so instead the engine earns the right to sample: verify every click until a run
has proven itself, then check periodically, and fall back to verifying
everything the instant a check fails.

These tests exercise the policy directly. It is pure decision logic, so no game,
no Qt event loop and no frames are needed.
"""

from __future__ import annotations

import pytest

from config import AppConfig, SafetyConfig
from core.bot_engine import BotEngine


def engine_with(**safety) -> BotEngine:
    settings = dict(
        verify_after_click=True,
        verify_trust_after=3,
        verify_sample_every=5,
    )
    settings.update(safety)
    return BotEngine(config=AppConfig(safety=SafetyConfig(**settings)))


def decisions(engine: BotEngine, count: int) -> list:
    return [engine._should_verify() for _ in range(count)]


class TestUntrustedRun:
    """Before the run proves itself, everything is checked."""

    def test_the_first_clicks_are_all_verified(self):
        engine = engine_with()
        assert decisions(engine, 3) == [True, True, True]

    def test_it_keeps_verifying_while_nothing_succeeds(self):
        """Without recorded successes the streak never grows, so sampling never starts."""
        engine = engine_with()
        assert all(decisions(engine, 12))


class TestEarningTrust:
    def test_sampling_begins_once_the_streak_is_reached(self):
        engine = engine_with(verify_trust_after=3, verify_sample_every=5)
        engine._verified_streak = 3

        outcomes = decisions(engine, 10)
        # One check per sampling period, not one per click.
        assert outcomes.count(True) == 2
        assert outcomes.count(False) == 8

    def test_the_first_decision_after_trust_still_checks(self):
        """Trust starts with a check, so a bad run is caught immediately."""
        engine = engine_with()
        engine._verified_streak = 3
        assert engine._should_verify() is True

    @pytest.mark.parametrize("every", [2, 5, 10])
    def test_sampling_rate_follows_the_setting(self, every):
        engine = engine_with(verify_sample_every=every)
        engine._verified_streak = 99
        outcomes = decisions(engine, every * 3)
        assert outcomes.count(True) == 3


class TestLosingTrust:
    def test_a_failed_check_forces_the_next_click_to_be_checked(self):
        engine = engine_with()
        engine._verified_streak = 10
        engine._verify_countdown = 5

        # What _verify_click does when the target is still there.
        engine._verified_streak = 0
        engine._verify_countdown = 0

        assert engine._should_verify() is True

    def test_trust_must_be_earned_again_from_scratch(self):
        engine = engine_with(verify_trust_after=3, verify_sample_every=5)
        engine._verified_streak = 0
        assert decisions(engine, 3) == [True, True, True]
        engine._verified_streak = 3
        assert engine._should_verify() is True


class TestSwitches:
    def test_verification_can_be_disabled_entirely(self):
        engine = engine_with(verify_after_click=False)
        engine._verified_streak = 99
        assert not any(decisions(engine, 8))

    def test_sample_every_one_means_always_verify(self):
        engine = engine_with(verify_sample_every=1)
        engine._verified_streak = 99
        assert all(decisions(engine, 8))

    def test_sample_every_zero_is_treated_as_always(self):
        """A nonsensical value must not silently disable self-correction."""
        engine = engine_with(verify_sample_every=0)
        engine._verified_streak = 99
        assert all(decisions(engine, 8))


class TestThroughputClaim:
    def test_a_healthy_run_verifies_a_small_fraction_of_clicks(self):
        """The whole point: cost per collected item drops without losing the check."""
        engine = engine_with(verify_trust_after=3, verify_sample_every=5)

        verified = 0
        for index in range(100):
            if engine._should_verify():
                verified += 1
            # Simulate every check passing.
            engine._verified_streak += 1

        assert verified < 25, "expected sampling to cut verification well below 1:1"
        assert verified > 0, "self-correction must not be switched off entirely"
