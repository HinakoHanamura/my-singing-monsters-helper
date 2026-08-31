"""Escalating cooldown: the wait grows only for positions that keep failing.

Why the design is shaped this way
---------------------------------
A flat cooldown after every click was pure waste. A successful click removes its
target, so there is nothing left to click twice; pausing afterwards only lowers
throughput. What genuinely deserves backing off is a position that survives
clicks, because that is the signature of a false positive.

So the base wait is short enough to merely stop the same spot firing twice, and
the wait escalates per consecutive failure at that position before a ban.

Time is injected, so these tests are deterministic and instant.
"""

from __future__ import annotations

import pytest

from config import SafetyConfig
from core.click_guard import ClickGuard
from core.vision_agent import Detection

COIN = (900, 400)
OTHER = (1400, 700)


class Clock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def detection_at(center, size=40, confidence=0.9) -> Detection:
    return Detection(
        name="coin",
        x=center[0] - size // 2,
        y=center[1] - size // 2,
        width=size,
        height=size,
        confidence=confidence,
    )


@pytest.fixture
def clock():
    return Clock()


def guard_with(clock, **overrides) -> ClickGuard:
    settings = dict(
        confirm_frames=1,  # keep the confirmation gate out of the way here
        click_cooldown=0.35,
        failure_cooldown_ladder=(1.0, 3.0, 8.0),
        blacklist_after_failures=4,
        blacklist_duration=90.0,
        position_tolerance=20,
    )
    settings.update(overrides)
    return ClickGuard(SafetyConfig(**settings), scale=1.0, time_source=clock)


def confirm(guard, detection, times=1):
    for _ in range(times):
        guard.begin_frame()
        guard.observe([detection])


class TestLadderShape:
    """The mapping from failure count to wait is the contract."""

    def test_no_failures_uses_the_short_base_wait(self, clock):
        guard = guard_with(clock)
        assert guard.required_cooldown(0) == pytest.approx(0.35)

    @pytest.mark.parametrize("failures, expected", [(1, 1.0), (2, 3.0), (3, 8.0)])
    def test_each_failure_climbs_one_step(self, clock, failures, expected):
        guard = guard_with(clock)
        assert guard.required_cooldown(failures) == pytest.approx(expected)

    def test_the_last_step_repeats_beyond_the_ladder(self, clock):
        guard = guard_with(clock)
        assert guard.required_cooldown(9) == pytest.approx(8.0)

    def test_an_empty_ladder_degrades_to_the_base_wait(self, clock):
        guard = guard_with(clock, failure_cooldown_ladder=())
        assert guard.required_cooldown(3) == pytest.approx(0.35)

    def test_the_base_wait_is_much_shorter_than_the_first_penalty(self, clock):
        """Throughput depends on this gap; assert it rather than assume it."""
        guard = guard_with(clock)
        assert guard.required_cooldown(0) < guard.required_cooldown(1) / 2


class TestSuccessfulClicksDoNotStall:
    def test_a_verified_success_leaves_only_the_base_wait(self, clock):
        guard = guard_with(clock)
        detection = detection_at(COIN)
        confirm(guard, detection)
        guard.register_click(COIN)
        guard.register_result(COIN, target_gone=True)

        clock.advance(0.4)
        confirm(guard, detection)
        assert guard.status(detection) is None

    def test_immediately_after_a_click_the_spot_is_still_blocked(self, clock):
        """Stops the same position firing twice inside one animation."""
        guard = guard_with(clock)
        detection = detection_at(COIN)
        confirm(guard, detection)
        guard.register_click(COIN)

        blocked = guard.status(detection)
        assert blocked is not None
        assert blocked.code == "COOLING_DOWN"

    def test_a_different_position_is_never_delayed(self, clock):
        guard = guard_with(clock)
        first = detection_at(COIN)
        second = detection_at(OTHER)
        confirm(guard, first)
        confirm(guard, second)

        guard.register_click(COIN)
        assert guard.status(first) is not None
        assert guard.status(second) is None


class TestEscalation:
    def _fail_once(self, guard, point=COIN):
        guard.register_click(point)
        return guard.register_result(point, target_gone=False)

    def test_one_failure_moves_past_the_base_wait(self, clock):
        guard = guard_with(clock)
        detection = detection_at(COIN)
        confirm(guard, detection)
        self._fail_once(guard)

        # The base wait would already have expired; the penalty has not.
        clock.advance(0.5)
        confirm(guard, detection)
        blocked = guard.status(detection)
        assert blocked is not None
        assert blocked.code == "COOLING_DOWN"

        clock.advance(0.6)  # past 1.0s total
        confirm(guard, detection)
        assert guard.status(detection) is None

    def test_failures_accumulate_into_longer_waits(self, clock):
        guard = guard_with(clock)
        detection = detection_at(COIN)
        confirm(guard, detection)

        self._fail_once(guard)
        clock.advance(1.1)
        self._fail_once(guard)  # second failure -> 3.0s

        clock.advance(1.5)
        confirm(guard, detection)
        assert guard.status(detection) is not None

        clock.advance(2.0)
        confirm(guard, detection)
        assert guard.status(detection) is None

    def test_a_success_drops_the_position_back_to_the_base_wait(self, clock):
        guard = guard_with(clock)
        detection = detection_at(COIN)
        confirm(guard, detection)

        self._fail_once(guard)
        clock.advance(1.1)
        self._fail_once(guard)
        clock.advance(3.1)

        guard.register_click(COIN)
        guard.register_result(COIN, target_gone=True)
        clock.advance(0.4)
        confirm(guard, detection)
        assert guard.status(detection) is None

    def test_the_ban_still_arrives_after_the_ladder(self, clock):
        guard = guard_with(clock, blacklist_after_failures=4)
        detection = detection_at(COIN)
        confirm(guard, detection)

        notes = []
        for _ in range(4):
            clock.advance(10.0)
            confirm(guard, detection)
            notes.append(self._fail_once(guard))

        assert notes[:3] == [None, None, None]
        assert notes[3] is not None
        blocked = guard.status(detection)
        assert blocked is not None
        assert blocked.code == "BLACKLISTED"

    def test_the_message_says_how_many_failures_there_were(self, clock):
        """The log line is the operator's only view into the escalation."""
        guard = guard_with(clock)
        detection = detection_at(COIN)
        confirm(guard, detection)
        self._fail_once(guard)

        blocked = guard.status(detection)
        assert blocked is not None
        assert "1" in blocked.message


class TestBookkeeping:
    def test_failing_positions_are_countable(self, clock):
        guard = guard_with(clock)
        guard.register_click(COIN)
        guard.register_result(COIN, target_gone=False)
        assert guard.failing_count() == 1
        assert guard.blacklisted_count() == 0

    def test_a_success_clears_the_failing_count(self, clock):
        guard = guard_with(clock)
        guard.register_click(COIN)
        guard.register_result(COIN, target_gone=False)
        guard.register_result(COIN, target_gone=True)
        assert guard.failing_count() == 0

    def test_pruning_accounts_for_the_longest_ladder_step(self, clock):
        """A position mid-escalation must not be forgotten before its wait ends."""
        guard = guard_with(
            clock,
            click_cooldown=0.35,
            failure_cooldown_ladder=(1.0, 3.0, 8.0),
            blacklist_duration=1.0,
        )
        guard.register_click(COIN)
        guard.register_result(COIN, target_gone=False)

        clock.advance(5.0)
        guard.register_click(OTHER)  # triggers pruning
        assert guard.failing_count() == 1

    def test_stats_mentions_both_backoff_and_bans(self, clock):
        guard = guard_with(clock)
        text = guard.stats()
        assert isinstance(text, str) and text
