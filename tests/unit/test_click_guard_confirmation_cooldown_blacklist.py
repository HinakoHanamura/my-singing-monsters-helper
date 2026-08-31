"""Click guard: cross-frame confirmation, cooldown, and the self-correcting blacklist.

Time is injected, so these tests are deterministic and instant - no sleeping.
Assertions key off ``GuardBlock.code`` rather than the displayed message.
"""

from __future__ import annotations

import pytest

from config import SafetyConfig
from core.click_guard import ClickGuard
from core.vision_agent import Detection


class Clock:
    """Manually advanced monotonic clock."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def detection_at(center, size=40, confidence=0.97) -> Detection:
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
        confirm_frames=2,
        confirm_tolerance=12,
        track_expire_frames=3,
        click_cooldown=3.0,
        blacklist_after_failures=2,
        blacklist_duration=60.0,
        position_tolerance=20,
    )
    settings.update(overrides)
    return ClickGuard(SafetyConfig(**settings), scale=1.0, time_source=clock)


COIN = (900, 400)


class TestCrossFrameConfirmation:
    """Animated targets produce one-off false hits; confirmation filters them."""

    def test_first_sighting_is_not_clickable(self, clock):
        guard = guard_with(clock)
        detection = detection_at(COIN)
        guard.begin_frame()
        guard.observe([detection])

        blocked = guard.status(detection)
        assert blocked is not None
        assert blocked.code == "AWAITING_CONFIRMATION"

    def test_becomes_clickable_once_confirmed(self, clock):
        guard = guard_with(clock)
        detection = detection_at(COIN)
        for _ in range(2):
            guard.begin_frame()
            guard.observe([detection])
        assert guard.status(detection) is None

    def test_confirm_frames_one_disables_the_gate(self, clock):
        guard = guard_with(clock, confirm_frames=1)
        detection = detection_at(COIN)
        guard.begin_frame()
        guard.observe([detection])
        assert guard.status(detection) is None

    def test_small_drift_keeps_the_same_track(self, clock):
        """An animated coin wobbles a few pixels; that is still the same coin."""
        guard = guard_with(clock, confirm_frames=3)
        for dx in (0, 4, 8):
            guard.begin_frame()
            guard.observe([detection_at((COIN[0] + dx, COIN[1]))])
        assert guard.confirmations(detection_at((COIN[0] + 8, COIN[1]))) == 3

    def test_a_distant_target_starts_its_own_track(self, clock):
        guard = guard_with(clock, confirm_frames=3)
        for _ in range(3):
            guard.begin_frame()
            guard.observe([detection_at(COIN)])
        assert guard.confirmations(detection_at((1400, 700))) == 0

    def test_multiple_hits_in_one_frame_do_not_inflate_the_count(self, clock):
        """Overlapping duplicates within a frame must count once."""
        guard = guard_with(clock, confirm_frames=3)
        guard.begin_frame()
        guard.observe(
            [
                detection_at(COIN),
                detection_at((COIN[0] + 2, COIN[1] + 2)),
                detection_at((COIN[0] - 3, COIN[1] + 1)),
            ]
        )
        assert guard.confirmations(detection_at(COIN)) == 1

    def test_track_expires_when_the_target_disappears(self, clock):
        guard = guard_with(clock, confirm_frames=2, track_expire_frames=2)
        guard.begin_frame()
        guard.observe([detection_at(COIN)])
        # Several empty frames pass.
        for _ in range(4):
            guard.begin_frame()
            guard.observe([])
        assert guard.confirmations(detection_at(COIN)) == 0

    def test_tolerance_scales_with_the_window(self, clock):
        """At half size, a 12 px reference tolerance becomes 6 px."""
        guard = ClickGuard(
            SafetyConfig(confirm_frames=2, confirm_tolerance=12),
            scale=0.5,
            time_source=clock,
        )
        guard.begin_frame()
        guard.observe([detection_at((400, 200), size=20)])
        guard.begin_frame()
        guard.observe([detection_at((404, 200), size=20)])
        assert guard.confirmations(detection_at((404, 200), size=20)) == 2

    def test_set_scale_rejects_nonsense_values(self, clock):
        guard = guard_with(clock)
        guard.set_scale(0.0)
        # Must not divide by zero or collapse the tolerance to nothing.
        guard.begin_frame()
        guard.observe([detection_at(COIN)])
        assert guard.confirmations(detection_at(COIN)) == 1


class TestCooldown:
    def _confirm(self, guard, detection):
        for _ in range(2):
            guard.begin_frame()
            guard.observe([detection])

    def test_click_starts_a_cooldown(self, clock):
        guard = guard_with(clock)
        detection = detection_at(COIN)
        self._confirm(guard, detection)

        guard.register_click(detection.center)
        blocked = guard.status(detection)
        assert blocked is not None
        assert blocked.code == "COOLING_DOWN"

    def test_cooldown_expires(self, clock):
        guard = guard_with(clock)
        detection = detection_at(COIN)
        self._confirm(guard, detection)
        guard.register_click(detection.center)

        clock.advance(3.5)
        guard.begin_frame()
        guard.observe([detection])
        assert guard.status(detection) is None

    def test_cooldown_is_per_position(self, clock):
        guard = guard_with(clock)
        first = detection_at(COIN)
        second = detection_at((1300, 700))
        for _ in range(2):
            guard.begin_frame()
            guard.observe([first, second])

        guard.register_click(first.center)
        assert guard.status(first) is not None
        # A different coin elsewhere is unaffected.
        assert guard.status(second) is None


class TestFalsePositiveBlacklist:
    """The only self-correcting mechanism in the stack."""

    def _confirm_and_click(self, guard, detection):
        for _ in range(2):
            guard.begin_frame()
            guard.observe([detection])
        guard.register_click(detection.center)

    def test_successful_click_leaves_no_penalty(self, clock):
        guard = guard_with(clock)
        detection = detection_at(COIN)
        self._confirm_and_click(guard, detection)

        assert guard.register_result(detection.center, target_gone=True) is None
        assert guard.blacklisted_count() == 0

    def test_one_failure_does_not_ban(self, clock):
        guard = guard_with(clock, blacklist_after_failures=2)
        detection = detection_at(COIN)
        self._confirm_and_click(guard, detection)

        assert guard.register_result(detection.center, target_gone=False) is None
        assert guard.blacklisted_count() == 0

    def test_repeated_failures_trigger_a_ban(self, clock):
        guard = guard_with(clock, blacklist_after_failures=2)
        detection = detection_at(COIN)
        self._confirm_and_click(guard, detection)

        guard.register_result(detection.center, target_gone=False)
        note = guard.register_result(detection.center, target_gone=False)

        assert note is not None
        assert guard.blacklisted_count() == 1
        blocked = guard.status(detection)
        assert blocked is not None
        assert blocked.code == "BLACKLISTED"

    def test_a_success_resets_the_failure_streak(self, clock):
        guard = guard_with(clock, blacklist_after_failures=3)
        detection = detection_at(COIN)
        self._confirm_and_click(guard, detection)

        guard.register_result(detection.center, target_gone=False)
        guard.register_result(detection.center, target_gone=False)
        guard.register_result(detection.center, target_gone=True)  # streak reset
        guard.register_result(detection.center, target_gone=False)

        assert guard.blacklisted_count() == 0

    def test_ban_expires_after_its_duration(self, clock):
        guard = guard_with(clock, blacklist_after_failures=1, blacklist_duration=60.0)
        detection = detection_at(COIN)
        self._confirm_and_click(guard, detection)
        guard.register_result(detection.center, target_gone=False)

        assert guard.status(detection) is not None
        clock.advance(61.0)
        guard.begin_frame()
        guard.observe([detection])
        assert guard.status(detection) is None

    def test_ban_is_confined_to_its_position(self, clock):
        guard = guard_with(clock, blacklist_after_failures=1)
        bad = detection_at(COIN)
        good = detection_at((1300, 700))
        for _ in range(2):
            guard.begin_frame()
            guard.observe([bad, good])
        guard.register_click(bad.center)
        guard.register_result(bad.center, target_gone=False)

        assert guard.status(bad) is not None
        assert guard.status(good) is None


class TestBookkeeping:
    def test_site_records_do_not_grow_without_bound(self, clock):
        guard = guard_with(clock, click_cooldown=1.0, blacklist_duration=1.0)
        for i in range(50):
            guard.register_click((100 + i * 50, 400))
            clock.advance(90.0)
        # Stale, unbanned records get pruned rather than accumulating forever.
        assert len(guard._sites) < 10

    def test_banned_records_survive_pruning(self, clock):
        guard = guard_with(clock, blacklist_after_failures=1, blacklist_duration=600.0)
        guard.register_click(COIN)
        guard.register_result(COIN, target_gone=False)
        for i in range(30):
            guard.register_click((200 + i * 40, 800))
        assert guard.blacklisted_count() == 1

    def test_stats_is_human_readable(self, clock):
        guard = guard_with(clock)
        guard.begin_frame()
        guard.observe([detection_at(COIN)])
        assert isinstance(guard.stats(), str)
        assert guard.stats()
