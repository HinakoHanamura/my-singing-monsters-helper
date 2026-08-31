"""Click guard: cross-frame confirmation, escalating cooldown, and a blacklist.

The rule chain answers "should this be clicked at all". This module answers
"should it be clicked right now":

1. Cross-frame confirmation. Targets are animated, so a single high-scoring
   frame can be a coincidence. Requiring a target to hold roughly the same
   position for several consecutive frames removes most one-off false detections.

2. Escalating cooldown. Deliberately not a flat wait. A successful click removes
   its target, so pausing afterwards buys nothing and only costs throughput; the
   base cooldown is short enough to merely stop the same spot firing twice. What
   does deserve backing off is a position that keeps surviving clicks, so the
   wait grows with consecutive failures at that position.

3. False-positive blacklist. After the ladder is exhausted the position is
   banned for a while. This is the only self-correcting mechanism here: a spot
   that never responds was probably never a target, and the bot stops spinning
   on it without anyone tuning a threshold.

All spatial tolerances are calibrated in reference-resolution pixels and scaled
to the live window, so changing resolution needs no retuning.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from config import SafetyConfig
from core.geometry import Point, distance, scale_length

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GuardBlock:
    """Reason a click was withheld.

    ``code`` is the stable identifier tests and metrics use; ``label`` and
    ``message`` are for the log.
    """

    code: str
    label: str
    message: str


@dataclass
class _Track:
    """A position being observed across frames."""

    point: Point
    seen_frames: int = 1
    last_frame: int = 0


@dataclass
class _Site:
    """A previously clicked position with its cooldown and failure history."""

    point: Point
    last_click: float = 0.0
    failures: int = 0
    blacklisted_until: float = 0.0
    last_touch: float = field(default_factory=time.monotonic)


class ClickGuard:
    """Click admission control.

    The engine uses it per round in the order:
    ``begin_frame()`` -> ``observe(detections)`` -> ``status(detection)``.
    """

    def __init__(
        self,
        safety: SafetyConfig,
        scale: float = 1.0,
        time_source: Callable[[], float] = time.monotonic,
    ) -> None:
        self._safety = safety
        self._scale = scale
        self._now = time_source

        self._frame = 0
        self._tracks: List[_Track] = []
        self._sites: List[_Site] = []

    # ------------------------------------------------------------- scale sync

    def set_scale(self, scale: float) -> None:
        """Called by the engine when the window size changes."""
        self._scale = scale if scale > 0 else 1.0

    @property
    def _confirm_tolerance(self) -> int:
        return scale_length(self._safety.confirm_tolerance, self._scale)

    @property
    def _position_tolerance(self) -> int:
        return scale_length(self._safety.position_tolerance, self._scale)

    # --------------------------------------------------- cross-frame tracking

    def begin_frame(self) -> None:
        """Advance the frame counter and drop stale tracks."""
        self._frame += 1
        expire = self._safety.track_expire_frames
        self._tracks = [t for t in self._tracks if self._frame - t.last_frame <= expire]

    def observe(self, detections: Sequence) -> None:
        """Fold this frame's detections into the tracks."""
        tolerance = self._confirm_tolerance
        for detection in detections:
            center = detection.center
            track = self._nearest_track(center, tolerance)
            if track is None:
                self._tracks.append(
                    _Track(point=center, seen_frames=1, last_frame=self._frame)
                )
                continue
            # Overlapping detections within one frame must not inflate the count.
            if track.last_frame == self._frame:
                continue
            track.seen_frames += 1
            track.last_frame = self._frame
            # Follow small drift so an animated target keeps its identity.
            track.point = center

    def confirmations(self, detection) -> int:
        """How many consecutive frames this position has been seen."""
        track = self._nearest_track(detection.center, self._confirm_tolerance)
        return track.seen_frames if track else 0

    def _nearest_track(self, point: Point, tolerance: int) -> Optional[_Track]:
        best: Optional[_Track] = None
        best_distance = float(tolerance)
        for track in self._tracks:
            d = distance(point, track.point)
            if d <= best_distance:
                best = track
                best_distance = d
        return best

    # ----------------------------------------------- cooldown and blacklist

    def required_cooldown(self, failures: int) -> float:
        """Seconds a position must rest, given its consecutive failure count.

        Zero failures means the last click there worked (or it has never been
        clicked), so only the short base wait applies. Each subsequent failure
        moves one step up the ladder; the final step repeats.
        """
        if failures <= 0:
            return self._safety.click_cooldown
        ladder = self._safety.failure_cooldown_ladder
        if not ladder:
            return self._safety.click_cooldown
        return ladder[min(failures, len(ladder)) - 1]

    def status(self, detection) -> Optional[GuardBlock]:
        """Return None if clicking is allowed, otherwise why it is withheld."""
        needed = self._safety.confirm_frames
        if needed > 1:
            seen = self.confirmations(detection)
            if seen < needed:
                return GuardBlock(
                    code="AWAITING_CONFIRMATION",
                    label="等待跨帧确认",
                    message="等待跨帧确认(%d/%d)" % (seen, needed),
                )

        site = self._nearest_site(detection.center)
        if site is None:
            return None

        now = self._now()
        if site.blacklisted_until > now:
            return GuardBlock(
                code="BLACKLISTED",
                label="位置已拉黑",
                message="位置已拉黑(剩余%.0fs)" % (site.blacklisted_until - now),
            )

        required = self.required_cooldown(site.failures)
        elapsed = now - site.last_click
        if elapsed < required:
            return GuardBlock(
                code="COOLING_DOWN",
                label="冷却中",
                message="冷却中(%.1fs，该位置已失败 %d 次)"
                % (required - elapsed, site.failures),
            )

        return None

    def register_click(self, point: Point) -> None:
        """Record a click and start the cooldown for that position."""
        site = self._nearest_site(point)
        now = self._now()
        if site is None:
            site = _Site(point=point)
            self._sites.append(site)
        site.point = point
        site.last_click = now
        site.last_touch = now
        self._prune_sites()

    def register_result(self, point: Point, target_gone: bool) -> Optional[str]:
        """Record the effect of a click.

        Args:
            point: the position that was clicked.
            target_gone: whether the target had disappeared on re-read.
                True means the click worked.

        Returns:
            An explanatory message if this call triggered a blacklist, else None.
        """
        site = self._nearest_site(point)
        if site is None:
            site = _Site(point=point)
            self._sites.append(site)

        site.last_touch = self._now()

        if target_gone:
            site.failures = 0
            return None

        site.failures += 1
        if site.failures >= self._safety.blacklist_after_failures:
            site.blacklisted_until = self._now() + self._safety.blacklist_duration
            site.failures = 0
            return (
                "(%d, %d) 连续点击无效，判定为假阳性并拉黑 %.0fs"
                % (point[0], point[1], self._safety.blacklist_duration)
            )
        return None

    def _nearest_site(self, point: Point) -> Optional[_Site]:
        best: Optional[_Site] = None
        best_distance = float(self._position_tolerance)
        for site in self._sites:
            d = distance(point, site.point)
            if d <= best_distance:
                best = site
                best_distance = d
        return best

    def _prune_sites(self) -> None:
        """Drop records that are neither cooling down nor banned and are stale."""
        now = self._now()
        longest_cooldown = max(
            [self._safety.click_cooldown]
            + list(self._safety.failure_cooldown_ladder or ())
        )
        keep_window = max(longest_cooldown, self._safety.blacklist_duration) + 60.0
        self._sites = [
            s
            for s in self._sites
            if s.blacklisted_until > now or now - s.last_touch <= keep_window
        ]

    # ------------------------------------------------------------ diagnostics

    def blacklisted_count(self) -> int:
        now = self._now()
        return sum(1 for s in self._sites if s.blacklisted_until > now)

    def failing_count(self) -> int:
        """Positions with a live failure streak, i.e. currently escalating."""
        return sum(1 for s in self._sites if s.failures > 0)

    def stats(self) -> str:
        return "轨迹 %d ｜ 记录位置 %d ｜ 退避中 %d ｜ 已拉黑 %d" % (
            len(self._tracks),
            len(self._sites),
            self.failing_count(),
            self.blacklisted_count(),
        )
