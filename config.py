"""Central configuration.

Every tunable lives here; the other layers read it and never write it.

Perception values are **measured**, not guessed. Each one below records where its
number came from, so it is obvious what has evidence behind it and what does
not. Re-derive them with:

    python tools/calibrate.py --frames captures/<recording>

Calibration baseline (2026-08-29): windowed mode, in-game resolution 1024x768,
camera zoom and pan untouched. 24 frames across two recordings, 219 detections.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

# Normalized rectangle (x1, y1, x2, y2) with values in 0.0..1.0, relative to the
# client area. Ratios rather than absolute pixels, so one configuration stays
# correct at any window size.
NormRect = Tuple[float, float, float, float]

#: Repository root, derived from this file's location. Never a literal path, so
#: the value is correct on any machine and for anyone who clones the repository.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _in_project(relative: str) -> str:
    """Anchor a repository-relative resource path on the project root.

    Why this is not left as a plain relative string: a relative path resolves
    against the *current working directory*, which is set by whoever launches
    the process, not by the project. Launching from a shell that had cd'd into
    the repository worked; launching the same file from an editor, which uses a
    different working directory, silently found no templates at all -- and a
    detector with no templates reports "nothing found" rather than an error, so
    the failure looked like a perception bug for as long as it took to compare
    the two launch methods.

    Measured, on this repository: with the cwd at the project root the coin
    template glob returns 1 file; with the cwd at the user's home directory it
    returns 0.

    The unit tests could not catch this. Every one of them injects its own
    temporary template directory, so none exercises the configured default --
    the suite passed at 455 while the application saw nothing.

    Already-absolute paths pass through untouched, so an explicit override on the
    command line still wins.
    """
    if os.path.isabs(relative):
        return relative
    return os.path.normpath(os.path.join(PROJECT_ROOT, relative))


@dataclass(frozen=True)
class WindowConfig:
    """Target window lookup and capture."""

    # Must match what win32gui.FindWindow can resolve.
    title: str = "My Singing Monsters"

    # PrintWindow flags: 3 = PW_CLIENTONLY(1) | PW_RENDERFULLCONTENT(2).
    # PW_RENDERFULLCONTENT is what makes hardware-accelerated frames readable;
    # without it the capture comes back black. Verified against the real game.
    print_window_flag: int = 3


@dataclass(frozen=True)
class ClickConfig:
    """Humanisation of background clicks.

    Fixed coordinates on a fixed cadence are the obvious machine signature, so
    both are randomised.
    """

    # Random offset radius around the target, in reference-resolution pixels
    # (scaled to the live window at runtime). The coin icon measures 50x38, so
    # 5px stays comfortably inside it.
    jitter_radius: int = 5

    # Random hold time between button down and up, seconds.
    # A real button press is rarely shorter than ~20ms, so this is about the
    # floor before the timing stops resembling a hand at all.
    press_duration: Tuple[float, float] = (0.02, 0.05)

    # Random pause after a completed click, seconds.
    post_click_delay: Tuple[float, float] = (0.03, 0.09)

    # Minimum extra settle time between two clicks of the same batch, seconds.
    #
    # Within a batch the pointer jumps hundreds of pixels between targets, and
    # the sequence MOVE -> DOWN -> UP -> MOVE(far) -> DOWN -> UP can look like a
    # drag gesture to an input layer that has not finished processing the UP.
    # A drag pans the camera in this game, which would move every remaining
    # target. This gap is unverified as a cause but cheap as a precaution.
    batch_settle_delay: Tuple[float, float] = (0.05, 0.12)

    # Send a WM_MOUSEMOVE first; some games only respond to a click after hover.
    move_before_click: bool = True

    # Random gap between the move and the button press, seconds.
    pre_press_delay: Tuple[float, float] = (0.008, 0.025)

    # True  -> PostMessage: asynchronous, never blocks the worker thread.
    # False -> SendMessage: synchronous, broader compatibility, but blocks this
    #          thread while the game is busy. Matches the verified spike.
    use_post_message: bool = False


@dataclass(frozen=True)
class LoopConfig:
    """Main loop pacing."""

    # Random interval between rounds, seconds.
    #
    # Collecting a stack is iterative (one click per round), so this interval is
    # the dominant cost per collected item, not the click itself. A round is
    # roughly:
    #   detect + optional verify_delay + post_click_delay + tick_interval
    #
    # Tightened twice against live runs: (0.9, 1.8) -> (0.35, 0.75) -> here.
    # With sampled verification (see SafetyConfig) a healthy round now costs
    # about 0.4s rather than the original 2.1s.
    #
    # Tradeoff worth knowing: a faster, tighter cadence is also a more
    # machine-like one. Widen the range back out if that matters more than
    # throughput.
    tick_interval: Tuple[float, float] = (0.05, 0.15)

    # Retry delay after the window is lost or a capture fails, seconds.
    retry_interval: float = 2.0

    # Stop automatically after this many consecutive failures rather than
    # spinning uselessly.
    max_consecutive_errors: int = 5

    # Maximum clicks per round.
    #
    # Above 1, the engine only batches candidates whose boxes do not overlap.
    # That distinction is the whole point:
    #
    #   - Stacked plaques must be peeled one at a time. A front plaque hides the
    #     ones behind it, so their positions are simply not known yet; firing at
    #     them would fire at stale or invented coordinates.
    #   - Plaques that do not overlap are independent. Clicking one cannot move
    #     or reveal another, so making them wait a full capture-and-detect cycle
    #     each buys nothing.
    #
    # Batching the independent ones amortises capture and detection across
    # several collections, which roughly halves cost per item on a busy island
    # while keeping the peeling behaviour intact.
    max_clicks_per_tick: int = 3

    # Extra padding added to each box before the overlap test, in reference
    # pixels. Boxes that merely touch are treated as interfering, since a
    # collection animation on one can bleed over its neighbour.
    click_separation_margin: int = 6


@dataclass(frozen=True)
class VisionConfig:
    """Perception layer."""

    # Anchored on the project root, not the cwd -- see _in_project above.
    template_dir: str = _in_project("assets/templates")

    # Client-area size the templates were cropped from. Everything calibrated in
    # "reference pixels" is relative to this. Change it only together with the
    # template assets.
    #
    # Measured: the calibration recordings are exactly 1024x768 (game setting,
    # windowed).
    reference_size: Tuple[int, int] = (1024, 768)

    # Confidence floor for a template match.
    #
    # Measured: with a permissive 0.50 sampling threshold there were zero false
    # positives on 24 frames - no background, monster, UI element or non-coin
    # plaque was ever matched. Precision is not the binding constraint here.
    # What the score actually tracks is occlusion: fully visible icons land at
    # 0.80-0.98, partially covered ones trail down to 0.51.
    #   score distribution : p5=0.53  p50=0.80  p95=0.89   (calib_1024)
    #                        p5=0.77  p50=0.84  p95=0.92   (calib_mixed)
    # 0.75 keeps essentially every unoccluded target while dropping the heavily
    # covered ones, which are exactly the ones not worth clicking yet. Raising
    # it to 0.85 would cut detections to 1.75 per frame and stall whenever every
    # plaque is partly covered.
    match_threshold: float = 0.75

    # Extra scale steps probed around the window ratio, to absorb rounding and
    # mild distortion.
    #
    # Every step is a full matchTemplate pass, so this list is the detection
    # cost multiplier. Narrowed from five steps to three: the window is run at
    # the reference resolution, where the base scale is 1.000, and the outer
    # +/-6% steps existed only to absorb rounding that does not occur at 1:1.
    # Widen it again if templates are ever used at a different window size.
    #
    # Honest limitation: this band covers window-size rounding, NOT the game's
    # own camera zoom. Zoom changes sprite size independently of window size, so
    # detection will miss outside the band. Making that robust is separate work;
    # the zoom/pan recordings in captures/ are its test set.
    scale_steps: Tuple[float, ...] = (0.97, 1.0, 1.03)

    # Minimum centre separation for de-duplication, in reference pixels.
    #
    # Measured: with suppression effectively disabled, the true centre-to-centre
    # distance between distinct adjacent plaques bottoms out at 38.6px
    # (calib_1024) and 39.1px (calib_mixed). Cross-scale duplicate hits on a
    # single plaque were observed as close as 26px. 30 sits between the two:
    # it merges duplicates yet keeps genuinely adjacent plaques apart, with 22%
    # margin below the measured floor.
    nms_distance: int = 30

    # Per-frame cap on one target type. Measured median is 11 per frame.
    max_detections: int = 20

    # Match on a shrunken copy of the frame. Template matching is O(pixels), so
    # this is the single largest lever on detection cost - and detection turned
    # out to BE the cost. Measured with tools/benchmark_vision.py on 24 real
    # frames, against full-resolution output as the reference:
    #
    #   full res, 3 steps   103-108 ms/frame   (the baseline)
    #   0.75 res            114-118 ms/frame   slower: the resize is not repaid
    #   0.50 res             30- 48 ms/frame   2.2-3.6x, loses 1.8% of detections
    #   0.40 res             20- 35 ms/frame   3.0-5.5x, loses 4.5%
    #
    # Worst centre drift at 0.5 was 2px, against a 20px click tolerance and a
    # 50x38 icon, so localisation is not the limiting factor.
    #
    # Recall loss matters less than it looks: the loop re-detects several times a
    # second, so a target missed in one frame is almost certainly found in the
    # next. 0.5 was chosen over the faster 0.4 because 0.4 only saves ~13ms more
    # per detection while doubling the miss rate.
    #
    # No false positives were introduced at either setting: total detections
    # never exceeded the full-resolution count on frames containing the other
    # plaque types (heart, purple, flame).
    match_downscale: float = 0.5

    # Optional normalized rect limiting where matching runs.
    #
    # Left off deliberately, and this is a measured negative result rather than
    # an oversight. Cropping to the playfield to skip the UI bands sounded free,
    # but it measured 105-129 ms/frame against a 103-108 ms baseline: the crop
    # copies memory and hands matchTemplate a non-contiguous view, costing more
    # than the ~25% of pixels it removes.
    search_region: Optional[Tuple[float, float, float, float]] = None

    # ---- placeholder mode, only for running without assets ----
    # True returns synthetic coordinates and reads no templates.
    use_fake_detection: bool = False
    simulate_miss_rate: float = 0.25


@dataclass(frozen=True)
class SafetyConfig:
    """Anti-misclick. Prefer missing a target over clicking the wrong thing."""

    # ---- Layer 1: hard no-click regions ----
    # Measured from real 1024x768 frames. Rejection triggers when a detection
    # box overlaps a zone, not merely when its centre does.
    #
    # Observed coin plaques occupy roughly x 0.49-0.81, y 0.26-0.55, so these
    # zones do not compete with the playfield. Verified: zero real detections
    # were rejected by these zones across 24 calibration frames.
    exclusion_zones: Tuple[NormRect, ...] = (
        (0.00, 0.00, 1.00, 0.075),  # top status strip
        (0.00, 0.00, 0.14, 0.17),   # level avatar, top-left
        (0.52, 0.00, 1.00, 0.13),   # currency widgets and the More button
        (0.00, 0.15, 0.13, 0.80),   # left column: GOALS, timers, ACTIVITIES
        (0.00, 0.83, 1.00, 1.00),   # bottom button bar
        (0.88, 0.69, 1.00, 0.86),   # COLLECT ALL button
    )

    # Do not click the outermost ring; targets there are usually clipped.
    edge_margin: float = 0.012

    # ---- Layer 2: texture probe below the target ----
    #
    # DISABLED, and deliberately not retuned.
    #
    # It was designed as a cheap proxy for "a collectible plaque hangs above a
    # monster": empty ground is flat, a monster is not. Measurement killed it:
    # across 219 real detections the probe value ranged 41.8 to 72.4, while the
    # threshold was 12.0, so it rejected 0% - active in the log, protecting
    # nothing. Real terrain is far busier than any synthetic background.
    #
    # No separating value can be derived either: every measured sample is a true
    # positive, so there is no negative distribution to separate from. Picking a
    # number now would be inventing evidence.
    #
    # It is also redundant. The icon template already discriminates plaque
    # contents: coin plaques match, while heart, purple and flame plaques scored
    # below 0.50 and never registered at all.
    require_texture_below: bool = False
    texture_probe_height: float = 1.3
    texture_min_std: float = 12.0

    # ---- Layer 3: neighbour rules (need extra templates, off by default) ----
    # Enable once assets/templates/monster.png exists.
    require_monster_below: bool = False
    monster_target_name: str = "monster"
    monster_max_distance: int = 160

    # Enable once assets/templates/cloud.png exists. Note: what looked like a
    # "cloud covering a coin" turned out to be a different plaque type rather
    # than an overlay, which the icon template already rejects. This rule is
    # kept for genuine overlay cases only.
    reject_cloud_overlap: bool = False
    cloud_target_name: str = "cloud"

    # ---- Layer 4: cross-frame confirmation ----
    # A target must hold roughly the same position for this many consecutive
    # frames before it may be clicked. 1 disables the gate.
    confirm_frames: int = 2
    # Tolerance for "same position", in reference pixels.
    confirm_tolerance: int = 12
    # Drop a track after this many frames without a sighting.
    track_expire_frames: int = 3

    # ---- Layer 5: escalating cooldown and false-positive blacklist ----
    #
    # Cooldown is deliberately NOT flat. A successful click removes its target,
    # so waiting afterwards buys nothing; only a position that keeps surviving
    # clicks deserves to be backed off from. So the base cooldown is short
    # enough to just prevent firing twice at the same spot, and the wait
    # escalates per consecutive failure at that position before a ban.
    #
    # Base cooldown, applied when the position has no failure history.
    click_cooldown: float = 0.35
    # Wait after 1, 2, 3... consecutive failures at one position. The last entry
    # repeats if there are more failures than steps.
    failure_cooldown_ladder: Tuple[float, ...] = (1.0, 3.0, 8.0)
    # Consecutive "clicked but the target stayed" events before banning.
    # One higher than before, because the ladder now absorbs early failures.
    blacklist_after_failures: int = 4
    # Ban duration, seconds.
    blacklist_duration: float = 90.0
    # Spatial tolerance for "same position", in reference pixels. Must stay
    # below the 38.6px measured plaque spacing so neighbouring plaques are not
    # treated as one site.
    position_tolerance: int = 20

    # ---- Staleness guard for batched clicks ----
    #
    # Every coordinate in a batch comes from the single frame captured at the
    # start of that round. If anything moves the view after the first click - a
    # camera pan, a reflow, an animation nudging things - the remaining clicks in
    # that batch would fire at coordinates that no longer describe the screen.
    # Observed as occasional misclicks landing on scenery.
    #
    # With this on, each click after the first re-captures and confirms its
    # target is still within position_tolerance of where it was. If it is not,
    # the click is skipped and left for the next round.
    #
    # Cost: one extra capture-and-detect per additional click in a batch. Still
    # cheaper than splitting the batch into separate rounds, which would pay the
    # loop interval and the cross-frame confirmation gate again. Turn it off only
    # if throughput matters more than never misclicking.
    recheck_before_batch_click: bool = True

    # ---- Verification: the only self-correcting part of the stack ----
    #
    # Re-reading a frame after every click is the single largest fixed cost per
    # collected item, but dropping it would remove the mechanism that detects
    # false positives. So it is sampled instead of removed: verify every click
    # until the run has proven itself, then check periodically, and fall back to
    # verifying every click the moment a check fails.
    verify_after_click: bool = True
    # Delay before the re-read, to let the collection animation finish.
    # Too short and a successful click reads as a failure, which would poison the
    # blacklist with false bans. 0.28 still cleared the animation in live runs;
    # if false bans appear, raise this first.
    verify_delay: float = 0.28
    # Consecutive verified successes before switching to sampled verification.
    #
    # Lowered from 3 after live measurement: verification was 19-35% of round
    # time, and short sessions never left the "verify everything" phase at all,
    # so they paid the full cost throughout.
    verify_trust_after: int = 2
    # While trusted, verify one click in this many. Raised from 5 for the same
    # reason; the failure branch still snaps back to checking every click, so
    # sampling less often delays detection of a problem by at most one click.
    verify_sample_every: int = 8


@dataclass(frozen=True)
class DiagnosticsConfig:
    """Failure forensics.

    Off by default because it writes files. Turn it on the moment the live run
    disagrees with the calibration numbers: it dumps the actual frame the bot
    was looking at, which is the only way to tell a zoom change from a
    resolution change from a covering dialog.
    """

    # Save the frame whenever a round ends with nothing clickable.
    dump_frames_on_miss: bool = False
    # Anchored like template_dir, so dumps always land in the repository's
    # reports/ directory instead of wherever the process happened to start.
    dump_dir: str = _in_project("reports/misses")
    # Bounded so a long unattended session cannot fill the disk.
    max_dumps_per_session: int = 12
    # Draw detection boxes on the dump when there were any.
    annotate_dumps: bool = True


@dataclass(frozen=True)
class AppConfig:
    """Aggregate configuration; the single entry point for every layer."""

    window: WindowConfig = field(default_factory=WindowConfig)
    click: ClickConfig = field(default_factory=ClickConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)


DEFAULT_CONFIG = AppConfig()
