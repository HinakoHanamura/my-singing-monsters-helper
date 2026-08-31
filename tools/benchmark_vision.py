"""Measure where detection time actually goes, on real captured frames.

Why this exists
---------------
Throughput was tuned several times by shrinking sleep constants, without anyone
measuring what the non-sleep part of a round costs. That is the same mistake as
inventing a threshold: adjusting the part that is easy to see instead of the part
that dominates.

This benchmark runs the real `VisionAgent` over real saved frames, so it needs no
game running and no guesswork. It reports, per configuration:

  - milliseconds per frame,
  - how many targets were found,
  - whether the detections still agree with the full-quality baseline.

That last column is the important one. A configuration that is four times faster
but silently loses targets, or moves them by more than the click tolerance, is
not an optimisation - it is a regression with a nice number next to it.

Usage
-----
    python tools/benchmark_vision.py
    python tools/benchmark_vision.py --frames captures/calib_mixed --limit 20
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import DEFAULT_CONFIG  # noqa: E402
from core.geometry import distance  # noqa: E402
from core.vision_agent import Detection, VisionAgent  # noqa: E402


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(PROJECT_ROOT, path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure detection cost on real frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--frames", default="captures/calib_1024")
    parser.add_argument("--target", default="coin")
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=3,
                        help="timing passes per configuration; the best is reported")
    parser.add_argument("--out", default="reports/benchmark_vision.txt")
    return parser


class Tee:
    def __init__(self) -> None:
        self._lines: List[str] = []

    def __call__(self, text: str = "") -> None:
        print(text)
        self._lines.append(text)

    def save(self, path: str) -> Optional[str]:
        target = resolve(path)
        try:
            os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
            with open(target, "w", encoding="utf-8") as handle:
                handle.write("\n".join(self._lines) + "\n")
            return target
        except Exception as exc:
            print("could not write transcript: %r" % (exc,))
            return None


def load_frames(directory: str, limit: int) -> List[np.ndarray]:
    if not os.path.isdir(directory):
        raise SystemExit("frame directory not found: %s" % directory)
    names = sorted(n for n in os.listdir(directory) if n.lower().endswith(".png"))
    if not names:
        raise SystemExit("no PNG frames in %s" % directory)
    if limit and len(names) > limit:
        step = len(names) / float(limit)
        names = [names[int(i * step)] for i in range(limit)]

    frames = []
    for name in names:
        raw = np.fromfile(os.path.join(directory, name), dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if image is not None:
            frames.append(image)
    return frames


def make_agent(reference_size, **overrides) -> VisionAgent:
    settings = dict(
        template_dir=resolve(DEFAULT_CONFIG.vision.template_dir),
        match_threshold=DEFAULT_CONFIG.vision.match_threshold,
        reference_size=reference_size,
        scale_steps=DEFAULT_CONFIG.vision.scale_steps,
        nms_distance=DEFAULT_CONFIG.vision.nms_distance,
        max_detections=DEFAULT_CONFIG.vision.max_detections,
        use_fake_detection=False,
    )
    settings.update(overrides)
    return VisionAgent(**settings)


def time_agent(
    agent: VisionAgent, frames: Sequence[np.ndarray], target: str, repeats: int
) -> Tuple[float, List[List[Detection]]]:
    """Return (best milliseconds per frame, detections from the first pass)."""
    best = None
    captured: List[List[Detection]] = []
    for attempt in range(max(1, repeats)):
        results: List[List[Detection]] = []
        start = time.perf_counter()
        for frame in frames:
            results.append(agent.detect(target, frame))
        elapsed = (time.perf_counter() - start) / len(frames) * 1000.0
        if best is None or elapsed < best:
            best = elapsed
        if attempt == 0:
            captured = results
    return float(best or 0.0), captured


def compare(
    baseline: Sequence[Sequence[Detection]],
    candidate: Sequence[Sequence[Detection]],
    tolerance: float,
) -> Dict[str, int]:
    """How faithfully a configuration reproduces the baseline detections."""
    matched = 0
    missing = 0
    extra = 0
    drift: List[float] = []

    for base_frame, cand_frame in zip(baseline, candidate):
        remaining = list(cand_frame)
        for base in base_frame:
            best = None
            best_gap = tolerance + 1.0
            for other in remaining:
                gap = distance(base.center, other.center)
                if gap < best_gap:
                    best, best_gap = other, gap
            if best is not None and best_gap <= tolerance:
                matched += 1
                drift.append(best_gap)
                remaining.remove(best)
            else:
                missing += 1
        extra += len(remaining)

    return {
        "matched": matched,
        "missing": missing,
        "extra": extra,
        "max_drift": int(round(max(drift))) if drift else 0,
    }


def main() -> int:
    args = build_parser().parse_args()
    say = Tee()

    frames_dir = resolve(args.frames)
    frames = load_frames(frames_dir, args.limit)
    height, width = frames[0][:2] if False else frames[0].shape[:2]
    reference_size = (width, height)
    tolerance = float(DEFAULT_CONFIG.safety.position_tolerance)

    say("frames      : %d from %s" % (len(frames), frames_dir))
    say("frame size  : %dx%d" % (width, height))
    say("target      : %s" % args.target)
    say("tolerance   : %.0f px (position_tolerance; drift beyond this matters)" % tolerance)
    say("repeats     : %d (best pass reported)" % args.repeats)
    say("")

    # Baseline: exactly what the shipped configuration does today.
    baseline_agent = make_agent(reference_size)
    baseline_ms, baseline_hits = time_agent(
        baseline_agent, frames, args.target, args.repeats
    )
    baseline_count = sum(len(h) for h in baseline_hits)

    say("=" * 76)
    say("BASELINE  (shipped config: %d scale steps, full frame, full resolution)"
        % len(DEFAULT_CONFIG.vision.scale_steps))
    say("  %.1f ms/frame   %d detections total (%.1f per frame)"
        % (baseline_ms, baseline_count, baseline_count / len(frames)))
    say("")

    say("CANDIDATES")
    say("%-46s %9s %7s %7s %6s %6s %6s" % (
        "configuration", "ms/frame", "speedup", "found", "match", "miss", "drift"))
    say("-" * 76)

    five = (0.94, 0.97, 1.0, 1.03, 1.06)
    three = (0.97, 1.0, 1.03)

    candidates: List[Tuple[str, VisionAgent]] = [
        # Does step count actually earn its cost? Measured, not assumed.
        ("5 steps, full res", make_agent(reference_size, scale_steps=five)),
        ("1 step, full res", make_agent(reference_size, scale_steps=(1.0,))),
        # Resolution is the other multiplier.
        ("3 steps, 0.75 res", make_agent(reference_size, match_downscale=0.75)),
        ("3 steps, half res", make_agent(reference_size, match_downscale=0.5)),
        (
            "5 steps, half res",
            make_agent(reference_size, scale_steps=five, match_downscale=0.5),
        ),
        (
            "3 steps, 0.4 res",
            make_agent(reference_size, match_downscale=0.4),
        ),
        ("1 step, half res", make_agent(reference_size, match_downscale=0.5,
                                        scale_steps=(1.0,))),
        # Cropping to the playfield: measured, and it turned out to cost more
        # than it saves, because the crop copies memory and leaves matchTemplate
        # working on a non-contiguous view.
        (
            "3 steps, playfield crop",
            make_agent(reference_size, search_region=(0.14, 0.08, 1.0, 0.83)),
        ),
        (
            "5 steps, half res, playfield crop",
            make_agent(
                reference_size,
                scale_steps=five,
                match_downscale=0.5,
                search_region=(0.14, 0.08, 1.0, 0.83),
            ),
        ),
    ]
    _ = three  # documented for reference; it is the baseline

    for label, agent in candidates:
        try:
            ms, hits = time_agent(agent, frames, args.target, args.repeats)
        except TypeError as exc:
            say("%-46s  UNSUPPORTED (%s)" % (label, exc))
            continue
        found = sum(len(h) for h in hits)
        stats = compare(baseline_hits, hits, tolerance)
        say("%-46s %9.1f %6.1fx %7d %6d %6d %6d" % (
            label,
            ms,
            baseline_ms / ms if ms else 0.0,
            found,
            stats["matched"],
            stats["missing"],
            stats["max_drift"],
        ))

    say("-" * 76)
    say("match = baseline targets still found within tolerance")
    say("miss  = baseline targets lost")
    say("drift = worst centre displacement, in pixels")
    say("")
    say("A configuration is only usable if miss is 0 and drift stays well under")
    say("the click tolerance. Speed with misses is a regression, not a win.")

    written = say.save(args.out)
    if written:
        print("")
        print("transcript written to %s" % written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
