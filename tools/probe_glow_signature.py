"""Measure whether the match-flash detector can tell a flash from ordinary art.

Why this exists
---------------
The glow thresholds in ``core/minigames/fingerprint.py`` were the one set of
numbers in this project chosen from a visual description rather than from
measurement, and the first live run showed what that costs: a single freshly
revealed card was reported as "glowing" for 4.3 seconds, and the report stopped
only after the pair had already resolved. A real flash is brief and happens *on*
a match, so the signal was the card's own artwork.

The consequence is not cosmetic. ``is_settled`` gates fingerprinting, so a card
whose art sits in the flash's hue window can never be sampled, and the runner
times out on it with CARD_NOT_REVEALED. Card faces include gold coins, XP stars
and diamonds, so this is the common case rather than an edge one.

Labelling without labels
------------------------
Which frames show a flash is not written down anywhere, but it does not need to
be. A settled card is a static sprite and correlates almost perfectly with itself
one capture later; a card mid-animation does not. So self-similarity across
consecutive captures splits the face-up cards into two groups without anyone
labelling anything:

    stable   (sim >= stability_threshold)  -> settled art. Any glow reported here
                                              is a false positive by construction.
    changing (sim <  stability_threshold)  -> animating, flash included.

The number that matters is the share of *stable* cards the current threshold
would reject, because each one is a card the runner would hang on.

Usage
-----
    python tools/probe_glow_signature.py captures/memory
    python tools/probe_glow_signature.py captures/memory --range 194:228

Results land in reports/glow_signature.txt.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from dataclasses import replace
from typing import Dict, List, Optional

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.minigames.board import SlotMap  # noqa: E402
from core.minigames.card_vision import (  # noqa: E402
    CARD_BACK,
    CardBackVision,
    board_screen_verdict,
)
from core.minigames.fingerprint import (  # noqa: E402
    DEFAULT_FINGERPRINT_PARAMS,
    crop_face,
    glow_score,
    similarity,
)

DEFAULT_RANGES = [
    (22, 27),
    (33, 44),
    (50, 62),
    (68, 80),
    (81, 102),
    (108, 131),
    (132, 155),
    (194, 228),
]


def parse_range(text: str):
    start, _, end = text.partition(":")
    return int(start), int(end)


def percentiles(values: List[float]) -> str:
    if not values:
        return "(no samples)"
    array = np.array(values)
    return "n=%-5d min %.3f  p50 %.3f  p90 %.3f  p99 %.3f  max %.3f" % (
        array.size,
        array.min(),
        np.percentile(array, 50),
        np.percentile(array, 90),
        np.percentile(array, 99),
        array.max(),
    )


def hue_profile(frame: np.ndarray, box) -> str:
    """Where the bright, saturated pixels of a card actually sit in hue.

    Reported for the worst false positives so the window can be re-derived rather
    than nudged: if ordinary art fills the same hues as the flash, no threshold on
    this window will work and the discriminator has to change.
    """
    x1, y1, x2, y2 = box
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    bright = (saturation >= 90) & (value >= 200)
    if not bright.any():
        return "no bright saturated pixels"
    selected = hue[bright]
    share = bright.mean()
    return "bright share %.2f, hue p10 %d p50 %d p90 %d" % (
        share,
        np.percentile(selected, 10),
        np.percentile(selected, 50),
        np.percentile(selected, 90),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure the glow detector.")
    parser.add_argument("directory")
    parser.add_argument("--range", action="append", dest="ranges", default=None)
    parser.add_argument("--out", default="reports/glow_signature.txt")
    # Window overrides, so a candidate can be measured before it is committed to
    # code. The first measurement showed the false positives clustered at the low
    # end of the hue window, which is a reason to move the window rather than to
    # nudge the cutoff.
    parser.add_argument("--hue-low", type=int, default=None)
    parser.add_argument("--hue-high", type=int, default=None)
    parser.add_argument("--sat-low", type=int, default=None)
    parser.add_argument("--val-low", type=int, default=None)
    args = parser.parse_args()

    source = (ROOT / args.directory).resolve()
    frames = sorted(source.glob("*.png"))
    if not frames:
        print("no png files in " + str(source))
        return 2

    ranges = [parse_range(r) for r in args.ranges] if args.ranges else DEFAULT_RANGES

    out_path = (ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.unlink(missing_ok=True)

    vision = CardBackVision()
    params = DEFAULT_FINGERPRINT_PARAMS
    overrides = {
        name: value
        for name, value in (
            ("glow_hue_low", args.hue_low),
            ("glow_hue_high", args.hue_high),
            ("glow_saturation_low", args.sat_low),
            ("glow_value_low", args.val_low),
        )
        if value is not None
    }
    if overrides:
        params = replace(params, **overrides)

    lines = [
        "glow signature over " + source.as_posix(),
        "current window: H [%d,%d]  S>=%d  V>=%d  reject when share >= %.2f"
        % (
            params.glow_hue_low,
            params.glow_hue_high,
            params.glow_saturation_low,
            params.glow_value_low,
            params.glow_fraction,
        ),
        "stability threshold %.2f" % params.stability_threshold,
        "",
    ]

    stable_scores: List[float] = []
    changing_scores: List[float] = []
    worst: List[tuple] = []

    for start, end in ranges:
        scanned = []
        for index in range(start, min(end, len(frames) - 1) + 1):
            image = cv2.imread(str(frames[index]))
            if image is None:
                continue
            boxes = [d.bbox for d in vision.detect(CARD_BACK, image)]
            scanned.append((index, image, boxes))
        if not scanned:
            continue

        anchor = max(scanned, key=lambda item: (len(item[2]), -item[0]))
        if not board_screen_verdict(anchor[2]).ok:
            continue
        slot_map = SlotMap(anchor[2])

        previous: Dict[int, np.ndarray] = {}
        for index, image, boxes in scanned:
            observation = slot_map.observe(boxes)
            if not observation.ok:
                continue
            for slot in sorted(observation.face_up):
                box = slot_map.box(slot)
                crop = crop_face(image, box, params)
                if crop is None:
                    continue
                score = glow_score(image, box, params)
                earlier = previous.get(slot)
                previous[slot] = crop
                if earlier is None:
                    continue
                sim = similarity(earlier, crop, params)
                if sim >= params.stability_threshold:
                    stable_scores.append(score)
                    if score >= params.glow_fraction:
                        worst.append((score, index, slot, image, box))
                else:
                    changing_scores.append(score)

    lines.append("=" * 74)
    lines.append("glow score of SETTLED cards (any hit here is a false positive)")
    lines.append("  " + percentiles(stable_scores))
    lines.append("")
    lines.append("glow score of CHANGING cards (flash lives in here, mixed with flips)")
    lines.append("  " + percentiles(changing_scores))
    lines.append("=" * 74)
    lines.append("")

    if stable_scores:
        stable = np.array(stable_scores)
        rejected = int((stable >= params.glow_fraction).sum())
        lines.append(
            "settled cards the current threshold would refuse to sample: %d / %d (%.1f%%)"
            % (rejected, stable.size, 100.0 * rejected / stable.size)
        )
        lines.append("each one is a card the runner would time out on.")
        lines.append("")

        # Highest score among settled cards is the floor any usable threshold has
        # to clear; report it next to the changing distribution to show whether
        # room exists at all.
        lines.append("highest settled score : %.3f" % stable.max())
        if changing_scores:
            changing = np.array(changing_scores)
            lines.append("changing p50 / max    : %.3f / %.3f" % (np.median(changing), changing.max()))
            if changing.max() <= stable.max():
                lines.append("")
                lines.append(
                    "NO ROOM: animating cards never score above the settled maximum,"
                )
                lines.append(
                    "so no threshold on this window separates them. The discriminator"
                )
                lines.append("itself has to change, not its cutoff.")

    if worst:
        worst.sort(key=lambda item: -item[0])
        lines.append("")
        lines.append("worst false positives, with where their bright pixels actually sit:")
        for score, index, slot, image, box in worst[:12]:
            lines.append(
                "  frame %-5d slot %-3d score %.3f   %s"
                % (index, slot, score, hue_profile(image, box))
            )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote " + out_path.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    sys.exit(main())
