"""Calibrate the face-similarity threshold against the real recordings.

The idea
--------
A matched pair stays on the board showing its picture, so on any mid-level frame
the cards that are *not* face down are exactly the matched ones -- an even number
of them, forming that many halves of pairs. Which card pairs with which is never
written down anywhere, but it does not need to be: a correct threshold is one
under which every visible card's best match is the card it pairs with, and the
pairing comes out complete with nothing left over.

That gives a labelling-free calibration. For each frame:

  * take the slot map built from the level's opening frame,
  * crop the faces of every slot that is no longer face down,
  * skip any card still lit by the match flash, which would be compared in the
    wrong state,
  * compute the full similarity matrix and pair up mutual best matches.

If the pairing is complete, the frame contributes two numbers: the *lowest*
similarity inside a true pair, and the *highest* similarity between cards from
different pairs. Over all frames, any threshold between the global maximum of the
second and the global minimum of the first separates them. The width of that
window is the margin, and reporting it is the point -- a threshold quoted without
a margin says nothing about whether it will hold.

Usage
-----
    python tools/calibrate_face_similarity.py captures/memory
    python tools/calibrate_face_similarity.py captures/memory --range 194:228

Results land in reports/face_similarity.txt.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

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
    mutual_best_pairs,
    pairwise_scores,
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate face similarity.")
    parser.add_argument("directory")
    parser.add_argument("--range", action="append", dest="ranges", default=None)
    parser.add_argument("--out", default="reports/face_similarity.txt")
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

    lines = [
        "face similarity calibration over " + source.as_posix(),
        "inset %.2f  canonical %dpx  glow cutoff %.2f"
        % (params.inset_ratio, params.canonical_size, params.glow_fraction),
        "",
    ]

    # Every measurement is kept, not just the extremes. A single frame caught
    # mid-flip -- past the glow but not yet showing its full face -- would
    # otherwise dictate the threshold on its own, and the interesting question is
    # whether such frames are rare outliers or the normal case.
    intra_samples: list = []
    inter_samples: list = []
    low_intra_frames: list = []

    usable_frames = 0
    skipped_incomplete = 0
    skipped_glow = 0

    #: Below this, a "true pair" score is treated as suspect and reported by
    #: frame, because real pairs measured about 0.96.
    SUSPECT_INTRA = 0.60

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

        anchor_index, anchor_image, anchor_boxes = max(
            scanned, key=lambda item: (len(item[2]), -item[0])
        )
        if not board_screen_verdict(anchor_boxes).ok:
            lines.append("frames %d..%d: no full board recorded, skipped" % (start, end))
            lines.append("")
            continue

        slot_map = SlotMap(anchor_boxes)
        lines.append("=" * 74)
        lines.append(
            "frames %d..%d   anchor %d   %d slots   card %dx%d"
            % (
                start,
                end,
                anchor_index,
                slot_map.slot_count,
                slot_map.card_size[0],
                slot_map.card_size[1],
            )
        )
        lines.append("=" * 74)
        lines.append(
            "  %-6s %-7s %-7s %-9s %-9s %s"
            % ("frame", "faceup", "used", "intra_min", "inter_max", "note")
        )
        lines.append("  " + "-" * 68)

        for index, image, boxes in scanned:
            observation = slot_map.observe(boxes)
            if not observation.ok:
                continue

            if not observation.face_down:
                # Nothing face down means the level is over and the board is
                # fading out. Measured on level 7: during the fade every card is
                # semi-transparent, correlation collapses, and the pairing comes
                # apart (6 pairs found, 10 cards left over). Those frames are not
                # evidence about the threshold, they are evidence about the fade.
                continue

            face_up = sorted(observation.face_up)
            if len(face_up) < 4:
                # Need at least two pairs, otherwise there is no "between
                # different pairs" number to measure.
                continue

            settled = []
            glowing = 0
            for slot in face_up:
                box = slot_map.box(slot)
                if glow_score(image, box, params) >= params.glow_fraction:
                    glowing += 1
                    continue
                crop = crop_face(image, box, params)
                if crop is not None:
                    settled.append((slot, crop))

            if glowing:
                skipped_glow += 1

            if len(settled) < 4 or len(settled) % 2 != 0:
                lines.append(
                    "  %-6d %-7d %-7d %-9s %-9s %s"
                    % (
                        index,
                        len(face_up),
                        len(settled),
                        "-",
                        "-",
                        "glow=%d skipped(odd or too few)" % glowing,
                    )
                )
                skipped_incomplete += 1
                continue

            crops = [crop for _, crop in settled]
            scores = pairwise_scores(crops)
            pairs, leftovers = mutual_best_pairs(scores)

            if leftovers or len(pairs) * 2 != len(crops):
                lines.append(
                    "  %-6d %-7d %-7d %-9s %-9s %s"
                    % (
                        index,
                        len(face_up),
                        len(settled),
                        "-",
                        "-",
                        "INCOMPLETE pairing: %d pairs, %d left over"
                        % (len(pairs), len(leftovers)),
                    )
                )
                skipped_incomplete += 1
                continue

            partner = {}
            for i, j in pairs:
                partner[i] = j
                partner[j] = i

            intra = [scores[i][j] for i, j in pairs]
            inter = []
            for i in range(len(crops)):
                for j in range(i + 1, len(crops)):
                    if partner[i] == j:
                        continue
                    inter.append(scores[i][j])

            intra_min = float(min(intra))
            inter_max = float(max(inter)) if inter else -1.0
            intra_samples.extend(float(v) for v in intra)
            inter_samples.extend(float(v) for v in inter)
            if intra_min < SUSPECT_INTRA:
                low_intra_frames.append((index, intra_min))
            usable_frames += 1

            lines.append(
                "  %-6d %-7d %-7d %-9.3f %-9.3f %s"
                % (
                    index,
                    len(face_up),
                    len(settled),
                    intra_min,
                    inter_max,
                    "glow=%d" % glowing if glowing else "",
                )
            )

        lines.append("")

    lines.append("=" * 74)
    lines.append("summary")
    lines.append("=" * 74)
    lines.append("usable frames          : %d" % usable_frames)
    lines.append("frames with glow skips : %d" % skipped_glow)
    lines.append("frames not usable      : %d" % skipped_incomplete)

    if usable_frames == 0 or not intra_samples or not inter_samples:
        lines.append("")
        lines.append("NO USABLE FRAMES -- cannot calibrate")
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("wrote " + out_path.relative_to(ROOT).as_posix())
        return 0

    intra = np.array(intra_samples)
    inter = np.array(inter_samples)

    lines.append("true-pair samples      : %d" % intra.size)
    lines.append(
        "  min %.4f   p1 %.4f   p5 %.4f   median %.4f"
        % (
            intra.min(),
            np.percentile(intra, 1),
            np.percentile(intra, 5),
            np.median(intra),
        )
    )
    lines.append("non-pair samples       : %d" % inter.size)
    lines.append(
        "  max %.4f   p99 %.4f   p95 %.4f   median %.4f"
        % (
            inter.max(),
            np.percentile(inter, 99),
            np.percentile(inter, 95),
            np.median(inter),
        )
    )

    if low_intra_frames:
        lines.append("")
        lines.append(
            "frames whose best true-pair score fell below %.2f (suspect: a card"
            % SUSPECT_INTRA
        )
        lines.append("caught mid-flip, or a wrong pair inferred):")
        for index, score in low_intra_frames:
            lines.append("  frame %-5d %.4f" % (index, score))

    lines.append("")
    absolute_gap = float(intra.min() - inter.max())
    robust_gap = float(np.percentile(intra, 1) - np.percentile(inter, 99))

    lines.append("absolute gap (min - max)   : %+.4f" % absolute_gap)
    lines.append("robust gap (p1 - p99)      : %+.4f" % robust_gap)

    if robust_gap <= 0:
        lines.append("")
        lines.append("NO SEPARATION even on robust percentiles.")
        lines.append("Revisit inset, canonical size or the glow cutoff before")
        lines.append("choosing any number.")
    else:
        suggested = float(np.percentile(intra, 1) + np.percentile(inter, 99)) / 2.0
        lines.append("")
        lines.append("suggested threshold        : %.3f" % suggested)
        lines.append(
            "  midpoint of the robust gap (%.4f, %.4f)"
            % (np.percentile(inter, 99), np.percentile(intra, 1))
        )
        if absolute_gap <= 0:
            lines.append("")
            lines.append(
                "NOTE: %d outlier frame(s) fall below this. A card measured while"
                % len(low_intra_frames)
            )
            lines.append(
                "still animating reads as a different picture, so the runtime must"
            )
            lines.append(
                "re-read rather than trust one frame -- which is what is_settled()"
            )
            lines.append("and returning None from identify() are for.")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote " + out_path.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    sys.exit(main())
