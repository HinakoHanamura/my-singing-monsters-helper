"""Measure how well "all card backs look alike" separates a card face.

Why this exists
---------------
Card-back detection is colour segmentation: violet hue, saturated, bright, and
filling most of its box. That is scale invariant, which is why it survives cards
from 122px down to 63px without recalibration.

What it cannot do is tell a violet *back* from a violet *card face*. A live level
8 board showed exactly that: a lavender monster card, revealed at slot 13, passed
every colour and shape filter.

    accepted box     size     aspect  fill   hue p10/p50/p90
    real backs       63x60    1.03    0.93   131/144/151
    the monster      64x57    1.12    0.76   139/144/145

Fill is the widest of those gaps and it is only 0.76 against 0.89, the lowest
real back ever measured. Raising the fill threshold into that gap would fix this
one card and nothing more general: a violet card that fills its frame would walk
straight through.

The fact worth using instead is one the recordings established long ago: within a
level, every back is the *same picture*. So a real back correlates almost
perfectly with the other backs in the same frame, while a card face does not
correlate with them at all. That is a much larger difference than any of the
colour margins, and this script measures it rather than assuming it.

What it does
------------
Runs the real detector over a frame, crops every accepted box the same way the
fingerprint layer does, and reports for each box its best correlation against any
other accepted box. Real backs cluster at the top; an impostor sits alone.

Usage
-----
    python tools/calibrate_back_similarity.py <frame.png> [<frame.png> ...]

Writes reports/back_similarity.txt and prints one line, per the tooling shape in
workflow.md.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.minigames.card_vision import CARD_BACK, CardBackVision  # noqa: E402
from core.minigames.fingerprint import (  # noqa: E402
    DEFAULT_FINGERPRINT_PARAMS,
    UNCOMPARABLE,
    crop_face,
    similarity,
)

REPORT = os.path.join(PROJECT_ROOT, "reports", "back_similarity.txt")


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.normpath(
        os.path.join(PROJECT_ROOT, path)
    )


def read_frame(path: str) -> np.ndarray:
    # imdecode rather than imread: imread fails on non-ASCII paths.
    buffer = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def measure(path: str, lines: List[str]) -> List[float]:
    frame = read_frame(path)
    if frame is None:
        lines.append("could not decode %s" % path)
        return []

    vision = CardBackVision()
    boxes = [d.bbox for d in vision.detect(CARD_BACK, frame)]
    crops = {}
    for box in boxes:
        crop = crop_face(frame, box, DEFAULT_FINGERPRINT_PARAMS)
        if crop is not None:
            crops[box] = crop

    lines.append("")
    lines.append("%s  %dx%d" % (os.path.basename(path), frame.shape[1], frame.shape[0]))
    lines.append("  accepted %d boxes, croppable %d" % (len(boxes), len(crops)))
    if len(crops) < 2:
        lines.append("  fewer than two crops; nothing to compare")
        return []

    rows = []
    for box, crop in crops.items():
        scores = [
            similarity(crop, other, DEFAULT_FINGERPRINT_PARAMS)
            for key, other in crops.items()
            if key != box
        ]
        scores = [s for s in scores if s > UNCOMPARABLE]
        if not scores:
            continue
        rows.append((max(scores), box))

    rows.sort()
    lines.append("  best correlation against any other accepted box, worst first:")
    for best, box in rows:
        lines.append("    %.3f  box (%d,%d,%d,%d)" % (best, box[0], box[1], box[2], box[3]))
    return [best for best, _ in rows]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("frames", nargs="+", help="frame images to measure")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    if os.path.exists(REPORT):
        os.unlink(REPORT)

    lines = ["measuring whether backs in one frame look alike"]
    every: List[float] = []
    for raw in args.frames:
        every.extend(measure(resolve(raw), lines))

    if every:
        array = np.array(every)
        lines.append("")
        lines.append("all boxes, best-correlation distribution")
        lines.append(
            "  min %.3f  p1 %.3f  p5 %.3f  median %.3f  max %.3f"
            % (
                array.min(),
                float(np.percentile(array, 1)),
                float(np.percentile(array, 5)),
                float(np.median(array)),
                array.max(),
            )
        )
        lines.append("")
        lines.append(
            "A threshold belongs in the gap between the impostors at the bottom of "
            "this list and the real backs above them. Both groups have to be "
            "present in the frames measured for the gap to mean anything."
        )

    with open(REPORT, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("wrote %s" % REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
