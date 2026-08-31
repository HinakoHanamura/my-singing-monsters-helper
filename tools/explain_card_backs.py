"""Explain, for one frame, every box the card-back detector produced and why.

Why this exists
---------------
A field failure gives you a saved frame and a symptom, not a measurement. The
first live run of the memory minigame stopped with CARD_NOT_REVEALED on a card
that had visibly flipped, and the saved frame showed a detection box *inside* the
revealed artwork. If that box lands within matching tolerance of the slot's
centre, the slot map keeps reporting the card as still face down and the runner
waits forever.

Eyeballing box sizes off a rendered PNG is not measurement, so this prints the
numbers: every accepted box with its width, height, aspect and fill, plus the
boxes that were *rejected* and which filter rejected them. That is what says
whether the fix belongs in the aspect bound, the size bound, the fill floor, or
somewhere else entirely.

It also groups accepted boxes into rows and reports, for each box, the distance
to the nearest other box centre -- because a spurious detection sitting on top of
a real card is exactly the case the slot matcher cannot distinguish from the real
thing.

Usage
-----
    python tools/explain_card_backs.py reports/manual_memory/stopped.png
    python tools/explain_card_backs.py captures/memory/frame_...png --debug

Results land next to the input as <name>_explained.txt, and with --debug an
annotated copy showing accepted boxes in green and rejected candidates in red.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.minigames.card_vision import (  # noqa: E402
    DEFAULT_CARD_BACK_PARAMS,
    CARD_BACK,
    CardBackVision,
    board_screen_verdict,
    card_back_mask,
    median_card_size,
    width_spread,
)


def hue_stats(frame, mask, x, y, w, h) -> Tuple[int, int, int]:
    """Hue percentiles of the in-window pixels inside a box.

    Reported because the same failure has now appeared twice: a colour window
    whose lower bound reaches far enough to swallow artwork it was never meant to
    match. Knowing where a box's hues actually sit says whether the window should
    move or whether some other filter is the right lever.
    """
    region = frame[y : y + h, x : x + w]
    region_mask = mask[y : y + h, x : x + w]
    if region.size == 0 or not (region_mask > 0).any():
        return (-1, -1, -1)
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0][region_mask > 0]
    return (
        int(np.percentile(hue, 10)),
        int(np.percentile(hue, 50)),
        int(np.percentile(hue, 90)),
    )


def classify_contour(contour, mask, params, frame) -> Tuple[str, dict]:
    """Repeat the detector's filters, but report which one rejects a candidate."""
    x, y, w, h = cv2.boundingRect(contour)
    facts = {
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "aspect": (w / float(h)) if h else 0.0,
        "fill": float(mask[y : y + h, x : x + w].mean()) / 255.0 if w and h else 0.0,
        "hue": hue_stats(frame, mask, x, y, w, h),
    }

    if not (params.min_side <= w <= params.max_side):
        return "REJECT_WIDTH", facts
    if not (params.min_side <= h <= params.max_side):
        return "REJECT_HEIGHT", facts
    if not (params.min_aspect <= facts["aspect"] <= params.max_aspect):
        return "REJECT_ASPECT", facts
    if facts["fill"] < params.min_fill:
        return "REJECT_FILL", facts
    return "ACCEPT", facts


def main() -> int:
    parser = argparse.ArgumentParser(description="Explain card-back detections.")
    parser.add_argument("image")
    parser.add_argument("--debug", action="store_true", help="also write an annotated png")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    image_path = (ROOT / args.image).resolve()
    frame = cv2.imread(str(image_path))
    if frame is None:
        print("cannot read " + str(image_path))
        return 2

    params = DEFAULT_CARD_BACK_PARAMS
    mask = card_back_mask(frame, params)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    accepted: List[dict] = []
    rejected: List[Tuple[str, dict]] = []
    for contour in contours:
        verdict, facts = classify_contour(contour, mask, params, frame)
        if verdict == "ACCEPT":
            accepted.append(facts)
        else:
            # Tiny specks are noise and would drown the report.
            if facts["w"] >= 12 and facts["h"] >= 12:
                rejected.append((verdict, facts))

    lines = [
        "explaining " + image_path.as_posix(),
        "frame %dx%d" % (frame.shape[1], frame.shape[0]),
        "filters: side [%d,%d]  aspect [%.2f,%.2f]  fill >= %.2f  H [%d,%d] S>=%d V>=%d"
        % (
            params.min_side,
            params.max_side,
            params.min_aspect,
            params.max_aspect,
            params.min_fill,
            params.hue_low,
            params.hue_high,
            params.saturation_low,
            params.value_low,
        ),
        "",
    ]

    accepted.sort(key=lambda f: (f["y"], f["x"]))
    boxes = [(f["x"], f["y"], f["x"] + f["w"], f["y"] + f["h"]) for f in accepted]
    centers = [((b[0] + b[2]) // 2, (b[1] + b[3]) // 2) for b in boxes]

    lines.append("ACCEPTED  %d boxes" % len(accepted))
    lines.append(
        "  %-4s %-13s %-9s %-7s %-6s %-16s %s"
        % ("idx", "box", "size", "aspect", "fill", "hue p10/p50/p90", "nearest")
    )
    lines.append("  " + "-" * 82)
    for index, facts in enumerate(accepted):
        cx, cy = centers[index]
        nearest = None
        for other_index, (ox, oy) in enumerate(centers):
            if other_index == index:
                continue
            distance = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
            if nearest is None or distance < nearest[0]:
                nearest = (distance, other_index)
        lines.append(
            "  %-4d (%3d,%3d)     %3dx%-3d  %-7.2f %-6.2f %-16s %s"
            % (
                index,
                facts["x"],
                facts["y"],
                facts["w"],
                facts["h"],
                facts["aspect"],
                facts["fill"],
                "%d/%d/%d" % facts["hue"],
                ("%.1f px to #%d" % nearest) if nearest else "-",
            )
        )

    lines.append("")
    if accepted:
        median = median_card_size(boxes)
        lines.append("median card size : %dx%d" % median)
        lines.append("width spread     : %d px" % width_spread(boxes))
        verdict = board_screen_verdict(boxes)
        lines.append("board gate       : %s  %s" % (verdict.code, verdict.reason))

        # A spurious box sitting on a real card is the case the slot matcher
        # cannot tell from the real thing, so call it out by name.
        tolerance = median[0] * 0.5
        lines.append("")
        lines.append(
            "slot matching tolerance would be %.1f px (half the median width)."
            % tolerance
        )
        crowded = []
        for index, (cx, cy) in enumerate(centers):
            for other in range(index + 1, len(centers)):
                ox, oy = centers[other]
                distance = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
                if distance <= tolerance:
                    crowded.append((index, other, distance))
        if crowded:
            lines.append(
                "PAIRS CLOSER THAN TOLERANCE -- these are indistinguishable to the"
            )
            lines.append("slot matcher, and one of them must be spurious:")
            for a, b, distance in crowded:
                lines.append("  #%d and #%d are %.1f px apart" % (a, b, distance))
        else:
            lines.append("no two accepted boxes fall within tolerance of each other.")

    lines.append("")
    lines.append("REJECTED  %d candidates larger than 12px" % len(rejected))
    if rejected:
        lines.append(
            "  %-16s %-13s %-9s %-7s %s" % ("filter", "box", "size", "aspect", "fill")
        )
        lines.append("  " + "-" * 68)
        rejected.sort(key=lambda item: -(item[1]["w"] * item[1]["h"]))
        for verdict, facts in rejected[:25]:
            lines.append(
                "  %-16s (%3d,%3d)     %3dx%-3d  %-7.2f %.2f"
                % (
                    verdict,
                    facts["x"],
                    facts["y"],
                    facts["w"],
                    facts["h"],
                    facts["aspect"],
                    facts["fill"],
                )
            )

    out_path = (
        (ROOT / args.out).resolve()
        if args.out
        else image_path.with_name(image_path.stem + "_explained.txt")
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if args.debug:
        annotated = frame.copy()
        for verdict, facts in rejected:
            cv2.rectangle(
                annotated,
                (facts["x"], facts["y"]),
                (facts["x"] + facts["w"], facts["y"] + facts["h"]),
                (0, 0, 255),
                1,
            )
        for index, box in enumerate(boxes):
            cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                str(index),
                (box[0] + 3, box[1] + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )
        debug_path = image_path.with_name(image_path.stem + "_explained.png")
        cv2.imwrite(str(debug_path), annotated)

    print("wrote " + out_path.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
