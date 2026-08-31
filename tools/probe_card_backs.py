"""Probe how reliably face-down memory-game cards can be located.

Why this exists
---------------
The minigame draws a different number of cards on each of its nine levels and
scales them to fit, so a card back measured on level 1 is roughly 125 px across
while the same art on level 9 is roughly 72 px. A single fixed-scale template
therefore cannot cover the range, and before committing to an approach it is
worth measuring whether a scale-free one works on the real recordings.

The card back is a saturated violet tile on a dark, desaturated wood
background, and every other card state (freshly flipped, matched-and-settled)
shows artwork in unrelated hues. That makes an HSV window on the violet plus a
squareness filter a scale-free detector: it keys off colour, not size.

This script reports, per frame, how many backs were found, their median size,
and the row shape that core.minigames.grid derives from them. Running it over
one frame per level answers three questions at once: does the colour window
hold across levels, does the squareness filter reject the UI furniture, and
does the row banding recover a sensible layout from a genuinely irregular
board.

Nothing here is the perception layer. It is a measurement tool, kept so the
same numbers can be reproduced after any threshold change.

Usage
-----
    python tools/probe_card_backs.py captures/memory
    python tools/probe_card_backs.py captures/memory --frames 22,33,108,194
    python tools/probe_card_backs.py captures/memory --debug-dir reports/card_backs

Results land in reports/card_backs_probe.txt because the interactive terminal
here truncates anything slower than about a second.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.minigames.grid import describe_grid, reading_order  # noqa: E402

# HSV window for the violet card back, in OpenCV ranges (H 0-179, S/V 0-255).
# Violet sits near H=135; the back is far brighter and more saturated than the
# wood background, which is what separates them.
BACK_HUE_LOW = 120
BACK_HUE_HIGH = 155
BACK_SAT_LOW = 45
BACK_VAL_LOW = 90

# A card is close to square. Generous bounds because the art has a wobbly,
# hand-drawn border and the drop shadow leaks into the contour.
MIN_ASPECT = 0.70
MAX_ASPECT = 1.40

# Absolute area bounds in pixels. The smallest observed card is about 72x72 and
# the largest about 130x130; the window is widened so a new level size does not
# silently fall outside it.
MIN_AREA = 40 * 40
MAX_AREA = 200 * 200

# Fraction of the bounding box that must actually be violet. Rejects the wood
# grain and any large background blob that happens to pass the colour window.
MIN_FILL = 0.45


def find_card_backs(image: np.ndarray):
    """Return bounding boxes of face-down cards as (x1, y1, x2, y2)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([BACK_HUE_LOW, BACK_SAT_LOW, BACK_VAL_LOW], dtype=np.uint8),
        np.array([BACK_HUE_HIGH, 255, 255], dtype=np.uint8),
    )

    # Close small gaps so the darker swirl in the middle of the tile does not
    # split one card into a ring of fragments.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < MIN_AREA or area > MAX_AREA:
            continue
        aspect = w / float(h) if h else 0.0
        if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
            continue
        fill = float(mask[y : y + h, x : x + w].mean()) / 255.0
        if fill < MIN_FILL:
            continue
        boxes.append((x, y, x + w, y + h))

    return boxes, mask


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe card-back detection.")
    parser.add_argument("directory", help="capture directory, e.g. captures/memory")
    parser.add_argument(
        "--frames",
        default=None,
        help="comma-separated frame indices to probe (default: every frame)",
    )
    parser.add_argument(
        "--debug-dir",
        default=None,
        help="write annotated copies of the probed frames here",
    )
    parser.add_argument("--out", default="reports/card_backs_probe.txt")
    args = parser.parse_args()

    source = (ROOT / args.directory).resolve()
    frames = sorted(source.glob("*.png"))
    if not frames:
        print("no png files in " + str(source))
        return 2

    if args.frames:
        wanted = [int(part) for part in args.frames.split(",") if part.strip()]
        selection = [(i, frames[i]) for i in wanted if 0 <= i < len(frames)]
    else:
        selection = list(enumerate(frames))

    out_path = (ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.unlink(missing_ok=True)

    debug_dir = None
    if args.debug_dir:
        debug_dir = (ROOT / args.debug_dir).resolve()
        debug_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("card-back probe over " + source.as_posix())
    lines.append(
        "hsv window H[%d,%d] S>=%d V>=%d   aspect [%.2f,%.2f]   fill >= %.2f"
        % (
            BACK_HUE_LOW,
            BACK_HUE_HIGH,
            BACK_SAT_LOW,
            BACK_VAL_LOW,
            MIN_ASPECT,
            MAX_ASPECT,
            MIN_FILL,
        )
    )
    lines.append("")
    lines.append(
        "%-5s %-7s %-9s %-9s %s" % ("idx", "backs", "med w x h", "size range", "row shape")
    )
    lines.append("-" * 78)

    for index, path in selection:
        image = cv2.imread(str(path))
        if image is None:
            lines.append("%-5d unreadable" % index)
            continue

        boxes, mask = find_card_backs(image)
        if boxes:
            widths = sorted(b[2] - b[0] for b in boxes)
            heights = sorted(b[3] - b[1] for b in boxes)
            median_text = "%dx%d" % (
                widths[len(widths) // 2],
                heights[len(heights) // 2],
            )
            range_text = "%d-%d" % (widths[0], widths[-1])
            shape = describe_grid(boxes)
        else:
            median_text = "-"
            range_text = "-"
            shape = "(none)"

        lines.append(
            "%-5d %-7d %-9s %-9s %s" % (index, len(boxes), median_text, range_text, shape)
        )

        if debug_dir is not None:
            annotated = image.copy()
            order = reading_order(boxes)
            for rank, box_index in enumerate(order):
                x1, y1, x2, y2 = boxes[box_index]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    str(rank),
                    (x1 + 4, y1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
            cv2.imwrite(str(debug_dir / ("probe_%04d.png" % index)), annotated)
            cv2.imwrite(str(debug_dir / ("mask_%04d.png" % index)), mask)

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote " + out_path.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    sys.exit(main())
