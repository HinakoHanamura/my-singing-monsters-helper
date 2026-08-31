"""Replay a recorded level through the real detection and tracking pipeline.

Why this exists
---------------
The tracking design rests on three claims that unit tests with synthetic boxes
cannot settle, because they are claims about this game's pixels:

1. Colour segmentation finds exactly the face-down cards, on every level, at
   every card size the game uses.
2. Slot identities assigned from the opening frame keep matching the same cards
   for the rest of the level, even though the set of face-down cards shrinks.
3. The screen gate refuses frames that are not a live board -- island, results
   screen, replay prompt -- instead of handing back plausible-looking boxes.

Choosing an anchor, and why it is not trivial
---------------------------------------------
A slot map may only be built from a frame where the *whole* layout is still face
down; that is a precondition of ``SlotMap``, not something it can check.

An early version of this script took the first frame the gate accepted, which
looked fine and was wrong. Cards come in pairs, so a board with an even number
already revealed passes an evenness test just as a full board does. On level 8
the recording opens with one card flipped (31 backs, correctly rejected as odd),
and the next frame has two flipped -- 30 backs, even, accepted. The level really
holds 32 cards, so the map would have been permanently blind to two of them.

There is no way to settle this from one frame with colour alone. A brightness
threshold that finds cards in either state was measured and rejected: adjacent
cards merge into one blob while header text and buttons add spurious ones, giving
32 where 30 was right and 27 where 32 was right.

So this script picks the frame with the *most* face-down cards in the range, and
separately reports whether any frame in the range ever showed more backs than the
anchor has slots. That second number is the honest check: if it is ever higher,
the anchor was not a full board and the replay for that level means nothing.

In live operation the question does not arise the same way. The bot performs every
click itself, so it knows when a level has just started, and the opening board is
verified to have no preview -- everything is face down. Inferring pristineness
from a single frame is only needed when joining somebody else's session midway,
which is what replaying a recording does.

Usage
-----
    python tools/probe_board_tracking.py captures/memory
    python tools/probe_board_tracking.py captures/memory --range 194:228

Results land in reports/board_tracking.txt.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import cv2

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.minigames.board import SlotMap  # noqa: E402
from core.minigames.card_vision import (  # noqa: E402
    CARD_BACK,
    CardBackVision,
    board_screen_verdict,
)

#: Levels of the recording in captures/memory, from reports/memory_survey.txt.
DEFAULT_RANGES = [
    (22, 27),
    (33, 44),
    (50, 62),
    (68, 80),
    (81, 102),
    (108, 131),
    (132, 155),
    (156, 188),
    (194, 228),
]

#: Frames that are known not to be a board, used as negative controls.
NEGATIVE_CONTROLS = [(229, "jiesuan-page"), (231, "replay-prompt"), (236, "island")]


def parse_range(text: str):
    start, _, end = text.partition(":")
    return int(start), int(end)


def boxes_for(vision: CardBackVision, frame) -> list:
    return [d.bbox for d in vision.detect(CARD_BACK, frame)]


def scan_range(vision, frames, start, end):
    """Detect once per frame in the range; returns [(index, boxes), ...]."""
    scanned = []
    for index in range(start, min(end, len(frames) - 1) + 1):
        image = cv2.imread(str(frames[index]))
        if image is None:
            continue
        scanned.append((index, boxes_for(vision, image)))
    return scanned


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay board tracking over frames.")
    parser.add_argument("directory")
    parser.add_argument(
        "--range",
        action="append",
        dest="ranges",
        default=None,
        help="frame index range as START:END, repeatable (default: all nine levels)",
    )
    parser.add_argument("--out", default="reports/board_tracking.txt")
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
    lines = ["board tracking replay over " + source.as_posix(), ""]
    problems = 0
    suspect_anchors = 0

    for start, end in ranges:
        lines.append("=" * 74)
        lines.append("frames %d..%d" % (start, end))
        lines.append("=" * 74)

        scanned = scan_range(vision, frames, start, end)
        if not scanned:
            lines.append("  NO READABLE FRAMES")
            problems += 1
            lines.append("")
            continue

        max_backs = max(len(boxes) for _, boxes in scanned)

        # Most face-down cards wins, earliest frame breaking ties. A full board
        # is the maximum by definition, so this cannot land below it.
        anchor_index, anchor_boxes = max(scanned, key=lambda item: (len(item[1]), -item[0]))
        verdict = board_screen_verdict(anchor_boxes)

        if not verdict.ok:
            lines.append(
                "  anchor %-4d rejected: %-18s %s"
                % (anchor_index, verdict.code, verdict.reason)
            )
            lines.append(
                "  max backs seen in range: %d -- no full board was recorded here"
                % max_backs
            )
            suspect_anchors += 1
            lines.append("")
            continue

        slot_map = SlotMap(anchor_boxes)
        lines.append(
            "  anchor %-4d accepted: %d slots, %s, card %dx%d, tol %.1f"
            % (
                anchor_index,
                slot_map.slot_count,
                slot_map.initial_shape,
                slot_map.card_size[0],
                slot_map.card_size[1],
                slot_map.match_tolerance(),
            )
        )

        if max_backs > slot_map.slot_count:
            lines.append(
                "  SUSPECT ANCHOR: some frame showed %d backs, more than the %d slots"
                % (max_backs, slot_map.slot_count)
            )
            suspect_anchors += 1

        lines.append("")
        lines.append(
            "  %-6s %-7s %-8s %-18s %s" % ("frame", "backs", "faceup", "code", "note")
        )
        lines.append("  " + "-" * 68)

        for index, boxes in scanned:
            gate = board_screen_verdict(boxes, expected_count=slot_map.slot_count)
            observation = slot_map.observe(boxes)

            notes = []
            if not gate.ok:
                notes.append("GATE:" + gate.code)
            if observation.ok:
                total = len(observation.face_down) + len(observation.face_up)
                if total != slot_map.slot_count:
                    notes.append("TOTAL_MISMATCH:%d" % total)
                    problems += 1
            else:
                problems += 1

            lines.append(
                "  %-6d %-7s %-8s %-18s %s"
                % (
                    index,
                    len(observation.face_down) if observation.ok else "-",
                    len(observation.face_up) if observation.ok else "-",
                    observation.code,
                    " ".join(notes),
                )
            )

        # A rising back count is normal, not a fault: a mismatched pair turns
        # face down again. It is reported as a summary so the shape of a level is
        # visible without reading every row.
        series = [len(boxes) for _, boxes in scanned]
        lines.append("")
        lines.append("  backs over time: " + " ".join(str(n) for n in series))
        lines.append(
            "  (rises are mismatches flipping two cards back; falls are pairs cleared)"
        )
        lines.append("")

    lines.append("=" * 74)
    lines.append("negative controls (must all be rejected)")
    lines.append("=" * 74)
    for index, label in NEGATIVE_CONTROLS:
        if index >= len(frames):
            continue
        image = cv2.imread(str(frames[index]))
        if image is None:
            continue
        boxes = boxes_for(vision, image)
        verdict = board_screen_verdict(boxes)
        if verdict.ok:
            problems += 1
            status = "ACCEPTED -- PROBLEM"
        else:
            status = "REJECTED"
        lines.append(
            "  %-5d %-14s %-20s boxes=%-3d %s"
            % (index, label, status, len(boxes), verdict.code)
        )

    lines.append("")
    lines.append("problems: %d" % problems)
    lines.append("suspect anchors: %d" % suspect_anchors)

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote " + out_path.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    sys.exit(main())
