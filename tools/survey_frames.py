"""Survey a capture directory and write a per-frame report to reports/.

Why this exists
---------------
A recording session produces hundreds of frames spanning several unrelated
scenes (island view, menus, the minigame, plus degenerate frames captured while
the window was closing). Opening them one by one to find the interesting
segment does not scale, and the interactive terminal here truncates anything
that takes longer than about a second, so the result has to land in a file.

The report answers three questions cheaply:

- Which frames are degenerate and must be discarded (tiny file, collapsed
  client rect, near-uniform image)?
- Where does one scene end and the next begin (large frame-to-frame delta)?
- Which frames are near-duplicates of their neighbour and can be skipped when
  picking calibration material?

Usage
-----
    python tools/survey_frames.py captures/memory
    python tools/survey_frames.py captures/memory --out reports/memory_survey.txt
    python tools/survey_frames.py captures/memory --cut 6.0

Output is a fixed-width table plus a "segments" section listing runs of frames
separated by scene cuts. Stale output is removed up front, because reading a
previous run's report has produced wrong conclusions before.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent

# A frame smaller than this many pixels in either axis cannot be a real game
# client area; it is what PrintWindow returns while the window is closing.
MIN_SIDE = 64

# Standard deviation below this means the image carries no structure at all.
FLAT_STD = 3.0

# Downscale used for the frame-to-frame delta. Small enough to be fast, large
# enough that a single card flipping still moves the number.
DELTA_SIZE = (256, 192)

# Default mean-absolute-difference threshold that counts as a scene cut.
DEFAULT_CUT = 6.0


def parse_stamp(name: str) -> str:
    """Extract HH:MM:SS.mmm from frame_YYYYMMDD_HHMMSS_mmm.png."""
    parts = name.replace(".png", "").split("_")
    if len(parts) < 4:
        return "?"
    hhmmss, millis = parts[2], parts[3]
    return "%s:%s:%s.%s" % (hhmmss[0:2], hhmmss[2:4], hhmmss[4:6], millis)


def stamp_seconds(name: str) -> float:
    """Seconds since midnight, for computing gaps between frames."""
    parts = name.replace(".png", "").split("_")
    if len(parts) < 4:
        return float("nan")
    hhmmss, millis = parts[2], parts[3]
    return (
        int(hhmmss[0:2]) * 3600
        + int(hhmmss[2:4]) * 60
        + int(hhmmss[4:6])
        + int(millis) / 1000.0
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Survey captured frames.")
    parser.add_argument("directory", help="capture directory, e.g. captures/memory")
    parser.add_argument("--out", default=None, help="report path (default reports/<name>_survey.txt)")
    parser.add_argument(
        "--cut",
        type=float,
        default=DEFAULT_CUT,
        help="mean-abs-diff above which a frame starts a new segment",
    )
    args = parser.parse_args()

    source = (ROOT / args.directory).resolve()
    if not source.is_dir():
        print("not a directory: " + str(source))
        return 2

    out_path = (
        (ROOT / args.out).resolve()
        if args.out
        else ROOT / "reports" / (source.name + "_survey.txt")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Stale reports have caused wrong conclusions; never leave one behind.
    out_path.unlink(missing_ok=True)

    frames = sorted(source.glob("*.png"))
    if not frames:
        print("no png files in " + str(source))
        return 2

    lines = []
    lines.append("survey of " + source.as_posix())
    lines.append("frames: %d   cut threshold: %.1f" % (len(frames), args.cut))
    lines.append("")
    lines.append(
        "%-4s %-13s %6s %-11s %8s %6s %7s %s"
        % ("idx", "time", "gap", "size", "bytes", "std", "delta", "flag")
    )
    lines.append("-" * 76)

    previous_small = None
    previous_seconds = None
    segments = []
    current_segment = None
    bad = []

    for index, path in enumerate(frames):
        raw_size = path.stat().st_size
        image = cv2.imread(str(path))
        stamp = parse_stamp(path.name)
        seconds = stamp_seconds(path.name)
        gap = (
            ""
            if previous_seconds is None or seconds != seconds
            else "%.2f" % (seconds - previous_seconds)
        )
        previous_seconds = seconds

        if image is None:
            lines.append(
                "%-4d %-13s %6s %-11s %8d %6s %7s %s"
                % (index, stamp, gap, "unreadable", raw_size, "-", "-", "DISCARD")
            )
            bad.append(index)
            previous_small = None
            continue

        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        std = float(gray.std())

        flags = []
        if height < MIN_SIDE or width < MIN_SIDE:
            flags.append("COLLAPSED")
        if std < FLAT_STD:
            flags.append("FLAT")

        small = cv2.resize(gray, DELTA_SIZE, interpolation=cv2.INTER_AREA).astype(np.int16)
        if previous_small is None or previous_small.shape != small.shape:
            delta = float("nan")
            delta_text = "-"
        else:
            delta = float(np.abs(small - previous_small).mean())
            delta_text = "%.2f" % delta
        previous_small = small

        is_bad = bool(flags)
        if is_bad:
            bad.append(index)

        cut = is_bad or delta != delta or delta >= args.cut
        if cut:
            if current_segment is not None:
                segments.append(current_segment)
            current_segment = [index, index, 1]
            flags.append("CUT")
        else:
            current_segment[1] = index
            current_segment[2] += 1

        lines.append(
            "%-4d %-13s %6s %-11s %8d %6.1f %7s %s"
            % (
                index,
                stamp,
                gap,
                "%dx%d" % (width, height),
                raw_size,
                std,
                delta_text,
                " ".join(flags),
            )
        )

    if current_segment is not None:
        segments.append(current_segment)

    lines.append("")
    lines.append("segments (start, end, length) -- a new segment starts at each CUT")
    lines.append("-" * 76)
    for start, end, length in segments:
        lines.append(
            "%-4d .. %-4d  n=%-4d  %s .. %s"
            % (start, end, length, parse_stamp(frames[start].name), parse_stamp(frames[end].name))
        )

    lines.append("")
    lines.append("discard candidates: " + (", ".join(str(i) for i in bad) if bad else "none"))
    lines.append("")
    lines.append("index -> filename")
    lines.append("-" * 76)
    for index, path in enumerate(frames):
        lines.append("%-4d %s" % (index, path.name))

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote " + out_path.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    sys.exit(main())
