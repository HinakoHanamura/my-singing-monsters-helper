"""Capture frames from the game window for template cropping and dataset work.

Why a dedicated tool
--------------------
Template matching is not scale invariant, so a template must be cropped from a
frame produced by the exact same capture path the bot uses at runtime
(PrintWindow on the client area). Cropping from a system screenshot tool gives
different pixel dimensions once display scaling is involved, and the template
silently stops matching.

This script reuses core.game_window.GameWindow, so what you crop here is
byte-for-byte the kind of frame the bot will see.

Two modes
---------
single shot::

    python tools/grab_frame.py

continuous recording (recommended for collecting material)::

    python tools/grab_frame.py --watch --interval 0.7 --duration 90

Continuous mode is the useful one for an animated target like a coin: it
captures several animation phases of the same object, records the
appear -> collect -> absent transitions, and yields negative frames (no target
present) for free. The same recordings later serve as a YOLO training set.

Near-duplicate frames are skipped by default so that a static screen does not
produce hundreds of identical files. Tune with --dedup (0 disables).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

# Project root is derived from this file's location, never from the current
# working directory, so the script behaves identically no matter where it is
# invoked from (and keeps working if the project folder is moved to another
# drive).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import DEFAULT_CONFIG  # noqa: E402
from core.game_window import GameWindow  # noqa: E402


def resolve_output_dir(raw: str) -> str:
    """Resolve --out against the project root, not the shell's cwd.

    Using os.path.abspath alone would silently scatter captures into whatever
    directory the script happened to be launched from.
    """
    if os.path.isabs(raw):
        return os.path.normpath(raw)
    return os.path.normpath(os.path.join(PROJECT_ROOT, raw))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture game window frames via PrintWindow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--title",
        default=DEFAULT_CONFIG.window.title,
        help="target window title",
    )
    parser.add_argument(
        "--out",
        default="captures",
        help="output directory (created if missing)",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="keep capturing until --duration elapses or Ctrl+C",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.25,
        help=(
            "seconds between captures in --watch mode. Keep this small: an "
            "animated target needs several phases per cycle, and transitions "
            "(a plaque vanishing after a click) are easy to miss at a coarse "
            "interval. Near-duplicate frames are dropped anyway, so a short "
            "interval costs little disk"
        ),
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=90.0,
        help="total seconds to record in --watch mode (0 = until Ctrl+C)",
    )
    parser.add_argument(
        "--dedup",
        type=float,
        default=1.5,
        help=(
            "skip a frame if its mean pixel difference from the previous saved "
            "frame is below this value (0 disables deduplication)"
        ),
    )
    parser.add_argument(
        "--prefix",
        default="frame",
        help="filename prefix",
    )
    return parser


def imwrite_unicode(path: str, image: np.ndarray) -> bool:
    """Write a PNG through imencode so non-ASCII paths work.

    cv2.imwrite goes through a narrow-char API and fails on paths containing
    non-ASCII characters, which is easy to hit on a Chinese Windows install.
    """
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        return False
    with open(path, "wb") as handle:
        handle.write(buffer.tobytes())
    return True


def frame_signature(image: np.ndarray) -> np.ndarray:
    """Small grayscale thumbnail used for cheap near-duplicate detection."""
    small = cv2.resize(image, (96, 54), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.int16)


def timestamp_name(prefix: str) -> str:
    now = datetime.now()
    return "%s_%s_%03d.png" % (prefix, now.strftime("%Y%m%d_%H%M%S"), now.microsecond // 1000)


def capture_once(window: GameWindow) -> Optional[np.ndarray]:
    frame = window.capture()
    if frame is None:
        return None
    return frame


def main() -> int:
    args = build_parser().parse_args()

    out_dir = resolve_output_dir(args.out)
    os.makedirs(out_dir, exist_ok=True)

    window = GameWindow(
        title=args.title,
        print_window_flag=DEFAULT_CONFIG.window.print_window_flag,
    )

    if not window.attach():
        print("Window not found: %r" % args.title)
        print("Start the game first. The window must not be minimized -")
        print("PrintWindow cannot read a minimized window.")
        return 1

    print("Attached to %r (HWND=%s)" % (args.title, window.hwnd))
    width, height = window.client_size()
    print("Client area: %dx%d" % (width, height))
    if (width, height) != tuple(DEFAULT_CONFIG.vision.reference_size):
        print(
            "NOTE: this differs from VisionConfig.reference_size %s."
            % (DEFAULT_CONFIG.vision.reference_size,)
        )
        print(
            "      If you crop templates from these frames, set reference_size "
            "to %dx%d." % (width, height)
        )
    print("Output: %s" % out_dir)

    if not args.watch:
        frame = capture_once(window)
        if frame is None:
            print("Capture failed (window minimized, or PrintWindow refused).")
            return 1
        name = timestamp_name(args.prefix)
        path = os.path.join(out_dir, name)
        if not imwrite_unicode(path, frame):
            print("Failed to encode PNG.")
            return 1
        print("Saved %s (%dx%d)" % (name, frame.shape[1], frame.shape[0]))
        return 0

    # ---- continuous recording ----
    print("")
    print("Recording every %.2fs for %s. Press Ctrl+C to stop early." % (
        args.interval,
        ("%.0fs" % args.duration) if args.duration > 0 else "an unlimited time",
    ))
    print("Go play: let coins appear, collect a few by hand, and leave some")
    print("stretches with no coins on screen. Negative frames matter too.")
    print("")

    started = time.monotonic()
    saved = 0
    skipped = 0
    failed = 0
    previous: Optional[np.ndarray] = None

    try:
        while True:
            if args.duration > 0 and time.monotonic() - started >= args.duration:
                break

            loop_start = time.monotonic()
            frame = capture_once(window)

            if frame is None:
                failed += 1
                if failed in (1, 10, 50):
                    print("  capture failed (minimized or window lost) x%d" % failed)
            else:
                signature = frame_signature(frame)
                duplicate = False
                if args.dedup > 0 and previous is not None:
                    diff = float(np.mean(np.abs(signature - previous)))
                    duplicate = diff < args.dedup

                if duplicate:
                    skipped += 1
                else:
                    name = timestamp_name(args.prefix)
                    if imwrite_unicode(os.path.join(out_dir, name), frame):
                        saved += 1
                        previous = signature
                        print("  [%3d] %s" % (saved, name))
                    else:
                        failed += 1

            elapsed = time.monotonic() - loop_start
            remaining = args.interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print("")
        print("Stopped by user.")

    print("")
    print("Saved %d frames, skipped %d near-duplicates, %d capture failures."
          % (saved, skipped, failed))
    print("Directory: %s" % out_dir)
    if saved == 0:
        print("Nothing was saved. Is the window minimized or fully static?")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
