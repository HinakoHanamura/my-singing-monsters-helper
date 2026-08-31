"""Determine whether the game honours the coordinates carried by a click message.

The question this answers
------------------------
Background clicking assumes the game reads the position out of the message
(the lparam of WM_LBUTTONDOWN). Many engines do not: they treat the message only
as "a click happened" and take the position from the real cursor via
GetCursorPos. If this game is one of those, every synthetic click lands wherever
the physical mouse happens to rest, no matter what coordinates are sent - which
looks exactly like "all clicks go to the same wrong place".

Sending a message without error proves nothing about where it landed. This probe
measures the effect instead of the call.

How it discriminates
--------------------
1. Capture a frame and detect targets.
2. Park the OS cursor on one target (cursor position is saved and restored).
3. Send a click whose message coordinates point at a *different*, distant target.
4. Capture again and see which target disappeared.

   target from the message vanished  -> coordinates are honoured
   target under the parked cursor    -> the game uses the cursor, not the message
   nothing vanished                  -> the click had no effect at all

Note that "a coin was collected" on its own settles nothing: both outcomes
collect something. Which one disappeared is the whole answer, so read the RESULT
block rather than watching the screen.

Side effects, stated plainly: this clicks once in the live game (collecting one
resource) and briefly moves the mouse cursor, restoring it afterwards. Pass
--no-park to leave the cursor alone, at the cost of a weaker conclusion.

Usage
-----
    python tools/probe_click.py
    python tools/probe_click.py --no-park
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from ctypes import c_int, c_void_p, windll
from typing import List, Optional, Sequence, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import win32api  # noqa: E402
import win32con  # noqa: E402
import win32gui  # noqa: E402

from config import DEFAULT_CONFIG  # noqa: E402
from core.game_window import GameWindow, dpi_unaware_thread  # noqa: E402
from core.geometry import distance  # noqa: E402
from core.vision_agent import Detection, VisionAgent  # noqa: E402

_AWARENESS_NAMES = {
    0: "UNAWARE",
    1: "SYSTEM_AWARE",
    2: "PER_MONITOR_AWARE",
    3: "PER_MONITOR_AWARE_V2",
    4: "UNAWARE_GDISCALED",
}


def describe_dpi_awareness() -> str:
    """Report this thread's DPI awareness, so the transcript explains itself."""
    try:
        get_context = windll.user32.GetThreadDpiAwarenessContext
        get_context.restype = c_void_p
        from_context = windll.user32.GetAwarenessFromDpiAwarenessContext
        from_context.restype = c_int
        from_context.argtypes = [c_void_p]
        value = from_context(get_context())
        return _AWARENESS_NAMES.get(value, "unknown(%s)" % value)
    except Exception as exc:  # pragma: no cover
        return "unavailable (%r)" % (exc,)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check whether click message coordinates are honoured."
    )
    parser.add_argument("--target", default="coin")
    parser.add_argument(
        "--no-park",
        action="store_true",
        help="do not move the cursor (weaker conclusion, no cursor side effect)",
    )
    parser.add_argument(
        "--settle",
        type=float,
        default=1.0,
        help="seconds to wait after the click before re-reading the frame",
    )
    parser.add_argument(
        "--out",
        default="reports/probe_click.txt",
        help=(
            "also write the full transcript here, as UTF-8. Saves copying the "
            "console by hand"
        ),
    )
    parser.add_argument(
        "--load-qt",
        action="store_true",
        help=(
            "construct a QApplication before probing, which makes this process "
            "DPI-aware exactly like main.py. Use it to test whether the sender's "
            "DPI context is what breaks click placement in the app: the plain "
            "probe (no Qt) places clicks correctly, the app did not"
        ),
    )
    parser.add_argument(
        "--send-mode",
        choices=("raw", "guarded"),
        default="raw",
        help=(
            "raw sends the click exactly as the app used to; guarded sends it "
            "inside the game's DPI context, as ActionAgent does now. Combine "
            "with --load-qt to tell the two apart"
        ),
    )
    parser.add_argument(
        "--min-separation",
        type=float,
        default=120.0,
        help=(
            "refuse to draw a conclusion unless the two chosen targets are at "
            "least this far apart. Plaques stack about 39px apart, so with a "
            "small separation a freshly revealed plaque behind the collected one "
            "can masquerade as 'the other target survived' and invert the "
            "verdict. The instrument has to out-resolve the effect it measures"
        ),
    )
    parser.add_argument(
        "--frames-dir",
        default="reports/probe_frames",
        help=(
            "write annotated before/after frames here. The verdict below is a "
            "heuristic over disappearances; these images are the primary evidence"
        ),
    )
    return parser


class Tee:
    """Collects everything printed so it can be written to a file verbatim."""

    def __init__(self) -> None:
        self._lines: List[str] = []

    def __call__(self, text: str = "") -> None:
        print(text)
        self._lines.append(text)

    def save(self, path: str) -> Optional[str]:
        if not path:
            return None
        target = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
        try:
            os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
            with open(target, "w", encoding="utf-8") as handle:
                handle.write("\n".join(self._lines) + "\n")
            return target
        except Exception as exc:
            print("could not write transcript: %r" % (exc,))
            return None


def describe(detections: Sequence[Detection]) -> str:
    if not detections:
        return "(none)"
    return ", ".join(
        "(%d,%d)@%.2f" % (d.center[0], d.center[1], d.confidence) for d in detections
    )


def vanished(
    before: Sequence[Detection], after: Sequence[Detection], tolerance: float
) -> List[Detection]:
    """Targets present before and absent afterwards."""
    survivors = [d.center for d in after]
    return [
        d
        for d in before
        if all(distance(d.center, other) > tolerance for other in survivors)
    ]


def pick_pair(detections: Sequence[Detection]) -> Optional[Tuple[Detection, Detection]]:
    """Two detections as far apart as possible, so the outcome is unambiguous."""
    if len(detections) < 2:
        return None
    best = None
    best_gap = -1.0
    for index, first in enumerate(detections):
        for second in detections[index + 1 :]:
            gap = distance(first.center, second.center)
            if gap > best_gap:
                best_gap = gap
                best = (first, second)
    return best


def _raw_click(hwnd: int, x: int, y: int) -> None:
    lparam = win32api.MAKELONG(x, y)
    win32gui.SendMessage(hwnd, win32con.WM_MOUSEMOVE, 0, lparam)
    time.sleep(0.05)
    win32gui.SendMessage(hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lparam)
    time.sleep(0.06)
    win32gui.SendMessage(hwnd, win32con.WM_LBUTTONUP, 0, lparam)


def send_click(hwnd: int, x: int, y: int, mode: str) -> None:
    """Deliver a click, either as-is or inside the game's DPI context.

    The two modes exist to isolate one variable. `raw` reproduces what the app
    used to do; `guarded` reproduces what ActionAgent does now.
    """
    if mode == "guarded":
        with dpi_unaware_thread():
            _raw_click(hwnd, x, y)
    else:
        _raw_click(hwnd, x, y)


def run(args, say: Tee) -> int:
    window = GameWindow(
        DEFAULT_CONFIG.window.title, DEFAULT_CONFIG.window.print_window_flag
    )
    if not window.attach():
        say("Window not found: %r. Start the game first." % DEFAULT_CONFIG.window.title)
        return 1

    hwnd = window.hwnd
    client_w, client_h = window.client_size()
    say("attached HWND=%s  client=%dx%d" % (hwnd, client_w, client_h))
    say("process DPI awareness: %s" % describe_dpi_awareness())
    say("qt loaded: %s   send mode: %s" % (bool(args.load_qt), args.send_mode))
    say("cursor before probe: %s" % (win32api.GetCursorPos(),))

    vision = VisionAgent(
        template_dir=os.path.join(PROJECT_ROOT, DEFAULT_CONFIG.vision.template_dir),
        match_threshold=DEFAULT_CONFIG.vision.match_threshold,
        reference_size=DEFAULT_CONFIG.vision.reference_size,
        scale_steps=DEFAULT_CONFIG.vision.scale_steps,
        nms_distance=DEFAULT_CONFIG.vision.nms_distance,
        max_detections=DEFAULT_CONFIG.vision.max_detections,
        use_fake_detection=False,
    )

    frame = window.capture()
    if frame is None:
        say("capture failed; is the window minimized?")
        return 1
    before = vision.detect(args.target, frame)
    say("frame captured  %dx%d" % (frame.shape[1], frame.shape[0]))
    say("detected %d target(s): %s" % (len(before), describe(before)))

    if not before:
        say("")
        say("Nothing detected, so there is nothing to probe with.")
        say("Open an island with collectible plaques and run this again.")
        return 1

    pair = pick_pair(before)
    if pair is None:
        park_target = None
        message_target = before[0]
        say("")
        say("Only one target on screen, so the cursor cannot be parked elsewhere.")
        say("This run can only tell you whether the click had an effect, not")
        say("which position it used. Re-run with several plaques on screen.")
    else:
        park_target, message_target = pair
        say("")
        say("cursor will be parked at %s" % (park_target.center,))
        say("click message will carry  %s" % (message_target.center,))
        say("separation: %.0f px" % distance(park_target.center, message_target.center))

    saved_cursor = None
    parked = False
    if park_target is not None and not args.no_park:
        try:
            saved_cursor = win32api.GetCursorPos()
            screen_point = win32gui.ClientToScreen(hwnd, park_target.center)
            win32api.SetCursorPos(screen_point)
            parked = True
            say("cursor parked at screen %s (was %s)" % (screen_point, saved_cursor))
            time.sleep(0.2)
        except Exception as exc:
            say("could not park the cursor (%r); continuing without it" % (exc,))

    try:
        send_click(
            hwnd,
            message_target.center[0],
            message_target.center[1],
            args.send_mode,
        )
        say("click sent to client %s (mode=%s)" % (message_target.center, args.send_mode))
        time.sleep(args.settle)
        after_frame = window.capture()
        after = vision.detect(args.target, after_frame) if after_frame is not None else []
    finally:
        if parked and saved_cursor is not None:
            try:
                win32api.SetCursorPos(saved_cursor)
                say("cursor restored to %s" % (saved_cursor,))
            except Exception:
                say("WARNING: could not restore the cursor position")

    say("")
    say("after: %d target(s): %s" % (len(after), describe(after)))

    tolerance = max(12.0, float(DEFAULT_CONFIG.safety.position_tolerance))
    gone = vanished(before, after, tolerance)
    say("disappeared: %d -> %s" % (len(gone), describe(gone)))

    gone_centers = [d.center for d in gone]
    hit_message = any(
        distance(c, message_target.center) <= tolerance for c in gone_centers
    )
    hit_cursor = park_target is not None and any(
        distance(c, park_target.center) <= tolerance for c in gone_centers
    )

    say("")
    say("=" * 68)
    if hit_message and not hit_cursor:
        say("RESULT: message coordinates ARE honoured.")
        say("The target named in the message disappeared while the one under the")
        say("cursor survived. Background clicking at arbitrary coordinates works,")
        say("so any wrong-position bug lies elsewhere: coordinate space, a fixed")
        say("offset, or scaling.")
        verdict = 0
    elif hit_cursor and not hit_message:
        say("RESULT: the game uses the CURSOR position, not the message.")
        say("The target under the parked cursor disappeared while the one named in")
        say("the message survived. Clicking arbitrary coordinates without moving")
        say("the mouse is therefore not possible through window messages alone.")
        verdict = 2
    elif hit_message and hit_cursor:
        say("RESULT: both disappeared - inconclusive.")
        say("Two plaques vanished at once, so this cannot separate the two cases.")
        say("Likely the scene changed on its own. Run it again on a calmer screen.")
        verdict = 4
    elif not gone:
        say("RESULT: the click had NO effect.")
        say("Nothing disappeared. Either the game ignores injected mouse messages")
        say("entirely, or it needs the window to be active.")
        verdict = 3
    else:
        say("RESULT: inconclusive.")
        say("Something vanished but it matches neither position. The scene probably")
        say("changed by itself (a plaque appearing, or an animation).")
        verdict = 4
    say("=" * 68)
    return verdict


def main() -> int:
    args = build_parser().parse_args()

    app = None
    if args.load_qt:
        # Constructing a QApplication is what flips this process to DPI-aware,
        # which is the only relevant difference between this probe and main.py.
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])

    say = Tee()
    try:
        return run(args, say)
    finally:
        written = say.save(args.out)
        if written:
            print("")
            print("transcript written to %s" % written)
        if app is not None:
            app.quit()


if __name__ == "__main__":
    sys.exit(main())
