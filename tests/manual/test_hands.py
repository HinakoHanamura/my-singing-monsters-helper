"""Earliest proof of concept: can we click without moving the real mouse?

Counterpart to test_eyes.py. It established that ``SendMessage`` with
WM_LBUTTONDOWN/UP reaches the game while the physical cursor stays where the
user left it, which is the entire safety argument of the project.
``core/action_agent.py`` is this idea plus jitter, cooldowns and verification.

Kept as the smallest possible smoke check: if this does not register in game,
the problem is below the framework.

WARNING: this really clicks. It fires at a fixed client coordinate with no
target checking and no anti-misclick guard, so whatever sits at (200, 200) gets
hit. For anything beyond "does input arrive at all", use tools/probe_click.py.

Run manually (the game must be running):
    python tests/manual/test_hands.py
"""

import time

import win32api
import win32con
import win32gui

WINDOW_NAME = "My Singing Monsters"

#: Arbitrary point in the client area. Not a target, just somewhere to poke.
CLICK_X, CLICK_Y = 200, 200


def main():
    hwnd = win32gui.FindWindow(None, WINDOW_NAME)
    if not hwnd:
        print("Failed! Window not found. Check if the game is running.")
        return 1

    lparam = win32api.MAKELONG(CLICK_X, CLICK_Y)
    print(f"({CLICK_X}, {CLICK_Y})")

    win32gui.SendMessage(hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lparam)
    time.sleep(0.05)
    win32gui.SendMessage(hwnd, win32con.WM_LBUTTONUP, 0, lparam)
    return 0


# Guarded so that importing this file, which pytest would otherwise do because
# the name matches test_*.py, cannot fire a real click as a side effect.
if __name__ == "__main__":
    raise SystemExit(main())
