"""Earliest proof of concept: can we read the window without touching it?

This is the script that established the project's core premise, that
``PrintWindow`` can grab a game that is not focused and may be partly covered.
Everything in ``core/game_window.py`` grew out of these forty lines, so it is
kept as a historical record and as the smallest possible smoke check when
capture stops working: if this fails, the problem is below the framework.

Deliberately dependency-light and free of any project imports, so it still runs
when the rest of the codebase is broken. Uses PIL rather than the OpenCV path
the real code takes, which makes it an independent second opinion.

Run manually (the game must be running):
    python tests/manual/test_eyes.py

Note this does not apply the DPI fix that ``core/game_window.py`` does, so on a
scaled display it reports the oversized client rect and captures the game in the
top-left corner. That discrepancy is exactly how the DPI bug was found.
"""

import os
from ctypes import windll

import win32gui
import win32ui
from PIL import Image

WINDOW_NAME = "My Singing Monsters"


def main():
    hwnd = win32gui.FindWindow(None, WINDOW_NAME)
    if not hwnd:
        print("Failed! Window not found. Check if the game is running.")
        return 1

    print(f"Window found! HWND: {hwnd}")
    left, top, right, bot = win32gui.GetClientRect(hwnd)
    w = right - left
    h = bot - top

    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()
    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
    saveDC.SelectObject(saveBitMap)

    # Flag 3 == PW_RENDERFULLCONTENT, required for hardware-accelerated content.
    windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 3)

    bmpinfo = saveBitMap.GetInfo()
    bmpstr = saveBitMap.GetBitmapBits(True)
    im = Image.frombuffer(
        "RGB",
        (bmpinfo["bmWidth"], bmpinfo["bmHeight"]),
        bmpstr,
        "raw",
        "BGRX",
        0,
        1,
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "test_background.png")
    im.save(save_path)
    print(f"Success! Image saved exactly at: {save_path}")

    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)
    return 0


# Guarded so that importing this file, which pytest would otherwise do because
# the name matches test_*.py, cannot trigger a real capture as a side effect.
if __name__ == "__main__":
    raise SystemExit(main())
