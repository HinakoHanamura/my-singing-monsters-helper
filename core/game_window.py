"""System adaptation layer: window lookup and background capture.

The lowest layer, with one job:
  1. resolve a window title to an HWND and keep it valid,
  2. copy the window's client area with PrintWindow and hand back a BGR ndarray
     that OpenCV can consume directly.

It is kept separate from the perception layer because "seeing the screen" and
"understanding the screen" are different problems. Capture needs Windows APIs;
recognition should be pure image maths. Splitting them means the perception layer
can be tested against files on disk with no game running.

Nothing here activates, raises, or moves the target window, so the game may sit
behind other windows the whole time. It may not be **minimized**: Windows stops
rendering a minimized window, so PrintWindow has nothing to copy.

DPI awareness, and why this file cares
--------------------------------------
This bit is subtle and cost a live failure, so it is documented at length.

Windows reports window geometry differently depending on the *calling process's*
DPI awareness. The game is a DPI-unaware application: on a display scaled to
150%, Windows gives it a 1024x768 logical client area and stretches the result
for presentation.

  - From a DPI-unaware process, GetClientRect returns 1024x768 and PrintWindow
    produces a 1024x768 surface that the game fills completely.
  - From a DPI-aware process, GetClientRect returns the physical 1536x1152, and
    PrintWindow produces a 1536x1152 surface into which the game still renders
    only 1024x768, leaving black padding on the right and bottom.

Constructing a QApplication makes the process DPI-aware. So a plain capture
script and the Qt application saw *different* frame sizes for the same window,
which silently broke template matching in the app while the offline tools looked
fine: the frame appeared 1.5x larger, so every template was scaled up 1.5x and
matched nothing.

The fix is to capture in the same DPI context the game renders in, by switching
the calling thread to DPI-unaware for the duration of the capture. A padding
detector stays in place as a safety net for any case this does not cover.
"""

from __future__ import annotations

import contextlib
import logging
from ctypes import c_void_p, windll
from typing import Optional, Tuple

import numpy as np
import win32gui
import win32ui

logger = logging.getLogger(__name__)

#: DPI_AWARENESS_CONTEXT_UNAWARE, from windef.h.
_DPI_AWARENESS_CONTEXT_UNAWARE = c_void_p(-1)

#: SetThreadDpiAwarenessContext needs Windows 10 1607 or newer. Older systems
#: predate per-monitor DPI, so the mismatch this guards against cannot arise.
try:
    _set_thread_dpi_context = windll.user32.SetThreadDpiAwarenessContext
    _set_thread_dpi_context.restype = c_void_p
    _set_thread_dpi_context.argtypes = [c_void_p]
    _DPI_CONTEXT_AVAILABLE = True
except AttributeError:  # pragma: no cover
    _set_thread_dpi_context = None  # type: ignore[assignment]
    _DPI_CONTEXT_AVAILABLE = False


@contextlib.contextmanager
def dpi_unaware_thread():
    """Run a block with this thread reporting DPI-unaware geometry.

    Restores the previous context on the way out, including on exception, so it
    cannot leak into Qt's own DPI handling.
    """
    if not _DPI_CONTEXT_AVAILABLE:
        yield False
        return

    previous = None
    try:
        previous = _set_thread_dpi_context(_DPI_AWARENESS_CONTEXT_UNAWARE)
    except Exception:  # pragma: no cover
        logger.debug("could not switch thread DPI context", exc_info=True)
        yield False
        return

    try:
        yield bool(previous)
    finally:
        if previous:
            try:
                _set_thread_dpi_context(previous)
            except Exception:  # pragma: no cover
                logger.debug("could not restore thread DPI context", exc_info=True)


def content_bounds(frame: np.ndarray, black_level: int = 4) -> Tuple[int, int, int, int]:
    """Bounding box of non-black content as (x1, y1, x2, y2).

    Used to detect the padding described in the module docstring. Returns the
    full frame when everything is black, so callers never get an empty box.
    """
    if frame is None or frame.size == 0:
        return 0, 0, 0, 0

    height, width = frame.shape[:2]
    # Max across channels: a pixel counts as content if any channel is lit.
    lit = frame.max(axis=2) > black_level
    columns = np.where(lit.any(axis=0))[0]
    rows = np.where(lit.any(axis=1))[0]
    if len(columns) == 0 or len(rows) == 0:
        return 0, 0, width, height
    return int(columns[0]), int(rows[0]), int(columns[-1]) + 1, int(rows[-1]) + 1


def strip_black_padding(
    frame: np.ndarray, min_ratio: float = 0.02
) -> Tuple[np.ndarray, bool]:
    """Trim a pure-black right/bottom band, if there is a substantial one.

    Deliberately narrow, because cropping the wrong thing shifts every
    coordinate the bot produces. It only acts when:
      - the content starts at the very top-left corner, and
      - the discarded band is at least `min_ratio` of the frame.

    A dark in-game scene therefore cannot trigger it: its content still reaches
    the frame edges.

    Returns (frame_or_crop, was_cropped).
    """
    if frame is None or frame.size == 0:
        return frame, False

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = content_bounds(frame)

    if (x1, y1) != (0, 0):
        return frame, False
    if x2 <= 0 or y2 <= 0:
        return frame, False

    trimmed = (width - x2) / float(width), (height - y2) / float(height)
    if max(trimmed) < min_ratio:
        return frame, False

    return frame[0:y2, 0:x2].copy(), True


class GameWindow:
    """One target window, with handle management and background capture."""

    def __init__(self, title: str, print_window_flag: int = 3) -> None:
        self._title = title
        self._flag = print_window_flag
        self._hwnd: Optional[int] = None
        self._warned_padding = False

        # Cached GDI objects, reused across captures.
        #
        # Creating and destroying a DC and a bitmap for every frame measured as a
        # meaningful slice of the 48-57 ms per capture seen in live runs, and none
        # of that work changes between frames as long as the window keeps the same
        # size. The cache is keyed on (hwnd, width, height) and rebuilt whenever
        # any of those change, so a resized or re-launched window cannot be served
        # a stale bitmap.
        self._cache_key: Optional[Tuple[int, int, int]] = None
        self._hwnd_dc = None
        self._mfc_dc = None
        self._save_dc = None
        self._bitmap = None

    # ------------------------------------------------------------- properties

    @property
    def title(self) -> str:
        return self._title

    @property
    def hwnd(self) -> Optional[int]:
        """Current handle, or None. The action layer reads this."""
        return self._hwnd

    # -------------------------------------------------------- handle lifecycle

    def attach(self) -> bool:
        """Resolve the title to a handle. True when found."""
        hwnd = win32gui.FindWindow(None, self._title)
        if not hwnd:
            self._hwnd = None
            logger.debug("window not found: %s", self._title)
            return False
        self._hwnd = hwnd
        logger.info("attached to %r (HWND=%s)", self._title, hwnd)
        return True

    def is_alive(self) -> bool:
        """Whether the handle still refers to an existing window."""
        if self._hwnd is None:
            return False
        try:
            return bool(win32gui.IsWindow(self._hwnd))
        except Exception:
            return False

    def ensure_attached(self) -> bool:
        """Re-resolve the handle if it went stale. Called once per round."""
        if self.is_alive():
            return True
        return self.attach()

    def detach(self) -> None:
        self._release_gdi()
        self._hwnd = None

    def is_minimized(self) -> bool:
        """Minimized windows are not rendered, so they cannot be captured."""
        if self._hwnd is None:
            return False
        try:
            return bool(win32gui.IsIconic(self._hwnd))
        except Exception:
            return False

    # ---------------------------------------------------------------- geometry

    def client_size(self) -> Tuple[int, int]:
        """Client area size as the game itself sees it.

        Queried in a DPI-unaware context so it matches what PrintWindow will
        actually produce; see the module docstring.
        """
        if self._hwnd is None:
            return 0, 0
        try:
            with dpi_unaware_thread():
                left, top, right, bottom = win32gui.GetClientRect(self._hwnd)
            return right - left, bottom - top
        except Exception:
            logger.exception("GetClientRect failed")
            return 0, 0

    # ----------------------------------------------------------------- capture

    def capture(self) -> Optional[np.ndarray]:
        """Copy the client area without focusing or raising the window.

        Returns:
            A (h, w, 3) BGR array on success. None when the window is gone,
            minimized, or PrintWindow refuses.
        """
        if not self.ensure_attached():
            return None

        if self.is_minimized():
            logger.debug("window is minimized; nothing is rendered to capture")
            return None

        # The whole capture runs in the game's DPI context so that the rect we
        # allocate for and the surface PrintWindow fills agree.
        with dpi_unaware_thread():
            frame = self._capture_locked()

        if frame is None:
            return None

        cropped, was_cropped = strip_black_padding(frame)
        if was_cropped and not self._warned_padding:
            self._warned_padding = True
            logger.warning(
                "capture had black padding: %dx%d surface, %dx%d of content. "
                "Trimming it. This usually means a DPI-scaling mismatch between "
                "this process and the game",
                frame.shape[1],
                frame.shape[0],
                cropped.shape[1],
                cropped.shape[0],
            )
        return cropped

    def park_physical_cursor(self) -> None:
        """If physical mouse cursor is hovering inside the game client area, park it safely."""
        if not self._hwnd:
            return
        try:
            pt = win32gui.GetCursorPos()
            cl = win32gui.GetClientRect(self._hwnd)
            tl = win32gui.ClientToScreen(self._hwnd, (cl[0], cl[1]))
            br = win32gui.ClientToScreen(self._hwnd, (cl[2], cl[3]))
            if tl[0] + 30 <= pt[0] <= br[0] - 30 and tl[1] + 30 <= pt[1] <= br[1] - 30:
                win32api.SetCursorPos((tl[0] + 5, tl[1] + 5))
        except Exception:
            pass

    def _release_gdi(self) -> None:
        """Free the cached GDI objects. Safe to call repeatedly.

        Each handle is released independently: a failure on one must not leak the
        others, because GDI leaks accumulate until capture stops working at all.
        """
        if self._bitmap is not None:
            try:
                win32gui.DeleteObject(self._bitmap.GetHandle())
            except Exception:
                pass
        if self._save_dc is not None:
            try:
                self._save_dc.DeleteDC()
            except Exception:
                pass
        if self._mfc_dc is not None:
            try:
                self._mfc_dc.DeleteDC()
            except Exception:
                pass
        if self._hwnd_dc is not None and self._cache_key is not None:
            try:
                win32gui.ReleaseDC(self._cache_key[0], self._hwnd_dc)
            except Exception:
                pass
        self._bitmap = None
        self._save_dc = None
        self._mfc_dc = None
        self._hwnd_dc = None
        self._cache_key = None

    def _ensure_gdi(self, width: int, height: int) -> bool:
        """Make sure cached GDI objects match this window and size."""
        key = (int(self._hwnd or 0), width, height)
        if self._cache_key == key and self._save_dc is not None:
            return True

        # Anything stale must go before new handles are taken.
        self._release_gdi()
        try:
            self._hwnd_dc = win32gui.GetWindowDC(self._hwnd)
            self._mfc_dc = win32ui.CreateDCFromHandle(self._hwnd_dc)
            self._save_dc = self._mfc_dc.CreateCompatibleDC()
            self._bitmap = win32ui.CreateBitmap()
            self._bitmap.CreateCompatibleBitmap(self._mfc_dc, width, height)
            self._save_dc.SelectObject(self._bitmap)
            self._cache_key = key
            logger.debug("GDI capture objects created for %dx%d", width, height)
            return True
        except Exception:
            logger.exception("could not create GDI capture objects")
            self._release_gdi()
            return False

    def _capture_locked(self) -> Optional[np.ndarray]:
        """PrintWindow proper. Assumes the caller set the DPI context."""
        try:
            left, top, right, bottom = win32gui.GetClientRect(self._hwnd)
        except Exception:
            logger.exception("GetClientRect failed")
            return None

        width, height = right - left, bottom - top
        if width <= 0 or height <= 0:
            logger.debug("client area is %dx%d; window is not renderable", width, height)
            return None

        if not self._ensure_gdi(width, height):
            return None

        try:
            # 1 means success. Exclusive fullscreen or an uncooperative driver
            # returns 0.
            result = windll.user32.PrintWindow(
                self._hwnd, self._save_dc.GetSafeHdc(), self._flag
            )
            if result != 1:
                logger.debug("PrintWindow returned %s; discarding this frame", result)
                return None

            info = self._bitmap.GetInfo()
            bits = self._bitmap.GetBitmapBits(True)

            # GetBitmapBits yields 32-bit BGRA; drop alpha for BGR.
            # frombuffer is a read-only view over a buffer the next capture will
            # overwrite, so it must be copied before returning.
            frame = np.frombuffer(bits, dtype=np.uint8).reshape(
                info["bmHeight"], info["bmWidth"], 4
            )
            return frame[:, :, :3].copy()

        except Exception:
            # A failure may have left the cached objects unusable; drop them so
            # the next call rebuilds rather than reusing something broken.
            logger.exception("capture raised")
            self._release_gdi()
            return None

    def __del__(self) -> None:  # pragma: no cover
        # Best-effort net for the case where detach() was never called.
        try:
            self._release_gdi()
        except Exception:
            pass
