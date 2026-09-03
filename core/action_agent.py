"""Action layer: turn "click here" into Windows messages.

Clicks are delivered as window messages rather than by driving the real mouse:
  - the physical cursor never moves and focus is never stolen, so the game can
    stay in the background,
  - you can keep using the machine while the bot runs.

Humanisation, which is the main reason this layer exists at all:
  - positional jitter, so successive clicks are not pixel-identical,
  - timing jitter on the hold and on the pause afterwards, so there is no fixed
    cadence,
  - an optional WM_MOUSEMOVE first, matching the order a real mouse produces.

Jitter radii are calibrated in reference-resolution pixels and scaled to the live
window, otherwise the same 5px would be invisible on a 4K window and could land
outside a small one.

DPI context, and why it matters here too
----------------------------------------
Window geometry and mouse-message coordinates are interpreted relative to a DPI
context. The capture layer already switches to the game's own (DPI-unaware)
context so that the frame it gets back matches the rectangle it asked for; see
core.game_window for the full account of the live failure that motivated it.

Message delivery is wrapped the same way, for symmetry and for a concrete
reason: a standalone probe script (no Qt, therefore DPI-unaware) placed clicks
correctly, while the Qt application (QApplication makes the process DPI-aware)
did not. Sending from the same context the game lives in removes that difference
as a variable, and costs nothing when the contexts already agree.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Optional, Protocol, Tuple

import win32api
import win32con
import win32gui

from config import ClickConfig
from core.game_window import dpi_unaware_thread
from core.geometry import scale_length

logger = logging.getLogger(__name__)


class WindowHandleProvider(Protocol):
    """All this layer needs: a handle and a client size.

    Typed as a Protocol rather than depending on GameWindow directly, so tests
    can pass a stand-in without a running game.
    """

    @property
    def hwnd(self) -> Optional[int]: ...

    def ensure_attached(self) -> bool: ...

    def client_size(self) -> Tuple[int, int]: ...


class ActionAgent:
    """Background click executor."""

    def __init__(
        self,
        window: WindowHandleProvider,
        config: Optional[ClickConfig] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        self._window = window
        self._cfg = config or ClickConfig()
        self._rng = rng or random.Random()
        # Window scale relative to the reference resolution; synced per round.
        self._scale = 1.0

    # -------------------------------------------------------------- public API

    def set_scale(self, scale: float) -> None:
        """Sync the window scale so jitter feels the same at any resolution."""
        self._scale = scale if scale > 0 else 1.0

    def click(self, x: int, y: int) -> bool:
        """Click once near client-area coordinates (x, y).

        Args:
            x, y: target centre from the perception layer, in client pixels.

        Returns:
            True when the messages were delivered; False if the window went away
            or delivery raised.
        """
        if not self._window.ensure_attached():
            logger.warning("target window unavailable; click cancelled")
            return False

        hwnd = self._window.hwnd
        if hwnd is None:
            return False

        target_x, target_y = self._humanize_point(x, y)
        lparam = win32api.MAKELONG(target_x, target_y)

        try:
            # Deliver in the game's DPI context, matching the capture path.
            with dpi_unaware_thread():
                if self._cfg.move_before_click:
                    self._send(hwnd, win32con.WM_MOUSEMOVE, 0, lparam)
                    self._sleep_in_range(self._cfg.pre_press_delay)

                self._send(
                    hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lparam
                )
                if self._cfg.press_duration[1] > 0:
                    self._sleep_in_range(self._cfg.press_duration)
                self._send(hwnd, win32con.WM_LBUTTONUP, 0, lparam)

            logger.debug(
                "clicked (%s, %s) for target (%s, %s)", target_x, target_y, x, y
            )

        except Exception:
            logger.exception("failed to deliver click messages")
            return False

        # Random pause so consecutive actions are not back to back.
        self._sleep_in_range(self._cfg.post_click_delay)
        return True

    def drag(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration: float = 0.25,
        steps: int = 10,
    ) -> bool:
        """Drag smoothly from (start_x, start_y) to (end_x, end_y).

        Args:
            start_x, start_y: drag start coordinates in client pixels.
            end_x, end_y: drag end coordinates in client pixels.
            duration: total duration of the drag in seconds.
            steps: number of intermediate mouse move messages.

        Returns:
            True if delivered successfully, False otherwise.
        """
        if not self._window.ensure_attached():
            logger.warning("target window unavailable; drag cancelled")
            return False

        hwnd = self._window.hwnd
        if hwnd is None:
            return False

        sx, sy = self._humanize_point(start_x, start_y)
        ex, ey = self._humanize_point(end_x, end_y)

        steps = max(2, steps)
        step_dt = duration / steps if duration > 0 else 0.01

        try:
            with dpi_unaware_thread():
                # Move to start position
                start_lparam = win32api.MAKELONG(sx, sy)
                self._send(hwnd, win32con.WM_MOUSEMOVE, 0, start_lparam)
                self._sleep_in_range((0.02, 0.04))

                # Mouse down at start position
                self._send(
                    hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, start_lparam
                )
                self._sleep_in_range((0.02, 0.04))

                # Interpolate moves
                for i in range(1, steps + 1):
                    alpha = i / steps
                    cx = int(round(sx + (ex - sx) * alpha))
                    cy = int(round(sy + (ey - sy) * alpha))
                    move_lparam = win32api.MAKELONG(cx, cy)
                    self._send(
                        hwnd, win32con.WM_MOUSEMOVE, win32con.MK_LBUTTON, move_lparam
                    )
                    time.sleep(step_dt)

                # Settle at destination before release to bleed off kinetic velocity (eliminates inertial fling)
                time.sleep(0.18)

                # Mouse up at end position
                end_lparam = win32api.MAKELONG(ex, ey)
                self._send(hwnd, win32con.WM_LBUTTONUP, 0, end_lparam)

            logger.debug(
                "dragged from (%s, %s) to (%s, %s)", sx, sy, ex, ey
            )
        except Exception:
            logger.exception("failed to deliver drag messages")
            return False

        self._sleep_in_range(self._cfg.post_click_delay)
        return True

    def park_cursor(self) -> bool:
        """Move cursor to a safe neutral coordinate to clear hover glow effects."""
        if not self._window.ensure_attached():
            return False
        hwnd = self._window.hwnd
        if hwnd is None:
            return False
        try:
            with dpi_unaware_thread():
                lparam = win32api.MAKELONG(10, 10)
                self._send(hwnd, win32con.WM_MOUSEMOVE, 0, lparam)
            return True
        except Exception:
            return False

    # ---------------------------------------------------------- humanisation

    def _humanize_point(self, x: int, y: int) -> Tuple[int, int]:
        """Offset the target randomly and clamp it inside the client area."""
        radius = scale_length(self._cfg.jitter_radius, self._scale, minimum=0)
        if radius > 0:
            x += self._rng.randint(-radius, radius)
            y += self._rng.randint(-radius, radius)

        # Jitter can push a point past the edge; clamp so nothing is delivered
        # outside the client area.
        width, height = self._window.client_size()
        if width > 0 and height > 0:
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
        return int(x), int(y)

    def _sleep_in_range(self, bounds: Tuple[float, float]) -> None:
        low, high = bounds
        if high <= 0:
            return
        time.sleep(self._rng.uniform(low, max(low, high)))

    # ------------------------------------------------------------- delivery

    def _send(self, hwnd: int, msg: int, wparam: int, lparam: int) -> None:
        """Single delivery point, so Send/Post semantics switch in one place.

        SendMessage blocks until the game has processed the message, so a stalled
        game stalls this worker thread (the UI is unaffected, being a different
        thread). Set ClickConfig.use_post_message when that becomes a problem.
        """
        if self._cfg.use_post_message:
            win32gui.PostMessage(hwnd, msg, wparam, lparam)
        else:
            win32gui.SendMessage(hwnd, msg, wparam, lparam)
