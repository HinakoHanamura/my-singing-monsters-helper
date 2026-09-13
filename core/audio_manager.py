"""Game process audio management layer via Windows Core Audio (WASAPI).

Controls the volume / mute state of the My Singing Monsters process specifically,
leaving all other application audio and master system volume untouched.
Features defensive exception handling for headless, disconnected, or hotplugged
audio endpoints.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


class GameAudioManager:
    """Manages mute and volume state for the target game process."""

    def __init__(
        self,
        target_process_name: str = "MySingingMonsters.exe",
        window_title: Optional[str] = "My Singing Monsters",
    ) -> None:
        self._target_process_name = target_process_name.lower()
        self._window_title = window_title
        self._muted_by_helper: bool = False

    @property
    def muted_by_helper(self) -> bool:
        """Whether the helper is currently responsible for the active mute state."""
        return self._muted_by_helper

    def _get_target_pid(self, hwnd: Optional[int] = None) -> Optional[int]:
        if not hwnd and self._window_title:
            try:
                import win32gui

                hwnd = win32gui.FindWindow(None, self._window_title)
            except Exception:
                hwnd = None
        if not hwnd:
            return None
        try:
            import win32process

            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            return pid
        except Exception:
            logger.debug("Failed to resolve PID from HWND %s", hwnd, exc_info=True)
            return None

    def _get_sessions(self, hwnd: Optional[int] = None) -> List[Any]:
        """Enumerate active audio sessions matching the target game process or PID."""
        target_pid = self._get_target_pid(hwnd)
        matching_sessions: List[Any] = []

        try:
            try:
                import pythoncom

                pythoncom.CoInitialize()
            except Exception:
                pass
            from pycaw.pycaw import AudioUtilities

            sessions = AudioUtilities.GetAllSessions()
        except Exception as e:
            logger.debug("Audio session enumeration unavailable: %s", e)
            return []

        for s in sessions:
            try:
                matched = False
                if target_pid and getattr(s, "ProcessId", None) == target_pid:
                    matched = True
                elif s.Process and getattr(s.Process, "name", None):
                    pname = s.Process.name()
                    if pname and pname.lower() == self._target_process_name:
                        matched = True

                if matched:
                    matching_sessions.append(s)
            except Exception:
                continue

        return matching_sessions

    def is_muted(self, hwnd: Optional[int] = None) -> bool:
        """Query if the target game process is currently muted."""
        sessions = self._get_sessions(hwnd)
        if not sessions:
            return False

        for s in sessions:
            try:
                vol = getattr(s, "SimpleAudioVolume", None)
                if vol and vol.GetMute():
                    return True
            except Exception:
                continue
        return False

    def set_mute(self, mute: bool, hwnd: Optional[int] = None) -> bool:
        """Set the mute state for all audio sessions of the game process."""
        sessions = self._get_sessions(hwnd)
        if not sessions:
            logger.debug("No active audio session found for target process.")
            return False

        success = False
        for s in sessions:
            try:
                vol = getattr(s, "SimpleAudioVolume", None)
                if vol:
                    vol.SetMute(1 if mute else 0, None)
                    success = True
            except Exception as e:
                logger.warning("Failed to set audio session mute: %s", e)

        return success

    def on_task_start(self, enabled: bool, hwnd: Optional[int] = None) -> bool:
        """Hook called when a bot task starts.

        If enabled and the game is not already muted, mute the game and take ownership.
        """
        if not enabled:
            return False

        # If already muted externally, don't claim ownership
        if self.is_muted(hwnd):
            logger.info("Game is already muted externally; keeping current state.")
            return False

        if self.set_mute(True, hwnd):
            self._muted_by_helper = True
            logger.info("Game audio muted by helper.")
            return True

        return False

    def on_task_finish(self, hwnd: Optional[int] = None) -> bool:
        """Hook called when a bot task stops or finishes.

        Restores audio if it was muted by the helper.
        """
        if self._muted_by_helper:
            self.set_mute(False, hwnd)
            self._muted_by_helper = False
            logger.info("Game audio restored by helper.")
            return True
        return False

    def restore_sound(self, hwnd: Optional[int] = None) -> bool:
        """Restore sound unconditionally if muted by helper (e.g. during app shutdown)."""
        return self.on_task_finish(hwnd)
