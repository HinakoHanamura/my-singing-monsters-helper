"""Unit tests for the ActionAgent drag / swipe functionality."""

from __future__ import annotations

from typing import List, Optional, Tuple
import win32con
import pytest

from config import ClickConfig
from core.action_agent import ActionAgent, WindowHandleProvider


class DummyWindow(WindowHandleProvider):
    def __init__(self, attached: bool = True, hwnd: Optional[int] = 12345) -> None:
        self._attached = attached
        self._hwnd = hwnd

    @property
    def hwnd(self) -> Optional[int]:
        return self._hwnd

    def ensure_attached(self) -> bool:
        return self._attached

    def client_size(self) -> Tuple[int, int]:
        return (1024, 768)


def test_drag_not_attached_returns_false() -> None:
    """If window is unavailable, drag immediately aborts and returns False."""
    window = DummyWindow(attached=False)
    agent = ActionAgent(window=window)
    assert agent.drag(100, 200, 100, 400) is False


def test_drag_message_sequence() -> None:
    """Verify drag posts start move, button down, intermediate moves, and button up."""
    window = DummyWindow(attached=True, hwnd=999)
    # Zero out delays for fast test
    cfg = ClickConfig(post_click_delay=(0.0, 0.0), jitter_radius=0)
    agent = ActionAgent(window=window, config=cfg)

    sent_messages: List[Tuple[int, int, int, int]] = []

    def fake_send(hwnd: int, msg: int, wparam: int, lparam: int) -> None:
        sent_messages.append((hwnd, msg, wparam, lparam))

    agent._send = fake_send  # type: ignore

    success = agent.drag(
        start_x=200,
        start_y=500,
        end_x=200,
        end_y=300,
        duration=0.01,
        steps=5,
    )
    assert success is True

    # 1st: WM_MOUSEMOVE to start
    assert sent_messages[0][1] == win32con.WM_MOUSEMOVE
    assert sent_messages[0][2] == 0

    # 2nd: WM_LBUTTONDOWN at start
    assert sent_messages[1][1] == win32con.WM_LBUTTONDOWN
    assert sent_messages[1][2] == win32con.MK_LBUTTON

    # Next steps: intermediate WM_MOUSEMOVE with MK_LBUTTON
    for i in range(2, len(sent_messages) - 1):
        assert sent_messages[i][1] == win32con.WM_MOUSEMOVE
        assert sent_messages[i][2] == win32con.MK_LBUTTON

    # Final message: WM_LBUTTONUP
    assert sent_messages[-1][1] == win32con.WM_LBUTTONUP
    assert sent_messages[-1][2] == 0


def test_drag_settle_pause_before_release(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that drag pauses at the end position before sending WM_LBUTTONUP.
    This guarantees zero release velocity, eliminating kinetic/inertial fling."""
    window = DummyWindow(attached=True, hwnd=999)
    cfg = ClickConfig(post_click_delay=(0.0, 0.0), jitter_radius=0)
    agent = ActionAgent(window=window, config=cfg)

    events: List[str] = []

    def fake_send(hwnd: int, msg: int, wparam: int, lparam: int) -> None:
        if msg == win32con.WM_LBUTTONUP:
            events.append("up")

    agent._send = fake_send  # type: ignore

    import time
    original_sleep = time.sleep

    def fake_sleep(duration: float) -> None:
        if duration >= 0.15:
            events.append("settle_sleep")
        else:
            original_sleep(0.001)

    monkeypatch.setattr(time, "sleep", fake_sleep)

    success = agent.drag(
        start_x=200,
        start_y=480,
        end_x=200,
        end_y=320,
        duration=0.01,
        steps=3,
    )
    assert success is True
    assert "settle_sleep" in events
    assert events.index("settle_sleep") < events.index("up"), "Settle sleep must happen before WM_LBUTTONUP!"
