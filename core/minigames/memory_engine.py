"""QThread wrapper that lets the UI drive the memory minigame.

Why a separate engine from BotEngine
------------------------------------
``BotEngine`` runs the coin patrol, and the two loops have genuinely different
shapes: the patrol watches a scene that changes on its own and never finishes,
while a minigame run is a closed sequence of puzzles that ends when the boards
stop appearing. Sharing one thread class would mean one loop carrying branches for
the other.

What is shared instead is the *interface to the UI*. This class emits exactly the
signals ``BotEngine`` does -- ``log``, ``state_changed``, ``stats_changed`` -- so
the window can hold either kind of worker in the same attribute, connect the same
slots, and stop it the same way. That keeps the UI free of "which engine is this"
branching except where the wording genuinely differs.

This module is also the UI's only door to the runner
----------------------------------------------------
``RunnerParams`` is re-exported here on purpose. The window needs to express one
choice -- which flip order to use -- and routing that through the engine keeps the
UI from importing the control loop itself. There is a single way to set it, the
``params`` argument, rather than a convenience flag duplicating the same field.

Threading rules this obeys
--------------------------
The whole point of a worker thread here is that the UI stays responsive while
capture, matching and clicking happen, so nothing in this class touches widgets.
Communication is one-way through signals, which Qt delivers on the receiving
thread.

Stopping is cooperative. ``requestInterruption`` sets a flag that the runner polls
between steps, so a stop takes effect at a turn boundary rather than in the middle
of one. Interrupting mid-turn could leave a card face up with the solver believing
otherwise, and since the next run rebuilds its state from a fresh board that would
cost nothing -- but a half-delivered click is not something to invite.
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import QThread, Signal

from config import DEFAULT_CONFIG, AppConfig
from core.action_agent import ActionAgent
from core.game_window import GameWindow
from core.minigames.memory_runner import (
    CODE_NO_BOARD,
    CODE_RUN_COMPLETE,
    CODE_UNRECOVERABLE,
    MemoryRunner,
    RunnerParams,
)

__all__ = ["MemoryEngine", "RunnerParams"]

logger = logging.getLogger(__name__)

#: Log levels, matching the ones the UI colours.
LEVEL_INFO = "INFO"
LEVEL_SUCCESS = "SUCCESS"
LEVEL_WARN = "WARN"
LEVEL_ERROR = "ERROR"


class MemoryEngine(QThread):
    """Runs one minigame session on a worker thread."""

    # (level, text) -> forwarded to the UI log
    log = Signal(str, str)

    # State change, carries a display string
    state_changed = Signal(str)

    # (levels cleared, turns played)
    stats_changed = Signal(int, int)

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        params: RunnerParams = RunnerParams(),
    ) -> None:
        super().__init__()
        self._cfg = config or DEFAULT_CONFIG
        self._params = params
        self._runner: Optional[MemoryRunner] = None

    # ------------------------------------------------------------ public API

    def stop(self) -> None:
        """Request a stop. Called from the UI thread; returns immediately."""
        if self.isRunning():
            self._emit(LEVEL_WARN, "收到停止指令，将在本轮翻牌结束后收尾…")
            self.requestInterruption()

    # ---------------------------------------------------------- thread body

    def run(self) -> None:  # noqa: D102  (QThread entry point)
        """Worker entry point. Wrapped so the thread never dies mutely."""
        self.state_changed.emit("准备中")
        window = GameWindow(
            title=self._cfg.window.title,
            print_window_flag=self._cfg.window.print_window_flag,
        )
        try:
            if not window.ensure_attached():
                self._emit(
                    LEVEL_ERROR,
                    f"找不到游戏窗口：{self._cfg.window.title}。请先启动游戏。",
                )
                self.state_changed.emit("未找到窗口")
                return

            width, height = window.client_size()
            self._emit(LEVEL_INFO, f"已附着窗口，客户区 {width}x{height}")
            self._emit(
                LEVEL_INFO,
                "请确认已手动进入小游戏某一关的开局界面，所有卡牌背面朝下。",
            )
            self._emit(LEVEL_INFO, f"翻牌策略：{self._order_label()}")

            action = ActionAgent(window, self._cfg.click)
            self._runner = MemoryRunner(
                window=window,
                action=action,
                params=self._params,
                log=lambda message: self._emit(LEVEL_INFO, message),
            )

            self.state_changed.emit("运行中")
            result = self._runner.run(
                should_continue=lambda: not self.isInterruptionRequested()
            )
            self._report(result)
        except Exception as error:  # pragma: no cover - defensive
            logger.exception("memory minigame worker crashed")
            self._emit(LEVEL_ERROR, f"小游戏线程异常退出：{error!r}")
            self.state_changed.emit("异常停止")
        finally:
            window.detach()
            self._publish_stats()

    # ------------------------------------------------------------- internals

    def _order_label(self) -> str:
        """How the chosen flip order reads in the log."""
        if self._params.scan_first:
            return "加速配对（先按序翻开全部卡牌，再逐对收牌）"
        return "边翻边收（一旦认出成对就立刻收掉）"

    def _report(self, result) -> None:
        """Turn the runner's verdict into user-facing wording.

        The codes are stable ASCII and the messages are Chinese; only the extra
        advice is added here, because what the user should *do* about a stop is a
        UI concern rather than the loop's.
        """
        self._publish_stats()

        if self.isInterruptionRequested():
            self._emit(LEVEL_WARN, "已按请求停止。")
            self.state_changed.emit("已停止")
            return

        if result.code == CODE_RUN_COMPLETE:
            self._emit(LEVEL_SUCCESS, result.message)
            self.state_changed.emit("本轮完成")
            return

        if result.ok:
            self._emit(LEVEL_INFO, result.message)
            self.state_changed.emit("已停止")
            return

        self._emit(LEVEL_ERROR, f"{result.code} — {result.message}")
        if result.code == CODE_UNRECOVERABLE:
            self._emit(
                LEVEL_WARN,
                "已经按盘面重建过记忆仍然对不上，通常是卡面识别在这一关不可靠。"
                "请保留日志，并在某一关刚开始、所有卡牌都是背面时重试。",
            )
        elif result.code == CODE_NO_BOARD:
            self._emit(
                LEVEL_WARN,
                "一关都没打完就找不到盘面，说明启动时并不在小游戏的开局界面。",
            )
        self.state_changed.emit("已停止")

    def _publish_stats(self) -> None:
        if self._runner is None:
            return
        self.stats_changed.emit(self._runner.levels_cleared, self._runner.turns_played)

    def _emit(self, level: str, message: str) -> None:
        self.log.emit(level, message)
