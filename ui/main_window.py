"""UI layer: one deliberately plain main window.

Its remit is exactly three things:
  1. turn the user's clicks into start() / stop() calls on a worker thread;
  2. render the Signals that thread sends into text on the screen;
  3. own the thread's lifetime, above all shutting it down cleanly on close.

Nothing here captures, detects or clicks. All of that lives in ``core``.

Two jobs, one window
--------------------
Coin collecting is driven by `BotEngine`, the memory minigame by `MemoryEngine`.
Their loops are fundamentally different shapes -- an endless patrol versus a
closed puzzle with a terminal state -- so they were never merged into one class.
But they **emit exactly the same three signals** (`log` / `state_changed` /
`stats_changed`), so one field can hold either, one set of slots can receive from
either, and one button can stop either.

The only thing that has to differ is the wording of the statistics label:
"ticks/clicks" and "levels/turns" are not the same quantity. So `self._mode`
remembers which job is running, and the code branches only where text is drawn.

Why the "faster pairing" switch is left to the user
---------------------------------------------------
It selects the flip order. The default collects as it goes: the moment a pair is
recognised it is taken. Ticked, every card is turned over in order first, the
whole board is read, and only then are pairs collected. The worst-case mismatch
counts differ by exactly one (n-1 versus n) and both sit far inside the
ceil(1.5n) the game allows, so **which one is faster cannot be settled by
reasoning**: the dominant cost of a mismatch is the flip-back animation the game
imposes, and that is independent of the order cards are turned in.

So this window does not choose for the user. It offers the switch and has the
runner report how long each level took -- the two orders are compared on the real
game instead of argued about in a document.

The switch is read only at the instant a run starts. Changing it mid-run would
mean nothing, so once something is running it is disabled alongside the two start
buttons.
"""

from __future__ import annotations

import html
from datetime import datetime
from typing import Optional, Union

from PySide6.QtCore import Qt, QThread, Slot
from PySide6.QtGui import QCloseEvent, QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from config import DEFAULT_CONFIG, AppConfig
from core.bot_engine import BotEngine
from core.minigames.memory_engine import MemoryEngine, RunnerParams

# Log level -> colour
LEVEL_COLORS = {
    "INFO": "#8b95a5",
    "SUCCESS": "#4ade80",
    "WARN": "#fbbf24",
    "ERROR": "#f87171",
}

# Cap on the lines kept in the log view, so a long session cannot eat memory.
MAX_LOG_BLOCKS = 2000

# Which job is running. Used only to decide how the statistics label reads.
MODE_COIN = "coin"
MODE_MEMORY = "memory"

#: Upper bound on waiting for either thread to wind down. A coin tick takes a
#: few seconds; the minigame's stop is cooperative and lets the current turn
#: play out, so it is given more room.
SHUTDOWN_WAIT_MS = 8000

STYLE_SHEET = """
QWidget#Root {
    background-color: #14161a;
}
QLabel#TitleLabel {
    color: #f2f4f8;
    font-size: 17px;
    font-weight: 600;
}
QLabel#SubtitleLabel {
    color: #6b7280;
    font-size: 11px;
}
QLabel#StatusLabel {
    color: #9aa4b2;
    font-size: 12px;
    padding: 5px 12px;
    border: 1px solid #2a2f38;
    border-radius: 11px;
    background-color: #1b1f25;
}
QLabel#StatsLabel {
    color: #5f6875;
    font-size: 11px;
}
QFrame#Divider {
    background-color: #23272e;
    max-height: 1px;
    border: none;
}
QPushButton {
    font-size: 13px;
    font-weight: 600;
    padding: 10px 22px;
    border-radius: 6px;
    border: none;
}
QPushButton#StartButton {
    background-color: #2f6df6;
    color: #ffffff;
}
QPushButton#StartButton:hover {
    background-color: #4680ff;
}
QPushButton#StartButton:pressed {
    background-color: #2559d0;
}
QPushButton#MinigameButton {
    background-color: #7c5cf0;
    color: #ffffff;
}
QPushButton#MinigameButton:hover {
    background-color: #8f73ff;
}
QPushButton#MinigameButton:pressed {
    background-color: #6748d6;
}
QPushButton#StopButton {
    background-color: #262b33;
    color: #d6dae1;
}
QPushButton#StopButton:hover {
    background-color: #323841;
}
QPushButton#StopButton:pressed {
    background-color: #1e2229;
}
QPushButton:disabled {
    background-color: #1c1f25;
    color: #4a5058;
}
QCheckBox#ScanFirstBox {
    color: #9aa4b2;
    font-size: 11px;
    spacing: 6px;
}
QCheckBox#ScanFirstBox:disabled {
    color: #4a5058;
}
QCheckBox#ScanFirstBox::indicator {
    width: 13px;
    height: 13px;
    border-radius: 3px;
    border: 1px solid #3a414b;
    background-color: #1b1f25;
}
QCheckBox#ScanFirstBox::indicator:checked {
    border: 1px solid #7c5cf0;
    background-color: #7c5cf0;
}
QTextBrowser#LogView {
    background-color: #0f1115;
    color: #c8ced8;
    border: 1px solid #23272e;
    border-radius: 8px;
    padding: 10px;
    selection-background-color: #2f6df6;
}
QScrollBar:vertical {
    background: transparent;
    width: 9px;
    margin: 4px 2px 4px 0px;
}
QScrollBar::handle:vertical {
    background: #363c45;
    border-radius: 4px;
    min-height: 28px;
}
QScrollBar::handle:vertical:hover {
    background: #454c57;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
}
"""


class MainWindow(QMainWindow):
    """Main window of the helper."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        super().__init__()
        self._cfg = config or DEFAULT_CONFIG
        self._worker: Optional[Union[BotEngine, MemoryEngine]] = None
        self._mode: str = MODE_COIN

        self.setWindowTitle("MSM Helper")
        self.setMinimumSize(620, 480)
        self.resize(740, 590)

        self._build_ui()
        self.setStyleSheet(STYLE_SHEET)

        self._append_log("INFO", "就绪。请先启动游戏。")
        self._append_log("INFO", f"当前目标窗口：{self._cfg.window.title}")
        self._append_log(
            "INFO", "「收集金币」在岛屿界面使用；「记忆小游戏」需先手动进入某一关的开局界面。"
        )

    # --------------------------------------------------------- building the UI

    def _build_ui(self) -> None:
        root = QWidget(objectName="Root")
        self.setCentralWidget(root)

        outer = QVBoxLayout(root)
        outer.setContentsMargins(22, 20, 22, 18)
        outer.setSpacing(14)

        # ---- top: title + status
        header = QHBoxLayout()
        header.setSpacing(12)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title_box.addWidget(QLabel("My Singing Monsters 挂机助手", objectName="TitleLabel"))
        title_box.addWidget(
            QLabel("后台截图 + 后台点击，不接管你的鼠标", objectName="SubtitleLabel")
        )
        header.addLayout(title_box)

        header.addStretch(1)

        self._status_label = QLabel("状态：空闲", objectName="StatusLabel")
        self._status_label.setAccessibleName("运行状态")
        header.addWidget(self._status_label, 0, Qt.AlignmentFlag.AlignVCenter)

        outer.addLayout(header)
        outer.addWidget(QFrame(objectName="Divider"))

        # ---- middle: control buttons
        controls = QHBoxLayout()
        controls.setSpacing(10)

        self._start_button = QPushButton("收集金币", objectName="StartButton")
        self._start_button.setAccessibleName("开始收集金币")
        self._start_button.setToolTip("在岛屿界面循环收集金币（F9）")
        self._start_button.setShortcut("F9")
        self._start_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._start_button.clicked.connect(self._on_start_coin_clicked)

        self._minigame_button = QPushButton("记忆小游戏", objectName="MinigameButton")
        self._minigame_button.setAccessibleName("开始记忆小游戏")
        self._minigame_button.setToolTip(
            "自动完成记忆小游戏，直到盘面不再出现（F11）。"
            "启动前请手动进入某一关的开局界面，所有卡牌背面朝下。"
        )
        self._minigame_button.setShortcut("F11")
        self._minigame_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._minigame_button.clicked.connect(self._on_start_minigame_clicked)

        # Flip-order switch. Read only at the instant the minigame is started.
        self._scan_first_box = QCheckBox("加速配对", objectName="ScanFirstBox")
        self._scan_first_box.setAccessibleName("加速配对：先翻开全部卡牌再逐对收牌")
        self._scan_first_box.setToolTip(
            "改变记忆小游戏的翻牌顺序：先按序把所有卡牌翻开看清，再逐对收牌。\n"
            "最坏失配次数 n 次，比默认的「边翻边收」多一次，仍远在游戏给的机会数之内。\n"
            "哪种更快取决于实机节奏——每关结束会报出耗时，可以自己比。\n"
            "不勾选则一旦认出成对就立刻收掉。"
        )
        self._scan_first_box.setCursor(Qt.CursorShape.PointingHandCursor)

        self._stop_button = QPushButton("停止", objectName="StopButton")
        self._stop_button.setAccessibleName("停止当前任务")
        self._stop_button.setToolTip("停止当前任务（F10）")
        self._stop_button.setShortcut("F10")
        self._stop_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._stop_button.setEnabled(False)
        self._stop_button.clicked.connect(self._on_stop_clicked)

        controls.addWidget(self._start_button)
        controls.addWidget(self._minigame_button)
        controls.addWidget(self._scan_first_box)
        controls.addWidget(self._stop_button)
        controls.addStretch(1)

        self._stats_label = QLabel("轮次 0 ｜ 点击 0", objectName="StatsLabel")
        self._stats_label.setAccessibleName("运行统计")
        controls.addWidget(self._stats_label, 0, Qt.AlignmentFlag.AlignVCenter)

        outer.addLayout(controls)

        # ---- bottom: log
        log_caption = QLabel("运行日志", objectName="SubtitleLabel")
        outer.addWidget(log_caption)

        self._log_view = QTextBrowser(objectName="LogView")
        self._log_view.setReadOnly(True)
        self._log_view.setAccessibleName("运行日志")
        self._log_view.setOpenExternalLinks(False)
        self._log_view.document().setMaximumBlockCount(MAX_LOG_BLOCKS)
        self._log_view.setFont(QFont("Consolas", 9))
        outer.addWidget(self._log_view, 1)

    # --------------------------------------------------------- button handlers

    @Slot()
    def _on_start_coin_clicked(self) -> None:
        self._start(MODE_COIN)

    @Slot()
    def _on_start_minigame_clicked(self) -> None:
        self._start(MODE_MEMORY)

    def _start(self, mode: str) -> None:
        """Start a worker thread. Both jobs share this path; only the class differs."""
        if self._worker is not None and self._worker.isRunning():
            return

        # A fresh instance every time: reusing a stopped QThread is unwise.
        if mode == MODE_MEMORY:
            worker: Union[BotEngine, MemoryEngine] = MemoryEngine(
                config=self._cfg,
                params=RunnerParams(scan_first=self._scan_first_box.isChecked()),
            )
        else:
            worker = BotEngine(config=self._cfg)

        self._mode = mode
        self._worker = worker
        worker.log.connect(self._append_log)
        worker.state_changed.connect(self._on_state_changed)
        worker.stats_changed.connect(self._on_stats_changed)
        worker.finished.connect(self._on_worker_finished)

        self._on_stats_changed(0, 0)
        self._set_running(True)
        worker.start()

    @Slot()
    def _on_stop_clicked(self) -> None:
        if self._worker is None or not self._worker.isRunning():
            return
        # Only request the stop, so the window stays responsive straight away;
        # the actual winding down happens on the worker thread.
        self._stop_button.setEnabled(False)
        self._worker.stop()

    def _set_running(self, running: bool) -> None:
        """Both start buttons have to be disabled together.

        Otherwise the minigame could be launched while the coin loop is running.
        The strategy switch is disabled with them: it is read only at the instant
        a run starts, so a change made mid-run would have no effect, and leaving
        it clickable would be a promise the code does not keep.
        """
        self._start_button.setEnabled(not running)
        self._minigame_button.setEnabled(not running)
        self._scan_first_box.setEnabled(not running)
        self._stop_button.setEnabled(running)

    # ---------------------------------------------------- worker signals -> UI

    @Slot(str, str)
    def _append_log(self, level: str, message: str) -> None:
        """Render one log line into the log view.

        This arrives as a cross-thread signal, so Qt marshals the call onto the
        main thread by itself.
        """
        color = LEVEL_COLORS.get(level, LEVEL_COLORS["INFO"])
        timestamp = datetime.now().strftime("%H:%M:%S")
        safe_text = html.escape(message)

        self._log_view.append(
            f'<span style="color:#4a5058;">[{timestamp}]</span> '
            f'<span style="color:{color};">{safe_text}</span>'
        )
        # Always scroll to the bottom, so the newest output stays in view.
        scrollbar = self._log_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    @Slot(str)
    def _on_state_changed(self, state: str) -> None:
        self._status_label.setText(f"状态：{state}")

    @Slot(int, int)
    def _on_stats_changed(self, first: int, second: int) -> None:
        """One signal, two meanings -- hence the branch on mode."""
        if self._mode == MODE_MEMORY:
            self._stats_label.setText(f"已完成 {first} 关 ｜ 翻牌 {second} 轮")
        else:
            self._stats_label.setText(f"轮次 {first} ｜ 点击 {second}")

    @Slot()
    def _on_worker_finished(self) -> None:
        """Re-enable the buttons only once the thread has actually exited.

        Otherwise a second run could be started while the first is still winding
        down.
        """
        self._set_running(False)

    # ----------------------------------------------------------------- closing

    def closeEvent(self, event: QCloseEvent) -> None:
        """The worker has to be shut down before the window closes.

        Destroying a QThread that is still running produces a Qt warning.
        """
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.stop()
            # The minigame's stop lets the current turn play out, so its wait
            # window is wider than the coin loop needs.
            if not worker.wait(SHUTDOWN_WAIT_MS):
                worker.terminate()
                worker.wait(500)
        event.accept()
