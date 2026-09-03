"""PySide6 control window with tabbed layout, blacklist table, and testing suite.

Architecture & Concurrency
--------------------------
The UI runs strictly on the main thread; long-running bot operations (coin
loop, multi-island tour, island tracking, or memory minigame) run inside
worker QThreads. Communication is purely via Qt signals and slots.

Tabbed Layout
-------------
- Tab 1: Multi-island tour, Memory minigame and toggle, dynamic Blacklist table.
- Tab 2: Single-island collection tests and Island tracking test.
- Tab 3: Dedicated spacious log viewer with timestamp formatting.
"""

from __future__ import annotations

import html
from datetime import datetime
from dataclasses import replace
from typing import List, Optional, Sequence, Union

from PySide6.QtCore import QEvent, Qt, QThread, Signal, Slot
from PySide6.QtGui import QCloseEvent, QFont
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QTabWidget,
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

MAX_LOG_BLOCKS = 3000

MODE_COIN = "coin"
MODE_MEMORY = "memory"
MODE_ALL_RESOURCES = "all_resources"
MODE_DIAMOND = "diamond"
MODE_TREATS = "treats"
MODE_PIGGY = "piggy"
MODE_MAP_TOUR = "map_tour"
MODE_TRACK_ISLAND = "track_island"

SHUTDOWN_WAIT_MS = 8000

STYLE_SHEET = """
QWidget#Root {
    background-color: #14161a;
}
QLabel#TitleLabel {
    color: #f2f4f8;
    font-size: 16px;
    font-weight: 600;
}
QLabel#SubtitleLabel {
    color: #6b7280;
    font-size: 11px;
}
QLabel#StatusLabel {
    color: #9aa4b2;
    font-size: 12px;
    padding: 4px 10px;
    border: 1px solid #2a2f38;
    border-radius: 10px;
    background-color: #1b1f25;
}
QLabel#StatsLabel {
    color: #5f6875;
    font-size: 11px;
}
QLabel#SectionTitle {
    color: #e5e9f0;
    font-size: 13px;
    font-weight: 600;
}
QLabel#SectionDesc {
    color: #8892b0;
    font-size: 11px;
}
QFrame#Divider {
    background-color: #23272e;
    max-height: 1px;
    border: none;
}
QTabWidget::pane {
    border: 1px solid #2a2f38;
    background-color: #16191f;
    border-radius: 8px;
    padding: 12px;
}
QTabBar::tab {
    background-color: #1b1f25;
    color: #9aa4b2;
    padding: 8px 18px;
    margin-right: 4px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    font-size: 12px;
    font-weight: 600;
}
QTabBar::tab:selected {
    background-color: #262b33;
    color: #ffffff;
    border-bottom: 2px solid #2f6df6;
}
QTabBar::tab:hover:!selected {
    background-color: #21252c;
    color: #d6dae1;
}
QPushButton {
    font-size: 12px;
    font-weight: 600;
    padding: 8px 16px;
    border-radius: 5px;
    border: none;
}
QPushButton#TourButton {
    background-color: #d35400;
    color: #ffffff;
    font-weight: bold;
}
QPushButton#TourButton:hover {
    background-color: #e67e22;
}
QPushButton#TourButton:pressed {
    background-color: #ba4a00;
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
QPushButton#ResourceButton {
    background-color: #10ac84;
    color: #ffffff;
    font-weight: bold;
}
QPushButton#ResourceButton:hover {
    background-color: #1dd1a1;
}
QPushButton#ResourceButton:pressed {
    background-color: #109873;
}
QPushButton#TrackButton {
    background-color: #2980b9;
    color: #ffffff;
    font-weight: bold;
}
QPushButton#TrackButton:hover {
    background-color: #3498db;
}
QPushButton#TrackButton:pressed {
    background-color: #2471a3;
}
QPushButton#SubActionButton {
    background-color: #262b33;
    color: #d6dae1;
}
QPushButton#SubActionButton:hover {
    background-color: #323841;
}
QPushButton#SubActionButton:pressed {
    background-color: #1e2229;
}
QPushButton#StopButton {
    background-color: #c0392b;
    color: #ffffff;
    font-weight: bold;
    padding: 5px 14px;
}
QPushButton#StopButton:hover {
    background-color: #e74c3c;
}
QPushButton#StopButton:pressed {
    background-color: #a93226;
}
QPushButton#ConfirmRowButton {
    background-color: #10ac84;
    color: #ffffff;
    font-size: 13px;
    font-weight: bold;
    border-radius: 4px;
    padding: 0px;
}
QPushButton#ConfirmRowButton:hover {
    background-color: #1dd1a1;
}
QPushButton#DeleteRowButton {
    background-color: #7f1d1d;
    color: #fca5a5;
    font-size: 14px;
    font-weight: bold;
    border-radius: 4px;
    padding: 0px;
}
QPushButton#DeleteRowButton:hover {
    background-color: #991b1b;
    color: #ffffff;
}
QPushButton:disabled {
    background-color: #1b1e23;
    color: #4a505a;
}
QLineEdit {
    background-color: #1b1f25;
    color: #d6dae1;
    border: 1px solid #2a2f38;
    border-radius: 5px;
    padding: 6px 10px;
    font-size: 12px;
}
QLineEdit:focus {
    border-color: #2f6df6;
}
QScrollArea#BlacklistScrollArea {
    border: 1px solid #2a2f38;
    background-color: #121418;
    border-radius: 6px;
}
QScrollArea#BlacklistScrollArea > QWidget {
    background-color: #121418;
}
QScrollArea#BlacklistScrollArea > QWidget > QWidget {
    background-color: #121418;
}
QWidget#BlacklistContainer {
    background-color: #121418;
}
QLineEdit#BlacklistRowEdit {
    background-color: #181b20;
    color: #d6dae1;
    border: 1px solid #2a2f38;
    border-radius: 4px;
    padding: 5px 8px;
}
QLineEdit#BlacklistRowEdit:focus {
    border-color: #2f6df6;
    background-color: #1e2229;
}
QScrollBar:vertical {
    border: none;
    background-color: #121418;
    width: 12px;
    margin: 0px;
    border-radius: 6px;
}
QScrollBar::handle:vertical {
    background-color: #2b303a;
    min-height: 24px;
    border-radius: 6px;
}
QScrollBar::handle:vertical:hover {
    background-color: #3b4250;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
    background: none;
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
}
QCheckBox {
    color: #9aa4b2;
    font-size: 12px;
    spacing: 6px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 3px;
    border: 1px solid #2a2f38;
    background-color: #1b1f25;
}
QCheckBox::indicator:checked {
    background-color: #2f6df6;
    border-color: #2f6df6;
}
QTextBrowser#LogView {
    background-color: #101216;
    border: 1px solid #22262e;
    border-radius: 6px;
    color: #c9d1d9;
    padding: 10px;
    font-family: Consolas, monospace;
}
"""


class BlacklistRow(QWidget):
    """Single row entry in the blacklist table.

    Displays an editable input and an action toggle button.
    """

    confirmed = Signal(object)  # Emits self on confirm
    deleted = Signal(object)    # Emits self on delete

    def __init__(
        self,
        initial_text: str = "",
        is_confirmed: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._confirmed_text = initial_text.strip()
        self._is_confirmed = is_confirmed

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(6)

        self.edit = QLineEdit(self)
        self.edit.setObjectName("BlacklistRowEdit")
        self.edit.setText(self._confirmed_text)
        self.edit.setPlaceholderText("输入屏蔽岛屿英文名称（例如：Gold Island）")
        self.edit.textEdited.connect(self._on_text_edited)
        self.edit.installEventFilter(self)

        self.btn = QPushButton(self)
        self.btn.setFixedSize(30, 28)
        self.btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn.clicked.connect(self._on_btn_clicked)

        layout.addWidget(self.edit, 1)
        layout.addWidget(self.btn, 0)

        self._update_btn_ui()

    def _update_btn_ui(self) -> None:
        if self._is_confirmed:
            self.btn.setText("×")
            self.btn.setToolTip("删除此屏蔽岛屿")
            self.btn.setObjectName("DeleteRowButton")
        else:
            self.btn.setText("✔")
            self.btn.setToolTip("确认并屏蔽此岛屿")
            self.btn.setObjectName("ConfirmRowButton")
        self.btn.setStyleSheet("")

    def _on_text_edited(self, _text: str) -> None:
        # Dynamically switch back to confirm mode when user begins editing
        if self._is_confirmed:
            self._is_confirmed = False
            self._update_btn_ui()

    def _on_btn_clicked(self) -> None:
        if self.btn.text() == "✔":
            # Confirm action
            text = self.edit.text().strip()
            if text:
                self._confirmed_text = text
                self._is_confirmed = True
                self._update_btn_ui()
                self.confirmed.emit(self)
        else:
            # Delete action
            self.deleted.emit(self)

    def eventFilter(self, obj: object, event: QEvent) -> bool:
        if obj == self.edit and event.type() == QEvent.Type.FocusOut:
            # If mouse is hovering over the action button, user is clicking it;
            # do not prematurely revert
            if self.btn.underMouse():
                return super().eventFilter(obj, event)
            # If user clicked away without confirming: revert to previously confirmed text
            if not self._is_confirmed and self._confirmed_text:
                self.edit.setText(self._confirmed_text)
                self._is_confirmed = True
                self._update_btn_ui()
        return super().eventFilter(obj, event)

    @property
    def is_confirmed(self) -> bool:
        return self._is_confirmed

    @property
    def confirmed_text(self) -> str:
        return self._confirmed_text if self._is_confirmed else ""


class BlacklistTableWidget(QWidget):
    """Scrollable dynamic list of blacklist island names with action buttons."""

    def __init__(
        self,
        initial_items: Sequence[str] = (),
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._rows: List[BlacklistRow] = []

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(4)

        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setObjectName("BlacklistScrollArea")
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.viewport().setObjectName("BlacklistViewport")
        self._scroll.viewport().setStyleSheet("background-color: #121418;")

        self._container = QWidget()
        self._container.setObjectName("BlacklistContainer")
        self._row_layout = QVBoxLayout(self._container)
        self._row_layout.setContentsMargins(6, 6, 6, 6)
        self._row_layout.setSpacing(4)
        self._row_layout.addStretch(1)

        self._scroll.setWidget(self._container)
        outer_layout.addWidget(self._scroll)

        # Pre-populate initial confirmed entries
        for item in initial_items:
            clean = item.strip()
            if clean:
                self._add_row(clean, is_confirmed=True)

        # Always append one blank unconfirmed row at the bottom
        self._add_row("", is_confirmed=False)

    def _add_row(self, text: str, is_confirmed: bool) -> BlacklistRow:
        row = BlacklistRow(
            initial_text=text, is_confirmed=is_confirmed, parent=self._container
        )
        row.confirmed.connect(self._on_row_confirmed)
        row.deleted.connect(self._on_row_deleted)
        # Insert before the trailing stretch item
        insert_idx = max(0, self._row_layout.count() - 1)
        self._row_layout.insertWidget(insert_idx, row)
        self._rows.append(row)
        return row

    def _on_row_confirmed(self, row: BlacklistRow) -> None:
        if row not in self._rows:
            return
        if row == self._rows[-1]:
            self._add_row("", is_confirmed=False)

    def _on_row_deleted(self, row: BlacklistRow) -> None:
        if row not in self._rows:
            return
        self._rows.remove(row)
        self._row_layout.removeWidget(row)
        row.deleteLater()
        if not self._rows or self._rows[-1].is_confirmed:
            self._add_row("", is_confirmed=False)

    def get_blacklist(self) -> List[str]:
        """Return list of confirmed non-empty island names."""
        items: List[str] = []
        for r in self._rows:
            t = r.confirmed_text.strip()
            if t and t not in items:
                items.append(t)
        return items

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable all inputs and buttons during execution."""
        for r in self._rows:
            r.edit.setEnabled(enabled)
            r.btn.setEnabled(enabled)


class MainWindow(QMainWindow):
    """Main application window hosting three tabs."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        super().__init__()
        self._cfg = config or DEFAULT_CONFIG
        self._worker: Optional[Union[BotEngine, MemoryEngine]] = None
        self._mode: str = MODE_MAP_TOUR

        self.setWindowTitle("MSM Helper - 挂机与巡航助手")
        self.setMinimumSize(680, 520)
        self.resize(780, 620)

        self._build_ui()
        self.setStyleSheet(STYLE_SHEET)

        self._append_log("INFO", "就绪。请先启动游戏窗口。")
        self._append_log("INFO", f"当前目标窗口：{self._cfg.window.title}")
        self._append_log(
            "INFO",
            "「常规控制」页支持全岛巡检与记忆游戏；「功能测试」页支持单岛各单项采集与岛屿追踪测试；「记忆游戏」需手动进入某一关开局界面。",
        )

    def _build_ui(self) -> None:
        root = QWidget(objectName="Root")
        self.setCentralWidget(root)

        outer = QVBoxLayout(root)
        outer.setContentsMargins(18, 16, 18, 16)
        outer.setSpacing(12)

        # ---- Persistent Header (Visible across all tabs)
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

        self._stats_label = QLabel("轮次 0 ｜ 点击 0", objectName="StatsLabel")
        self._stats_label.setAccessibleName("运行统计")
        header.addWidget(self._stats_label, 0, Qt.AlignmentFlag.AlignVCenter)

        self._status_label = QLabel("状态：空闲", objectName="StatusLabel")
        self._status_label.setAccessibleName("运行状态")
        header.addWidget(self._status_label, 0, Qt.AlignmentFlag.AlignVCenter)

        self._stop_button = QPushButton("停止", objectName="StopButton")
        self._stop_button.setAccessibleName("停止当前任务")
        self._stop_button.setToolTip("停止当前正在运行的任务（F10）")
        self._stop_button.setShortcut("F10")
        self._stop_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._stop_button.setEnabled(False)
        self._stop_button.clicked.connect(self._on_stop_clicked)
        header.addWidget(self._stop_button, 0, Qt.AlignmentFlag.AlignVCenter)

        outer.addLayout(header)
        outer.addWidget(QFrame(objectName="Divider"))

        # ---- Tab Widget
        self._tabs = QTabWidget()
        outer.addWidget(self._tabs, 1)

        # Tab 1: Main control
        tab_main = QWidget()
        self._build_tab_main(tab_main)
        self._tabs.addTab(tab_main, "常规控制")

        # Tab 2: Testing
        tab_test = QWidget()
        self._build_tab_test(tab_test)
        self._tabs.addTab(tab_test, "功能测试")

        # Tab 3: Execution log
        tab_log = QWidget()
        self._build_tab_log(tab_log)
        self._tabs.addTab(tab_log, "运行日志")

        # Tab 4: Settings
        tab_settings = QWidget()
        self._build_tab_settings(tab_settings)
        self._tabs.addTab(tab_settings, "偏好设置")

    # ------------------------------------------------ Tab 1: Main control
    def _build_tab_main(self, parent: QWidget) -> None:
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Row 1: Tour, Memory Game & Pair Toggle on the same line
        row1 = QHBoxLayout()
        row1.setSpacing(10)

        self._tour_button = QPushButton("收集各岛资源", objectName="TourButton")
        self._tour_button.setAccessibleName("收集各岛资源")
        self._tour_button.setToolTip(
            "自动打开地图，遍历未屏蔽的各个岛屿并执行资源全收，直至列表到底（F7）"
        )
        self._tour_button.setShortcut("F7")
        self._tour_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._tour_button.clicked.connect(self._on_start_tour_clicked)

        self._minigame_button = QPushButton("记忆游戏", objectName="MinigameButton")
        self._minigame_button.setAccessibleName("开始记忆游戏")
        self._minigame_button.setToolTip(
            "自动完成记忆小游戏，直到盘面不再出现（F11）。"
            "启动前请手动进入某一关的开局界面，所有卡牌背面朝下。"
        )
        self._minigame_button.setShortcut("F11")
        self._minigame_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._minigame_button.clicked.connect(self._on_start_minigame_clicked)

        row1.addWidget(self._tour_button)
        row1.addWidget(self._minigame_button)
        row1.addStretch(1)
        layout.addLayout(row1)

        layout.addWidget(QFrame(objectName="Divider"))

        # Blacklist Table Section
        bl_title = QLabel(
            "屏蔽岛屿黑名单（列表可上下滚动，点击 ✔ 确认新增，点击 × 删除，修改失焦自动复原）：",
            objectName="SectionTitle",
        )
        layout.addWidget(bl_title)

        self._blacklist_table = BlacklistTableWidget(
            initial_items=self._cfg.map.blacklist, parent=parent
        )
        layout.addWidget(self._blacklist_table, 1)

    # ------------------------------------------------ Tab 2: Testing
    def _build_tab_test(self, parent: QWidget) -> None:
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(14)

        # Section 1: Single-island collection tests
        sec1_title = QLabel("单岛收集测试：‘金币’ ‘钻石’ ‘食物’ ‘储蓄罐’ ‘全部’", objectName="SectionTitle")
        sec1_desc = QLabel(
            "针对当前所处岛屿执行单项或复合收集测试（起点须为岛屿界面）：",
            objectName="SectionDesc",
        )
        layout.addWidget(sec1_title)
        layout.addWidget(sec1_desc)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self._test_coin_btn = QPushButton("金币", objectName="SubActionButton")
        self._test_coin_btn.setToolTip("在当前岛屿收集金币（F9）")
        self._test_coin_btn.setShortcut("F9")
        self._test_coin_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._test_coin_btn.clicked.connect(self._on_start_coin_clicked)
        self._test_coin_btn.setAccessibleName("开始收集金币")
        self._start_button = self._test_coin_btn

        self._test_diamond_btn = QPushButton("钻石", objectName="SubActionButton")
        self._test_diamond_btn.setToolTip("在当前岛屿检索并收集钻石矿")
        self._test_diamond_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._test_diamond_btn.clicked.connect(self._on_start_diamond_clicked)
        self._test_diamond_btn.setAccessibleName("开始收集钻石")
        self._diamond_button = self._test_diamond_btn

        self._test_treats_btn = QPushButton("食物", objectName="SubActionButton")
        self._test_treats_btn.setToolTip("在当前岛屿检索并收集烘焙坊食物")
        self._test_treats_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._test_treats_btn.clicked.connect(self._on_start_treats_clicked)
        self._test_treats_btn.setAccessibleName("开始收集食物")
        self._treats_button = self._test_treats_btn

        self._test_piggy_btn = QPushButton("储蓄罐", objectName="SubActionButton")
        self._test_piggy_btn.setToolTip("在当前岛屿检查高光小猪储蓄罐并确认弹窗")
        self._test_piggy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._test_piggy_btn.clicked.connect(self._on_start_piggy_clicked)
        self._test_piggy_btn.setAccessibleName("使用小猪储蓄罐")
        self._piggy_button = self._test_piggy_btn

        self._test_all_btn = QPushButton("全部", objectName="ResourceButton")
        self._test_all_btn.setToolTip("当前岛屿全量流水线：小猪 -> 钻石 -> 食物 -> 金币（F8）")
        self._test_all_btn.setShortcut("F8")
        self._test_all_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._test_all_btn.clicked.connect(self._on_start_resource_clicked)
        self._test_all_btn.setAccessibleName("开始资源全收流水线")
        self._resource_button = self._test_all_btn

        btn_row.addWidget(self._test_coin_btn)
        btn_row.addWidget(self._test_diamond_btn)
        btn_row.addWidget(self._test_treats_btn)
        btn_row.addWidget(self._test_piggy_btn)
        btn_row.addWidget(self._test_all_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        layout.addWidget(QFrame(objectName="Divider"))

        # Section 2: Island tracking test
        sec2_title = QLabel("岛屿追踪测试", objectName="SectionTitle")
        sec2_desc = QLabel(
            "输入对应岛屿英文名称，脚本自动在地图列表中滑动检索目标岛屿并进入（起点可以是岛屿或地图）：",
            objectName="SectionDesc",
        )
        layout.addWidget(sec2_title)
        layout.addWidget(sec2_desc)

        track_row = QHBoxLayout()
        track_row.setSpacing(8)

        self._track_input = QLineEdit()
        self._track_input.setPlaceholderText("输入目标岛屿英文名称，例如：Cold Island 或 Air Island")
        self._track_input.setAccessibleName("目标追踪岛屿名称")

        self._track_btn = QPushButton("追踪并进入", objectName="TrackButton")
        self._track_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._track_btn.clicked.connect(self._on_start_track_clicked)

        track_row.addWidget(self._track_input, 1)
        track_row.addWidget(self._track_btn, 0)
        layout.addLayout(track_row)

        layout.addStretch(1)

    # ------------------------------------------------ Tab 3: Execution log
    def _build_tab_log(self, parent: QWidget) -> None:
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        log_bar = QHBoxLayout()
        log_label = QLabel("详细运行日志", objectName="SectionTitle")
        log_bar.addWidget(log_label)
        log_bar.addStretch(1)

        clear_btn = QPushButton("清空日志", objectName="SubActionButton")
        clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        clear_btn.clicked.connect(self._on_clear_log_clicked)
        log_bar.addWidget(clear_btn)

        layout.addLayout(log_bar)

        self._log_view = QTextBrowser(objectName="LogView")
        self._log_view.setReadOnly(True)
        self._log_view.setAccessibleName("运行日志")
        self._log_view.setOpenExternalLinks(False)
        self._log_view.document().setMaximumBlockCount(MAX_LOG_BLOCKS)
        self._log_view.setFont(QFont("Consolas", 9))
        layout.addWidget(self._log_view, 1)

    # ------------------------------------------------ Tab 4: Settings
    def _build_tab_settings(self, parent: QWidget) -> None:
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(14)

        # Section 1: Memory game settings
        sec1_title = QLabel("记忆游戏选项：", objectName="SectionTitle")
        layout.addWidget(sec1_title)

        self._scan_first_box = QCheckBox("加速配对（先翻开全部卡牌再逐对收牌）", objectName="ScanFirstBox")
        self._scan_first_box.setAccessibleName("加速配对：先翻开全部卡牌再逐对收牌")
        self._scan_first_box.setToolTip(
            "改变记忆小游戏的翻牌顺序：先按序把所有卡牌翻开看清，再逐对收牌。\n"
            "最坏失配次数 n 次，比默认的「边翻边收」多一次，仍远在游戏给的机会数之内。\n"
            "默认开启以获得最大通关速度；不勾选则一旦认出成对就立刻收掉。"
        )
        self._scan_first_box.setChecked(False)
        self._scan_first_box.setCursor(Qt.CursorShape.PointingHandCursor)
        layout.addWidget(self._scan_first_box)

        layout.addWidget(QFrame(objectName="Divider"))

        # Section 2: Map initialization description
        sec2_title = QLabel("地图初始化说明：", objectName="SectionTitle")
        layout.addWidget(sec2_title)

        sec2_desc = QLabel(
            "智能动态防回弹置顶：无固定首岛名称或坐标假设，自动通过列表物理拖拽内容不变性判定真实物理天花板。",
            objectName="SectionDesc",
        )
        sec2_desc.setWordWrap(True)
        layout.addWidget(sec2_desc)

        layout.addStretch(1)

    # --------------------------------------------------------- Button Handlers

    @Slot()
    def _on_start_tour_clicked(self) -> None:
        self._start(MODE_MAP_TOUR)

    @Slot()
    def _on_start_minigame_clicked(self) -> None:
        self._start(MODE_MEMORY)

    @Slot()
    def _on_start_resource_clicked(self) -> None:
        self._start(MODE_ALL_RESOURCES)

    @Slot()
    def _on_start_coin_clicked(self) -> None:
        self._start(MODE_COIN)

    @Slot()
    def _on_start_diamond_clicked(self) -> None:
        self._start(MODE_DIAMOND)

    @Slot()
    def _on_start_treats_clicked(self) -> None:
        self._start(MODE_TREATS)

    @Slot()
    def _on_start_piggy_clicked(self) -> None:
        self._start(MODE_PIGGY)

    @Slot()
    def _on_start_track_clicked(self) -> None:
        target = self._track_input.text().strip()
        if not target:
            self._append_log("WARN", "请先输入要追踪的目标岛屿名称！")
            return
        self._start(MODE_TRACK_ISLAND)

    @Slot()
    def _on_clear_log_clicked(self) -> None:
        self._log_view.clear()

    def _start(self, mode: str) -> None:
        """Start a worker thread."""
        if self._worker is not None and self._worker.isRunning():
            return

        if mode == MODE_MEMORY:
            worker: Union[BotEngine, MemoryEngine] = MemoryEngine(
                config=self._cfg,
                params=RunnerParams(scan_first=self._scan_first_box.isChecked()),
            )
        else:
            worker = BotEngine(config=self._cfg, mode=mode)
            if mode == MODE_MAP_TOUR:
                worker.set_blacklist(self._blacklist_table.get_blacklist())
            elif mode == MODE_TRACK_ISLAND:
                worker.set_track_target(self._track_input.text().strip())

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
        self._stop_button.setEnabled(False)
        self._worker.stop()

    def _set_running(self, running: bool) -> None:
        """Update controls enabled status across all tabs."""
        # Tab 1 controls
        self._tour_button.setEnabled(not running)
        self._minigame_button.setEnabled(not running)
        self._blacklist_table.set_enabled(not running)

        # Tab 2 controls
        self._test_coin_btn.setEnabled(not running)
        self._test_diamond_btn.setEnabled(not running)
        self._test_treats_btn.setEnabled(not running)
        self._test_piggy_btn.setEnabled(not running)
        self._test_all_btn.setEnabled(not running)
        self._track_input.setEnabled(not running)
        self._track_btn.setEnabled(not running)

        # Tab 4 controls
        self._scan_first_box.setEnabled(not running)

        # Header controls
        self._stop_button.setEnabled(running)

    # ---------------------------------------------------- Worker Signals -> UI

    @Slot(str, str)
    def _append_log(self, level: str, message: str) -> None:
        """Render one log line into the log view."""
        color = LEVEL_COLORS.get(level, LEVEL_COLORS["INFO"])
        timestamp = datetime.now().strftime("%H:%M:%S")
        safe_text = html.escape(message)

        self._log_view.append(
            f'<span style="color:#4a5058;">[{timestamp}]</span> '
            f'<span style="color:{color};">{safe_text}</span>'
        )
        scrollbar = self._log_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    @Slot(str)
    def _on_state_changed(self, state: str) -> None:
        self._status_label.setText(f"状态：{state}")

    @Slot(int, int)
    def _on_stats_changed(self, first: int, second: int) -> None:
        """Update stats label based on active mode."""
        if self._mode == MODE_MEMORY:
            self._stats_label.setText(f"已完成 {first} 关 ｜ 翻牌 {second} 轮")
        elif self._mode == MODE_MAP_TOUR:
            self._stats_label.setText(f"巡航轮次 {first} ｜ 点击 {second}")
        elif self._mode == MODE_TRACK_ISLAND:
            self._stats_label.setText(f"追踪测试 ｜ 检索轮次 {first}")
        else:
            self._stats_label.setText(f"轮次 {first} ｜ 点击 {second}")

    @Slot()
    def _on_worker_finished(self) -> None:
        self._set_running(False)

    def closeEvent(self, event: QCloseEvent) -> None:
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.stop()
            if not worker.wait(SHUTDOWN_WAIT_MS):
                worker.terminate()
                worker.wait(500)
        event.accept()
