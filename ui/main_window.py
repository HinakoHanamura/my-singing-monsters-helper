from __future__ import annotations

import html
import os
from datetime import datetime
from dataclasses import replace
from typing import List, Optional, Sequence, Union

from PySide6.QtCore import QEvent, Qt, QThread, Signal, Slot
from PySide6.QtGui import QCloseEvent, QFont
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
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

from config import DEFAULT_CONFIG, AppConfig, load_user_settings, save_user_settings
from core.bot_engine import BotEngine
from core.minigames.memory_engine import MemoryEngine, RunnerParams

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

TRANSLATIONS = {
    "en": {
        "window_title": "My Singing Monsters Helper",
        "app_title": "My Singing Monsters Helper",
        "app_subtitle": "",
        "status_idle": "Status: Idle",
        "status_running": "Status: Running",
        "status_prefix": "Status: ",
        "stop": "Stop",
        "stop_tip": "Stop currently running task (F10)",
        "tab_main": "General Controls",
        "tab_test": "Function Tests",
        "tab_log": "Execution Log",
        "tab_settings": "Settings",
        "start_tour": "Harvest All Islands",
        "start_tour_tip": "Open map, tour all un-blacklisted islands and harvest resources until list ends (F7)",
        "start_memory": "Memory Game",
        "start_memory_tip": "Automatically solve memory minigame (F11). Enter starting grid manually before launch.",
        "blacklist_title": "Island Blacklist:",
        "blacklist_placeholder": "Enter island name to skip (e.g. Gold Island)",
        "blacklist_confirm_tip": "Confirm and blacklist this island",
        "blacklist_delete_tip": "Delete from blacklist",
        "test_sec1_title": "Single-Island Collection Tests",
        "test_sec1_desc": "Test individual or full resource collection on current island:",
        "test_coin": "Coin",
        "test_coin_tip": "Collect coins on current island (F9)",
        "test_diamond": "Diamond",
        "test_diamond_tip": "Search and collect diamond mine on current island",
        "test_treats": "Treats",
        "test_treats_tip": "Search and collect bakery treats on current island",
        "test_piggy": "Piggy Bank",
        "test_piggy_tip": "Check highlighted piggy bank and confirm modal on current island",
        "test_all": "All",
        "test_all_tip": "Full collection pipeline on current island: Piggy -> Diamond -> Treats -> Coin (F8)",
        "test_sec2_title": "Island Tracking Test",
        "test_sec2_desc": "Search and navigate to an island on the map list:",
        "track_placeholder": "Enter target island name (e.g. Cold Island)",
        "track_btn": "Search & Enter",
        "log_title": "Execution Log",
        "clear_log": "Clear Log",
        "settings_lang_title": "Interface Language / 界面语言:",
        "settings_memory_title": "Memory Game Options:",
        "scan_first": "Fast Pairing (Flip all cards before harvesting pairs)",
        "scan_first_tip": "Flip and record all cards before harvesting pairs. Maximizes completion speed.",
        "settings_map_title": "Map Navigation & Initialization Options:",
        "reset_map_box": "Auto-scroll map list to top before starting tour",
        "reset_map_box_tip": "When enabled, automatically scrolls to the top of the island list before starting the tour.",
        "brake_dynamic": "Dynamic top detection (physical bounce invariance)",
        "brake_dynamic_tip": "Automatically detects physical top ceiling via scroll physics invariance without island name assumptions.",
        "brake_first_island": "Stop at specified first island:",
        "brake_first_island_tip": "Stops scrolling to top when the specified island appears at the top.",
        "first_island_placeholder": "First island name (e.g. Plant Island)",
        "first_island_save_tip": "Save first island name",
        "stats_idle": "Rounds 0 ｜ Clicks 0",
        "stats_memory": "Completed {first} Levels ｜ Flipped {second} Rounds",
        "stats_tour": "Tour Rounds {first} ｜ Clicks {second}",
        "stats_track": "Tracking Test ｜ Search Rounds {first}",
        "stats_generic": "Rounds {first} ｜ Clicks {second}",
        "warn_empty_target": "Please enter target island name first!",
        "log_ready": "Ready. Please launch the game window first.",
        "log_target": "Current target window: {title}",
        "log_help": "General Controls supports full island tour and memory game; Function Tests supports single resource tests.",
    },
    "zh": {
        "window_title": "My Singing Monsters Helper",
        "app_title": "My Singing Monsters Helper",
        "app_subtitle": "",
        "status_idle": "状态：空闲",
        "status_running": "状态：运行中",
        "status_prefix": "状态：",
        "stop": "停止",
        "stop_tip": "停止当前正在运行的任务（F10）",
        "tab_main": "常规控制",
        "tab_test": "功能测试",
        "tab_log": "运行日志",
        "tab_settings": "偏好设置",
        "start_tour": "收集各岛资源",
        "start_tour_tip": "自动打开地图，遍历未屏蔽的各个岛屿并执行资源全收，直至列表到底（F7）",
        "start_memory": "记忆游戏",
        "start_memory_tip": "自动完成记忆小游戏，直到盘面不再出现（F11）。启动前请手动进入某一关的开局界面。",
        "blacklist_title": "屏蔽岛屿黑名单：",
        "blacklist_placeholder": "输入屏蔽岛屿英文名称（例如：Gold Island）",
        "blacklist_confirm_tip": "确认并屏蔽此岛屿",
        "blacklist_delete_tip": "删除此屏蔽岛屿",
        "test_sec1_title": "单岛收集测试",
        "test_sec1_desc": "在当前岛屿执行单项或全量资源收集测试：",
        "test_coin": "金币",
        "test_coin_tip": "在当前岛屿收集金币（F9）",
        "test_diamond": "钻石",
        "test_diamond_tip": "在当前岛屿检索并收集钻石矿",
        "test_treats": "食物",
        "test_treats_tip": "在当前岛屿检索并收集烘焙坊食物",
        "test_piggy": "储蓄罐",
        "test_piggy_tip": "在当前岛屿检查高光小猪储蓄罐并确认弹窗",
        "test_all": "全部",
        "test_all_tip": "当前岛屿全量流水线：小猪 -> 钻石 -> 食物 -> 金币（F8）",
        "test_sec2_title": "岛屿追踪测试",
        "test_sec2_desc": "输入岛屿名称，在地图列表中滑动检索并进入：",
        "track_placeholder": "输入目标岛屿英文名称，例如：Cold Island 或 Air Island",
        "track_btn": "追踪并进入",
        "log_title": "运行日志",
        "clear_log": "清空日志",
        "settings_lang_title": "界面语言 / Language:",
        "settings_memory_title": "记忆游戏选项：",
        "scan_first": "加速配对（先翻开全部卡牌再逐对收牌）",
        "scan_first_tip": "改变记忆小游戏的翻牌顺序：先按序把所有卡牌翻开看清，再逐对收牌。",
        "settings_map_title": "地图导航与初始化选项：",
        "reset_map_box": "开启巡岛前自动滑动置顶地图列表（从第一座岛开始）",
        "reset_map_box_tip": "开启巡岛时会自动将地图列表滑动置顶，从第一座岛开始遍历。",
        "brake_dynamic": "动态置顶（自适应物理防回弹检测，无需指定岛名）",
        "brake_dynamic_tip": "通过列表物理下拉不变性判定真实物理天花板，无需假设首岛名称。",
        "brake_first_island": "根据首岛名称置顶：",
        "brake_first_island_tip": "滑动置顶列表时，一旦在顶部检测到该首岛名称即完成初始化。",
        "first_island_placeholder": "首岛名称（如 Plant Island）",
        "first_island_save_tip": "保存首岛名称",
        "stats_idle": "轮次 0 ｜ 点击 0",
        "stats_memory": "已完成 {first} 关 ｜ 翻牌 {second} 轮",
        "stats_tour": "巡航轮次 {first} ｜ 点击 {second}",
        "stats_track": "追踪测试 ｜ 检索轮次 {first}",
        "stats_generic": "轮次 {first} ｜ 点击 {second}",
        "warn_empty_target": "请先输入要追踪的目标岛屿名称！",
        "log_ready": "就绪。请先启动游戏窗口。",
        "log_target": "当前目标窗口：{title}",
        "log_help": "「常规控制」页支持全岛巡检与记忆游戏（启动前请手动进入某一关的开局界面）；「功能测试」页支持单岛各单项采集与岛屿追踪测试。",
    },
}

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
QPushButton#TourButton {
    background-color: #1f242c;
    color: #f2f4f8;
    border: 1px solid #3b82f6;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
    padding: 8px 18px;
}
QPushButton#TourButton:hover {
    background-color: #262c36;
    border-color: #60a5fa;
}
QPushButton#TourButton:disabled {
    background-color: #181b21;
    color: #4b5260;
    border-color: #2a2f38;
}
QPushButton#MinigameButton {
    background-color: #1f242c;
    color: #f2f4f8;
    border: 1px solid #4ade80;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
    padding: 8px 18px;
}
QPushButton#MinigameButton:hover {
    background-color: #262c36;
    border-color: #86efac;
}
QPushButton#MinigameButton:disabled {
    background-color: #181b21;
    color: #4b5260;
    border-color: #2a2f38;
}
QPushButton#StopButton {
    background-color: #3b181a;
    color: #fca5a5;
    border: 1px solid #b71c1c;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 600;
    padding: 6px 14px;
}
QPushButton#StopButton:hover {
    background-color: #4c1d20;
    border-color: #ef4444;
}
QPushButton#StopButton:disabled {
    background-color: #181b21;
    color: #4b5260;
    border-color: #2a2f38;
}
QPushButton#SubActionButton, QPushButton#ResourceButton, QPushButton#TrackButton {
    background-color: #1b1f25;
    color: #d8dee9;
    border: 1px solid #2a2f38;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 500;
    padding: 6px 14px;
}
QPushButton#SubActionButton:hover, QPushButton#ResourceButton:hover, QPushButton#TrackButton:hover {
    background-color: #222730;
    border-color: #3b4252;
}
QPushButton#SubActionButton:disabled, QPushButton#ResourceButton:disabled, QPushButton#TrackButton:disabled {
    background-color: #181b21;
    color: #4b5260;
    border-color: #222730;
}
QPushButton#ResourceButton {
    border-color: #3b82f6;
    color: #93c5fd;
}
QPushButton#TrackButton {
    border-color: #4ade80;
    color: #86efac;
}
QComboBox#LangCombo {
    background-color: #1b1f25;
    color: #e5e9f0;
    border: 1px solid #2a2f38;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 11px;
    min-width: 75px;
}
QComboBox#LangCombo:hover {
    border-color: #3b4252;
}
QComboBox#LangCombo::drop-down {
    border: none;
    width: 14px;
}
QComboBox#LangCombo QAbstractItemView {
    background-color: #1b1f25;
    color: #e5e9f0;
    selection-background-color: #2e3440;
    border: 1px solid #2a2f38;
    outline: none;
}
QTextBrowser#LogView {
    background-color: #0f1114;
    border: 1px solid #222730;
    border-radius: 8px;
    padding: 8px;
    color: #8b95a5;
    font-family: Consolas, "Courier New", monospace;
    font-size: 11px;
}
QFrame#Divider {
    color: #222730;
    max-height: 1px;
}
QTabWidget::pane {
    border: 1px solid #222730;
    border-radius: 8px;
    background-color: #14161a;
    top: -1px;
}
QTabBar::tab {
    background-color: #181b21;
    color: #7b8494;
    border: 1px solid #222730;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 8px 16px;
    font-size: 12px;
    font-weight: 500;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background-color: #14161a;
    color: #f2f4f8;
    border-color: #222730;
    font-weight: 600;
}
QTabBar::tab:hover:!selected {
    background-color: #1c2027;
    color: #c5cdd8;
}
QCheckBox {
    color: #c5cdd8;
    font-size: 12px;
    spacing: 6px;
}
QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid #3b4252;
    border-radius: 3px;
    background-color: #1b1f25;
}
QCheckBox::indicator:checked {
    background-color: #3b82f6;
    border-color: #60a5fa;
}
QRadioButton {
    color: #ffffff;
    font-size: 12px;
    spacing: 6px;
}
QRadioButton:disabled {
    color: #4b5260;
}
QRadioButton#BrakeDynamicRadio, QRadioButton#BrakeFirstIslandRadio {
    color: #ffffff;
    font-size: 12px;
}
QRadioButton#BrakeDynamicRadio:disabled, QRadioButton#BrakeFirstIslandRadio:disabled {
    color: #4b5260;
}
QRadioButton::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid #3b4252;
    border-radius: 7px;
    background-color: #1b1f25;
}
QRadioButton::indicator:checked {
    background-color: #3b82f6;
    border-color: #60a5fa;
}
QRadioButton::indicator:disabled {
    background-color: #181b21;
    border-color: #2a2f38;
}
QLineEdit {
    background-color: #181b21;
    border: 1px solid #2a2f38;
    border-radius: 6px;
    color: #e5e9f0;
    padding: 6px 10px;
    font-size: 12px;
}
QLineEdit:focus {
    border-color: #3b82f6;
}
QScrollArea#BlacklistScrollArea {
    border: 1px solid #222730;
    border-radius: 6px;
    background-color: #121418;
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
    background-color: #1a1d24;
    border: 1px solid #2b303c;
    border-radius: 4px;
    color: #eceff4;
    padding: 4px 8px;
    font-size: 12px;
}
QLineEdit#BlacklistRowEdit:focus {
    border: 1px solid #4c566a;
    background-color: #1e222a;
}
QPushButton#ConfirmRowButton {
    background-color: #1e3a24;
    color: #4ade80;
    border: 1px solid #2e5a36;
    border-radius: 4px;
    font-weight: bold;
    font-size: 13px;
}
QPushButton#ConfirmRowButton:hover {
    background-color: #274d30;
    border-color: #4ade80;
}
QPushButton#DeleteRowButton {
    background-color: #3a1e20;
    color: #f87171;
    border: 1px solid #5a2e32;
    border-radius: 4px;
    font-weight: bold;
    font-size: 14px;
}
QPushButton#DeleteRowButton:hover {
    background-color: #4d272a;
    border-color: #f87171;
}
"""


class BlacklistRow(QWidget):
    confirmed = Signal(object)
    deleted = Signal(object)

    def __init__(
        self,
        initial_text: str = "",
        is_confirmed: bool = False,
        lang: str = "en",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._confirmed_text = initial_text.strip()
        self._is_confirmed = is_confirmed
        self._lang = lang

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(6)

        self.edit = QLineEdit(self)
        self.edit.setObjectName("BlacklistRowEdit")
        self.edit.setText(self._confirmed_text)
        self.edit.textEdited.connect(self._on_text_edited)
        self.edit.installEventFilter(self)

        self.btn = QPushButton(self)
        self.btn.setFixedSize(30, 28)
        self.btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn.clicked.connect(self._on_btn_clicked)

        layout.addWidget(self.edit, 1)
        layout.addWidget(self.btn, 0)

        self._update_btn_ui()
        self.update_language(self._lang)

    def update_language(self, lang: str) -> None:
        self._lang = lang
        t = TRANSLATIONS.get(lang, TRANSLATIONS["en"])
        self.edit.setPlaceholderText(t["blacklist_placeholder"])
        if self._is_confirmed:
            self.btn.setToolTip(t["blacklist_delete_tip"])
        else:
            self.btn.setToolTip(t["blacklist_confirm_tip"])

    def _update_btn_ui(self) -> None:
        t = TRANSLATIONS.get(self._lang, TRANSLATIONS["en"])
        if self._is_confirmed:
            self.btn.setText("×")
            self.btn.setToolTip(t["blacklist_delete_tip"])
            self.btn.setObjectName("DeleteRowButton")
        else:
            self.btn.setText("✔")
            self.btn.setToolTip(t["blacklist_confirm_tip"])
            self.btn.setObjectName("ConfirmRowButton")
        self.btn.setStyleSheet("")

    def _on_text_edited(self, _text: str) -> None:
        if self._is_confirmed:
            self._is_confirmed = False
            self._update_btn_ui()

    def _on_btn_clicked(self) -> None:
        if self.btn.text() == "✔":
            text = self.edit.text().strip()
            if text:
                self._confirmed_text = text
                self._is_confirmed = True
                self._update_btn_ui()
                self.confirmed.emit(self)
        else:
            self.deleted.emit(self)

    def eventFilter(self, obj: object, event: QEvent) -> bool:
        if obj == self.edit and event.type() == QEvent.Type.FocusOut:
            if self.btn.underMouse():
                return super().eventFilter(obj, event)
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
        return self._confirmed_text


class BlacklistTableWidget(QWidget):
    changed = Signal()

    def __init__(
        self,
        initial_items: Sequence[str] = (),
        lang: str = "en",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._lang = lang
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

        for item in initial_items:
            clean = item.strip()
            if clean:
                self._add_row(clean, is_confirmed=True)

        self._add_row("", is_confirmed=False)

    def _add_row(self, text: str, is_confirmed: bool) -> BlacklistRow:
        row = BlacklistRow(
            initial_text=text, is_confirmed=is_confirmed, lang=self._lang, parent=self._container
        )
        row.confirmed.connect(self._on_row_confirmed)
        row.deleted.connect(self._on_row_deleted)
        insert_idx = max(0, self._row_layout.count() - 1)
        self._row_layout.insertWidget(insert_idx, row)
        self._rows.append(row)
        return row

    def _on_row_confirmed(self, row: BlacklistRow) -> None:
        if row not in self._rows:
            return
        if row == self._rows[-1]:
            self._add_row("", is_confirmed=False)
        self.changed.emit()

    def _on_row_deleted(self, row: BlacklistRow) -> None:
        if row not in self._rows:
            return
        self._rows.remove(row)
        self._row_layout.removeWidget(row)
        row.deleteLater()
        if not self._rows or self._rows[-1].is_confirmed:
            self._add_row("", is_confirmed=False)
        self.changed.emit()

    def set_language(self, lang: str) -> None:
        self._lang = lang
        for r in self._rows:
            r.update_language(lang)

    def get_blacklist(self) -> List[str]:
        items: List[str] = []
        for r in self._rows:
            t = r.confirmed_text.strip()
            if t and t not in items:
                items.append(t)
        return items

    def set_enabled(self, enabled: bool) -> None:
        for r in self._rows:
            r.edit.setEnabled(enabled)
            r.btn.setEnabled(enabled)


class MainWindow(QMainWindow):
    def __init__(self, config: Optional[AppConfig] = None) -> None:
        super().__init__()
        self._cfg = config or DEFAULT_CONFIG
        self._worker: Optional[Union[BotEngine, MemoryEngine]] = None
        self._mode: str = MODE_MAP_TOUR

        settings = load_user_settings() if "PYTEST_CURRENT_TEST" not in os.environ else {}
        self._lang: str = settings.get("language", "zh")
        saved_bl = settings.get("blacklist", None)
        self._initial_bl = saved_bl if saved_bl is not None else self._cfg.map.blacklist
        self._initial_scan_first = bool(settings.get("scan_first", False))
        self._initial_reset_map = bool(settings.get("reset_map_to_top", True))
        self._initial_brake_mode = str(settings.get("init_brake_mode", getattr(self._cfg.map, "init_brake_mode", "dynamic")))
        self._initial_first_island_name = str(settings.get("first_island_name", getattr(self._cfg.map, "first_island_name", "Plant Island")))
        self._saved_first_island_name = self._initial_first_island_name

        self.setMinimumSize(700, 530)
        self.resize(790, 620)

        self._build_ui()

        self._resource_button = self._test_all_btn
        self._diamond_button = self._test_diamond_btn
        self._treats_button = self._test_treats_btn
        self._piggy_button = self._test_piggy_btn
        self._start_button = self._test_coin_btn

        self.setStyleSheet(STYLE_SHEET)
        self._retranslate_ui()

        t = TRANSLATIONS.get(self._lang, TRANSLATIONS["en"])
        self._append_log("INFO", t["log_ready"])
        self._append_log("INFO", t["log_target"].format(title=self._cfg.window.title))
        self._append_log("INFO", t["log_help"])

    def _build_ui(self) -> None:
        root = QWidget(objectName="Root")
        self.setCentralWidget(root)

        outer = QVBoxLayout(root)
        outer.setContentsMargins(18, 16, 18, 16)
        outer.setSpacing(12)

        header = QHBoxLayout()
        header.setSpacing(12)

        self._title_label = QLabel(objectName="TitleLabel")
        self._subtitle_label = QLabel(objectName="SubtitleLabel")
        header.addWidget(self._title_label, 0, Qt.AlignmentFlag.AlignVCenter)

        header.addStretch(1)

        self._lang_combo = QComboBox(objectName="LangCombo")
        self._lang_combo.addItem("English", "en")
        self._lang_combo.addItem("中文", "zh")
        current_idx = 0 if self._lang == "en" else 1
        self._lang_combo.setCurrentIndex(current_idx)
        self._lang_combo.currentIndexChanged.connect(self._on_language_changed)
        header.addWidget(self._lang_combo, 0, Qt.AlignmentFlag.AlignVCenter)

        self._stats_label = QLabel(objectName="StatsLabel")
        header.addWidget(self._stats_label, 0, Qt.AlignmentFlag.AlignVCenter)

        self._status_label = QLabel(objectName="StatusLabel")
        header.addWidget(self._status_label, 0, Qt.AlignmentFlag.AlignVCenter)

        self._stop_button = QPushButton(objectName="StopButton")
        self._stop_button.setAccessibleName("停止当前任务")
        self._stop_button.setShortcut("F10")
        self._stop_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._stop_button.setEnabled(False)
        self._stop_button.clicked.connect(self._on_stop_clicked)
        header.addWidget(self._stop_button, 0, Qt.AlignmentFlag.AlignVCenter)

        outer.addLayout(header)
        outer.addWidget(QFrame(objectName="Divider"))

        self._tabs = QTabWidget()
        outer.addWidget(self._tabs, 1)

        tab_main = QWidget()
        self._build_tab_main(tab_main)
        self._tabs.addTab(tab_main, "")

        tab_test = QWidget()
        self._build_tab_test(tab_test)
        self._tabs.addTab(tab_test, "")

        tab_log = QWidget()
        self._build_tab_log(tab_log)
        self._tabs.addTab(tab_log, "")

        tab_settings = QWidget()
        self._build_tab_settings(tab_settings)
        self._tabs.addTab(tab_settings, "")

    def _build_tab_main(self, parent: QWidget) -> None:
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        row1 = QHBoxLayout()
        row1.setSpacing(10)

        self._tour_button = QPushButton(objectName="TourButton")
        self._tour_button.setAccessibleName("开始全岛巡航")
        self._tour_button.setShortcut("F7")
        self._tour_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._tour_button.clicked.connect(self._on_start_tour_clicked)

        self._minigame_button = QPushButton(objectName="MinigameButton")
        self._minigame_button.setAccessibleName("开始记忆小游戏")
        self._minigame_button.setShortcut("F11")
        self._minigame_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._minigame_button.clicked.connect(self._on_start_minigame_clicked)

        row1.addWidget(self._tour_button)
        row1.addWidget(self._minigame_button)
        row1.addStretch(1)
        layout.addLayout(row1)

        layout.addWidget(QFrame(objectName="Divider"))

        self._bl_title = QLabel(objectName="SectionTitle")
        layout.addWidget(self._bl_title)

        self._blacklist_table = BlacklistTableWidget(
            initial_items=self._initial_bl, lang=self._lang, parent=parent
        )
        self._blacklist_table.changed.connect(self._save_settings)
        layout.addWidget(self._blacklist_table, 1)

    def _build_tab_test(self, parent: QWidget) -> None:
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(14)

        self._sec1_title = QLabel(objectName="SectionTitle")
        self._sec1_desc = QLabel(objectName="SectionDesc")
        layout.addWidget(self._sec1_title)
        layout.addWidget(self._sec1_desc)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self._test_coin_btn = QPushButton(objectName="SubActionButton")
        self._test_coin_btn.setAccessibleName("开始收集金币")
        self._test_coin_btn.setShortcut("F9")
        self._test_coin_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._test_coin_btn.clicked.connect(self._on_start_coin_clicked)

        self._test_diamond_btn = QPushButton(objectName="SubActionButton")
        self._test_diamond_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._test_diamond_btn.clicked.connect(self._on_start_diamond_clicked)

        self._test_treats_btn = QPushButton(objectName="SubActionButton")
        self._test_treats_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._test_treats_btn.clicked.connect(self._on_start_treats_clicked)

        self._test_piggy_btn = QPushButton(objectName="SubActionButton")
        self._test_piggy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._test_piggy_btn.clicked.connect(self._on_start_piggy_clicked)

        self._test_all_btn = QPushButton(objectName="ResourceButton")
        self._test_all_btn.setAccessibleName("全部收集")
        self._test_all_btn.setShortcut("F8")
        self._test_all_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._test_all_btn.clicked.connect(self._on_start_resource_clicked)

        btn_row.addWidget(self._test_coin_btn)
        btn_row.addWidget(self._test_diamond_btn)
        btn_row.addWidget(self._test_treats_btn)
        btn_row.addWidget(self._test_piggy_btn)
        btn_row.addWidget(self._test_all_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        layout.addWidget(QFrame(objectName="Divider"))

        self._sec2_title = QLabel(objectName="SectionTitle")
        self._sec2_desc = QLabel(objectName="SectionDesc")
        layout.addWidget(self._sec2_title)
        layout.addWidget(self._sec2_desc)

        track_row = QHBoxLayout()
        track_row.setSpacing(8)

        self._track_input = QLineEdit()
        self._track_btn = QPushButton(objectName="TrackButton")
        self._track_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._track_btn.clicked.connect(self._on_start_track_clicked)

        track_row.addWidget(self._track_input, 1)
        track_row.addWidget(self._track_btn, 0)
        layout.addLayout(track_row)

        layout.addStretch(1)

    def _build_tab_log(self, parent: QWidget) -> None:
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        log_bar = QHBoxLayout()
        self._log_title = QLabel(objectName="SectionTitle")
        log_bar.addWidget(self._log_title)
        log_bar.addStretch(1)

        self._clear_btn = QPushButton(objectName="SubActionButton")
        self._clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._clear_btn.clicked.connect(self._on_clear_log_clicked)
        log_bar.addWidget(self._clear_btn)

        layout.addLayout(log_bar)

        self._log_view = QTextBrowser(objectName="LogView")
        self._log_view.setAccessibleName("运行日志")
        self._log_view.setReadOnly(True)
        self._log_view.setOpenExternalLinks(False)
        self._log_view.document().setMaximumBlockCount(MAX_LOG_BLOCKS)
        self._log_view.setFont(QFont("Consolas", 9))
        layout.addWidget(self._log_view, 1)

    def _build_tab_settings(self, parent: QWidget) -> None:
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(14)

        lang_box = QHBoxLayout()
        lang_box.setSpacing(10)
        self._settings_lang_title = QLabel(objectName="SectionTitle")
        lang_box.addWidget(self._settings_lang_title)

        self._settings_lang_combo = QComboBox(objectName="LangCombo")
        self._settings_lang_combo.addItem("English", "en")
        self._settings_lang_combo.addItem("中文", "zh")
        current_idx = 0 if self._lang == "en" else 1
        self._settings_lang_combo.setCurrentIndex(current_idx)
        self._settings_lang_combo.currentIndexChanged.connect(self._on_settings_lang_changed)
        lang_box.addWidget(self._settings_lang_combo)
        lang_box.addStretch(1)
        layout.addLayout(lang_box)

        layout.addWidget(QFrame(objectName="Divider"))

        self._sec1_settings_title = QLabel(objectName="SectionTitle")
        layout.addWidget(self._sec1_settings_title)

        self._scan_first_box = QCheckBox(objectName="ScanFirstBox")
        self._scan_first_box.setAccessibleName("加速配对：先翻开全部卡牌再逐对收牌")
        self._scan_first_box.setChecked(self._initial_scan_first)
        self._scan_first_box.setCursor(Qt.CursorShape.PointingHandCursor)
        self._scan_first_box.toggled.connect(self._save_settings)
        layout.addWidget(self._scan_first_box)

        layout.addWidget(QFrame(objectName="Divider"))

        self._sec2_settings_title = QLabel(objectName="SectionTitle")
        layout.addWidget(self._sec2_settings_title)

        self._reset_map_box = QCheckBox(objectName="ResetMapBox")
        self._reset_map_box.setChecked(self._initial_reset_map)
        self._reset_map_box.setCursor(Qt.CursorShape.PointingHandCursor)
        self._reset_map_box.toggled.connect(self._on_reset_map_toggled)
        layout.addWidget(self._reset_map_box)

        brake_container = QVBoxLayout()
        brake_container.setContentsMargins(20, 0, 0, 0)
        brake_container.setSpacing(6)

        self._brake_group = QButtonGroup(self)
        self._brake_dynamic_radio = QRadioButton(objectName="BrakeDynamicRadio")
        self._brake_dynamic_radio.setCursor(Qt.CursorShape.PointingHandCursor)
        self._brake_dynamic_radio.setChecked(self._initial_brake_mode != "first_island")
        self._brake_dynamic_radio.toggled.connect(self._on_brake_radio_toggled)
        self._brake_group.addButton(self._brake_dynamic_radio)
        brake_container.addWidget(self._brake_dynamic_radio)

        first_row = QHBoxLayout()
        first_row.setSpacing(8)
        self._brake_first_island_radio = QRadioButton(objectName="BrakeFirstIslandRadio")
        self._brake_first_island_radio.setCursor(Qt.CursorShape.PointingHandCursor)
        self._brake_first_island_radio.setChecked(self._initial_brake_mode == "first_island")
        self._brake_first_island_radio.toggled.connect(self._on_brake_radio_toggled)
        self._brake_group.addButton(self._brake_first_island_radio)
        first_row.addWidget(self._brake_first_island_radio)

        self._first_island_edit = QLineEdit(self._saved_first_island_name, objectName="FirstIslandEdit")
        self._first_island_edit.setFixedWidth(160)
        self._first_island_edit.textChanged.connect(self._on_first_island_text_changed)
        first_row.addWidget(self._first_island_edit)

        self._first_island_btn = QPushButton("✔", objectName="FirstIslandConfirmBtn")
        self._first_island_btn.setFixedSize(28, 24)
        self._first_island_btn.clicked.connect(self._on_first_island_save_clicked)
        first_row.addWidget(self._first_island_btn)
        first_row.addStretch(1)
        brake_container.addLayout(first_row)

        layout.addLayout(brake_container)
        self._update_brake_controls_state()

        layout.addStretch(1)

    def _on_reset_map_toggled(self, checked: bool) -> None:
        self._update_brake_controls_state()
        self._save_settings()

    def _on_brake_radio_toggled(self, checked: bool) -> None:
        self._update_brake_controls_state()
        self._save_settings()

    def _on_first_island_text_changed(self, text: str) -> None:
        self._update_first_island_btn_state()

    def _on_first_island_save_clicked(self) -> None:
        clean = self._first_island_edit.text().strip()
        if clean:
            self._saved_first_island_name = clean
            self._first_island_edit.setText(clean)
            self._update_first_island_btn_state()
            self._save_settings()

    def _update_first_island_btn_state(self) -> None:
        is_first_mode = self._brake_first_island_radio.isChecked() and self._reset_map_box.isChecked()
        cur_text = self._first_island_edit.text().strip()
        is_modified = is_first_mode and (cur_text != self._saved_first_island_name) and bool(cur_text)
        if is_modified:
            self._first_island_btn.setEnabled(True)
            self._first_island_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self._first_island_btn.setStyleSheet(
                "QPushButton#FirstIslandConfirmBtn { background-color: #2e7d32; color: #ffffff; font-weight: bold; border-radius: 4px; border: 1px solid #4caf50; }"
            )
        else:
            self._first_island_btn.setEnabled(False)
            self._first_island_btn.setCursor(Qt.CursorShape.ArrowCursor)
            self._first_island_btn.setStyleSheet(
                "QPushButton#FirstIslandConfirmBtn { background-color: #23272e; color: #555e6c; font-weight: bold; border-radius: 4px; border: 1px solid #333842; }"
            )

    def _update_brake_controls_state(self) -> None:
        reset_enabled = self._reset_map_box.isChecked()
        self._brake_dynamic_radio.setEnabled(reset_enabled)
        self._brake_first_island_radio.setEnabled(reset_enabled)
        first_mode = reset_enabled and self._brake_first_island_radio.isChecked()
        self._first_island_edit.setEnabled(first_mode)
        self._update_first_island_btn_state()

    def _retranslate_ui(self) -> None:
        t = TRANSLATIONS.get(self._lang, TRANSLATIONS["en"])

        self.setWindowTitle(t["window_title"])
        self._title_label.setText(t["app_title"])
        self._subtitle_label.setText(t["app_subtitle"])
        self._status_label.setText(t["status_idle"])
        self._stats_label.setText(t["stats_idle"])

        self._stop_button.setText(t["stop"])
        self._stop_button.setToolTip(t["stop_tip"])

        self._tabs.setTabText(0, t["tab_main"])
        self._tabs.setTabText(1, t["tab_test"])
        self._tabs.setTabText(2, t["tab_log"])
        self._tabs.setTabText(3, t["tab_settings"])

        self._tour_button.setText(t["start_tour"])
        self._tour_button.setToolTip(t["start_tour_tip"])
        self._minigame_button.setText(t["start_memory"])
        self._minigame_button.setToolTip(t["start_memory_tip"])
        self._bl_title.setText(t["blacklist_title"])
        self._blacklist_table.set_language(self._lang)

        self._sec1_title.setText(t["test_sec1_title"])
        self._sec1_desc.setText(t["test_sec1_desc"])
        self._test_coin_btn.setText(t["test_coin"])
        self._test_coin_btn.setToolTip(t["test_coin_tip"])
        self._test_diamond_btn.setText(t["test_diamond"])
        self._test_diamond_btn.setToolTip(t["test_diamond_tip"])
        self._test_treats_btn.setText(t["test_treats"])
        self._test_treats_btn.setToolTip(t["test_treats_tip"])
        self._test_piggy_btn.setText(t["test_piggy"])
        self._test_piggy_btn.setToolTip(t["test_piggy_tip"])
        self._test_all_btn.setText(t["test_all"])
        self._test_all_btn.setToolTip(t["test_all_tip"])

        self._sec2_title.setText(t["test_sec2_title"])
        self._sec2_desc.setText(t["test_sec2_desc"])
        self._track_input.setPlaceholderText(t["track_placeholder"])
        self._track_btn.setText(t["track_btn"])

        self._log_title.setText(t["log_title"])
        self._clear_btn.setText(t["clear_log"])

        self._settings_lang_title.setText(t["settings_lang_title"])
        self._sec1_settings_title.setText(t["settings_memory_title"])
        self._scan_first_box.setText(t["scan_first"])
        self._scan_first_box.setToolTip(t["scan_first_tip"])
        self._sec2_settings_title.setText(t["settings_map_title"])
        self._reset_map_box.setText(t["reset_map_box"])
        self._reset_map_box.setToolTip(t["reset_map_box_tip"])
        self._brake_dynamic_radio.setText(t["brake_dynamic"])
        self._brake_dynamic_radio.setToolTip(t["brake_dynamic_tip"])
        self._brake_first_island_radio.setText(t["brake_first_island"])
        self._brake_first_island_radio.setToolTip(t["brake_first_island_tip"])
        self._first_island_edit.setPlaceholderText(t["first_island_placeholder"])
        self._first_island_btn.setToolTip(t["first_island_save_tip"])

        self._tour_button.setShortcut("F7")
        self._minigame_button.setShortcut("F11")
        self._stop_button.setShortcut("F10")
        self._test_coin_btn.setShortcut("F9")
        self._test_all_btn.setShortcut("F8")

    def _on_language_changed(self, index: int) -> None:
        new_lang = self._lang_combo.itemData(index)
        if new_lang != self._lang:
            self._lang = new_lang
            if self._settings_lang_combo.currentIndex() != index:
                self._settings_lang_combo.blockSignals(True)
                self._settings_lang_combo.setCurrentIndex(index)
                self._settings_lang_combo.blockSignals(False)
            self._retranslate_ui()
            self._save_settings()

    def _on_settings_lang_changed(self, index: int) -> None:
        new_lang = self._settings_lang_combo.itemData(index)
        if new_lang != self._lang:
            self._lang = new_lang
            if self._lang_combo.currentIndex() != index:
                self._lang_combo.blockSignals(True)
                self._lang_combo.setCurrentIndex(index)
                self._lang_combo.blockSignals(False)
            self._retranslate_ui()
            self._save_settings()

    def _save_settings(self) -> None:
        brake_mode = "first_island" if hasattr(self, "_brake_first_island_radio") and self._brake_first_island_radio.isChecked() else "dynamic"
        data = {
            "language": self._lang,
            "blacklist": self._blacklist_table.get_blacklist(),
            "scan_first": self._scan_first_box.isChecked(),
            "reset_map_to_top": self._reset_map_box.isChecked(),
            "init_brake_mode": brake_mode,
            "first_island_name": getattr(self, "_saved_first_island_name", "Plant Island"),
        }
        save_user_settings(data)

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
        t = TRANSLATIONS.get(self._lang, TRANSLATIONS["en"])
        target = self._track_input.text().strip()
        if not target:
            self._append_log("WARN", t["warn_empty_target"])
            return
        self._start(MODE_TRACK_ISLAND)

    @Slot()
    def _on_clear_log_clicked(self) -> None:
        self._log_view.clear()

    def _start(self, mode: str) -> None:
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
                worker.set_reset_map_to_top(self._reset_map_box.isChecked())
                brake_mode = "first_island" if self._brake_first_island_radio.isChecked() else "dynamic"
                worker.set_map_init_brake(brake_mode, self._saved_first_island_name)
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
        self._tour_button.setEnabled(not running)
        self._minigame_button.setEnabled(not running)
        self._blacklist_table.set_enabled(not running)

        self._test_coin_btn.setEnabled(not running)
        self._test_diamond_btn.setEnabled(not running)
        self._test_treats_btn.setEnabled(not running)
        self._test_piggy_btn.setEnabled(not running)
        self._test_all_btn.setEnabled(not running)
        self._track_input.setEnabled(not running)
        self._track_btn.setEnabled(not running)

        self._scan_first_box.setEnabled(not running)
        self._reset_map_box.setEnabled(not running)
        if running:
            self._brake_dynamic_radio.setEnabled(False)
            self._brake_first_island_radio.setEnabled(False)
            self._first_island_edit.setEnabled(False)
            self._first_island_btn.setEnabled(False)
        else:
            self._update_brake_controls_state()
        self._stop_button.setEnabled(running)

    @Slot(str, str)
    def _append_log(self, level: str, message: str) -> None:
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
        t = TRANSLATIONS.get(self._lang, TRANSLATIONS["en"])
        self._status_label.setText(f"{t['status_prefix']}{state}")

    @Slot(int, int)
    def _on_stats_changed(self, first: int, second: int) -> None:
        t = TRANSLATIONS.get(self._lang, TRANSLATIONS["en"])
        if self._mode == MODE_MEMORY:
            self._stats_label.setText(t["stats_memory"].format(first=first, second=second))
        elif self._mode == MODE_MAP_TOUR:
            self._stats_label.setText(t["stats_tour"].format(first=first, second=second))
        elif self._mode == MODE_TRACK_ISLAND:
            self._stats_label.setText(t["stats_track"].format(first=first, second=second))
        else:
            self._stats_label.setText(t["stats_generic"].format(first=first, second=second))

    @Slot()
    def _on_worker_finished(self) -> None:
        self._set_running(False)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._save_settings()
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.stop()
            if not worker.wait(SHUTDOWN_WAIT_MS):
                worker.terminate()
                worker.wait(500)
        event.accept()

