"""UI wiring: button state, log rendering, and worker lifecycle.

Safety note, and it is not only about determinism
------------------------------------------------
These tests start real workers, and both of them post real mouse messages to
whichever window matches the configured title. If they ran with the default title
while the game happened to be open, the suite would be driving the live game.

So the fixture points them at a title no process can own. Each worker can then
only ever exercise its "window not found" branch, which is also exactly what a
user sees when pressing start before launching the game: it must degrade
gracefully rather than freeze or crash.

Two workers, one interface
--------------------------
``BotEngine`` (coins) and ``MemoryEngine`` (minigame) have different loops but
emit the same three signals, which is what lets the window hold either in
``_worker`` and stop it the same way. These tests cover both through that single
path, plus the one place the wording genuinely diverges: the statistics label,
where the same ``stats_changed`` numbers mean rounds and clicks for one worker and
levels and turns for the other.

The flip-order switch is checked at the seam, not by its effect
--------------------------------------------------------------
What the two flip orders *do* is the control loop's business and is covered
there. What matters here is that the checkbox reaches the worker it configures:
the window reads it once, at start, and hands it over as ``RunnerParams``. So
these tests assert on the parameters the constructed engine is holding, which is
the only thing the UI is responsible for.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import QTimer

from config import AppConfig, WindowConfig
from core.bot_engine import BotEngine
from core.minigames.memory_engine import MemoryEngine
from ui.main_window import MODE_COIN, MODE_MEMORY, MainWindow

#: A window title no real process will ever own.
ABSENT_WINDOW_TITLE = "__msm_helper_test_window_that_does_not_exist__"


@pytest.fixture
def test_config() -> AppConfig:
    return AppConfig(window=WindowConfig(title=ABSENT_WINDOW_TITLE))


@pytest.fixture
def window(qapp, test_config):
    win = MainWindow(config=test_config)
    win.show()
    yield win
    win.close()


class TestInitialState:
    def test_both_starts_are_available_and_stop_is_not(self, window):
        assert window._start_button.isEnabled()
        assert window._minigame_button.isEnabled()
        assert not window._stop_button.isEnabled()

    def test_log_view_is_read_only(self, window):
        assert window._log_view.isReadOnly()

    def test_startup_message_is_rendered(self, window):
        assert window._log_view.toPlainText().strip()

    def test_stylesheet_is_applied(self, window):
        assert window.styleSheet()

    def test_no_worker_exists_before_start(self, window):
        assert window._worker is None

    def test_buttons_expose_accessible_names(self, window):
        assert window._start_button.accessibleName()
        assert window._minigame_button.accessibleName()
        assert window._stop_button.accessibleName()
        assert window._scan_first_box.accessibleName()
        assert window._log_view.accessibleName()

    def test_log_view_has_a_bounded_backlog(self, window):
        """An unbounded log would grow without limit during long sessions."""
        assert window._log_view.document().maximumBlockCount() > 0

    def test_configured_window_title_is_surfaced(self, window):
        assert ABSENT_WINDOW_TITLE in window._log_view.toPlainText()

    def test_the_minigame_precondition_is_stated_up_front(self, window):
        """The slot map can only be built from a full board, so the user has to
        be told before pressing the button, not after it fails."""
        assert "开局" in window._log_view.toPlainText()
        assert window._minigame_button.toolTip()

    def test_the_flip_order_switch_starts_off_and_explains_itself(self, window):
        """Off by default: the opportunistic order is the measured one, and a
        switch whose trade-off is unresolved should not be opt-out."""
        assert window._scan_first_box.isEnabled()
        assert not window._scan_first_box.isChecked()
        assert window._scan_first_box.toolTip()


class TestStartStopCycle:
    def test_full_cycle(self, qapp, window):
        collected = {}

        def press_start():
            window._start_button.click()
            collected["running"] = window._worker.isRunning()
            collected["start_disabled"] = not window._start_button.isEnabled()
            collected["minigame_disabled"] = not window._minigame_button.isEnabled()
            collected["stop_enabled"] = window._stop_button.isEnabled()

        def press_stop():
            window._stop_button.click()
            # Stopping must not block the UI thread waiting for the worker.
            collected["stop_disabled_immediately"] = not window._stop_button.isEnabled()

        def finish():
            collected["stopped"] = not window._worker.isRunning()
            collected["start_reenabled"] = window._start_button.isEnabled()
            collected["minigame_reenabled"] = window._minigame_button.isEnabled()
            qapp.quit()

        QTimer.singleShot(100, press_start)
        QTimer.singleShot(1200, press_stop)
        QTimer.singleShot(3500, finish)
        qapp.exec()

        assert collected["running"]
        assert collected["start_disabled"]
        assert collected["minigame_disabled"]
        assert collected["stop_enabled"]
        assert collected["stop_disabled_immediately"]
        assert collected["stopped"]
        assert collected["start_reenabled"]
        assert collected["minigame_reenabled"]

    def test_engine_output_reaches_the_log_widget(self, qapp, window):
        QTimer.singleShot(100, window._start_button.click)
        QTimer.singleShot(1500, window._stop_button.click)
        QTimer.singleShot(3800, qapp.quit)
        qapp.exec()

        text = window._log_view.toPlainText()
        assert "挂机引擎已启动" in text
        assert "NotInExclusionZones" in text  # rule chain announcement

    def test_missing_game_window_is_reported_not_fatal(self, qapp, window):
        QTimer.singleShot(100, window._start_button.click)
        QTimer.singleShot(1800, window._stop_button.click)
        QTimer.singleShot(4200, qapp.quit)
        qapp.exec()

        text = window._log_view.toPlainText()
        assert "未找到窗口" in text
        # It must never have reached the click stage.
        assert "已点击" not in text
        assert window._start_button.isEnabled()

    def test_double_start_does_not_spawn_a_second_worker(self, qapp, window):
        seen = {}

        def start_twice():
            window._start_button.click()
            first = window._worker
            window._on_start_coin_clicked()  # simulate a stray second trigger
            seen["same_worker"] = window._worker is first

        QTimer.singleShot(100, start_twice)
        QTimer.singleShot(1200, window._stop_button.click)
        QTimer.singleShot(3500, qapp.quit)
        qapp.exec()

        assert seen["same_worker"]

    def test_stop_without_start_is_harmless(self, window):
        window._on_stop_clicked()
        assert window._worker is None


class TestMinigameWorker:
    """The minigame button, through the same single worker slot."""

    def test_pressing_it_starts_a_memory_engine(self, qapp, window):
        seen = {}

        def press():
            window._minigame_button.click()
            seen["type"] = type(window._worker)
            seen["mode"] = window._mode
            seen["coin_disabled"] = not window._start_button.isEnabled()

        QTimer.singleShot(100, press)
        QTimer.singleShot(2500, qapp.quit)
        qapp.exec()

        assert seen["type"] is MemoryEngine
        assert seen["mode"] == MODE_MEMORY
        assert seen["coin_disabled"]

    def test_missing_game_window_is_reported_not_fatal(self, qapp, window):
        """The same graceful degradation the coin engine has.

        With no window there is nothing to capture, so the run must end with a
        message rather than an exception -- and above all without clicking.
        """
        QTimer.singleShot(100, window._minigame_button.click)
        QTimer.singleShot(2500, qapp.quit)
        qapp.exec()

        text = window._log_view.toPlainText()
        assert "找不到游戏窗口" in text
        assert "已点击槽位" not in text
        assert window._minigame_button.isEnabled()
        assert window._start_button.isEnabled()

    def test_switching_between_tasks_reuses_the_same_slot(self, qapp, window):
        """Coins then minigame: one attribute, one stop button, no leaks."""
        seen = {}

        def start_coin():
            window._start_button.click()
            seen["first"] = window._worker

        def stop_coin():
            window._stop_button.click()

        def start_minigame():
            window._minigame_button.click()
            seen["second"] = window._worker
            seen["different"] = seen["second"] is not seen["first"]
            seen["first_stopped"] = not seen["first"].isRunning()

        QTimer.singleShot(100, start_coin)
        QTimer.singleShot(1200, stop_coin)
        QTimer.singleShot(3600, start_minigame)
        QTimer.singleShot(6000, qapp.quit)
        qapp.exec()

        assert seen["different"]
        assert seen["first_stopped"]
        assert isinstance(seen["second"], MemoryEngine)

    def test_a_coin_run_still_builds_a_bot_engine(self, qapp, window):
        seen = {}

        def press():
            window._start_button.click()
            seen["type"] = type(window._worker)
            seen["mode"] = window._mode

        QTimer.singleShot(100, press)
        QTimer.singleShot(1200, window._stop_button.click)
        QTimer.singleShot(3500, qapp.quit)
        qapp.exec()

        assert seen["type"] is BotEngine
        assert seen["mode"] == MODE_COIN


class TestFlipOrderSwitch:
    """The one setting the window owns has to arrive at the worker."""

    def test_unchecked_gives_the_opportunistic_order(self, qapp, window):
        seen = {}

        def press():
            window._scan_first_box.setChecked(False)
            window._minigame_button.click()
            seen["scan_first"] = window._worker._params.scan_first

        QTimer.singleShot(100, press)
        QTimer.singleShot(2500, qapp.quit)
        qapp.exec()

        assert seen["scan_first"] is False

    def test_checked_gives_the_scan_first_order(self, qapp, window):
        seen = {}

        def press():
            window._scan_first_box.setChecked(True)
            window._minigame_button.click()
            seen["scan_first"] = window._worker._params.scan_first

        QTimer.singleShot(100, press)
        QTimer.singleShot(2500, qapp.quit)
        qapp.exec()

        assert seen["scan_first"] is True

    def test_it_is_locked_while_a_run_is_in_flight(self, qapp, window):
        """It is read once, at start. Leaving it clickable would promise an
        effect it cannot have."""
        seen = {}

        def press():
            window._minigame_button.click()
            seen["locked"] = not window._scan_first_box.isEnabled()

        def after():
            seen["unlocked"] = window._scan_first_box.isEnabled()
            qapp.quit()

        QTimer.singleShot(100, press)
        QTimer.singleShot(2500, after)
        qapp.exec()

        assert seen["locked"]
        assert seen["unlocked"]

    def test_a_coin_run_is_unaffected_by_it(self, qapp, window):
        """The switch belongs to the minigame; the patrol must not read it."""
        seen = {}

        def press():
            window._scan_first_box.setChecked(True)
            window._start_button.click()
            seen["type"] = type(window._worker)

        QTimer.singleShot(100, press)
        QTimer.singleShot(1200, window._stop_button.click)
        QTimer.singleShot(3500, qapp.quit)
        qapp.exec()

        assert seen["type"] is BotEngine


class TestLogRendering:
    @pytest.mark.parametrize(
        "level", ["INFO", "SUCCESS", "WARN", "ERROR", "UNKNOWN_LEVEL"]
    )
    def test_every_level_renders(self, window, level):
        window._append_log(level, "message body")
        assert "message body" in window._log_view.toPlainText()

    def test_html_in_a_message_is_escaped_not_interpreted(self, window):
        """A stray angle bracket must never be able to inject markup."""
        window._append_log("INFO", "<b>not bold</b> & <script>")
        text = window._log_view.toPlainText()
        assert "<b>not bold</b>" in text
        assert "<script>" in text

    def test_entries_carry_a_timestamp(self, window):
        window._append_log("INFO", "timed entry")
        # Rendered as [HH:MM:SS] in front of the message.
        assert "[" in window._log_view.toPlainText()

    def test_state_label_updates(self, window):
        window._on_state_changed("搜索目标")
        assert "搜索目标" in window._status_label.text()

    def test_stats_label_reads_as_rounds_and_clicks_for_coins(self, window):
        window._mode = MODE_COIN
        window._on_stats_changed(7, 3)

        text = window._stats_label.text()
        assert "7" in text and "3" in text
        assert "轮次" in text

    def test_stats_label_reads_as_levels_and_turns_for_the_minigame(self, window):
        """Same signal, different meaning -- the one place the two diverge."""
        window._mode = MODE_MEMORY
        window._on_stats_changed(4, 21)

        text = window._stats_label.text()
        assert "4" in text and "21" in text
        assert "关" in text


class TestShutdown:
    def test_closing_while_running_stops_the_worker(self, qapp, test_config):
        win = MainWindow(config=test_config)
        win.show()
        result = {}

        def start_then_close():
            win._start_button.click()
            worker = win._worker
            win.close()  # closeEvent must join the worker
            result["stopped"] = not worker.isRunning()
            qapp.quit()

        QTimer.singleShot(300, start_then_close)
        qapp.exec()

        assert result["stopped"]

    def test_closing_while_the_minigame_runs_stops_it_too(self, qapp, test_config):
        win = MainWindow(config=test_config)
        win.show()
        result = {}

        def start_then_close():
            win._minigame_button.click()
            worker = win._worker
            win.close()
            result["stopped"] = not worker.isRunning()
            qapp.quit()

        QTimer.singleShot(300, start_then_close)
        qapp.exec()

        assert result["stopped"]
