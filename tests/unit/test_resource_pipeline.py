"""Tests for resource collection extensions and pipeline orchestration."""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from config import (
    TARGET_COIN,
    TARGET_DIAMOND,
    TARGET_MODAL_CONFIRM,
    TARGET_PIGGY_BANK,
    TARGET_TREATS,
    AppConfig,
    LoopConfig,
    SafetyConfig,
    VisionConfig,
    WindowConfig,
)
from core.bot_engine import BotEngine, EngineMode
from core.vision_agent import BaseVisionAgent, Detection, VisionAgent, check_piggy_status
from ui.main_window import (
    MODE_ALL_RESOURCES,
    MODE_COIN,
    MODE_DIAMOND,
    MODE_PIGGY,
    MODE_TREATS,
    MainWindow,
)


class TestPiggyStatus:
    def test_empty_or_none_returns_false(self):
        assert check_piggy_status(None) is False
        assert check_piggy_status(np.zeros((0, 0, 3), dtype=np.uint8)) is False

    def test_dim_piggy_detected_as_inactive(self):
        # Create a dim crop (Value in HSV ~ 90)
        crop = np.full((50, 50, 3), 90, dtype=np.uint8)
        assert check_piggy_status(crop, min_v=130.0) is False

    def test_bright_piggy_detected_as_active(self):
        # Create a bright crop (Value in HSV ~ 170)
        crop = np.full((50, 50, 3), 170, dtype=np.uint8)
        assert check_piggy_status(crop, min_v=130.0) is True


class TestVisionTargetThresholds:
    def test_falls_back_to_match_threshold_for_unknown_target(self):
        agent = VisionAgent(
            match_threshold=0.82,
            target_thresholds={"diamond": 0.70},
            use_fake_detection=True,
        )
        assert agent.threshold_for("coin") == 0.82
        assert agent.threshold_for("diamond") == 0.70

    def test_returns_match_threshold_when_no_thresholds_dict(self):
        agent = VisionAgent(match_threshold=0.80, use_fake_detection=True)
        assert agent.threshold_for("coin") == 0.80
        assert agent.threshold_for("diamond") == 0.80


class MockGameWindow:
    def __init__(self, frame=None, attached=True):
        self.frame = frame if frame is not None else np.full((768, 1024, 3), 150, dtype=np.uint8)
        self.attached = attached
        self.detach_called = False

    def ensure_attached(self) -> bool:
        return self.attached

    def capture(self):
        return self.frame.copy() if self.frame is not None else None

    def detach(self):
        self.detach_called = True


class MockActionAgent:
    def __init__(self):
        self.clicks = []
        self.scale = 1.0

    def set_scale(self, scale: float):
        self.scale = scale

    def click(self, x, y):
        self.clicks.append((x, y))
        return True


class ScriptedVision(BaseVisionAgent):
    def __init__(self, script: dict[str, list[list[Detection]]]):
        self.script = script
        self.calls: dict[str, int] = {k: 0 for k in script}

    def detect(self, target_name: str, screenshot: np.ndarray) -> list[Detection]:
        if target_name not in self.script:
            return []
        idx = self.calls[target_name]
        responses = self.script[target_name]
        if idx < len(responses):
            res = responses[idx]
        else:
            res = []
        self.calls[target_name] += 1
        return res


class TestPiggyStage:
    def test_piggy_dim_is_skipped_without_click(self):
        dim_frame = np.full((768, 1024, 3), 90, dtype=np.uint8)
        piggy_det = Detection(TARGET_PIGGY_BANK, 920, 600, 70, 70, 0.9)
        vision = ScriptedVision({TARGET_PIGGY_BANK: [[piggy_det]]})

        engine = BotEngine(
            config=AppConfig(
                window=WindowConfig(title="mock"),
                loop=LoopConfig(tick_interval=(0.01, 0.01)),
            ),
            vision_agent=vision,
            mode=EngineMode.PIGGY,
        )
        engine._window = MockGameWindow(frame=dim_frame)
        engine._action = MockActionAgent()

        ok = engine._run_piggy_stage()
        assert ok is False
        assert len(engine._action.clicks) == 0

    def test_piggy_bright_clicks_and_confirms(self):
        bright_frame = np.full((768, 1024, 3), 180, dtype=np.uint8)
        piggy_det = Detection(TARGET_PIGGY_BANK, 920, 600, 70, 70, 0.9)
        confirm_det = Detection(TARGET_MODAL_CONFIRM, 340, 520, 70, 70, 0.95)

        vision = ScriptedVision({
            TARGET_PIGGY_BANK: [[piggy_det]],
            TARGET_MODAL_CONFIRM: [
                [],             # Initial check: modal not open yet
                [confirm_det],  # Modal opens after clicking piggy
                [],             # Modal closes after clicking confirm
            ],
        })

        engine = BotEngine(
            config=AppConfig(
                window=WindowConfig(title="mock"),
                vision=VisionConfig(modal_timeout=1.0),
                loop=LoopConfig(tick_interval=(0.01, 0.01)),
            ),
            vision_agent=vision,
            mode=EngineMode.PIGGY,
        )
        engine._window = MockGameWindow(frame=bright_frame)
        engine._action = MockActionAgent()

        ok = engine._run_piggy_stage()
        assert ok is True
        assert len(engine._action.clicks) == 2
        assert engine._action.clicks[0] == piggy_det.center
        assert engine._action.clicks[1] == confirm_det.center

    def test_piggy_modal_already_open_confirms_directly(self):
        frame = np.full((768, 1024, 3), 100, dtype=np.uint8)
        confirm_det = Detection(TARGET_MODAL_CONFIRM, 340, 520, 70, 70, 0.95)

        vision = ScriptedVision({
            TARGET_MODAL_CONFIRM: [
                [confirm_det],  # Initial check: modal already open!
                [confirm_det],  # During confirm loop
                [],             # Closed after click
            ],
        })

        engine = BotEngine(
            config=AppConfig(
                window=WindowConfig(title="mock"),
                vision=VisionConfig(modal_timeout=1.0),
                loop=LoopConfig(tick_interval=(0.01, 0.01)),
            ),
            vision_agent=vision,
            mode=EngineMode.PIGGY,
        )
        engine._window = MockGameWindow(frame=frame)
        engine._action = MockActionAgent()

        ok = engine._run_piggy_stage()
        assert ok is True
        assert len(engine._action.clicks) == 1
        assert engine._action.clicks[0] == confirm_det.center


class TestCollectTargetUntilEmpty:
    def test_stops_after_consecutive_empty_rounds(self):
        target_det = Detection(TARGET_DIAMOND, 500, 300, 50, 50, 0.9)
        vision = ScriptedVision({
            TARGET_DIAMOND: [
                [target_det],
                [],
                [],
            ]
        })

        engine = BotEngine(
            config=AppConfig(
                window=WindowConfig(title="mock"),
                safety=SafetyConfig(confirm_frames=1, verify_after_click=False),
                loop=LoopConfig(tick_interval=(0.01, 0.01)),
            ),
            vision_agent=vision,
            mode=EngineMode.DIAMOND,
        )
        engine._window = MockGameWindow()
        engine._action = MockActionAgent()

        clicks = engine._collect_target_until_empty(
            TARGET_DIAMOND, max_rounds=5, max_consecutive_empty=2
        )
        assert clicks == 1
        assert len(engine._action.clicks) == 1


class TestAllResourcesPipeline:
    def test_pipeline_executes_all_four_stages_in_order(self):
        call_order = []

        engine = BotEngine(
            config=AppConfig(window=WindowConfig(title="mock")),
            mode=EngineMode.ALL_RESOURCES,
        )

        def mock_piggy_stage():
            call_order.append("piggy")
            return True

        def mock_collect(target_name, max_rounds, max_consecutive_empty=4):
            call_order.append(f"collect_{target_name}")
            return 2

        engine._run_piggy_stage = mock_piggy_stage
        engine._collect_target_until_empty = mock_collect

        engine._run_all_resources()

        assert call_order == [
            "piggy",
            f"collect_{TARGET_DIAMOND}",
            f"collect_{TARGET_TREATS}",
            f"collect_{TARGET_COIN}",
        ]


class TestUIResourceControls:
    @pytest.fixture
    def app(self):
        return QApplication.instance() or QApplication([])

    @pytest.fixture
    def window(self, app):
        win = MainWindow(config=AppConfig(window=WindowConfig(title="mock_ui")))
        win.show()
        yield win
        win.close()

    def test_all_resource_buttons_are_available_initially(self, window):
        assert window._resource_button.isEnabled()
        assert window._diamond_button.isEnabled()
        assert window._treats_button.isEnabled()
        assert window._piggy_button.isEnabled()
        assert window._start_button.isEnabled()
        assert not window._stop_button.isEnabled()

    def test_resource_button_has_shortcut_f8(self, window):
        assert window._resource_button.shortcut().toString() == "F8"

    def test_set_running_toggles_all_resource_buttons(self, window):
        window._set_running(True)
        assert not window._resource_button.isEnabled()
        assert not window._diamond_button.isEnabled()
        assert not window._treats_button.isEnabled()
        assert not window._piggy_button.isEnabled()
        assert not window._start_button.isEnabled()
        assert window._stop_button.isEnabled()

        window._set_running(False)
        assert window._resource_button.isEnabled()
        assert window._diamond_button.isEnabled()
        assert window._treats_button.isEnabled()
        assert window._piggy_button.isEnabled()
        assert window._start_button.isEnabled()
        assert not window._stop_button.isEnabled()
