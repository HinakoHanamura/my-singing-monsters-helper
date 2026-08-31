"""End-to-end decision pipeline, with real template matching and a stubbed window.

What is real here: the vision layer (actual cv2 template matching), the rule
chain, the click guard, and the engine's orchestration and threading.

What is stubbed: the game window (serves a fixed synthetic frame) and the action
layer (records clicks instead of posting Windows messages). Stubbing those is
what makes the test deterministic and runnable without the game - and keeps the
suite from ever driving a live game window.

Performance note: the shared engine run is module scoped on purpose. Re-running
a multi-second session once per assertion would push this file past two minutes
for no extra coverage. One run, many assertions; only tests that genuinely need
a different configuration start their own engine.

Assertion style: behavioural facts are checked against engine and guard state,
not by searching the log transcript. Log text carries status banners that happen
to contain the same words as the events being tested, so substring matching over
the whole transcript passes for the wrong reasons.
"""

from __future__ import annotations

import threading

import pytest

from config import AppConfig, ClickConfig, LoopConfig, SafetyConfig, VisionConfig
from core.bot_engine import BotEngine, BotState
from tests.synthetic import (
    REFERENCE_SIZE,
    make_coin_template,
    make_scene,
    render_at,
    scale_point,
    write_template,
)

WINDOW = (1280, 720)

# Reference-resolution positions with deliberately different fates.
COIN_VALID = (900, 400)     # above a textured blob -> should be clicked
COIN_FLAT = (1300, 620)     # on empty ground -> rejected by TextureBelow
COIN_IN_PANEL = (200, 450)  # inside the left UI panel -> rejected by exclusion zone


class StubWindow:
    """Serves one fixed frame and counts how often it was read."""

    hwnd = 4242

    def __init__(self, frame):
        self.frame = frame
        self.captures = 0
        self.detached = False

    def ensure_attached(self) -> bool:
        return True

    def client_size(self):
        return self.frame.shape[1], self.frame.shape[0]

    def capture(self):
        self.captures += 1
        return self.frame.copy()

    def detach(self) -> None:
        self.detached = True


class DeadWindow(StubWindow):
    """Attaches but never yields a frame, as a minimized window would."""

    def capture(self):
        self.captures += 1
        return None


class RecordingAction:
    """Records clicks; never touches the OS."""

    def __init__(self):
        self.clicks = []
        self.threads = set()
        self.scales = []

    def set_scale(self, scale):
        self.scales.append(scale)

    def click(self, x, y) -> bool:
        self.clicks.append((x, y))
        self.threads.add(threading.get_ident())
        return True


class FailingAction(RecordingAction):
    """Simulates message delivery failing."""

    def click(self, x, y) -> bool:
        super().click(x, y)
        return False


class EngineHarness:
    """Runs the engine on its own thread and collects everything it emits."""

    def __init__(self, config, frame, action=None, window=None):
        self.engine = BotEngine(config=config)
        self.window = window or StubWindow(frame)
        self.action = action or RecordingAction()
        self.engine._window = self.window
        self.engine._action = self.action

        self.logs = []
        self.states = []
        self.stats = []
        self.engine.log.connect(lambda lvl, msg: self.logs.append((lvl, msg)))
        self.engine.state_changed.connect(self.states.append)
        self.engine.stats_changed.connect(lambda r, c: self.stats.append((r, c)))

    def run_for(self, qapp, seconds: float):
        from PySide6.QtCore import QTimer

        self.engine.start()
        QTimer.singleShot(int(seconds * 1000), self.engine.stop)
        QTimer.singleShot(int(seconds * 1000) + 1500, qapp.quit)
        qapp.exec()
        assert self.engine.wait(4000), "engine thread did not finish in time"
        return self

    @property
    def log_text(self) -> str:
        return " || ".join(message for _, message in self.logs)

    @property
    def blacklisted(self) -> int:
        return self.engine._guard.blacklisted_count()


# --------------------------------------------------------------------- helpers


def build_config(template_dir, **safety_overrides) -> AppConfig:
    safety = dict(
        confirm_frames=2,
        click_cooldown=0.0,
        failure_cooldown_ladder=(0.0,),
        blacklist_after_failures=2,
        blacklist_duration=30.0,
        verify_after_click=True,
        verify_delay=0.05,
        # Verify every click here. Sampling is a throughput optimisation with its
        # own unit tests; these tests are about the pipeline, and sampling would
        # make click-by-click outcomes depend on how many rounds happened to fit
        # in the time budget.
        verify_sample_every=1,
        # Pinned on purpose. Calibration disabled this rule for the real game,
        # where it measured as rejecting nothing, but these tests exercise the
        # rule's wiring through the engine and must not inherit that decision
        # from the shipped defaults.
        require_texture_below=True,
        # Likewise pinned: the shipped zones are calibrated for 1024x768 real
        # frames, while this file works in the synthetic 1707x960 reference
        # space. Only the left column matters here, for COIN_IN_PANEL.
        exclusion_zones=(
            (0.00, 0.00, 1.00, 0.07),
            (0.00, 0.07, 0.31, 1.00),
            (0.00, 0.86, 1.00, 1.00),
        ),
    )
    safety.update(safety_overrides)
    return AppConfig(
        click=ClickConfig(post_click_delay=(0.0, 0.0), press_duration=(0.0, 0.01)),
        loop=LoopConfig(tick_interval=(0.02, 0.05), max_clicks_per_tick=1),
        vision=VisionConfig(
            template_dir=template_dir,
            match_threshold=0.80,
            reference_size=REFERENCE_SIZE,
            use_fake_detection=False,
        ),
        safety=SafetyConfig(**safety),
    )


def near(point, target, tolerance=8) -> bool:
    return (
        abs(point[0] - target[0]) <= tolerance
        and abs(point[1] - target[1]) <= tolerance
    )


def expected(reference_point):
    return scale_point(reference_point, REFERENCE_SIZE, WINDOW)


# -------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def module_template_dir(tmp_path_factory) -> str:
    directory = tmp_path_factory.mktemp("templates")
    write_template(directory, "coin", make_coin_template())
    return str(directory)


@pytest.fixture(scope="module")
def pipeline_frame():
    """A resized window frame; every asset scales together as the game would."""
    reference = make_scene(
        *REFERENCE_SIZE,
        coin_centers=[COIN_VALID, COIN_FLAT, COIN_IN_PANEL],
        monster_under=[COIN_VALID],
    )
    return render_at(reference, WINDOW)


@pytest.fixture(scope="module")
def harness(qapp, module_template_dir, pipeline_frame):
    """One shared engine session that every read-only assertion inspects."""
    return EngineHarness(build_config(module_template_dir), pipeline_frame).run_for(
        qapp, 3.0
    )


# ----------------------------------------------------------------------- tests


class TestThreadingContract:
    """The whole point of the QThread design: the UI thread must stay free."""

    def test_engine_terminates_cleanly(self, harness):
        assert not harness.engine.isRunning()

    def test_clicks_execute_off_the_calling_thread(self, harness):
        assert harness.action.threads
        assert threading.get_ident() not in harness.action.threads

    def test_window_is_released_on_exit(self, harness):
        assert harness.window.detached

    def test_final_state_is_stopped(self, harness):
        assert harness.states[-1] == BotState.STOPPED.value


class TestTargetSelection:
    """Only the coin that passes every rule may be clicked."""

    def test_at_least_one_click_happened(self, harness):
        assert harness.action.clicks

    def test_the_valid_coin_is_the_only_target(self, harness):
        target = expected(COIN_VALID)
        assert all(near(point, target) for point in harness.action.clicks)

    def test_coin_on_empty_ground_is_never_clicked(self, harness):
        forbidden = expected(COIN_FLAT)
        assert not any(near(point, forbidden) for point in harness.action.clicks)

    def test_coin_inside_the_ui_panel_is_never_clicked(self, harness):
        forbidden = expected(COIN_IN_PANEL)
        assert not any(near(point, forbidden) for point in harness.action.clicks)

    def test_all_three_coins_were_actually_detected(self, harness):
        """Proves the rejections came from the rules, not from a detection miss."""
        assert "识别到 3 个金币" in harness.log_text

    def test_rejection_reasons_are_reported_by_stable_code(self, harness):
        codes = set(harness.engine.last_rejections.by_code)
        assert "NO_TEXTURE_BELOW" in codes


class TestWindowAdaptation:
    def test_scale_is_pushed_to_the_action_layer(self, harness):
        assert harness.action.scales
        assert harness.action.scales[0] == pytest.approx(0.75, abs=0.02)

    def test_adaptation_is_announced_once(self, harness):
        announcements = [m for _, m in harness.logs if "已自适应" in m]
        assert len(announcements) == 1
        assert "1280x720" in announcements[0]


class TestConfirmationGate:
    def test_first_round_withholds_the_click(self, harness):
        assert "等待跨帧确认" in harness.log_text

    def test_a_later_round_lets_it_through(self, harness):
        assert "通过 0 个" in harness.log_text
        assert "通过 1 个" in harness.log_text


class TestVerifyAndSelfCorrection:
    """A click that never removes the target must eventually be abandoned."""

    def test_verification_reads_extra_frames(self, harness):
        assert harness.window.captures > len(harness.action.clicks)

    def test_verification_actually_ran(self, harness):
        assert harness.engine._verifications > 0

    def test_ineffective_clicks_get_the_position_blacklisted(self, harness):
        """The recording action never removes a coin, so the spot must get banned.

        Asserted against guard state, not by searching the log. The startup
        banner and the shutdown summary both mention blacklisting by name, so a
        substring check over the whole transcript would pass for the wrong
        reason - which is exactly what happened before this was tightened.
        """
        assert harness.blacklisted >= 1

    def test_every_verification_reported_a_surviving_target(self, harness):
        assert harness.engine._verified_failures == harness.engine._verifications


class TestStateMachine:
    def test_every_pipeline_stage_is_visited(self, harness):
        visited = set(harness.states)
        for state in (
            BotState.ATTACHING,
            BotState.SEARCHING,
            BotState.VALIDATING,
            BotState.ACTING,
            BotState.VERIFYING,
            BotState.STOPPED,
        ):
            assert state.value in visited, state.name


class TestStatsReporting:
    def test_round_counter_only_increases(self, harness):
        rounds = [r for r, _ in harness.stats]
        assert rounds == sorted(rounds)
        assert rounds[-1] >= 2

    def test_click_counter_matches_recorded_clicks(self, harness):
        assert harness.stats[-1][1] == len(harness.action.clicks)


class TestAlternativeConfigurations:
    """These need their own engine run because the configuration differs."""

    def test_verify_can_be_switched_off(self, qapp, module_template_dir, pipeline_frame):
        run = EngineHarness(
            build_config(
                module_template_dir, verify_after_click=False, confirm_frames=1
            ),
            pipeline_frame,
        ).run_for(qapp, 1.2)

        assert run.action.clicks
        # With verification off nothing can be judged a failure, so no position
        # can ever be banned. Both facts are checked directly rather than by
        # matching log text, which now contains status banners mentioning bans.
        assert run.engine._verifications == 0
        assert run.blacklisted == 0

    def test_a_failing_action_is_logged_and_does_not_crash(
        self, qapp, module_template_dir, pipeline_frame
    ):
        run = EngineHarness(
            build_config(module_template_dir, confirm_frames=1),
            pipeline_frame,
            action=FailingAction(),
        ).run_for(qapp, 1.2)
        assert "投递失败" in run.log_text
        assert not run.engine.isRunning()

    def test_capture_failure_is_survived(
        self, qapp, module_template_dir, pipeline_frame
    ):
        run = EngineHarness(
            build_config(module_template_dir),
            pipeline_frame,
            window=DeadWindow(pipeline_frame),
        ).run_for(qapp, 1.2)
        assert "截图失败" in run.log_text
        assert not run.action.clicks

    def test_empty_scene_reports_no_target_rather_than_erroring(
        self, qapp, module_template_dir
    ):
        blank = render_at(make_scene(*REFERENCE_SIZE, coin_centers=[]), WINDOW)
        run = EngineHarness(build_config(module_template_dir), blank).run_for(qapp, 1.0)
        assert "本轮未发现金币" in run.log_text
        assert not run.action.clicks

    def test_exclusion_zone_rejection_is_reported(
        self, qapp, module_template_dir, pipeline_frame
    ):
        """Isolated so the shared run's later rounds cannot mask this code."""
        run = EngineHarness(
            build_config(module_template_dir, confirm_frames=1),
            pipeline_frame,
        ).run_for(qapp, 1.0)
        assert "EXCLUSION_ZONE" in run.engine.last_rejections.by_code

    def test_escalating_cooldown_delays_a_repeatedly_failing_position(
        self, qapp, module_template_dir, pipeline_frame
    ):
        """With a real ladder, a spot that never responds must be backed off.

        The recording action never removes anything, so every verification fails
        and the position climbs the ladder instead of being retried immediately.
        """
        run = EngineHarness(
            build_config(
                module_template_dir,
                confirm_frames=1,
                click_cooldown=0.05,
                failure_cooldown_ladder=(5.0, 10.0),
                blacklist_after_failures=99,  # keep it on the ladder, not banned
            ),
            pipeline_frame,
        ).run_for(qapp, 2.0)

        assert run.action.clicks, "expected at least one click before backing off"
        # A 5s penalty inside a 2s run means it must not have been clicked twice.
        assert len(run.action.clicks) == 1
        assert "COOLING_DOWN" in run.engine.last_rejections.by_code
