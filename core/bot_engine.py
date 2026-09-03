"""Control layer: the idle-bot main loop and its decision pipeline.

The whole loop runs on a QThread worker, so the Qt main thread is never blocked:
the window stays draggable and the log keeps updating while the bot works.

Threading contract (please keep this when extending):
  - This class never touches a QWidget; everything it wants to say goes out
    through a Signal.
  - The UI only calls start() / stop(); it does not read or write internal state.
  - Cross-thread Signal connections are queued by Qt, so slots run on the main
    thread automatically.

Single round pipeline
---------------------
    capture
     -> detect        (vision.detect, returns Detection boxes)
         -> validate  (validators: exclusion zones / confidence / neighbours)
             -> guard (click_guard: confirmation / escalating cooldown / blacklist)
                 -> batch (pick candidates that cannot interfere with each other)
                     -> recheck (is this target still where the frame said?)
                         -> click   (action.click, with coordinate/timing jitter)
                             -> verify (sampled: did the target disappear?)

Every stage can be switched off or retuned independently, which makes failures
diagnosable layer by layer. Rejected candidates are grouped by reason and
summarised into a single log line per round, so the log stays readable and
doubles as the calibration instrument.

Capture requires the game window to be **rendered**, not focused. It may sit
behind other windows and does not need the mouse. It may not be minimized:
Windows stops rendering a minimized window, so PrintWindow has nothing to copy.

Where the time goes, and how that was established
-------------------------------------------------
Throughput was tuned twice by shrinking sleep constants before anyone measured
the non-sleep part of a round. Benchmarking the real vision agent over real saved
frames then showed detection alone cost 103-108 ms per frame, more than every
sleep in the loop combined - so the earlier tuning had been optimising the
visible part rather than the dominant part.

This class now accumulates per-stage timings and reports them when it stops, so
the next such question is answered by the log instead of by argument. Capture in
particular can only be measured against the live game, which is why it is
instrumented here rather than in the offline benchmark.

Three throughput decisions worth understanding
----------------------------------------------
Verification is sampled, not unconditional. Re-reading a frame after every click
was the largest fixed cost per collected item, but removing it would remove the
only mechanism that notices a false positive. So the engine verifies every click
until a run has proven itself, then samples, and drops straight back to verifying
everything the moment a check fails.

Clicking is batched, but only across candidates whose boxes do not overlap.
Stacked plaques must still be peeled one per round: a front plaque hides the ones
behind it, so their positions are not known yet. Plaques that do not overlap are
independent, and making each of them wait for its own capture-and-detect cycle
buys nothing.

Batched coordinates are rechecked before use. Every coordinate in a batch comes
from one frame, so anything that moves the view after the first click leaves the
rest of the batch aiming at stale positions - observed live as occasional clicks
landing on scenery, which in this game pans the camera and makes the next click
worse still. Each click after the first therefore re-captures and confirms its
target is still there, and skips rather than guesses.
"""

from __future__ import annotations

import logging
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PySide6.QtCore import QThread, Signal

from config import (
    DEFAULT_CONFIG,
    TARGET_COIN,
    TARGET_DIAMOND,
    TARGET_MODAL_CONFIRM,
    TARGET_PIGGY_BANK,
    TARGET_TREATS,
    AppConfig,
)
from core.action_agent import ActionAgent
from core.click_guard import ClickGuard
from core.diagnostics import MissRecorder
from core.game_window import GameWindow
from core.geometry import (
    PixelRect,
    Point,
    distance,
    rects_overlap,
    scale_factor,
    scale_length,
)
from core.validators import ValidationContext, build_rule_chain
from core.vision_agent import (
    BaseVisionAgent,
    Detection,
    VisionAgent,
    check_piggy_status,
)
from collections import deque
import cv2
import difflib
from core.map_navigator import MapNavigator, ScreenState, IslandCardInfo, hash_distance

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueuedIsland:
    """Item in the monotonic FIFO tour queue."""

    name: str
    canon_name: str
    card_hash: int = 0
    card_hist: Optional[np.ndarray] = None


class EngineMode(str, Enum):
    """Execution modes for the bot engine."""

    COIN = "coin"
    DIAMOND = "diamond"
    TREATS = "treats"
    PIGGY = "piggy"
    ALL_RESOURCES = "all_resources"
    MAP_TOUR = "map_tour"
    TRACK_ISLAND = "track_island"


class LogLevel(str, Enum):
    """Log severity; the UI maps these to colours."""

    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARN = "WARN"
    ERROR = "ERROR"


class BotState(str, Enum):
    """Runtime state. Display strings are Chinese because the UI is Chinese."""

    IDLE = "空闲"
    ATTACHING = "查找窗口"
    SEARCHING = "搜索目标"
    VALIDATING = "校验候选"
    ACTING = "执行点击"
    VERIFYING = "回读验证"
    ERROR = "异常"
    STOPPED = "已停止"


def inflate(box: PixelRect, margin: int) -> PixelRect:
    """Grow a box on all sides. Used to treat touching boxes as interfering."""
    x1, y1, x2, y2 = box
    return x1 - margin, y1 - margin, x2 + margin, y2 + margin


def select_independent(
    candidates: Sequence[Detection], limit: int, margin: int
) -> List[Detection]:
    """Pick up to `limit` candidates that cannot interfere with each other.

    Candidates are expected in descending confidence order, so the clearest
    target is always chosen first and anything overlapping it is deferred to a
    later round, by which time its true position is known.

    Kept as a module-level function so it can be tested without an engine, a
    window, or a Qt event loop.
    """
    if limit <= 1:
        return list(candidates[:1])

    chosen: List[Detection] = []
    for candidate in candidates:
        if len(chosen) >= limit:
            break
        padded = inflate(candidate.bbox, margin)
        if any(rects_overlap(padded, inflate(other.bbox, margin)) for other in chosen):
            continue
        chosen.append(candidate)
    return chosen


@dataclass
class StageTimings:
    """Cumulative wall time per pipeline stage, in seconds.

    Exists because the loop was tuned twice without measuring: the sleeps were
    obvious and got shrunk, while detection quietly dominated. Reported on stop.
    """

    capture: float = 0.0
    detect: float = 0.0
    validate: float = 0.0
    click: float = 0.0
    verify: float = 0.0
    sleep: float = 0.0

    def add(self, stage: str, seconds: float) -> None:
        setattr(self, stage, getattr(self, stage) + seconds)

    def total(self) -> float:
        return (
            self.capture
            + self.detect
            + self.validate
            + self.click
            + self.verify
            + self.sleep
        )

    def describe(self, rounds: int) -> str:
        total = self.total()
        if total <= 0 or rounds <= 0:
            return "no timing data"
        parts = []
        for name, label in (
            ("capture", "截图"),
            ("detect", "识别"),
            ("validate", "校验"),
            ("click", "点击"),
            ("verify", "回读"),
            ("sleep", "间隔"),
        ):
            value = getattr(self, name)
            parts.append(
                "%s %.0fms(%.0f%%)" % (label, value / rounds * 1000.0, 100.0 * value / total)
            )
        return "单轮 %.0fms ｜ %s" % (total / rounds * 1000.0, "，".join(parts))


@dataclass
class RejectionStats:
    """Per-round tally of why candidates were dropped.

    Counts are keyed by stable code so tests and future metrics do not depend on
    wording; ``labels`` keeps the human-readable name for the log line.
    """

    by_code: Counter = field(default_factory=Counter)
    labels: Dict[str, str] = field(default_factory=dict)

    def add(self, code: str, label: str) -> None:
        self.by_code[code] += 1
        self.labels.setdefault(code, label or code)

    def total(self) -> int:
        return sum(self.by_code.values())

    def describe(self) -> str:
        return "，".join(
            "%s×%d" % (self.labels.get(code, code), count)
            for code, count in self.by_code.most_common()
        )

    def dominant_code(self) -> str:
        if not self.by_code:
            return "nohit"
        return self.by_code.most_common(1)[0][0]

    def __bool__(self) -> bool:
        return self.total() > 0


class BotEngine(QThread):
    """The idle bot. One instance per session."""

    # (level, text) -> forwarded to the UI log
    log = Signal(str, str)
    # State change, carries the BotState display string
    state_changed = Signal(str)
    # (total rounds, successful clicks)
    stats_changed = Signal(int, int)

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        vision_agent: Optional[BaseVisionAgent] = None,
        mode: str = EngineMode.COIN,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._cfg = config or DEFAULT_CONFIG
        self._mode = EngineMode(mode)
        self._rng = random.Random()

        # Dependency injection: pass a YoloVisionAgent here and the engine needs
        # no changes at all.
        self._vision: BaseVisionAgent = vision_agent or VisionAgent(
            template_dir=self._cfg.vision.template_dir,
            match_threshold=self._cfg.vision.match_threshold,
            reference_size=self._cfg.vision.reference_size,
            scale_steps=self._cfg.vision.scale_steps,
            nms_distance=self._cfg.vision.nms_distance,
            max_detections=self._cfg.vision.max_detections,
            use_fake_detection=self._cfg.vision.use_fake_detection,
            simulate_miss_rate=self._cfg.vision.simulate_miss_rate,
            match_downscale=self._cfg.vision.match_downscale,
            search_region=self._cfg.vision.search_region,
            target_thresholds=self._cfg.vision.target_thresholds,
        )

        self._window = GameWindow(
            title=self._cfg.window.title,
            print_window_flag=self._cfg.window.print_window_flag,
        )
        self._action = ActionAgent(self._window, self._cfg.click)

        self._rules = build_rule_chain(
            self._cfg.safety, self._cfg.vision.match_threshold
        )
        self._guard = ClickGuard(self._cfg.safety)

        self._recorder = MissRecorder(
            directory=self._cfg.diagnostics.dump_dir,
            enabled=self._cfg.diagnostics.dump_frames_on_miss,
            max_dumps=self._cfg.diagnostics.max_dumps_per_session,
            annotate=self._cfg.diagnostics.annotate_dumps,
        )

        # Extra categories neighbour rules need (monster / cloud), excluding the
        # primary target itself.
        self._context_targets = tuple(self._rules.required_targets - {TARGET_COIN})

        self._state = BotState.IDLE
        self._rounds = 0
        self._clicks = 0
        self._consecutive_errors = 0
        self._last_client_size: Optional[Tuple[int, int]] = None
        # Kept for inspection by tests and future metrics.
        self._last_rejections = RejectionStats()
        self._last_batch_size = 0
        self._stale_skips = 0
        self._timings = StageTimings()

        # Sampled-verification state.
        self._verified_streak = 0
        self._verify_countdown = 0
        self._verifications = 0
        self._verified_failures = 0
        self._blacklist: List[str] = list(self._cfg.map.blacklist)
        self._track_target: str = ""

    # ------------------------------------------------------------ public API

    def set_blacklist(self, blacklist: Sequence[str]) -> None:
        """Update the list of island names to skip during map tours."""
        self._blacklist = [str(b).strip() for b in blacklist if str(b).strip()]

    def set_track_target(self, target: str) -> None:
        """Set the target island name for tracking tests."""
        self._track_target = target.strip()

    def stop(self) -> None:
        """Request a stop. Called from the UI thread; returns immediately.

        Uses Qt's interruption flag rather than a custom bool: built into
        QThread, unambiguous, and needs no lock.
        """
        if self.isRunning():
            self._emit_log(LogLevel.WARN, "收到停止指令，正在收尾…")
            self.requestInterruption()

    @property
    def mode(self) -> EngineMode:
        return self._mode

    @property
    def state(self) -> BotState:
        return self._state

    @property
    def last_rejections(self) -> RejectionStats:
        return self._last_rejections

    @property
    def timings(self) -> StageTimings:
        return self._timings

    @property
    def blacklist(self) -> List[str]:
        return list(self._blacklist)

    # ---------------------------------------------------------- thread body

    def run(self) -> None:  # noqa: D102  (QThread entry point)
        """Worker entry point. Wrapped in try/except so the thread never dies mutely."""
        self._rounds = 0
        self._clicks = 0
        self._consecutive_errors = 0
        self._last_client_size = None
        self._last_batch_size = 0
        self._stale_skips = 0
        self._timings = StageTimings()
        self._verified_streak = 0
        self._verify_countdown = 0
        self._verifications = 0
        self._verified_failures = 0

        if self._mode == EngineMode.MAP_TOUR:
            self._announce_startup("收集各岛资源任务已启动")
        elif self._mode == EngineMode.TRACK_ISLAND:
            self._announce_startup("岛屿追踪测试已启动")
        elif self._mode == EngineMode.ALL_RESOURCES:
            self._announce_startup("单岛资源全收测试已启动")
        elif self._mode == EngineMode.PIGGY:
            self._announce_startup("小猪储蓄罐任务已启动")
        elif self._mode == EngineMode.DIAMOND:
            self._announce_startup("钻石收集任务已启动")
        elif self._mode == EngineMode.TREATS:
            self._announce_startup("食物收集任务已启动")
        else:
            self._announce_startup("金币挂机引擎已启动")

        try:
            if self._mode == EngineMode.MAP_TOUR:
                self._run_map_tour()
            elif self._mode == EngineMode.TRACK_ISLAND:
                self._run_track_island()
            elif self._mode == EngineMode.ALL_RESOURCES:
                self._run_all_resources()
            elif self._mode == EngineMode.PIGGY:
                self._run_piggy_mode()
            elif self._mode == EngineMode.DIAMOND:
                self._collect_target_until_empty(
                    TARGET_DIAMOND, max_rounds=6, max_consecutive_empty=4
                )
            elif self._mode == EngineMode.TREATS:
                self._collect_target_until_empty(
                    TARGET_TREATS, max_rounds=30, max_consecutive_empty=4
                )
            else:
                self._run_coin_loop()

        except Exception as exc:  # pragma: no cover
            logger.exception("engine main loop crashed")
            self._set_state(BotState.ERROR)
            self._emit_log(LogLevel.ERROR, "主循环崩溃: %r" % (exc,))

        finally:
            self._window.detach()
            self._set_state(BotState.STOPPED)
            if self._cfg.diagnostics.dump_frames_on_miss:
                self._emit_log(LogLevel.INFO, self._recorder.summary())
            self._emit_log(
                LogLevel.INFO,
                "引擎已停止 ｜ 共 %d 轮，成功点击 %d 次（平均 %.2f 次/轮）"
                " ｜ 坐标过期跳过 %d 次 ｜ 抽检 %d 次（%d 次失败）｜ %s"
                % (
                    self._rounds,
                    self._clicks,
                    self._clicks / self._rounds if self._rounds else 0.0,
                    self._stale_skips,
                    self._verifications,
                    self._verified_failures,
                    self._guard.stats(),
                ),
            )
            self._emit_log(LogLevel.INFO, "耗时分布: %s" % self._timings.describe(self._rounds))

    def _run_coin_loop(self) -> None:
        """Continuous coin collection loop until interrupted."""
        while not self.isInterruptionRequested():
            self._tick()

            if self._consecutive_errors >= self._cfg.loop.max_consecutive_errors:
                self._emit_log(
                    LogLevel.ERROR,
                    "连续 %d 轮异常，自动停止以避免空转" % self._consecutive_errors,
                )
                break

            self._sleep_timed(self._rng.uniform(*self._cfg.loop.tick_interval))

    def _run_piggy_mode(self) -> None:
        """Collect via piggy bank and finish."""
        self._run_piggy_stage()

    def _run_piggy_stage(self) -> bool:
        """Attempt to activate and confirm the piggy bank if bright.

        Returns True if the piggy bank was clicked and confirmed closed, False otherwise.
        """
        self._set_state(BotState.SEARCHING)
        if not self._window.ensure_attached():
            self._emit_log(LogLevel.WARN, "未找到游戏窗口，跳过储蓄罐")
            return False

        frame = self._timed("capture", self._window.capture)
        if frame is None:
            self._emit_log(LogLevel.WARN, "截图失败，跳过储蓄罐")
            return False

        height, width = frame.shape[:2]
        client_size = (width, height)
        scale = scale_factor(client_size, self._cfg.vision.reference_size)
        self._action.set_scale(scale)
        self._guard.set_scale(scale)

        # Check if the confirmation modal is already open on screen
        confirms = self._timed(
            "detect", self._vision.detect, TARGET_MODAL_CONFIRM, frame
        )
        if not confirms:
            piggies = self._timed(
                "detect", self._vision.detect, TARGET_PIGGY_BANK, frame
            )
            if not piggies:
                self._emit_log(LogLevel.INFO, "未检测到小猪储蓄罐图标（本岛无储蓄罐或被遮挡），跳过")
                return False

            piggy = piggies[0]
            crop = frame[
                max(0, piggy.y) : min(height, piggy.y + piggy.height),
                max(0, piggy.x) : min(width, piggy.x + piggy.width),
            ]
            is_bright = check_piggy_status(
                crop, min_v=self._cfg.vision.piggy_min_brightness_v
            )
            if not is_bright:
                self._emit_log(LogLevel.INFO, "小猪储蓄罐处于黯淡状态（无可收集货币），跳过")
                return False

            self._set_state(BotState.ACTING)
            self._emit_log(LogLevel.INFO, "检测到高光储蓄罐，执行点击...")
            if not self._timed(
                "click", self._action.click, piggy.center[0], piggy.center[1]
            ):
                self._emit_log(LogLevel.WARN, "点击储蓄罐失败")
                return False

            self._clicks += 1
            self.stats_changed.emit(self._rounds, self._clicks)

        # Wait for modal confirmation button to appear and click with retry until closed
        self._set_state(BotState.SEARCHING)
        modal_deadline = time.monotonic() + self._cfg.vision.modal_timeout
        confirmed_any = False

        while time.monotonic() < modal_deadline and not self.isInterruptionRequested():
            self._sleep_timed(0.2)
            mframe = self._timed("capture", self._window.capture)
            if mframe is None:
                continue

            confirms = self._timed(
                "detect", self._vision.detect, TARGET_MODAL_CONFIRM, mframe
            )
            if confirms:
                confirm_btn = confirms[0]
                self._set_state(BotState.ACTING)
                self._emit_log(LogLevel.INFO, "检测到确认弹窗，点击绿色确认按钮...")
                self._timed(
                    "click",
                    self._action.click,
                    confirm_btn.center[0],
                    confirm_btn.center[1],
                )
                self._clicks += 1
                self.stats_changed.emit(self._rounds, self._clicks)
                confirmed_any = True
                self._sleep_timed(0.35)
                continue

            if confirmed_any:
                self._emit_log(LogLevel.SUCCESS, "储蓄罐收集完成，确认弹窗已关闭")
                self._sleep_timed(0.4)
                return True

        if not confirmed_any:
            self._emit_log(LogLevel.WARN, "等待确认弹窗超时，未发现确认按钮")
            return False

        self._emit_log(LogLevel.SUCCESS, "储蓄罐收集完成")
        self._sleep_timed(0.4)
        return True

    def _collect_target_until_empty(
        self,
        target_name: str,
        max_rounds: int = 30,
        max_consecutive_empty: int = 4,
    ) -> int:
        """Collect specified target iteratively until empty or max rounds reached."""
        target_labels = {
            TARGET_COIN: "金币",
            TARGET_DIAMOND: "钻石",
            TARGET_TREATS: "食物",
        }
        label = target_labels.get(target_name, target_name)
        consecutive_empty = 0
        clicks_start = self._clicks

        for _ in range(max_rounds):
            if self.isInterruptionRequested():
                break

            self._rounds += 1
            self.stats_changed.emit(self._rounds, self._clicks)

            if not self._window.ensure_attached():
                self._consecutive_errors += 1
                self._sleep_timed(self._cfg.loop.retry_interval)
                continue

            frame = self._timed("capture", self._window.capture)
            if frame is None:
                self._consecutive_errors += 1
                self._sleep_timed(self._cfg.loop.retry_interval)
                continue

            height, width = frame.shape[:2]
            client_size = (width, height)
            scale = scale_factor(client_size, self._cfg.vision.reference_size)
            self._action.set_scale(scale)
            self._guard.set_scale(scale)

            self._set_state(BotState.SEARCHING)
            self._guard.begin_frame()
            targets = self._timed("detect", self._vision.detect, target_name, frame)
            self._guard.observe(targets)

            if not targets:
                consecutive_empty += 1
                self._emit_log(
                    LogLevel.INFO,
                    f"未发现可收集的{label}（{consecutive_empty}/{max_consecutive_empty}）",
                )
                if consecutive_empty >= max_consecutive_empty:
                    break
                self._sleep_timed(self._rng.uniform(*self._cfg.loop.tick_interval))
                continue

            consecutive_empty = 0
            self._set_state(BotState.VALIDATING)
            candidates, rejections = self._timed(
                "validate",
                self._filter_candidates,
                targets,
                frame,
                client_size,
                scale,
                target_name,
            )
            self._last_rejections = rejections

            if not candidates:
                summary = f"识别到 {len(targets)} 个{label}，通过 0 个"
                if rejections:
                    summary += f" ｜ 过滤: {rejections.describe()}"
                self._emit_log(LogLevel.INFO, summary)
                self._sleep_timed(self._rng.uniform(*self._cfg.loop.tick_interval))
                continue

            batch = select_independent(
                candidates,
                self._cfg.loop.max_clicks_per_tick,
                scale_length(self._cfg.loop.click_separation_margin, scale, minimum=0),
            )
            self._last_batch_size = len(batch)

            summary = (
                f"识别到 {len(targets)} 个{label}，通过 {len(candidates)} 个，本轮点 {len(batch)} 个"
            )
            if len(batch) < len(candidates):
                summary += "（其余重叠，留待下轮）"
            if rejections:
                summary += f" ｜ 过滤: {rejections.describe()}"
            self._emit_log(LogLevel.INFO, summary)

            self._click_batch(batch, scale, target_name=target_name)
            self.stats_changed.emit(self._rounds, self._clicks)
            self._sleep_timed(self._rng.uniform(*self._cfg.loop.tick_interval))

        return self._clicks - clicks_start

    def _run_all_resources(self) -> None:
        """Run the composite resource collection pipeline in sequence."""
        self._emit_log(LogLevel.INFO, "===== 开始执行「资源全收」流水线 =====")

        # Stage 1: Piggy Bank
        if self.isInterruptionRequested():
            return
        self._emit_log(LogLevel.INFO, "【阶段 1/4】 检查小猪储蓄罐...")
        piggy_ok = self._run_piggy_stage()
        if piggy_ok:
            self._emit_log(LogLevel.SUCCESS, "【阶段 1/4】 储蓄罐全收完成")
        else:
            self._emit_log(LogLevel.INFO, "【阶段 1/4】 储蓄罐已跳过，继续后续流程")

        # Stage 2: Diamond
        if self.isInterruptionRequested():
            return
        self._emit_log(LogLevel.INFO, "【阶段 2/4】 收集钻石...")
        d_clicks = self._collect_target_until_empty(
            TARGET_DIAMOND, max_rounds=6, max_consecutive_empty=4
        )
        self._emit_log(LogLevel.INFO, f"【阶段 2/4】 钻石收集完毕（点击 {d_clicks} 次）")

        # Stage 3: Treats
        if self.isInterruptionRequested():
            return
        self._emit_log(LogLevel.INFO, "【阶段 3/4】 收集食物 (Treats)...")
        t_clicks = self._collect_target_until_empty(
            TARGET_TREATS, max_rounds=30, max_consecutive_empty=4
        )
        self._emit_log(LogLevel.INFO, f"【阶段 3/4】 食物收集完毕（点击 {t_clicks} 次）")

        # Stage 4: Coins
        if self.isInterruptionRequested():
            return
        self._emit_log(LogLevel.INFO, "【阶段 4/4】 收集金币...")
        c_clicks = self._collect_target_until_empty(
            TARGET_COIN, max_rounds=40, max_consecutive_empty=4
        )
        self._emit_log(LogLevel.INFO, f"【阶段 4/4】 金币收集完毕（点击 {c_clicks} 次）")

        self._emit_log(
            LogLevel.SUCCESS,
            f"===== 资源全收流水线已完成 ｜ 总点击 {self._clicks} 次 =====",
        )

    def _run_map_tour(self) -> None:
        """Execute multi-island traversal with blacklist filtering and full resource collection."""
        self._set_state(BotState.SEARCHING)
        self._emit_log(LogLevel.INFO, "===== 开始执行「收集各岛资源」任务 =====")

        nav = MapNavigator(
            action_agent=self._action,
            window=self._window,
            config=self._cfg,
        )

        if not self._window.ensure_attached():
            self._emit_log(LogLevel.ERROR, "未能连接到游戏窗口，退出巡岛")
            return

        frame = self._timed("capture", self._window.capture)
        if frame is None:
            self._emit_log(LogLevel.ERROR, "未能捕获游戏画面，退出巡岛")
            return

        visited_names: Set[str] = set()
        visited_hashes: List[int] = []
        island_queue: deque[QueuedIsland] = deque()
        last_anchor: Optional[QueuedIsland] = None

        def words_match(a: str, b: str) -> bool:
            if a == b:
                return True
            if len(a) <= 4 and len(b) <= 4 and a[0] != b[0]:
                return False
            return difflib.SequenceMatcher(None, a, b).ratio() >= 0.70

        def names_fuzzy_match(s1: str, s2: str) -> bool:
            if not s1 or not s2:
                return False
            c1 = s1.strip().lower()
            c2 = s2.strip().lower()
            if c1 == c2:
                return True
            w1 = c1.split()
            w2 = c2.split()
            if len(w1) > 1 and len(w2) > 1:
                if len(w1) != len(w2):
                    return False
                return all(words_match(a, b) for a, b in zip(w1, w2))
            return words_match(c1, c2)

        def match_card_to_island(card: IslandCardInfo, target: QueuedIsland) -> bool:
            # 1. Perceptual dHash match (visual identity - primary invariant)
            if card.card_hash != 0 and target.card_hash != 0:
                if hash_distance(card.card_hash, target.card_hash) <= 6:
                    return True

            c_raw = card.name.strip().lower()
            t_raw = target.name.strip().lower()

            # 2. Text name matching (exact or structural fuzzy)
            if names_fuzzy_match(c_raw, t_raw):
                return True

            return False

        def is_card_already_visited(card: IslandCardInfo) -> bool:
            # 1. Perceptual dHash check (primary visual invariant - unconditional deduplication)
            if card.card_hash != 0:
                for vh in visited_hashes:
                    if hash_distance(card.card_hash, vh) <= 6:
                        return True

            c_raw = card.name.strip().lower()

            # 2. Text name matching against visited names
            if c_raw:
                for v in visited_names:
                    if names_fuzzy_match(c_raw, v):
                        return True

            return False

        def mark_card_visited(card: IslandCardInfo) -> None:
            clean_name = card.name.strip().lower()
            if clean_name:
                visited_names.add(clean_name)
            if card.card_hash != 0:
                visited_hashes.append(card.card_hash)

        def scan_and_enqueue_downward(cards_list: List[IslandCardInfo]) -> int:
            nonlocal last_anchor
            # 1. Lock anchor vertical position on current screen
            anchor_y = -1
            if last_anchor is not None:
                for c in cards_list:
                    if match_card_to_island(c, last_anchor):
                        anchor_y = c.rect[1]
                        break

            # 2. Fuzzy range: mask out everything above anchor; search entire region below
            downward_cards: List[IslandCardInfo] = []
            for c in cards_list:
                if anchor_y >= 0 and c.rect[1] <= anchor_y + 10:
                    continue
                downward_cards.append(c)

            # 3. Sort candidates strictly by Y axis (top to bottom)
            downward_cards.sort(key=lambda c: c.rect[1])

            # 4. Enqueue new unvisited islands in order
            enqueued = 0
            for c in downward_cards:
                # Guard against enqueuing partially clipped bottom boundary slivers
                if not c.is_fully_visible:
                    continue
                if not c.name or len(c.name.strip()) < 2:
                    continue
                if is_card_already_visited(c):
                    continue

                # Dual deduplication against items currently in queue
                already_queued = any(
                    (c.card_hash != 0 and q.card_hash != 0 and hash_distance(c.card_hash, q.card_hash) <= 6)
                    or match_card_to_island(c, q)
                    for q in island_queue
                )
                if already_queued:
                    continue

                item = QueuedIsland(
                    name=c.name,
                    canon_name=c.name,
                    card_hash=c.card_hash,
                    card_hist=c.card_hist,
                )
                island_queue.append(item)
                enqueued += 1
                self._emit_log(
                    LogLevel.INFO,
                    "【队列排队】 发现新岛屿 '%s'，加入待巡检队列（队列长度：%d）" % (c.name, len(island_queue)),
                )
            return enqueued

        cur_state = nav.detect_state(frame)

        if cur_state != ScreenState.MAP:
            self._emit_log(LogLevel.INFO, "正在打开地图界面以开始巡岛…")
            if not nav.open_map():
                self._emit_log(LogLevel.ERROR, "未能进入地图界面，退出巡岛")
                return
        else:
            self._emit_log(LogLevel.INFO, "检测到当前处于地图界面，执行地图初始化…")

        # Park cursor to eliminate hover glow effects before list scanning
        self._action.park_cursor()

        # Map Initialization: scroll to the very top so traversal begins at card 0
        self._emit_log(LogLevel.INFO, "【地图初始化】 正在滑动置顶岛屿列表…")
        nav.scroll_to_top()
        self._sleep_timed(0.4)

        consecutive_no_progress = 0
        target_miss_count = 0

        while not self.isInterruptionRequested():
            nav.wait_for_list_stable(timeout=1.2)

            frame = self._timed("capture", self._window.capture)
            if frame is None:
                break

            cards = nav.get_visible_cards(frame)
            if not cards:
                self._emit_log(LogLevel.WARN, "未检测到可见岛屿卡片，尝试滑动列表…")
                nav.scroll_down()
                continue

            # Update queue with cards discovered below current anchor
            scan_and_enqueue_downward(cards)

            # Check if queue is empty
            if not island_queue:
                self._emit_log(LogLevel.INFO, "当前视野待访队列为空，向下滑动探索后续新岛屿…")
                nav.scroll_down()
                nav.wait_for_list_stable(timeout=1.2)
                fresh_frame = self._timed("capture", self._window.capture)
                if fresh_frame is None:
                    break
                fresh_cards = nav.get_visible_cards(fresh_frame)
                new_found = scan_and_enqueue_downward(fresh_cards) if fresh_cards else 0

                if new_found == 0:
                    consecutive_no_progress += 1
                    if consecutive_no_progress >= 2:
                        self._emit_log(
                            LogLevel.SUCCESS,
                            "【巡岛完成】 待访队列已清空且向下滑动无新岛屿，所有岛屿收集完毕！",
                        )
                        break
                else:
                    consecutive_no_progress = 0
                continue

            # Strict FIFO: target next island at the head of queue
            target = island_queue[0]
            target_disp_name = target.name

            # Locate target card on current screen
            target_card: Optional[IslandCardInfo] = None
            for c in cards:
                if match_card_to_island(c, target):
                    target_card = c
                    break

            if target_card is not None and target_card.is_fully_visible:
                # Target is fully visible and ready for interaction!
                island_queue.popleft()
                target_miss_count = 0

                # Check blacklist
                if nav._recognizer.is_blacklisted(target_disp_name, self._blacklist):
                    self._emit_log(
                        LogLevel.WARN,
                        "【黑名单跳过】 岛屿 '%s' 在屏蔽列表中，自动跳过" % target_disp_name,
                    )
                    mark_card_visited(target_card)
                    last_anchor = target
                    continue

                # Select target island card
                self._emit_log(LogLevel.INFO, "【前往岛屿】 正在选中卡片 '%s'…" % target_disp_name)
                nav.select_island(target_card)
                self._sleep_timed(0.18)

                self._emit_log(LogLevel.INFO, "【进入岛屿】 正在确认进入 '%s'…" % target_disp_name)
                if not nav.enter_selected_island():
                    self._emit_log(LogLevel.WARN, "未能进入岛屿 '%s'，跳过该岛屿" % target_disp_name)
                    mark_card_visited(target_card)
                    last_anchor = target
                    continue

                # Inside island: run full resource collection pipeline
                self._emit_log(
                    LogLevel.SUCCESS,
                    "【岛上采集】 已成功进入 '%s'，开始执行资源全收…" % target_disp_name,
                )
                self._run_all_resources()

                if self.isInterruptionRequested():
                    break

                # Return to map with retry
                self._emit_log(LogLevel.INFO, "【返回地图】 资源收集完毕，正在返回地图界面…")
                return_ok = False
                for _ in range(3):
                    if nav.open_map():
                        return_ok = True
                        break
                    self._sleep_timed(0.5)

                if not return_ok:
                    self._emit_log(LogLevel.ERROR, "未能从岛屿返回地图，巡岛中止")
                    return

                # Mark visited and update anchor
                mark_card_visited(target_card)
                last_anchor = target

                # Synchronize post-return state: register the card matching target on returned screen
                post_frame = self._timed("capture", self._window.capture)
                if post_frame is not None:
                    post_cards = nav.get_visible_cards(post_frame)
                    if post_cards:
                        for pc in post_cards:
                            if match_card_to_island(pc, target):
                                mark_card_visited(pc)
                                break
                continue
            else:
                if target_card is not None:
                    # Target is visible in viewport but partially clipped by bottom edge: advance slightly to bring it into full view
                    self._emit_log(
                        LogLevel.INFO,
                        "待访目标 '%s' 位于下方边缘，微距推进至全貌视野…" % target_disp_name,
                    )
                    nav.scroll_down()
                    continue
                else:
                    target_miss_count += 1
                    if target_miss_count <= 2:
                        self._emit_log(
                            LogLevel.INFO,
                            "待访目标 '%s' 暂未在当前视野出现，向下推进寻找（尝试 %d/2）…" % (target_disp_name, target_miss_count),
                        )
                        nav.scroll_down()
                        continue
                    else:
                        # Target missing after 2 consecutive downward scrolls; pop it to break endless scroll cascade
                        self._emit_log(
                            LogLevel.WARN,
                            "待访目标 '%s' 连续滑动未现身，已从队列移出以防阻碍后续岛屿" % target_disp_name,
                        )
                        island_queue.popleft()
                        target_miss_count = 0
                        continue

    def _run_track_island(self) -> None:
        """Search the map list for a specified target island and enter it."""
        self._set_state(BotState.SEARCHING)
        target = self._track_target
        if not target:
            self._emit_log(LogLevel.WARN, "未指定目标岛屿名称，退出追踪测试")
            return

        self._emit_log(LogLevel.INFO, f"【岛屿追踪】 开始检索目标岛屿：'{target}'…")
        nav = MapNavigator(
            action_agent=self._action,
            window=self._window,
            config=self._cfg,
        )

        if not self._window.ensure_attached():
            self._emit_log(LogLevel.ERROR, "未能连接到游戏窗口，退出追踪测试")
            return

        def on_located(card_name: str) -> None:
            self._emit_log(LogLevel.INFO, f"【岛屿追踪】 已在列表中定位到目标卡片 '{card_name}'，正在进入…")

        ok, status = nav.find_and_enter_island(target, on_located=on_located)
        if ok:
            self._emit_log(LogLevel.SUCCESS, f"【追踪成功】 已成功定位并进入目标岛屿：'{target}'！")
        elif status == "entry_timeout":
            self._emit_log(LogLevel.ERROR, f"【进入超时】 已在列表中定位并选中目标岛屿：'{target}'，但确认进入超时！")
        else:
            self._emit_log(LogLevel.ERROR, f"【追踪失败】 列表中未检索到匹配的岛屿：'{target}'")

    def _announce_startup(self, title: str = "挂机引擎已启动") -> None:
        """One-time banner. Prints the settings that most often explain a miss."""
        vision = self._cfg.vision
        safety = self._cfg.safety
        loop = self._cfg.loop
        self._emit_log(LogLevel.INFO, title)
        self._emit_log(
            LogLevel.INFO,
            "目标窗口: %s ｜ 识别器: %s"
            % (self._cfg.window.title, type(self._vision).__name__),
        )
        self._emit_log(
            LogLevel.INFO,
            "模板目录: %s ｜ 参考分辨率: %dx%d ｜ 匹配阈值: %.2f ｜ 尺度 %d 档 ｜ 匹配降采样 %.2f"
            % (
                vision.template_dir,
                vision.reference_size[0],
                vision.reference_size[1],
                vision.match_threshold,
                len(vision.scale_steps),
                vision.match_downscale,
            ),
        )
        self._emit_log(LogLevel.INFO, "防误触规则链: %s" % self._rules.describe())
        self._emit_log(
            LogLevel.INFO,
            "节奏: 轮间隔 %.2f~%.2fs ｜ 每轮最多点 %d 个（仅互不重叠的）｜ 批内间隔 %.2f~%.2fs"
            % (
                loop.tick_interval[0],
                loop.tick_interval[1],
                loop.max_clicks_per_tick,
                self._cfg.click.batch_settle_delay[0],
                self._cfg.click.batch_settle_delay[1],
            ),
        )
        if loop.max_clicks_per_tick > 1:
            if safety.recheck_before_batch_click:
                self._emit_log(LogLevel.INFO, "批内每次点击前重新取证，坐标过期则跳过")
            else:
                self._emit_log(
                    LogLevel.WARN,
                    "批内取证已关闭：若画面在批次中途移动，后续点击会落在过期坐标上",
                )
        self._emit_log(
            LogLevel.INFO,
            "退避: 基础冷却 %.2fs ｜ 失败阶梯 %s ｜ %d 次后拉黑 %.0fs"
            % (
                safety.click_cooldown,
                "/".join("%.1fs" % v for v in safety.failure_cooldown_ladder) or "无",
                safety.blacklist_after_failures,
                safety.blacklist_duration,
            ),
        )
        if safety.verify_after_click:
            self._emit_log(
                LogLevel.INFO,
                "回读验证: 前 %d 次逐次验证，之后每 %d 次抽检一次（延迟 %.2fs）"
                % (
                    safety.verify_trust_after,
                    safety.verify_sample_every,
                    safety.verify_delay,
                ),
            )
        else:
            self._emit_log(LogLevel.WARN, "回读验证已关闭，假阳性不会被自动发现")
        if self._context_targets:
            self._emit_log(
                LogLevel.INFO,
                "邻居规则需额外识别: %s" % ", ".join(self._context_targets),
            )
        if vision.use_fake_detection:
            self._emit_log(
                LogLevel.WARN,
                "当前为假数据模式，坐标是随机生成的。"
                "备好模板后把 VisionConfig.use_fake_detection 设为 False",
            )
        if self._cfg.diagnostics.dump_frames_on_miss:
            self._emit_log(
                LogLevel.INFO,
                "取证已开启：未命中的画面会存到 %s" % self._cfg.diagnostics.dump_dir,
            )

    # ------------------------------------------------------------ one round

    def _tick(self) -> None:
        """One full perceive -> validate -> act -> verify cycle."""
        self._rounds += 1
        self.stats_changed.emit(self._rounds, self._clicks)

        # 1) make sure the window is there
        self._set_state(BotState.ATTACHING)
        if not self._window.ensure_attached():
            self._consecutive_errors += 1
            self._emit_log(
                LogLevel.WARN,
                "未找到窗口「%s」，请确认游戏已启动" % self._cfg.window.title,
            )
            self._sleep_timed(self._cfg.loop.retry_interval)
            return

        # 2) capture
        frame = self._timed("capture", self._window.capture)
        if frame is None:
            self._consecutive_errors += 1
            self._emit_log(
                LogLevel.WARN,
                "截图失败。窗口最小化时 Windows 不渲染它，PrintWindow 取不到画面；"
                "被其他窗口盖住是可以的，最小化不行",
            )
            self._sleep_timed(self._cfg.loop.retry_interval)
            return

        # 3) sync window size: everything calibrated in reference pixels is
        #    converted here, once per round.
        height, width = frame.shape[:2]
        client_size = (width, height)
        scale = scale_factor(client_size, self._cfg.vision.reference_size)
        self._action.set_scale(scale)
        self._guard.set_scale(scale)
        self._announce_resize(client_size, scale)

        # 4) detect
        self._set_state(BotState.SEARCHING)
        self._guard.begin_frame()
        coins = self._timed("detect", self._vision.detect, TARGET_COIN, frame)
        self._guard.observe(coins)

        if not coins:
            # Finding nothing is a normal branch, not an error.
            self._consecutive_errors = 0
            self._emit_log(LogLevel.INFO, "本轮未发现金币，继续巡查")
            self._record_miss(frame, coins, "nohit")
            return

        # 5) rules + guard
        self._set_state(BotState.VALIDATING)
        candidates, rejections = self._timed(
            "validate", self._filter_candidates, coins, frame, client_size, scale
        )
        self._last_rejections = rejections

        if not candidates:
            self._consecutive_errors = 0
            summary = "识别到 %d 个金币，通过 0 个" % len(coins)
            if rejections:
                summary += " ｜ 过滤: %s" % rejections.describe()
            self._emit_log(LogLevel.INFO, summary)
            self._record_miss(frame, coins, rejections.dominant_code())
            return

        # 6) batch only the candidates that cannot interfere with each other
        batch = select_independent(
            candidates,
            self._cfg.loop.max_clicks_per_tick,
            scale_length(self._cfg.loop.click_separation_margin, scale, minimum=0),
        )
        self._last_batch_size = len(batch)

        summary = "识别到 %d 个金币，通过 %d 个，本轮点 %d 个" % (
            len(coins),
            len(candidates),
            len(batch),
        )
        if len(batch) < len(candidates):
            summary += "（其余重叠，留待下轮）"
        if rejections:
            summary += " ｜ 过滤: %s" % rejections.describe()
        self._emit_log(LogLevel.INFO, summary)

        # 7) click + sampled verify
        self._click_batch(batch, scale)
        self.stats_changed.emit(self._rounds, self._clicks)

    # ------------------------------------------------------ candidate filter

    def _filter_candidates(
        self,
        targets: Sequence[Detection],
        frame: np.ndarray,
        client_size: Tuple[int, int],
        scale: float,
        target_name: str = TARGET_COIN,
    ) -> Tuple[List[Detection], RejectionStats]:
        """Run the rule chain and the guard; return survivors plus a reason tally."""
        # Neighbour categories are detected on demand, so a config without such
        # rules pays nothing.
        detections: Dict[str, List[Detection]] = {target_name: list(targets)}
        for name in self._context_targets:
            detections[name] = self._vision.detect(name, frame)

        ctx = ValidationContext(
            frame=frame,
            client_size=client_size,
            scale=scale,
            detections=detections,
        )

        candidates: List[Detection] = []
        rejections = RejectionStats()

        for target in targets:
            verdict = self._rules.evaluate(target, ctx)
            if not verdict.ok:
                rejections.add(verdict.code, verdict.label)
                logger.debug("rejected %s: %s", target.center, verdict.describe())
                continue

            blocked = self._guard.status(target)
            if blocked is not None:
                rejections.add(blocked.code, blocked.label)
                logger.debug("guard blocked %s: %s", target.center, blocked.message)
                continue

            candidates.append(target)

        return candidates, rejections

    # ------------------------------------------------------------- clicking

    def _click_batch(
        self,
        batch: Sequence[Detection],
        scale: float,
        target_name: str = TARGET_COIN,
    ) -> None:
        recheck = (
            self._cfg.safety.recheck_before_batch_click
            and self._cfg.loop.max_clicks_per_tick > 1
        )
        labels = {
            TARGET_COIN: "金币",
            TARGET_DIAMOND: "钻石",
            TARGET_TREATS: "食物",
        }
        target_label = labels.get(target_name, "目标")

        for index, item in enumerate(batch):
            if self.isInterruptionRequested():
                return

            if index > 0:
                # Let the input layer settle before jumping the pointer again.
                self._sleep_timed(
                    self._rng.uniform(*self._cfg.click.batch_settle_delay)
                )
                if self.isInterruptionRequested():
                    return

                if recheck and not self._still_present(
                    item, scale, target_name=target_name
                ):
                    self._stale_skips += 1
                    self._emit_log(
                        LogLevel.WARN,
                        "%s 的坐标已过期（画面可能移动过），跳过，留待下轮"
                        % (item.center,),
                    )
                    continue

            self._set_state(BotState.ACTING)
            point = item.center
            if not self._timed("click", self._action.click, point[0], point[1]):
                self._consecutive_errors += 1
                self._emit_log(LogLevel.WARN, "点击 %s 投递失败" % (point,))
                continue

            self._clicks += 1
            self._consecutive_errors = 0
            self._guard.register_click(point)
            self._emit_log(
                LogLevel.SUCCESS,
                "已点击%s %s（置信度 %.2f）" % (target_label, point, item.confidence),
            )

            if self._should_verify():
                self._verify_click(point, target_name=target_name)

    def _still_present(
        self, item: Detection, scale: float, target_name: str = TARGET_COIN
    ) -> bool:
        """Confirm a batched target is still where the round's frame said it was.

        Returns False when there is no evidence, including when the capture
        fails. Absence of evidence must not become permission to click: the whole
        point is that the coordinate may already be describing the wrong pixels.
        """
        frame = self._timed("capture", self._window.capture)
        if frame is None:
            return False

        tolerance = scale_length(self._cfg.safety.position_tolerance, scale)
        target = item.center
        found = self._timed("detect", self._vision.detect, target_name, frame)
        for detection in found:
            if distance(target, detection.center) <= tolerance:
                return True
        return False

    # ------------------------------------------------------ verification policy

    def _should_verify(self) -> bool:
        """Decide whether to spend a re-read on this click.

        Verify everything until the run has proven itself, then sample. A single
        failed check resets to verifying everything, so a run that starts
        misbehaving is caught within one click rather than one sampling period.
        """
        safety = self._cfg.safety
        if not safety.verify_after_click:
            return False
        if safety.verify_sample_every <= 1:
            return True
        if self._verified_streak < safety.verify_trust_after:
            return True

        self._verify_countdown -= 1
        if self._verify_countdown <= 0:
            self._verify_countdown = safety.verify_sample_every
            return True
        return False

    def _verify_click(self, point: Point, target_name: str = TARGET_COIN) -> None:
        """Re-read a frame after clicking to confirm the target actually vanished.

        This is the only self-correcting part of the anti-misclick stack: if a
        position keeps surviving clicks it was never a target, and the guard bans
        it so the bot stops spinning on the same spot.
        """
        self._set_state(BotState.VERIFYING)
        started = time.perf_counter()
        self._interruptible_sleep(self._cfg.safety.verify_delay)
        if self.isInterruptionRequested():
            self._timings.add("verify", time.perf_counter() - started)
            return

        frame = self._window.capture()
        if frame is None:
            # No frame means no judgement; do not blame a good click on a bad read.
            self._timings.add("verify", time.perf_counter() - started)
            return

        height, width = frame.shape[:2]
        scale = scale_factor((width, height), self._cfg.vision.reference_size)
        tolerance = scale_length(self._cfg.safety.position_tolerance, scale)

        still_there = any(
            distance(point, d.center) <= tolerance
            for d in self._vision.detect(target_name, frame)
        )
        self._timings.add("verify", time.perf_counter() - started)

        self._verifications += 1
        if still_there:
            self._verified_failures += 1
            self._verified_streak = 0
            # Check the next click too, rather than waiting for the next sample.
            self._verify_countdown = 0
        else:
            self._verified_streak += 1

        note = self._guard.register_result(point, target_gone=not still_there)
        if note:
            self._emit_log(LogLevel.WARN, note)
        elif still_there:
            self._emit_log(
                LogLevel.WARN,
                "%s 点击后目标仍在，已记一次失败，该位置进入退避" % (point,),
            )

    # ---------------------------------------------------------------- helpers

    def _timed(self, stage: str, func, *args):
        """Call func, adding its wall time to the named stage."""
        started = time.perf_counter()
        try:
            return func(*args)
        finally:
            self._timings.add(stage, time.perf_counter() - started)

    def _sleep_timed(self, seconds: float) -> None:
        started = time.perf_counter()
        self._interruptible_sleep(seconds)
        self._timings.add("sleep", time.perf_counter() - started)

    def _record_miss(
        self, frame: np.ndarray, detections: Sequence[Detection], note: str
    ) -> None:
        """Persist a frame that yielded nothing clickable, when diagnostics are on."""
        path = self._recorder.record(frame, self._rounds, detections, note)
        if path:
            self._emit_log(LogLevel.INFO, "已保存取证画面: %s" % path)
            if self._recorder.exhausted:
                self._emit_log(
                    LogLevel.INFO,
                    "取证张数已达上限 %d，后续不再保存"
                    % self._cfg.diagnostics.max_dumps_per_session,
                )

    def _announce_resize(self, client_size: Tuple[int, int], scale: float) -> None:
        """Announce a window size change once, so adaptation is visible in the log."""
        if client_size == self._last_client_size:
            return
        self._last_client_size = client_size
        ref_w, ref_h = self._cfg.vision.reference_size
        self._emit_log(
            LogLevel.INFO,
            "窗口尺寸 %dx%d（参考 %dx%d，缩放 %.3f），已自适应"
            % (client_size[0], client_size[1], ref_w, ref_h, scale),
        )
        if abs(scale - 1.0) > 0.02:
            self._emit_log(
                LogLevel.WARN,
                "当前尺寸与模板的参考分辨率不同，模板会被缩放匹配。"
                "若识别率明显下降，请在该尺寸下重新裁一套模板",
            )

    def _interruptible_sleep(self, seconds: float) -> None:
        """Sleep that a stop request can cut short.

        A plain msleep(2000) would make the stop button feel up to two seconds
        laggy, so this sleeps in 50 ms slices and checks the flag between them.
        """
        if seconds <= 0:
            return
        remaining_ms = int(seconds * 1000)
        slice_ms = 50
        while remaining_ms > 0 and not self.isInterruptionRequested():
            step = min(slice_ms, remaining_ms)
            self.msleep(step)
            remaining_ms -= step

    def _set_state(self, state: BotState) -> None:
        if state is not self._state:
            self._state = state
            self.state_changed.emit(state.value)

    def _emit_log(self, level: LogLevel, message: str) -> None:
        logger.log(
            logging.WARNING
            if level in (LogLevel.WARN, LogLevel.ERROR)
            else logging.INFO,
            message,
        )
        self.log.emit(level.value, message)
