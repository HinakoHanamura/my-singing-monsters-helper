"""Resource collection pipeline for single-island and full-tour harvesting."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from config import (
    TARGET_COIN,
    TARGET_DIAMOND,
    TARGET_MODAL_CONFIRM,
    TARGET_PIGGY_BANK,
    TARGET_TREATS,
    AppConfig,
)
from core.action_agent import ActionAgent
from core.click_guard import ClickGuard
from core.game_window import GameWindow
from core.geometry import scale_factor, scale_length, select_independent
from core.vision_agent import BaseVisionAgent, check_piggy_status


@dataclass(frozen=True)
class ResourceOptions:
    """Options for which resource types to harvest during tours or full collections."""

    piggy: bool = True
    diamond: bool = True
    treats: bool = True
    coin: bool = True


class ResourceHarvestPipeline:
    """Orchestrates detection, validation, and collection of resources on an island."""

    def __init__(
        self,
        window: Any,
        action: Any,
        vision: BaseVisionAgent,
        guard: ClickGuard,
        cfg: AppConfig,
        emit_log: Callable[[str, str], None],
        set_state: Callable[[Any], None],
        sleep_timed: Callable[[float], None],
        timed: Callable[..., Any],
        filter_candidates: Callable[..., Any],
        click_batch: Callable[..., None],
        is_interrupted: Callable[[], bool],
        on_click: Optional[Callable[[int], None]] = None,
        on_round: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[], None]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._window = window
        self._action = action
        self._vision = vision
        self._guard = guard
        self._cfg = cfg
        self._emit_log = emit_log
        self._set_state = set_state
        self._sleep_timed = sleep_timed
        self._timed = timed
        self._filter_candidates = filter_candidates
        self._click_batch = click_batch
        self._is_interrupted = is_interrupted
        self._on_click = on_click or (lambda count: None)
        self._on_round = on_round or (lambda: None)
        self._on_error = on_error or (lambda: None)
        self._rng = rng or np.random.default_rng()

    def run_piggy_stage(self) -> bool:
        """Attempt to activate and confirm the piggy bank if bright.

        Returns True if the piggy bank was clicked and confirmed closed, False otherwise.
        """
        # BotState.SEARCHING
        self._set_state("searching")
        if not self._window.ensure_attached():
            self._emit_log("WARN", "未找到游戏窗口，跳过储蓄罐")
            return False

        frame = self._timed("capture", self._window.capture)
        if frame is None:
            self._emit_log("WARN", "截图失败，跳过储蓄罐")
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
                self._emit_log("INFO", "未检测到小猪储蓄罐图标（本岛无储蓄罐或被遮挡），跳过")
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
                self._emit_log("INFO", "小猪储蓄罐处于黯淡状态（无可收集货币），跳过")
                return False

            self._set_state("acting")
            self._emit_log("INFO", "检测到高光储蓄罐，执行点击...")
            if not self._timed(
                "click", self._action.click, piggy.center[0], piggy.center[1]
            ):
                self._emit_log("WARN", "点击储蓄罐失败")
                return False

            self._on_click(1)

        # Wait for modal confirmation button to appear and click with retry until closed
        self._set_state("searching")
        modal_deadline = time.monotonic() + self._cfg.vision.modal_timeout
        confirmed_any = False

        while time.monotonic() < modal_deadline and not self._is_interrupted():
            self._sleep_timed(0.2)
            mframe = self._timed("capture", self._window.capture)
            if mframe is None:
                continue

            confirms = self._timed(
                "detect", self._vision.detect, TARGET_MODAL_CONFIRM, mframe
            )
            if confirms:
                confirm_btn = confirms[0]
                self._set_state("acting")
                self._emit_log("INFO", "检测到确认弹窗，点击绿色确认按钮...")
                self._timed(
                    "click",
                    self._action.click,
                    confirm_btn.center[0],
                    confirm_btn.center[1],
                )
                self._on_click(1)
                confirmed_any = True
                self._sleep_timed(0.35)
                continue

            if confirmed_any:
                self._emit_log("SUCCESS", "储蓄罐收集完成，确认弹窗已关闭")
                self._sleep_timed(0.4)
                return True

        if not confirmed_any:
            self._emit_log("WARN", "等待确认弹窗超时，未发现确认按钮")
            return False

        self._emit_log("SUCCESS", "储蓄罐收集完成")
        self._sleep_timed(0.4)
        return True

    def collect_target_until_empty(
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
        clicks_collected = 0

        for _ in range(max_rounds):
            if self._is_interrupted():
                break

            self._on_round()

            if not self._window.ensure_attached():
                self._on_error()
                self._sleep_timed(self._cfg.loop.retry_interval)
                continue

            frame = self._timed("capture", self._window.capture)
            if frame is None:
                self._on_error()
                self._sleep_timed(self._cfg.loop.retry_interval)
                continue

            height, width = frame.shape[:2]
            client_size = (width, height)
            scale = scale_factor(client_size, self._cfg.vision.reference_size)
            self._action.set_scale(scale)
            self._guard.set_scale(scale)

            self._set_state("searching")
            self._guard.begin_frame()
            targets = self._timed("detect", self._vision.detect, target_name, frame)
            self._guard.observe(targets)

            if not targets:
                consecutive_empty += 1
                self._emit_log(
                    "INFO",
                    f"未发现可收集的{label}（{consecutive_empty}/{max_consecutive_empty}）",
                )
                if consecutive_empty >= max_consecutive_empty:
                    break
                self._sleep_timed(self._rng.uniform(*self._cfg.loop.tick_interval))
                continue

            consecutive_empty = 0
            self._set_state("validating")
            candidates, rejections = self._timed(
                "validate",
                self._filter_candidates,
                targets,
                frame,
                client_size,
                scale,
                target_name,
            )

            if not candidates:
                summary = f"识别到 {len(targets)} 个{label}，通过 0 个"
                if rejections:
                    summary += f" ｜ 过滤: {rejections.describe()}"
                self._emit_log("INFO", summary)
                self._sleep_timed(self._rng.uniform(*self._cfg.loop.tick_interval))
                continue

            batch = select_independent(
                candidates,
                self._cfg.loop.max_clicks_per_tick,
                scale_length(self._cfg.loop.click_separation_margin, scale, minimum=0),
            )

            summary = (
                f"识别到 {len(targets)} 个{label}，通过 {len(candidates)} 个，本轮点 {len(batch)} 个"
            )
            if len(batch) < len(candidates):
                summary += "（其余重叠，留待下轮）"
            if rejections:
                summary += f" ｜ 过滤: {rejections.describe()}"
            self._emit_log("INFO", summary)

            self._click_batch(batch, scale, target_name=target_name)
            self._on_click(len(batch))
            clicks_collected += len(batch)
            self._sleep_timed(self._rng.uniform(*self._cfg.loop.tick_interval))

        return clicks_collected

    def run(self, options: Optional[ResourceOptions] = None) -> None:
        """Run the composite resource collection pipeline in sequence according to options."""
        opts = options or ResourceOptions()
        self._emit_log("INFO", "===== 开始执行「资源全收」流水线 =====")

        # Stage 1: Piggy Bank
        if self._is_interrupted():
            return
        if opts.piggy:
            self._emit_log("INFO", "【阶段 1/4】 检查小猪储蓄罐...")
            piggy_ok = self.run_piggy_stage()
            if piggy_ok:
                self._emit_log("SUCCESS", "【阶段 1/4】 储蓄罐全收完成")
            else:
                self._emit_log("INFO", "【阶段 1/4】 储蓄罐已跳过，继续后续流程")
        else:
            self._emit_log("INFO", "【阶段 1/4】 储蓄罐已由设置跳过，继续后续流程")

        # Stage 2: Diamond
        if self._is_interrupted():
            return
        if opts.diamond:
            self._emit_log("INFO", "【阶段 2/4】 收集钻石...")
            d_clicks = self.collect_target_until_empty(
                TARGET_DIAMOND, max_rounds=6, max_consecutive_empty=4
            )
            self._emit_log("INFO", f"【阶段 2/4】 钻石收集完毕（点击 {d_clicks} 次）")
        else:
            self._emit_log("INFO", "【阶段 2/4】 钻石已由设置跳过，继续后续流程")

        # Stage 3: Treats
        if self._is_interrupted():
            return
        if opts.treats:
            self._emit_log("INFO", "【阶段 3/4】 收集食物 (Treats)...")
            t_clicks = self.collect_target_until_empty(
                TARGET_TREATS, max_rounds=30, max_consecutive_empty=4
            )
            self._emit_log("INFO", f"【阶段 3/4】 食物收集完毕（点击 {t_clicks} 次）")
        else:
            self._emit_log("INFO", "【阶段 3/4】 食物已由设置跳过，继续后续流程")

        # Stage 4: Coins
        if self._is_interrupted():
            return
        if opts.coin:
            self._emit_log("INFO", "【阶段 4/4】 收集金币...")
            c_clicks = self.collect_target_until_empty(
                TARGET_COIN, max_rounds=40, max_consecutive_empty=4
            )
            self._emit_log("INFO", f"【阶段 4/4】 金币收集完毕（点击 {c_clicks} 次）")
        else:
            self._emit_log("INFO", "【阶段 4/4】 金币已由设置跳过，继续后续流程")

        self._emit_log("SUCCESS", "【资源全收完毕】 本岛指定资源收集流程已顺利完成")
