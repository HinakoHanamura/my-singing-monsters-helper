"""Run the memory minigame against the live game, by hand.

Why this is not an automated test
---------------------------------
Everything in the minigame pipeline is covered by unit tests and by replaying the
recordings, but none of that touches the real game. Three things can only be
answered here: whether the card back really segments the same way on a live,
animating window, whether clicks land where the slot map says, and whether the
loop's waits are long enough for animations this machine actually renders.

Its record justifies itself. Three separate faults were found here and nowhere
else: a glow threshold chosen from a description rather than measured, that same
threshold turning out to be unfixable in principle, and a patch of violet *inside*
revealed artwork that the detector accepted as a card back. None was reachable
from the recordings.

Note that the UI can now run the minigame too (the "记忆小游戏" button). This
script remains the diagnostic route: observe mode prints the per-slot numbers
behind every gate the runner applies, which the UI deliberately does not.

Two modes, and why the safe one is the default
----------------------------------------------
``--observe`` (default) clicks nothing at all. It captures, detects, gates the
screen, builds the slot map and then keeps reporting what it sees. That is enough
to confirm the perception half at zero risk.

``--play`` hands control to :class:`MemoryRunner`, which does click, and keeps
going until boards stop appearing. Losing a level opens a prompt offering a replay
for two diamonds, so the loop is built so that no click outside a card is
reachable -- see the module docstring of ``core/minigames/memory_runner.py``.

Comparing the two flip orders
-----------------------------
``--scan-first`` selects the same order the UI's "加速配对" checkbox does: turn
every card over before collecting anything. It costs at most one more miss than
the default and both fit the game's allowance, so which one is *faster* is a
measurement rather than an argument. Run each order once and compare the per-level
seconds this script prints at the end.

Before running
--------------
Open the minigame by hand and stop at the **opening board of a level**, with every
card face down. The slot map can only be built from a full layout, and an even
number of already-revealed cards is indistinguishable from a complete board, so
starting midway would leave the bot blind to whatever it missed. It detects that
and stops, but starting correctly avoids the whole question.

The window may be covered and need not be in the foreground. It must not be
minimised: Windows stops rendering the client area, and there is nothing to
capture.

What the observe table means
----------------------------
For every card that is face up, three numbers decide whether the runner would be
able to fingerprint it, and all three are printed:

    sim    correlation with the same slot one capture earlier. The runner needs
           this at or above the stability threshold before it will sample a card.
           Card faces here are static images, so a settled card reads exactly
           1.000; anything lower means the card is still animating.
    glow   share of the card matching the match-flash colour window. Reported
           because it helps when diagnosing, but it no longer gates anything:
           ordinary artwork was measured reaching the old cutoff, and no cutoff
           can be justified when there is no bound on ordinary art.
    std    texture of the crop. Correlation is undefined on a uniform patch, so
           a crop below the floor is refused rather than trusted.

Usage
-----
    cd /d <repository root>
    python tests/manual/test_memory_minigame.py                 # observe only
    python tests/manual/test_memory_minigame.py --seconds 30
    python tests/manual/test_memory_minigame.py --play --max-levels 1
    python tests/manual/test_memory_minigame.py --play           # until it ends
    python tests/manual/test_memory_minigame.py --play --scan-first

Annotated frames land in ``reports/manual_memory/``. When something looks wrong,
``python tools/explain_card_backs.py reports/manual_memory/stopped.png --debug``
prints the per-box measurements for that exact frame, which is how the violet-
inside-artwork problem was pinned down.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import cv2  # noqa: E402

from config import DEFAULT_CONFIG  # noqa: E402
from core.action_agent import ActionAgent  # noqa: E402
from core.game_window import GameWindow  # noqa: E402
from core.minigames.board import SlotMap  # noqa: E402
from core.minigames.card_vision import (  # noqa: E402
    CARD_BACK,
    CardBackVision,
    board_screen_verdict,
)
from core.minigames.fingerprint import (  # noqa: E402
    DEFAULT_FINGERPRINT_PARAMS,
    crop_face,
    glow_score,
    patch_std,
    similarity,
)
from core.minigames.memory_runner import MemoryRunner, RunnerParams  # noqa: E402

REPORT_DIR = os.path.join(PROJECT_ROOT, "reports", "manual_memory")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the memory minigame against the live game."
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="actually play (clicks cards). Without this, nothing is clicked.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=15.0,
        help="how long to observe in observe mode (default 15)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="seconds between captures in observe mode (default 0.5)",
    )
    parser.add_argument(
        "--max-levels",
        type=int,
        default=None,
        dest="max_levels",
        help=(
            "stop after this many levels. Omit to play until the boards stop "
            "appearing, which is how a run actually ends."
        ),
    )
    parser.add_argument(
        "--scan-first",
        action="store_true",
        dest="scan_first",
        help=(
            "turn every card over before collecting any pair, the same order as "
            "the UI's 加速配对 checkbox. One more miss at worst; run both and "
            "compare the per-level seconds."
        ),
    )
    return parser


def attach_window() -> Optional[GameWindow]:
    window = GameWindow(
        title=DEFAULT_CONFIG.window.title,
        print_window_flag=DEFAULT_CONFIG.window.print_window_flag,
    )
    if not window.ensure_attached():
        print("找不到游戏窗口：%r" % DEFAULT_CONFIG.window.title)
        print("请确认游戏正在运行，且窗口标题与 config.py 中的一致。")
        return None
    width, height = window.client_size()
    print("已附着窗口，客户区 %dx%d" % (width, height))
    reference = tuple(DEFAULT_CONFIG.vision.reference_size)
    if (width, height) != reference:
        print("注意：与参考分辨率 %s 不同，坐标会按比例换算。" % (reference,))
    return window


def save_annotated(frame, boxes, slot_map: Optional[SlotMap], name: str) -> str:
    """Write a frame with detections drawn on it, for checking by eye later."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    annotated = frame.copy()
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if slot_map is not None:
        for slot in range(slot_map.slot_count):
            x1, y1, _, _ = slot_map.box(slot)
            cv2.putText(
                annotated,
                str(slot),
                (x1 + 4, y1 + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
    path = os.path.join(REPORT_DIR, name)
    cv2.imwrite(path, annotated)
    return path


def observe(window: GameWindow, seconds: float, interval: float) -> int:
    """Watch the board without clicking anything."""
    vision = CardBackVision()
    params = DEFAULT_FINGERPRINT_PARAMS

    print("")
    print("=== 观察模式：不会发出任何点击 ===")
    print("")

    frame = window.capture()
    if frame is None:
        print("截图失败。窗口是否被最小化了？最小化后客户区停止渲染。")
        return 1

    boxes = [d.bbox for d in vision.detect(CARD_BACK, frame)]
    verdict = board_screen_verdict(boxes)
    print("检出卡背 %d 个｜门禁 %s：%s" % (len(boxes), verdict.code, verdict.reason))
    print("卡牌宽度极差 %d px" % verdict.spread)

    if not verdict.ok:
        path = save_annotated(frame, boxes, None, "rejected.png")
        print("")
        print("这一帧不被当作可玩的盘面。标注图：%s" % path)
        print("请确认已经手动进入某一关的开局界面，且所有卡牌都是背面朝下。")
        print("想知道检测器在这一帧上究竟看到了什么，跑：")
        print("  python tools/explain_card_backs.py reports/manual_memory/rejected.png --debug")
        return 1

    slot_map = SlotMap(boxes)
    print("")
    print("槽位表已建立：%s" % slot_map.initial_shape)
    print(
        "卡牌中位尺寸 %dx%d｜匹配容差 %.1f px"
        % (slot_map.card_size[0], slot_map.card_size[1], slot_map.match_tolerance())
    )
    path = save_annotated(frame, boxes, slot_map, "board.png")
    print("标注图（含槽位编号）：%s" % path)
    print("")
    print("接下来持续观察 %.0f 秒。**请手动翻开一两张牌**，重点看 sim 一列。" % seconds)
    print(
        "sim 要达到 %.2f 求解器才肯取样；std 要高于 %.1f。glow 只做诊断，不再拦截。"
        % (params.stability_threshold, params.min_patch_std)
    )
    print("")
    print("%-7s %-6s %-7s %-16s %s" % ("时刻", "背面", "已翻开", "code", "已翻开槽位的三个数"))
    print("-" * 88)

    started = time.monotonic()
    drifts = 0
    unstable_seen = False
    previous_crops: Dict[int, np.ndarray] = {}
    sim_samples: List[float] = []

    while time.monotonic() - started < seconds:
        time.sleep(interval)
        frame = window.capture()
        if frame is None:
            print("%-7.1f 截图失败" % (time.monotonic() - started))
            continue

        boxes = [d.bbox for d in vision.detect(CARD_BACK, frame)]
        observation = slot_map.observe(boxes)

        detail = ""
        if not observation.ok:
            drifts += 1
            detail = observation.reason
        else:
            pieces = []
            for slot in sorted(observation.face_up):
                box = slot_map.box(slot)
                crop = crop_face(frame, box, params)
                if crop is None:
                    pieces.append("%d:裁片不可用" % slot)
                    continue
                earlier = previous_crops.get(slot)
                previous_crops[slot] = crop
                glow = glow_score(frame, box, params)
                std = patch_std(crop)
                if earlier is None:
                    pieces.append("%d:sim=首帧 glow=%.2f std=%.0f" % (slot, glow, std))
                    continue
                sim = similarity(earlier, crop, params)
                sim_samples.append(sim)
                mark = "" if sim >= params.stability_threshold else " 不稳定"
                if sim < params.stability_threshold:
                    unstable_seen = True
                pieces.append(
                    "%d:sim=%.3f glow=%.2f std=%.0f%s" % (slot, sim, glow, std, mark)
                )
            # Slots that turned back over should not keep a stale crop around.
            for slot in list(previous_crops):
                if slot in observation.face_down:
                    previous_crops.pop(slot, None)
            detail = "  ".join(pieces)

        print(
            "%-7.1f %-6s %-7s %-16s %s"
            % (
                time.monotonic() - started,
                len(observation.face_down) if observation.ok else "-",
                len(observation.face_up) if observation.ok else "-",
                observation.code,
                detail,
            )
        )

    print("")
    if drifts:
        print("出现 %d 次几何漂移。若窗口没有被移动或缩放，这是需要查的信号。" % drifts)
    else:
        print("全程没有几何漂移，槽位编号稳定。")

    if sim_samples:
        array = np.array(sim_samples)
        print(
            "sim 统计：n=%d  最低 %.3f  中位 %.3f  最高 %.3f（阈值 %.2f）"
            % (
                array.size,
                array.min(),
                float(np.median(array)),
                array.max(),
                params.stability_threshold,
            )
        )
        if unstable_seen:
            print("低于阈值的样本应当只出现在翻牌、翻回、配对高光这些动画瞬间。")
            print("若已沉降的牌也长期低于阈值，才说明稳定性阈值需要重新标定。")
        else:
            print("全部样本达标。")
    else:
        print("没有采到 sim 样本——观察期间没有牌保持翻开状态。请重跑并手动翻一张牌。")

    print("观察结束，未发出任何点击。")
    return 0


def play(
    window: GameWindow, max_levels: Optional[int], scan_first: bool = False
) -> int:
    """Hand control to the runner, which does click."""
    action = ActionAgent(window, DEFAULT_CONFIG.click)
    runner = MemoryRunner(
        window=window,
        action=action,
        params=RunnerParams(max_levels=max_levels, scan_first=scan_first),
        log=lambda message: print("  " + message),
    )

    order = "先全部翻开（加速配对）" if scan_first else "边翻边收（默认）"
    print("")
    print("=== 实战模式：会真的点击卡牌 ===")
    print("翻牌策略：%s" % order)
    if max_levels is None:
        print("一直打到盘面不再出现（结算画面）为止。中途可以按 Ctrl+C 停止。")
    else:
        print("最多 %d 关。中途可以按 Ctrl+C 停止。" % max_levels)
    print("")

    started = time.monotonic()
    try:
        result = runner.run()
    except KeyboardInterrupt:
        print("")
        print("已被手动中断。已完成 %d 关。" % runner.levels_cleared)
        report_level_times(runner, order)
        return 130

    elapsed = time.monotonic() - started
    print("")
    print("结束：%s — %s" % (result.code, result.message))
    print(
        "已完成 %d 关，翻牌 %d 轮，耗时 %.1f 秒。"
        % (runner.levels_cleared, runner.turns_played, elapsed)
    )
    report_level_times(runner, order)

    if not result.ok:
        frame = window.capture()
        if frame is not None:
            vision = CardBackVision()
            boxes = [d.bbox for d in vision.detect(CARD_BACK, frame)]
            path = save_annotated(frame, boxes, None, "stopped.png")
            print("停止时的画面已存到：%s" % path)
            print("逐框测量：")
            print("  python tools/explain_card_backs.py reports/manual_memory/stopped.png --debug")
        return 1
    return 0


def report_level_times(runner: MemoryRunner, order: str) -> None:
    """Print the per-level seconds, which is what settles the strategy question.

    Printed even on an interrupt or a stop: a partial run of each order is still
    comparable, and throwing the numbers away because the run ended early would
    waste the only measurement that can answer the question.
    """
    if not runner.level_seconds:
        return
    print("")
    print("每关耗时（策略 %s）：" % order)
    for index, seconds in enumerate(runner.level_seconds, start=1):
        print("  第 %d 关  %.1f 秒" % (index, seconds))
    average = sum(runner.level_seconds) / len(runner.level_seconds)
    print("  平均    %.1f 秒" % average)
    print("请用两种策略各跑一轮，再比这一行。")


def main() -> int:
    args = build_parser().parse_args()

    window = attach_window()
    if window is None:
        return 1

    try:
        if args.play:
            return play(window, args.max_levels, args.scan_first)
        return observe(window, args.seconds, args.interval)
    finally:
        window.detach()


if __name__ == "__main__":
    sys.exit(main())
