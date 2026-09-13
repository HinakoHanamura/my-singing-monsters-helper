"""Tour coordinator for traversing all islands on the map and harvesting resources."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Callable, List, Optional, Sequence, Set

from config import AppConfig
from core.letter_recognizer import LetterRecognizer
from core.map_navigator import IslandCardInfo, MapNavigator, ScreenState, hash_distance


@dataclass(frozen=True)
class QueuedIsland:
    """Represents a discovered island awaiting traversal during map tour."""

    name: str
    canon_name: str = ""
    card_hash: int = 0
    card_hist: Optional[Any] = None


class IslandTourCoordinator:
    """Orchestrates full-island tour traversal, queue management, and island entry."""

    def __init__(
        self,
        nav: MapNavigator,
        window: Any,
        action: Any,
        cfg: AppConfig,
        blacklist: Sequence[str],
        reset_map_to_top: bool,
        init_brake_mode: str,
        first_island_name: str,
        on_harvest: Callable[[], None],
        emit_log: Callable[[str, str], None],
        sleep_timed: Callable[[float], None],
        timed: Callable[..., Any],
        is_interrupted: Callable[[], bool],
        set_state: Optional[Callable[[Any], None]] = None,
    ) -> None:
        self._nav = nav
        self._window = window
        self._action = action
        self._cfg = cfg
        self._blacklist = blacklist
        self._reset_map_to_top = reset_map_to_top
        self._init_brake_mode = init_brake_mode
        self._first_island_name = first_island_name
        self._on_harvest = on_harvest
        self._emit_log = emit_log
        self._sleep_timed = sleep_timed
        self._timed = timed
        self._is_interrupted = is_interrupted
        self._set_state = set_state or (lambda state: None)

    def run(self) -> None:
        """Execute multi-island traversal with blacklist filtering and full resource collection."""
        self._set_state("searching")
        self._emit_log("INFO", "===== 开始执行「收集各岛资源」任务 =====")

        nav = self._nav
        if not self._window.ensure_attached():
            self._emit_log("ERROR", "未能连接到游戏窗口，退出巡岛")
            return

        frame = self._timed("capture", self._window.capture)
        if frame is None:
            self._emit_log("ERROR", "未能捕获游戏画面，退出巡岛")
            return

        visited_names: Set[str] = set()
        visited_hashes: List[int] = []
        island_queue: deque[QueuedIsland] = deque()
        last_anchor: Optional[QueuedIsland] = None

        names_fuzzy_match = LetterRecognizer.names_fuzzy_match

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
                if not c.name or not c.name.strip():
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
                    "INFO",
                    "【队列排队】 发现新岛屿 '%s'，加入待巡检队列（队列长度：%d）" % (c.name, len(island_queue)),
                )
            return enqueued

        # Clean up any startup popups/ads dynamically until no close button remains
        for _ in range(8):
            if self._is_interrupted():
                return
            frame = self._timed("capture", self._window.capture)
            if frame is None:
                break
            if hasattr(nav, "find_modal_cancel") and nav.find_modal_cancel(frame) is not None:
                self._emit_log("INFO", "【开局自愈】 检测到界面存在广告/弹窗遮挡，正在自动关闭…")
                nav.dismiss_modal(frame)
                self._sleep_timed(0.35)
            else:
                break

        frame = self._timed("capture", self._window.capture)
        if frame is None:
            self._emit_log("ERROR", "未能捕获游戏画面，退出巡岛")
            return

        cur_state = nav.detect_state(frame)

        if cur_state != ScreenState.MAP:
            self._emit_log("INFO", "正在打开地图界面以开始巡岛…")
            if not nav.open_map():
                self._emit_log("ERROR", "未能进入地图界面，退出巡岛")
                return
        else:
            self._emit_log("INFO", "检测到当前处于地图界面，执行地图初始化…")

        # Park cursor to eliminate hover glow effects before list scanning
        if hasattr(self._action, "park_cursor"):
            self._action.park_cursor()

        # Map Initialization: scroll to the very top so traversal begins at card 0
        if self._reset_map_to_top:
            if self._init_brake_mode == "first_island":
                self._emit_log("INFO", f"【地图初始化】 正在滑动置顶至目标首岛 '{self._first_island_name}'…")
            else:
                self._emit_log("INFO", "【地图初始化】 正在自适应滑动置顶岛屿列表…")
            nav.scroll_to_top()
            self._sleep_timed(0.4)

        consecutive_no_progress = 0
        target_miss_count = 0
        empty_cards_streak = 0

        while not self._is_interrupted():
            frame = self._timed("capture", self._window.capture)
            if frame is None:
                break

            cur_st = nav.detect_state(frame)
            if cur_st == ScreenState.ISLAND:
                self._emit_log(
                    "WARN",
                    "【界面校准】 检测到当前画面在岛屿界面（MAP 按钮存在），正在打开地图界面…",
                )
                if not nav.open_map():
                    self._emit_log("ERROR", "未能返回地图界面，巡岛中止")
                    return
                continue
            elif cur_st != ScreenState.MAP:
                self._emit_log(
                    "WARN",
                    "【界面校准】 检测到当前画面未在地图界面（状态: %s），正在解除界面遮挡…" % cur_st.value,
                )
                if hasattr(nav, "dismiss_modal") and nav.dismiss_modal(frame):
                    self._emit_log("INFO", "【弹窗自愈】 检测到界面存在弹窗/全屏遮挡，已通过红 X/ESC 成功关闭…")
                    self._sleep_timed(0.3)
                    continue
                if not nav.open_map():
                    self._emit_log("ERROR", "未能返回地图界面，巡岛中止")
                    return
                continue

            cards = nav.get_visible_cards(frame)
            if not cards:
                empty_cards_streak += 1
                if hasattr(nav, "dismiss_modal") and nav.dismiss_modal(frame):
                    self._emit_log(
                        "INFO",
                        "【弹窗自愈】 检测到弹窗阻挡地图卡片列表，已通过红 X/ESC 成功关闭",
                    )
                    empty_cards_streak = 0
                    self._sleep_timed(0.3)
                    continue

                self._emit_log("WARN", "未检测到可见岛屿卡片，尝试滑动列表…")
                nav.scroll_down()
                continue
            else:
                empty_cards_streak = 0

            # Update queue with cards discovered below current anchor
            scan_and_enqueue_downward(cards)

            # Check if queue is empty
            if not island_queue:
                self._emit_log("INFO", "当前视野待访队列为空，向下滑动探索后续新岛屿…")
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
                            "SUCCESS",
                            "【巡岛完成】 待访队列已清空且向下滑动无新岛屿，所有岛屿收集完毕！",
                        )
                        break
                else:
                    consecutive_no_progress = 0
                continue

            # Peek next target in queue
            target = island_queue[0]
            target_disp_name = target.name or target.canon_name

            # Check if target island is blacklisted
            recognizer = getattr(nav, "letter_recognizer", getattr(nav, "_recognizer", None))
            if self._blacklist and recognizer:
                if recognizer.is_blacklisted(target.name, self._blacklist):
                    self._emit_log(
                        "WARN",
                        "【黑名单拦截】 岛屿 '%s' 命中用户黑名单，跳过采集" % target_disp_name,
                    )
                    island_queue.popleft()
                    target_miss_count = 0
                    for c in cards:
                        if match_card_to_island(c, target):
                            mark_card_visited(c)
                            last_anchor = target
                            break
                    continue

            # Find matching card in current visible viewport
            target_card: Optional[IslandCardInfo] = None
            for c in cards:
                if match_card_to_island(c, target):
                    target_card = c
                    break

            if target_card is not None and target_card.is_fully_visible:
                # Target located and fully visible: proceed to enter
                island_queue.popleft()
                target_miss_count = 0

                self._emit_log("INFO", "【前往岛屿】 正在选中卡片 '%s'…" % target_disp_name)
                nav.select_island(target_card)

                self._emit_log("INFO", "【进入岛屿】 正在确认进入 '%s'…" % target_disp_name)
                entered = nav.enter_selected_island(target_card=target_card)
                if not entered:
                    # Double check if screen actually entered island (state self-healing)
                    fresh_frame = self._timed("capture", self._window.capture)
                    if fresh_frame is not None and nav.detect_state(fresh_frame) == ScreenState.ISLAND:
                        entered = True
                        self._emit_log("INFO", "【状态自愈】 画面已确认进入岛屿 '%s'" % target_disp_name)
                    else:
                        self._emit_log("WARN", "未能进入岛屿 '%s'，保留待巡检队列稍后重试" % target_disp_name)
                        island_queue.appendleft(target)
                        continue

                # Inside island: run resource collection pipeline
                self._emit_log(
                    "SUCCESS",
                    "【岛上采集】 已成功进入 '%s'，开始执行资源全收…" % target_disp_name,
                )
                self._on_harvest()

                if self._is_interrupted():
                    break

                # Return to map with retry
                self._emit_log("INFO", "【返回地图】 资源收集完毕，正在返回地图界面…")
                return_ok = False
                for _ in range(3):
                    if nav.open_map():
                        return_ok = True
                        break
                    self._sleep_timed(0.5)

                if not return_ok:
                    self._emit_log("ERROR", "未能从岛屿返回地图，巡岛中止")
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
                    # Target is visible in viewport but partially clipped by bottom edge: advance slightly
                    self._emit_log(
                        "INFO",
                        "待访目标 '%s' 位于下方边缘，微距推进至全貌视野…" % target_disp_name,
                    )
                    nav.scroll_down()
                    continue
                else:
                    target_miss_count += 1
                    if target_miss_count <= 2:
                        self._emit_log(
                            "INFO",
                            "待访目标 '%s' 暂未在当前视野出现，向下推进寻找（尝试 %d/2）…" % (target_disp_name, target_miss_count),
                        )
                        nav.scroll_down()
                        continue
                    else:
                        # Target missing after 2 consecutive downward scrolls; pop it to break endless scroll cascade
                        self._emit_log(
                            "WARN",
                            "待访目标 '%s' 连续滑动未现身，已从队列移出以防阻碍后续岛屿" % target_disp_name,
                        )
                        island_queue.popleft()
                        target_miss_count = 0
                        continue
