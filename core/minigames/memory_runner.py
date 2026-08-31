"""Drive the memory minigame: capture, decide, click, verify, repeat.

Where this sits
---------------
This is the minigame's own control loop, not a reuse of ``BotEngine``. The coin
patrol is an endless watch over a scene that changes on its own; a memory board is
a closed puzzle with a terminal state, driven entirely by the bot's own clicks.

    GameWindow      background capture (PrintWindow)
    CardBackVision  where the face-down cards are, by colour
    SlotMap         which card is which, re-measured every capture
    FaceRegistry    whether two revealed cards show the same picture
    MemorySolver    which card to flip next
    ActionAgent     background click (SendMessage)

The game is the authority on the outcome of a turn
--------------------------------------------------
An earlier version decided whether two cards matched by comparing its own
fingerprints. That was the wrong source of truth, and live runs showed why.

The game announces the result unambiguously and for free: a matched pair stays
face up forever, a mismatched pair turns back over. Fingerprinting is a *guess*
about the same question, taken at the worst possible moment -- the second card of
a turn is sampled exactly while the match celebration washes its artwork out. On a
four-card board that produced a false "mismatch" on the final pair, followed by a
three-second wait for a flip back that was never going to happen.

So the fingerprint is a *hint* and observation is the authority. Where the hint
says the cards agree there is something specific to confirm, which is quick; where
it does not, the flip back is waited for and its absence is itself the answer.
That inverts the cost of a fingerprint error: it used to strand the level, now it
costs a little waiting and corrects itself.

...except once the level is over, when there is nothing left to observe
-----------------------------------------------------------------------
The one thing the board cannot be asked about is a board that is no longer there.
When the last pair matches, the level ends and the game replaces it with the next
one, which has a different number of cards in different places. A look taken after
that point measures the *new* board against the finished level's slot map, and the
honest reading of "six cards in four slots" is that the geometry has diverged --
which stops the run.

A live four-card level laid out AABB found this. Both scanning turns matched, so
there was no flip back to wait out and the level was over about a second after it
began; the next board was already up by the time the loop looked again, and the run
stopped with ``GEOMETRY_DRIFT`` having done nothing wrong.

The fix is not more patience, it is not looking. Every pair matched means the level
is finished, and that is known from the record without a capture. So
:meth:`play_turn` asks the solver first, and only reaches for the camera when there
is still something on the board to decide about.

How many misses this can cost
-----------------------------
A miss requires the first card flipped to be the *first* copy of its picture to be
seen -- otherwise it matches something known and the pair is simply collected. So
every miss consumes a first copy, of which there is one per pair. The opening turn
consumes two of them, because nothing is known yet, and every later miss consumes
at least one. With ``n`` pairs that gives

    misses <= 1 + (n - 2) = n - 1

and the bound is tight: a board laid out so that the second card of every turn
after the first duplicates an already-seen picture reaches it exactly. The game
grants ceil(1.5 * n), so there is room to spare.

Two orders to flip in
---------------------
``scan_first`` selects between them.

*Opportunistic* (default) walks the unread cards in reading order but collects any
pair it can already complete, including one it stumbles into mid-turn. Worst case
n - 1 misses, as above.

*Scan first* splits the level into two phases: turn over every card in order to
learn the whole board, then collect all n pairs. Its worst case is n misses --
exactly one more -- because with partners never sharing a turn, every one of the n
scanning turns misses.

So scanning is strictly worse on misses, and both fit the budget. Whether it is
*faster* is a separate question that arithmetic cannot settle, which is why every
level reports how long it took.

What the scanning phase can do that the other order cannot
----------------------------------------------------------
A scanning turn is not trying to complete a pair. Its two cards are two
independent reads that the game allows at the same time, so there is no reason to
finish one before starting the other: both are clicked, then both are read from
*the same* captures.

Deciding that a card has stopped moving is the expensive part of a turn, and one
sequence of frames answers it for every card that is up. So a scanning turn pays
for one settling wait instead of two.

    scanning turn    click, click, settle once, read both
    ordinary turn    click, settle, read, choose, click, settle, read

The opportunistic order must not do this, and the reason is exactly what makes it
opportunistic. There the second card is chosen *because* of what the first turned
out to be: reading the first is what lets ``partner_for`` offer a card that
completes a pair instead of a card that merely has not been seen. Clicking blind
would trade a collection for a miss. In the scanning phase there is nothing to
trade, because completing pairs is deferred on purpose.

Clicking the second card without waiting is not a new assumption:
:meth:`_collect_known_pair` has always done it, with nothing in between at all,
across ten live levels. Nothing here needs the game to be more permissive than it
has already been observed to be.

What is still dead time, and what would remove it
-------------------------------------------------
After both cards are read, a mismatched pair still has to turn back over before
the next turn, which live observation put at roughly 1.6s. Both cards were read
well before that finishes, so the remainder is spent watching an animation.

Removing it needs a fact about the game rather than an argument: whether a click on
a third card is accepted while a mismatched pair is still on display. If it is, the
next turn can start immediately and the scanning phase loses its largest remaining
cost. If it is not, a speculative click would be silently dropped and the wait
would happen anyway, so this is a measurement rather than something to assume.

Reading a card is the expensive part, so it is skipped whenever possible
-----------------------------------------------------------------------
Sampling means polling until the artwork holds still. Two kinds of turn need none:
collecting a pair whose faces are both on file, and finishing a pair a probe just
walked into, where the solver's second choice is a known partner. Only turns that
actually learn something pay for sampling.

Polling rate and the stability gap are separate concerns
--------------------------------------------------------
A capture plus detection costs roughly 100ms here, so sleeping another 120ms
between looks doubled every wait for nothing. Waiting for a card to turn over, or
for a pair to turn back, wants the fastest polling available.

Judging that a card has *stopped changing* is the one thing that does need a real
interval between the two frames compared, because two captures taken microseconds
apart look identical whatever the card is doing. So that interval is stated
separately and applied only where it is the point.

Verification depth follows the miss budget
------------------------------------------
Extra confirmation buys insurance against a misread, and insurance is worth
different amounts depending on how much slack is left. The margin is

    ceil(1.5 * pairs) - misses so far - worst case for the cards still unread

and while it is comfortable the loop samples with a single stability confirmation.
As it tightens, full confirmation comes back. This is only safe because a misread
is now recoverable.

Waiting for the board to go quiet is skipped when it is already known to be
--------------------------------------------------------------------------
Sampling wants a still board, because a celebration elsewhere would keep the crop
moving and be blamed on the card being read. But a turn that ended in a mismatch
has already been watched until both its cards were face down again, which *is* the
board being still. Asking again afterwards measures nothing, so the level tracks
whether its last turn left the board quiet.

Recovering instead of stopping, where the board can say what is true
-------------------------------------------------------------------
Three failures used to end the run: a card the record thought was finished turning
out to be face down again, more distinct pictures on file than the level has pairs,
and a solver with cards left but no legal move. All three mean the record and the
board have diverged, and all three are repairable, because the board itself says
which cards are done -- anything not face down at the start of a turn has been
matched.

So the response is to rebuild the level's knowledge from that observation and carry
on, at the cost of re-reading the cards still in play. It is capped, because a
repair that keeps being needed is a real fault rather than a hiccup, and an
uncapped loop would hide it.

Safety: why REPLAY cannot be clicked
------------------------------------
Losing a level opens a prompt offering a replay for two diamonds -- real currency.
Rather than blacklist that button's coordinates, no click outside a card is
reachable at all: every target comes from a card-back detection in the frame just
captured, :meth:`_click_slot` re-checks that the target still lies inside one of
those boxes, and the results screen produces no card-back boxes.

When a run ends
---------------
Boards stop appearing. The results screen is recognisable by absence, so
:meth:`seek_board` running out of patience means there is no board left; whether
that is good news depends on whether anything was cleared first. The patience
window is several times a level transition, so a transition cannot be mistaken for
the end.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from config import DEFAULT_CONFIG
from core.geometry import PixelRect, point_in_rect, scale_factor
from core.minigames.board import CODE_STRAY_BOXES, BoardObservation, SlotMap
from core.minigames.card_vision import (
    CARD_BACK,
    CardBackVision,
    board_screen_verdict,
)
from core.minigames.fingerprint import (
    DEFAULT_FINGERPRINT_PARAMS,
    UNCOMPARABLE,
    FaceRegistry,
    FingerprintParams,
    crop_face,
    looks_stable,
    similarity,
)
from core.minigames.memory_game import FaceKey, MemorySolver

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunnerParams:
    """Timing, patience and strategy."""

    # Sleep between looks while waiting for the board to change state.
    #
    # Near zero on purpose: a look already costs about 100ms of capture and
    # detection, which is the real polling period. The previous 120ms sleep on top
    # of that doubled every wait and bought nothing.
    poll_interval: float = 0.02

    # Seconds to leave between the two frames compared for stability.
    #
    # This is the one place an interval is the point rather than an overhead. Two
    # captures taken back to back look identical whatever the card is doing, so
    # the measured separation between settled art (1.000) and animation (0.207 to
    # 0.932) only holds if the frames are genuinely apart in time.
    stability_gap: float = 0.15

    #: Looks to wait for a clicked card to reveal itself and stop moving.
    settle_polls: int = 40

    # Consecutive stable comparisons before a card is fingerprinted, when being
    # careful. Relaxed to one while the miss budget has room to spare.
    stability_passes: int = 2

    # Looks showing both cards still up, taken as a match.
    #
    # Deliberately short, and deliberately not called proof: a mismatched pair
    # also stays up for over a second, so this cannot distinguish the two. It is a
    # cheap sanity check on an expected outcome, and the case it misses is caught
    # at the start of the next turn and repaired.
    confirm_polls: int = 2

    #: Looks to wait for a mismatched pair to turn face down again. Live
    #: observation put the display-then-flip-back sequence at roughly 1.6s.
    flip_back_polls: int = 40

    #: Consecutive identical face-down sets that count as "the board is quiet".
    quiet_polls: int = 2

    #: Upper bound on the wait for quiet, after which sampling is tried anyway.
    quiet_budget: int = 25

    #: Looks to wait for a playable board. Doubles as the end-of-run signal, so it
    #: has to be clearly longer than a level transition.
    board_polls: int = 60

    #: How much a box must look like this level's card back to count as one.
    #:
    #: Colour segmentation cannot separate a violet back from a violet card face.
    #: A lavender monster card, revealed on a live level 8 board, passed every
    #: colour and shape filter: 64x57, aspect 1.12, fill 0.76, hue 139/144/145
    #: against real backs at 63x60, aspect 1.03, fill 0.93, hue 131/144/151.
    #: Fill is the widest of those margins and it is only 0.76 against 0.89, the
    #: lowest real back on record -- and a violet card that filled its frame
    #: would pass that too.
    #:
    #: Within a level every back is the same picture, so the separation that does
    #: exist is enormous. Measured by tools/calibrate_back_similarity.py on the
    #: frame that stalled:
    #:
    #:     the monster card   -0.152
    #:     24 real backs       0.992 .. 1.000
    #:     a 2-back frame      0.994
    #:
    #: 0.50 sits in the middle of that gap: 0.65 above the impostor and 0.49
    #: below the lowest real back.
    #:
    #: It was 0.90 first, chosen on the argument that rejecting a real back is
    #: cheap because "the next turn re-examines it". That argument is wrong, and
    #: the next live run showed how: a real back rejected here is absent from the
    #: observation, so the slot reads as face up, and the first attempt to click
    #: it stops the run with CLICK_REFUSED. There is no re-examination -- the
    #: structural rule that every click target must appear in the current frame is
    #: the project's best safety argument and it does not bend.
    #:
    #: So both errors are expensive, and the threshold belongs where the measured
    #: gap is widest rather than 0.09 away from the backs it must not reject. A
    #: card face would have to look half like the back to get through 0.50, while
    #: a back would have to stop looking like itself to be refused.
    back_similarity: float = 0.50

    #: How many times one slot may fail to yield a readable face before the scan
    #: stops putting it first. See _scan_order for what happens without it.
    max_read_attempts: int = 2

    #: How many extra captures to spend when a frame shows stray detections.
    #:
    #: A stray blob is a property of one frame, so looking again is the whole
    #: cure and it costs a capture rather than a mismatch. Two retries because
    #: the alternative -- believing the first frame -- used to stop a level that
    #: was still perfectly playable.
    stray_box_retries: int = 2

    # Margin, in misses, at or below which full verification comes back.
    #
    # Two spare misses is enough to absorb one misread and still have something
    # left, which is the point at which cutting corners stops being free.
    careful_margin: int = 2

    #: How many times one level may rebuild its knowledge from the board before
    #: the run gives up. A repair that keeps being needed is a fault, not a hiccup.
    max_recoveries: int = 2

    # Turn over every card before collecting anything, instead of taking pairs as
    # they appear. One more miss in the worst case; here to be timed rather than
    # argued about.
    scan_first: bool = False

    #: Upper bound on levels in one run, or None for "play until the boards stop".
    max_levels: Optional[int] = None


DEFAULT_RUNNER_PARAMS = RunnerParams()


# Stop and step codes. Stable ASCII, asserted on by tests; the human-readable
# Chinese lives in the accompanying message.
CODE_LEVEL_CLEARED = "LEVEL_CLEARED"
CODE_RUN_COMPLETE = "RUN_COMPLETE"
CODE_TURN_MATCHED = "TURN_MATCHED"
CODE_TURN_MISSED = "TURN_MISSED"
CODE_NO_WINDOW = "NO_WINDOW"
CODE_NO_BOARD = "NO_BOARD"
CODE_GEOMETRY_DRIFT = "GEOMETRY_DRIFT"
CODE_CLICK_FAILED = "CLICK_FAILED"
CODE_CLICK_REFUSED = "CLICK_REFUSED"
CODE_FLIP_BACK_TIMEOUT = "FLIP_BACK_TIMEOUT"
CODE_NOTHING_TO_FLIP = "NOTHING_TO_FLIP"
CODE_BOARD_READY = "BOARD_READY"
CODE_RESYNCED = "RESYNCED"
CODE_UNRECOVERABLE = "UNRECOVERABLE"
CODE_OUT_OF_BUDGET = "OUT_OF_BUDGET"
CODE_REGROUPED = "REGROUPED"

#: Why the record and the board disagree. Kept separate from the message so the
#: response can branch on the cause instead of parsing prose.
TROUBLE_STALE_SLOTS = "STALE_SLOTS"
TROUBLE_GROUP_OVERFLOW = "GROUP_OVERFLOW"
TROUBLE_SOLVER_STUCK = "SOLVER_STUCK"


@dataclass(frozen=True)
class StepResult:
    ok: bool
    code: str
    message: str


@dataclass
class LevelSession:
    """Everything scoped to one level, discarded when it ends."""

    slot_map: SlotMap
    registry: FaceRegistry
    solver: MemorySolver
    #: Cards turned over but not readable. A rising count is the early warning
    #: that sampling is struggling.
    unread_slots: int = 0
    #: How many times knowledge here has been rebuilt from the board.
    recoveries: int = 0
    #: How many times two face groups have been rejoined into one.
    regroupings: int = 0
    #: Failed read attempts per slot, for slots whose face will not settle.
    read_failures: Dict[int, int] = field(default_factory=dict)
    #: What this level's card back looks like, learned from the opening frame.
    back_reference: Optional[np.ndarray] = None
    #: True when the last turn ended with both its cards watched back to face
    #: down, which makes a further wait for stillness pointless.
    board_is_quiet: bool = False
    started_at: float = 0.0

    @property
    def pair_count(self) -> int:
        return self.slot_map.slot_count // 2


class MemoryRunner:
    """Plays the memory minigame until the boards stop appearing."""

    def __init__(
        self,
        window,
        action,
        vision: Optional[CardBackVision] = None,
        params: RunnerParams = DEFAULT_RUNNER_PARAMS,
        fingerprint_params: FingerprintParams = DEFAULT_FINGERPRINT_PARAMS,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.monotonic,
        log: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._window = window
        self._action = action
        self._vision = vision or CardBackVision()
        self._params = params
        self._fp = fingerprint_params
        self._sleep = sleep
        self._now = now
        self._log = log or (lambda message: logger.info(message))

        self.session: Optional[LevelSession] = None
        self.levels_cleared = 0
        self.turns_played = 0
        #: Seconds each cleared level took, so the two flip orders can be compared.
        self.level_seconds: List[float] = []

    # ------------------------------------------------------------------ capture

    def _capture(self) -> Optional[np.ndarray]:
        frame = self._window.capture()
        if frame is None:
            return None
        # Keep click jitter proportional to the window, as the coin loop does.
        height, width = frame.shape[:2]
        self._action.set_scale(
            scale_factor((width, height), tuple(DEFAULT_CONFIG.vision.reference_size))
        )
        return frame

    def _capture_and_detect(self) -> Tuple[Optional[np.ndarray], List[PixelRect]]:
        frame = self._capture()
        if frame is None:
            return None, []
        return frame, [d.bbox for d in self._vision.detect(CARD_BACK, frame)]

    def _learn_back(
        self, frame: np.ndarray, boxes: Sequence[PixelRect]
    ) -> Optional[np.ndarray]:
        """Crop this level's card back from the opening frame.

        Safe to take from any box here and only here: the board gate has just
        confirmed a full face-down board, so every box is a back. The picture
        changes between levels, so it is learned per level and never stored --
        the same arrangement as the face registry.

        The most representative crop is chosen rather than the first, so a single
        box caught mid-animation cannot become the yardstick every later frame is
        measured against.
        """
        crops = []
        for box in boxes:
            crop = crop_face(frame, box, self._fp)
            if crop is not None:
                crops.append(crop)
        if not crops:
            return None
        if len(crops) == 1:
            return crops[0]

        best_crop = None
        best_median = None
        for crop in crops:
            scores = [
                similarity(crop, other, self._fp) for other in crops if other is not crop
            ]
            scores = [s for s in scores if s > UNCOMPARABLE]
            if not scores:
                continue
            median = float(np.median(scores))
            if best_median is None or median > best_median:
                best_median = median
                best_crop = crop
        return best_crop if best_crop is not None else crops[0]

    def _only_real_backs(
        self, session: LevelSession, frame: np.ndarray, boxes: Sequence[PixelRect]
    ) -> List[PixelRect]:
        """Drop boxes that do not look like this level's back.

        The colour gate answers "is this violet, square and solid", which a violet
        card face also is. This answers "is this *the back*", which only the back
        is. Without it a revealed lavender monster kept its slot marked face down,
        the settle wait therefore never completed for that slot or its partner,
        and the scan re-probed the same two cards until the level's whole
        allowance was gone.

        With no reference learned there is nothing to compare against, so every
        box is kept: refusing them all would be far worse than the fault this
        guards against.
        """
        reference = session.back_reference
        if reference is None:
            return list(boxes)

        kept: List[PixelRect] = []
        for box in boxes:
            crop = crop_face(frame, box, self._fp)
            if crop is None:
                # Unreadable as a picture, but it passed the colour gate; keep it
                # rather than invent a verdict from a crop that does not exist.
                kept.append(box)
                continue
            score = similarity(crop, reference, self._fp)
            if score <= UNCOMPARABLE or score >= self._params.back_similarity:
                kept.append(box)
            else:
                logger.debug(
                    "框 %s 与本关卡背相关度 %.3f，低于 %.2f，不算背面",
                    box,
                    score,
                    self._params.back_similarity,
                )
        return kept

    def _look(
        self, session: LevelSession
    ) -> Tuple[Optional[np.ndarray], Optional[BoardObservation]]:
        """A capture, turned into a slot-level view. (None, None) on failure.

        Stray detections are retried rather than believed. One box matching no
        slot while every other box matches its own is a detector false positive,
        not a moved layout, and it belongs to the frame it appeared in -- so the
        answer is another frame, which costs a capture and no mismatches. Real
        drift survives the retries and is reported as before.
        """
        observation: Optional[BoardObservation] = None
        for attempt in range(self._params.stray_box_retries + 1):
            frame, boxes = self._capture_and_detect()
            if frame is None:
                return None, None
            observation = session.slot_map.observe(
                self._only_real_backs(session, frame, boxes)
            )
            if observation.code != CODE_STRAY_BOXES:
                return frame, observation
            if attempt < self._params.stray_box_retries:
                self._sleep(self._params.poll_interval)
        return frame, observation

    # ------------------------------------------------------- starting a level

    def seek_board(self) -> StepResult:
        """Wait for a full, face-down board and start a level on it."""
        for _ in range(self._params.board_polls):
            frame, boxes = self._capture_and_detect()
            if frame is None:
                self._sleep(self._params.poll_interval)
                continue

            verdict = board_screen_verdict(boxes)
            if verdict.ok:
                slot_map = SlotMap(boxes)
                self.session = LevelSession(
                    slot_map=slot_map,
                    registry=FaceRegistry(self._fp),
                    solver=MemorySolver(slot_map.slot_count),
                    started_at=self._now(),
                    back_reference=self._learn_back(frame, boxes),
                )
                order = "先全部翻开" if self._params.scan_first else "边翻边收"
                self._log(
                    f"开始第 {self.levels_cleared + 1} 关："
                    f"{slot_map.initial_shape}，"
                    f"卡牌 {slot_map.card_size[0]}x{slot_map.card_size[1]}，"
                    f"策略 {order}"
                )
                return StepResult(True, CODE_BOARD_READY, verdict.reason)

            self._sleep(self._params.poll_interval)

        return StepResult(False, CODE_NO_BOARD, "等待盘面超时，未看到完整的一盘卡牌")

    # ------------------------------------------------------------- one turn

    def play_turn(self) -> StepResult:
        session = self.session
        if session is None:
            return StepResult(False, CODE_NO_BOARD, "尚未建立盘面")

        if session.solver.is_solved():
            # Every pair has been matched, so the level is over and the game has
            # moved on. Looking now would measure the *next* board against this
            # level's slot map, which reads as a geometry divergence and used to
            # stop the run. The record answers this without a capture.
            return self._finish_level(session)

        frame, observation = self._look(session)
        if frame is None or observation is None:
            return StepResult(False, CODE_NO_WINDOW, "截图失败，窗口可能已关闭或最小化")
        if not observation.ok:
            return StepResult(False, CODE_GEOMETRY_DRIFT, observation.reason)

        if not observation.face_down:
            return self._finish_level(session)

        trouble = self._diagnose(session, observation.face_down)
        if trouble is not None:
            cause, reason = trouble
            if cause == TROUBLE_GROUP_OVERFLOW:
                rejoined = self._rejoin_split_group(session, reason)
                if rejoined is not None:
                    return rejoined
            return self._recover(session, observation.face_down, reason)

        if self._scanning(session):
            return self._scan_turn(session)

        pair = session.solver.known_pair()
        if pair is not None:
            return self._collect_known_pair(session, pair)
        return self._probe(session)

    def _finish_level(self, session: LevelSession) -> StepResult:
        self.levels_cleared += 1
        elapsed = self._now() - session.started_at
        self.level_seconds.append(elapsed)
        notes = []
        if session.unread_slots:
            notes.append(f"{session.unread_slots} 张没读出图案")
        if session.recoveries:
            notes.append(f"重建记忆 {session.recoveries} 次")
        suffix = ("（" + "，".join(notes) + "）") if notes else ""
        self._log(
            f"第 {self.levels_cleared} 关完成{suffix}，耗时 {elapsed:.1f} 秒。"
            f"{session.solver.summary()}"
        )
        self.session = None
        return StepResult(True, CODE_LEVEL_CLEARED, "盘面已全部翻开")

    def _scanning(self, session: LevelSession) -> bool:
        """True while the scan-first order still has cards left to turn over."""
        return bool(self._params.scan_first and session.solver.unknown_slots())

    # ------------------------------------------------------ repairing the record

    def _diagnose(
        self, session: LevelSession, face_down: FrozenSet[int]
    ) -> Optional[Tuple[str, str]]:
        """Why the record and the board disagree, as ``(cause, message)``.

        The cause is returned separately because the three disagreements do not
        deserve the same answer: too many face groups is repairable without
        touching the board, while the other two are not.
        """
        active = set(session.solver.active_slots())
        stale = [slot for slot in sorted(face_down) if slot not in active]
        if stale:
            return (
                TROUBLE_STALE_SLOTS,
                f"槽位 {stale} 本应已配对完成，却又是背面",
            )
        if session.registry.group_count > session.pair_count:
            return (
                TROUBLE_GROUP_OVERFLOW,
                f"卡面分组 {session.registry.group_count} 组，"
                f"多于本关 {session.pair_count} 对",
            )
        if session.solver.is_stuck():
            return (TROUBLE_SOLVER_STUCK, "所有牌面已认识但无法配对")
        return None

    def _rejoin_split_group(
        self, session: LevelSession, reason: str
    ) -> Optional[StepResult]:
        """Rejoin the two most similar face groups. None when not possible.

        The board fixes the number of pictures for free: a level of n pairs shows
        exactly n of them, so n+1 groups means the similarity threshold split one
        picture in two. The two halves are, by construction, the most similar
        groups present.

        Why this beats rebuilding from the board, which is what used to happen
        here: a rebuild discards every learned face to fix one wrong grouping. On
        the run that motivated this, 11 of the 12 groups were correct, the rebuild
        threw all of them away, and re-reading what it already knew consumed the
        level's remaining allowance. Rejoining flips nothing and keeps the
        knowledge that was never in question.

        Rejoining is still a guess, so it is only ever a proposal. The solver
        offers the rejoined slots as a pair, the game answers, and a refusal is
        recorded by ``forbid_pair`` so the same guess cannot be made twice. Same
        division of labour as everywhere else here: fingerprints suggest,
        observation decides.
        """
        closest = session.registry.closest_groups()
        if closest is None:
            return None

        keep, drop, score = closest
        session.registry.merge(keep, drop)
        moved = session.solver.rename_face(drop, keep)
        session.regroupings += 1
        self._log(
            f"{reason}；合并最相似的两组（组 {drop} → 组 {keep}，相似度 {score:.3f}），"
            f"{moved} 张牌改记，未翻动任何牌"
        )
        return StepResult(True, CODE_REGROUPED, reason)

    def _recover(
        self, session: LevelSession, face_down: FrozenSet[int], reason: str
    ) -> StepResult:
        """Rebuild this level's knowledge from what the board currently shows.

        The board is the authority: at the start of a turn nothing should be
        mid-flip, so every card that is not face down has been matched.

        A rebuild is not free, and an earlier version of this docstring claimed it
        was -- it said the cost was "re-reading the cards still in play and
        nothing else". Re-reading costs *mismatches*, and mismatches are the one
        resource the game rations. That sentence is why the check below did not
        exist, and its absence lost a level:

            22 cards / 11 pairs, budget ceil(1.5 * 11) = 17.
            A full scan had already spent 11, leaving 6.
            Rebuilding meant re-reading all 22, worst case 11 more.
            It rebuilt anyway, spent the remaining 6, and the level failed.

        So a rebuild is only a repair while the budget can pay for it. Past that
        point it cannot win the level, and refusing is arithmetic rather than
        caution: stopping leaves the board playable and spends nothing, while
        rebuilding guarantees the loss it was meant to prevent.
        """
        # Worst case after a rebuild: one mismatch per two cards to re-read, the
        # same bound _careful uses for the cards it has not read yet.
        budget = math.ceil(1.5 * session.pair_count)
        spent = session.solver.mismatches
        worst_case_rescan = (len(face_down) + 1) // 2
        if spent + worst_case_rescan > budget:
            return StepResult(
                False,
                CODE_OUT_OF_BUDGET,
                f"{reason}；重建要重认 {len(face_down)} 张（最坏 {worst_case_rescan} 次失配），"
                f"而机会数只剩 {budget - spent} 次，重建必定输掉本关，停手",
            )

        if session.recoveries >= self._params.max_recoveries:
            return StepResult(
                False,
                CODE_UNRECOVERABLE,
                f"{reason}；已重建记忆 {session.recoveries} 次仍不一致，停止",
            )

        session.recoveries += 1
        done = [
            slot
            for slot in range(session.slot_map.slot_count)
            if slot not in face_down
        ]
        session.solver = MemorySolver(session.slot_map.slot_count)
        if done:
            session.solver.mark_cleared(done)
        session.registry = FaceRegistry(self._fp)
        session.board_is_quiet = False
        self._log(
            f"{reason}，以盘面为准重建记忆："
            f"{len(done)} 张已完成，{len(face_down)} 张重新认牌"
        )
        return StepResult(True, CODE_RESYNCED, reason)

    # --------------------------------------------------- collecting a known pair

    def _collect_known_pair(
        self, session: LevelSession, pair: Tuple[int, int]
    ) -> StepResult:
        """Click a pair we already know matches. Nothing is read or waited for."""
        a, b = pair
        for slot in (a, b):
            clicked = self._click_slot(session, slot)
            if not clicked.ok:
                return clicked
            key = session.solver.known_face(slot)
            if key is None:  # pragma: no cover - known_pair guarantees a key
                session.solver.record_flip_unknown(slot)
            else:
                session.solver.record_flip(slot, key)

        matched = self._confirm_stayed_up(session, a, b)
        return self._close_turn(session, a, b, matched, collected=True)

    # ------------------------------------------------------- the scanning phase

    def _scan_order(self, session: LevelSession) -> List[int]:
        """Unread slots, with the ones that keep refusing to be read moved last.

        ``record_flip_unknown`` leaves a slot unread on purpose, and the comment
        on it says the cost is "one more probe later". There was no later: the
        scan always took the two lowest unread slots, so a card whose crop cannot
        be read is immediately the lowest again. Live, on a 26 card level:

            failed 12 & 13, failed 12 & 13, failed 12 & 13, ...

        one mismatch per round against a budget of ceil(1.5 * 13) = 20, and the
        two cards never became readable. It is deterministic rather than unlucky,
        because whatever makes a crop unreadable -- too little texture to
        correlate, a picture that never settles -- is a property of that card, not
        of that moment.

        Deferring is enough to turn a stall into a cost. The other cards still get
        read, and a deferred slot is retried once the rest are done, by which time
        elimination may have answered it for free: when it is one of the last two
        in play, its partner is the other one and no reading is needed.

        Attempts are counted per slot rather than per pair because the failure
        belongs to the card. Two attempts before deferring, so a genuinely
        transient miss -- sparks from a neighbouring celebration -- still gets its
        cheap retry.
        """
        unread = session.solver.unknown_slots()
        limit = self._params.max_read_attempts
        ready = [s for s in unread if session.read_failures.get(s, 0) < limit]
        deferred = [s for s in unread if session.read_failures.get(s, 0) >= limit]
        # Fewest failures first, so the least hopeless card is tried soonest.
        deferred.sort(key=lambda s: session.read_failures.get(s, 0))
        return ready + deferred

    def _scan_turn(self, session: LevelSession) -> StepResult:
        """Turn over the next two unread cards and read both from the same frames.

        Nothing is being collected in this phase, so the two cards are independent
        reads rather than a first choice and a consequence of it. Both clicks go
        out before either card is examined, and one sequence of captures then
        decides that both have stopped moving -- one settling wait for two cards
        instead of one each.

        Clicking back to back with nothing in between is what
        :meth:`_collect_known_pair` has always done, over ten live levels, so it
        asks nothing new of the game. Waiting for the first card to appear is not
        needed either: :meth:`_await_settled_faces` restarts its comparison for as
        long as any of its slots is still showing a back, so a card that has not
        finished turning simply delays the first useful frame.

        A click the game ignores is tolerated exactly as elsewhere: the card is
        recorded as flipped with no face, keeps its place in the unread list and
        comes round again. If the record and the board really have diverged, the
        next turn's diagnosis rebuilds from the board.
        """
        careful = self._careful(session)
        if careful and not session.board_is_quiet:
            # Same protection the other order gets: a celebration still playing
            # would keep the crops moving and be blamed on the cards being read.
            self._wait_for_quiet(session)

        unread = self._scan_order(session)
        if len(unread) < 2:
            # One unread card left, or none. There is nothing in this phase to
            # pair it with, so the ordinary turn machinery finishes the level.
            return self._probe(session)

        first, second = unread[0], unread[1]
        for slot in (first, second):
            clicked = self._click_slot(session, slot)
            if not clicked.ok:
                return clicked

        passes = self._params.stability_passes if careful else 1
        crops = self._await_settled_faces(session, (first, second), passes)

        keys: Dict[int, FaceKey] = {}
        for slot in (first, second):
            crop = crops.get(slot)
            match = session.registry.identify(crop) if crop is not None else None
            if match is None:
                session.unread_slots += 1
                session.read_failures[slot] = session.read_failures.get(slot, 0) + 1
                session.solver.record_flip_unknown(slot)
                logger.debug("槽位 %d 扫描期间没读出稳定卡面，留到后面重读", slot)
                continue
            session.solver.record_flip(slot, match.key)
            keys[slot] = match.key

        hint = len(keys) == 2 and keys[first] == keys[second]
        matched = self._decide_outcome(session, first, second, hint)
        return self._close_turn(session, first, second, matched, collected=False)

    # ------------------------------------------------------------- probing

    def _probe(self, session: LevelSession) -> StepResult:
        """Flip cards we have not read yet, learning what we can from them."""
        careful = self._careful(session)
        if careful and not session.board_is_quiet:
            # Sampling needs a still board; a celebration elsewhere would keep the
            # crop moving and be blamed on the card being read. A turn that ended
            # in a watched flip back has already established stillness.
            self._wait_for_quiet(session)

        first = self._pick_first(session)
        if first is None:
            return StepResult(False, CODE_NOTHING_TO_FLIP, "求解器没有可翻的牌")

        first_key, failure = self._flip_and_read(session, first, careful)
        if failure is not None:
            return failure

        second = self._pick_second(session, first)
        if second is None:
            return StepResult(
                False, CODE_NOTHING_TO_FLIP, "翻开第一张后没有可翻的第二张"
            )

        # If the second card is one we have already read, there is nothing to
        # learn from it and the outcome is already expected.
        known_second = session.solver.known_face(second)
        if known_second is not None:
            clicked = self._click_slot(session, second)
            if not clicked.ok:
                return clicked
            session.solver.record_flip(second, known_second)
            matched = self._confirm_stayed_up(session, first, second)
            return self._close_turn(session, first, second, matched, collected=True)

        second_key, failure = self._flip_and_read(session, second, careful)
        if failure is not None:
            return failure

        hint = first_key is not None and first_key == second_key
        matched = self._decide_outcome(session, first, second, hint)
        return self._close_turn(session, first, second, matched, collected=False)

    def _pick_first(self, session: LevelSession) -> Optional[int]:
        """The first card of a probe.

        Scanning takes the next card it has not read. The solver would offer the
        first half of a known pair instead, and taking that would mean turning over
        a card already read -- exactly the opportunism this order exists to decline,
        and a wasted turn rather than a cheaper one.
        """
        if self._scanning(session):
            unread = session.solver.unknown_slots()
            if unread:
                return unread[0]
        return session.solver.next_flip()

    def _pick_second(self, session: LevelSession, first: int) -> Optional[int]:
        """The second card of a probe.

        Scanning declines the solver's offer to complete a pair it just walked
        into, taking the next unread card instead; that is the whole difference
        between the two orders. With nothing left unread it falls back to the
        solver, which will pair off what is on the board.
        """
        if self._scanning(session):
            for slot in session.solver.unknown_slots():
                if slot != first:
                    return slot
        second = session.solver.next_flip()
        if second is None or second == first:
            return None
        return second

    def _close_turn(
        self,
        session: LevelSession,
        a: int,
        b: int,
        matched: bool,
        collected: bool,
    ) -> StepResult:
        """Record the outcome and leave the board ready for the next turn."""
        session.solver.record_pair_result(a, b, matched)
        self.turns_played += 1

        if matched:
            # The celebration is an animation, so the board is no longer still.
            session.board_is_quiet = False
            self._log(("收牌：" if collected else "配对成功：") + f"槽位 {a} 与 {b}")
            return StepResult(True, CODE_TURN_MATCHED, f"{a} 与 {b} 匹配")

        if collected:
            # Two different pictures had been given the same key, so anything else
            # that comparison decided is suspect. The next turn's diagnosis will
            # rebuild from the board; here it is enough to wait the pair out.
            self._log(f"槽位 {a} 与 {b} 本以为同图案却不匹配，下一轮将以盘面为准重建记忆")
            if not self._wait_for_flip_back(session, (a, b)):
                return StepResult(
                    False, CODE_FLIP_BACK_TIMEOUT, f"槽位 {a}、{b} 迟迟没有翻回背面"
                )
            session.solver.forget_faces()
        else:
            self._log(f"失配：槽位 {a} 与 {b}")
        return StepResult(True, CODE_TURN_MISSED, f"{a} 与 {b} 不匹配")

    def _flip_and_read(
        self, session: LevelSession, slot: int, careful: bool
    ) -> Tuple[Optional[FaceKey], Optional[StepResult]]:
        """Click a card and try to read it. Returns (key, failure_or_None).

        Reading is best effort. A card that cannot be read is recorded as flipped
        with an unknown face, because the outcome comes from watching the game and
        inventing a key would corrupt every later comparison.
        """
        clicked = self._click_slot(session, slot)
        if not clicked.ok:
            return None, clicked

        passes = self._params.stability_passes if careful else 1
        crop = self._await_settled_face(session, slot, passes)
        match = session.registry.identify(crop) if crop is not None else None
        if match is None:
            session.unread_slots += 1
            session.solver.record_flip_unknown(slot)
            logger.debug("槽位 %d 翻开了但没读出稳定的卡面，本轮按未知处理", slot)
            return None, None

        session.solver.record_flip(slot, match.key)
        logger.debug(
            "槽位 %d 归入卡面组 %d（得分 %.3f，%s）",
            slot,
            match.key,
            match.best_score,
            "新组" if match.is_new else "已知组",
        )
        return match.key, None

    # ------------------------------------------------- how careful to be

    def _careful(self, session: LevelSession) -> bool:
        """Whether the miss budget is tight enough to pay for full verification.

        The budget is computed, not read off the screen: ceil(1.5 * pairs), which
        held on every level measured. Against it stands the worst case still
        ahead, one miss per two cards not yet read.
        """
        budget = math.ceil(1.5 * session.pair_count)
        unread = len(session.solver.unknown_slots())
        worst_case_left = (unread + 1) // 2
        margin = budget - session.solver.mismatches - worst_case_left
        return margin <= self._params.careful_margin

    # --------------------------------------------------- deciding the outcome

    def _decide_outcome(
        self, session: LevelSession, a: int, b: int, hint: bool
    ) -> bool:
        """Whether the game matched the pair. Observation decides; the hint hurries."""
        if hint:
            return self._confirm_stayed_up(session, a, b)
        return not self._wait_for_flip_back(session, (a, b))

    def _confirm_stayed_up(self, session: LevelSession, a: int, b: int) -> bool:
        """True when neither slot returns to face down over a few looks."""
        wanted = {a, b}
        for _ in range(self._params.confirm_polls):
            self._sleep(self._params.poll_interval)
            _, observation = self._look(session)
            if observation is None or not observation.ok:
                continue
            if wanted & observation.face_down:
                return False
        return True

    def _wait_for_flip_back(
        self, session: LevelSession, slots: Sequence[int]
    ) -> bool:
        """Wait until every slot in ``slots`` is face down again.

        Every slot, not just one: live observation caught a board reporting eleven
        backs and one face-up card mid-flip-back, because the two do not turn over
        in step.

        Success also establishes that the board is still, which is why the next
        probe can skip asking the same question again.
        """
        wanted = set(slots)
        for _ in range(self._params.flip_back_polls):
            self._sleep(self._params.poll_interval)
            _, observation = self._look(session)
            if observation is None or not observation.ok:
                continue
            if wanted.issubset(observation.face_down):
                session.board_is_quiet = True
                return True
        return False

    def _wait_for_quiet(self, session: LevelSession) -> Optional[FrozenSet[int]]:
        """Wait until the set of face-down slots stops changing.

        Uses the stability interval rather than the polling one: this is a question
        about motion having stopped, and two looks taken back to back cannot answer
        it. Not settling is not fatal -- sampling is tried anyway and has its own
        timeout, so a restless board costs a probe rather than the run.
        """
        previous: Optional[FrozenSet[int]] = None
        repeats = 0
        for _ in range(self._params.quiet_budget):
            _, observation = self._look(session)
            if observation is not None and observation.ok:
                current = observation.face_down
                if previous is not None and current == previous:
                    repeats += 1
                    if repeats >= self._params.quiet_polls - 1:
                        session.board_is_quiet = True
                        return current
                else:
                    repeats = 0
                previous = current
            self._sleep(self._params.stability_gap)
        return None

    # ------------------------------------------------------------- primitives

    def _click_slot(self, session: LevelSession, slot: int) -> StepResult:
        """Click one card, refusing any target that is not currently a card.

        The refusal is what makes "the bot cannot click REPLAY" a property of the
        code rather than a promise.
        """
        frame, observation = self._look(session)
        if frame is None or observation is None:
            return StepResult(False, CODE_NO_WINDOW, "点击前截图失败")
        if not observation.ok:
            return StepResult(False, CODE_GEOMETRY_DRIFT, observation.reason)
        if slot not in observation.face_down:
            return StepResult(
                False, CODE_CLICK_REFUSED, f"槽位 {slot} 当前不是背面，拒绝点击"
            )

        target = session.slot_map.center(slot)
        detected = [session.slot_map.box(index) for index in observation.face_down]
        if not any(point_in_rect(target, box) for box in detected):
            return StepResult(
                False,
                CODE_CLICK_REFUSED,
                f"槽位 {slot} 的中心不在任何检出的卡牌内，拒绝点击",
            )

        if not self._action.click(*target):
            return StepResult(False, CODE_CLICK_FAILED, f"槽位 {slot} 点击未送达")
        return StepResult(True, "CLICKED", f"已点击槽位 {slot}")

    def _await_settled_face(
        self, session: LevelSession, slot: int, passes: int
    ) -> Optional[np.ndarray]:
        """Poll until the card is revealed and has stopped changing; return its crop.

        Returns None when the card never settles, which is a "did not see it"
        answer rather than a failure -- the caller records the flip with no face.
        """
        crops = self._await_settled_faces(session, (slot,), passes)
        return crops.get(slot)

    def _await_settled_faces(
        self, session: LevelSession, slots: Sequence[int], passes: int
    ) -> Dict[int, np.ndarray]:
        """Poll until every slot in ``slots`` is revealed and still; return crops.

        Waiting for the cards to turn over polls as fast as capture allows; deciding
        that they have stopped changing waits ``stability_gap`` between the frames
        being compared, because that comparison is meaningless without a real
        interval.

        Comparing frames is what costs the time, and one sequence of frames answers
        "has it stopped moving" for as many cards as are up. So two cards read
        together cost what one used to, which is the scanning phase's advantage.

        An empty mapping means they never settled together. That is a "did not see
        them" answer rather than a failure; the caller records the flips with no
        face and the cards come round again.
        """
        previous: Dict[int, np.ndarray] = {}
        stable_streak = 0

        for _ in range(self._params.settle_polls):
            frame, observation = self._look(session)
            if frame is None or observation is None:
                self._sleep(self._params.poll_interval)
                continue
            if not observation.ok:
                # Drift mid-flip: do not sample slots whose coordinates are no
                # longer trustworthy.
                return {}
            if any(slot in observation.face_down for slot in slots):
                # Still turning over, or the click did not take. Either way there
                # is nothing to compare yet.
                previous = {}
                stable_streak = 0
                self._sleep(self._params.poll_interval)
                continue

            crops: Dict[int, np.ndarray] = {}
            for slot in slots:
                crop = crop_face(frame, session.slot_map.box(slot), self._fp)
                if crop is not None:
                    crops[slot] = crop
            if len(crops) != len(slots):
                self._sleep(self._params.poll_interval)
                continue

            if previous and all(
                looks_stable(previous[slot], crops[slot], self._fp) for slot in slots
            ):
                stable_streak += 1
                if stable_streak >= passes:
                    return crops
            else:
                stable_streak = 0
            previous = crops
            # The next crops are comparison partners, so they have to be taken
            # after a real interval rather than immediately.
            self._sleep(self._params.stability_gap)

        return {}

    # ------------------------------------------------------------------- loop

    def step(self) -> StepResult:
        """One unit of progress: start a level, or play one turn of it."""
        if self.session is not None:
            return self.play_turn()

        limit = self._params.max_levels
        if limit is not None and self.levels_cleared >= limit:
            return StepResult(
                True, CODE_RUN_COMPLETE, f"已完成 {self.levels_cleared} 关，达到设定上限"
            )

        found = self.seek_board()
        if found.ok:
            return found

        if self.levels_cleared > 0:
            return StepResult(True, CODE_RUN_COMPLETE, self._run_summary())
        return found

    def _run_summary(self) -> str:
        if not self.level_seconds:
            return f"盘面不再出现，本轮结束，共完成 {self.levels_cleared} 关"
        total = sum(self.level_seconds)
        average = total / len(self.level_seconds)
        order = "先全部翻开" if self._params.scan_first else "边翻边收"
        return (
            f"盘面不再出现，本轮结束：{self.levels_cleared} 关，"
            f"翻牌 {self.turns_played} 轮，总耗时 {total:.1f} 秒，"
            f"每关平均 {average:.1f} 秒（策略 {order}）"
        )

    def run(self, should_continue: Callable[[], bool] = lambda: True) -> StepResult:
        """Play until the run finishes or something stops it."""
        last = StepResult(True, "IDLE", "尚未开始")
        while should_continue():
            last = self.step()
            if not last.ok:
                self._log(f"停止：{last.code} — {last.message}")
                if last.code == CODE_UNRECOVERABLE:
                    self._log(
                        "反复重建仍不一致，通常意味着卡面识别在这一关上不可靠。"
                        "请把 reports/manual_memory 下的画面和日志一起留存。"
                    )
                elif last.code == CODE_NO_BOARD:
                    self._log(
                        "一关都没打完就找不到盘面，说明启动时并不在小游戏的开局界面。"
                    )
                return last
            if last.code == CODE_RUN_COMPLETE:
                self._log(last.message)
                return last
        return last
