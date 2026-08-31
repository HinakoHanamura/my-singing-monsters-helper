"""Solver for the card-matching minigame.

The game shows an even number of identical-looking face-down cards. Every card
has exactly one partner sharing the same face. A turn flips two cards: equal
faces take the pair out of play, unequal faces spend one of a limited number of
misses and both cards turn back over.

Note that "out of play" is not "off the screen". A matched pair keeps showing its
pictures, face up, in the same positions until the whole board fades out at the
end of the level. That makes no difference to this module -- a cleared slot simply
stops being a candidate -- but it matters a great deal to whoever is watching the
board through a camera, so it is stated here rather than left to be discovered.
See :mod:`core.minigames.board` for what that costs the perception side.

Separation of concerns
----------------------
This module contains **no image code at all**. It never learns what a card
depicts, only whether two cards look the same, and it receives that judgement
from the caller as an opaque ``FaceKey``. Anything hashable works: a perceptual
hash, a cluster id, or in tests a plain string.

That boundary is deliberate. The faces are drawn from a large and open set --
monster portraits of many kinds, plus food, XP stars, diamonds, keys, pumpkins
and coins -- with far too many variants to template-match, so recognition has to
be a runtime *similarity* question rather than a classification one. Keeping the
solver ignorant of how that similarity is computed means the whole decision layer
is testable without a running game, and the vision side can be swapped or
re-calibrated without touching a line of game logic.

The solver is also not the judge of a turn
------------------------------------------
It records the outcome it is told. Whether two cards actually matched is something
the game answers unambiguously -- a matched pair never turns back over -- and the
caller is expected to watch for that rather than trust a fingerprint comparison.
See :mod:`core.minigames.memory_runner` for why: the second card of a turn is
sampled exactly while the match celebration is washing its colours out, so the
fingerprint is at its least reliable precisely when the answer matters most.

Slot indices
------------
A slot index is the card's position in reading order (left-to-right,
top-to-bottom) as produced by :mod:`core.minigames.grid`. The solver always
prefers the lowest index when several choices are equally good, which is what
makes the flip order predictable to someone watching.

Strategy
--------
1. A pair whose faces are both already known is free: it clears without risking
   a miss, so take it first.
2. Otherwise flip the lowest-index unseen card. If its face turns out to match
   a card seen earlier, complete that pair immediately.
3. Otherwise flip the next unseen card. This may miss, but it is the only way
   to gain information, and the face learned is kept for later turns.

Knowledge is never discarded on a miss, which is the entire point of the game.
Note that no branch of this strategy gambles: known pairs are free and probes
are forced, so knowing the exact miss allowance would not change any decision.
The miss counter is tracked for reporting only.

That is also why the allowance is never read off the screen. Every miss reveals
exactly two previously unseen cards and nothing learned is thrown away, so the
worst case is one miss per pair; the game grants ceil(1.5 * pairs), measured
across six levels. The budget therefore cannot bind, and the only way a level can
be lost is a perception error.
"""

from __future__ import annotations

import logging
from typing import Dict, FrozenSet, Hashable, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

#: Opaque "these two cards look the same" token. The solver only ever compares
#: keys for equality; it attaches no meaning to their contents.
FaceKey = Hashable


class MemorySolver:
    """Tracks board knowledge and decides which card to flip next.

    The caller drives a turn like this::

        a = solver.next_flip()
        solver.record_flip(a, fingerprint_of(a))
        b = solver.next_flip()
        solver.record_flip(b, fingerprint_of(b))
        solver.record_pair_result(a, b, matched)

    ``next_flip`` is a pure query and may be called as often as needed; only the
    ``record_*`` methods mutate state.
    """

    def __init__(self, slot_count: int) -> None:
        if slot_count < 0:
            raise ValueError(f"slot_count must not be negative, got {slot_count}")
        if slot_count % 2 != 0:
            # A real board always pairs up, so an odd count means perception
            # missed a card or invented one. Not fatal (the solver degrades to
            # leaving one card unmatched) but worth surfacing loudly.
            logger.warning(
                "卡牌数为奇数 (%d)，可能漏检或误检了一张，剩余一张将无法配对",
                slot_count,
            )

        self._slot_count = slot_count
        self._cleared: Set[int] = set()
        self._faces: Dict[int, FaceKey] = {}
        self._face_up: List[int] = []
        # Slot pairs the game itself has rejected. Faces are a guess; a mismatch
        # is a fact, and this is where the facts go so a guess cannot overrule
        # them twice. See forbid_pair.
        self._rejected: Set[FrozenSet[int]] = set()

        self.turns = 0
        self.matches = 0
        self.mismatches = 0

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def slot_count(self) -> int:
        return self._slot_count

    @property
    def face_up(self) -> Tuple[int, ...]:
        """Slots flipped during the current, unresolved turn."""
        return tuple(self._face_up)

    def active_slots(self) -> List[int]:
        """Slots still in play, in reading order.

        "In play" rather than "on the board": a matched pair stays visible on
        screen and only leaves this list.
        """
        return [i for i in range(self._slot_count) if i not in self._cleared]

    def unknown_slots(self) -> List[int]:
        """Slots still in play whose face we have never seen."""
        return [i for i in self.active_slots() if i not in self._faces]

    def known_face(self, index: int) -> Optional[FaceKey]:
        return self._faces.get(index)

    def is_solved(self) -> bool:
        return not self.active_slots()

    def known_pair(self) -> Optional[Tuple[int, int]]:
        """A pair of face-down slots already known to match, or None.

        Picks the pair with the smallest leading index so the flip order stays
        in reading order.
        """
        groups: Dict[FaceKey, List[int]] = {}
        for index in self.active_slots():
            if index in self._face_up:
                continue
            key = self._faces.get(index)
            if key is None:
                continue
            groups.setdefault(key, []).append(index)

        best: Optional[Tuple[int, int]] = None
        for members in groups.values():
            for i, first in enumerate(members):
                partner = next(
                    (
                        second
                        for second in members[i + 1:]
                        if not self.is_rejected(first, second)
                    ),
                    None,
                )
                if partner is None:
                    continue
                candidate = (first, partner)
                if best is None or candidate[0] < best[0]:
                    best = candidate
                break
        return best

    def partner_for(self, index: int) -> Optional[int]:
        """A face-down slot known to carry the same face as ``index``."""
        key = self._faces.get(index)
        if key is None:
            return None
        for other in self.active_slots():
            if other == index or other in self._face_up:
                continue
            if self._faces.get(other) != key:
                continue
            if self.is_rejected(index, other):
                continue
            return other
        return None

    def next_flip(self) -> Optional[int]:
        """The slot to flip next, or None if there is nothing safe to do.

        None means one of three things, distinguished by :meth:`is_solved` and
        :meth:`is_stuck`: the board is finished, the current turn still needs
        :meth:`record_pair_result`, or knowledge has become inconsistent.
        """
        if len(self._face_up) >= 2:
            # Two cards are already up; the turn has to be resolved first.
            return None

        if not self._face_up:
            pair = self.known_pair()
            if pair is not None:
                return pair[0]
            return self._lowest_unknown()

        first = self._face_up[0]
        partner = self.partner_for(first)
        if partner is not None:
            return partner
        return self._lowest_unknown(exclude=frozenset(self._face_up))

    def is_stuck(self) -> bool:
        """True when the board is unfinished yet no move can be chosen.

        On a consistent board this is unreachable: every remaining card has a
        partner. In practice it means fingerprinting split one pair into two
        different keys, so the caller should :meth:`forget_faces` and re-probe
        rather than spin.
        """
        if self.is_solved():
            return False
        if len(self._face_up) >= 2:
            return False
        return self.next_flip() is None

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------

    def learn_face(self, index: int, key: FaceKey) -> None:
        """Record what a card looks like without marking it as flipped.

        Split out from :meth:`record_flip` for the case where the game reveals
        several cards at once, such as an opening preview. This game turned out
        to have no such preview, but the split costs nothing and the mechanism is
        the obvious one to reach for if a later version adds one.
        """
        self._require_active(index)
        if key is None:
            raise ValueError("face key must not be None; use a real fingerprint")
        self._faces[index] = key

    def record_flip(self, index: int, key: FaceKey) -> None:
        """Record that ``index`` was flipped this turn and shows ``key``."""
        self._require_active(index)
        self._require_room_for_flip(index)
        self.learn_face(index, key)
        if index not in self._face_up:
            self._face_up.append(index)

    def record_flip_unknown(self, index: int) -> None:
        """Mark a slot as flipped this turn without learning its face.

        Perception can fail to read a card the game really did turn over. The
        celebration on a neighbouring pair throws sparks across its neighbours,
        for instance, which can keep a crop unstable for as long as the caller is
        willing to wait.

        The turn still has to be finished and its result recorded, so the flip is
        registered and the face left unknown. That costs one more probe later,
        which the miss budget absorbs comfortably. Inventing a face instead would
        be far worse: a wrong key does not stay wrong locally, it poisons every
        comparison for the rest of the level.
        """
        self._require_active(index)
        self._require_room_for_flip(index)
        if index not in self._face_up:
            self._face_up.append(index)

    def record_pair_result(self, a: int, b: int, matched: bool) -> None:
        """Close out a turn: take the pair out of play on a match, else count a miss.

        ``matched`` comes from the caller watching the game, not from comparing
        faces here; see the module docstring. Learned faces survive a miss on
        purpose.
        """
        self._require_active(a)
        self._require_active(b)
        if a == b:
            raise ValueError("一轮里的两张牌不能是同一个槽位")

        self.turns += 1
        if matched:
            self._cleared.update((a, b))
            self.matches += 1
        else:
            self.mismatches += 1
            self.forbid_pair(a, b)
        self._face_up.clear()

    def mark_cleared(self, indices: Sequence[int]) -> None:
        """Force slots out of play, for resynchronising after a lost turn.

        Covers the case where an animation or a popup hid the outcome of a turn
        and the caller had to work out from the board which cards are done.
        Note that a done card is still *visible* -- it shows its picture face up
        -- so "cleared" here means "no longer a candidate", not "gone".
        """
        for index in indices:
            self._require_index(index)
        self._cleared.update(indices)
        self._face_up = [i for i in self._face_up if i not in self._cleared]

    def forget_faces(self) -> None:
        """Drop all recorded faces, keeping cleared slots. Stuck-board recovery."""
        self._faces.clear()
        self._face_up.clear()

    def rename_face(self, old: FaceKey, new: FaceKey) -> int:
        """Repoint every slot recorded as ``old`` at ``new``. Returns how many.

        Used when two face groups turn out to be one picture that the similarity
        threshold split in two. The board fixes the group count for free -- a
        level of n pairs shows exactly n pictures -- so n+1 groups means one
        picture got split, and the two halves can be rejoined without flipping
        anything.
        """
        if old == new:
            return 0
        changed = [index for index, key in self._faces.items() if key == old]
        for index in changed:
            self._faces[index] = new
        return len(changed)

    def forbid_pair(self, a: int, b: int) -> None:
        """Record that the game refused ``a`` and ``b`` as a pair.

        Faces are inferred; a mismatch is observed. Without this record an
        incorrect inference can be acted on repeatedly: if two slots are believed
        to carry the same picture, ``known_pair`` keeps proposing them, the game
        keeps refusing, and the level spends its whole allowance on one wrong
        guess. That is exactly the shape of failure that rejoining split groups
        would otherwise introduce, so the two belong together.

        Cheap in the ordinary case too. When a miss happens between two cards
        whose faces were not both known, their keys end up different anyway and
        this record changes nothing.
        """
        self._require_index(a)
        self._require_index(b)
        if a == b:
            raise ValueError("一轮里的两张牌不能是同一个槽位")
        self._rejected.add(frozenset((a, b)))

    def is_rejected(self, a: int, b: int) -> bool:
        """Whether the game has already refused this pair in this level."""
        return frozenset((a, b)) in self._rejected

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> str:
        return (
            f"共 {self._slot_count} 张 ｜ 已消除 {len(self._cleared)} 张"
            f"（{self.matches} 对）｜ 已认识 {len(self._faces)} 张 ｜ "
            f"翻牌 {self.turns} 轮，失配 {self.mismatches} 次"
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _lowest_unknown(self, exclude: FrozenSet[int] = frozenset()) -> Optional[int]:
        for index in self.unknown_slots():
            if index not in exclude and index not in self._face_up:
                return index
        return None

    def _require_index(self, index: int) -> None:
        if not 0 <= index < self._slot_count:
            raise IndexError(
                f"slot index {index} out of range for {self._slot_count} slots"
            )

    def _require_active(self, index: int) -> None:
        self._require_index(index)
        if index in self._cleared:
            raise ValueError(f"槽位 {index} 已被消除，不能再操作")

    def _require_room_for_flip(self, index: int) -> None:
        if index not in self._face_up and len(self._face_up) >= 2:
            raise ValueError(
                "两张牌已翻开，需先调用 record_pair_result 结束本轮"
            )
