"""Decide whether two revealed cards show the same picture, at runtime.

Why this is a similarity problem and not a classification one
------------------------------------------------------------
The card faces are drawn from a large and open set: monster portraits of many
kinds, plus food, XP stars, diamonds, keys and pumpkins. Every pair within a
level is distinct. Recognising *what* a card depicts would mean maintaining a
catalogue that the next game update invalidates, and it is not even the question
being asked -- the solver only ever needs to know whether two cards match.

So nothing here is pre-built. No face templates ship with the project, and no
fixed vocabulary of pictures is assumed. Faces are cropped from the frame in
front of us and compared against the faces seen earlier in this same level; that
is the whole mechanism. A card the game has never shown before is handled by
being unlike everything already on file.

How identity is assigned
------------------------
:class:`FaceRegistry` keeps one representative crop per distinct picture found so
far and hands out an opaque integer per group. A new crop is compared with every
representative: close enough to one means the same picture and the same key,
close to none means a new key. That is agglomerative clustering with a single
pass and a fixed threshold, which is all the structure this problem has.

``MemorySolver`` never learns what the keys mean -- it compares them for equality
and nothing else -- so the clustering can be re-tuned or replaced without the
game logic noticing.

Four details that come straight from the recordings
--------------------------------------------------
*Scale.* The same artwork is drawn at 122 px on level 1 and 63 px on level 7, a
factor of about 1.9. Correlation between differently sized crops is meaningless,
so every crop is resampled to one canonical size first.

*Borders.* A card is a picture inside a wobbly hand-drawn frame with a drop
shadow, and the frame differs between a freshly flipped card and a settled one.
Including it would compare frames as much as pictures, so a fixed inset is
dropped from every edge.

*Animation.* A pair that has just matched is lit with a yellow-green flash bright
enough to wash the sprite towards white -- in the recordings a card looks white
mid-flash and eggplant-purple once settled. Comparing during the flash produces
larger differences between one card's two states than between two different
cards. :func:`glow_score` catches the flash, but calibration showed that is not
the whole animation: two frames out of forty-eight measured a true pair at 0.57
and 0.34 with no glow present at all, which is a card caught part-way through its
flip. :func:`looks_stable` exists for that, and the rule it enforces is that a
card is only fingerprinted once it has stopped changing between captures.

*Flat patches.* Normalised correlation divides by each patch's standard
deviation, so between two patches of near-uniform colour it is undefined and
whatever comes back is not a measurement. Card art always has structure, but a
card washed out by the match flash can approach uniform, and the failure mode is
the dangerous direction: an undefined score can read as a perfect match, and a
false match makes the solver click a pair that cannot match while recording a
wrong face. :func:`similarity` therefore refuses to answer for such patches.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.geometry import PixelRect

logger = logging.getLogger(__name__)

#: Opaque group id. Matches ``memory_game.FaceKey`` (anything hashable).
FaceKey = int

#: Returned when two patches cannot be meaningfully compared. Below every real
#: threshold, so a caller that ignores the distinction still errs towards
#: "different", which is the safe direction.
UNCOMPARABLE = -1.0


@dataclass(frozen=True)
class FingerprintParams:
    """Knobs for cropping and comparing card faces.

    Every default here was measured against ``captures/memory`` by
    ``tools/calibrate_face_similarity.py`` and ``tools/probe_glow_signature.py``,
    never chosen from synthetic images or from a visual impression.
    """

    #: Fraction of width and height dropped from each edge before comparing, to
    #: exclude the card frame and its drop shadow. 0.18 keeps the picture and
    #: little else at every card size the game uses.
    inset_ratio: float = 0.18

    #: Every crop is resampled to this square before comparison. Large enough to
    #: keep a monster's features on the smallest 63 px card, small enough that a
    #: full pairwise comparison over a 32-card board stays trivial.
    canonical_size: int = 48

    # Correlation at or above this counts as the same picture.
    #
    # Measured over 263 true-pair and 3196 non-pair comparisons:
    #
    #   true pairs   median 0.983   p5 0.922   p1 0.691   min 0.337
    #   non-pairs    median 0.062   p95 0.299  p99 0.458  max 0.462
    #
    # 0.75 sits 0.29 above the highest non-pair score ever measured and 0.17
    # below the fifth percentile of true pairs, so neither side is close.
    #
    # The asymmetry of the two failure modes decided which end of that range to
    # favour. Judging two different cards to be the same makes the solver click a
    # pair that cannot match: a mismatch is spent *and* its memory is now wrong,
    # so the error compounds. Judging one card to be two costs an extra probe and
    # nothing else. The threshold therefore sits well clear of the non-pair
    # distribution even though that leaves the animation outliers below it --
    # falling below only mints a spare group, which the caller detects and retries.
    match_threshold: float = 0.75

    # Hue window of the match flash, in OpenCV units (H 0-179).
    #
    # The lower bound started at 20 and was wrong. It was the one number in this
    # project picked from a description ("yellow-green") rather than measured, and
    # the first live run showed the cost: an ordinary revealed card was reported
    # as glowing for 4.3 seconds, and stopped only once the pair had resolved.
    # Since is_settled gates fingerprinting, such a card can never be sampled and
    # the runner times out on it -- and gold coins, XP stars and diamonds are all
    # card faces, so this was the common case.
    #
    # Measured with tools/probe_glow_signature.py, using self-similarity across
    # consecutive captures to separate settled cards from animating ones. Any
    # score on a settled card is a false positive by construction:
    #
    #   window        settled max   animating p99 / max   settled cards refused
    #   H [20,45]         0.245         0.801 / 0.901        5 / 1054  (0.5%)
    #   H [30,45]         0.037         0.709 / 0.779        0 / 1054  (0.0%)
    #
    # The false positives clustered at hue 22-26 with 44% of the card bright and
    # saturated -- gold and amber artwork, not the flash. Moving the lower bound
    # excludes them at the source, which is why the window moved rather than the
    # cutoff being loosened.
    glow_hue_low: int = 30
    glow_hue_high: int = 45

    #: The flash is both bright and saturated; either alone also describes normal
    #: artwork, so both are required.
    glow_saturation_low: int = 90
    glow_value_low: int = 200

    # Share of a card's pixels in that window above which it is treated as still
    # animating and unsafe to compare.
    #
    # Kept at 0.15, which is four times the highest score any settled card
    # reaches under the corrected window (0.037) while still far below the flash
    # (p99 0.709). Raising it further would only cost flash sensitivity: the
    # failure directions are not symmetric. A false positive stalls the level,
    # while a missed flash is still caught by looks_stable, so this gate is tuned
    # never to fire on settled art rather than to catch every flash.
    glow_fraction: float = 0.15

    # Correlation between the same slot in two consecutive captures, above which
    # the card is considered to have stopped moving. Settled cards are static
    # sprites and correlate essentially perfectly with themselves; a flip in
    # progress does not. Set below 1.0 only to absorb capture noise.
    stability_threshold: float = 0.98

    # Standard deviation, in grey levels, below which a patch carries too little
    # structure for correlation to mean anything. Real card art measures far
    # above this; the value only has to exclude the near-uniform case.
    min_patch_std: float = 3.0


DEFAULT_FINGERPRINT_PARAMS = FingerprintParams()


def _inset_box(box: PixelRect, ratio: float) -> PixelRect:
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    dx = int(round(width * ratio))
    dy = int(round(height * ratio))
    # Never inset a degenerate box out of existence.
    if width - 2 * dx < 4 or height - 2 * dy < 4:
        return box
    return x1 + dx, y1 + dy, x2 - dx, y2 - dy


def patch_std(patch: np.ndarray) -> float:
    """Standard deviation of a patch in grey levels, as a structure measure."""
    if patch is None or patch.size == 0:
        return 0.0
    if patch.ndim == 3:
        grey = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    else:
        grey = patch
    return float(grey.std())


def has_structure(
    patch: np.ndarray, params: FingerprintParams = DEFAULT_FINGERPRINT_PARAMS
) -> bool:
    """True when a patch varies enough for correlation to be meaningful."""
    return patch_std(patch) >= params.min_patch_std


def crop_face(
    frame: np.ndarray,
    box: PixelRect,
    params: FingerprintParams = DEFAULT_FINGERPRINT_PARAMS,
) -> Optional[np.ndarray]:
    """Return a comparable crop of the picture on the card at ``box``.

    Colour is kept. Two monsters can share a silhouette and differ only in hue,
    so discarding it would throw away the cheapest distinguishing signal.

    Returns None when the box lies outside the frame or is too small to inset,
    which is a "cannot judge" answer rather than a guess.
    """
    if frame is None or frame.size == 0:
        return None

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = _inset_box(box, params.inset_ratio)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None

    patch = frame[y1:y2, x1:x2]
    side = params.canonical_size
    # INTER_AREA downsamples without the ringing that would add texture the card
    # does not have; cards are almost always being shrunk here.
    return cv2.resize(patch, (side, side), interpolation=cv2.INTER_AREA)


def similarity(
    a: np.ndarray,
    b: np.ndarray,
    params: FingerprintParams = DEFAULT_FINGERPRINT_PARAMS,
) -> float:
    """Normalised correlation between two equally sized crops, in [-1, 1].

    ``TM_CCOEFF_NORMED`` subtracts each patch's mean before correlating, which
    makes it tolerant of the overall brightness differences between a freshly
    flipped card and a settled one while still keeping them apart from a
    genuinely different picture.

    Returns :data:`UNCOMPARABLE` rather than a number when either patch is
    missing, differently shaped, or too uniform to correlate; see the module
    docstring for why the uniform case has to be refused rather than trusted.
    """
    if a is None or b is None:
        return UNCOMPARABLE
    if a.shape != b.shape:
        return UNCOMPARABLE
    if not has_structure(a, params) or not has_structure(b, params):
        return UNCOMPARABLE
    result = cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)
    value = float(result[0][0])
    # matchTemplate can still emit a non-finite value on pathological input.
    if not np.isfinite(value):
        return UNCOMPARABLE
    return value


def glow_score(
    frame: np.ndarray,
    box: PixelRect,
    params: FingerprintParams = DEFAULT_FINGERPRINT_PARAMS,
) -> float:
    """Fraction of the card that looks like the match flash."""
    if frame is None or frame.size == 0:
        return 0.0

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return 0.0

    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array(
            [params.glow_hue_low, params.glow_saturation_low, params.glow_value_low],
            np.uint8,
        ),
        np.array([params.glow_hue_high, 255, 255], np.uint8),
    )
    return float(mask.mean()) / 255.0


def looks_stable(
    previous_crop: Optional[np.ndarray],
    current_crop: Optional[np.ndarray],
    params: FingerprintParams = DEFAULT_FINGERPRINT_PARAMS,
) -> bool:
    """True when the same slot looks the same in two consecutive captures.

    This is the guard the glow test cannot provide. Calibration found true pairs
    scoring 0.57 and 0.34 on frames with no glow at all, because a card part-way
    through its flip shows neither its back nor its face. Such a card correlates
    poorly with itself one capture later, which is exactly what is measured here.

    A missing previous crop returns False: not yet known to be stable, so wait
    one more capture rather than assume.
    """
    if previous_crop is None or current_crop is None:
        return False
    return similarity(previous_crop, current_crop, params) >= params.stability_threshold


def is_settled(
    frame: np.ndarray,
    box: PixelRect,
    params: FingerprintParams = DEFAULT_FINGERPRINT_PARAMS,
) -> bool:
    """True when the card is not mid-flash.

    Necessary but not sufficient on its own; pair it with :func:`looks_stable`
    before fingerprinting.
    """
    return glow_score(frame, box, params) < params.glow_fraction


@dataclass
class FaceMatch:
    """Outcome of asking the registry to identify a crop.

    ``best_score`` is carried even when a new key is minted, because the margin
    between it and the threshold is the only evidence available at runtime that
    the threshold is holding up. A stream of near-misses in the log is the signal
    that calibration has drifted.
    """

    key: FaceKey
    is_new: bool
    best_score: float
    rival_key: Optional[FaceKey] = None


class FaceRegistry:
    """Groups card faces seen in one level, by similarity, as they arrive.

    One registry per level. Faces learned in a previous level are meaningless in
    the next -- the pictures change -- so there is deliberately no persistence
    and nothing is written to disk.
    """

    def __init__(self, params: FingerprintParams = DEFAULT_FINGERPRINT_PARAMS) -> None:
        self.params = params
        self._representatives: Dict[FaceKey, np.ndarray] = {}
        self._next_key: FaceKey = 0

    @property
    def group_count(self) -> int:
        return len(self._representatives)

    def keys(self) -> List[FaceKey]:
        return sorted(self._representatives)

    def identify(self, crop: np.ndarray) -> Optional[FaceMatch]:
        """Assign ``crop`` to an existing group or start a new one.

        Returns None when the crop cannot be judged -- missing, empty, or too
        uniform to correlate -- so a caller can retry on the next frame rather
        than record a face it did not really see. Minting a group for an
        unreadable crop would be worse than waiting: the spurious group has no
        partner anywhere on the board, so the solver would probe for it until the
        level stalled.
        """
        if crop is None or crop.size == 0:
            return None
        if not has_structure(crop, self.params):
            logger.debug("卡面裁片纹理不足，拒绝取样（标准差 %.2f）", patch_std(crop))
            return None

        best_key: Optional[FaceKey] = None
        best_score = UNCOMPARABLE
        for key, representative in self._representatives.items():
            score = similarity(crop, representative, self.params)
            if score > best_score:
                best_score = score
                best_key = key

        if best_key is not None and best_score >= self.params.match_threshold:
            return FaceMatch(key=best_key, is_new=False, best_score=best_score)

        key = self._next_key
        self._next_key += 1
        # The first crop of a group becomes its representative. Averaging members
        # was considered and rejected: with exactly two cards per picture there is
        # nothing to average away, and a blurred mean would only weaken the next
        # comparison.
        self._representatives[key] = crop.copy()
        logger.debug(
            "新卡面组 %d（最接近的已知组得分 %.3f，阈值 %.3f）",
            key,
            best_score,
            self.params.match_threshold,
        )
        return FaceMatch(
            key=key, is_new=True, best_score=best_score, rival_key=best_key
        )

    def group_overflow(self, pair_count: int) -> bool:
        """True when more groups exist than the board can possibly hold.

        A level of ``pair_count`` pairs has exactly that many distinct pictures.
        More groups than that means the threshold split one picture in two --
        usually a card fingerprinted while still animating. Worth acting on
        rather than logging, because the solver will never find a partner for the
        spurious group and would probe until the level stalls.
        """
        return self.group_count > pair_count

    def closest_groups(self) -> Optional[Tuple[FaceKey, FaceKey, float]]:
        """The two most similar groups, as ``(keep, drop, score)``.

        Returns None when there is nothing to compare or no pair of groups can be
        scored at all. ``keep`` is the lower key purely so the outcome is
        deterministic and easy to read in a log.
        """
        keys = self.keys()
        if len(keys) < 2:
            return None

        best: Optional[Tuple[FaceKey, FaceKey, float]] = None
        for i, left in enumerate(keys):
            for right in keys[i + 1:]:
                score = similarity(
                    self._representatives[left],
                    self._representatives[right],
                    self.params,
                )
                if score <= UNCOMPARABLE:
                    continue
                if best is None or score > best[2]:
                    best = (left, right, score)
        return best

    def merge(self, keep: FaceKey, drop: FaceKey) -> None:
        """Fold ``drop`` into ``keep``, discarding the dropped representative.

        The kept representative is not averaged with the dropped one, for the same
        reason a group's first crop becomes its representative in the first place:
        with two cards per picture there is nothing to average away, and a blurred
        mean only weakens the next comparison.

        The caller is responsible for repointing anything that recorded ``drop``;
        this class knows nothing about slots.
        """
        if keep == drop:
            raise ValueError("不能把一个卡面组合并到它自己")
        if keep not in self._representatives or drop not in self._representatives:
            raise KeyError(f"卡面组不存在：keep={keep} drop={drop}")
        del self._representatives[drop]


def pairwise_scores(
    crops: List[np.ndarray],
    params: FingerprintParams = DEFAULT_FINGERPRINT_PARAMS,
) -> np.ndarray:
    """Full similarity matrix, diagonal set to :data:`UNCOMPARABLE`.

    Used by the calibration tool: on a board where every visible card is part of
    a matched pair, a correct threshold is exactly one that makes mutual best
    matches form a perfect pairing. That check needs no manual labelling, which
    is why the matrix is exposed here rather than hidden inside the registry.
    """
    count = len(crops)
    scores = np.full((count, count), UNCOMPARABLE, dtype=np.float64)
    for i in range(count):
        for j in range(i + 1, count):
            value = similarity(crops[i], crops[j], params)
            scores[i][j] = value
            scores[j][i] = value
    return scores


def mutual_best_pairs(scores: np.ndarray) -> Tuple[List[Tuple[int, int]], List[int]]:
    """Pair up indices that are each other's best match.

    Returns the pairs and whatever was left over. A perfect pairing with nothing
    left over is the evidence that a threshold candidate is sound.
    """
    count = scores.shape[0]
    if count == 0:
        return [], []

    best = [int(np.argmax(scores[i])) for i in range(count)]
    pairs: List[Tuple[int, int]] = []
    taken = set()
    for i in range(count):
        j = best[i]
        if i in taken or j in taken or j == i:
            continue
        if best[j] == i:
            pairs.append((i, j))
            taken.add(i)
            taken.add(j)
    leftovers = [i for i in range(count) if i not in taken]
    return pairs, leftovers
