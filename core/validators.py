"""Anti-misclick: composable context validation rules.

Core idea: detecting a target is **not** the same as being allowed to click it.
A detection must survive a chain of rules; the first rule that rejects it wins
and records why. The bias is deliberate: missing a coin costs a little income,
clicking the wrong thing can spend currency, open a menu, or buy an item.

Rules come in two flavours:

* Self-contained rules look only at the detection box and raw pixels
  (confidence, position, texture underneath).
* Neighbour rules need other categories detected in the same frame
  ("there must be a monster below", "skip coins wrapped in a cloud").
  They advertise this through `required_targets`, and the engine detects those
  extra categories on demand.

Each Verdict carries three separate things, which matters more than it looks:

* ``code``   - stable ASCII identifier. Tests and metrics key off this, so
               rewording a message never breaks them.
* ``label``  - short human-readable group name, used for the aggregated log.
* ``detail`` - the concrete numbers behind this particular rejection.

This module imports neither win32 nor PySide6; it is pure logic and can be
tested without a game or a GUI.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from config import SafetyConfig
from core.geometry import (
    PixelRect,
    denorm_rect,
    point_in_rect,
    rects_overlap,
    scale_length,
)
from core.vision_agent import Detection

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Verdict:
    """Outcome of one rule.

    ``code`` is the machine-readable reason and is what tests assert on.
    ``label`` / ``detail`` exist for humans reading the log.
    """

    ok: bool
    code: str = ""
    label: str = ""
    detail: str = ""

    def describe(self) -> str:
        if self.ok:
            return "accepted"
        if self.detail:
            return f"{self.label}({self.detail})"
        return self.label or self.code


ACCEPT = Verdict(True, code="ACCEPTED", label="通过")


@dataclass
class ValidationContext:
    """Per-frame context shared by every rule, so nothing is computed twice."""

    frame: np.ndarray
    client_size: Tuple[int, int]
    # Current window scale relative to the reference resolution. Rules use it to
    # convert reference-resolution pixels from the config into actual pixels.
    scale: float
    # Detections for each category in this frame; neighbour rules read from here.
    detections: Dict[str, List[Detection]] = field(default_factory=dict)


class Rule(ABC):
    """Base class for validation rules."""

    #: Stable machine-readable rejection code.
    code: str = "REJECTED"
    #: Short human-readable group name shown in the aggregated log.
    label: str = "rule"

    @abstractmethod
    def check(self, detection: Detection, ctx: ValidationContext) -> Verdict:
        raise NotImplementedError

    @property
    def required_targets(self) -> Set[str]:
        """Extra categories the engine must detect for this rule. Empty by default."""
        return set()

    def reject(self, detail: str = "") -> Verdict:
        return Verdict(False, code=self.code, label=self.label, detail=detail)


# ------------------------------------------------------------ self-contained


class MinConfidence(Rule):
    """Confidence floor. Drops matches that barely cleared the threshold."""

    code = "LOW_CONFIDENCE"
    label = "置信度不足"

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def check(self, detection: Detection, ctx: ValidationContext) -> Verdict:
        if detection.confidence < self.threshold:
            return self.reject(f"{detection.confidence:.2f}<{self.threshold:.2f}")
        return ACCEPT


class NotNearEdge(Rule):
    """Never click the outer ring: targets there are usually clipped."""

    code = "NEAR_EDGE"
    label = "贴近画面边缘"

    def __init__(self, margin_ratio: float) -> None:
        self.margin_ratio = margin_ratio

    def check(self, detection: Detection, ctx: ValidationContext) -> Verdict:
        width, height = ctx.client_size
        mx = width * self.margin_ratio
        my = height * self.margin_ratio
        cx, cy = detection.center
        if cx < mx or cx > width - mx or cy < my or cy > height - my:
            return self.reject(f"{detection.center}")
        return ACCEPT


class NotInExclusionZones(Rule):
    """Hard no-click regions. The most reliable line of defence.

    Zones are configured as normalized ratios and converted per frame, so they
    stay correct no matter how the window is resized.
    """

    code = "EXCLUSION_ZONE"
    label = "落在禁点区"

    def __init__(self, zones: Sequence[Tuple[float, float, float, float]]) -> None:
        self.zones = tuple(zones)
        # Cache the pixel conversion: one computation per distinct window size.
        self._cache: Dict[Tuple[int, int], Tuple[PixelRect, ...]] = {}

    def _pixel_zones(self, client_size: Tuple[int, int]) -> Tuple[PixelRect, ...]:
        cached = self._cache.get(client_size)
        if cached is None:
            width, height = client_size
            cached = tuple(denorm_rect(z, width, height) for z in self.zones)
            self._cache[client_size] = cached
        return cached

    def check(self, detection: Detection, ctx: ValidationContext) -> Verdict:
        zones = self._pixel_zones(ctx.client_size)
        center = detection.center
        for index, zone in enumerate(zones):
            # Either the centre falls inside, or the box overlaps the zone.
            if point_in_rect(center, zone) or rects_overlap(detection.bbox, zone):
                return self.reject(f"#{index}")
        return ACCEPT


class TextureBelow(Rule):
    """Require visible texture below the target.

    This is a cheap proxy for the domain rule "a collectible coin always sits
    above a monster": empty ground is flat (low standard deviation) while a
    monster has strong colour and edge variation. It avoids needing a template
    for every monster in the game.

    Known limitation, stated plainly: decorated or patterned ground can also
    pass this check, and the threshold must be calibrated against real frames.
    On a busy background this rule can degenerate into passing everything, which
    fails silently. For a strict check, enable NeighborBelow with a monster
    template, or let a trained model learn "monster" as its own class.
    """

    code = "NO_TEXTURE_BELOW"
    label = "下方无怪物特征"

    def __init__(self, probe_height_ratio: float, min_std: float) -> None:
        self.probe_height_ratio = probe_height_ratio
        self.min_std = min_std

    def measure(self, detection: Detection, frame: np.ndarray) -> float:
        """Standard deviation of the probe region below the box, -1 if unusable.

        Exposed separately so calibration tooling can collect the distribution
        of this value over real frames instead of guessing a threshold.
        """
        height, width = frame.shape[:2]
        x1 = max(0, detection.x)
        x2 = min(width, detection.x + detection.width)
        y1 = min(height, detection.y + detection.height)
        y2 = min(height, y1 + int(round(detection.height * self.probe_height_ratio)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return -1.0
        return float(np.std(frame[y1:y2, x1:x2]))

    def check(self, detection: Detection, ctx: ValidationContext) -> Verdict:
        std = self.measure(detection, ctx.frame)
        if std < 0:
            return self.reject("probe out of bounds")
        if std < self.min_std:
            return self.reject(f"std={std:.1f}<{self.min_std:.1f}")
        return ACCEPT


# ------------------------------------------------------------------ neighbour


class NeighborBelow(Rule):
    """Require a detection of another category below the target (e.g. a monster).

    Needs the corresponding template (assets/templates/monster.png). Without it
    the category never gets detected and this rule rejects everything, which is
    why it is disabled by default.
    """

    code = "MISSING_NEIGHBOR"
    label = "下方缺少指定邻居"

    def __init__(self, neighbor_name: str, max_distance_ref: int) -> None:
        self.neighbor_name = neighbor_name
        self.max_distance_ref = max_distance_ref

    @property
    def required_targets(self) -> Set[str]:
        return {self.neighbor_name}

    def check(self, detection: Detection, ctx: ValidationContext) -> Verdict:
        neighbors = ctx.detections.get(self.neighbor_name, [])
        if not neighbors:
            return self.reject(f"{self.neighbor_name} 未检出")

        max_distance = scale_length(self.max_distance_ref, ctx.scale)
        cx, cy = detection.center
        for neighbor in neighbors:
            nx, ny = neighbor.center
            # "Below" means strictly lower on screen and not far off sideways.
            if ny <= cy:
                continue
            if ny - cy > max_distance:
                continue
            if abs(nx - cx) > max_distance:
                continue
            return ACCEPT

        return self.reject(f"{self.neighbor_name} 不在范围内")


class NoOverlapWith(Rule):
    """Reject a target overlapped by another category (e.g. cloud-wrapped coins).

    Needs the corresponding template (assets/templates/cloud.png). Disabled by
    default.
    """

    code = "BLOCKED_BY_OVERLAP"
    label = "被遮挡物覆盖"

    def __init__(self, blocker_name: str) -> None:
        self.blocker_name = blocker_name

    @property
    def required_targets(self) -> Set[str]:
        return {self.blocker_name}

    def check(self, detection: Detection, ctx: ValidationContext) -> Verdict:
        for blocker in ctx.detections.get(self.blocker_name, []):
            if rects_overlap(detection.bbox, blocker.bbox):
                return self.reject(self.blocker_name)
        return ACCEPT


# ---------------------------------------------------------------- rule chain


class RuleChain:
    """Run rules in order, short-circuiting on the first rejection.

    Order matters: cheap rules with a high rejection rate go first (exclusion
    zones, confidence), and rules that require extra detection work go last, so
    the expensive checks run as rarely as possible.
    """

    def __init__(self, rules: Sequence[Rule]) -> None:
        self.rules = tuple(rules)

    def evaluate(self, detection: Detection, ctx: ValidationContext) -> Verdict:
        for rule in self.rules:
            verdict = rule.check(detection, ctx)
            if not verdict.ok:
                return verdict
        return ACCEPT

    @property
    def required_targets(self) -> Set[str]:
        """Every extra category the engine must detect for this chain."""
        needed: Set[str] = set()
        for rule in self.rules:
            needed |= rule.required_targets
        return needed

    def describe(self) -> str:
        return " -> ".join(type(rule).__name__ for rule in self.rules) or "(no rules)"


def build_rule_chain(safety: SafetyConfig, min_confidence: float) -> RuleChain:
    """Assemble the default chain from config.

    Rules whose switch is off are not added at all, so they cost nothing.
    """
    rules: List[Rule] = [
        NotInExclusionZones(safety.exclusion_zones),
        NotNearEdge(safety.edge_margin),
        MinConfidence(min_confidence),
    ]

    if safety.require_texture_below:
        rules.append(TextureBelow(safety.texture_probe_height, safety.texture_min_std))

    if safety.reject_cloud_overlap:
        rules.append(NoOverlapWith(safety.cloud_target_name))

    if safety.require_monster_below:
        rules.append(
            NeighborBelow(safety.monster_target_name, safety.monster_max_distance)
        )

    return RuleChain(rules)
