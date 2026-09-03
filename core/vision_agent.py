"""Perception layer: locate targets in a single frame.

Designed to be swappable. The contract is deliberately narrow so a learned
detector can replace the template matcher without touching anything else:

  - no win32, no PySide6 imports; this is pure image maths,
  - one abstract method, `BaseVisionAgent.detect`,
  - the engine depends on the abstract base, never on a concrete class.

Window-size adaptation
----------------------
`cv2.matchTemplate` has no scale invariance: a template cropped at 1024x768 will
not match once the same sprite is drawn larger or smaller.

The compensation here scales the **template**, not the frame:
  1. `geometry.scale_factor` gives the current window's ratio to the reference
     resolution the templates were cropped at,
  2. the template is resized by that ratio and matched against the native frame,
  3. a few extra steps around that ratio absorb rounding and mild distortion.

Scaling the template rather than the frame is cheaper (a template is tiny) and,
more importantly, keeps matching in native frame coordinates, so results need no
inverse transform - one fewer place to get wrong.

Caveat worth stating plainly: if the game's own camera zoom is free to change,
window size alone does not determine sprite size, and `scale_steps` must span
the zoom range or detection will miss. Wide multi-scale search is the fallback;
a learned detector handles it natively.

Replacing this with a model later:

    class YoloVisionAgent(BaseVisionAgent):
        def __init__(self, weights, conf=0.4):
            self.model = YOLO(weights)

        def detect(self, target_name, screenshot):
            results = self.model(screenshot, verbose=False)
            return [Detection(...) for box in results[0].boxes if ...]

The engine and UI need no changes at all.
"""

from __future__ import annotations

import glob
import logging
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.geometry import (
    PixelRect,
    Point,
    aspect_mismatch,
    denorm_rect,
    scale_factor,
    scale_length,
)

logger = logging.getLogger(__name__)

# OpenCV is only needed by the template implementation. Keep it a soft
# dependency so the framework still starts without it installed.
try:
    import cv2

    _CV2_AVAILABLE = True
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
    _CV2_AVAILABLE = False


#: Hard ceiling on peaks taken from one match pass. Bounds the work regardless
#: of how permissive the threshold is.
_MAX_PEAKS_PER_PASS = 300


@dataclass(frozen=True)
class Detection:
    """One structured hit.

    Carrying a box and a score rather than a bare point is what makes the
    anti-misclick rules possible: without a box there is no "what is underneath
    this", and without a score there is no way to rank candidates.
    Coordinates are client-area pixels of the current window and can be handed
    straight to the action layer.
    """

    name: str
    x: int          # box top-left
    y: int
    width: int
    height: int
    confidence: float

    @property
    def center(self) -> Point:
        return self.x + self.width // 2, self.y + self.height // 2

    @property
    def bbox(self) -> PixelRect:
        return self.x, self.y, self.x + self.width, self.y + self.height


class BaseVisionAgent(ABC):
    """The contract every detector implements (template, learned model, OCR)."""

    @abstractmethod
    def detect(self, target_name: str, screenshot: np.ndarray) -> List[Detection]:
        """Find every instance of a target in one frame.

        Args:
            target_name: logical name, e.g. "coin", "monster", "cloud".
            screenshot: BGR array shaped (h, w, 3), origin at the client area's
                top-left corner.

        Returns:
            Detections sorted by descending confidence; empty when nothing hits.
        """
        raise NotImplementedError

    # ---- convenience wrappers, implemented once for every subclass ----

    def find_target(self, target_name: str, screenshot: np.ndarray) -> Optional[Point]:
        """Locate the single best target and return its centre, or None.

        Kept for simple call sites (clicking one known button). The main loop
        uses detect(), because the anti-misclick rules need bounding boxes.
        """
        detections = self.detect(target_name, screenshot)
        return detections[0].center if detections else None

    def find_all_targets(self, target_name: str, screenshot: np.ndarray) -> List[Point]:
        """Centres of every detected target."""
        return [d.center for d in self.detect(target_name, screenshot)]


class VisionAgent(BaseVisionAgent):
    """OpenCV template matching with window-size adaptation.

    Template naming: `{template_dir}/{target_name}*.png`. So coin.png,
    coin_phase2.png and coin_phase3.png are all treated as variants of one
    logical target - useful for an animated sprite, where a couple of cropped
    animation phases noticeably reduce misses.

    With `use_fake_detection=True` no templates are read and synthetic
    coordinates are returned instead, so the whole pipeline can be exercised
    before any assets exist.
    """

    def __init__(
        self,
        template_dir: str = "assets/templates",
        match_threshold: float = 0.82,
        reference_size: Tuple[int, int] = (1024, 768),
        scale_steps: Sequence[float] = (0.97, 1.0, 1.03),
        nms_distance: int = 24,
        max_detections: int = 20,
        use_fake_detection: bool = True,
        simulate_miss_rate: float = 0.25,
        match_downscale: float = 1.0,
        search_region: Optional[Tuple[float, float, float, float]] = None,
        target_thresholds: Optional[Dict[str, float]] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.template_dir = template_dir
        self.match_threshold = match_threshold
        self.reference_size = reference_size
        self.scale_steps = tuple(scale_steps) or (1.0,)
        self.nms_distance = nms_distance
        self.max_detections = max_detections
        self.use_fake_detection = use_fake_detection
        self.simulate_miss_rate = simulate_miss_rate
        self.target_thresholds = dict(target_thresholds) if target_thresholds else {}

        # Cost controls. Template matching is O(pixels), so both of these are
        # direct multipliers on detection time.
        #
        # match_downscale shrinks frame and template together before matching and
        # maps results back afterwards. Localisation gets coarser by roughly
        # 1/factor pixels, which is only acceptable because the click tolerance
        # and the icon are both far larger than that error - the benchmark tool
        # measures the actual drift rather than assuming it.
        self.match_downscale = float(match_downscale) if match_downscale else 1.0
        # search_region is a normalized rect limiting where matching happens.
        # The UI bands are never clickable anyway, so searching them is waste.
        self.search_region = search_region

        self._rng = rng or random.Random()

        # Raw templates: {target_name: [ndarray, ...]}
        self._templates: Dict[str, List[np.ndarray]] = {}
        # Resized templates: {(target_name, variant index, quantised scale)}
        self._scaled_cache: Dict[Tuple[str, int, int], np.ndarray] = {}
        # Window sizes already warned about, so the log is not spammed per frame.
        self._warned_aspect: set = set()

    def threshold_for(self, target_name: str) -> float:
        """Confidence floor for a specific target, falling back to match_threshold."""
        if self.target_thresholds and target_name in self.target_thresholds:
            return self.target_thresholds[target_name]
        return self.match_threshold

    # ------------------------------------------------------------ public API

    def detect(self, target_name: str, screenshot: np.ndarray) -> List[Detection]:
        if screenshot is None or screenshot.size == 0:
            logger.warning("empty frame received, skipping detection")
            return []

        height, width = screenshot.shape[:2]

        if self.use_fake_detection:
            return self._fake_detect(target_name, width, height)

        return self._template_detect(target_name, screenshot, width, height)

    # -------------------------------------------------------- template match

    def _template_detect(
        self, target_name: str, screenshot: np.ndarray, width: int, height: int
    ) -> List[Detection]:
        if not _CV2_AVAILABLE:
            logger.error("opencv-python is not installed; template matching unavailable")
            return []

        templates = self._load_templates(target_name)
        if not templates:
            return []

        base_scale = scale_factor((width, height), self.reference_size)
        self._warn_if_aspect_off((width, height))

        # Restrict the search area, then optionally shrink it. Both reduce the
        # pixel count matching has to walk; coordinates are mapped back below so
        # callers always receive full-frame client coordinates.
        search, offset = self._search_window(screenshot, width, height)
        if search is None:
            return []

        shrink = self._effective_downscale(search)
        if shrink != 1.0:
            search = cv2.resize(
                search, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA
            )

        raw: List[Detection] = []
        for index, template in enumerate(templates):
            for step in self.scale_steps:
                scaled = self._get_scaled_template(
                    target_name, index, template, base_scale * step * shrink
                )
                if scaled is None:
                    continue
                raw.extend(self._match_one(target_name, search, scaled))

        if not raw:
            return []

        if shrink != 1.0 or offset != (0, 0):
            raw = [self._to_frame_coords(d, shrink, offset) for d in raw]

        merged = self._suppress_duplicates(raw, base_scale)
        return merged[: self.max_detections]

    def _search_window(
        self, screenshot: np.ndarray, width: int, height: int
    ) -> Tuple[Optional[np.ndarray], Tuple[int, int]]:
        """Crop to the configured search region. Returns (view, (dx, dy))."""
        if not self.search_region:
            return screenshot, (0, 0)

        x1, y1, x2, y2 = denorm_rect(self.search_region, width, height)
        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))
        if x2 - x1 < 8 or y2 - y1 < 8:
            logger.warning("search_region collapses to nothing; ignoring it")
            return screenshot, (0, 0)
        return screenshot[y1:y2, x1:x2], (x1, y1)

    def _effective_downscale(self, view: np.ndarray) -> float:
        """Clamp the requested downscale so the view stays usable."""
        shrink = self.match_downscale
        if shrink >= 1.0 or shrink <= 0.0:
            return 1.0
        height, width = view.shape[:2]
        if min(width * shrink, height * shrink) < 64:
            # Too small to hold a scaled template with any structure left.
            return 1.0
        return shrink

    @staticmethod
    def _to_frame_coords(
        detection: Detection, shrink: float, offset: Tuple[int, int]
    ) -> Detection:
        """Map a detection from the search view back to full-frame coordinates."""
        factor = 1.0 / shrink if shrink else 1.0
        dx, dy = offset
        return Detection(
            name=detection.name,
            x=int(round(detection.x * factor)) + dx,
            y=int(round(detection.y * factor)) + dy,
            width=max(1, int(round(detection.width * factor))),
            height=max(1, int(round(detection.height * factor))),
            confidence=detection.confidence,
        )

    def _match_one(
        self, target_name: str, screenshot: np.ndarray, template: np.ndarray
    ) -> List[Detection]:
        """Match one template at one scale and return its local score peaks.

        Only local maxima are returned, never every pixel above the threshold.
        That distinction is not cosmetic: a single real target produces a broad
        plateau of above-threshold pixels, so on a busy frame a permissive
        threshold yields tens of thousands of raw hits. Feeding those into the
        pairwise duplicate suppression below turns it into an O(n^2) Python loop
        with n in the tens of thousands, which effectively hangs the caller.

        Peak extraction is done with a dilation, which is vectorised in OpenCV
        and costs a single pass over the score map.
        """
        th, tw = template.shape[:2]
        sh, sw = screenshot.shape[:2]
        if th > sh or tw > sw:
            return []

        # TM_CCOEFF_NORMED tolerates overall brightness shifts, which suits
        # game icons drawn with glow and lighting effects.
        result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)

        # A peak must dominate a neighbourhood roughly half a template wide:
        # two genuinely distinct targets are not packed closer than that.
        window = max(3, (min(th, tw) // 2) | 1)
        neighbourhood_max = cv2.dilate(result, np.ones((window, window), np.uint8))

        threshold = self.threshold_for(target_name)
        peaks = (result >= threshold) & (result >= neighbourhood_max - 1e-6)
        ys, xs = np.where(peaks)
        if len(xs) == 0:
            return []

        scores = result[ys, xs]

        # Even after peak extraction, a pathological frame could yield many
        # points. Cap by score so downstream work is always bounded.
        if len(scores) > _MAX_PEAKS_PER_PASS:
            keep = np.argpartition(-scores, _MAX_PEAKS_PER_PASS)[:_MAX_PEAKS_PER_PASS]
            xs, ys, scores = xs[keep], ys[keep], scores[keep]

        return [
            Detection(
                name=target_name,
                x=int(x),
                y=int(y),
                width=tw,
                height=th,
                confidence=float(score),
            )
            for x, y, score in zip(xs, ys, scores)
        ]

    def _suppress_duplicates(
        self, detections: List[Detection], base_scale: float
    ) -> List[Detection]:
        """Greedy de-duplication across variants and scales.

        One target still hits at several scales and from several template
        variants. Keep the highest score and drop anything whose centre is too
        close. The gap is expressed in reference-resolution pixels and scaled to
        the live window, so behaviour is identical at any resolution.

        Input size is bounded by `_match_one`, so the pairwise loop here is
        cheap.
        """
        min_gap = scale_length(self.nms_distance, base_scale)
        min_gap_sq = min_gap * min_gap

        ordered = sorted(detections, key=lambda d: d.confidence, reverse=True)
        kept: List[Detection] = []
        for candidate in ordered:
            cx, cy = candidate.center
            crowded = False
            for existing in kept:
                ex, ey = existing.center
                if (cx - ex) ** 2 + (cy - ey) ** 2 < min_gap_sq:
                    crowded = True
                    break
            if not crowded:
                kept.append(candidate)
        return kept

    # ------------------------------------------------------- template loading

    def _load_templates(self, target_name: str) -> List[np.ndarray]:
        """Load and cache every `{target_name}*.png` variant."""
        if target_name in self._templates:
            return self._templates[target_name]

        pattern = os.path.join(self.template_dir, "%s*.png" % target_name)
        paths = sorted(glob.glob(pattern))
        if not paths:
            logger.error(
                "no template found for %r (looked for %s)", target_name, pattern
            )
            self._templates[target_name] = []
            return []

        loaded: List[np.ndarray] = []
        for path in paths:
            # imdecode instead of imread: imread goes through a narrow-char API
            # and fails on paths containing non-ASCII characters.
            try:
                buffer = np.fromfile(path, dtype=np.uint8)
                image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            except Exception:
                logger.exception("failed to read template %s", path)
                continue
            if image is None:
                logger.error("template could not be decoded: %s", path)
                continue
            loaded.append(image)
            logger.info(
                "loaded template %s (%dx%d)", path, image.shape[1], image.shape[0]
            )

        self._templates[target_name] = loaded
        return loaded

    def _get_scaled_template(
        self, target_name: str, index: int, template: np.ndarray, scale: float
    ) -> Optional[np.ndarray]:
        """Resize a template, caching by scale quantised to one percent."""
        quantised = int(round(scale * 100))
        if quantised <= 0:
            return None

        key = (target_name, index, quantised)
        cached = self._scaled_cache.get(key)
        if cached is not None:
            return cached

        actual = quantised / 100.0
        if abs(actual - 1.0) < 1e-6:
            self._scaled_cache[key] = template
            return template

        th, tw = template.shape[:2]
        new_w = max(1, int(round(tw * actual)))
        new_h = max(1, int(round(th * actual)))
        if new_w < 4 or new_h < 4:
            # Too small to carry recognisable structure; skip this step.
            return None

        interpolation = cv2.INTER_AREA if actual < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(template, (new_w, new_h), interpolation=interpolation)
        self._scaled_cache[key] = resized
        return resized

    def _warn_if_aspect_off(self, client_size: Tuple[int, int]) -> None:
        """Warn once when the aspect ratio differs enough to hurt match rates."""
        if client_size in self._warned_aspect:
            return
        mismatch = aspect_mismatch(client_size, self.reference_size)
        if mismatch > 0.08:
            logger.warning(
                "window %dx%d differs from reference %dx%d by %.1f%% in aspect "
                "ratio; match rate may drop, consider re-cropping templates at "
                "this size",
                client_size[0],
                client_size[1],
                self.reference_size[0],
                self.reference_size[1],
                mismatch * 100,
            )
        self._warned_aspect.add(client_size)

    # ------------------------------------------------------------- fake data

    def _fake_detect(
        self, target_name: str, width: int, height: int
    ) -> List[Detection]:
        """Synthetic boxes for wiring up the pipeline before assets exist.

        Returns nothing with probability `simulate_miss_rate` so the "target not
        found" branch of the main loop is genuinely exercised. Delete this
        method once a real detector is in place.
        """
        if self._rng.random() < self.simulate_miss_rate:
            logger.debug("[fake] no %s this round", target_name)
            return []

        factor = scale_factor((width, height), self.reference_size)
        box = scale_length(46, factor)

        results: List[Detection] = []
        for _ in range(self._rng.randint(1, 3)):
            # Middle 60% of the frame, away from edges and UI furniture.
            cx = self._rng.randint(int(width * 0.2), int(width * 0.8))
            cy = self._rng.randint(int(height * 0.2), int(height * 0.8))
            results.append(
                Detection(
                    name=target_name,
                    x=cx - box // 2,
                    y=cy - box // 2,
                    width=box,
                    height=box,
                    confidence=self._rng.uniform(0.85, 0.99),
                )
            )

        results.sort(key=lambda d: d.confidence, reverse=True)
        logger.debug("[fake] %d hits for %s", len(results), target_name)
        return results


def check_piggy_status(crop: np.ndarray, min_v: float = 130.0) -> bool:
    """Determine whether the piggy bank is bright (active) or dim (inactive).

    Measured on live captures: bright active piggy has center body HSV Value V ~ 152-168,
    while dim inactive piggy has V ~ 89-96.
    Returns True if bright, False if dim or empty.
    """
    if crop is None or crop.size == 0 or not _CV2_AVAILABLE:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, w = crop.shape[:2]
    # Sample the central 60% of the crop where the pig body sits.
    pig_center = hsv[int(h * 0.2) : int(h * 0.8), int(w * 0.2) : int(w * 0.8), 2]
    if pig_center.size == 0:
        return False
    return float(pig_center.mean()) >= min_v
