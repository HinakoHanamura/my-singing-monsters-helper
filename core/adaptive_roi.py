"""Adaptive dynamic Region-of-Interest (ROI) tracker with global scan fallback.

Eliminates all fixed/hardcoded coordinates when matching templates or features.
Operates in two modes:
1. Global scan (default initially, or whenever ROI prediction fails).
2. Dynamic localized ROI (once target is detected, tracks it within an expanding
   adaptive bounding box for high-speed sub-millisecond verification).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import cv2
import numpy as np


@dataclass
class MatchResult:
    """Detection result containing center point, bounding rectangle, and score."""

    center: Tuple[int, int]
    rect: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in full-frame coordinates
    score: float


class DynamicROI:
    """Tracks a visual target's location and provides an adaptive search region.

    Starts with no ROI (executes full-screen global search by default).
    Once the target is found, memorizes its center and size with an adaptive margin.
    If a subsequent search within the dynamic ROI fails, automatically
    invalidates the ROI so the caller falls back to an exhaustive global scan.
    """

    def __init__(
        self,
        name: str,
        margin_x: int = 60,
        margin_y: int = 60,
        min_dim: int = 40,
    ) -> None:
        self.name = name
        self.margin_x = margin_x
        self.margin_y = margin_y
        self.min_dim = min_dim
        self._center: Optional[Tuple[int, int]] = None
        self._size: Tuple[int, int] = (0, 0)
        self._consecutive_hits: int = 0

    @property
    def has_memorized_roi(self) -> bool:
        """Return True if an adaptive ROI is currently memorized."""
        return self._center is not None

    @property
    def center(self) -> Optional[Tuple[int, int]]:
        return self._center

    def get_crop_bounds(
        self, frame_shape: Tuple[int, ...]
    ) -> Optional[Tuple[int, int, int, int]]:
        """Return (y1, y2, x1, x2) for current memorized ROI, or None if no ROI memorized."""
        if self._center is None:
            return None
        h, w = frame_shape[:2]
        cx, cy = self._center
        tw, th = self._size
        hw = max(tw // 2, self.min_dim // 2) + self.margin_x
        hh = max(th // 2, self.min_dim // 2) + self.margin_y
        x1 = max(0, cx - hw)
        x2 = min(w, cx + hw)
        y1 = max(0, cy - hh)
        y2 = min(h, cy + hh)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None
        return (y1, y2, x1, x2)

    def update(self, center: Tuple[int, int], size: Tuple[int, int]) -> None:
        """Update memorized target location and size."""
        self._center = center
        self._size = size
        self._consecutive_hits += 1

    def invalidate(self) -> None:
        """Clear memorized target location to trigger full global scan next time."""
        self._center = None
        self._consecutive_hits = 0

    def match(
        self,
        frame: np.ndarray,
        template: Optional[np.ndarray],
        threshold: float = 0.65,
    ) -> Optional[MatchResult]:
        """Perform template matching using dynamic ROI with automatic global scan fallback.

        Step 1: If a dynamic ROI is memorized, attempt fast match in that local crop.
        Step 2: If local match succeeds (score >= threshold), update center and return.
        Step 3: If local match fails (or no ROI memorized), invalidate ROI and execute
                an exhaustive global scan over the full frame.
        Step 4: If global match succeeds, memorize new ROI and return.
        Step 5: Return None (target not present on screen).
        """
        if frame is None or template is None or frame.size == 0 or template.size == 0:
            return None

        th, tw = template.shape[:2]
        fh, fw = frame.shape[:2]
        if th > fh or tw > fw:
            return None

        # 1. Fast path: test memorized dynamic ROI if available
        bounds = self.get_crop_bounds(frame.shape)
        if bounds is not None:
            y1, y2, x1, x2 = bounds
            roi = frame[y1:y2, x1:x2]
            if roi.shape[0] >= th and roi.shape[1] >= tw:
                res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
                _, max_v, _, loc = cv2.minMaxLoc(res)
                if max_v >= threshold:
                    full_x = x1 + loc[0]
                    full_y = y1 + loc[1]
                    center = (full_x + tw // 2, full_y + th // 2)
                    rect = (full_x, full_y, full_x + tw, full_y + th)
                    self.update(center, (tw, th))
                    return MatchResult(center=center, rect=rect, score=float(max_v))

            # If dynamic ROI failed: target moved, shifted, or vanished -> invalidate ROI
            self.invalidate()

        # 2. Fallback / Initial path: Full global scan across entire frame
        res_global = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        _, max_v_g, _, loc_g = cv2.minMaxLoc(res_global)
        if max_v_g >= threshold:
            full_x = loc_g[0]
            full_y = loc_g[1]
            center = (full_x + tw // 2, full_y + th // 2)
            rect = (full_x, full_y, full_x + tw, full_y + th)
            self.update(center, (tw, th))
            return MatchResult(center=center, rect=rect, score=float(max_v_g))

        return None

    def match_any(
        self,
        frame: np.ndarray,
        templates: Sequence[Tuple[str, Optional[np.ndarray]]],
        threshold: float = 0.65,
    ) -> Optional[Tuple[str, MatchResult]]:
        """Match against multiple template candidates using dynamic ROI and global fallback."""
        valid_templates: List[Tuple[str, np.ndarray]] = [
            (name, tmpl) for name, tmpl in templates if tmpl is not None and tmpl.size > 0
        ]
        if not valid_templates:
            return None

        # 1. Fast path: check all templates in memorized dynamic ROI
        bounds = self.get_crop_bounds(frame.shape)
        if bounds is not None:
            y1, y2, x1, x2 = bounds
            roi = frame[y1:y2, x1:x2]
            best_local_match: Optional[Tuple[str, MatchResult]] = None
            best_local_score = -1.0

            for name, tmpl in valid_templates:
                th, tw = tmpl.shape[:2]
                if roi.shape[0] >= th and roi.shape[1] >= tw:
                    res = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
                    _, max_v, _, loc = cv2.minMaxLoc(res)
                    if max_v >= threshold and max_v > best_local_score:
                        full_x = x1 + loc[0]
                        full_y = y1 + loc[1]
                        center = (full_x + tw // 2, full_y + th // 2)
                        rect = (full_x, full_y, full_x + tw, full_y + th)
                        best_local_match = (
                            name,
                            MatchResult(center=center, rect=rect, score=float(max_v)),
                        )
                        best_local_score = max_v

            if best_local_match is not None:
                _, match_res = best_local_match
                w = match_res.rect[2] - match_res.rect[0]
                h = match_res.rect[3] - match_res.rect[1]
                self.update(match_res.center, (w, h))
                return best_local_match

            # Local ROI failed
            self.invalidate()

        # 2. Fallback / Initial path: Full global scan across entire frame
        best_global_match: Optional[Tuple[str, MatchResult]] = None
        best_global_score = -1.0

        for name, tmpl in valid_templates:
            th, tw = tmpl.shape[:2]
            if frame.shape[0] >= th and frame.shape[1] >= tw:
                res = cv2.matchTemplate(frame, tmpl, cv2.TM_CCOEFF_NORMED)
                _, max_v, _, loc = cv2.minMaxLoc(res)
                if max_v >= threshold and max_v > best_global_score:
                    full_x = loc[0]
                    full_y = loc[1]
                    center = (full_x + tw // 2, full_y + th // 2)
                    rect = (full_x, full_y, full_x + tw, full_y + th)
                    best_global_match = (
                        name,
                        MatchResult(center=center, rect=rect, score=float(max_v)),
                    )
                    best_global_score = max_v

        if best_global_match is not None:
            _, match_res = best_global_match
            w = match_res.rect[2] - match_res.rect[0]
            h = match_res.rect[3] - match_res.rect[1]
            self.update(match_res.center, (w, h))
            return best_global_match

        return None
