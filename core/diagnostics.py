"""Failure forensics: dump frames when a round produces no clickable target.

Why this exists
---------------
"It does not detect anything" is not actionable. The calibration numbers were
derived from saved frames, so when the live run disagrees with them, the live
frame must differ from the calibration set - different camera zoom, different
resolution, a dialog covering the playfield, a different island. Guessing which
is a waste of time; capturing the frame settles it.

This writer is deliberately dumb and bounded: it saves raw PNGs, caps how many
it will write per session, and never raises into the caller. A diagnostic that
can break the thing it diagnoses is worse than none.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2

    _CV2_AVAILABLE = True
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
    _CV2_AVAILABLE = False


class MissRecorder:
    """Saves frames from rounds that produced nothing clickable."""

    def __init__(
        self,
        directory: str,
        enabled: bool = False,
        max_dumps: int = 12,
        annotate: bool = True,
    ) -> None:
        self.directory = directory
        self.enabled = enabled and _CV2_AVAILABLE
        self.max_dumps = max_dumps
        self.annotate = annotate
        self._written = 0
        self._session_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

        if enabled and not _CV2_AVAILABLE:
            logger.warning("diagnostics requested but opencv-python is unavailable")

    @property
    def written(self) -> int:
        return self._written

    @property
    def exhausted(self) -> bool:
        return self._written >= self.max_dumps

    def record(
        self,
        frame: np.ndarray,
        round_index: int,
        detections: Sequence = (),
        note: str = "",
    ) -> Optional[str]:
        """Write one frame. Returns the path, or None when nothing was written.

        Args:
            frame: the BGR frame the round worked on.
            round_index: loop round number, so dumps line up with the log.
            detections: whatever the vision layer returned, drawn on the frame
                when annotation is on. Zero detections is itself the useful
                signal: it means the template never matched.
            note: short reason tag folded into the filename.
        """
        if not self.enabled or self.exhausted or frame is None:
            return None

        try:
            os.makedirs(self.directory, exist_ok=True)

            canvas = frame
            if self.annotate and detections:
                canvas = frame.copy()
                for detection in detections:
                    x1, y1, x2, y2 = detection.bbox
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 140, 255), 2)
                    cv2.putText(
                        canvas,
                        "%.2f" % detection.confidence,
                        (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 140, 255),
                        1,
                        cv2.LINE_AA,
                    )

            safe_note = "".join(c if c.isalnum() or c in "-_" else "_" for c in note)
            name = "miss_%s_r%04d_d%02d_%s.png" % (
                self._session_tag,
                round_index,
                len(detections),
                safe_note or "nohit",
            )
            path = os.path.join(self.directory, name)

            # imencode + write so non-ASCII paths work.
            ok, buffer = cv2.imencode(".png", canvas)
            if not ok:
                return None
            with open(path, "wb") as handle:
                handle.write(buffer.tobytes())

            self._written += 1
            return path

        except Exception:
            # Never let a diagnostic take the bot down.
            logger.exception("failed to write diagnostic frame")
            return None

    def summary(self) -> str:
        if not self.enabled:
            return "diagnostics off"
        return "diagnostic frames written: %d/%d -> %s" % (
            self._written,
            self.max_dumps,
            self.directory,
        )
