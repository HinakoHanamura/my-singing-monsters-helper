"""Deterministic synthetic frames for the logic tests.

Scope, stated up front
----------------------
Synthetic images are used here **only** to test code that must be exercised with
fully controlled inputs:

* scale conversion arithmetic,
* non-maximum suppression / de-duplication,
* rule-chain short-circuit behaviour,
* guard state transitions,
* engine orchestration.

For those, a hand-drawn icon on a known background is the *correct* input: the
expected output is knowable exactly, and the test cannot flake.

They are explicitly **not** a substitute for validating detection quality.
Nothing here resembles a real animated, semi-transparent game sprite, so no
perception threshold (match_threshold, texture_min_std, ...) may be justified
from these images. Those belong in tests/real_frames/, driven by captures taken
with tools/grab_frame.py.

Important convention: when producing a frame for a smaller or larger window,
build it at the reference resolution and resize the **whole frame**. That is how
the game actually renders - every asset scales together. Pasting a full-size
sprite onto a small canvas creates a scene that cannot occur in practice, and a
test built on it fails for the wrong reason.
"""

from __future__ import annotations

import pathlib
from typing import Iterable, Sequence, Tuple

import cv2
import numpy as np

#: Client area of the reference capture the templates are calibrated against.
REFERENCE_SIZE: Tuple[int, int] = (1707, 960)

#: Side length of the synthetic coin icon at the reference resolution.
COIN_SIZE = 40

Point = Tuple[int, int]


def make_coin_template(size: int = COIN_SIZE) -> np.ndarray:
    """A high-contrast icon whose structure survives resampling.

    Smooth, strongly coloured shapes are used on purpose: random noise would
    change character when resized and make scale tests meaningless.
    """
    image = np.zeros((size, size, 3), np.uint8)
    center = size // 2
    cv2.circle(image, (center, center), center - 2, (40, 200, 240), -1)
    cv2.circle(image, (center, center), center - 2, (10, 60, 90), 3)
    cv2.circle(
        image,
        (center - 5, center - 5),
        max(2, center // 4),
        (210, 255, 255),
        -1,
    )
    cv2.line(
        image,
        (center - 7, center + 7),
        (center + 7, center - 7),
        (15, 110, 170),
        3,
    )
    return image


def write_template(directory, name: str, image: np.ndarray) -> str:
    """Save a template as PNG and return its path."""
    path = pathlib.Path(directory) / f"{name}.png"
    ok, buffer = cv2.imencode(".png", image)
    assert ok, "failed to encode template"
    path.write_bytes(buffer.tobytes())
    return str(path)


def paste(scene: np.ndarray, patch: np.ndarray, center: Point) -> None:
    height, width = patch.shape[:2]
    x = center[0] - width // 2
    y = center[1] - height // 2
    scene[y : y + height, x : x + width] = patch


def make_scene(
    width: int,
    height: int,
    coin_centers: Sequence[Point] = (),
    monster_under: Iterable[Point] = (),
    seed: int = 7,
) -> np.ndarray:
    """Build a frame with a deliberately flat background.

    The background is smoothed low-amplitude noise so that its standard
    deviation stays well under the TextureBelow threshold. That makes the
    "nothing underneath" branch testable. Real game ground is far busier - which
    is exactly the assumption the calibration step has to check.
    """
    rng = np.random.default_rng(seed)
    scene = np.full((height, width, 3), 45, np.uint8)
    noise = rng.integers(0, 20, (height, width, 3), dtype=np.uint8)
    scene = cv2.add(scene, noise)
    scene = cv2.GaussianBlur(scene, (0, 0), 2.0)

    coin = make_coin_template()

    for center in monster_under:
        anchor = (center[0], center[1] + 45)
        cv2.ellipse(scene, anchor, (34, 26), 0, 0, 360, (30, 190, 60), -1)
        cv2.ellipse(scene, anchor, (34, 26), 0, 0, 360, (200, 40, 30), 4)
        cv2.circle(scene, (center[0] - 10, center[1] + 40), 5, (250, 250, 250), -1)
        cv2.circle(scene, (center[0] + 10, center[1] + 40), 5, (250, 250, 250), -1)

    for center in coin_centers:
        paste(scene, coin, center)

    return scene


def render_at(scene: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize a whole reference-resolution frame, mimicking a resized game window."""
    return cv2.resize(scene, size, interpolation=cv2.INTER_AREA)


def scale_point(point: Point, from_size: Tuple[int, int], to_size: Tuple[int, int]) -> Point:
    """Map a point between two window sizes."""
    return (
        int(round(point[0] * to_size[0] / from_size[0])),
        int(round(point[1] * to_size[1] / from_size[1])),
    )
