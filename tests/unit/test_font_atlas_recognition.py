"""Unit tests for authentic game font atlas extraction and character recognition."""

from __future__ import annotations

import json
import os
from collections import deque

import cv2
import numpy as np
import pytest

from config import DEFAULT_CONFIG, PROJECT_ROOT
from core.bot_engine import BotEngine, QueuedIsland
from core.letter_recognizer import LetterRecognizer, KNOWN_ISLANDS
from core.map_navigator import MapNavigator, hash_distance


def test_font_atlases_present() -> None:
    """Verify authentic font atlas files are stored in assets/fonts."""
    fonts_dir = os.path.join(PROJECT_ROOT, "assets", "fonts")
    assert os.path.isdir(fonts_dir), "assets/fonts directory missing"

    for atlas_name in ("font_atlas_retina.png", "font_atlas_standard.png", "font_atlas_mask.png"):
        p = os.path.join(fonts_dir, atlas_name)
        assert os.path.isfile(p), f"Atlas file missing: {atlas_name}"
        assert os.path.getsize(p) > 10000, f"Atlas file too small: {atlas_name}"


def test_extracted_letters_complete() -> None:
    """Verify all 52 letters and letters.json metadata exist in assets/letters."""
    letters_dir = os.path.join(PROJECT_ROOT, "assets", "letters")
    meta_path = os.path.join(letters_dir, "letters.json")
    assert os.path.isfile(meta_path), "letters.json missing"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert len(meta) == 52, f"Expected 52 letter metadata entries, found {len(meta)}"

    for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        k = f"upper_{c}"
        assert k in meta, f"Missing metadata for {k}"
        assert meta[k]["is_upper"] is True
        assert os.path.isfile(os.path.join(letters_dir, f"{k}.png")), f"Missing PNG for {k}"

    for c in "abcdefghijklmnopqrstuvwxyz":
        k = f"lower_{c}"
        assert k in meta, f"Missing metadata for {k}"
        assert meta[k]["is_upper"] is False
        assert os.path.isfile(os.path.join(letters_dir, f"{k}.png")), f"Missing PNG for {k}"


def test_template_properties_and_topology() -> None:
    """Verify templates are 24x24 normalized arrays and have correct hole topologies."""
    rec = LetterRecognizer()
    assert len(rec._templates) >= 52

    # Check canvas shape and bounds
    for key, (info, tmpl) in rec._templates.items():
        assert tmpl.shape == (24, 24), f"Wrong shape for {key}: {tmpl.shape}"
        assert tmpl.dtype == np.float32
        assert tmpl.min() >= 0.0
        assert tmpl.max() <= 1.0

    # Hole topology assertions
    assert rec._template_holes["upper_P"] == 1, "upper_P must have 1 hole"
    assert rec._template_holes["upper_O"] == 1, "upper_O must have 1 hole"
    assert rec._template_holes["upper_D"] == 1, "upper_D must have 1 hole"
    assert rec._template_holes["upper_A"] == 1, "upper_A must have 1 hole"
    assert rec._template_holes["upper_F"] == 0, "upper_F must have 0 holes"
    assert rec._template_holes["upper_E"] == 0, "upper_E must have 0 holes"
    assert rec._template_holes["upper_C"] == 0, "upper_C must have 0 holes"


def test_real_capture_all_six_islands_recognized() -> None:
    """Verify all 6 visible island cards in real capture frame resolve to canonical names."""
    fpath = os.path.join(PROJECT_ROOT, "captures", "map", "map_20260902_185615_714.png")
    if not os.path.isfile(fpath):
        pytest.skip("capture frame not found")

    frame = cv2.imread(fpath)
    nav = MapNavigator(None, None, DEFAULT_CONFIG)
    rec = LetterRecognizer()
    cards = nav.get_visible_cards(frame)
    assert len(cards) == 6, f"Expected 6 visible cards, found {len(cards)}"

    # All 6 cards have valid non-empty raw OCR strings and unique card hashes
    assert all(len(c.name.strip()) >= 3 for c in cards)
    assert len({c.card_hash for c in cards}) == 6

    expected_islands = [
        "Plant Island",
        "Cold Island",
        "Air Island",
        "Water Island",
        "Earth Island",
        "Shugabush Island",
    ]

    # Direct OCR output matches expected islands with 100% accuracy
    for idx, (exp, actual) in enumerate(zip(expected_islands, [c.name for c in cards])):
        assert actual == exp, f"Card {idx} mismatch: expected {exp!r}, got {actual!r}"


def test_queue_enqueue_all_six_without_collision() -> None:
    """Verify that queue enqueues all 6 cards without collision using perceptual hash and names."""
    fpath = os.path.join(PROJECT_ROOT, "captures", "map", "map_20260902_185615_714.png")
    if not os.path.isfile(fpath):
        pytest.skip("capture frame not found")

    frame = cv2.imread(fpath)
    nav = MapNavigator(None, None, DEFAULT_CONFIG)
    cards = nav.get_visible_cards(frame)

    island_queue: deque[QueuedIsland] = deque()
    for c in cards:
        already = any(hash_distance(c.card_hash, q.card_hash) <= 6 for q in island_queue)
        if not already:
            island_queue.append(
                QueuedIsland(name=c.name, canon_name=c.name, card_hash=c.card_hash, card_hist=c.card_hist)
            )

    assert len(island_queue) == 6, f"Expected queue length 6, got {len(island_queue)}"
