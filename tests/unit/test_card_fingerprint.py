from __future__ import annotations

import os
import cv2
import numpy as np
import pytest

from config import DEFAULT_CONFIG, PROJECT_ROOT
from core.letter_recognizer import KNOWN_ISLANDS, LetterRecognizer
from core.map_navigator import IslandCardInfo, compute_card_hash, hash_distance


def test_atomic_click_configuration_zero_hold_time() -> None:
    """Verify that press_duration is configured to (0.0, 0.0) so clicks are atomic,
    preventing hardware mouse movement from interleaving and converting clicks into drags."""
    assert DEFAULT_CONFIG.click.press_duration == (0.0, 0.0)


def test_compute_card_hash_and_distance() -> None:
    """Verify dHash produces identical hashes for identical images and large distance for different ones."""
    # Blank/invalid inputs return 0
    assert compute_card_hash(np.zeros((0, 0), dtype=np.uint8)) == 0

    # Image 1: horizontal gradient
    img1 = np.tile(np.linspace(0, 255, 30, dtype=np.uint8), (30, 1))
    # Image 2: slight noise on image 1
    img2 = np.clip(img1.astype(np.int16) + np.random.randint(-2, 3, img1.shape), 0, 255).astype(np.uint8)
    # Image 3: inverted gradient (completely different)
    img3 = 255 - img1

    h1 = compute_card_hash(img1)
    h2 = compute_card_hash(img2)
    h3 = compute_card_hash(img3)

    assert hash_distance(h1, h2) <= 2
    assert hash_distance(h1, h3) >= 30


def test_noisy_ocr_names_resolve_to_canonical() -> None:
    """Verify that real in-game noisy OCR readings from user runs map correctly to canonical islands."""
    rec = LetterRecognizer()

    noisy_samples = [
        ("rlaOt lSIaNU", "Plant Island"),
        ("COld lSlOnd", "Cold Island"),
        ("AiE lSIONd", "Air Island"),
        ("The Colossingum", "The Colossingum"),
        ("Fire Haven", "Fire Haven"),
    ]
    vocab = [expected for _, expected in noisy_samples]

    for raw, expected in noisy_samples:
        canon = rec.resolve_canonical_name(raw, vocabulary=vocab)
        assert canon == expected, f"Failed to map {raw!r} to {expected!r}, got {canon!r}"


def test_island_card_info_carries_hash_and_hist() -> None:
    """Verify IslandCardInfo correctly stores card_hash and card_hist."""
    hist = np.ones((16, 16), dtype=np.float32)
    card = IslandCardInfo(
        index=0,
        name="Plant Island",
        rect=(0, 70, 360, 179),
        click_point=(180, 124),
        is_fully_visible=True,
        card_hash=0x123456789ABCDEF0,
        card_hist=hist,
    )
    assert card.card_hash == 0x123456789ABCDEF0
    assert card.card_hist is not None
    assert card.is_fully_visible is True


def test_card_hist_invariance_across_selection_states() -> None:
    """Verify HSV palette histogram has high correlation (>= 0.95) across selected and unselected card states."""
    from core.map_navigator import compute_card_hist
    p_sel = os.path.join(PROJECT_ROOT, "scratch", "plant_selected.png")
    p_unsel = os.path.join(PROJECT_ROOT, "scratch", "plant_unselected.png")
    if not (os.path.isfile(p_sel) and os.path.isfile(p_unsel)):
        pytest.skip("scratch plant test crops not found")

    img_sel = cv2.imread(p_sel)
    img_unsel = cv2.imread(p_unsel)

    h_sel = compute_card_hist(img_sel)
    h_unsel = compute_card_hist(img_unsel)

    sim = cv2.compareHist(h_sel, h_unsel, cv2.HISTCMP_CORREL)
    assert sim >= 0.95, f"Expected histogram correlation >= 0.95, got {sim:.3f}"


def test_card_hist_separation_across_different_islands() -> None:
    """Verify HSV palette histogram has low correlation (< 0.30) between different islands."""
    from core.map_navigator import compute_card_hist
    fpath = os.path.join(PROJECT_ROOT, "captures", "map", "map_20260902_185615_714.png")
    if not os.path.isfile(fpath):
        pytest.skip("map capture not found")

    frame = cv2.imread(fpath)
    plant_crop = frame[70:179, 0:360]
    cold_crop = frame[179:267, 0:360]

    h_plant = compute_card_hist(plant_crop)
    h_cold = compute_card_hist(cold_crop)

    sim = cv2.compareHist(h_plant, h_cold, cv2.HISTCMP_CORREL)
    assert sim < 0.30, f"Expected low correlation between plant and cold, got {sim:.3f}"


def test_distinct_known_islands_are_not_falsely_marked_visited() -> None:
    """Verify that Air Island, Water Island, and Earth Island are not falsely considered
    visited when Plant Island and Cold Island are already in visited_names."""
    import difflib

    def strip_island_suffix(name: str) -> str:
        s = name.strip().lower()
        for suffix in (" island", " lsland", " lslond", " lsiond", " iuland", " islaou", " lula", " lzfund"):
            if s.endswith(suffix):
                s = s[: -len(suffix)].strip()
        return s

    visited_names = {"plant island", "cold island"}

    for candidate in ["Air Island", "Water Island", "Earth Island", "Shugabush Island", "Fire Haven"]:
        clean = candidate.strip().lower()
        is_known = any(k.lower() == clean for k in KNOWN_ISLANDS)

        # 1. Exact
        exact = clean in visited_names
        # 2. Fuzzy only if not known
        fuzzy = False
        if not is_known:
            dist = strip_island_suffix(clean)
            for v in visited_names:
                v_dist = strip_island_suffix(v)
                if difflib.SequenceMatcher(None, dist, v_dist).ratio() >= 0.70:
                    fuzzy = True
                    break

        assert not exact, f"{candidate} had false exact match"
        assert not fuzzy, f"{candidate} had false fuzzy match"
