"""Unit tests for the character-level letter recognizer and blacklist filter."""

from __future__ import annotations

import os
import cv2
import numpy as np
import pytest

from config import PROJECT_ROOT
from core.letter_recognizer import LetterRecognizer


@pytest.fixture
def recognizer() -> LetterRecognizer:
    return LetterRecognizer()


def test_letter_templates_loaded(recognizer: LetterRecognizer) -> None:
    """Verify all 52 letter templates are loaded."""
    assert len(recognizer._templates) >= 52


def test_is_blacklisted_exact_and_case_insensitive(recognizer: LetterRecognizer) -> None:
    """Test blacklist matching is case-insensitive and whitespace tolerant."""
    blacklist = ["Cold Island", "the colossingum", "  Gold Island  "]

    assert recognizer.is_blacklisted("Cold Island", blacklist) is True
    assert recognizer.is_blacklisted("cold island", blacklist) is True
    assert recognizer.is_blacklisted("COLD ISLAND", blacklist) is True
    assert recognizer.is_blacklisted("The Colossingum", blacklist) is True
    assert recognizer.is_blacklisted("Gold Island", blacklist) is True
    assert recognizer.is_blacklisted("gold island", blacklist) is True

    # Non-blacklisted islands
    assert recognizer.is_blacklisted("Plant Island", blacklist) is False
    assert recognizer.is_blacklisted("Water Island", blacklist) is False
    assert recognizer.is_blacklisted("Fire Oasis", blacklist) is False


def test_is_blacklisted_empty_inputs(recognizer: LetterRecognizer) -> None:
    """Test blacklist handles empty and invalid inputs gracefully."""
    assert recognizer.is_blacklisted("", ["Cold Island"]) is False
    assert recognizer.is_blacklisted("Cold Island", []) is False
    assert recognizer.is_blacklisted("Cold Island", ["", "   "]) is False


def test_recognize_card_real_context(recognizer: LetterRecognizer) -> None:
    """Test recognition on a real card crop from captures/map."""
    map_frame_path = os.path.join(PROJECT_ROOT, "captures", "map", "map_20260902_185615_714.png")
    if not os.path.isfile(map_frame_path):
        pytest.skip("captures/map frame not found")

    frame = cv2.imread(map_frame_path)
    # Cold Island crop (y: 179..267, x: 0..360)
    card_crop = frame[179:267, 0:360]
    name = recognizer.recognize_card(card_crop)
    # High-accuracy OCR string
    assert name == "Cold Island"
    # Ensure blacklist detects it robustly
    assert recognizer.is_blacklisted(name, ["Cold Island"]) is True


def test_resolve_canonical_name(recognizer: LetterRecognizer) -> None:
    """Verify raw OCR variations resolve to correct canonical island names when vocabulary provided."""
    vocab = [
        "Plant Island", "Cold Island", "Air Island", "Water Island",
        "Earth Island", "Gold Island", "Shugabush Island", "The Colossingum",
    ]
    # Without vocabulary: returns raw text unmodified (zero hardcoded assumptions)
    assert recognizer.resolve_canonical_name("rlaOt lSIaNU") == "rlaOt lSIaNU"

    # With explicit vocabulary: resolves to closest target
    assert recognizer.resolve_canonical_name("rlaOt lSIaNU", vocabulary=vocab) == "Plant Island"
    assert recognizer.resolve_canonical_name("COld lSlOnd", vocabulary=vocab) == "Cold Island"
    assert recognizer.resolve_canonical_name("AiE lSIONd", vocabulary=vocab) == "Air Island"
    assert recognizer.resolve_canonical_name("wataC iulaNd", vocabulary=vocab) == "Water Island"
    assert recognizer.resolve_canonical_name("EaFtn lSIaNd", vocabulary=vocab) == "Earth Island"
    assert recognizer.resolve_canonical_name("Goid lSiONd", vocabulary=vocab) == "Gold Island"
    assert recognizer.resolve_canonical_name("Bhugabuuh lulaNd", vocabulary=vocab) == "Shugabush Island"
    assert recognizer.resolve_canonical_name("Tha OOluSulOgum", vocabulary=vocab) == "The Colossingum"
    assert recognizer.resolve_canonical_name("rlaNt lulONU", vocabulary=vocab) == "Plant Island"
    assert recognizer.resolve_canonical_name("ShuaabuSh lSlaOd", vocabulary=vocab) == "Shugabush Island"


def test_blacklist_distinguishes_cold_and_gold(recognizer: LetterRecognizer) -> None:
    """Verify Cold Island and Gold Island are never confused by blacklist matching."""
    # Blacklisting Gold Island should NOT blacklist Cold Island
    assert recognizer.is_blacklisted("Cold Island", ["Gold Island"]) is False
    assert recognizer.is_blacklisted("COld lSlOnd", ["Gold Island"]) is False

    # Blacklisting Air Island should match Air Island OCR
    assert recognizer.is_blacklisted("AiE lSIONd", ["Air Island"]) is True
    assert recognizer.is_blacklisted("Air Island", ["Air Island"]) is True
    assert recognizer.is_blacklisted("Plant Island", ["Air Island"]) is False
