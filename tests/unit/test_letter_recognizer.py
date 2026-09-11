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


def test_recognize_card_1080p_plant_island_no_tv_noise(recognizer: LetterRecognizer) -> None:
    """Verify 1080p Plant Island card ignores TV/icon noise and resolves cleanly to Plant Island."""
    map_frame_path = os.path.join(PROJECT_ROOT, "captures", "map", "map_20260902_185615_714.png")
    if not os.path.isfile(map_frame_path):
        pytest.skip("captures/map frame not found")

    frame = cv2.imread(map_frame_path)
    f1080 = cv2.resize(frame, (1920, 1080))
    y1 = int(70 * 1080 / 768)
    y2 = int(179 * 1080 / 768)
    w_card = int(360 * 1920 / 1024)
    plant_crop = f1080[y1:y2, 0:w_card]

    name = recognizer.recognize_card(plant_crop)
    assert name == "Plant Island"
    assert "TV" not in name
    assert "Istana" not in name
    assert "iU" not in name


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


def test_clean_title_tokens_badge_noise_and_filtering(recognizer: LetterRecognizer) -> None:
    """Verify clean_title_tokens normalizes casing, punctuation, and whitespace."""
    assert recognizer.clean_title_tokens("Earth Island") == "earth island"
    assert recognizer.clean_title_tokens("Plant Island") == "plant island"
    assert recognizer.clean_title_tokens("  Cold   Island  ") == "cold island"
    assert recognizer.clean_title_tokens("，。，") == ""
    assert recognizer.clean_title_tokens("。%") == ""


def test_names_fuzzy_match_badge_and_typo_tolerance(recognizer: LetterRecognizer) -> None:
    """Verify names_fuzzy_match matches target islands despite badge suffixes and OCR typos."""
    # Badge suffixes
    assert recognizer.names_fuzzy_match("Earth Island A", "Earth Island") is True
    assert recognizer.names_fuzzy_match("Light Island @", "Light Island") is True

    # Visited/blacklist resilience to badge suffixes
    assert recognizer.is_blacklisted("Earth Island A", ["Earth Island"]) is True
    assert recognizer.is_blacklisted("Light Island @", ["Light Island"]) is True

    # Distinct islands must never match
    assert recognizer.names_fuzzy_match("Cold Island", "Gold Island") is False
    assert recognizer.names_fuzzy_match("Water Island", "Fire Oasis") is False


def test_extract_title_crop_excludes_building_noise(recognizer: LetterRecognizer) -> None:
    """Verify extract_title_crop excludes distant artwork/building graphics to the right."""
    # Create synthetic card image (109x360)
    card = np.zeros((109, 360, 3), dtype=np.uint8)
    card[:] = (30, 40, 50)  # dark background

    # Draw title letters: "COLD" with black border and white interior
    # Letter height ~16px, outline border 2px
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Draw black stroke first
    cv2.putText(card, "COLD", (140, 30), font, 0.7, (0, 0, 0), thickness=4, lineType=cv2.LINE_AA)
    # Draw white fill
    cv2.putText(card, "COLD", (140, 30), font, 0.7, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    # Draw distant building with white highlight on the right (x=330, gap > 30px)
    # Building highlight has no black border (or partial border)
    card[15:35, 320:345] = (240, 240, 240)

    crop = recognizer.extract_title_crop(card)
    assert crop is not None
    # Verify the crop does NOT extend to x=320..345
    # The crop width should cover the title (around 140..250 with pad) but not reach the building
    assert crop.shape[1] < 200

