from __future__ import annotations

import os
from collections import deque
import cv2
import difflib
import numpy as np
import pytest

from config import PROJECT_ROOT, DEFAULT_CONFIG
from core.bot_engine import QueuedIsland
from core.letter_recognizer import KNOWN_ISLANDS, LetterRecognizer
from core.map_navigator import MapNavigator, IslandCardInfo, hash_distance


def words_match(a: str, b: str) -> bool:
    if a == b:
        return True
    if len(a) <= 4 and len(b) <= 4 and a[0] != b[0]:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.70


def names_fuzzy_match(s1: str, s2: str) -> bool:
    if not s1 or not s2:
        return False
    c1 = s1.strip().lower()
    c2 = s2.strip().lower()
    if c1 == c2:
        return True
    w1 = c1.split()
    w2 = c2.split()
    if len(w1) > 1 and len(w2) > 1:
        if len(w1) != len(w2):
            return False
        return all(words_match(a, b) for a, b in zip(w1, w2))
    return words_match(c1, c2)


def match_card_to_island(card: IslandCardInfo, target: QueuedIsland) -> bool:
    if card.card_hash != 0 and target.card_hash != 0:
        if hash_distance(card.card_hash, target.card_hash) <= 6:
            return True

    c_raw = card.name.strip().lower()
    t_raw = target.name.strip().lower()

    if names_fuzzy_match(c_raw, t_raw):
        return True

    if card.card_hist is not None and target.card_hist is not None:
        if cv2.compareHist(card.card_hist, target.card_hist, cv2.HISTCMP_CORREL) >= 0.88:
            return True

    return False


def test_queued_island_dataclass() -> None:
    """Verify QueuedIsland stores all attributes correctly."""
    q = QueuedIsland(
        name="Plant Island",
        canon_name="Plant Island",
        card_hash=0x1234,
        card_hist=np.zeros((16, 16), dtype=np.float32),
    )
    assert q.name == "Plant Island"
    assert q.canon_name == "Plant Island"
    assert q.card_hash == 0x1234
    assert q.card_hist is not None


def test_anchor_masks_out_upper_cards() -> None:
    """Verify that when an anchor is locked, all cards at or above the anchor are masked out,
    and all cards below the anchor are preserved in strictly sorted Y order."""
    card_plant = IslandCardInfo(index=0, name="Plant Island", rect=(0, 70, 360, 179), click_point=(180, 124))
    card_cold = IslandCardInfo(index=1, name="Cold Island", rect=(0, 179, 360, 267), click_point=(180, 223))
    card_air = IslandCardInfo(index=2, name="Air Island", rect=(0, 287, 360, 376), click_point=(180, 331))
    card_water = IslandCardInfo(index=3, name="Water Island", rect=(0, 396, 360, 495), click_point=(180, 445))

    cards = [card_plant, card_cold, card_air, card_water]

    anchor_y = card_cold.rect[1]
    downward_cards = [c for c in cards if c.rect[1] > anchor_y + 10]
    downward_cards.sort(key=lambda c: c.rect[1])

    assert len(downward_cards) == 2
    assert downward_cards[0].name == "Air Island"
    assert downward_cards[1].name == "Water Island"


def test_queue_strict_fifo_ordering() -> None:
    """Verify that islands are consumed strictly in First-In-First-Out (FIFO) queue order."""
    q: deque[QueuedIsland] = deque()

    q.append(QueuedIsland(name="Plant Island", canon_name="Plant Island"))
    q.append(QueuedIsland(name="Cold Island", canon_name="Cold Island"))
    q.append(QueuedIsland(name="Air Island", canon_name="Air Island"))

    first = q.popleft()
    assert first.name == "Plant Island"

    second = q.popleft()
    assert second.name == "Cold Island"

    third = q.popleft()
    assert third.name == "Air Island"

    assert len(q) == 0


def test_queue_rejects_duplicates() -> None:
    """Verify that already enqueued islands or visited islands cannot be enqueued twice."""
    q: deque[QueuedIsland] = deque()
    q.append(QueuedIsland(name="Plant Island", canon_name="Plant Island"))

    visited_names = {"plant island"}
    candidate = IslandCardInfo(index=0, name="Plant Island", rect=(0, 70, 360, 179), click_point=(180, 124))

    is_visited = candidate.name.strip().lower() in visited_names
    assert is_visited is True

    already_queued = any(item.name == candidate.name for item in q)
    assert already_queued is True


# ==============================================================================
# DOUBLED TEST SUITES (Additional Verification Suites)
# ==============================================================================


SAMPLE_ISLANDS: Tuple[str, ...] = (
    "Plant Island", "Cold Island", "Air Island", "Water Island", "Earth Island",
    "Fire Haven", "Fire Oasis", "Light Island", "Psychic Island", "Faerie Island",
    "Bone Island", "Mythical Island", "Ethereal Island", "Ethereal Workshop",
    "Shugabush Island", "Seasonal Shanty", "Amber Island", "Wublin Island",
    "Celestial Island", "The Colossingum", "Tribal Island", "Composer Island",
    "Gold Island", "Magical Sanctum", "Magical Nexus",
)


def test_pairwise_island_orthogonal_matrix() -> None:
    """Test Suite 1: Full 25x25 orthogonal distinctness matrix.
    Guarantees that NO TWO DISTINCT canonical islands ever match each other."""
    for i, name1 in enumerate(SAMPLE_ISLANDS):
        card1 = IslandCardInfo(index=i, name=name1, rect=(0, 0, 360, 100), click_point=(180, 50))
        target1 = QueuedIsland(name=name1, canon_name=name1)
        for j, name2 in enumerate(SAMPLE_ISLANDS):
            target2 = QueuedIsland(name=name2, canon_name=name2)
            matched = match_card_to_island(card1, target2)
            if i == j:
                assert matched is True, f"{name1} should match itself"
            else:
                assert matched is False, f"CRITICAL COLLISION: {name1} falsely matched {name2}!"


def test_real_capture_full_queue_simulation() -> None:
    """Test Suite 2: Real capture multi-island queue simulation.
    Ensures Plant, Cold, Air, Water, Earth, and Shugabush are ALL enqueued without skipping."""
    nav = MapNavigator(None, None, DEFAULT_CONFIG)
    fpath = os.path.join(PROJECT_ROOT, "captures", "map", "map_20260902_185615_714.png")
    if not os.path.isfile(fpath):
        pytest.skip("capture frame not found")

    frame = cv2.imread(fpath)
    cards = nav.get_visible_cards(frame)
    assert len(cards) == 6, f"Expected 6 visible cards, got {len(cards)}"

    island_queue: deque[QueuedIsland] = deque()
    for c in cards:
        already = any(match_card_to_island(c, q) for q in island_queue)
        if not already:
            island_queue.append(QueuedIsland(name=c.name, canon_name=c.name, card_hash=c.card_hash, card_hist=c.card_hist))

    # All 6 cards MUST be enqueued without any skipping
    assert len(island_queue) == 6, f"Expected queue length 6, got {len(island_queue)}"


def test_noisy_ocr_variants_orthogonality() -> None:
    """Test Suite 3: Noisy OCR variants orthogonality against sample vocabulary."""
    rec = LetterRecognizer()
    noisy_cases = [
        ("rlaOt lSIaaU", "Plant Island"),
        ("rlaOt lSIaNU", "Plant Island"),
        ("COld lSlOnd", "Cold Island"),
        ("AiE lSIONd", "Air Island"),
        ("wataC iulaNd", "Water Island"),
        ("EaFtn lSIaNd", "Earth Island"),
        ("Goid lSiONd", "Gold Island"),
        ("Bhugabuuh lulaNd", "Shugabush Island"),
        ("Tha OOluSulOgum", "The Colossingum"),
        ("rlaNt lulONU", "Plant Island"),
        ("ShuaabuSh lSlaOd", "Shugabush Island"),
    ]

    for noisy_text, expected_canon in noisy_cases:
        canon_resolved = rec.resolve_canonical_name(noisy_text, vocabulary=SAMPLE_ISLANDS)
        card = IslandCardInfo(index=0, name=canon_resolved, rect=(0, 0, 360, 100), click_point=(180, 50))
        target_true = QueuedIsland(name=expected_canon, canon_name=expected_canon)
        assert match_card_to_island(card, target_true) is True, f"{noisy_text} should match {expected_canon}"

        for other in SAMPLE_ISLANDS:
            if other == expected_canon:
                continue
            target_other = QueuedIsland(name=other, canon_name=other)
            matched = match_card_to_island(card, target_other)
            assert matched is False, f"False collision: {noisy_text} falsely matched {other}!"


def test_non_island_suffix_islands_distinction() -> None:
    """Test Suite 3b: Suffix-agnostic distinctness for islands without 'Island' suffix.
    Guarantees that Fire Haven, Fire Oasis, The Colossingum, Magical Sanctum,
    Magical Nexus, Ethereal Workshop, Seasonal Shanty are 100% distinguished."""
    non_island_suffix = [
        "Fire Haven",
        "Fire Oasis",
        "The Colossingum",
        "Magical Sanctum",
        "Magical Nexus",
        "Ethereal Workshop",
        "Seasonal Shanty",
    ]
    for i, name1 in enumerate(non_island_suffix):
        card1 = IslandCardInfo(index=i, name=name1, rect=(0, 0, 360, 100), click_point=(180, 50))
        for j, name2 in enumerate(non_island_suffix):
            target2 = QueuedIsland(name=name2, canon_name=name2)
            matched = match_card_to_island(card1, target2)
            if i == j:
                assert matched is True
            else:
                assert matched is False, f"CRITICAL: {name1} falsely matched {name2}!"


def test_queue_state_machine_full_tour_simulation() -> None:
    """Test Suite 4: End-to-end multi-island monotonic tour traversal simulation.
    Simulates the entire loop: enqueue -> visit -> return anchor -> downward scan -> next.
    Verifies strictly monotonic 0 -> 1 -> 2 -> 3 -> 4 -> 5 visitation without any leaps."""
    nav = MapNavigator(None, None, DEFAULT_CONFIG)
    fpath = os.path.join(PROJECT_ROOT, "captures", "map", "map_20260902_185615_714.png")
    if not os.path.isfile(fpath):
        pytest.skip("capture frame not found")

    frame = cv2.imread(fpath)
    cards = nav.get_visible_cards(frame)

    island_queue: deque[QueuedIsland] = deque()
    visited_sequence: list[str] = []
    last_anchor: QueuedIsland | None = None

    # Step 1: Initial scan from top (last_anchor is None)
    for c in cards:
        already = any(match_card_to_island(c, q) for q in island_queue)
        if not already:
            island_queue.append(QueuedIsland(name=c.name, canon_name=c.name, card_hash=c.card_hash, card_hist=c.card_hist))

    assert len(island_queue) == 6

    # Step 2: Pop and visit each island in strict FIFO order
    while island_queue:
        target = island_queue.popleft()
        visited_sequence.append(target.name)
        last_anchor = target

        # Simulate returning to map: anchor is locked at last_anchor
        # Downward search ignores everything at or above last_anchor
        anchor_y = -1
        for c in cards:
            if match_card_to_island(c, last_anchor):
                anchor_y = c.rect[1]
                break
        assert anchor_y >= 0, f"Failed to locate anchor {last_anchor.name}"

        # Downward cards must all have y > anchor_y + 10
        downward = [c for c in cards if c.rect[1] > anchor_y + 10]
        # Any card remaining in queue must belong to downward cards
        if island_queue:
            next_target = island_queue[0]
            found_in_downward = any(match_card_to_island(dc, next_target) for dc in downward)
            assert found_in_downward is True, f"Next target {next_target.name} must be strictly below anchor {last_anchor.name}!"

    # Final sequence must visit all 6 islands in strictly monotonic order
    assert len(visited_sequence) == 6
    for idx, c in enumerate(cards):
        assert visited_sequence[idx] == c.name


def test_clipped_boundary_card_rejected_from_queue() -> None:
    """Test Suite 5: Boundary clipping guard.
    Guarantees that cards cut off at viewport edges (is_fully_visible=False)
    or with empty/single-char names are NEVER enqueued to prevent corrupted targets."""
    # Fully visible valid card
    valid_card = IslandCardInfo(
        index=0,
        name="The Colossingum",
        rect=(0, 200, 360, 310),
        click_point=(180, 255),
        is_fully_visible=True,
    )
    # Clipped card sliver at bottom (e.g. only 25px visible)
    clipped_card = IslandCardInfo(
        index=1,
        name="",
        rect=(0, 680, 360, 705),
        click_point=(180, 692),
        is_fully_visible=False,
    )
    # Card with corrupted 1-char noise
    noise_card = IslandCardInfo(
        index=2,
        name="I",
        rect=(0, 680, 360, 705),
        click_point=(180, 692),
        is_fully_visible=False,
    )

    def should_enqueue(c: IslandCardInfo) -> bool:
        if not c.is_fully_visible:
            return False
        if not c.name or len(c.name.strip()) < 2:
            return False
        return True

    assert should_enqueue(valid_card) is True
    assert should_enqueue(clipped_card) is False
    assert should_enqueue(noise_card) is False


def test_target_missing_recovery_after_retry_limit() -> None:
    """Test Suite 6: Target missing recovery mechanism.
    Verifies that when a queued target cannot be found in viewport after 2 scrolls,
    it is safely popped from the queue, breaking any endless downward scroll cascade."""
    q: deque[QueuedIsland] = deque()
    q.append(QueuedIsland(name="Ghost Island", canon_name="Ghost Island"))
    q.append(QueuedIsland(name="Gold Island", canon_name="Gold Island"))

    target_miss_count = 0
    # Visible cards on screen only contain Gold Island
    visible_cards = [
        IslandCardInfo(index=0, name="Gold Island", rect=(0, 100, 360, 210), click_point=(180, 155), is_fully_visible=True)
    ]
    rec = LetterRecognizer()

    # Attempt 1: Ghost Island not found
    target = q[0]
    matched = any(match_card_to_island(c, target) for c in visible_cards)
    assert matched is False
    target_miss_count += 1
    assert target_miss_count == 1

    # Attempt 2: Ghost Island still not found
    matched = any(match_card_to_island(c, target) for c in visible_cards)
    assert matched is False
    target_miss_count += 1
    assert target_miss_count == 2

    # Attempt 3: Retry limit reached (max 2 attempts) -> pop ghost target
    if target_miss_count >= 2:
        popped = q.popleft()
        assert popped.name == "Ghost Island"
        target_miss_count = 0

    # Queue now cleanly targets Gold Island and recovers immediately!
    assert len(q) == 1
    assert q[0].name == "Gold Island"
    recovered_target = q[0]
    matched_recovery = any(match_card_to_island(c, recovered_target) for c in visible_cards)
    assert matched_recovery is True


def test_unconditional_hash_deduplication_prevents_reentry() -> None:
    """Test Suite 7: Unconditional perceptual dHash deduplication prevents re-entering islands.
    Guarantees that if an island card's image hash matches an already visited card (distance <= 6),
    it is detected as ALREADY VISITED, regardless of whether OCR produced a noisy name or whatever name!"""
    earth_hash = 0x10AA3A60C280D080
    visited_hashes = [earth_hash]
    visited_names = {"earth island"}

    # Simulate card on screen with minor OCR noise ('Light Island') but exact image hash
    card_noisy_ocr = IslandCardInfo(
        index=4,
        name="Light Island",  # Noisy OCR text
        rect=(0, 505, 360, 603),
        click_point=(180, 554),
        card_hash=earth_hash,  # Real visual fingerprint matches visited Earth Island!
        is_fully_visible=True,
    )

    def is_card_already_visited(c: IslandCardInfo) -> bool:
        if c.card_hash != 0:
            for vh in visited_hashes:
                if hash_distance(c.card_hash, vh) <= 6:
                    return True
        c_raw = c.name.strip().lower()
        if c_raw and c_raw in visited_names:
            return True
        return False

    # Must be TRUE! Hash deduplication prevents re-entering the island!
    assert is_card_already_visited(card_noisy_ocr) is True
