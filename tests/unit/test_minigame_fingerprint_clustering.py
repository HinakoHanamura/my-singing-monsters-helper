"""Runtime clustering of card faces: comparison, refusal, and grouping.

Scope
-----
The patches here are generated, not captured, which makes them right for checking
behaviour (same picture groups together, different pictures do not, unreadable
input is refused) and wrong for choosing numbers. **No threshold may be
calibrated from this file.** The measured distributions come from
``tools/calibrate_face_similarity.py`` over ``captures/memory``:

    true pairs   median 0.983   p5 0.922   p1 0.691   min 0.337
    non-pairs    median 0.062   p95 0.299  p99 0.458  max 0.462

Every generated patch is textured on purpose. Normalised correlation divides by
each patch's standard deviation, so between two uniform patches it is undefined --
and an earlier version of the runner tests used flat colours, which made every
card compare as identical and the loop cheerfully "match" pairs that do not
match. That is exactly the failure the refusal path now prevents, so it is
asserted here rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.minigames.fingerprint import (
    DEFAULT_FINGERPRINT_PARAMS,
    UNCOMPARABLE,
    FaceRegistry,
    FingerprintParams,
    crop_face,
    glow_score,
    has_structure,
    is_settled,
    looks_stable,
    mutual_best_pairs,
    pairwise_scores,
    patch_std,
    similarity,
)

SIDE = DEFAULT_FINGERPRINT_PARAMS.canonical_size


def textured(seed: int, side: int = SIDE) -> np.ndarray:
    """A reproducible, high-variance patch standing in for card art."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(side, side, 3), dtype=np.uint8)


def flat(value: int = 120, side: int = SIDE) -> np.ndarray:
    patch = np.zeros((side, side, 3), np.uint8)
    patch[:, :] = (value, value, value)
    return patch


def noisy_copy(patch: np.ndarray, amplitude: int = 4, seed: int = 7) -> np.ndarray:
    """The same picture as seen one capture later, with a little capture noise."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(-amplitude, amplitude + 1, size=patch.shape, dtype=np.int16)
    return np.clip(patch.astype(np.int16) + noise, 0, 255).astype(np.uint8)


# --- comparing ---------------------------------------------------------------


def test_a_patch_matches_itself():
    patch = textured(1)

    assert similarity(patch, patch) == pytest.approx(1.0, abs=1e-6)


def test_the_same_picture_survives_capture_noise():
    patch = textured(2)

    score = similarity(patch, noisy_copy(patch))

    assert score > DEFAULT_FINGERPRINT_PARAMS.match_threshold


def test_different_pictures_score_low():
    assert similarity(textured(3), textured(4)) < DEFAULT_FINGERPRINT_PARAMS.match_threshold


@pytest.mark.parametrize(
    "a, b",
    [
        (None, None),
        (None, textured(5)),
        (textured(5), None),
    ],
    ids=["both_missing", "left_missing", "right_missing"],
)
def test_missing_patches_are_uncomparable(a, b):
    assert similarity(a, b) == UNCOMPARABLE


def test_mismatched_shapes_are_uncomparable():
    """Crops of different sizes must never be correlated.

    Cards run from 63 px to 122 px across the nine levels, so this is the guard
    that turns "forgot to normalise the scale" into a refusal instead of a number
    that looks plausible.
    """
    assert similarity(textured(6, side=48), textured(6, side=32)) == UNCOMPARABLE


# --- refusing patches with no structure --------------------------------------


def test_uniform_patches_are_uncomparable():
    """The dangerous case: correlation between two constants is undefined.

    Left to OpenCV this can read as a perfect match, and a false match makes the
    solver click an impossible pair while recording a face that is not there.
    """
    assert similarity(flat(120), flat(200)) == UNCOMPARABLE
    assert similarity(flat(120), flat(120)) == UNCOMPARABLE


def test_structure_is_measured_not_assumed():
    assert has_structure(textured(7)) is True
    assert has_structure(flat()) is False
    assert patch_std(flat()) == pytest.approx(0.0)
    assert patch_std(textured(7)) > DEFAULT_FINGERPRINT_PARAMS.min_patch_std


def test_a_washed_out_card_is_refused_rather_than_grouped():
    """A card lit by the match flash approaches uniform white."""
    registry = FaceRegistry()

    assert registry.identify(flat(250)) is None
    assert registry.group_count == 0


# --- grouping ---------------------------------------------------------------


def test_the_same_picture_gets_the_same_key():
    registry = FaceRegistry()
    patch = textured(8)

    first = registry.identify(patch)
    second = registry.identify(noisy_copy(patch))

    assert first is not None and second is not None
    assert first.is_new is True
    assert second.is_new is False
    assert second.key == first.key
    assert registry.group_count == 1


def test_different_pictures_get_different_keys():
    registry = FaceRegistry()

    keys = {registry.identify(textured(seed)).key for seed in (10, 11, 12, 13)}

    assert len(keys) == 4
    assert registry.group_count == 4


def test_a_new_group_reports_what_it_nearly_matched():
    """The margin is the only runtime evidence that the threshold still holds."""
    registry = FaceRegistry()
    registry.identify(textured(14))

    match = registry.identify(textured(15))

    assert match is not None
    assert match.is_new is True
    assert match.rival_key == 0
    assert match.best_score < DEFAULT_FINGERPRINT_PARAMS.match_threshold


def test_empty_input_is_refused():
    registry = FaceRegistry()

    assert registry.identify(None) is None
    assert registry.identify(np.zeros((0, 0, 3), np.uint8)) is None
    assert registry.group_count == 0


def test_group_overflow_detects_a_split_picture():
    """More pictures than the level has pairs means perception has diverged.

    Acting on it matters: a spurious group has no partner on the board, so the
    solver would keep probing for one until the level stalled.
    """
    registry = FaceRegistry()
    for seed in (20, 21, 22):
        registry.identify(textured(seed))

    assert registry.group_overflow(pair_count=2) is True
    assert registry.group_overflow(pair_count=3) is False


def test_the_closest_two_groups_are_the_split_halves():
    """Which two groups to rejoin is a measurement, not a choice.

    A picture split by a threshold that was very slightly too strict leaves two
    groups that still look far more like each other than like anything else on
    the board, so "most similar" identifies them.
    """
    registry = FaceRegistry(FingerprintParams(match_threshold=0.999))
    patch = textured(90)
    registry.identify(patch)
    registry.identify(noisy_copy(patch, amplitude=30, seed=91))
    registry.identify(textured(92))
    assert registry.group_count == 3

    closest = registry.closest_groups()

    assert closest is not None
    keep, drop, score = closest
    assert {keep, drop} == {0, 1}
    assert score > similarity(patch, textured(92))


def test_rejoining_two_groups_leaves_one():
    registry = FaceRegistry(FingerprintParams(match_threshold=0.999))
    patch = textured(93)
    registry.identify(patch)
    registry.identify(noisy_copy(patch, amplitude=30, seed=94))
    assert registry.group_count == 2

    registry.merge(0, 1)

    assert registry.group_count == 1
    assert registry.keys() == [0]


def test_a_single_group_has_nothing_to_rejoin():
    """The caller has to be able to tell "no repair available" from "repaired"."""
    registry = FaceRegistry()
    registry.identify(textured(95))

    assert registry.closest_groups() is None


@pytest.mark.parametrize(
    "keep, drop",
    [(0, 0), (0, 99)],
    ids=["same_group", "missing_group"],
)
def test_an_impossible_rejoin_is_refused(keep, drop):
    """Silently doing nothing would leave the group count wrong for the level."""
    registry = FaceRegistry()
    registry.identify(textured(96))

    with pytest.raises((ValueError, KeyError)):
        registry.merge(keep, drop)


def test_nothing_is_carried_between_levels():
    """A fresh registry per level, because next level's pictures are different."""
    first = FaceRegistry()
    first.identify(textured(30))

    second = FaceRegistry()

    assert second.group_count == 0


# --- animation gating -------------------------------------------------------


def test_a_card_is_stable_only_after_two_matching_captures():
    patch = textured(40)

    # Nothing to compare against yet: wait one more capture rather than assume.
    assert looks_stable(None, patch) is False
    assert looks_stable(patch, noisy_copy(patch, amplitude=1)) is True


def test_a_changing_card_is_not_stable():
    """A card part-way through its flip does not correlate with itself."""
    assert looks_stable(textured(41), textured(42)) is False


def solid_hsv(hue: int, saturation: int, value: int, side: int = 60):
    import cv2

    hsv = np.zeros((side, side, 3), np.uint8)
    hsv[:, :] = (hue, saturation, value)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_the_match_flash_is_detected():
    """Yellow-green and bright: the celebration the recordings show."""
    frame = solid_hsv(35, 200, 240)

    assert glow_score(frame, (0, 0, 60, 60)) > DEFAULT_FINGERPRINT_PARAMS.glow_fraction
    assert is_settled(frame, (0, 0, 60, 60)) is False


def test_gold_artwork_is_not_mistaken_for_the_flash():
    """The regression the first live run exposed.

    An ordinary revealed card was reported as glowing for 4.3 seconds, and only
    stopped once the pair had resolved -- so it was the card's own art, not the
    flash. Because is_settled gates fingerprinting, such a card can never be
    sampled and the runner times out on it. Gold coins, XP stars and diamonds are
    all card faces, so this was the common case rather than an edge one.

    Measurement put the false positives at hue 22-26. The lower bound of the
    window now sits above them; this pins that down.
    """
    for hue in (18, 22, 26):
        frame = solid_hsv(hue, 200, 240)
        assert glow_score(frame, (0, 0, 60, 60)) == 0.0, f"hue {hue} leaked in"
        assert is_settled(frame, (0, 0, 60, 60)) is True


def test_ordinary_artwork_is_not_mistaken_for_the_flash():
    frame = textured(43, side=60)

    assert is_settled(frame, (0, 0, 60, 60)) is True


# --- cropping ---------------------------------------------------------------


def test_crops_are_normalised_to_one_size():
    """Different levels draw different card sizes; comparison needs one scale."""
    frame = textured(50, side=200)

    small = crop_face(frame, (10, 10, 73, 73))
    large = crop_face(frame, (10, 10, 132, 132))

    assert small is not None and large is not None
    assert small.shape == large.shape == (SIDE, SIDE, 3)


def test_the_card_border_is_excluded():
    """The frame differs between a freshly flipped card and a settled one.

    Comparing borders would measure the card's state as much as its picture, so a
    fixed inset is dropped. Here the border is one colour and the picture another;
    the crop must show the picture.
    """
    frame = np.zeros((120, 120, 3), np.uint8)
    frame[:, :] = (30, 30, 30)
    frame[30:90, 30:90] = textured(51, side=60)

    crop = crop_face(frame, (10, 10, 110, 110))

    assert crop is not None
    # A crop dominated by the flat border would fail the structure test.
    assert has_structure(crop) is True


@pytest.mark.parametrize(
    "box",
    [(0, 0, 2, 2), (500, 500, 560, 560)],
    ids=["too_small", "outside_frame"],
)
def test_unusable_boxes_yield_no_crop(box):
    assert crop_face(textured(52, side=100), box) is None


def test_no_crop_from_a_missing_frame():
    assert crop_face(None, (0, 0, 40, 40)) is None


# --- the calibration helpers ------------------------------------------------


def test_mutual_best_pairs_recovers_a_perfect_pairing():
    """The property the labelling-free calibration rests on.

    On a board where every visible card belongs to a matched pair, the right
    threshold is one under which mutual best matches pair everything up with
    nothing left over.
    """
    crops = []
    for seed in (60, 61, 62):
        patch = textured(seed)
        crops.append(patch)
        crops.append(noisy_copy(patch, seed=seed))

    scores = pairwise_scores(crops)
    pairs, leftovers = mutual_best_pairs(scores)

    assert leftovers == []
    assert sorted(sorted(pair) for pair in pairs) == [[0, 1], [2, 3], [4, 5]]


def test_an_unpaired_card_is_reported_as_left_over():
    crops = [textured(70), noisy_copy(textured(70), seed=70), textured(71)]

    _, leftovers = mutual_best_pairs(pairwise_scores(crops))

    assert leftovers == [2]


def test_pairwise_scores_are_symmetric_with_a_neutral_diagonal():
    crops = [textured(80), textured(81), textured(82)]

    scores = pairwise_scores(crops)

    assert np.allclose(scores, scores.T)
    assert all(scores[i][i] == UNCOMPARABLE for i in range(3))


def test_a_custom_threshold_is_honoured():
    """Calibration must be able to move the number without editing code.

    The same two patches are grouped together under the calibrated threshold and
    split apart under an unreasonably strict one, so what changed is the number
    and not the input.
    """
    patch = textured(90)
    noisier = noisy_copy(patch, amplitude=40)

    default = FaceRegistry()
    default.identify(patch)
    grouped = default.identify(noisier)

    strict = FaceRegistry(FingerprintParams(match_threshold=0.999))
    strict.identify(patch)
    split = strict.identify(noisier)

    assert grouped is not None and grouped.is_new is False
    assert split is not None and split.is_new is True
