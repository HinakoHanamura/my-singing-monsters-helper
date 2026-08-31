"""Measure detection thresholds from real captured frames.

Why this exists
---------------
Every perception parameter in config.py starts life as a guess. A guess that
looks plausible can fail silently: set `texture_min_std` too low and the rule
passes everything while the log still looks healthy. This tool replaces guesses
with measurements taken on real frames.

It deliberately drives the **production code path** - the same `VisionAgent`,
the same scale steps, the same duplicate suppression, the same rule objects the
bot uses at runtime. A separate reimplementation would measure something the bot
never does.

What it reports
---------------
1. Score distribution of detections, so a separating threshold can be chosen
   from the histogram rather than invented.
2. A threshold sweep: detections per frame at each candidate threshold. Real
   targets form a plateau; background noise grows without bound as the
   threshold drops.
3. Nearest-neighbour spacing between detections, which is what
   `nms_distance` must stay below or stacked plaques get merged into one.
4. The `TextureBelow` metric per detection, to check whether that rule can
   separate "target above a monster" from "target over empty ground" at all.
5. Annotated frames, so the numbers can be confirmed by eye. This step is not
   optional: a histogram cannot tell you whether a peak is a coin or a
   look-alike decoration.

Usage
-----
    python tools/calibrate.py
    python tools/calibrate.py --frames captures/calib_mixed --target coin
    python tools/calibrate.py --threshold 0.55 --annotate 8
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import DEFAULT_CONFIG  # noqa: E402
from core.geometry import scale_factor  # noqa: E402
from core.validators import TextureBelow, ValidationContext, build_rule_chain  # noqa: E402
from core.vision_agent import Detection, VisionAgent  # noqa: E402

SWEEP = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


def resolve(path: str) -> str:
    """Resolve against the project root, never the shell's cwd."""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(PROJECT_ROOT, path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Derive detection thresholds from real frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--frames", default="captures/calib_1024",
                        help="directory of captured PNG frames")
    parser.add_argument("--target", default="coin", help="target name to measure")
    parser.add_argument("--templates", default=None,
                        help="template directory (defaults to the configured one)")
    parser.add_argument("--threshold", type=float, default=0.50,
                        help="permissive threshold used while collecting samples")
    parser.add_argument("--limit", type=int, default=40,
                        help="maximum number of frames to analyse (0 = all)")
    parser.add_argument("--annotate", type=int, default=6,
                        help="how many annotated frames to write for visual review")
    parser.add_argument("--out", default="reports/calibration",
                        help="directory for annotated frames")
    parser.add_argument("--max-detections", type=int, default=60,
                        help="per-frame detection cap while sampling")
    parser.add_argument(
        "--nms",
        type=int,
        default=None,
        help=(
            "override nms_distance while sampling. Measuring true target "
            "spacing REQUIRES a small value here: duplicate suppression removes "
            "any pair closer than its own gap, so measuring with the configured "
            "gap can never report a spacing below it, and the result is "
            "circular. Pass something like 6 to see the real distribution"
        ),
    )
    return parser


def load_frames(directory: str, limit: int) -> List[Tuple[str, np.ndarray]]:
    if not os.path.isdir(directory):
        raise SystemExit("frame directory not found: %s" % directory)

    names = sorted(n for n in os.listdir(directory) if n.lower().endswith(".png"))
    if not names:
        raise SystemExit("no PNG frames in %s" % directory)

    if limit and len(names) > limit:
        # Even spacing beats taking the first N: a recording changes over time.
        step = len(names) / float(limit)
        names = [names[int(i * step)] for i in range(limit)]

    frames: List[Tuple[str, np.ndarray]] = []
    for name in names:
        raw = np.fromfile(os.path.join(directory, name), dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if image is None:
            print("  skipped unreadable frame: %s" % name)
            continue
        frames.append((name, image))
    return frames


def percentiles(values: Sequence[float], points=(0, 5, 25, 50, 75, 95, 100)) -> str:
    if not values:
        return "n/a"
    ordered = sorted(values)
    parts = []
    for point in points:
        index = min(len(ordered) - 1, int(round(point / 100.0 * (len(ordered) - 1))))
        parts.append("p%d=%.3f" % (point, ordered[index]))
    return "  ".join(parts)


def histogram(values: Sequence[float], low=0.4, high=1.0, bins=12) -> None:
    if not values:
        print("    (no samples)")
        return
    width = (high - low) / bins
    counts = [0] * bins
    for value in values:
        index = int((value - low) / width)
        index = max(0, min(bins - 1, index))
        counts[index] += 1
    peak = max(counts) or 1
    for index, count in enumerate(counts):
        start = low + index * width
        bar = "#" * int(round(40.0 * count / peak))
        print("    %.2f-%.2f | %-40s %d" % (start, start + width, bar, count))


def nearest_neighbour_gaps(detections: Sequence[Detection]) -> List[float]:
    gaps: List[float] = []
    for i, first in enumerate(detections):
        best: Optional[float] = None
        fx, fy = first.center
        for j, second in enumerate(detections):
            if i == j:
                continue
            sx, sy = second.center
            distance = ((fx - sx) ** 2 + (fy - sy) ** 2) ** 0.5
            if best is None or distance < best:
                best = distance
        if best is not None:
            gaps.append(best)
    return gaps


def annotate(image: np.ndarray, detections: Sequence[Detection], verdicts: Dict[int, str]) -> np.ndarray:
    canvas = image.copy()
    for index, detection in enumerate(detections):
        code = verdicts.get(index, "ACCEPTED")
        accepted = code == "ACCEPTED"
        colour = (0, 220, 0) if accepted else (0, 120, 255)
        x1, y1, x2, y2 = detection.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
        label = "%.2f" % detection.confidence
        if not accepted:
            label += " " + code
        cv2.putText(canvas, label, (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)
    return canvas


def main() -> int:
    args = build_parser().parse_args()

    frames_dir = resolve(args.frames)
    template_dir = resolve(args.templates or DEFAULT_CONFIG.vision.template_dir)
    out_dir = resolve(args.out)

    frames = load_frames(frames_dir, args.limit)
    height, width = frames[0][1].shape[:2]
    sizes = {(f.shape[1], f.shape[0]) for _, f in frames}
    if len(sizes) > 1:
        print("WARNING: mixed frame sizes %s - calibrate one size at a time" % sizes)

    # Frames used for calibration define the reference resolution by construction:
    # the templates were cropped from frames of exactly this size.
    reference_size = (width, height)

    print("frames        : %d from %s" % (len(frames), frames_dir))
    print("frame size    : %dx%d" % (width, height))
    print("templates     : %s" % template_dir)
    print("target        : %s" % args.target)
    print("sample thresh : %.2f (permissive on purpose)" % args.threshold)
    print("")

    nms_distance = (
        args.nms if args.nms is not None else DEFAULT_CONFIG.vision.nms_distance
    )
    print("nms distance  : %d%s" % (
        nms_distance,
        " (overridden for spacing measurement)" if args.nms is not None else "",
    ))

    agent = VisionAgent(
        template_dir=template_dir,
        match_threshold=args.threshold,
        reference_size=reference_size,
        scale_steps=DEFAULT_CONFIG.vision.scale_steps,
        nms_distance=nms_distance,
        max_detections=args.max_detections,
        use_fake_detection=False,
    )

    texture_rule = TextureBelow(
        DEFAULT_CONFIG.safety.texture_probe_height,
        DEFAULT_CONFIG.safety.texture_min_std,
    )
    chain = build_rule_chain(DEFAULT_CONFIG.safety, args.threshold)
    scale = scale_factor(reference_size, reference_size)

    all_scores: List[float] = []
    all_gaps: List[float] = []
    texture_values: List[float] = []
    per_frame_counts: List[int] = []
    sweep_counts = {t: 0 for t in SWEEP}
    box_sizes: List[Tuple[int, int]] = []
    rejection_tally: Dict[str, int] = {}

    os.makedirs(out_dir, exist_ok=True)
    annotate_every = max(1, len(frames) // args.annotate) if args.annotate else 0
    written = 0

    for index, (name, image) in enumerate(frames):
        detections = agent.detect(args.target, image)
        per_frame_counts.append(len(detections))

        for detection in detections:
            all_scores.append(detection.confidence)
            box_sizes.append((detection.width, detection.height))
            for threshold in SWEEP:
                if detection.confidence >= threshold:
                    sweep_counts[threshold] += 1
            value = texture_rule.measure(detection, image)
            if value >= 0:
                texture_values.append(value)

        all_gaps.extend(nearest_neighbour_gaps(detections))

        ctx = ValidationContext(
            frame=image,
            client_size=reference_size,
            scale=scale,
            detections={args.target: list(detections)},
        )
        verdicts: Dict[int, str] = {}
        for slot, detection in enumerate(detections):
            verdict = chain.evaluate(detection, ctx)
            code = "ACCEPTED" if verdict.ok else verdict.code
            verdicts[slot] = code
            if not verdict.ok:
                rejection_tally[code] = rejection_tally.get(code, 0) + 1

        if annotate_every and index % annotate_every == 0 and written < args.annotate:
            canvas = annotate(image, detections, verdicts)
            target_path = os.path.join(out_dir, "annotated_%02d_%s" % (written, name))
            ok, buffer = cv2.imencode(".png", canvas)
            if ok:
                with open(target_path, "wb") as handle:
                    handle.write(buffer.tobytes())
                written += 1

    total = len(all_scores)
    print("=" * 72)
    print("1. DETECTION VOLUME")
    print("   total detections      : %d across %d frames" % (total, len(frames)))
    if per_frame_counts:
        print("   per frame             : min=%d median=%.1f max=%d"
              % (min(per_frame_counts),
                 statistics.median(per_frame_counts),
                 max(per_frame_counts)))
    if not total:
        print("\nNo detections at all. Either the template does not match these "
              "frames, or the template directory is wrong.")
        return 1

    print("")
    print("2. MATCH SCORE DISTRIBUTION")
    print("   %s" % percentiles(all_scores))
    histogram(all_scores, low=min(0.4, args.threshold), high=1.0)
    print("   Look for a gap between a high cluster (real targets) and a low")
    print("   tail (background). Put match_threshold inside that gap.")

    print("")
    print("3. THRESHOLD SWEEP  (detections per frame)")
    for threshold in SWEEP:
        per_frame = sweep_counts[threshold] / float(len(frames))
        bar = "#" * int(round(min(40.0, per_frame * 4)))
        print("   %.2f | %-40s %.2f" % (threshold, bar, per_frame))
    print("   A plateau means real targets; a steep rise means noise creeping in.")

    print("")
    print("4. DETECTION BOX SIZE")
    widths = [w for w, _ in box_sizes]
    heights = [h for _, h in box_sizes]
    print("   width  : min=%d median=%d max=%d" % (min(widths), int(statistics.median(widths)), max(widths)))
    print("   height : min=%d median=%d max=%d" % (min(heights), int(statistics.median(heights)), max(heights)))

    print("")
    print("5. NEAREST-NEIGHBOUR SPACING  (drives nms_distance)")
    if all_gaps:
        ordered = sorted(all_gaps)
        print("   min=%.1f p5=%.1f median=%.1f max=%.1f px"
              % (ordered[0],
                 ordered[max(0, int(0.05 * (len(ordered) - 1)))],
                 statistics.median(ordered),
                 ordered[-1]))
        print("   nms_distance must stay clearly BELOW the small percentiles,")
        print("   otherwise adjacent stacked targets collapse into one.")
    else:
        print("   only one detection per frame; no spacing data")

    print("")
    print("6. TEXTURE-BELOW METRIC  (drives texture_min_std)")
    if texture_values:
        ordered = sorted(texture_values)
        print("   %s" % percentiles(texture_values))
        print("   min=%.1f median=%.1f max=%.1f"
              % (ordered[0], statistics.median(ordered), ordered[-1]))
        print("   Current threshold is %.1f." % DEFAULT_CONFIG.safety.texture_min_std)
        below = sum(1 for v in texture_values if v < DEFAULT_CONFIG.safety.texture_min_std)
        print("   %d of %d detections (%.0f%%) fall below it."
              % (below, len(texture_values), 100.0 * below / len(texture_values)))
        print("   If that share is ~0%%, the rule is currently rejecting nothing")
        print("   and provides no protection despite looking active in the log.")
    else:
        print("   no measurable probe regions")

    print("")
    print("7. RULE CHAIN OUTCOME  (at sample threshold, current config)")
    if rejection_tally:
        for code, count in sorted(rejection_tally.items(), key=lambda kv: -kv[1]):
            print("   %-22s %d" % (code, count))
    else:
        print("   nothing rejected")
    accepted = total - sum(rejection_tally.values())
    print("   ACCEPTED               %d" % accepted)

    print("")
    print("Annotated frames written: %d -> %s" % (written, out_dir))
    print("Review them before trusting any number above. Green = accepted,")
    print("orange = rejected with its reason code.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
