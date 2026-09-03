"""Extract authentic character templates from native game font atlas.

Extracts all 52 alphabet character templates (upper_A..upper_Z, lower_a..lower_z)
directly from the 1x standard font atlas (assets/fonts/font_atlas_standard.png)
which matches the pixel proportions of the in-game UI at 1024x768 reference resolution.

Generates 24x24 standardized grayscale templates in assets/letters/ and letters.json metadata.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATLAS_PATH = os.path.join(PROJECT_ROOT, "assets", "fonts", "font_atlas_standard.png")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "assets", "letters")


def extract_letters(
    atlas_path: str = ATLAS_PATH,
    output_dir: str = OUTPUT_DIR,
) -> Dict[str, Dict]:
    """Extract all 52 authentic letters and write templates and metadata."""
    if not os.path.isfile(atlas_path):
        raise FileNotFoundError(f"Font atlas not found: {atlas_path}")

    os.makedirs(output_dir, exist_ok=True)

    img_std = cv2.imread(atlas_path, cv2.IMREAD_UNCHANGED)
    if img_std is None or img_std.shape[2] != 4:
        raise ValueError(f"Invalid RGBA font atlas image: {atlas_path}")

    white_std = (
        (img_std[:, :, 0] > 180)
        & (img_std[:, :, 1] > 180)
        & (img_std[:, :, 2] > 180)
        & (img_std[:, :, 3] > 100)
    ).astype(np.uint8) * 255

    metadata: Dict[str, Dict] = {}

    def save_template(key: str, char: str, is_upper: bool, row_img: np.ndarray, box: Tuple[int, int, int, int]) -> None:
        x, y, w, h = box
        core = row_img[y : y + h, x : x + w]
        target_dim = 18
        scale = target_dim / max(h, w)
        nh = max(1, int(round(h * scale)))
        nw = max(1, int(round(w * scale)))
        resized = cv2.resize(core, (nw, nh), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((24, 24), dtype=np.uint8)
        yo = (24 - nh) // 2
        xo = (24 - nw) // 2
        canvas[yo : yo + nh, xo : xo + nw] = (resized > 100).astype(np.uint8) * 255

        out_png = os.path.join(output_dir, f"{key}.png")
        cv2.imwrite(out_png, canvas)

        ar = round(float(w / h), 3) if h > 0 else 1.0
        metadata[key] = {
            "char": char,
            "is_upper": is_upper,
            "orig_w": int(w),
            "orig_h": int(h),
            "ar": ar,
        }

    # Row 0: A..K (y=7..29)
    r0 = white_std[7:29, :]
    c0, _ = cv2.findContours(r0, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    b0 = [cv2.boundingRect(c) for c in c0 if cv2.boundingRect(c)[2] >= 2 and cv2.boundingRect(c)[3] >= 3]
    b0.sort(key=lambda b: b[0])
    for i in range(11):
        ch = chr(ord("A") + i)
        save_template(f"upper_{ch}", ch, True, r0, b0[i])

    # Row 1: L..U (y=42..68)
    r1 = white_std[42:68, :]
    c1, _ = cv2.findContours(r1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    b1 = [cv2.boundingRect(c) for c in c1 if cv2.boundingRect(c)[2] >= 2 and cv2.boundingRect(c)[3] >= 3]
    b1.sort(key=lambda b: b[0])
    for i in range(10):
        ch = chr(ord("L") + i)
        save_template(f"upper_{ch}", ch, True, r1, b1[i])

    # Row 2: V..Z, a..f (y=77..99)
    r2 = white_std[77:99, :]
    c2, _ = cv2.findContours(r2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    b2 = [cv2.boundingRect(c) for c in c2 if cv2.boundingRect(c)[2] >= 2 and cv2.boundingRect(c)[3] >= 3]
    b2.sort(key=lambda b: b[0])
    for i in range(5):
        ch = chr(ord("V") + i)
        save_template(f"upper_{ch}", ch, True, r2, b2[i])
    for i in range(6):
        ch = chr(ord("a") + i)
        save_template(f"lower_{ch}", ch, False, r2, b2[5 + i])

    # Row 3: g..s (y=113..139)
    r3 = white_std[113:139, :]
    c3, _ = cv2.findContours(r3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw_b3 = [cv2.boundingRect(c) for c in c3 if cv2.boundingRect(c)[2] >= 2 and cv2.boundingRect(c)[3] >= 3]
    raw_b3.sort(key=lambda b: b[0])
    b3: List[Tuple[int, int, int, int]] = []
    # g (merge raw_b3[0] and raw_b3[1])
    gx1 = min(raw_b3[0][0], raw_b3[1][0])
    gy1 = min(raw_b3[0][1], raw_b3[1][1])
    gx2 = max(raw_b3[0][0] + raw_b3[0][2], raw_b3[1][0] + raw_b3[1][2])
    gy2 = max(raw_b3[0][1] + raw_b3[0][3], raw_b3[1][1] + raw_b3[1][3])
    b3.append((gx1, gy1, gx2 - gx1, gy2 - gy1))
    # h
    b3.append(raw_b3[2])
    # i (merge raw_b3[3] and raw_b3[4])
    ix1 = min(raw_b3[3][0], raw_b3[4][0])
    iy1 = min(raw_b3[3][1], raw_b3[4][1])
    ix2 = max(raw_b3[3][0] + raw_b3[3][2], raw_b3[4][0] + raw_b3[4][2])
    iy2 = max(raw_b3[3][1] + raw_b3[3][3], raw_b3[4][1] + raw_b3[4][3])
    b3.append((ix1, iy1, ix2 - ix1, iy2 - iy1))
    # j (merge raw_b3[5] and raw_b3[6])
    jx1 = min(raw_b3[5][0], raw_b3[6][0])
    jy1 = min(raw_b3[5][1], raw_b3[6][1])
    jx2 = max(raw_b3[5][0] + raw_b3[5][2], raw_b3[6][0] + raw_b3[6][2])
    jy2 = max(raw_b3[5][1] + raw_b3[5][3], raw_b3[6][1] + raw_b3[6][3])
    b3.append((jx1, jy1, jx2 - jx1, jy2 - jy1))
    for idx in range(7, 16):
        b3.append(raw_b3[idx])

    for i in range(13):
        ch = chr(ord("g") + i)
        save_template(f"lower_{ch}", ch, False, r3, b3[i])

    # Row 4: t..z (y=147..173)
    r4 = white_std[147:173, :]
    c4, _ = cv2.findContours(r4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    b4 = [cv2.boundingRect(c) for c in c4 if cv2.boundingRect(c)[2] >= 2 and cv2.boundingRect(c)[3] >= 3]
    b4.sort(key=lambda b: b[0])
    for i in range(7):
        ch = chr(ord("t") + i)
        save_template(f"lower_{ch}", ch, False, r4, b4[i])

    meta_file = os.path.join(output_dir, "letters.json")
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


if __name__ == "__main__":
    meta = extract_letters()
    print(f"Successfully extracted {len(meta)} authentic letter templates to {OUTPUT_DIR}")
