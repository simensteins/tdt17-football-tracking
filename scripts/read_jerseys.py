"""Read jersey numbers for every person track, with Florence-2 OCR.

For each track, up to --per-track frames spread over its lifetime are chosen
(larger boxes first), the upper body is cropped (15-60 % of the box height,
where back numbers sit), and Florence-2 reads the text in the crop. Only
one- or two-digit numbers are kept. The mean shirt colour of each crop is
stored too, so tracks can be split into teams later (both teams have a 5).

Runs in the RF-DETR environment (transformers >= 5), on a GPU if available:

    HF_HOME=/cluster/work/$USER/hf python scripts/read_jerseys.py data/tracking-2023 \
        <tracks dir> --split val --out <tracks dir>/jerseys.csv

Output: one row per crop: seq, track, frame, h, number ('' if none), text, shirt colour (L, a, b).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football.data import SPLITS, find_sequences  # noqa: E402
from football.tracking import BALL, read_mot  # noqa: E402

NUMBER = re.compile(r"\d+")


def pick_frames(track: pd.DataFrame, n: int, min_height: float) -> pd.DataFrame:
    """Up to n boxes spread over the track, preferring the larger ones."""
    tall = track[track["h"] >= min_height]
    track = tall if len(tall) else track
    if len(track) <= n:
        return track
    track = track.sort_values("frame")
    stretches = np.array_split(np.arange(len(track)), n)  # one box per stretch of the track
    return pd.concat([track.iloc[s].nlargest(1, "h") for s in stretches])


def upper_body(img: np.ndarray, x: float, y: float, w: float, h: float) -> np.ndarray:
    """The shirt below the head (15-60 % of the box height), where back numbers sit."""
    x0, x1 = int(max(x, 0)), int(min(x + w, img.shape[1]))
    y0, y1 = int(max(y + 0.15 * h, 0)), int(min(y + 0.6 * h, img.shape[0]))
    return img[y0:y1, x0:x1]


def shirt_colour(img: np.ndarray, x: float, y: float, w: float, h: float) -> np.ndarray:
    """Mean CIELAB colour of the shirt's centre (middle half of the width, 20-50 % of the height)."""
    import cv2

    patch = img[int(y + 0.2 * h):int(y + 0.5 * h), int(x + 0.25 * w):int(x + 0.75 * w)]
    if patch.size == 0:
        return np.full(3, np.nan)
    return cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="tracking-2023 folder")
    parser.add_argument("tracks", type=Path, help="folder with <sequence>.txt track files")
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--sequences", nargs="*", help="only these sequences (default: whole split)")
    parser.add_argument("--per-track", type=int, default=12)
    parser.add_argument("--min-height", type=float, default=60, help="prefer boxes at least this tall (px)")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--model", default="florence-community/Florence-2-large")
    parser.add_argument("--save-crops", type=Path, help="also write each crop as an image, named by its reading")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import cv2
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Florence2ForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(args.model)
    model = Florence2ForConditionalGeneration.from_pretrained(args.model, dtype=dtype).to(device).eval()

    def read(crops: list[np.ndarray]) -> list[str]:
        images = [Image.fromarray(c[:, :, ::-1]) for c in crops]
        inputs = processor(text=["<OCR>"] * len(images), images=images, return_tensors="pt").to(device, dtype)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=12, num_beams=1, do_sample=False)
        return [t.strip() for t in processor.batch_decode(out, skip_special_tokens=True)]

    sequences = [s for s in find_sequences(args.root) if s.game_id in SPLITS[args.split]
                 and (not args.sequences or s.name in args.sequences)]
    if args.save_crops:
        args.save_crops.mkdir(parents=True, exist_ok=True)
    rows, start, n_crops = [], time.time(), 0
    for seq in sequences:
        tracks = read_mot(args.tracks / f"{seq.name}.txt")
        tracks = tracks[tracks["cls"] != BALL]
        chosen = pd.concat([pick_frames(t, args.per_track, args.min_height) for _, t in tracks.groupby("id")])
        chosen = chosen.sort_values("frame")
        crops, meta = [], []
        for frame, boxes in chosen.groupby("frame"):
            img = cv2.imread(str(seq.image(int(frame))))
            for b in boxes.itertuples():
                crop = upper_body(img, b.x, b.y, b.w, b.h)
                if min(crop.shape[:2]) < 8:
                    continue
                crops.append(crop)
                meta.append((seq.name, int(b.id), int(frame), float(b.h), *shirt_colour(img, b.x, b.y, b.w, b.h)))
        texts = []
        for i in range(0, len(crops), args.batch):
            texts += read(crops[i:i + args.batch])
        for (name, track, frame, h, L, a, bb), text, crop in zip(meta, texts, crops):
            numbers = [n for n in NUMBER.findall(text) if len(n) <= 2]
            number = numbers[0] if numbers else ""
            rows.append((name, track, frame, h, number, text, L, a, bb))
            if args.save_crops:
                cv2.imwrite(str(args.save_crops / f"{name}_{track}_{frame}_{number or 'x'}.jpg"), crop)
        n_crops += len(crops)
        print(f"{seq.name}: {tracks['id'].nunique()} tracks, {len(crops)} crops, "
              f"{sum(1 for r in rows if r[0] == seq.name and r[4])} with a number")

    df = pd.DataFrame(rows, columns=["seq", "track", "frame", "h", "number", "text", "L", "a", "b"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    elapsed = time.time() - start
    print(f"{n_crops} crops in {elapsed:.0f} s ({n_crops / elapsed:.1f} crops/s) on {device}; "
          f"{(df['number'] != '').mean():.1%} with a number")


if __name__ == "__main__":
    main()
