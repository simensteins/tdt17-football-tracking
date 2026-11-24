"""Draw ground-truth boxes on frames of one sequence.

Each output image shows the full frame with boxes coloured by role, and a
zoomed crop around every ball box underneath.

    python scripts/show_frames.py data/tracking-2023/train/SNMOT-060 1 300 --out outputs/frames
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football.data import Sequence  # noqa: E402

COLOURS = {"player": (66, 135, 245), "goalkeeper": (245, 197, 66), "referee": (240, 60, 60),
           "ball": (60, 230, 90), "other": (200, 200, 200)}
CROP = 120      # half-size of the ball crop in source pixels
ZOOM = 2        # upscaling of the crop


def render(seq: Sequence, frame: int, boxes) -> Image.Image:
    image = Image.open(seq.image(frame)).convert("RGB")
    draw = ImageDraw.Draw(image)
    rows = boxes[boxes["frame"] == frame]
    for r in rows.itertuples():
        colour = COLOURS[r.role]
        draw.rectangle([r.x, r.y, r.x + r.w, r.y + r.h], outline=colour, width=2)
        draw.text((r.x, r.y - 12), f"{r.track}", fill=colour)
    draw.text((10, 10), f"{seq.name} frame {frame}", fill=(255, 255, 255))

    crops = []
    for r in rows[rows["role"] == "ball"].itertuples():
        cx, cy = r.x + r.w / 2, r.y + r.h / 2
        box = (int(cx - CROP), int(cy - CROP), int(cx + CROP), int(cy + CROP))
        crops.append(image.crop(box).resize((2 * CROP * ZOOM,) * 2, Image.NEAREST))
    if not crops:
        return image
    side = 2 * CROP * ZOOM
    sheet = Image.new("RGB", (image.width, image.height + side), (0, 0, 0))
    sheet.paste(image, (0, 0))
    for i, crop in enumerate(crops):
        sheet.paste(crop, (i * (side + 10), image.height))
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sequence", type=Path)
    parser.add_argument("frames", type=int, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    seq = Sequence(args.sequence)
    boxes = seq.boxes()
    args.out.mkdir(parents=True, exist_ok=True)
    for frame in args.frames:
        path = args.out / f"{seq.name}_{frame:06d}.jpg"
        render(seq, frame, boxes).save(path, quality=90)
        print(path)


if __name__ == "__main__":
    main()
