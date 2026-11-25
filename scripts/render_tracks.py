"""Draw tracks onto a sequence's frames and save them as an MP4 video.

    python scripts/render_tracks.py data/tracking-2023 SNMOT-097 outputs/tracks/val/<run> --out outputs/videos/SNMOT-097.mp4

Each track ID gets its own colour, labelled with its ID and role; the ball is
drawn as a ring. With --gt the ground truth is drawn instead, for comparison.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football.data import CLASSES, find_sequences  # noqa: E402
from football.tracking import BALL, read_mot  # noqa: E402

ROLE_LETTER = {c: c[0].upper() for c in CLASSES}


def colour(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id)
    return tuple(int(c) for c in rng.integers(64, 256, 3))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="tracking-2023 folder")
    parser.add_argument("sequence", help="e.g. SNMOT-097")
    parser.add_argument("tracks", type=Path, nargs="?", help="folder with <sequence>.txt track files")
    parser.add_argument("--gt", action="store_true", help="draw the ground truth instead of tracks")
    parser.add_argument("--scale", type=float, default=0.5, help="output size relative to 1920x1080")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import cv2

    seq = next(s for s in find_sequences(args.root) if s.name == args.sequence)
    if args.gt:
        boxes = seq.boxes()
        boxes = boxes[boxes["role"].isin(CLASSES)].rename(columns={"track": "id"})
        boxes["cls"] = boxes["role"].map(CLASSES.index)
    else:
        boxes = read_mot(args.tracks / f"{seq.name}.txt")
    by_frame = dict(tuple(boxes.groupby("frame")))

    width, height = seq.size
    size = (int(width * args.scale), int(height * args.scale))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), 25, size)
    for frame in range(1, seq.length + 1):
        img = cv2.imread(str(seq.image(frame)))
        rows = by_frame.get(frame)
        for row in ([] if rows is None else rows.itertuples()):
            x, y, w, h = int(row.x), int(row.y), int(row.w), int(row.h)
            c = colour(int(row.id))
            if int(row.cls) == BALL:
                cv2.circle(img, (x + w // 2, y + h // 2), max(w, h) + 6, (0, 255, 255), 3)
                continue
            cv2.rectangle(img, (x, y), (x + w, y + h), c, 3)
            label = f"{int(row.id)} {ROLE_LETTER[CLASSES[int(row.cls)]]}"
            cv2.putText(img, label, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2, cv2.LINE_AA)
        cv2.putText(img, f"{seq.name}  frame {frame}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                    cv2.LINE_AA)
        writer.write(cv2.resize(img, size, interpolation=cv2.INTER_AREA))
    writer.release()
    print(f"wrote {args.out} ({seq.length} frames)")


if __name__ == "__main__":
    main()
