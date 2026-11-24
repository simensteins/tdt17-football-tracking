"""Convert SoccerNet-Tracking sequences to an Ultralytics YOLO dataset.

Images are symlinked, not copied. Frames are named <sequence>_<frame>.jpg so
results can be traced back to a sequence and its clip type. Boxes of the
`other` role are dropped and boxes are clipped to the image.

Label sets:
    four  player, goalkeeper, referee, ball (classes 0-3)
    coco  people as COCO 'person' (0) and the ball as 'sports ball' (32), for
          evaluating a COCO-pretrained model without fine-tuning

    python scripts/make_yolo_dataset.py data/tracking-2023 --out /cluster/work/$USER/yolo/stride5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football.data import CLASSES, SPLITS, find_sequences  # noqa: E402

COCO_IDS = {"player": 0, "goalkeeper": 0, "referee": 0, "ball": 32}


def coco_names() -> list[str]:
    import ultralytics

    path = Path(ultralytics.__file__).parent / "cfg" / "datasets" / "coco.yaml"
    names = yaml.safe_load(path.read_text())["names"]
    return [names[i] for i in sorted(names)]


def yolo_lines(rows, width: int, height: int, class_ids: dict[str, int]) -> list[str]:
    lines = []
    for r in rows.itertuples():
        if r.role not in class_ids:
            continue
        x1, y1 = max(r.x, 0), max(r.y, 0)
        x2, y2 = min(r.x + r.w, width), min(r.y + r.h, height)
        if x2 <= x1 or y2 <= y1:
            continue
        cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
        w, h = (x2 - x1) / width, (y2 - y1) / height
        lines.append(f"{class_ids[r.role]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="tracking-2023 folder with train/ and test/")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=5, help="keep every n-th frame of training sequences")
    parser.add_argument("--eval-stride", type=int, default=5, help="keep every n-th frame of val/test sequences")
    parser.add_argument("--labels", choices=("four", "coco"), default="four")
    args = parser.parse_args()

    if args.labels == "four":
        class_ids = {c: i for i, c in enumerate(CLASSES)}
        names = list(CLASSES)
    else:
        class_ids = COCO_IDS
        names = coco_names()

    split_of_game = {g: split for split, games in SPLITS.items() for g in games}
    counts = {split: [0, 0] for split in SPLITS}  # images, boxes
    for seq in find_sequences(args.root):
        split = split_of_game.get(seq.game_id)
        if split is None:
            print(f"skipping {seq.name}: game {seq.game_id} is in no split")
            continue
        stride = args.stride if split == "train" else args.eval_stride
        image_dir = args.out / "images" / split
        label_dir = args.out / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        width, height = seq.size
        boxes = seq.boxes()
        by_frame = dict(tuple(boxes.groupby("frame")))
        for frame in range(1, seq.length + 1, stride):
            stem = f"{seq.name}_{frame:06d}"
            source = seq.image(frame).resolve()
            link = image_dir / f"{stem}.jpg"
            if not link.exists():
                os.symlink(source, link)
            rows = by_frame.get(frame, boxes.iloc[:0])
            lines = yolo_lines(rows, width, height, class_ids)
            (label_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            counts[split][0] += 1
            counts[split][1] += len(lines)

    config = {
        "path": str(args.out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(names)),
    }
    (args.out / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    for split, (images, boxes) in counts.items():
        print(f"{split}: {images} images, {boxes} boxes")
    print(f"wrote {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
