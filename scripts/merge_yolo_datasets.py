"""Combine YOLO-format image folders into one dataset, with symlinks.

RF-DETR reads one image folder per split from data.yaml, so training on the
course frames and the SoccerNet-GSR frames together needs them in one folder:

    python scripts/merge_yolo_datasets.py --out /cluster/work/$USER/yolo/rfdetr-gsr \
        --train /cluster/work/$USER/yolo/stride5/images/train /cluster/work/$USER/yolo/gsr-stride5/images/train \
        --val /cluster/work/$USER/yolo/stride5/images/val

Each image folder's labels are read from the matching 'labels' folder. File
names must not clash between the inputs (the converters prefix them by clip).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football.data import CLASSES  # noqa: E402


def link_split(sources: list[Path], out: Path, split: str, stride: int) -> int:
    images, labels = out / "images" / split, out / "labels" / split
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    count = 0
    for source in sources:
        label_dir = Path(str(source).replace("/images/", "/labels/"))
        for image in sorted(source.iterdir())[::stride]:
            if (images / image.name).exists():
                raise SystemExit(f"{image.name} appears in more than one input")
            (images / image.name).symlink_to(os.path.realpath(image))
            label = label_dir / f"{image.stem}.txt"
            if label.exists():
                (labels / label.name).symlink_to(label)
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, nargs="+", required=True, help="image folders for training")
    parser.add_argument("--val", type=Path, nargs="+", required=True, help="image folders for validation")
    parser.add_argument("--val-stride", type=int, default=1, help="keep every n-th validation image")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.out.exists():
        raise SystemExit(f"{args.out} exists")
    n_train = link_split(args.train, args.out, "train", 1)
    n_val = link_split(args.val, args.out, "val", args.val_stride)
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    (args.out / "data.yaml").write_text(f"path: {args.out}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    print(f"{n_train} training and {n_val} validation images in {args.out}")


if __name__ == "__main__":
    main()
