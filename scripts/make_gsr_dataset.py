"""Turn SoccerNet-GSR clips from other matches into extra YOLO training data.

SoccerNet-GSR (Somers et al., 2024; https://huggingface.co/datasets/SoccerNet/SN-GSR-2025,
GPL-3.0) re-releases the 200 SoccerNet-Tracking clips with labels. Its 'valid'
split holds 58 clips from three matches (game IDs 2, 3 and 5) that are not in
our course data. Our validation match (6) and test matches (7, 8, 11) are
refused, so nothing leaks into evaluation.

    python scripts/make_gsr_dataset.py <path to valid.zip> --out /cluster/work/$USER/yolo/gsr-stride5

Writes images/train and labels/train with every --stride-th frame, classes as
in football.data.CLASSES (player, goalkeeper, referee, ball); 'other' is dropped.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football.data import CLASSES, SPLITS  # noqa: E402

GSR_CLASSES = {1: "player", 2: "goalkeeper", 3: "referee", 4: "ball"}  # 7 = other, dropped
EVALUATION_GAMES = set(SPLITS["val"]) | set(SPLITS["test"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("zip", type=Path, help="a SoccerNet-GSR split zip, e.g. valid.zip")
    parser.add_argument("--stride", type=int, default=5, help="keep every n-th frame")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    images, labels = args.out / "images" / "train", args.out / "labels" / "train"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    archive = zipfile.ZipFile(args.zip)
    counts, games = Counter(), Counter()
    for label_file in sorted(n for n in archive.namelist() if n.endswith("Labels-GameState.json")):
        data = json.loads(archive.read(label_file))
        info, clip = data["info"], label_file.split("/")[0]
        game = int(info["game_id"])
        if game in EVALUATION_GAMES:
            raise SystemExit(f"{clip} is from game {game}, one of our validation or test matches")
        boxes = {}
        for a in data["annotations"]:
            if a.get("category_id") in GSR_CLASSES and "bbox_image" in a:
                boxes.setdefault(a["image_id"], []).append(a)
        for image in data["images"]:
            frame = int(Path(image["file_name"]).stem)
            if not image.get("is_labeled") or (frame - 1) % args.stride:
                continue
            width, height = image["width"], image["height"]
            lines = []
            for a in boxes.get(image["image_id"], []):
                b = a["bbox_image"]
                x0, y0 = max(b["x"], 0), max(b["y"], 0)
                x1, y1 = min(b["x"] + b["w"], width), min(b["y"] + b["h"], height)
                if x1 <= x0 or y1 <= y0:
                    continue
                role = GSR_CLASSES[a["category_id"]]
                counts[role] += 1
                lines.append(f"{CLASSES.index(role)} {(x0 + x1) / 2 / width:.6f} {(y0 + y1) / 2 / height:.6f} "
                             f"{(x1 - x0) / width:.6f} {(y1 - y0) / height:.6f}")
            name = f"{clip}_{frame:06d}"
            (images / f"{name}.jpg").write_bytes(archive.read(f"{clip}/{info['im_dir']}/{image['file_name']}"))
            (labels / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            counts["images"] += 1
        games[game] += 1
    print(f"clips per game: {dict(games)}; {dict(counts)}")


if __name__ == "__main__":
    main()
