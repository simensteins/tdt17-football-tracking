"""Measure a jersey reader on SoccerNet-GSR tracks with labelled numbers.

SoccerNet-GSR labels a jersey number for most player and goalkeeper tracks.
Its 'valid' split (games 2, 3 and 5) is not part of our validation or test
matches, so readers can be compared there without touching either. Frames
and crops are chosen as in scripts/read_jerseys.py, from the ground-truth
boxes, and each track's number is decided with link_tracks' rule.

    HF_HOME=/cluster/work/$USER/hf python scripts/bench_jersey_readers.py <path to valid.zip> \
        --reader qwen --out <readings.csv>

Prints, per rule and minimum answer probability: tracks given a number, how many of those are right, and
how many labelled tracks get their right number (coverage).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football.data import CLASSES, SPLITS  # noqa: E402
from football.jerseys import READERS, make_reader, parse_number, pick_frames, trusted_number, upper_body  # noqa: E402

ROLES = {1: CLASSES.index("player"), 2: CLASSES.index("goalkeeper")}  # GSR category -> our class
EVALUATION_GAMES = set(SPLITS["val"]) | set(SPLITS["test"])


def clip_tracks(data: dict) -> pd.DataFrame:
    """Labelled player and goalkeeper boxes of one clip, with the track's labelled number."""
    frames = {im["image_id"]: (int(Path(im["file_name"]).stem), im["file_name"])
              for im in data["images"] if im.get("is_labeled")}
    rows = []
    for a in data["annotations"]:
        if a.get("category_id") in ROLES and a["image_id"] in frames and "bbox_image" in a:
            b = a["bbox_image"]
            frame, file = frames[a["image_id"]]
            rows.append((frame, file, a["track_id"], b["x"], b["y"], b["w"], b["h"], ROLES[a["category_id"]],
                         (a.get("attributes") or {}).get("jersey")))
    boxes = pd.DataFrame(rows, columns=["frame", "file", "id", "x", "y", "w", "h", "cls", "jersey"])
    labelled = boxes.dropna(subset=["jersey"]).groupby("id")["jersey"].agg(lambda j: j.mode().iloc[0])
    boxes["jersey"] = boxes["id"].map(labelled)
    return boxes


def score(readings: pd.DataFrame, single_digits: bool, min_conf: float = 0.0, min_votes: int = 2,
          min_share: float = 0.6) -> dict:
    readings = readings.assign(number=readings["number"].where(~(readings["conf"] < min_conf), ""))
    tracks = readings.groupby(["clip", "track"]).agg(role=("cls", "first"), truth=("jersey", "first"))
    by_track = dict(tuple(readings.groupby(["clip", "track"])["number"]))
    tracks["number"] = [trusted_number(by_track[k].astype("string"), role, min_votes, min_share, single_digits)
                        for k, role in zip(tracks.index, tracks["role"])]
    given = tracks["number"].notna()
    right = given & (tracks["number"] == tracks["truth"])
    labelled = tracks["truth"].notna()
    return {"tracks": len(tracks), "labelled": int(labelled.sum()), "given": int(given.sum()),
            "right": int(right.sum()), "precision": right.sum() / max(given.sum(), 1),
            "coverage": (right & labelled).sum() / max(labelled.sum(), 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("zip", type=Path, help="SoccerNet-GSR valid.zip")
    parser.add_argument("--reader", choices=READERS, required=True)
    parser.add_argument("--model", help="model ID (default: the reader's own)")
    parser.add_argument("--clips", type=int, help="only the first n clips")
    parser.add_argument("--per-track", type=int, default=10)
    parser.add_argument("--min-height", type=float, default=60)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True, help="CSV with every reading")
    args = parser.parse_args()

    import cv2

    read = make_reader(args.reader, args.model)
    archive = zipfile.ZipFile(args.zip)
    label_files = sorted(n for n in archive.namelist() if n.endswith("Labels-GameState.json"))[:args.clips]
    rows, start = [], time.time()
    for label_file in label_files:
        data = json.loads(archive.read(label_file))
        clip, info = label_file.split("/")[0], data["info"]
        if int(info["game_id"]) in EVALUATION_GAMES:
            raise SystemExit(f"{clip} is from one of our validation or test matches")
        boxes = clip_tracks(data)
        chosen = pd.concat([pick_frames(t, args.per_track, args.min_height) for _, t in boxes.groupby("id")])
        crops, meta = [], []
        for file, group in chosen.groupby("file"):
            raw = np.frombuffer(archive.read(f"{clip}/{info['im_dir']}/{file}"), np.uint8)
            img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            for b in group.itertuples():
                crop = upper_body(img, b.x, b.y, b.w, b.h)
                if min(crop.shape[:2]) >= 8:
                    crops.append(crop)
                    meta.append((clip, b.id, b.frame, b.h, b.cls, b.jersey))
        texts = []
        for i in range(0, len(crops), args.batch):
            texts += read(crops[i:i + args.batch])
        rows += [(*m, parse_number(t), t, p) for m, (t, p) in zip(meta, texts)]
        print(f"{clip}: {boxes['id'].nunique()} tracks, {len(crops)} crops", flush=True)

    readings = pd.DataFrame(rows, columns=["clip", "track", "frame", "h", "cls", "jersey", "number", "text", "conf"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    readings.to_csv(args.out, index=False)
    elapsed = time.time() - start
    print(f"{len(readings)} crops in {elapsed:.0f} s ({len(readings) / elapsed:.1f} crops/s)")

    labelled = readings.dropna(subset=["jersey"])
    print(f"crops with any number: {(readings['number'] != '').mean():.1%}; "
          f"crops reading exactly the labelled number: {(labelled['number'] == labelled['jersey']).mean():.1%}; "
          f"'1' read on tracks not numbered 1: {((labelled['number'] == '1') & (labelled['jersey'] != '1')).mean():.1%}")
    thresholds = (0.0,) if readings["conf"].isna().all() else (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98)
    for single in (False, True):
        for min_conf in thresholds:  # readings below min_conf count as no number
            s = score(readings, single, min_conf)
            print(f"{'one- and two-digit' if single else 'two-digit only':>18}, probability >= {min_conf:.2f}: "
                  f"{s['given']} of {s['tracks']} tracks given a number, {s['right']} right "
                  f"(precision {s['precision']:.1%}); coverage {s['coverage']:.1%} of {s['labelled']} labelled tracks")


if __name__ == "__main__":
    main()
