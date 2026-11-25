"""Run a YOLO detector on every frame of a split and store the detections.

Low-confidence boxes are kept (default --conf 0.01): ByteTrack-style trackers
use them in their second association step, and the tracker decides what to
discard. Output per sequence: frame, -1, x, y, w, h, confidence, class.

    python scripts/detect.py data/tracking-2023 --split val \
        --model /cluster/work/$USER/runs/d2-m-1280/weights/best.pt --imgsz 1280 \
        --out /cluster/work/$USER/detections/val/d2-m-1280

With --coco, a COCO-pretrained model's 'person' becomes player and
'sports ball' becomes ball; its other classes are dropped.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football.data import CLASSES, SPLITS, find_sequences  # noqa: E402
from football.tracking import write_mot  # noqa: E402

COCO_TO_OURS = {0: CLASSES.index("player"), 32: CLASSES.index("ball")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="tracking-2023 folder")
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8, help="threads decoding images")
    parser.add_argument("--coco", action="store_true", help="map COCO classes to ours")
    parser.add_argument("--sequences", nargs="*", help="only these sequences (default: whole split)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    model = YOLO(args.model)
    sequences = [s for s in find_sequences(args.root) if s.game_id in SPLITS[args.split]
                 and (not args.sequences or s.name in args.sequences)]
    args.out.mkdir(parents=True, exist_ok=True)

    frames = 0
    inference_ms = 0.0
    start = time.time()
    # Decoding full-HD JPEGs one at a time left the GPU idle most of the time,
    # so a thread pool decodes the next batch while the model runs on this one.
    pool, prefetch = ThreadPoolExecutor(args.workers), ThreadPoolExecutor(1)
    for seq in sequences:
        rows = []
        batches = [range(first, min(first + args.batch, seq.length + 1))
                   for first in range(1, seq.length + 1, args.batch)]
        decode = lambda frames_in_batch: list(pool.map(lambda f: cv2.imread(str(seq.image(f))), frames_in_batch))
        upcoming = prefetch.submit(decode, batches[0])
        for i, frames_in_batch in enumerate(batches):
            images = upcoming.result()
            if i + 1 < len(batches):
                upcoming = prefetch.submit(decode, batches[i + 1])
            # Ultralytics runs a list as a single batch, so pass one batch at a time.
            results = model.predict(images, imgsz=args.imgsz, conf=args.conf, verbose=False)
            for frame, result in zip(frames_in_batch, results):
                inference_ms += result.speed["inference"]
                boxes = result.boxes.cpu().numpy()
                for (x1, y1, x2, y2), conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls.astype(int)):
                    if args.coco:
                        if cls not in COCO_TO_OURS:
                            continue
                        cls = COCO_TO_OURS[cls]
                    rows.append((frame, -1, x1, y1, x2 - x1, y2 - y1, conf, cls))
        df = pd.DataFrame(rows, columns=["frame", "id", "x", "y", "w", "h", "conf", "cls"])
        write_mot(df, args.out / f"{seq.name}.txt")
        frames += seq.length
        print(f"{seq.name}: {len(df)} boxes")

    elapsed = time.time() - start
    print(f"{frames} frames in {elapsed:.0f} s end to end ({frames / elapsed:.1f} frames/s); "
          f"model inference {inference_ms / frames:.1f} ms/frame")


if __name__ == "__main__":
    main()
