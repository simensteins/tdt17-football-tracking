"""Check the roles (player, goalkeeper, referee) in track files.

HOTA as computed by eval_tracking.py ignores roles. Here every ground-truth
person box is matched to the track box with the highest IoU (above 0.5) in
the same frame, and the two roles are compared.

    python scripts/eval_roles.py data/tracking-2023 outputs/tracks/test/<run> --split test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football.data import CLASSES, SPLITS, find_sequences  # noqa: E402
from football.tracking import BALL, gt_detections, read_mot  # noqa: E402

PEOPLE = [c for c in CLASSES if c != "ball"]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a and every box in b (x, y, w, h)."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 0] + a[:, None, 2], b[None, :, 0] + b[None, :, 2])
    y2 = np.minimum(a[:, None, 1] + a[:, None, 3], b[None, :, 1] + b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / (a[:, None, 2] * a[:, None, 3] + b[None, :, 2] * b[None, :, 3] - inter)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="tracking-2023 folder")
    parser.add_argument("tracks", type=Path, help="folder with <sequence>.txt track files")
    parser.add_argument("--split", choices=SPLITS, required=True)
    args = parser.parse_args()

    confusion = np.zeros((len(PEOPLE), len(PEOPLE)), dtype=int)  # rows: true role, columns: predicted
    for seq in (s for s in find_sequences(args.root) if s.game_id in SPLITS[args.split]):
        gt = gt_detections(seq)
        gt = gt[gt["cls"] != BALL]
        tracks = read_mot(args.tracks / f"{seq.name}.txt")
        by_frame = dict(tuple(tracks[tracks["cls"] != BALL].groupby("frame")))
        for frame, truth in gt.groupby("frame"):
            pred = by_frame.get(frame)
            if pred is None:
                continue
            iou = iou_matrix(truth[["x", "y", "w", "h"]].to_numpy(), pred[["x", "y", "w", "h"]].to_numpy())
            matched = iou.max(1) > 0.5
            for true, j in zip(truth["cls"].to_numpy()[matched], iou.argmax(1)[matched]):
                confusion[true, int(pred["cls"].iloc[j])] += 1

    print(f"role accuracy over {confusion.sum()} matched boxes: {np.trace(confusion) / confusion.sum():.3f}")
    print(f"{'true / predicted':18s}" + "".join(f"{c:>12s}" for c in PEOPLE) + f"{'recall':>10s}")
    for i, role in enumerate(PEOPLE):
        print(f"{role:18s}" + "".join(f"{n:12d}" for n in confusion[i]) + f"{confusion[i, i] / confusion[i].sum():10.3f}")
    precision = np.diag(confusion) / confusion.sum(0)
    print(f"{'precision':18s}" + "".join(f"{p:12.3f}" for p in precision))


if __name__ == "__main__":
    main()
