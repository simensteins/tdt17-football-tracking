"""Run an Ultralytics tracker over all sequences of a split.

Detections come either from the ground truth without IDs (--detections gt,
the perfect detector) or from files written by scripts/detect.py.

    python scripts/track.py data/tracking-2023 --split val --detections gt \
        --tracker bytetrack.yaml --set track_buffer=150 --out outputs/tracks/oracle-bytetrack-150

Writes one <sequence>.txt per sequence: frame, id, x, y, w, h, conf, class.
"""

from __future__ import annotations

import os

# The Kalman filter solves tiny 8x8 systems; multi-threaded BLAS only adds
# overhead, and several tracker processes side by side starve each other.
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(var, "1")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football.data import SPLITS, find_sequences  # noqa: E402
from football.tracking import (gt_detections, interpolate, load_tracker_config, merge_people,  # noqa: E402
                               read_mot, track_sequence, write_mot)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="tracking-2023 folder")
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--sequences", nargs="*", help="only these sequences (default: whole split)")
    parser.add_argument("--detections", required=True, help="'gt', or a folder of <sequence>.txt detection files")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker config")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="tracker config overrides")
    parser.add_argument("--images", action="store_true", help="read frames (needed for camera motion compensation and ReID)")
    parser.add_argument("--ball", choices=("tracker", "single", "path"), default="tracker",
                        help="ball with the tracker; the best detection per frame ('single'); "
                             "or the most plausible path through all candidates ('path', offline)")
    parser.add_argument("--ball-path", nargs="*", default=[], metavar="KEY=VALUE",
                        help="options for --ball path, e.g. tau=0.2 restart_cost=3")
    parser.add_argument("--ball-conf", type=float, default=0.0, help="minimum confidence for --ball single")
    parser.add_argument("--merge-iou", type=float,
                        help="merge people boxes of different classes overlapping by more than this IoU")
    parser.add_argument("--interpolate", type=int, default=0, metavar="FRAMES",
                        help="fill gaps of up to this many frames inside each track (offline post-processing)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_tracker_config(args.tracker, dict(s.split("=", 1) for s in args.set))
    sequences = [s for s in find_sequences(args.root) if s.game_id in SPLITS[args.split]
                 and (not args.sequences or s.name in args.sequences)]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.txt").write_text(
        f"tracker: {args.tracker}\noverrides: {' '.join(args.set)}\ndetections: {args.detections}\n"
        f"ball: {args.ball} (min conf {args.ball_conf}; path options {' '.join(args.ball_path)})\nmerge people above IoU: {args.merge_iou}\n"
        f"interpolate gaps up to: {args.interpolate} frames\n"
        + "".join(f"{k}: {v}\n" for k, v in vars(cfg).items()))

    start = time.time()
    frames = 0
    for seq in sequences:
        dets = gt_detections(seq) if args.detections == "gt" else read_mot(Path(args.detections) / f"{seq.name}.txt")
        if args.merge_iou is not None:
            dets = merge_people(dets, args.merge_iou)
        tracks = track_sequence(seq, dets, cfg, read_images=args.images, ball=args.ball, ball_conf=args.ball_conf,
                                ball_args={k: float(v) for k, v in (o.split("=", 1) for o in args.ball_path)})
        if args.interpolate:
            tracks = interpolate(tracks, args.interpolate)
        write_mot(tracks, args.out / f"{seq.name}.txt")
        frames += seq.length
        print(f"{seq.name}: {tracks['id'].nunique()} tracks, {len(tracks)} boxes")
    elapsed = time.time() - start
    print(f"{len(sequences)} sequences, {frames} frames in {elapsed:.0f} s ({frames / elapsed:.0f} frames/s)")


if __name__ == "__main__":
    main()
