"""Evaluate tracker outputs with TrackEval (HOTA, CLEAR, Identity).

Each tracker folder holds <sequence>.txt files from scripts/track.py. Results
are computed for three subsets: all four classes, people only (player,
goalkeeper, referee) and ball only. The `other` role is excluded from the
ground truth.

    python scripts/eval_tracking.py data/tracking-2023 --split val \
        outputs/tracks/val/oracle-bytetrack outputs/tracks/val/oracle-botsort --out outputs/eval/val

Prints a summary table and writes per-sequence metrics to --out/per_sequence.csv.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football.data import CLASSES, SPLITS, find_sequences  # noqa: E402
from football.tracking import read_mot  # noqa: E402

# TrackEval still uses numpy aliases that numpy 1.24 removed; restore them
# before it is imported.
np.float = float
np.int = int
np.bool = bool

SUBSETS = {
    "all": set(CLASSES),
    "people": {"player", "goalkeeper", "referee"},
    "ball": {"ball"},
}
SUMMARY = ["HOTA", "DetA", "AssA", "LocA", "IDF1", "MOTA", "IDSW"]


def _write_inputs(tmp: Path, sequences, tracker_dirs: list[Path], roles: set[str]) -> None:
    """GT and tracker files in MOTChallenge layout, restricted to the given roles."""
    class_ids = {CLASSES.index(r) for r in roles}
    for seq in sequences:
        gt = seq.boxes()
        gt = gt[gt["role"].isin(roles) & (gt["w"] > 0) & (gt["h"] > 0)]
        gt_dir = tmp / "gt" / seq.name / "gt"
        gt_dir.mkdir(parents=True)
        pd.DataFrame({"frame": gt["frame"], "id": gt["track"], "x": gt["x"], "y": gt["y"], "w": gt["w"],
                      "h": gt["h"], "keep": 1, "cls": 1, "vis": 1}).to_csv(gt_dir / "gt.txt", header=False, index=False)
        for tracker_dir in tracker_dirs:
            tracks = read_mot(tracker_dir / f"{seq.name}.txt")
            tracks = tracks[tracks["cls"].isin(class_ids)]
            out = tmp / "trackers" / tracker_dir.name / "data"
            out.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"frame": tracks["frame"], "id": tracks["id"], "x": tracks["x"], "y": tracks["y"],
                          "w": tracks["w"], "h": tracks["h"], "conf": tracks["conf"], "cls": 1, "a": -1, "b": -1}
                         ).to_csv(out / f"{seq.name}.txt", header=False, index=False, float_format="%.2f")


def run_trackeval(sequences, tracker_dirs: list[Path], roles: set[str]) -> dict:
    """Raw TrackEval results: {tracker: {sequence or 'COMBINED_SEQ': {metric_name: result_dict}}}."""
    import trackeval

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _write_inputs(tmp, sequences, tracker_dirs, roles)
        dataset_config = {
            "GT_FOLDER": str(tmp / "gt"),
            "TRACKERS_FOLDER": str(tmp / "trackers"),
            "TRACKERS_TO_EVAL": [d.name for d in tracker_dirs],
            "SEQ_INFO": {s.name: s.length for s in sequences},
            "SKIP_SPLIT_FOL": True,
            "DO_PREPROC": False,
            "PRINT_CONFIG": False,
        }
        eval_config = {"USE_PARALLEL": False, "PRINT_RESULTS": False, "PRINT_CONFIG": False, "TIME_PROGRESS": False,
                       "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False, "PLOT_CURVES": False, "LOG_ON_ERROR": None}
        evaluator = trackeval.Evaluator(eval_config)
        dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
        metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
                   trackeval.metrics.Identity({"PRINT_CONFIG": False})]
        results, _ = evaluator.evaluate([dataset], metrics)
    out = {}
    for tracker, per_seq in results["MotChallenge2DBox"].items():
        out[tracker] = {seq: res["pedestrian"] for seq, res in per_seq.items()}
    return out


def summarise(res: dict) -> dict[str, float]:
    """Headline numbers from one sequence's (or a combination's) metric results."""
    hota, clear, ident = res["HOTA"], res["CLEAR"], res["Identity"]
    return {
        "HOTA": 100 * float(np.mean(hota["HOTA"])),
        "DetA": 100 * float(np.mean(hota["DetA"])),
        "AssA": 100 * float(np.mean(hota["AssA"])),
        "LocA": 100 * float(np.mean(hota["LocA"])),
        "IDF1": 100 * float(ident["IDF1"]),
        "MOTA": 100 * float(clear["MOTA"]),
        "IDSW": int(clear["IDSW"]),
    }


def combine(per_seq: dict, names: list[str]) -> dict:
    """Combine the metric results of a group of sequences the way TrackEval does."""
    import trackeval

    metrics = {"HOTA": trackeval.metrics.HOTA(), "CLEAR": trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
               "Identity": trackeval.metrics.Identity({"PRINT_CONFIG": False})}
    return {m: metric.combine_sequences({n: per_seq[n][m] for n in names}) for m, metric in metrics.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="tracking-2023 folder")
    parser.add_argument("trackers", type=Path, nargs="+", help="tracker output folders")
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--sequences", nargs="*", help="only these sequences (default: whole split)")
    parser.add_argument("--subsets", nargs="+", choices=SUBSETS, default=list(SUBSETS))
    parser.add_argument("--by-action", action="store_true", help="also report HOTA per clip type")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    sequences = [s for s in find_sequences(args.root) if s.game_id in SPLITS[args.split]
                 and (not args.sequences or s.name in args.sequences)]
    action = {s.name: s.action for s in sequences}
    rows = []
    summary = {}
    for subset in args.subsets:
        results = run_trackeval(sequences, args.trackers, SUBSETS[subset])
        for tracker, per_seq in results.items():
            summary[(tracker, subset)] = summarise(per_seq["COMBINED_SEQ"])
            for seq in action:
                rows.append({"tracker": tracker, "subset": subset, "seq": seq, "action": action[seq],
                             **summarise(per_seq[seq])})
            if args.by_action and subset == "all":
                for act in sorted(set(action.values())):
                    names = [s for s, a in action.items() if a == act]
                    summary[(tracker, f"action: {act}")] = summarise(combine(per_seq, names))

    pd.set_option("display.width", 200)
    table = pd.DataFrame(summary).T.rename_axis(["tracker", "subset"])
    print(table[SUMMARY].round(1).to_string())
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).round(2).to_csv(args.out / "per_sequence.csv", index=False)
        table.round(2).to_csv(args.out / "summary.csv")


if __name__ == "__main__":
    main()
