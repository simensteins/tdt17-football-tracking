"""Running Ultralytics trackers on stored detections.

Detection files use MOT text format, one box per line:
    frame, -1, x, y, w, h, confidence, class
with class an index into football.data.CLASSES. Track files add the ID:
    frame, id, x, y, w, h, confidence, class

People (player, goalkeeper, referee) and the ball are tracked by separate
tracker instances. The ball moves far between frames and barely overlaps its
own previous box, and it should never be associated with a person.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from football.data import CLASSES, Sequence

BALL = CLASSES.index("ball")
BALL_ID_OFFSET = 100_000  # keeps ball IDs apart from people IDs in one file

DET_COLUMNS = ["frame", "id", "x", "y", "w", "h", "conf", "cls"]


def gt_detections(seq: Sequence) -> pd.DataFrame:
    """Ground-truth boxes without IDs: the perfect detector (identical to det.txt)."""
    boxes = seq.boxes()
    boxes = boxes[boxes["role"].isin(CLASSES) & (boxes["w"] > 0) & (boxes["h"] > 0)]
    return pd.DataFrame({
        "frame": boxes["frame"], "id": -1, "x": boxes["x"], "y": boxes["y"], "w": boxes["w"], "h": boxes["h"],
        "conf": 1.0, "cls": boxes["role"].map(CLASSES.index),
    })


def read_mot(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, header=None, names=DET_COLUMNS)


def write_mot(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df[DET_COLUMNS].to_csv(path, header=False, index=False, float_format="%.2f")


def load_tracker_config(name_or_path: str, overrides: dict[str, str]):
    """Ultralytics tracker config (e.g. 'bytetrack.yaml') with key=value overrides."""
    from ultralytics.utils import YAML, IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml

    cfg = YAML.load(check_yaml(name_or_path))
    for key, value in overrides.items():
        if key not in cfg:
            raise KeyError(f"unknown tracker option {key!r}; options: {sorted(cfg)}")
        cfg[key] = type(cfg[key])(value) if not isinstance(cfg[key], bool) else value.lower() in ("1", "true")
    return IterableSimpleNamespace(**cfg)


def make_tracker(cfg):
    from ultralytics.trackers.track import TRACKER_MAP

    return TRACKER_MAP[cfg.tracker_type](args=cfg)


def _boxes(rows: pd.DataFrame, shape: tuple[int, int]):
    from ultralytics.engine.results import Boxes

    data = np.column_stack([rows["x"], rows["y"], rows["x"] + rows["w"], rows["y"] + rows["h"],
                            rows["conf"], rows["cls"]]).astype(np.float32)
    return Boxes(data.reshape(-1, 6), shape)


def merge_people(detections: pd.DataFrame, iou: float) -> pd.DataFrame:
    """Class-agnostic NMS on people boxes.

    YOLO26 has no NMS step and picks boxes per class, so one person can come
    out as both a player and a goalkeeper. Each copy would start its own track.
    Keeps the most confident box, and its class, among people boxes overlapping
    by more than `iou`.
    """
    import torch
    from torchvision.ops import batched_nms

    people = detections[detections["cls"] != BALL]
    boxes = torch.tensor(np.column_stack([people["x"], people["y"], people["x"] + people["w"],
                                          people["y"] + people["h"]]), dtype=torch.float32)
    keep = batched_nms(boxes, torch.tensor(people["conf"].to_numpy(), dtype=torch.float32),
                       torch.tensor(people["frame"].to_numpy()), iou).numpy()
    return pd.concat([people.iloc[np.sort(keep)], detections[detections["cls"] == BALL]]).sort_values("frame", kind="stable")


def single_ball(detections: pd.DataFrame, min_conf: float = 0.0) -> pd.DataFrame:
    """Ball 'tracking' using the prior that there is one ball.

    Keeps the most confident ball detection in each frame and gives it one
    fixed ID for the whole sequence. Frames with two real balls (a replacement
    ball) lose the second one.
    """
    balls = detections[(detections["cls"] == BALL) & (detections["conf"] >= min_conf)]
    best = balls.loc[balls.groupby("frame")["conf"].idxmax()].copy()
    best["id"] = BALL_ID_OFFSET + 1
    return best[DET_COLUMNS]


def track_sequence(seq: Sequence, detections: pd.DataFrame, cfg, read_images: bool = False,
                   ball: str = "tracker", ball_conf: float = 0.0) -> pd.DataFrame:
    """Track people with the given tracker; the ball with it too, or with single_ball()."""
    import cv2

    width, height = seq.size
    shape = (height, width)
    trackers = {"people": make_tracker(cfg)}
    if ball == "tracker":
        trackers["ball"] = make_tracker(cfg)
    by_frame = dict(tuple(detections.groupby("frame")))
    empty = detections.iloc[:0]
    out = []
    for frame in range(1, seq.length + 1):
        img = cv2.imread(str(seq.image(frame))) if read_images else None
        dets = by_frame.get(frame, empty)
        for group, tracker in trackers.items():
            rows = dets[dets["cls"] == BALL] if group == "ball" else dets[dets["cls"] != BALL]
            tracks = tracker.update(_boxes(rows, shape), img)
            if len(tracks) == 0:
                continue
            # Report the matched detection's box rather than the Kalman-smoothed
            # state, so localisation reflects the detector and not the tracker.
            matched = rows.iloc[tracks[:, 7].astype(int)]
            ids = tracks[:, 4].astype(int) + (BALL_ID_OFFSET if group == "ball" else 0)
            out.append(pd.DataFrame({"frame": frame, "id": ids, "x": matched["x"].to_numpy(),
                                     "y": matched["y"].to_numpy(), "w": matched["w"].to_numpy(),
                                     "h": matched["h"].to_numpy(), "conf": matched["conf"].to_numpy(),
                                     "cls": matched["cls"].to_numpy().astype(int)}))
    if ball == "single":
        out.append(single_ball(detections, ball_conf))
    out = [o for o in out if len(o)]
    if not out:
        return pd.DataFrame(columns=DET_COLUMNS)
    return pd.concat(out, ignore_index=True).sort_values(["frame", "id"], ignore_index=True)
