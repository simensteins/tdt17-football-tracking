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


def interpolate(tracks: pd.DataFrame, max_gap: int) -> pd.DataFrame:
    """Fill gaps of up to `max_gap` frames inside each track by linear interpolation.

    Offline post-processing: a box missing for a few frames (missed detection,
    brief occlusion) is drawn between the last box before and the first box
    after the gap. Filled boxes get the lower confidence of the two ends.
    """
    filled = []
    for _, track in tracks.groupby("id"):
        track = track.sort_values("frame")
        frames = track["frame"].to_numpy()
        for i in np.flatnonzero((np.diff(frames) > 1) & (np.diff(frames) <= max_gap + 1)):
            a, b = track.iloc[i], track.iloc[i + 1]
            missing = np.arange(a["frame"] + 1, b["frame"])
            t = (missing - a["frame"]) / (b["frame"] - a["frame"])
            box = {k: a[k] + t * (b[k] - a[k]) for k in ("x", "y", "w", "h")}
            filled.append(pd.DataFrame({"frame": missing, "id": a["id"], **box,
                                        "conf": min(a["conf"], b["conf"]), "cls": a["cls"]}))
    if not filled:
        return tracks
    return pd.concat([tracks, *filled], ignore_index=True).astype({"frame": int, "id": int, "cls": int}) \
        .sort_values(["frame", "id"], ignore_index=True)


def ball_path(detections: pd.DataFrame, tau: float = 0.2, top_k: int = 10, max_gap: int = 12,
              free_speed: float = 20.0, speed_scale: float = 15.0, gap_cost: float = 0.3,
              restart_cost: float = 3.0) -> pd.DataFrame:
    """Offline ball 'tracking': the most plausible ball path through all candidates.

    Uses past and future frames, so it fits post-match analysis. Picks at most
    one ball candidate per frame, maximising
        sum over chosen candidates of log(conf / tau)
        - sum over links of ((speed - free_speed) / speed_scale)^2 for speeds above free_speed
        - gap_cost per skipped frame inside a segment
        - restart_cost per new segment (the ball left the picture, or was lost).
    A candidate weaker than tau can still be chosen when it continues a
    plausible path; a confident outlier far from the path can be skipped.
    Solved exactly by dynamic programming over candidates in frame order.
    Speeds are in pixels per frame (median ball speed 3.6, 90th percentile 20).
    """
    top_k, max_gap = int(top_k), int(max_gap)
    balls = detections[detections["cls"] == BALL]
    balls = balls.sort_values(["frame", "conf"], ascending=[True, False]).groupby("frame").head(top_k)
    if balls.empty:
        return balls[DET_COLUMNS]
    frames = balls["frame"].to_numpy()
    centres = np.column_stack([balls["x"] + balls["w"] / 2, balls["y"] + balls["h"] / 2])
    reward = np.log(np.maximum(balls["conf"].to_numpy(), 1e-6) / tau)

    n = len(balls)
    best = np.empty(n)
    back = np.full(n, -1)  # previous candidate in the path, or -1 at the start of the path
    # Best path ending in any earlier frame, from which a new segment can start.
    best_so_far, best_so_far_idx = 0.0, -1  # the empty path
    done = 0  # candidates before this index belong to finished frames
    order_start = np.searchsorted(frames, frames, side="left")
    for i in range(n):
        f = frames[i]
        if i == order_start[i]:  # first candidate of a new frame
            for j in range(done, i):
                if best[j] > best_so_far:
                    best_so_far, best_so_far_idx = best[j], j
            done = i
        score, prev = best_so_far - restart_cost, best_so_far_idx
        lo = np.searchsorted(frames, f - max_gap, side="left")
        hi = order_start[i]
        if hi > lo:
            k = f - frames[lo:hi]
            speed = np.linalg.norm(centres[lo:hi] - centres[i], axis=1) / k
            link = best[lo:hi] - (np.maximum(speed - free_speed, 0) / speed_scale) ** 2 - gap_cost * (k - 1)
            j = int(np.argmax(link))
            if link[j] > score:
                score, prev = link[j], lo + j
        best[i] = reward[i] + score
        back[i] = prev

    end = int(np.argmax(best))
    if best[end] <= 0:
        return balls.iloc[:0][DET_COLUMNS]
    chosen = []
    while end >= 0:
        chosen.append(end)
        end = back[end]
    path = balls.iloc[sorted(chosen)].copy()
    path["id"] = BALL_ID_OFFSET + 1
    return path[DET_COLUMNS]


def track_sequence(seq: Sequence, detections: pd.DataFrame, cfg, read_images: bool = False,
                   ball: str = "tracker", ball_conf: float = 0.0, ball_args: dict | None = None) -> pd.DataFrame:
    """Track people with the given tracker; the ball with it too, or with single_ball() or ball_path()."""
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
    elif ball == "path":
        out.append(ball_path(detections, **(ball_args or {})))
    out = [o for o in out if len(o)]
    if not out:
        return pd.DataFrame(columns=DET_COLUMNS)
    return pd.concat(out, ignore_index=True).sort_values(["frame", "id"], ignore_index=True)
