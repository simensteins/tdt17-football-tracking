"""Label-level exploration of SoccerNet-Tracking sequences.

Reads only the annotation files, so it runs on the labels without images.
Prints summary tables and writes figures and CSVs to --out.

    python scripts/explore_labels.py data/tracking-2023/train --out outputs/eda/train
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football.data import CLASSES, find_sequences  # noqa: E402

ROLE_ORDER = list(CLASSES) + ["other"]
SOURCE_WIDTH = 1920
INPUT_SIZES = (640, 1280)


def pairwise_iou(boxes: np.ndarray) -> np.ndarray:
    """IoU matrix for boxes given as (x, y, w, h)."""
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = x1 + boxes[:, 2], y1 + boxes[:, 3]
    iw = np.clip(np.minimum(x2[:, None], x2) - np.maximum(x1[:, None], x1), 0, None)
    ih = np.clip(np.minimum(y2[:, None], y2) - np.maximum(y1[:, None], y1), 0, None)
    inter = iw * ih
    area = boxes[:, 2] * boxes[:, 3]
    return inter / (area[:, None] + area - inter)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise IoU between two aligned arrays of (x, y, w, h) boxes."""
    iw = np.clip(np.minimum(a[:, 0] + a[:, 2], b[:, 0] + b[:, 2]) - np.maximum(a[:, 0], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, 1] + a[:, 3], b[:, 1] + b[:, 3]) - np.maximum(a[:, 1], b[:, 1]), 0, None)
    inter = iw * ih
    return inter / (a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter)


def section(title: str) -> None:
    print(f"\n## {title}\n")


def show(df: pd.DataFrame) -> None:
    print(df.to_string())


# --- analyses ---------------------------------------------------------------


def sequences_overview(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.pivot_table(index="seq", columns="role", values="frame", aggfunc="size", fill_value=0)
    meta = df.groupby("seq").agg(game=("game", "first"), action=("action", "first"),
                                 tracks=("track", "nunique"))
    per_frame = df.groupby(["seq", "frame"]).size().groupby("seq").median().rename("median_per_frame")
    return meta.join(counts).join(per_frame)


def per_game(df: pd.DataFrame, moves: pd.DataFrame, pan: pd.DataFrame, max_iou: np.ndarray) -> pd.DataFrame:
    """One row per match, to compare the matches with each other."""
    d = df.assign(max_iou=max_iou)
    people = d[d["role"] != "ball"]
    frames = d.groupby(["game", "seq", "frame"]).size()
    ball_frames = d[d["role"] == "ball"].groupby(["game", "seq"])["frame"].nunique()
    seq_frames = d.groupby(["game", "seq"])["frame"].max()
    seq_game = d.groupby("seq")["game"].first()
    pan_game = pan.join(seq_game, on="seq")
    out = pd.DataFrame({
        "sequences": d.groupby("game")["seq"].nunique(),
        "boxes_per_frame": frames.groupby("game").median(),
        "goalkeeper_%": d.groupby("game")["role"].apply(lambda s: 100 * (s == "goalkeeper").mean()),
        "player_h_p50": d[d["role"] == "player"].groupby("game")["h"].median(),
        "player_h_p5": d[d["role"] == "player"].groupby("game")["h"].quantile(0.05),
        "ball_h_p50": d[d["role"] == "ball"].groupby("game")["h"].median(),
        "ball_visible_%": (100 * ball_frames / seq_frames).groupby("game").mean(),
        "people_iou>0.3_%": people.groupby("game")["max_iou"].apply(lambda s: 100 * (s > 0.3).mean()),
        "camera_pan_p50": pan_game.groupby("game")["pan_speed"].median(),
        "camera_pan_p95": pan_game.groupby("game")["pan_speed"].quantile(0.95),
    })
    return out.round(1)


def actions(df: pd.DataFrame) -> pd.DataFrame:
    seqs = df.drop_duplicates("seq")
    return seqs.pivot_table(index="action", columns="game", values="seq", aggfunc="size", fill_value=0)


def max_overlap(df: pd.DataFrame) -> np.ndarray:
    """For every box, the highest IoU with another box in the same frame."""
    out = np.zeros(len(df))
    for _, idx in df.groupby(["seq", "frame"]).indices.items():
        if len(idx) < 2:
            continue
        boxes = df.iloc[idx][["x", "y", "w", "h"]].to_numpy(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            iou = np.nan_to_num(pairwise_iou(boxes))
        np.fill_diagonal(iou, 0)
        out[idx] = iou.max(axis=1)
    return out


def class_balance(df: pd.DataFrame) -> pd.DataFrame:
    boxes = df.groupby("role").size().rename("boxes")
    tracks = df.drop_duplicates(["seq", "track"]).groupby("role").size().rename("tracks")
    out = pd.concat([boxes, tracks], axis=1).reindex(ROLE_ORDER).dropna()
    out["box_share_%"] = (100 * out["boxes"] / out["boxes"].sum()).round(1)
    return out.astype({"boxes": int, "tracks": int})


def box_sizes(df: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for role, g in df.groupby("role"):
        row = {}
        for q in (5, 50, 95):
            row[f"h_p{q}"] = np.percentile(g["h"], q)
        row["w_p50"] = g["w"].median()
        for size in INPUT_SIZES:
            scale = size / SOURCE_WIDTH
            row[f"h_p50@{size}"] = g["h"].median() * scale
            row[f"<8px@{size}_%"] = 100 * (np.minimum(g["w"], g["h"]) * scale < 8).mean()
        rows[role] = row
    return pd.DataFrame(rows).T.reindex(ROLE_ORDER).dropna().round(1)


def crowding(df: pd.DataFrame, max_iou: np.ndarray) -> pd.DataFrame:
    """Share of boxes overlapping another box above IoU thresholds."""
    d = df.assign(max_iou=max_iou)
    out = pd.DataFrame({f"iou>{t}_%": d.groupby("role")["max_iou"].apply(lambda s: 100 * (s > t).mean())
                        for t in (0.1, 0.3, 0.5)})
    return out.reindex(ROLE_ORDER).dropna().round(1)


def border_distance(rows: pd.DataFrame, width: int = 1920, height: int = 1080) -> np.ndarray:
    """Distance in pixels from each box to the nearest image edge."""
    return np.minimum.reduce([rows["x"].to_numpy(), rows["y"].to_numpy(),
                              width - (rows["x"] + rows["w"]).to_numpy(),
                              height - (rows["y"] + rows["h"]).to_numpy()])


def track_stats(df: pd.DataFrame, edge_px: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-track length and gaps; returns (per-track table, gap table).

    A gap counts as 'left view' when the box just before or just after it is
    within edge_px of the image border, otherwise as 'inside view'
    (occluded, or not annotated).
    """
    tracks, gaps = [], []
    for (seq, track), g in df.groupby(["seq", "track"]):
        g = g.sort_values("frame")
        frames = g["frame"].to_numpy()
        edge = border_distance(g) < edge_px
        role = g["role"].iloc[0]
        starts = np.flatnonzero(np.diff(frames) > 1)
        tracks.append(dict(seq=seq, track=track, role=role, boxes=len(frames),
                           span=frames[-1] - frames[0] + 1, gaps=len(starts)))
        for i in starts:
            gaps.append(dict(seq=seq, track=track, role=role, start=frames[i] + 1,
                             length=frames[i + 1] - frames[i] - 1,
                             cause="left view" if edge[i] or edge[i + 1] else "inside view"))
    columns = ["seq", "track", "role", "start", "length", "cause"]
    return pd.DataFrame(tracks), pd.DataFrame(gaps, columns=columns)


def track_summary(tracks: pd.DataFrame, gaps: pd.DataFrame, fps: float) -> pd.DataFrame:
    t = tracks.groupby("role").agg(tracks=("track", "size"), median_boxes=("boxes", "median"),
                                   with_gaps_pct=("gaps", lambda s: 100 * (s > 0).mean()),
                                   gaps_per_track=("gaps", "mean"))
    g = gaps.groupby("role")["length"].agg(gap_p50="median", gap_p90=lambda s: np.percentile(s, 90),
                                          gap_max="max")
    inside = gaps.groupby("role")["cause"].apply(lambda s: 100 * (s == "inside view").mean()).rename("inside_view_%")
    out = t.join(g).join(inside)
    for col in ("gap_p50", "gap_p90", "gap_max"):
        out[col.replace("gap", "gap_s")] = out[col] / fps
    return out.reindex(ROLE_ORDER).dropna(how="all").round(2)


def motion(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Frame-to-frame displacement of box centres.

    The camera pan for a frame is estimated as the median displacement of all
    people (not the ball) between that frame and the previous one. A player's
    own motion is their displacement minus the pan.
    """
    d = df.sort_values(["seq", "track", "frame"]).copy()
    d["cx"] = d["x"] + d["w"] / 2
    d["cy"] = d["y"] + d["h"] / 2
    g = d.groupby(["seq", "track"])
    consecutive = g["frame"].diff() == 1
    d["dx"] = g["cx"].diff().where(consecutive)
    d["dy"] = g["cy"].diff().where(consecutive)
    d = d.dropna(subset=["dx"])
    people = d[d["role"] != "ball"]
    pan = people.groupby(["seq", "frame"])[["dx", "dy"]].median().rename(columns={"dx": "pan_x", "dy": "pan_y"})
    d = d.join(pan, on=["seq", "frame"])
    d["speed"] = np.hypot(d["dx"], d["dy"])
    d["own_speed"] = np.hypot(d["dx"] - d["pan_x"], d["dy"] - d["pan_y"])
    pan["pan_speed"] = np.hypot(pan["pan_x"], pan["pan_y"])
    return d, pan


def frame_to_frame_iou(df: pd.DataFrame, pan: pd.DataFrame, offsets=(1, 2, 5)) -> pd.DataFrame:
    """IoU between a track's box at frame t and t+k, as seen by an IoU-based tracker.

    'comp' shifts the earlier box by the estimated camera pan first, which is
    what camera motion compensation does. Reports the median and the share of
    pairs with IoU below 0.3, a typical matching threshold.
    """
    key = df.set_index(["seq", "track", "frame"])[["x", "y", "w", "h", "role"]]
    pan_xy = pan[["pan_x", "pan_y"]].fillna(0)
    stats = {}
    for k in offsets:
        later = key.reset_index()
        later["frame"] -= k
        m = key.join(later.set_index(["seq", "track", "frame"]), rsuffix="_k", how="inner").reset_index()
        a = m[["x", "y", "w", "h"]].to_numpy(float)
        b = m[["x_k", "y_k", "w_k", "h_k"]].to_numpy(float)
        shift = np.zeros((len(m), 2))
        for j in range(1, k + 1):
            idx = pd.MultiIndex.from_arrays([m["seq"], m["frame"] + j])
            shift += pan_xy.reindex(idx).fillna(0).to_numpy()
        compensated = a.copy()
        compensated[:, :2] += shift
        for name, boxes in (("raw", a), ("comp", compensated)):
            iou = pd.Series(box_iou(boxes, b), index=m["role"].to_numpy())
            stats[(f"+{k} {name}", "median")] = iou.groupby(level=0).median()
            stats[(f"+{k} {name}", "<0.3_%")] = iou.groupby(level=0).apply(lambda s: 100 * (s < 0.3).mean())
    return pd.DataFrame(stats).reindex(ROLE_ORDER).dropna().round(2)


def ball_presence(df: pd.DataFrame, lengths: dict[str, int]) -> pd.DataFrame:
    ball = df[df["role"] == "ball"]
    out = pd.DataFrame({
        "ball_tracks": ball.groupby("seq")["track"].nunique(),
        "frames_with_ball": ball.groupby("seq")["frame"].nunique(),
    }).reindex(lengths.keys(), fill_value=0)
    out["frames"] = pd.Series(lengths)
    out["ball_visible_%"] = (100 * out["frames_with_ball"] / out["frames"]).round(1)
    multi = ball.groupby(["seq", "frame"]).size()
    out["frames_multi_ball"] = multi[multi > 1].groupby("seq").size().reindex(out.index, fill_value=0)
    return out


# --- figures ----------------------------------------------------------------


def plot_sizes(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.logspace(np.log10(4), np.log10(400), 50)
    for role in ROLE_ORDER:
        h = df.loc[df["role"] == role, "h"]
        if len(h):
            ax.hist(h, bins=bins, histtype="step", linewidth=1.5, density=True, label=f"{role} (n={len(h)})")
    ax.set_xscale("log")
    ax.set_xlabel("box height at 1920×1080 [px]")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "box_heights.png", dpi=150)
    plt.close(fig)


def plot_positions(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for ax, roles, title in ((axes[0], ["player", "goalkeeper", "referee"], "people (box bottom centre)"),
                             (axes[1], ["ball"], "ball (box centre)")):
        d = df[df["role"].isin(roles)]
        x = d["x"] + d["w"] / 2
        y = d["y"] + (d["h"] if roles != ["ball"] else d["h"] / 2)
        ax.hist2d(x, y, bins=(64, 36), range=((0, 1920), (0, 1080)), cmap="magma")
        ax.set_xlim(0, 1920)
        ax.set_ylim(1080, 0)
        ax.set_aspect("equal")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out / "positions.png", dpi=150)
    plt.close(fig)


def plot_pan(pan: pd.DataFrame, out: Path) -> None:
    seqs = pan.index.get_level_values("seq").unique()
    fig, axes = plt.subplots(len(seqs), 1, figsize=(8, 1.6 * len(seqs)), sharex=True, squeeze=False)
    for ax, seq in zip(axes[:, 0], seqs):
        p = pan.loc[seq]
        ax.plot(p.index, p["pan_x"], label="horizontal")
        ax.plot(p.index, p["pan_y"], label="vertical")
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.set_ylabel(seq, rotation=0, ha="right", va="center")
    axes[0, 0].legend(loc="upper right")
    axes[-1, 0].set_xlabel("frame")
    fig.suptitle("estimated camera pan [px / frame]")
    fig.tight_layout()
    fig.savefig(out / "camera_pan.png", dpi=150)
    plt.close(fig)


def plot_gaps(gaps: pd.DataFrame, fps: float, out: Path) -> None:
    if gaps.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.logspace(0, np.log10(max(gaps["length"].max(), 2)), 30)
    for role in ROLE_ORDER:
        g = gaps.loc[gaps["role"] == role, "length"]
        if len(g):
            ax.hist(g, bins=bins, histtype="step", linewidth=1.5, label=f"{role} (n={len(g)})")
    ax.set_xscale("log")
    ax.set_xlabel(f"gap length [frames, {fps:g} fps]")
    ax.set_ylabel("gaps")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "track_gaps.png", dpi=150)
    plt.close(fig)


# --- main -------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="folder containing sequence folders")
    parser.add_argument("--out", type=Path, required=True, help="folder for figures and CSVs")
    args = parser.parse_args()

    sequences = find_sequences(args.root)
    if not sequences:
        sys.exit(f"no sequences found under {args.root}")
    args.out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)

    df = pd.concat([s.boxes() for s in sequences], ignore_index=True)
    fps = float(sequences[0].info["frameRate"])
    lengths = {s.name: s.length for s in sequences}
    sizes = {s.size for s in sequences}

    print(f"# {len(sequences)} sequences, {df['game'].nunique()} games, {len(df)} boxes, image sizes {sizes}")
    raw = sorted({t.raw for s in sequences for t in s.tracklets().values()})
    print("raw role labels:", raw)

    section("Sequences")
    overview = sequences_overview(df)
    show(overview)
    section("Class balance")
    show(class_balance(df))
    section("Box sizes [px at 1920 wide; @N = after resizing to N wide]")
    show(box_sizes(df))
    section("Overlap with the most-overlapping other box in the same frame")
    max_iou = max_overlap(df)
    show(crowding(df, max_iou))
    section("Tracks and gaps (gap = frames missing inside a track)")
    tracks, gaps = track_stats(df)
    show(track_summary(tracks, gaps, fps))
    section("Ball presence per sequence")
    show(ball_presence(df, lengths))
    section("Motion [px / frame]")
    moves, pan = motion(df)
    show(moves.groupby("role")[["speed", "own_speed"]].describe(percentiles=[0.5, 0.95])
         .loc[:, (slice(None), ["50%", "95%", "max"])].reindex(ROLE_ORDER).dropna().round(1))
    print()
    show(pan.groupby("seq")["pan_speed"].describe(percentiles=[0.5, 0.95])[["50%", "95%", "max"]]
         .round(1).rename(columns=lambda c: f"camera_pan_{c}"))
    section("IoU of a track's box at frame t and t+k (raw, and compensated for camera pan)")
    show(frame_to_frame_iou(df, pan))
    section("Per match")
    show(per_game(df, moves, pan, max_iou))
    section("Clip context (actionClass) per match")
    show(actions(df))

    overview.to_csv(args.out / "sequences.csv")
    tracks.to_csv(args.out / "tracks.csv", index=False)
    gaps.to_csv(args.out / "gaps.csv", index=False)
    plot_sizes(df, args.out)
    plot_positions(df, args.out)
    plot_pan(pan, args.out)
    plot_gaps(gaps, fps, args.out)
    print(f"\nfigures and CSVs written to {args.out}")


if __name__ == "__main__":
    main()
