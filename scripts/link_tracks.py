"""Link tracks of the same player using jersey numbers (offline post-processing).

A player who leaves the picture and comes back gets a new track ID: motion
alone cannot tell that it is the same person. This joins such tracks when
they show the same jersey number, role and team.

    python scripts/link_tracks.py data/tracking-2023 <tracks dir> <tracks dir>/jerseys.csv \
        --split val --out <linked tracks dir>

- A track's number comes from the readings of scripts/read_jerseys.py, and is
  trusted only when they agree (see trusted_number).
- Players are split into two teams per clip by shirt colour (2-means on the
  mean CIELAB colour of their crops). Goalkeepers and referees are kept apart by role.
- Tracks with the same (role, team, number) that never overlap in time get one
  ID. Overlapping tracks with the same key stay separate: one reading is wrong.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football.data import CLASSES, SPLITS, find_sequences  # noqa: E402
from football.tracking import BALL, read_mot, write_mot  # noqa: E402

PLAYER = CLASSES.index("player")


def two_means(x: np.ndarray, iterations: int = 20) -> np.ndarray:
    """Split rows of x into two clusters; returns labels 0/1."""
    centres = x[[np.argmin(x[:, 0]), np.argmax(x[:, 0])]]  # start from the darkest and lightest
    labels = np.zeros(len(x), dtype=int)
    for _ in range(iterations):
        labels = np.linalg.norm(x[:, None] - centres[None], axis=2).argmin(1)
        new = np.array([x[labels == k].mean(0) if (labels == k).any() else centres[k] for k in (0, 1)])
        if np.allclose(new, centres):
            break
        centres = new
    return labels


def trusted_number(numbers: pd.Series, role: int, min_votes: int, min_share: float,
                   single_digits: bool = True) -> str | None:
    """The track's jersey number, if the readings agree enough.

    Florence-2 often drops a digit of a two-digit number (24 read as 2) and
    reads a front view without a number as '1'. So a two-digit number read at
    least min_votes times wins; single digits need one more vote, and '1'
    only counts for goalkeepers (outfield players do not wear it).
    """
    numbers = numbers.dropna()
    two = numbers[numbers.str.len() == 2].value_counts()
    if len(two) and two.iloc[0] >= min_votes and two.iloc[0] / two.sum() >= min_share:
        return two.index[0]
    if not single_digits:
        return None
    one = numbers[numbers.str.len() == 1]
    if role != CLASSES.index("goalkeeper"):
        one = one[one != "1"]
    one = one.value_counts()
    if len(one) and one.iloc[0] >= min_votes + 1 and one.iloc[0] / one.sum() >= min_share:
        return one.index[0]
    return None


def track_keys(tracks: pd.DataFrame, readings: pd.DataFrame, min_votes: int, min_share: float,
               single_digits: bool = True) -> pd.DataFrame:
    """Per track: role, team, trusted number (or NaN), first and last frame."""
    info = tracks.groupby("id").agg(start=("frame", "min"), end=("frame", "max"),
                                    role=("cls", lambda c: c.mode().iloc[0]))
    by_track = dict(tuple(readings.groupby("track")["number"]))
    info["number"] = [trusted_number(by_track.get(t, pd.Series(dtype="string")), role, min_votes, min_share,
                                     single_digits)
                      for t, role in zip(info.index, info["role"])]

    colour = readings.groupby("track")[["L", "a", "b"]].mean().reindex(info.index)
    info["team"] = -1
    players = info.index[(info["role"] == PLAYER) & colour.notna().all(1)]
    if len(players) >= 2:
        info.loc[players, "team"] = two_means(colour.loc[players].to_numpy())
    return info


def link(info: pd.DataFrame) -> dict[int, int]:
    """New ID for every track: chains of non-overlapping tracks with the same key share an ID."""
    new_id = {int(t): int(t) for t in info.index}
    keyed = info[info["number"].notna()]
    for _, group in keyed.groupby(["role", "team", "number"]):
        chains: list[list] = []  # each chain: [id, last frame]
        for track, row in group.sort_values("start").iterrows():
            free = [c for c in chains if c[1] < row["start"]]
            if free:
                chain = max(free, key=lambda c: c[1])  # the one that ended most recently
                new_id[int(track)] = chain[0]
                chain[1] = row["end"]
            else:
                chains.append([int(track), row["end"]])
    return new_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="tracking-2023 folder")
    parser.add_argument("tracks", type=Path, help="folder with <sequence>.txt track files")
    parser.add_argument("readings", type=Path, help="jerseys.csv from scripts/read_jerseys.py")
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--min-votes", type=int, default=2, help="readings needed to trust a number")
    parser.add_argument("--min-share", type=float, default=0.6)
    parser.add_argument("--single-digits", action="store_true",
                        help="also trust one-digit numbers (often a two-digit number missing a digit)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    readings = pd.read_csv(args.readings, dtype={"number": "string"})
    args.out.mkdir(parents=True, exist_ok=True)
    linked_total = 0
    for seq in (s for s in find_sequences(args.root) if s.game_id in SPLITS[args.split]):
        path = args.tracks / f"{seq.name}.txt"
        if not path.exists():
            continue
        tracks = read_mot(path)
        people = tracks[tracks["cls"] != BALL]
        info = track_keys(people, readings[readings["seq"] == seq.name], args.min_votes, args.min_share,
                          args.single_digits)
        new_id = link(info)
        linked = sum(1 for old, new in new_id.items() if old != new)
        linked_total += linked
        tracks.loc[tracks["cls"] != BALL, "id"] = people["id"].map(new_id)
        write_mot(tracks.sort_values(["frame", "id"]), args.out / path.name)
        print(f"{seq.name}: {len(info)} tracks, {info['number'].notna().sum()} with a number, {linked} linked")
    print(f"{linked_total} tracks joined to an earlier track")


if __name__ == "__main__":
    main()
