"""Loading SoccerNet-Tracking 2023 sequences.

Each sequence folder holds:
    seqinfo.ini   frame rate, length and image size
    gameinfo.ini  match ID, clip context and the role of every track ID
    gt/gt.txt     MOT-format boxes: frame, id, x, y, w, h, conf, -1, -1, -1
    det/det.txt   identical to gt.txt with the IDs removed
    img1/         frames as 000001.jpg, 000002.jpg, ...

gt.txt carries no class, so each box gets its role by looking up its track ID
in gameinfo.ini.
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

CLASSES = ("player", "goalkeeper", "referee", "ball")

# Matches used for each split. Train and test share no match; validation is a
# whole training match held out, so clips of one match never sit on both sides.
SPLITS = {"train": (4, 9), "val": (6,), "test": (7, 8, 11)}

GT_COLUMNS = ["frame", "track", "x", "y", "w", "h", "conf", "c8", "c9", "c10"]


@dataclass(frozen=True)
class Tracklet:
    role: str      # one of CLASSES, or "other"
    team: str      # "left", "right" or "" when not applicable
    jersey: str    # jersey number, a placeholder letter, or a referee position
    raw: str       # the original label from gameinfo.ini


def parse_role(label: str) -> tuple[str, str]:
    """Map a gameinfo.ini role such as 'goalkeepers team left' to (role, team).

    The labels are not spelled consistently ('goalkeeper' and 'goalkeepers'
    both occur), so matching is done on the leading word.
    """
    words = label.lower().split()
    head = words[0] if words else ""
    team = words[-1] if len(words) >= 3 and words[-2] == "team" else ""
    if head.startswith("player"):
        return "player", team
    if head.startswith("goalkeeper"):
        return "goalkeeper", team
    if head.startswith("referee"):
        return "referee", ""
    if head.startswith("ball"):
        return "ball", ""
    return "other", team


def _read_ini(path: Path) -> configparser.SectionProxy:
    parser = configparser.ConfigParser()
    parser.read(path)
    return parser["Sequence"]


@dataclass
class Sequence:
    path: Path

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def info(self) -> configparser.SectionProxy:
        return _read_ini(self.path / "seqinfo.ini")

    @property
    def game(self) -> configparser.SectionProxy:
        return _read_ini(self.path / "gameinfo.ini")

    @property
    def game_id(self) -> int:
        return int(self.game["gameID"])

    @property
    def action(self) -> str:
        return self.game.get("actionClass", "").strip()

    @property
    def size(self) -> tuple[int, int]:
        return int(self.info["imWidth"]), int(self.info["imHeight"])

    @property
    def length(self) -> int:
        return int(self.info["seqLength"])

    def tracklets(self) -> dict[int, Tracklet]:
        out = {}
        for key, value in self.game.items():
            if not key.startswith("trackletid_"):
                continue
            label, _, jersey = value.strip().partition(";")
            role, team = parse_role(label)
            out[int(key.split("_")[1])] = Tracklet(role, team, jersey.strip(), label.strip())
        return out

    def image(self, frame: int) -> Path:
        return self.path / self.info["imDir"] / f"{frame:06d}{self.info['imExt']}"

    def boxes(self) -> pd.DataFrame:
        """Ground-truth boxes with role, team and sequence metadata attached."""
        df = pd.read_csv(self.path / "gt" / "gt.txt", header=None, names=GT_COLUMNS)
        df = df[["frame", "track", "x", "y", "w", "h"]]
        tracklets = self.tracklets()
        df["role"] = df["track"].map(lambda t: tracklets[t].role)
        df["team"] = df["track"].map(lambda t: tracklets[t].team)
        df["seq"] = self.name
        df["game"] = self.game_id
        df["action"] = self.action
        return df


def find_sequences(root: str | Path) -> list[Sequence]:
    """All sequence folders below root, sorted by name."""
    return [Sequence(p.parent) for p in sorted(Path(root).rglob("seqinfo.ini"))]
