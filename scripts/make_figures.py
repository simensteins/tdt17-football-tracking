"""Figures for the presentation, as PNG files sized for 16:9 slides, numbered in talk order.

    python scripts/make_figures.py data/tracking-2023 --out outputs/figures

Label statistics come from the ground truth (training and validation
matches), per-clip-type scores from outputs/eval/test-run4/summary.csv. All
other numbers are copied from the result tables in docs/decisions.md, which
is where they are recorded with their runs.

Colours are the first three slots of a colour-blind-checked categorical
palette (blue, orange, aqua); grey marks the runs a chart does not highlight.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football.data import SPLITS, find_sequences  # noqa: E402

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, OTHER = "#e1e0d9", "#c3c2b7", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SIZE = (12, 6.75)  # inches; 2400 x 1350 px at 200 dpi

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 15,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlesize": 17, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.titlepad": 12,
    "xtick.color": MUTED, "ytick.color": INK2, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "grid.color": GRID, "grid.linewidth": 1, "legend.frameon": False, "legend.fontsize": 14,
})


def save(fig: plt.Figure, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"wrote {out / name}.png")


def title(fig: plt.Figure, text: str, subtitle: str = "") -> None:
    fig.text(0.01, 0.995, text, ha="left", va="top", fontsize=21, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.01, 0.935, subtitle, ha="left", va="top", fontsize=15, color=INK2)


def hgrid(ax: plt.Axes) -> None:
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


# --- data analysis --------------------------------------------------------------------------------

def box_heights(root: Path, out: Path) -> None:
    games = set(SPLITS["train"]) | set(SPLITS["val"])
    boxes = pd.concat([s.boxes() for s in find_sequences(root) if s.game_id in games])
    roles = [("player", "Player"), ("referee", "Referee"), ("goalkeeper", "Goalkeeper"), ("ball", "Ball")]
    share = boxes["role"].value_counts(normalize=True)
    bins = np.logspace(np.log10(4), np.log10(400), 60)

    fig, axes = plt.subplots(4, 1, figsize=SIZE, sharex=True, gridspec_kw={"hspace": 0.55})
    for ax, (role, label) in zip(axes, roles):
        h = boxes.loc[boxes["role"] == role, "h"]
        weights = np.full(len(h), 100 / len(h))
        ax.hist(h, bins=bins, weights=weights, color=ORANGE if role == "ball" else BLUE, rwidth=0.85)
        ax.set_xscale("log")
        ax.set_title(f"{label}: {share[role]:.1%} of boxes, median {h.median():.0f} px", fontsize=14,
                     fontweight="normal", pad=4)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        for px, text in ((24, "8 px at 640 input"), (12, "8 px at 1280")):
            ax.axvline(px, color=MUTED, linewidth=1)
            if role == "player":
                ax.text(px * 1.04, 0.9, text, transform=ax.get_xaxis_transform(), fontsize=12, color=INK2, va="top")
    axes[-1].set_xticks([5, 10, 20, 50, 100, 200, 400], ["5", "10", "20", "50", "100", "200", "400"])
    axes[-1].set_xlabel("Box height in the 1920 × 1080 frame (px, log scale)")
    title(fig, "The ball is ten times smaller than a player",
          f"Box heights in the training and validation matches ({len(boxes):,} boxes); "
          "most balls shrink below 8 px at 640 input")
    fig.subplots_adjust(top=0.84)
    save(fig, out, "01-box-heights")


# --- method ---------------------------------------------------------------------------------------

def pipeline(out: Path) -> None:
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=SIZE)
    ax.set_xlim(0, 14.2)
    ax.set_ylim(0.3, 5.1)
    ax.axis("off")
    w, h = 2.0, 1.3
    people_y, ball_y = 3.45, 1.35
    x = dict(zip(("input", "detect", "associate", "post1", "post2", "output"), np.arange(6) * 2.4 + 0.1))

    # the offline part works on the whole clip at once, which suits post-match analysis
    ax.add_patch(FancyBboxPatch((x["post1"] - 0.15, 0.4), 2 * w + 0.7, 4.65, boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor="#f0efec", edgecolor="none"))
    for key, text in (("input", "Input"), ("detect", "Detection (GPU)"), ("associate", "Association (online)"),
                      ("post1", "Post-processing (offline, whole clip)"), ("output", "Output")):
        ax.text(x[key], 4.7, text, fontsize=14, fontweight="bold", color=INK2, va="center")

    def box(left, y, name, detail, colour, height=h):
        ax.add_patch(FancyBboxPatch((left, y - height / 2), w, height, boxstyle="round,pad=0,rounding_size=0.1",
                                    facecolor=SURFACE, edgecolor=AXIS, linewidth=1.2))
        ax.add_patch(FancyBboxPatch((left, y - height / 2), 0.09, height, boxstyle="square,pad=0", facecolor=colour,
                                    edgecolor="none"))
        ax.text(left + 0.22, y + height / 2 - 0.18, name, fontsize=12.5, fontweight="bold", color=INK, va="top")
        ax.text(left + 0.22, y + height / 2 - 0.5, detail, fontsize=11, color=INK2, va="top", linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops={"arrowstyle": "-|>", "color": MUTED, "linewidth": 1.5,
                                                        "shrinkA": 2, "shrinkB": 2, "mutation_scale": 16})

    tall = people_y - ball_y + h
    mid = (people_y + ball_y) / 2
    box(x["input"], mid, "Broadcast video", "1920 × 1080, 25 fps,\n30 s clips", MUTED, height=tall)
    box(x["detect"], people_y, "YOLO26m", "players, goalkeepers,\nreferees; 1280 px", BLUE)
    box(x["detect"], ball_y, "RF-DETR-M", "ball candidates only;\nDINOv2, 1280 px", ORANGE)
    box(x["associate"], people_y, "BoT-SORT + GMC", "camera motion\nremoved; lost tracks\nkept for 6 s", BLUE)
    box(x["post1"], people_y, "Fill gaps", "up to 10 frames,\nlinear", BLUE)
    box(x["post2"], people_y, "Jersey linking", "Qwen3.5-4B reads\nnumbers; same\nnumber → one ID",
        BLUE)
    box(x["post1"], ball_y, "Ball path", "best path through\nall candidates\nin the clip", ORANGE)
    box(x["post2"], ball_y, "Fill gaps", "up to 10 frames,\nlinear", ORANGE)
    box(x["output"], mid, "Tracks", "ID and role per\nperson, one ball,\nevery frame", MUTED, height=tall)

    for y in (people_y, ball_y):
        arrow(x["input"] + w, y, x["detect"], y)
        arrow(x["post1"] + w, y, x["post2"], y)
        arrow(x["post2"] + w, y, x["output"], y)
    arrow(x["detect"] + w, people_y, x["associate"], people_y)
    arrow(x["associate"] + w, people_y, x["post1"], people_y)
    arrow(x["detect"] + w, ball_y, x["post1"], ball_y)

    title(fig, "Pipeline: tracking by detection, with post-match refinement",
          "People (blue) and the ball (orange) are detected and tracked separately; "
          "the shaded steps use the whole clip")
    fig.subplots_adjust(top=0.86, bottom=0.02, left=0.01, right=0.99)
    save(fig, out, "02-pipeline")


# --- detectors ------------------------------------------------------------------------------------

DETECTORS = [  # validation match 6, last epoch; decisions.md D2c, D2f, D6, D7, D5 and the RF-DETR check
    ("YOLO, 2 matches (D2c)", 0.397, 64.1, 96.7),
    ("YOLO + hue (D2f)", 0.431, 64.1, 97.1),
    ("YOLO + 3 GSR matches (D6)", 0.407, 64.8, 92.2),
    ("YOLO + both (D7)", 0.474, 65.5, 98.2),
    ("RF-DETR, 2 matches", 0.448, 65.8, 97.9),
    ("RF-DETR + both", 0.478, 65.7, 98.9),
]


def detectors(out: Path) -> None:
    names = [d[0] for d in DETECTORS][::-1]
    panels = [("mAP@50:95", 1, "{:.3f}", (0.38, 0.50)), ("Tracking HOTA", 2, "{:.1f}", (63.5, 66.5)),
              ("Role accuracy (%)", 3, "{:.1f}", (91, 100))]
    fig, axes = plt.subplots(1, 3, figsize=SIZE, sharey=True, gridspec_kw={"wspace": 0.12})
    y = np.arange(len(names))
    for ax, (label, col, fmt, lim) in zip(axes, panels):
        values = [d[col] for d in DETECTORS][::-1]
        colours = [BLUE if n.startswith("YOLO + both") else OTHER for n in names]
        ax.scatter(values, y, s=140, color=colours, zorder=3, edgecolor=SURFACE, linewidth=2)
        for yi, v in zip(y, values):
            ax.text(v + (lim[1] - lim[0]) * 0.03, yi, fmt.format(v), va="center", fontsize=13, color=INK2)
        ax.set_xlim(*lim)
        ax.set_title(label, fontsize=16)
        hgrid(ax)
    axes[0].set_yticks(y, names)
    title(fig, "Detection gains barely reach tracking",
          "Validation match 6, last epoch. Blue: extra matches + hue, the recipe of the final YOLO")
    fig.subplots_adjust(top=0.8)
    save(fig, out, "03-detectors")


# --- tracking results -----------------------------------------------------------------------------

def test_runs(out: Path) -> None:
    runs = [("Off-the-shelf\nbaseline", 53.4), ("Run 1\nfine-tuned YOLO", 63.0), ("Run 2\n+ ball path", 63.1),
            ("Run 3\n+ GSR matches,\nhue, jerseys", 63.4), ("Run 4\n+ RF-DETR ball,\nQwen jerseys", 64.2)]
    fig, ax = plt.subplots(figsize=SIZE)
    x = np.arange(len(runs))
    ax.bar(x, [r[1] for r in runs], width=0.5, color=[OTHER] * 4 + [BLUE])
    for xi, (_, v) in zip(x, runs):
        ax.text(xi, v + 1.2, f"{v:.1f}", ha="center", fontsize=16, color=INK, fontweight="bold" if xi == 4 else None)
    ax.axhline(87.7, color=MUTED, linewidth=1.5)
    ax.text(len(runs) - 0.5, 88.8, "Same tracker on perfect detections: 87.7", ha="right", fontsize=14, color=INK2)
    ax.set_xticks(x, [r[0] for r in runs])
    ax.set_ylim(0, 100)
    ax.set_ylabel("HOTA (test set)")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    title(fig, "Test set: +10.8 HOTA over the off-the-shelf baseline",
          "Games 7, 8 and 11 (49 clips, never trained or tuned on); all four test runs reported")
    fig.subplots_adjust(top=0.84)
    save(fig, out, "04-test-runs")


def paper(out: Path) -> None:
    rows = [  # SoccerMaster (Yang et al. 2026) Table 5, and our test runs
        ("YOLOv8 + OC-SORT", 54.6), ("YOLOv8 + StrongSORT++", 56.2), ("SoccerMaster + MOTIP", 59.1),
        ("YOLOv8 + PRTReID", 59.8), ("Ours: off-the-shelf baseline", 53.4), ("Ours: final pipeline", 64.2)]
    fig, ax = plt.subplots(figsize=SIZE)
    y = np.arange(len(rows))[::-1]
    colours = [OTHER] * 4 + [AXIS, BLUE]
    ax.barh(y, [r[1] for r in rows], height=0.5, color=colours)
    for yi, (_, v) in zip(y, rows):
        ax.text(v + 0.8, yi, f"{v:.1f}", va="center", fontsize=15, color=INK)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.axhline(1.5, color=GRID, linewidth=1)
    ax.set_xlim(0, 75)
    ax.set_xlabel("HOTA on the SoccerNet-Tracking test set")
    hgrid(ax)
    title(fig, "Ahead of the published pipelines by 4.4 HOTA",
          "SoccerMaster (Yang et al., 2026), Table 5. Our gain is detection (DetA 72.2 vs ≤ 65.2);\n"
          "PRTReID still associates better (AssA 58.6 vs 57.3)")
    fig.subplots_adjust(top=0.78)
    save(fig, out, "05-paper-comparison")


def clip_types(out: Path) -> None:
    summary = pd.read_csv("outputs/eval/test-run4/summary.csv")
    rows = summary[(summary["tracker"] == "final-v3-linked") & summary["subset"].str.startswith("action: ")]
    rows = rows.assign(action=rows["subset"].str.removeprefix("action: ")).sort_values("HOTA")
    overall = summary.loc[(summary["tracker"] == "final-v3-linked") & (summary["subset"] == "all"), "HOTA"].item()
    fig, ax = plt.subplots(figsize=SIZE)
    y = np.arange(len(rows))
    hard = rows["action"].isin(["Corner", "Shots on target", "Direct free-kick"])
    ax.barh(y, rows["HOTA"], height=0.55, color=np.where(hard, ORANGE, BLUE))
    for yi, v in zip(y, rows["HOTA"]):
        ax.text(v + 0.6, yi, f"{v:.1f}", va="center", fontsize=13, color=INK2)
    ax.axvline(overall, color=INK2, linewidth=1.5)
    ax.text(overall + 0.5, len(rows) - 0.4, f"all clips {overall:.1f}", fontsize=13, color=INK2)
    ax.set_yticks(y, rows["action"])
    ax.set_xlim(0, 80)
    ax.set_xlabel("HOTA (test run 4)")
    hgrid(ax)
    title(fig, "Crowded set pieces are hardest",
          "Test set by clip type. Lowest (orange): set pieces and shots on target; corners and free kicks are\n"
          "the most crowded clips (15 % of people overlap another, against 4 % in kick-offs)")
    fig.subplots_adjust(top=0.84)
    save(fig, out, "06-clip-types")


def roles(out: Path) -> None:
    rows = [  # recall, test run 1 against test run 4 (decisions.md, roles on the test set)
        ("Referee, game 7", 99.6, 99.5), ("Referee, game 8", 99.7, 99.2), ("Referee, game 11", 22.5, 83.1),
        ("Goalkeeper, all games", 62.9, 78.6)]
    fig, ax = plt.subplots(figsize=SIZE)
    y = np.arange(len(rows))[::-1]
    for yi, (_, before, after) in zip(y, rows):
        ax.plot([before, after], [yi, yi], color=GRID, linewidth=3, zorder=1)
    ax.scatter([r[1] for r in rows], y, s=180, color=OTHER, zorder=2, edgecolor=SURFACE, linewidth=2,
               label="Runs 1–2: three training matches")
    ax.scatter([r[2] for r in rows], y, s=180, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=2,
               label="Run 4: six matches, hue augmentation")
    for yi, (_, before, after) in zip(y, rows):
        if abs(after - before) > 5:
            ax.text(before - 2, yi + 0.18, f"{before:.1f}", ha="right", fontsize=13, color=INK2)
            ax.text(after + 2, yi + 0.18, f"{after:.1f}", ha="left", fontsize=13, color=INK)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0, 105)
    ax.set_ylim(-0.6, len(rows) - 0.2)
    ax.set_xlabel("Recall (%) on the test set")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    hgrid(ax)
    title(fig, "Extra matches and hue augmentation find game 11's referee",
          "He wears a dark green kit none of the first three training matches had")
    fig.subplots_adjust(top=0.78)
    save(fig, out, "07-roles")


def jersey_readers(out: Path) -> None:
    # SoccerNet-GSR games 2, 3 and 5, 966 labelled tracks (decisions.md, "A jersey reader that can say no number")
    qwen_both = [(46.0, 94.7), (41.3, 97.6), (37.2, 98.1), (31.9, 98.7), (26.0, 99.2), (15.2, 98.7)]
    qwen_two = [(35.3, 94.5), (31.6, 97.1), (28.6, 97.9), (24.6, 98.3), (20.0, 99.0), (12.5, 98.4)]
    florence = [(18.9, 92.0), (26.1, 75.7)]
    fig, ax = plt.subplots(figsize=SIZE)
    for points, colour, label in ((qwen_both, BLUE, "Qwen3.5-4B, one- and two-digit numbers"),
                                  (qwen_two, ORANGE, "Qwen3.5-4B, two-digit numbers"),
                                  (florence, AQUA, "Florence-2 (no probability)")):
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=colour, linewidth=0 if points is florence else 2, marker="o", markersize=10,
                markeredgecolor=SURFACE, markeredgewidth=2, label=label, solid_capstyle="round")
    ax.annotate("chosen: answer probability ≥ 0.5\n94.7 % right, 46 % of tracks", (46.0, 94.7), (37.5, 84),
                fontsize=13, color=INK, arrowprops={"arrowstyle": "-", "color": INK2, "linewidth": 1})
    ax.text(13.0, 99.8, "← stricter probability cut-off", fontsize=12, color=INK2, va="bottom")
    ax.text(18.9, 90.6, "two-digit only\n(run 3's reader)", fontsize=12, color=INK2, ha="center", va="top")
    ax.text(25.3, 75.7, "one- and two-digit", fontsize=12, color=INK2, ha="right", va="center")
    ax.set_xlim(10, 50)
    ax.set_ylim(70, 101)
    ax.set_xlabel("Labelled tracks given their right number (%)")
    ax.set_ylabel("Numbers given that are right (%)")
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    title(fig, "A reader that can answer \"no number\" more than doubles coverage",
          "SoccerNet-GSR games 2, 3 and 5 (966 tracks with a labelled number, none from our validation or "
          "test matches)")
    fig.subplots_adjust(top=0.84)
    save(fig, out, "08-jersey-readers")


# --- sustainability -------------------------------------------------------------------------------

def energy(out: Path) -> None:
    rows = [  # decisions.md, "Sustainability" (scripts/energy_report.py)
        ("Training (27 runs)", 1.86, 0.42), ("Jersey reading", 0.13, 1.96), ("Detection", 0.37, 0.10),
        ("Tracking and evaluation", 0.0, 0.41), ("Other", 0.07, 0.02)]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=SIZE, gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.75})
    y = np.arange(len(rows))[::-1]
    gpu, cpu = np.array([r[1] for r in rows]), np.array([r[2] for r in rows])
    ax.barh(y, gpu, height=0.5, color=BLUE, label="GPU (measured)")
    ax.barh(y, cpu, left=gpu + 0.02 * (gpu > 0), height=0.5, color=ORANGE, label="CPU (estimated)")
    for yi, total in zip(y, gpu + cpu):
        ax.text(total + 0.06, yi, f"{total:.2f}", va="center", fontsize=13, color=INK2)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0, 2.8)
    ax.set_xlabel("kWh")
    ax.set_title("Whole project: 5.3 kWh", fontsize=16)
    ax.legend(loc="lower right")
    hgrid(ax)

    steps = [("YOLO26m detection", 0.11), ("RF-DETR (ball only)", 0.47), ("Tracking", 0.01),
             ("Qwen jersey reading", 0.13)]
    y2 = np.arange(len(steps))[::-1]
    ax2.barh(y2, [s[1] for s in steps], height=0.5, color=AQUA)  # GPU and CPU together, so neither series colour
    for yi, (_, v) in zip(y2, steps):
        ax2.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=13, color=INK2)
    ax2.set_yticks(y2, [s[0] for s in steps])
    ax2.set_xlim(0, 0.6)
    ax2.set_xlabel("kWh per 90-minute match")
    ax2.set_title("Final pipeline: 0.73 kWh per match", fontsize=16)
    hgrid(ax2)
    title(fig, "The project used as much energy as an electric car needs for 31 km",
          "All 236 IDUN jobs; CPU at 7 W per core; cooling not included. SoccerMaster's pretraining "
          "used 340× our GPU-hours")
    fig.subplots_adjust(top=0.78)
    save(fig, out, "09-energy")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="tracking-2023 folder (labels are enough)")
    parser.add_argument("--out", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    box_heights(args.root, args.out)
    pipeline(args.out)
    detectors(args.out)
    test_runs(args.out)
    paper(args.out)
    clip_types(args.out)
    roles(args.out)
    jersey_readers(args.out)
    energy(args.out)


if __name__ == "__main__":
    main()
