# TDT17 Mini-Project: Football Detection and Multi-Object Tracking

Object detection and multi-object tracking of players, goalkeepers, referees and the ball in broadcast football footage, using the [SoccerNet-Tracking 2023](https://www.soccer-net.org/tasks/tracking) dataset.

Part of TDT17 Visual Intelligence at NTNU (autumn 2026), Task 4, Option 3.

## Scope

- **Detection:** evaluated with precision, recall, mAP@50 and mAP@50:95.
- **Tracking:** associating detections across frames, evaluated with HOTA, DetA and AssA.
- **Sustainability:** compute and energy use are logged for training and inference.

## Data

The dataset is not included in this repository. On IDUN it is available at:

```
/cluster/projects/vc/courses/TDT17/2026/3_fotball/SoccerNet-Tracking-2023/tracking-2023/
```

Locally, place the sequences under `data/` (ignored by git):

```
data/tracking-2023/
├── train/SNMOT-xxx/{seqinfo.ini, gameinfo.ini, det/, gt/, img1/}
└── test/SNMOT-xxx/...
```

## Running on IDUN

One-time setup on a login node:

```bash
module load Python/3.12.3-GCCcore-13.3.0
python -m venv ~/venvs/tdt17 && source ~/venvs/tdt17/bin/activate
pip install torch==2.14.1 torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
mkdir -p data && ln -s /cluster/projects/vc/courses/TDT17/2026/3_fotball/SoccerNet-Tracking-2023/tracking-2023 data/tracking-2023
```

Build the YOLO dataset (symlinks to the course images, every 5th frame):

```bash
python scripts/make_yolo_dataset.py data/tracking-2023 --out /cluster/work/$USER/yolo/stride5
```

GPU work goes through `jobs/gpu.slurm`, which runs any command and logs GPU power draw to `logs/`:

```bash
sbatch --job-name=<name> jobs/gpu.slurm yolo detect train data=... model=... imgsz=1280
```

## Repository layout

| Path | Contents |
|---|---|
| `football/` | Dataset loading shared by all scripts |
| `scripts/` | Data exploration and dataset conversion |
| `jobs/` | Slurm job scripts for IDUN |
| `docs/decisions.md` | Decisions, findings and the experiment plan |
