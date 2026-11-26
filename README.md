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

## Tracking

Detect once on a GPU, then track and evaluate on CPU:

```bash
python scripts/detect.py data/tracking-2023 --split val --model <weights.pt> --imgsz 1280 --out <detections dir>
python scripts/track.py data/tracking-2023 --split val --detections <detections dir or 'gt'> \
    --tracker botsort.yaml --set track_buffer=150 fuse_score=false match_thresh=0.95 new_track_thresh=0.5 \
    --merge-iou 0.7 --ball path --images --interpolate 10 --out <tracks dir>
python scripts/eval_tracking.py data/tracking-2023 <tracks dir> [<tracks dir> ...] --split val --by-action
```

`--detections gt` uses the ground-truth boxes without IDs, i.e. a perfect detector.

`--ball-detections <dir>` takes the ball candidates from a second detector (people still from `--detections`); the final pipeline uses RF-DETR's balls with `--ball-path tau=0.3`.

Tracks of a player who left the picture and came back can be joined by jersey number (Qwen3.5-4B or Florence-2 reads the numbers, then tracks are linked by role, team and number):

```bash
python scripts/read_jerseys.py data/tracking-2023 <tracks dir> --split val --reader qwen --per-track 10 --out <tracks dir>/jerseys.csv
python scripts/link_tracks.py data/tracking-2023 <tracks dir> <tracks dir>/jerseys.csv --split val \
    --single-digits --min-conf 0.5 --out <linked tracks dir>
```

`scripts/bench_jersey_readers.py` measures a reader against the jersey numbers labelled in SoccerNet-GSR (see below).

## Extra models and data

RF-DETR and Florence-2 need newer libraries than Ultralytics, so they run in a second environment:

```bash
python -m venv ~/venvs/rfdetr && source ~/venvs/rfdetr/bin/activate
pip install torch==2.14.1 torchvision --index-url https://download.pytorch.org/whl/cu126
pip install "rfdetr[train,loggers]==1.12.1" opencv-python-headless pandas pyyaml
python scripts/train_rfdetr.py /cluster/work/$USER/yolo/stride5 --size medium --resolution 1280 --out <run dir>
```

`scripts/detect.py` runs an RF-DETR checkpoint (`.pth`) the same way as a YOLO model.

Extra training matches come from the `valid` split of [SoccerNet-GSR](https://huggingface.co/datasets/SoccerNet/SN-GSR-2025) (games 2, 3 and 5, not in the course data):

```bash
python scripts/make_gsr_dataset.py <path to valid.zip> --out /cluster/work/$USER/yolo/gsr-stride5
```

RF-DETR reads one folder per split, so the course and GSR frames are combined with symlinks first. `--hue 0.5` needs `pip install "rfdetr[augment]==1.12.1"`:

```bash
Y=/cluster/work/$USER/yolo
python scripts/merge_yolo_datasets.py --out $Y/rfdetr-gsr \
    --train $Y/stride5/images/train $Y/gsr-stride5/images/train --val $Y/stride5/images/val
python scripts/train_rfdetr.py $Y/rfdetr-gsr --size medium --resolution 1280 --epochs 2 --hue 0.5 --out <run dir>
```

## Repository layout

| Path | Contents |
|---|---|
| `football/` | Dataset loading shared by all scripts |
| `scripts/` | Data exploration, dataset conversion, detection, tracking and evaluation |
| `jobs/` | Slurm job scripts for IDUN |
| `docs/decisions.md` | Decisions, findings and the experiment plan |
