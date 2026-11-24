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

Locally, place it under `data/` (ignored by git).
