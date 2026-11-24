# Decision log

Decisions made during the project and the reasoning behind them. Source material for the *Approach / Strategy* part of the presentation.

---

## 2026-11-24

### Project option
Option 3: football object detection and multi-object tracking on SoccerNet-Tracking 2023.

### Use case: post-match analysis
The system is designed for post-match analysis (tactical reports, player statistics), not live use.

- **Why:** the task describes tactical analysis, player statistics and match understanding, which are offline uses. A customer for these values correct numbers over low latency.
- **Consequences:**
  - No latency budget. Larger models, higher input resolution and tiled inference are allowed.
  - Offline tracking steps that use future frames (gap interpolation, tracklet merging, smoothing) are allowed, but must be labelled as such since they would not work live.
  - Inference speed is still measured per model, to show what a real-time variant would cost in accuracy and to support the sustainability discussion.

### What "good detection" means here
mAP alone does not capture the downstream need. Priorities for this use case:

1. **Recall over precision.** Missed detections break tracks and cause identity switches; low-confidence false positives can be filtered by the tracker.
2. **Tight boxes.** Player positions come from the bottom of the box, so localisation matters: mAP@50:95, not only mAP@50.
3. **Roles.** Player, goalkeeper and referee must be separated, since referees are excluded from player statistics.
4. **Ball reported separately.** It is the most important object for match understanding and the hardest (median 15×14 px at 1080p). A class-averaged mAP would hide its performance.

### Compute: IDUN, data read in place
- Training and evaluation run on IDUN GPUs (account `share-ie-idi`). The dataset is read directly from the course folder instead of being copied.
- Code is developed locally against one sequence (SNMOT-060), pushed to GitHub and pulled on IDUN.
- **Environment:** Python module + venv. PyTorch is pinned to `2.14.1+cu126`, because the default `cu132` build has dropped support for V100 GPUs (compute capability 7.0). The `cu126` build runs on P100, V100, A100 and H100, so jobs are not restricted to the busiest GPU types.

### Dataset findings that shape the experiments
- **`det/det.txt` equals the ground truth in all 106 sequences** (IDs removed, confidence 1). There are no public detections, so the detector is entirely ours.
  - This enables an oracle experiment: tracking on `det.txt` isolates association performance. Comparing it with tracking on our own detections measures what detection errors cost in HOTA.
- **Classes are not in `gt.txt`.** Each track ID is mapped to a role in `gameinfo.ini`, with inconsistent spelling (`goalkeeper` / `goalkeepers`) that must be normalised.
- **Only six matches, no overlap between train and test.** Train: games 4, 6, 9 (57 sequences). Test: games 7, 8, 11 (49 sequences). The test set measures generalisation to unseen matches.
- **Validation split by match, not by sequence.** A random split would put clips from the same match (same kits, stadium, camera) on both sides and inflate validation scores. Proposed: hold out game 6 for validation, train on games 4 and 9.
- **Frame redundancy.** At 25 fps consecutive frames are near-identical. Training on every n-th frame should cost little accuracy and save compute; to be tested.

### Related work: SoccerMaster (Yang et al., 2026)
- Reports tracking baselines on SoccerNet-Tracking: YOLOv8 + OC-SORT (HOTA 54.6), + StrongSORT++ (56.2), + PRTReID (59.8), SoccerMaster + MOTIP (59.1).
- DetA is similar across methods (61–65) while AssA ranges from 47 to 59, so association, not detection, separates the trackers. Appearance-based ReID gives the largest gain.
- Their pretraining used 16 H800 GPUs for about 9 days, yet only matches a YOLOv8 + ReID pipeline on HOTA. A useful reference point for the sustainability discussion.
- They leave out the ball and report goalkeeper/player confusion, both of which we address.

---

## Open questions for data exploration

| Question | Decision it drives |
|---|---|
| Class counts across all sequences | Imbalance handling; whether goalkeepers are learnable as a class |
| Box sizes per class | Input resolution (640 / 1280 / tiling) |
| Ball annotation frequency, size, position | Special handling for the ball |
| Box overlap and crowd density | NMS settings, occlusion difficulty |
| Track lengths, gap lengths, re-entries | Tracker buffer length; need for ReID |
| Camera motion between frames | Need for camera motion compensation |
| Differences between matches | Expected generalisation to the test matches |
| Visual check of annotations | Unlabelled people (coaches, substitutes) as false positives; box tightness; ball labelled when hidden |
| Similarity of consecutive frames | Frame subsampling for training |
| Match situation per clip (`actionClass`) | Whether some situations are harder |
