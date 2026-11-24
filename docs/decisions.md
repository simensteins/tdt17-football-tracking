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

### First data exploration (3 local sequences, all from game 4)
Scripts: `scripts/explore_labels.py` (label statistics) and `scripts/show_frames.py` (annotations drawn on frames). Results must be confirmed on all 106 sequences, since these three come from one match.

- **Class balance:** players 83 % of boxes, referees 9 %, ball 6 %, goalkeepers 2 % (5 tracks).
- **The ball cannot be detected at 640 px input.** Median ball box is 13 px at 1920 wide, which becomes about 4 px at 640: 99.9 % of ball boxes fall below 8 px. At 1280 the median is about 9 px. → Train and infer at 1280 or higher, or tile the image; compare resolutions as an experiment. Side referees (linesmen) are also small: 18 % below 8 px at 640.
- **Every player and referee gap is the person leaving the picture**, not occlusion: median 5 s, up to 19 s, and they return with the same ID. 62 % of player tracks have at least one gap. → A motion-only tracker that keeps lost tracks for about 1 s will assign new IDs on return. Long-term re-identification (appearance, or position on the pitch) is needed for good association.
- **Camera motion is larger than player motion.** The camera pans mostly horizontally, median 4 px/frame and up to 43 px/frame, while players move about 2 px/frame relative to the camera. Correcting for the pan cuts the share of player boxes with IoU < 0.3 five frames apart from 43 % to 15 %. → Camera motion compensation is worth testing (e.g. BoT-SORT).
- **IoU matching does not work for the ball.** Its box overlaps its own previous box with median IoU 0.39, and 42 % of consecutive pairs fall below 0.3, with or without pan correction. → Associate the ball by distance or handle it separately from the people.
- **The ball is labelled when visible and dropped when hidden.** It is present in 98–100 % of frames; its gaps (about 17 frames) happen mid-picture, during duels.
- **There can be two balls.** In SNMOT-061 the match ball goes out of play and a ball boy supplies a new one; both are labelled, with different IDs. → Do not assume exactly one ball per frame.
- **People who are not labelled:** coaches and stewards on the sideline, ball boys, photographers, camera operators. → An off-the-shelf COCO "person" detector would count these as false positives; a fine-tuned detector learns to ignore them. Matters for any zero-shot baseline.
- **Crowding is moderate** in these clips: 3.4 % of player boxes overlap another with IoU > 0.3.

### Full label exploration (all 106 sequences, 6 matches)
Labels for all sequences were copied without images; `scripts/explore_labels.py data/tracking-2023` covers both splits. The findings from the three local sequences hold, with these corrections and additions:

- **Class balance (train):** players 82.5 %, referees 8.4 %, ball 5.6 %, goalkeepers 3.3 % (79 tracks), `other` 0.2 %.
- **Ball size confirmed:** 97 % of ball boxes are below 8 px at 640 input, 38 % at 1280. Ball heights (median 13 px) and people (median 100 px) are an order of magnitude apart; referees split into far linesmen (about 35 px) and the main referee (about 90 px).
- **Gaps confirmed:** 49 % of player tracks have a gap, median 4.4 s, up to 29 s; 99 % of them are the player leaving the picture.
- **Ball association:** median IoU between consecutive ball boxes is 0.54, and 32 % of pairs fall below 0.3. Pan correction does not help, since the ball moves on its own.
- **Crowding depends on the situation.** 9 % of people overlap another person with IoU > 0.3 overall, but 15 % in corners and free kicks against 4 % in kick-offs and clearances. → Report results per clip type (`actionClass`), not only overall.
- **Clip types are balanced across matches:** every match has about two corners, free kicks, shots, clearances and so on. The per-type breakdown is therefore not confounded by match.
- **The test matches are framed differently.** Median player height is 97–102 px in the three training matches, but 120 px in test match 7, 119 px in match 8 and 89 px in match 11. Match 8 also has faster camera pans (95th percentile 27 px/frame against 13–18 in training). → Scale augmentation matters; expect a train–test gap from framing, not only from kits and stadiums.
- **Camera cuts appear only in the test set.** In 6 test sequences (125, 133, 137, 147, 148, 150) most people jump at once in a single frame, i.e. a cut or abrupt zoom. No training sequence has one. → Motion-based association will break at these frames; worth showing as a failure case.
- **Label noise is small:** one zero-width box (SNMOT-065, dropped) and about 54 single-frame jumps where one track's box leaps elsewhere and back (largest: SNMOT-169 track 9, 934 px).
- **`other` role:** 10 tracks in 4 sequences (train 076, 166; test 133, 188), 0.2 % of boxes. In SNMOT-076 (a substitution) they are staff on the pitch: two medics escorting an injured player and the fourth official holding the substitution board. Staff on the sideline in the same frame are not labelled, so the labelling of non-players is inconsistent.
- **Decision: drop `other` from training and evaluation.** The system covers four classes: player, goalkeeper, referee, ball. Too few `other` tracks exist to learn a class, merging them into `player` would contradict the many unlabelled staff elsewhere and leak staff into player statistics, and staff are not part of the analysis. Cost: about 0.2 % of ground-truth boxes are excluded from tracking evaluation, which is stated with the results.
- **Validation split kept:** game 6 is held out. None of the training matches resembles the test framing, so the choice between them matters little.

### Scope: working alone, aiming at the two-person scope
Detection and tracking both get full experiments, not detection with a light tracking add-on. Deadline Friday 2026-11-27.

### Method: tracking-by-detection with Ultralytics YOLO
- **Detector:** Ultralytics YOLO fine-tuned from COCO weights, four classes (player, goalkeeper, referee, ball).
  - **Why:** mature, fast to train, handles high input resolution (needed for the ball), and the same library provides trackers. Time goes into experiments rather than plumbing.
  - **Rejected:** SoccerMaster and similar end-to-end models (compute far beyond reach); a DETR-only approach (slower to train, weaker on tiny objects). RT-DETR is kept as an architecture comparison.
- **Tracker:** Ultralytics ByteTrack and BoT-SORT, evaluated with TrackEval (HOTA, DetA, AssA, ID switches).
- **Splits:** train on games 4 and 9, validate on game 6, test once at the end on games 7, 8 and 11.
- **Augmentation:** scale augmentation (test framing differs); horizontal flips allowed, since team side is not a class.

### Model generation and hardware
- **YOLO26** (Ultralytics 8.4.218) is the main detector family. It is the newest generation, predicts boxes end-to-end without NMS, and targets small objects, which matters for the ball. YOLO11 is also available if a comparison is needed.
- **Jobs run on any A100, H100 or H200** (`--constraint=a100|h100|h200` in `jobs/gpu.slurm`). The first plan was A100 only, so that training times and energy would be comparable. But waiting for one GPU type cost more time than the comparison was worth, with three days left. IDUN also has P100 and V100, which are left out: they have at most 32 GB and are several times slower, and a 1280 training run would risk the 12-hour limit.
  - Accuracy (mAP, HOTA) doesn't depend on the GPU, so results stay comparable.
  - Time and energy do. Each job logs its GPU name and power draw, and training cost is reported per GPU type.
  - Inference speed is measured on one GPU type for all models, so speed comparisons stay fair.
- **Training runs in 2-hour chunks.** The 12-hour D2 jobs waited in the queue (reason `Priority`, estimated start after the deadline), while a 1-hour job started at once: Slurm fills gaps with short jobs. Each run is now submitted with `--time=02:00:00`, followed by resume jobs that continue from `last.pt` (`yolo detect train resume`). At most one epoch is lost per chunk.

### Experiment plan
Each experiment answers a question raised by the data.

| # | Experiment | Question |
|---|---|---|
| D1 | COCO-pretrained YOLO without fine-tuning (person + sports ball) | Why fine-tune? Expected: staff false positives, missed ball |
| D2 | Fine-tuned at 640 vs 1280 input | Does the ball become detectable? |
| D3 | Model size s / m / l at the best resolution | Speed vs accuracy vs energy |
| D4 | Training on every 5th vs every 25th frame | How redundant are consecutive frames? |
| D5 | RT-DETR, and tiled inference for the ball | Different architecture; dedicated small-object handling |
| T1 | ByteTrack on `det.txt` (perfect detections) vs on our detections | How much HOTA is lost to detection vs association? |
| T2 | ByteTrack vs BoT-SORT with camera motion compensation | Does pan correction help? |
| T3 | Track buffer about 1 s vs about 6 s | Do longer buffers prevent new IDs after players re-enter? |
| T4 | BoT-SORT with appearance ReID | Does appearance re-identification help on top? |
| T5 | Offline gap interpolation | What does post-match processing gain? |

Detection metrics: precision, recall, mAP@50, mAP@50:95, per class (ball separate), per test match and per clip type. Reference tracking results on the same benchmark: SoccerMaster Table 5, HOTA 54.6–59.8.

---

## Results

### D1: COCO-pretrained YOLO26m, no fine-tuning (validation match 6, 2,850 frames)
COCO classes `person` (players, goalkeepers and referees merged) and `sports ball`. Inference on an A100: 1.6 ms per image at 640, 5.8 ms at 1280.

| Input | Class | P | R | mAP@50 | mAP@50:95 |
|---|---|---|---|---|---|
| 640 | person | 0.913 | 0.873 | 0.887 | 0.496 |
| 640 | ball | 0.613 | 0.316 | 0.294 | 0.103 |
| 1280 | person | 0.889 | 0.922 | 0.904 | 0.520 |
| 1280 | ball | 0.643 | 0.377 | 0.375 | 0.134 |

- People are found well even without fine-tuning (recall 0.87–0.92), but mAP@50:95 stays around 0.5: the boxes do not match the SoccerNet annotation style tightly, and the model cannot tell players, goalkeepers and referees apart.
- The ball is the weak point: 32 % found at 640, 38 % at 1280.
- Doubling the resolution raises ball mAP@50 by 28 % (0.294 → 0.375) and person recall from 0.87 to 0.92, at 3.6× the inference time. The resolution effect is visible before any fine-tuning.

### Tracking pipeline design
- **Detect once, track many times.** `scripts/detect.py` (GPU) stores every detection down to confidence 0.01 per sequence; `scripts/track.py` (CPU) runs any tracker configuration on stored detections or on the ground truth. Tracker settings can be explored without rerunning the detector.
- **People and ball are tracked by separate tracker instances**, so the ball can never be matched to a person and each can use its own strategy.
- **Tracks report the matched detection's box**, not the tracker's Kalman-smoothed estimate, so localisation reflects the detector.
- **Evaluation:** TrackEval HOTA, CLEAR and Identity metrics on three subsets: all four classes, people only, ball only. Scoring the ground truth as a tracker gives 100 on every metric, which checks the pipeline end to end.

### T1/T3: trackers on perfect detections (validation match 6, 19 sequences)
Ground-truth boxes without IDs, so detection is perfect and only association is tested. People DetA is 99 rather than 100 because a new track is only reported once confirmed.

| Tracker (lost-track buffer) | HOTA all | HOTA people | AssA people | IDSW people | HOTA ball | AssA ball |
|---|---|---|---|---|---|---|
| ByteTrack (30 frames, 1.2 s) | 85.8 | 87.9 | 78.0 | 922 | 41.4 | 21.9 |
| ByteTrack (150 frames, 6 s) | 85.7 | 87.8 | 77.8 | 963 | 41.1 | 21.6 |
| OC-SORT (30 frames) | 86.7 | 88.7 | 79.3 | 745 | 42.5 | 22.9 |
| OC-SORT (150 frames) | 87.2 | 89.3 | 80.3 | 777 | 42.5 | 22.8 |
| OC-SORT (150) + single ball | **89.7** | 89.3 | 80.3 | 777 | **97.0** | **95.9** |

- **Association, not detection, limits people tracking:** with perfect detections AssA is still only 78–80.
- **A longer buffer barely helps (T3)**, although players are out of view for 4.4 s on median. A player re-entering the picture appears where the motion model cannot predict, especially while the camera pans, so the boxes do not overlap and the kept track cannot be re-attached. → Motion alone cannot re-identify; appearance ReID (T4) or pitch position is needed.
- **OC-SORT beats ByteTrack** by about 1 HOTA on people, fewer ID switches (745 vs 922).
- **Corners are hardest:** HOTA 77 against 90 for shots on target, matching the crowding measured in the data.
- **General trackers fail on the ball** even with perfect detections (AssA 22): IoU matching does not work for a small, fast object.
- **Decision: track the ball with a one-ball prior.** Keep the most confident ball detection per frame under one ID (`--ball single`). Ball HOTA rises from 42.5 to 97.0 on perfect detections. With real detections, false ball detections become the risk, so the confidence threshold needs tuning on validation.

---

## Open questions

| Question | How to answer it |
|---|---|
| How much does training on every n-th frame cost? | Train with different strides and compare (consecutive frames are near-identical) |
