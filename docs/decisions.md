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
- **Prior experience:** I worked with both YOLO and RT-DETR in my master's thesis. That is why I could pick and set up both families quickly: a YOLO detector as the main pipeline, and a DETR-style transformer detector (RF-DETR, a successor of RT-DETR built on a DINOv2 backbone) as the comparison.
- **Jobs run on any A100, H100 or H200** (`--constraint=a100|h100|h200` in `jobs/gpu.slurm`). The first plan was A100 only, so that training times and energy would be comparable. But waiting for one GPU type cost more time than the comparison was worth, with three days left. IDUN also has P100 and V100, which are left out: they have at most 32 GB and are several times slower, and a 1280 training run would risk the 12-hour limit.
  - Accuracy (mAP, HOTA) doesn't depend on the GPU, so results stay comparable.
  - Time and energy do. Each job logs its GPU name and power draw, and training cost is reported per GPU type.
  - Inference speed is measured on one GPU type for all models, so speed comparisons stay fair.
- **GPU job time limits come from measured runtimes plus about 25 %.** The 12-hour D2 jobs waited in the queue (reason `Priority`, estimated start after the deadline), while 1-hour jobs started at once: Slurm only fits low-priority jobs into gaps they fit in, so a generous limit means hours of waiting. Unknown runtimes are measured with a short job first. CPU jobs start at once anyway, so they get more room: a tracking job sized at 8 min + 25 % landed on a slower node and timed out one clip before the end.
- **No chained resume jobs.** D2 was given follow-up jobs (`yolo detect train resume`) in case it hit its limit. It finished early, and the follow-ups did not stop but trained a new model on Ultralytics' built-in 4-image demo dataset for 100 epochs (about 3 minutes each, wasted). Runs are now sized to finish within one job.

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

### D2: fine-tuned YOLO26m at 640 and 1280 (validation match 6)
Trained on games 4 and 9 (5,700 frames, every 5th), 50 epochs planned with early stopping after 15 epochs without improvement. Ultralytics chose AdamW with learning rate 0.00125. Scores are for the best epoch.

| Input | Class | P | R | mAP@50 | mAP@50:95 |
|---|---|---|---|---|---|
| 640 | all | 0.718 | 0.607 | 0.638 | 0.345 |
| 640 | player | 0.906 | 0.878 | 0.930 | 0.541 |
| 640 | goalkeeper | 0.453 | 0.488 | 0.448 | 0.235 |
| 640 | referee | 0.873 | 0.823 | 0.899 | 0.506 |
| 640 | ball | 0.638 | 0.240 | 0.275 | 0.099 |
| 1280 | all | 0.679 | 0.685 | 0.651 | 0.362 |
| 1280 | player | 0.859 | 0.937 | 0.952 | 0.580 |
| 1280 | goalkeeper | 0.195 | 0.609 | 0.276 | 0.162 |
| 1280 | referee | 0.906 | 0.862 | 0.940 | 0.548 |
| 1280 | ball | 0.758 | 0.333 | 0.437 | 0.158 |

| Run | GPU | Time | Mean power | GPU energy |
|---|---|---|---|---|
| 640 | A100 40 GB | 12.8 min (16 epochs, 48 s each) | 168 W | 0.04 kWh |
| 1280 | H100 80 GB | 18.4 min (16 epochs, 69 s each) | 493 W | 0.16 kWh |

- **The best epoch was the first one in both runs**, so both stopped after 16 epochs. Training loss fell steadily, but validation mAP@50:95 swung between 0.28 and 0.36 with no upward trend. Training on two matches overfits to their kits and stadiums, and the high learning rate makes the validation scores unstable. → The fine-tuning recipe needs fixing before the D2 comparison means much (lower learning rate, frozen backbone).
- **1280 beats 640 for the ball** even so: mAP@50 0.437 against 0.275, recall 0.33 against 0.24. The resolution finding from D1 holds.
- **Players and referees are learned; goalkeepers are not.** Goalkeepers wear a different kit in every match, and two training matches show only four goalkeeper kits. At 1280, goalkeeper precision is 0.195: other people are called goalkeepers.
- **The A100 ran at 168 W of its 250 W during training at 640**, so training was held back by image loading on the CPU, not by the GPU. Full-HD JPEGs are decoded four at a time for every mosaic image.

### D2b/D2c: gentler fine-tuning at 1280
Same data, 15 epochs without early stopping, AdamW at learning rate 0.0002 (about 6× lower than D2), mosaic off for the last 5 epochs. D2c also freezes the first 10 layers (the backbone).

| Run | GPU | Best epoch | mAP@50:95 all | Player | Goalkeeper | Referee | Ball mAP@50 | Ball recall | Time | GPU energy |
|---|---|---|---|---|---|---|---|---|---|---|
| D2 (lr 0.00125) | H100 | 1 | 0.362 | 0.580 | 0.162 | 0.548 | 0.437 | 0.33 | 18 min | 0.16 kWh |
| D2b (lr 0.0002) | H100 | 2 | **0.407** | 0.597 | **0.311** | 0.548 | 0.452 | 0.41 | 17 min | 0.15 kWh |
| D2c (lr 0.0002, frozen backbone) | A100 | 8 | 0.401 | **0.614** | 0.251 | **0.581** | **0.459** | **0.48** | 27 min | 0.10 kWh |

Validation mAP@50:95 per epoch:
- D2b: 0.357 0.408 0.351 0.357 0.357 0.329 0.368 0.386 0.361 0.359 0.372 0.372 0.375 0.371 0.369
- D2c: 0.388 0.389 0.380 0.389 0.391 0.385 0.380 0.401 0.393 0.398 0.395 0.391 0.401 0.398 0.397

- **The lower learning rate helps**: +0.045 mAP@50:95 over D2. But D2b still peaks at epoch 2 and swings by up to 0.08 between epochs, so its best score is partly luck in which epoch was picked.
- **Freezing the backbone makes training stable.** D2c stays within 0.380–0.401 and keeps improving slowly. Its last epoch (0.397) is as good as its best, while D2b's last epoch (0.369) is well below its best. With two training matches, keeping the COCO features fixed stops the model from overfitting to those matches. → **D2c is the detector going forward.**
- **Freezing is also cheaper:** about 0.10 kWh on an A100 against 0.15 kWh for D2b on an H100 (different GPUs, so only indicative).
- Best epochs are chosen on the validation match, so validation scores are slightly optimistic. The test set, used once at the end, gives the unbiased number.

### D3: larger model, YOLO26l at 1280 (D2c recipe, 8 epochs)
Halved to 8 epochs: D2c reached its best at epoch 8 and gained nothing after, and shorter jobs start sooner in the queue.

| Run | GPU | mAP@50:95 all | Player | Goalkeeper | Referee | Ball mAP@50 | Ball recall | Time |
|---|---|---|---|---|---|---|---|---|
| D2c, YOLO26m (15 epochs) | A100 40 GB | 0.401 | 0.614 | 0.251 | 0.581 | 0.459 | 0.48 | 27 min |
| D3, YOLO26l (8 epochs) | A100 80 GB | 0.403 | 0.608 | 0.259 | 0.576 | 0.463 | 0.45 | 16 min |

Validation mAP@50:95 per epoch: 0.360 0.390 0.394 0.395 0.394 0.403 0.398 0.391.

- **The larger model gives no gain** (0.403 against 0.401, within epoch-to-epoch noise). The limit is the training data (two matches), not model capacity. → **YOLO26m stays the detector**; YOLO26l would cost about 1.3× the compute per image for nothing. Training time barely differed, because training is limited by image loading, not by the GPU.

### D2d: higher resolution, YOLO26m at 1600 (D2c recipe, 8 epochs)
| Run | mAP@50:95 all | Goalkeeper | Ball mAP@50 | Ball mAP@50:95 | Ball recall |
|---|---|---|---|---|---|
| D2c at 1280 | **0.401** | **0.251** | 0.459 | 0.159 | **0.48** |
| D2d at 1600 | 0.390 | 0.203 | **0.505** | **0.170** | 0.47 |

- **1600 helps the ball's average precision a little (+0.046 mAP@50) but finds no more balls** (recall 0.47 against 0.48), and the other classes get slightly worse. At 1.56× the compute per image it is not worth it. → **1280 stays.** The jump that mattered was 640 → 1280 (ball mAP@50 0.275 → 0.437 in D2).
- Took 19 min on an A100 80 GB (SXM, a faster card than the PCIe A100s).

### D4: every frame instead of every 5th (D2c recipe, same compute)
28,500 training frames (stride 1) for 3 epochs, against D2c's 5,700 frames (stride 5) for 15 epochs: the same number of training steps. Warmup (0.6 epochs) and the final mosaic-off phase (1 epoch) are scaled to the same share of training.

| Run | Frames | Epochs | mAP@50:95 all | Goalkeeper | Ball mAP@50 | Ball recall | Time (A100 40 GB) |
|---|---|---|---|---|---|---|---|
| D2c | every 5th (5,700) | 15 | **0.401** | **0.251** | 0.459 | 0.48 | 27 min |
| D4 | every frame (28,500) | 3 | 0.389 | 0.193 | **0.487** | 0.49 | 26 min |

Validation mAP@50:95 per epoch: 0.388 0.389 0.377.

- **Five times the frames gives no gain** for the same compute: consecutive frames at 25 fps are near-identical, so seeing different frames is worth no more than seeing the same frames again. This settles the frame-redundancy question from the data analysis.
- **Sustainability:** training on every 5th frame costs nothing in accuracy and needs a fifth of the images to store, read and decode. The diversity that matters is across matches (kits, stadiums, referees), not across frames.

**Frame stride curve** (same recipe and the same number of training steps; validation only after the last epoch, so all runs are compared on their last epoch; datasets built with `make_yolo_dataset.py --stride`):

| Stride | Training frames | Epochs | mAP@50:95 (last epoch) | Goalkeeper | Ball mAP@50 | Time (A100 40 GB) |
|---|---|---|---|---|---|---|
| 1 (D4) | 28,500 | 3 | 0.377 | 0.193 | 0.487 | 26 min |
| 5 (D2c) | 5,700 | 15 | 0.397 | 0.251 | 0.459 | 27 min |
| **10** | **2,850** | **30** | **0.406** | **0.274** | **0.495** | 22 min |
| 25 | 1,140 | 75 | 0.352 | 0.165 | 0.442 | 23 min |

- **Strides 5 and 10 are equally good** (within the ±0.01 epoch noise), stride 1 is slightly worse, and stride 25 is clearly worse: 1,140 frames seen 75 times overfit.
- **A tenth of the footage is enough** for the best detector at this compute. Data volume beyond a few thousand frames from the same matches buys nothing, which supports adding matches (GSR) rather than frames.

### D2e: two-stage fine-tuning (D2c, then all layers trainable)
Continued from D2c's last epoch with the backbone unfrozen, learning rate 0.00002 (10× lower), 4 epochs, no warmup.

Validation mAP@50:95 per epoch: **0.412** 0.405 0.392 0.398 (D2c: 0.401). Goalkeeper 0.282, ball mAP@50 0.475 at the best epoch. 9 min on an A100.

- **The first unfrozen epoch is slightly better (+0.011), then the model overfits again**, even at this small learning rate. The gain is within D2c's epoch-to-epoch spread (0.380–0.401), and the final detector has no validation match left to pick that one epoch. → **Not used.** The detector experiments end here: larger model (D3), higher resolution (D2d), more frames (D4) and unfreezing (D2e) all land within ±0.012 of D2c. With two or three training matches, the data, not the model or the recipe, is the limit.

### D5: RF-DETR (DINOv2 backbone) instead of YOLO
**Why:** every YOLO variant landed within ±0.012 mAP of D2c, which points at the data (two training matches) as the limit. RF-DETR's DINOv2 backbone was pretrained on far more varied images than YOLO's COCO weights, and RF-DETR is built for transfer to new domains with little data. If better pretraining helps, unseen kits (goalkeepers, game 11's referee) should benefit most.

**Setup:** `rfdetr` 1.12.1 in its own environment (`~/venvs/rfdetr`, same PyTorch 2.14.1 + CUDA 12.6 build), RF-DETR-M (35.3 M parameters), trained on the same YOLO-format dataset (games 4 and 9, every 5th frame) with `scripts/train_rfdetr.py`.
- Input 1280 × 1280: RF-DETR squashes frames to a square without keeping the aspect ratio, so a 16 px ball becomes about 11 × 19 px.
- Its default scale jitter crops frames to 384–600 px first, which would shrink the ball to 2–3 px, so it is turned off.
- Constant learning rate 1e-4 (encoder 1.5e-4), EMA weights, effective batch 16.

**First epoch (measuring run, batch 2, A100 40 GB, 6.5 min per epoch, 8.4 GB GPU memory):**

| Validation match | mAP@50:95 | Player | Goalkeeper | Referee | Ball | Ball recall |
|---|---|---|---|---|---|---|
| YOLO26m D2c (15 epochs) | 0.401 | 0.614 | 0.251 | 0.581 | 0.159 | 0.48 |
| RF-DETR-M (1 epoch) | **0.431** | 0.588 | **0.406** | 0.549 | **0.179** | **0.53** |

- **After one epoch RF-DETR is ahead overall, and far ahead on goalkeepers** (AP 0.41 against 0.25), as the pretraining hypothesis predicts.
- Caveat: RF-DETR and Ultralytics compute mAP with different code, so the numbers are close but not identical in definition. The deciding comparison is tracking HOTA on validation with the same tracker.

**Tracking with the one-epoch model** (validation; BoT-SORT + GMC, merge 0.7, ball path, gaps ≤ 10 filled). Detection on the 14,250 validation frames: 39.7 ms model time per frame on an A100 (batch 8), against 5.5 ms for YOLO26m. RF-DETR keeps its 300 top queries per frame, so detection files are large (about 280 boxes per frame at confidence ≥ 0.01).

| Detector | Tracker thresholds | HOTA all | HOTA people | AssA people | IDSW | HOTA ball | Corners |
|---|---|---|---|---|---|---|---|
| YOLO26m D2c (15 epochs) | tuned for YOLO | 64.1 | 65.4 | **57.5** | **609** | 40.1 | **57.7** |
| RF-DETR-M (1 epoch) | tuned for YOLO | 63.8 | 64.9 | 56.3 | 787 | 43.6 | 53.5 |
| RF-DETR-M (1 epoch) | `match_thresh=0.95 new_track_thresh=0.6 track_high_thresh=0.4` | **64.3** | **65.5** | 57.2 | 650 | **43.6** | 56.6 |

- **RF-DETR detects better** (DetA 73.0 against 72.1) and **finds the ball much better** (+3.5 ball HOTA), but its tracks break more often with YOLO's tracker settings. Re-tuned thresholds (found with OC-SORT on the laptop, then confirmed with BoT-SORT) close the gap on people.
- After one epoch, the overall gain is small (+0.2) and comes from the ball. The six-epoch run decides.

**Six epochs** (same data and settings, A100, 39 min, 0.16 kWh GPU energy). Validation mAP@50:95 per epoch (EMA weights): 0.431 0.443 0.446 0.445 0.442 0.448.

| Epoch | All | Player | Goalkeeper | Referee | Ball |
|---|---|---|---|---|---|
| 1 | 0.431 | 0.584 | 0.418 | 0.546 | 0.176 |
| 6 (best) | **0.448** | **0.608** | 0.405 | **0.580** | **0.198** |

- **The best detector on mAP** (YOLO D2f: 0.434, D2c: 0.401), and it gains slowly but steadily. Players and referees improve; goalkeepers stay around 0.41.

Tracking on game 6, best checkpoint, with the chosen pipeline (merge 0.7, ball path, gaps ≤ 10):

| Detector | Tracker thresholds | HOTA | People | Ball | DetA | AssA | IDSW | Role accuracy | Goalkeeper recall / precision |
|---|---|---|---|---|---|---|---|---|---|
| YOLO D2c | YOLO's | 64.1 | 65.4 | 40.1 | 72.1 | 57.1 | 609 | 96.7 % | 0.39 / 0.66 |
| RF-DETR, 6 epochs | YOLO's | 65.4 | 66.6 | **44.6** | 73.9 | 58.0 | 726 | **97.9 %** | **0.54** / 0.91 |
| RF-DETR, 6 epochs | RF-DETR's (`match 0.95, new 0.6, high 0.4`) | **65.8** | **67.0** | **44.6** | **74.1** | **58.5** | **594** | **97.9 %** | **0.54 / 0.92** |

- **The first detector change that clearly carries over to tracking:** +1.7 HOTA over D2c, from better detection (+2.0 DetA) and better association (+1.4 AssA, fewer ID switches). The ball gains +4.5.
- **Roles are the best of all runs** too: 97.9 %, goalkeeper precision 0.92.
- The thresholds re-tuned after one epoch still help (+0.4, 132 fewer ID switches).
- RF-DETR costs about 7× YOLO's model time per frame, which weighs against it in the sustainability discussion unless the gain grows.

### Extra data: three more matches from SoccerNet-GSR
**Why:** every detector variant pointed at the number of training matches as the limit (two kits per match, few goalkeepers and referees; game 11's referee kit was never seen).

**Is extra data allowed?** The task description does not mention extra training data. It allows open-source resources and pretrained weights, provided all external resources are acknowledged and documented (Section 1.3). Extra data is treated the same way: documented here and credited.

**Source:** SoccerNet-Tracking has 200 clips from 2019 Swiss Super League matches, split into train 57, test 49, challenge 58 and 37 future clips. The course data is the 106 train and test clips from 6 matches. SoccerNet-GSR (Somers et al., CVPRW 2024; [Hugging Face SN-GSR-2025](https://huggingface.co/datasets/SoccerNet/SN-GSR-2025), GPL-3.0, no gating) re-releases all 200 clips with labels.
- Checked before downloading, by reading only the label headers from the remote zips:

  | GSR split | Labelled clips | Game IDs | Relation to course data |
  |---|---|---|---|
  | train | 57 | 4, 6, 9 | identical to our training clips (SNGS-060 = SNMOT-060) |
  | **valid** | **58** | **2, 3, 5** | **three new matches** |
  | test | 49 | 7, 8, 11 | identical to our test clips, never used for training |
  | challenge | 0 | | images only, labels withheld |
- Only the valid split is used (11.2 GB). `scripts/make_gsr_dataset.py` refuses any clip from our validation or test matches.
- Labels (v1.3) cover player, goalkeeper, referee, **ball** and other, with the same box conventions as SoccerNet-Tracking. The GSR paper says the ball is ignored in its benchmark, but the files do contain ball boxes (716 in one clip), so no borrowed labels are needed.
- Every 5th frame, `other` dropped: **8,700 frames** with 123,928 player, 12,381 referee, 5,203 goalkeeper and 8,179 ball boxes. Spot-checked visually (new stadium, sun and hard shadows, referees in dark kits).

**Experiment D6:** the D2c recipe on games 4 and 9 plus GSR games 2, 3 and 5 (14,400 frames), with the same compute as D2c (6 epochs × 14,400 ≈ 15 × 5,700 images), warmup and mosaic-off phase scaled to the same share. Validation on game 6 as always.

| Run | GPU | Best epoch | mAP@50:95 all | Player | Goalkeeper | Referee | Ball mAP@50 | Recall | Time | GPU energy |
|---|---|---|---|---|---|---|---|---|---|---|
| D2c (2 matches) | A100 | 8 | 0.401 | **0.614** | 0.251 | **0.581** | 0.459 | 0.68 | 27 min | 0.10 kWh |
| D6 (5 matches) | A100 | 1 | **0.425** | 0.607 | **0.342** | 0.573 | **0.472** | **0.73** | 28 min | 0.09 kWh |

Validation mAP@50:95 per epoch:
- D6: 0.425 0.410 0.390 0.412 0.401 0.407 (mean 0.408, last 0.407)
- D2c: mean 0.392, last 0.397

- **More matches help detection overall.** Every D6 epoch is at or above D2c's mean, and recall goes up on every epoch (0.71–0.76 against 0.67–0.69), so fewer people are missed.
- **The goalkeeper gain does not last.** At the best epoch (the first) goalkeepers gain +0.09, but at the last epoch, which the final detector would use, they fall to 0.223 (D2c: 0.251). Players (0.618), referees (0.588) and the ball (mAP@50 0.536) are all up at the last epoch.

Tracking with the chosen settings on game 6, last epoch of each (D2c re-run in the same job):

| Detector | HOTA | People | Ball | DetA | AssA | IDSW | Role accuracy | Goalkeeper recall / precision |
|---|---|---|---|---|---|---|---|---|
| D2c | 64.1 | 65.4 | 40.1 | 72.1 | 57.1 | 609 | **96.7 %** | 0.39 / **0.66** |
| D6 | **64.8** | **66.0** | **42.8** | **73.2** | **57.5** | **597** | 92.2 % | **0.66** / 0.29 |

- **Tracking improves (+0.7 HOTA), mostly through the ball (+2.7).**
- **But roles get worse:** 13,123 player boxes (7 %) are called goalkeepers, against 1,512 with D2c. The model now finds goalkeepers but also sees them where there are none. The GSR matches have different goalkeeper kits, and with only one goalkeeper per team per match, colour is an unreliable cue. → Combined with hue augmentation in D7 (below).

### D2f: strong hue augmentation
**Why:** the failures are unseen kit colours. Ultralytics' default augmentation already uses horizontal flips, mosaic, scaling, translation and colour jitter (`hsv_h=0.015`, `hsv_s=0.7`, `hsv_v=0.4`), so mirroring was on in every run. Shifting all hues of an image together (`hsv_h=0.5`, up to half the colour wheel) turns kits into colours never seen, so the model cannot rely on "referee = yellow" and has to use the referee being dressed unlike both teams. Same recipe as D2c otherwise.

| Run | GPU | Best epoch | mAP@50:95 all | Player | Goalkeeper | Referee | Ball mAP@50 | Recall | Time | GPU energy |
|---|---|---|---|---|---|---|---|---|---|---|
| D2c | A100 | 8 | 0.401 | **0.614** | 0.251 | **0.581** | 0.459 | 0.68 | 27 min | 0.10 kWh |
| D2f (`hsv_h=0.5`) | A100 | 13 | **0.434** | 0.611 | **0.384** | 0.564 | **0.490** | **0.72** | 23 min | 0.13 kWh |

Validation mAP@50:95 per epoch: 0.380 0.400 0.421 0.418 0.431 0.416 0.419 0.429 0.430 0.432 0.432 0.426 0.434 0.428 0.431 (last 0.431, D2c 0.397).

- **The largest single gain of all detector experiments** (+0.033 at the best epoch, +0.034 at the last), and as stable as D2c from epoch 5 on.
- **It is the goalkeepers:** +0.13 mAP@50:95, precision 0.77 and recall 0.54 against D2c's 0.50 and 0.41. With colours scrambled, the model has to find the goalkeeper by position, posture and gloves rather than by a kit colour learnt from two matches. Referees lose a little (−0.017): their kit colour was a useful cue.

Tracking with the chosen settings on game 6 (last epochs):

| Detector | HOTA | People | Ball | DetA | AssA | IDSW | Role accuracy | Goalkeeper recall / precision |
|---|---|---|---|---|---|---|---|---|
| D2c | 64.1 | **65.4** | 40.1 | 72.1 | **57.1** | **609** | 96.7 % | 0.39 / 0.66 |
| D2f | 64.1 | 65.3 | **40.9** | **72.5** | 56.7 | 624 | **97.1 %** | **0.51 / 0.87** |

- **The mAP gain does not show in HOTA.** HOTA treats all people as one class, while D2f's gain is in telling goalkeepers from players. It shows up where it belongs: in role accuracy, the best of all runs, with goalkeeper precision 0.87 against 0.66.
- For post-match analysis this matters: a player counted as a goalkeeper is lost from the player statistics.

### D7: five matches and hue augmentation
D6's data with D2f's `hsv_h=0.5`, same compute as D6 (6 epochs). A100, 25 min, 0.09 kWh GPU energy.

Validation mAP@50:95 per epoch: 0.444 0.455 0.467 0.469 0.472 0.474. Last epoch: player 0.624, goalkeeper 0.487, referee 0.580, ball mAP@50 0.533.

Tracking with the chosen settings on game 6 (last epochs; RF-DETR from D5 for comparison):

| Detector | HOTA | People | Ball | DetA | AssA | IDSW | Role accuracy | Goalkeeper recall / precision | Model time per frame |
|---|---|---|---|---|---|---|---|---|---|
| D2c | 64.1 | 65.4 | 40.1 | 72.1 | 57.1 | 609 | 96.7 % | 0.39 / 0.66 | 5.5 ms |
| D6 (5 matches) | 64.8 | 66.0 | 42.8 | 73.2 | 57.5 | 597 | 92.2 % | 0.66 / 0.29 | 5.5 ms |
| D2f (hue) | 64.1 | 65.3 | 40.9 | 72.5 | 56.7 | 624 | 97.1 % | 0.51 / 0.87 | 5.5 ms |
| **D7 (5 matches + hue)** | 65.5 | 66.7 | 43.2 | 73.4 | **58.6** | **538** | **98.2 %** | **0.69 / 0.93** | 5.5 ms |
| RF-DETR, 6 epochs (2 matches) | **65.8** | **67.0** | **44.6** | **74.1** | 58.5 | 594 | 97.9 % | 0.54 / 0.92 | 40 ms |

- **The two changes fix each other's weakness.** More matches gave better detection but confused goalkeepers; hue augmentation stops the model from learning goalkeepers by kit colour, so the extra matches now help roles too. D7 has the best mAP (0.474), the fewest ID switches and the best roles of all runs.
- **D7 almost reaches RF-DETR (−0.3 HOTA) at a seventh of the model time.** RF-DETR has not had either change yet.

### Final detector, second version: all six non-test matches and hue augmentation
- **Chosen after D7:** extra matches and hue augmentation are the only YOLO changes that improved tracking. Both are applied to RF-DETR as well. The other YOLO findings do not transfer or are already in place: RF-DETR trains at a constant low learning rate and was stable without freezing (0.442–0.448 over epochs 2–6), and stride, resolution and model size made no difference.
- **Hue for RF-DETR** is Albumentations' `HueSaturationValue` with a shift of up to ±90 OpenCV hue units, i.e. half the colour wheel, always on, like `hsv_h=0.5` (`scripts/train_rfdetr.py --hue 0.5`). RF-DETR's default horizontal flip is kept.
- **2 epochs instead of 6**, about the same training steps as D5's six epochs (28,800 and 34,500 images against 34,200). RF-DETR gained little after its second epoch. A first submission at 4 epochs (80–90 min limits) was estimated to start in 10 days; jobs under 50 minutes start within minutes on the busy GPU queue.
- **Three runs, in parallel:**
  - `rfdetr-m-gsr-hue`: games 4 and 9 plus GSR games 2, 3 and 5 (14,400 frames), validation on game 6. This checks on validation that the changes help RF-DETR, as they did YOLO, before the final model is used.
  - `rfdetr-m-all-hue`: all six non-test matches (games 2, 3, 4, 5, 6 and 9; 17,250 frames). The final RF-DETR. Its validation set (every 10th game 6 frame) is in the training data, so it only satisfies the loader, and the last epoch is used.
  - `final-m-1280-gsr-hue`: YOLO with D7's recipe on all six matches. If RF-DETR does not gain clearly over D7 on validation, the YOLO is the final detector: nearly as good at a seventh of the cost.
- `scripts/merge_yolo_datasets.py` combines the folders, since RF-DETR reads one folder per split.
- To save waiting, both final detectors run detection and tracking on the test set as soon as they finish. Only the one chosen on validation is evaluated there (test run 3).

**Validation check, `rfdetr-m-gsr-hue`** (H100, 19.5 min, 0.10 kWh GPU energy; both RF-DETR runs got H100s, so their times are not comparable with D5's A100 run). Validation mAP@50:95 per epoch (EMA weights): 0.472 0.478.

| Run | mAP@50:95 all | Player | Goalkeeper | Referee | Ball |
|---|---|---|---|---|---|
| RF-DETR D5 (2 matches, 6 epochs, last) | 0.448 | 0.608 | 0.405 | 0.580 | 0.198 |
| RF-DETR, 5 matches + hue (2 epochs, last) | **0.478** | **0.613** | **0.506** | 0.579 | **0.216** |

- **The same changes help RF-DETR as they helped YOLO**, and again mostly through goalkeepers (+0.10). This is the best mAP of all runs (YOLO D7: 0.474), at a third of D5's training steps per match.
- The final RF-DETR (`rfdetr-m-all-hue`, six matches) took 31.5 min on an H100, 0.13 kWh.

Tracking on game 6 with the chosen pipeline (RF-DETR's tracker thresholds; last epochs):

| Detector | HOTA | People | Ball | DetA | AssA | IDSW | Role accuracy | Goalkeeper recall / precision | Model time per frame |
|---|---|---|---|---|---|---|---|---|---|
| YOLO D7 (5 matches + hue) | 65.5 | 66.7 | 43.2 | 73.4 | **58.6** | **538** | 98.2 % | 0.69 / 0.93 | 5.5 ms |
| RF-DETR D5 (2 matches) | **65.8** | **67.0** | 44.6 | 74.1 | 58.5 | 594 | 97.9 % | 0.54 / 0.92 | 40 ms |
| RF-DETR, 5 matches + hue | 65.7 | 66.9 | **45.2** | **74.3** | 58.3 | 613 | **98.9 %** | **0.79 / 0.95** | 40 ms |

- **The mAP gain hardly reaches HOTA:** +0.2 over D7, −0.1 against RF-DETR without the changes. The changes again pay off in roles (98.9 %, the best of all runs; goalkeeper recall 0.79) and the ball (+2.0 over D7), not in people tracking.
- **The YOLO final (`final-m-1280-gsr-hue`) is the final detector**, by the rule set before this run: RF-DETR does not gain clearly in HOTA, and costs 7× the model time. Its role and ball gains are reported as the trade-off.
- Test detection with the YOLO final: 2.6 ms model time per frame on an H100, 6 min for the 36,750 test frames.

### Final detector, first version (test runs 1 and 2): trained on all three non-test matches
- **Settings chosen on validation:** YOLO26m at 1280, frozen backbone, AdamW at learning rate 0.0002, 8 epochs (`final-m-1280-trainval`).
- **Retrained on games 4, 6 and 9** (8,550 frames). Game 6 was only held out to make choices; with those made, a third match of kits, goalkeepers and stadium is worth more than a validation score. Goalkeepers should gain most.
- **The last epoch is used, not the "best" one.** Validation is now part of training, so picking an epoch by its validation score would be meaningless. D2c showed the last epoch is as good as the best with this recipe.
- **The test set (games 7, 8, 11) is used once**, with the final detector and the tracker settings chosen on validation.

### T1, continued: tracking on fine-tuned detections (D2 at 1280, best epoch 1)
| Detections + tracker settings | HOTA all | HOTA people | DetA people | AssA people | IDSW people | HOTA ball |
|---|---|---|---|---|---|---|
| COCO, OC-SORT (150) + single ball ≥ 0.3 | 56.1 | 57.3 | 67.3 | 49.0 | 1,221 | 30.5 |
| D2, same settings | 47.5 | 48.6 | 65.9 | 36.0 | 2,926 | 21.6 |
| D2, + merge duplicate people boxes | 52.3 | 53.4 | 68.4 | 41.8 | 1,740 | 34.8* |
| COCO, OC-SORT without score fusion | 57.4 | 58.6 | 66.1 | 52.2 | 1,007 | 30.5 |
| **D2, merge + OC-SORT without score fusion** | **59.8** | **61.3** | **71.9** | **52.4** | 1,135 | **34.8*** |

\* single ball without a confidence threshold. The fine-tuned model is less confident, and a 0.3 threshold drops ball HOTA to 21.6, so the threshold depends on the detector.

- **The fine-tuned model first tracked worse than the COCO model**, with more than twice the ID switches. Two causes were found:
  - **Duplicate boxes.** YOLO26 has no NMS step and picks boxes per class, so the same person often comes out as both player and goalkeeper (2.1 duplicate pairs per frame on SNMOT-097, against 0 for the one-class COCO model). Each copy starts its own track. → `scripts/track.py --merge-iou 0.7` keeps the most confident of people boxes overlapping by more than IoU 0.7. People HOTA +4.8.
  - **Score fusion.** Ultralytics trackers multiply the match score by the detection confidence (`fuse_score`). The fine-tuned model is less confident (median 0.69 against 0.85), so good matches fall below the matching threshold and tracks break. → `--set fuse_score=false`. People HOTA +7.9, ID switches −35 %. It also helps the COCO detections (+1.3).
- **With both fixes, fine-tuning pays off:** people HOTA 61.3 against 58.6, and DetA 71.9 against 66.1, mostly from fewer false people (MOTA 89.9 against 78.8). Association quality is equal (AssA 52), as expected, since the tracker is the same.
- Low-score matching (`use_byte=true`) adds only 0.8, so it stays off.
- These detections come from D2's epoch-1 checkpoint, which has barely learned the SoccerNet box style (LocA still 83). D2c should do better.
- **With D2c detections** (frozen backbone, same tracker settings): HOTA all **62.8**, people **64.5** (DetA 74.2, AssA 56.2, 977 ID switches), ball 35.8. +3.0 over D2 on all classes. A ball threshold of 0.2 raises ball HOTA to 37.8 and HOTA all to **63.1**. ByteTrack with the same settings is 0.7 lower than OC-SORT.

### Tracker tuning and gap interpolation (OC-SORT on D2c detections, validation match 6)
One setting changed at a time from the base (`track_buffer=150 fuse_score=false`, merge 0.7, single ball ≥ 0.2), then the best combined.

| Setting | HOTA all | HOTA people | AssA people | IDSW people |
|---|---|---|---|---|
| Base | 63.1 | 64.5 | 56.2 | 977 |
| `match_thresh` 0.6 / 0.7 / **0.9** (default 0.8) | 61.7 / 62.4 / **63.4** | 62.1 / 63.7 / **64.7** | | 1,780 / 1,181 / 880 |
| `new_track_thresh` 0.4 / 0.5 (default 0.25) | 63.2 / 63.2 | 64.5 / 64.6 | 57.0 | 809 / 756 |
| `track_high_thresh` 0.15 / 0.4 / 0.5 (default 0.25) | | 62.5 / 58.5 / 52.8 | | |
| `track_buffer` 60 / 300 | 63.3 / | 64.7 / 64.1 | | |
| `match_thresh=0.9 new_track_thresh=0.5` | 63.3 | 64.7 | 57.1 | 728 |
| **+ interpolate gaps ≤ 10 frames** | **63.6** | **65.1** | **57.3** | **629** |
| + interpolate gaps ≤ 25 / 50 frames | 63.5 / 63.1 | 65.0 / 64.7 | | |

- **The tracker is close to saturated:** the best combination adds 0.5 HOTA. It does cut ID switches by 36 % (977 → 629), which matters for post-match statistics per player.
- A looser match threshold (0.9, i.e. IoU ≥ 0.1) keeps tracks through fast camera pans; a stricter one breaks them.
- A higher threshold for starting new tracks (0.5) avoids short false tracks from weak detections.
- **Filling short gaps** offline (`--interpolate 10`) adds a little. Longer gaps get worse: across 2 s a player does not move in a straight line, and the ball even less.
- Remaining gains have to come from the detector (DetA 74, LocA 84) and from re-identifying players who come back into view (AssA 57).
- **Best OC-SORT settings:** `track_buffer=150 fuse_score=false match_thresh=0.9 new_track_thresh=0.5`, `--merge-iou 0.7 --ball single --ball-conf 0.2 --interpolate 10`.

### T2: BoT-SORT with camera motion compensation on D2c detections (validation match 6, run on IDUN)
Same detections and options as above (`fuse_score=false`, merge 0.7, single ball ≥ 0.2); sparse optical flow for camera motion.

| Tracker | HOTA all | HOTA people | AssA people | IDSW people | HOTA on corners |
|---|---|---|---|---|---|
| OC-SORT, tuned, + interpolation | 63.6 | 65.1 | 57.3 | 629 | 55.4 |
| BoT-SORT + GMC, default thresholds | 63.7 | 65.1 | 56.7 | 778 | 57.3 |
| BoT-SORT + GMC, `match_thresh=0.9 new_track_thresh=0.5` | 63.8 | 65.2 | 57.4 | 658 | 57.3 |
| **BoT-SORT + GMC, same + interpolation ≤ 10 frames** | **63.9** | **65.4** | **57.5** | **607** | **57.5** |

- **Camera motion compensation helps most where the camera moves most:** +2.1 HOTA on corners, +0.3 overall. The same tuned thresholds help BoT-SORT too.
- **Cost:** it needs the frames, and tracking drops from about 300 to about 30 frames/s on one CPU core. That is still faster than real time (25 frames/s) and about as fast as detection, so for post-match analysis it is affordable.
- **Chosen tracker settings:** BoT-SORT + GMC, `track_buffer=150 fuse_score=false match_thresh=0.9 new_track_thresh=0.5`, `--merge-iou 0.7 --ball single --ball-conf 0.2 --images --interpolate 10`. Updated after the first test run: the ball uses `--ball path` instead of `--ball single --ball-conf 0.2` (see the ball path section). Tuned on validation only; the test set is not used for any choice.

### Roles in the tracks (chosen OC-SORT settings, D2c detections)
HOTA as computed here ignores roles, so roles are checked separately: every ground-truth person box is matched to the track box with the highest IoU (> 0.5) in the same frame, and the roles are compared.

| Roles from | Role accuracy | Goalkeeper recall | Goalkeeper precision | Referee recall |
|---|---|---|---|---|
| Each box's own class | **96.8 %** | **0.40** | 0.67 | **0.99** |
| Majority vote per track (confidence-weighted) | 96.4 % | 0.29 | **0.90** | 0.94 |

- **Players and referees are labelled correctly almost always; goalkeepers are not.** Only 40 % of goalkeeper boxes are called goalkeeper; the rest become players. Two training matches show only four goalkeeper kits.
- **Voting per track does not help overall.** Goalkeeper labels get more reliable but rarer, and tracks that switch identity between a referee and a player drag the referee down. → Not used.
- → Training on all three non-test matches adds goalkeeper kits, which is the best available fix.

### T6: newer trackers, Deep OC-SORT and TrackTrack (D2c detections, validation match 6)
Both ship with Ultralytics 8.4 and were run with camera motion compensation, no ReID, and the same post-processing as the chosen pipeline (merge 0.7, ball path, gaps ≤ 10 filled). "Adjusted" means thresholds moved to suit our detector, which is less confident than TrackTrack's defaults assume (it starts tracks only above 0.7; our median confidence is 0.69).

| Tracker | HOTA all | HOTA people | AssA people | IDSW | HOTA on corners |
|---|---|---|---|---|---|
| **BoT-SORT + GMC, tuned (chosen)** | 64.1 | 65.4 | 57.5 | 607 | 57.7 |
| Deep OC-SORT + GMC, defaults | 61.0 | 62.2 | 52.2 | 884 | 54.0 |
| Deep OC-SORT + GMC, our tuned thresholds | **64.3** | **65.6** | **57.8** | **600** | **58.6** |
| TrackTrack, defaults (buffer 150) | 59.5 | 60.6 | 52.7 | 589 | 50.2 |
| TrackTrack, adjusted (high 0.4, new 0.5, low 0.1, lost rebinding 0.9) | 63.1 | 64.4 | 56.0 | 614 | 56.0 |

- **Neither beats the chosen tracker by a meaningful margin.** Deep OC-SORT with our thresholds is +0.15 HOTA, within noise; without them it is 3 points lower. → **BoT-SORT stays**, and the test set is not used again for this.
- **Thresholds matter more than the tracker:** the same tuning (`fuse_score=false`, looser matching, stricter track start) lifts Deep OC-SORT by 3.2 and TrackTrack by 3.6. Trackers tuned on pedestrian benchmarks assume a confident detector; ours, fine-tuned on two matches, is not.
- All motion-only trackers end up within about one point of each other: what remains is detection errors and players returning to the picture, which motion cannot solve.

### T7: linking tracks by jersey number (validation match 6)
**Why:** with perfect detections the tracker still makes 1,264 ID switches on the test set (AssA 78): players who leave the picture come back with a new ID, and motion cannot re-identify them. A jersey number can.

**Feasibility:** on 48 random player crops (SNMOT-061, median player height 106 px), about a quarter show a readable number. Players stay in view for several seconds, so most tracks get a few readable frames.

**Method:**
- `scripts/read_jerseys.py`: for every track, up to 10 frames spread over its life (larger boxes first); the shirt below the head (15–60 % of the box height) is cropped and read with **Florence-2-large** (Microsoft, 0.77 B parameters, MIT licence, zero-shot `<OCR>`). One- and two-digit numbers are kept, with the shirt's mean CIELAB colour.
  - A published jersey-number model fine-tuned on SoccerNet data was avoided: its training clips likely overlap our test set.
  - Ran on IDUN CPUs (16 cores per clip, about 1 crop/s), since the GPU queue was full.
- `scripts/link_tracks.py`:
  - Players are split into two teams per clip by shirt colour (2-means).
  - A track's number is trusted only when the readings agree: at least 2 identical readings and 60 % of its readings.
  - Tracks with the same role, team and number that never overlap in time get one ID.

**What Florence-2 gets wrong:** it reads clear numbers correctly (33, 7, 16, 31, 24, 62), but **"1" in 75 % of crops with any reading** (3,287 of 4,370), mostly front views and body outlines, and often drops a digit of two-digit numbers (24 → 2, 17 → 1).

**Results** (BoT-SORT + GMC pipeline with ball path; link correctness checked against ground-truth identities):

| Linking | Links | Correct | HOTA all | AssA people | IDF1 | IDSW |
|---|---|---|---|---|---|---|
| None | 0 | | 64.1 | 57.5 | 77.4 | 609 |
| One- and two-digit numbers, ≥ 2 readings ('1' ignored for outfield players) | 15 | 12 | 64.6 | 58.4 | 78.2 | 599 |
| **Two-digit numbers only, ≥ 2 readings** | **11** | **11** | **64.5** | **58.3** | **78.0** | **600** |
| Two-digit numbers only, ≥ 1 reading | 13 | not checked | 64.7 | 58.5 | 78.2 | 598 |

- **Linking by number works when it fires: +0.4–0.5 HOTA, and every link from a two-digit number joins the same real player.** All three wrong links came from the single digit "2", a two-digit number missing a digit.
- → **Chosen: two-digit numbers only, read at least twice.** On unseen matches a wrong merge costs more than a missed one.
- **Coverage is the limit:** only 136 of 672 tracks get a trusted number, and 11 links result. A reader that does not hallucinate "1" and reads both digits would find many more. This is the clearest path to further association gains.

### Ball: offline path through all candidates (`--ball path`, validation match 6)
**Why:** on validation the true ball is among the detector's candidates (confidence ≥ 0.01) in 83 % of frames with a ball, but it is the most confident candidate in only 77 %. 16 % of true balls have confidence below 0.2 and were thrown away by the threshold. Ball motion in the training labels: median 3.6 px/frame, 90th percentile 20, 99th 50.

**Method:** `football.tracking.ball_path` picks at most one candidate per frame and finds the best-scoring path by dynamic programming over all frames (offline, so the future is used too). Score: log(confidence / 0.2) per chosen candidate, minus a squared penalty for speeds above 20 px/frame, minus 0.3 per skipped frame inside a segment, minus 3 for starting a new segment (the ball left the picture). A weak candidate that continues the path is kept; a confident outlier far from it is dropped.

| Ball method | HOTA ball | DetA ball | AssA ball | MOTA ball | HOTA all |
|---|---|---|---|---|---|
| Most confident candidate ≥ 0.2 per frame | 38.1 | 37.2 | 39.2 | 24.6 | 63.9 |
| **Path search, default settings** | **40.1** | **39.0** | **41.3** | **33.4** | **64.1** |
| Path search, best of 9 settings tried (`gap_cost` 0.1) | 40.5 | 39.7 | 41.3 | 30.0 | 64.1 |

- **+2.0 ball HOTA and +8.8 ball MOTA**, mostly from fewer false balls. The default settings are kept; the best variant is within noise.
- **The ball is limited by the detector, not the selection.** In 17 % of frames the true ball is not among the candidates at all, and predicted boxes are on median 1.8 px off-centre and 7 % smaller than the labels (16 px wide). For a 16 px box that is enough to fail HOTA's stricter overlap thresholds (ball LocA 73).

### T2/T4 first check: camera motion compensation and appearance ReID (SNMOT-060, perfect detections)
One training clip with images on the laptop, so this is a quick check only, before spending cluster time.

| Tracker | HOTA people | AssA people | IDSW | Speed (laptop CPU) |
|---|---|---|---|---|
| OC-SORT (150) | 85.9 | 74.7 | 61 | about 300 frames/s |
| BoT-SORT, no camera motion compensation | 84.9 | 73.0 | 65 | |
| BoT-SORT + camera motion compensation (sparse optical flow) | **87.1** | 76.0 | 42 | 25 frames/s |
| BoT-SORT + camera motion compensation + ReID (`yolo26s-reid`) | 87.1 | 76.1 | 38 | 2 frames/s |

- **Camera motion compensation helps** (+2.1 HOTA over BoT-SORT without it, +1.2 over OC-SORT), as the camera-pan analysis predicted. It needs the images, so it runs on IDUN.
- **Appearance ReID adds nothing** here at 12× the compute. Players on the same team wear identical kits, and BoT-SORT only uses appearance to choose between nearby candidates, not to re-identify players who come back into view. → **ReID is dropped**: not worth the compute and energy. Re-identifying returning players would need jersey numbers or pitch positions, which is out of scope.
- Tracking jobs need no GPU and go through `jobs/cpu.slurm`, so they don't hold a GPU while it sits idle.

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

### T1: trackers on real detections from the COCO model (validation match 6)
Detections from COCO-pretrained YOLO26m at 1280, no fine-tuning (`scripts/detect.py --coco`, kept down to confidence 0.01). COCO has one person class, so people are evaluated without roles here; role errors are not counted yet.

| Detections + tracker | HOTA all | HOTA people | DetA people | AssA people | LocA people | IDSW people | HOTA ball |
|---|---|---|---|---|---|---|---|
| Perfect + OC-SORT (150) + single ball | 89.7 | 89.3 | 99.2 | 80.3 | 99.9 | 777 | 97.0 |
| COCO + ByteTrack (30) | 55.1 | 56.6 | 66.5 | 48.3 | 83.6 | 1,289 | 18.1 |
| COCO + OC-SORT (150) | 55.8 | 57.3 | 67.3 | 49.0 | 83.7 | 1,221 | 17.7 |
| COCO + OC-SORT (150) + single ball | 55.6 | 57.3 | 67.3 | 49.0 | 83.7 | 1,221 | 27.2 |
| COCO + OC-SORT (150) + single ball, conf ≥ 0.3 | **56.1** | 57.3 | 67.3 | 49.0 | 83.7 | 1,221 | **30.5** |

- **Detection errors cost 32 HOTA on people** (89.3 → 57.3). Both parts drop: DetA from 99 to 67, and AssA from 80 to 49 with the same tracker. Missed and false boxes also break tracks.
- **The COCO boxes are drawn differently from SoccerNet's.** LocA is 83.7 instead of 99.9. On SNMOT-097, matched COCO boxes are 9 % narrower and 4 % shorter than the ground truth, and start lower. A systematic annotation-style difference, not a bug; fine-tuning (D2) should learn it.
- **OC-SORT still beats ByteTrack** on real detections (+0.7 HOTA on people, fewer ID switches).
- **The one-ball prior still helps** (ball HOTA 17.7 → 27.2), and requiring confidence ≥ 0.3 adds 3.3 more by skipping frames whose best ball candidate is a false detection. The ball is now limited by the detector (D1 ball recall 0.38).
- **Tracking is cheap:** 290–430 frames/s on a laptop CPU, so the detector dominates the compute cost.
- **Reference point:** SoccerMaster reports HOTA 54.6 for YOLOv8 + OC-SORT on the test set. Our zero-shot COCO baseline reaches 55.8 on validation, but the split and the class handling differ, so the numbers are not directly comparable.

---

## Test set: final pipeline (games 7, 8, 11; 49 clips, 36,750 frames; three runs, all reported)
Final detector (YOLO26m at 1280, frozen backbone, trained on games 4, 6 and 9, last epoch) and the chosen tracker (BoT-SORT + camera motion compensation, tuned thresholds, merge 0.7, single ball ≥ 0.2, gaps ≤ 10 frames filled). Nothing was changed after seeing these numbers.

| Subset | HOTA | DetA | AssA | LocA | IDF1 | MOTA | IDSW |
|---|---|---|---|---|---|---|---|
| **All** | **63.0** | 71.4 | 55.7 | 84.2 | 73.6 | 85.9 | 1,344 |
| People | 64.4 | 74.0 | 56.1 | 84.7 | 74.6 | 89.9 | 1,339 |
| Ball | 39.8 | 38.8 | 41.0 | 73.6 | 57.6 | 22.6 | 0 |

| Group | Clips | HOTA all | HOTA people | HOTA ball |
|---|---|---|---|---|
| Game 7 | 16 | 65.3 | 66.8 | 40.4 |
| Game 8 | 19 | 61.5 | 62.9 | 42.7 |
| Game 11 | 14 | 62.0 | 63.5 | 34.4 |
| Clips with camera cuts | 6 | 63.6 | 65.2 | 42.1 |
| Clips without | 43 | 62.9 | 64.3 | 39.5 |

By clip type: goals 70.6, substitutions 69.9, yellow cards 66.7 … shots on target 57.7, corners 54.7.

- **The result holds on unseen matches:** 63.0 on test against 63.9 on validation.
- **Corners stay hardest** (54.7): crowded penalty areas, where players occlude each other.
- **Game 11, filmed from further away (players about 89 px tall against about 100 in training), has the weakest ball** (34.4).
- **Camera cuts do no visible harm** (63.6 against 62.9): after a cut the tracker starts new tracks, which costs identity across the cut but little else.
- **Speed:** detection ran at 82 frames/s end to end on an A100 (5.5 ms model time per frame) after decoding images in parallel, 3.3× faster than the first version. Tracking with camera motion compensation ran at 30 frames/s per CPU core, 8 clips in parallel. The whole test set took 7.5 + 3.3 min.

**Second test run: ball path search.** After the first test run, the ball path search was developed and chosen on validation (+2.0 ball HOTA there). It changes only the ball, and reuses the final detector's test detections. This is the second and last look at the test set; both runs are reported.

| Test run | HOTA all | DetA | AssA | HOTA people | HOTA ball | MOTA ball |
|---|---|---|---|---|---|---|
| 1: single ball ≥ 0.2 | 63.0 | 71.4 | 55.7 | 64.4 | 39.8 | 22.7 |
| **2: ball path search** | **63.1** | **71.6** | **55.8** | 64.4 | **41.4** | **30.3** |

The gain carries over to unseen matches (+1.6 ball HOTA, +7.7 ball MOTA).

**Third test run: second final detector and jersey linking.** After test run 2, extra matches and hue augmentation (D6, D2f, D7) and jersey linking (T7) were chosen on validation. The second final detector (YOLO `final-m-1280-gsr-hue`, six matches with hue augmentation, last epoch) was chosen over RF-DETR on validation before this run. Tracker settings are unchanged. This is the third look at the test set; all three runs are reported.

| Test run | HOTA all | DetA | AssA | HOTA people | AssA people | IDSW | HOTA ball | Role accuracy |
|---|---|---|---|---|---|---|---|---|
| 2: first final detector, ball path | 63.1 | 71.6 | 55.8 | 64.4 | 56.1 | **1,340** | **41.4** | 96.4 % |
| 3 without linking | 63.2 | **72.1** | 55.4 | 64.5 | 55.8 | 1,428 | 40.5 | **98.5 %** |
| **3: second final detector, jersey linking** | **63.4** | **72.1** | **56.0** | **64.8** | **56.3** | 1,404 | 40.5 | **98.5 %** |

| Group | HOTA all (run 1) | HOTA all (run 3) | Referee recall (run 1) | Referee recall (run 3) | Goalkeeper recall (run 3) |
|---|---|---|---|---|---|
| Game 7 | 65.3 | 65.1 | 99.6 % | 99.5 % | 70 % |
| Game 8 | 61.5 | 62.9 | 99.7 % | 99.2 % | 87 % |
| Game 11 | 62.0 | 62.2 | 22.5 % | **83.1 %** | 79 % |

| True role (run 3) | Recall | Precision |
|---|---|---|
| Player | 0.998 | 0.986 |
| Goalkeeper | 0.786 | 0.952 |
| Referee | 0.936 | 0.995 |

By clip type: goals 70.9, shots off target 66.6, shots on target 58.2, corners 55.4.

- **The new detector's gain is in roles, as on validation, not in HOTA.** Detection improves (+0.5 DetA), but association does not (−0.3 AssA, 88 more ID switches), so HOTA without linking is +0.1. The ball loses 0.9.
- **Hue augmentation fixes game 11's referee:** recall 83 % against 22.5 %. He wears a dark green kit no training match had; with colours scrambled in training, the model learned referees by more than their kit colour. Goalkeeper recall rises from 0.63 to 0.79 at the same precision.
- **Jersey linking carries over:** 27 tracks joined, +0.3 HOTA and +0.5 AssA on people, close to its +0.4 on validation. Whether each link is correct was not checked on the test set.
- Jersey reading: 49 CPU jobs (16 cores each), at most 12.7 min per clip.

**Baseline and upper bound on the test set:**

| Pipeline | HOTA all | HOTA people | AssA people | IDSW | HOTA ball |
|---|---|---|---|---|---|
| Off-the-shelf: COCO YOLO26m at 1280 + default ByteTrack | 53.4 | 55.0 | 46.4 | 3,902 | 13.7 |
| Ours (first final pipeline, test run 2) | 63.1 | 64.4 | 56.1 | 1,340 | 41.4 |
| **Ours (second final pipeline, test run 3)** | **63.4** | **64.8** | **56.3** | 1,404 | 40.5 |
| Our tracker on perfect detections | 87.7 | 87.1 | 76.7 | 1,264 | 95.7 |

- **The pipeline adds 10.0 HOTA over the off-the-shelf baseline** (9.7 in test run 2), cuts ID switches by almost two thirds and triples ball HOTA. The baseline also cannot tell players, goalkeepers and referees apart.

**Upper bound on the test set:** the same tracker on perfect detections (ground truth without IDs, no box merging) reaches **HOTA 87.7** (people 87.1, AssA 77.7; ball 95.7). Detection errors cost about 25 HOTA. Even with perfect boxes, association stops at AssA 78: players who leave the picture and come back get new IDs.

**Roles on the test set** (`scripts/eval_roles.py`, 500,798 matched person boxes): role accuracy 96.4 %.

| True role | Recall | Precision |
|---|---|---|
| Player | 0.999 | 0.963 |
| Goalkeeper | 0.629 | 0.966 |
| Referee | 0.737 | 0.989 |

- **Goalkeepers are recognised far better than in validation** (recall 0.63 against 0.39 with D2c), at precision 0.97. Training on a third match helped as hoped.
- **Referees fail in one match only:** recall 99.6 % in game 7 and 99.7 % in game 8, but **22.5 % in game 11**, where the referee wears a dark green kit in bright sun. No training match had a referee like that, so he is called a player. A clear example of what three training matches cannot cover.

---

## Open questions

| Question | How to answer it |
|---|---|
| How much does training on every n-th frame cost? | Train with different strides and compare (consecutive frames are near-identical) |
