> **⚠ Superseded numbers** (pre VO-scaling fix + GT label). Corrected risk
> Spearman **+0.840**. See [`../CORRECTED_RESULTS.md`](../CORRECTED_RESULTS.md).

# Research Question 1 — Predicting VO Degradation 1 Second Ahead

## Goal

Determine whether visual odometry (VO) degradation can be predicted ~1 second before it occurs, using only signals available to the robot at runtime. A reliable early warning would let the localization stack reduce its trust in VO before bad estimates corrupt downstream state.

---

## Dataset

The comprehensive dataset logs robot teleop runs at a roughly 25 Hz logger rate. Each frame contains:

- `pan_tilt` — current camera head pose
- `wheel_twist` — commanded body twist (linear, angular). *Verified this is the controller command, not a wheel encoder: only 3 unique non-zero values across runs, matching ROS `cmd_vel` patterns.*
- `vo_twist` — VO-reported motion since the last VO update (per-update displacement, ~10 Hz)
- `vo_features` — number of features VO is currently tracking
- `vo_covariance` — VO's self-reported covariance (6 elements, known miscalibrated)

**Phase A (10 runs) was excluded.** The camera was off; `vo_twist`, `vo_features`, and `vo_covariance` are NaN throughout, so no VO degradation can be defined on those frames.

We worked with the **60 Phase B + Phase C runs**, ~250k usable frames:

- **Phase B (30 runs)** — 5 static pan/tilt configurations × 6 repeats. Captures degradation driven by *where* the camera is pointing.
- **Phase C (30 runs)** — slow_pan / slow_tilt / combined_slow, 3 conditions × 10 repeats. Captures degradation driven by *head motion*.

---

## Data split (committed and reused across all four research questions)

**Splits happen at the run level, not the frame level.** At 25 Hz, consecutive frames are highly correlated; a random per-frame split would put near-duplicates in both train and test and silently inflate every metric.

Concretely:

- **12 test runs** — held out completely, untouched until final evaluation. Selection is stratified across all 8 conditions and balanced 6 LR / 6 RL routes. The hardest condition (tilt-up, where VO routinely fails catastrophically) gets one extra test run.
- **6 validation runs** — carved out of the remaining 48 train runs with a fixed seed. Used during training for early stopping and hyperparameter checks. Never used as a held-out evaluation.
- **42 train runs** — used for model fitting and for computing feature standardization statistics.

The split is saved as `Data/comprehensive_dataset/split.json` and loaded by every experiment.

| Set | Run IDs |
|---|---|
| Test (12) | 13, 22, 24, 32, 34, 37, 43, 48, 53, 58, 63, 68 |
| Validation (6) | 16, 44, 47, 49, 51, 57 |
| Train (42) | all remaining B+C runs |

---

## Defining VO degradation without ground truth

We have no motion capture or external pose reference, so we constructed a proxy truth signal from two independent estimates of body motion at every frame:

- **VO's estimate**: `vo_twist`, converted from per-update displacement to m/s by dividing by the inter-update period. Periods are detected from change-points in `vo_twist` (the logger samples faster than VO publishes, so most logged values are held repeats of the last real VO update).
- **The commanded velocity** (`wheel_twist`).

The per-frame **disagreement signal** is then

> `disagreement(t) = ‖ vo_twist_mps(t) − wheel_twist_command(t) ‖`

VO dropouts (NaN, zero output, or zero feature count — ~2% of B+C frames) are flagged and treated as worst-case in label construction.

**Why this is a defensible truth signal:**

| Property | Result |
|---|---|
| Orders conditions by physical difficulty | ✓ centered (median 0.70) < floor (0.82) < sides (1.1) < ceiling (1.5) m/s of disagreement |
| Run-level Pearson correlation with `estimated_lce_m` (manifest's per-run VO error) | **+0.77** on held-out test runs |
| Free of common-mode error with VO | ✓ command is generated independently of VO |

**Caveats we acknowledge explicitly:** the disagreement signal also contains small contributions from (a) command-tracking lag during acceleration ramps, (b) camera-body coordinate-frame mismatch when the head is panned, and (c) unit-conversion noise from variable VO timing. These add noise but do not flip the rank ordering across conditions. We treat the signal as reliable for *relative* risk prediction, not as an absolute physical quantity.

---

## Task and label

For each frame *t*, the **future-badness label** is the mean disagreement over the next 1 second:

> `label(t) = mean( disagreement[t+1 : t+1s] )`

VO dropouts inside that window are imputed at the 99th percentile of the training-set disagreement distribution. The label is computed strictly from frames after *t*; nothing from the prediction window enters the model's inputs.

---

## Inputs

For each frame *t*, we build a flat **29-dimensional feature vector** summarizing the **past 0.5 s** of:

- Pan/tilt position and its first derivative (current value, std, mean abs rate, max abs rate)
- Commanded twist (current, window mean, sum of absolute command transitions)
- VO velocity in m/s (current, std)
- VO feature count (current, mean) — included even though it's a VO-internal signal
- VO covariance diagonals at *t* — included as features even though known miscalibrated
- Recent disagreement signal (current, mean, max, drift over window) — strictly lagged
- Dropout fraction in the past window

Standardization (z-score per feature) is fit on training data only and applied to validation and test.

---

## Model architecture

A small fully-connected MLP in PyTorch:

```
Input (29)
  → Linear(29 → 128) → ReLU → Dropout(0.1)
  → Linear(128 → 128) → ReLU → Dropout(0.1)
  → Linear(128 → 1)   → ŷ
```

~17k parameters. Single regression head; MSE loss against the future-badness label.

We deliberately started with the simplest viable architecture — hand-rolled summary features and a shallow MLP — to establish a baseline before introducing temporal models or image-based components. The image-aware comparison is the focus of Research Question 2.

---

## Training procedure

- **Optimizer:** AdamW, learning rate 1e-3, weight decay 1e-4.
- **Batch size:** 4096.
- **Epochs:** 20.
- **Early stopping (implicit):** after each epoch we evaluate on the 6 validation runs and keep the checkpoint with the lowest validation MSE.
- **Reproducibility:** fixed seeds (`numpy`, `torch`).
- **Hardware:** trained on a single CUDA device; ~37 s end-to-end training.

The 12 test runs are touched **exactly once**, after training has completed, to produce the numbers below.

---

## Baselines

We compare the MLP against two non-ML predictors:

1. **`current_disagreement`** — naive persistence: predict the next-second mean to equal the current-frame disagreement. The floor any useful model must clear.
2. **`vo_covariance_sum`** — sum of VO's own translational and rotational variance diagonals. The interesting bar: does our model add anything beyond VO's self-reported uncertainty?

---

## Results

### Overall (12 held-out test runs, ~64k frames)

| Predictor | Spearman ρ | RMSE | MAE |
|---|---|---|---|
| **MLP (ours)** | **+0.762** | 4.70 | 1.52 |
| current_disagreement | +0.528 | 10.53 | 2.65 |
| vo_covariance_sum | +0.331 | (units differ) | (units differ) |

Spearman (rank correlation) is the headline metric: the label distribution is heavy-tailed and the baselines are on different units. The MLP improves over naive persistence by +0.23 Spearman and roughly doubles the rank correlation of VO's own self-uncertainty.

### Per-condition Pearson r (test set)

| Condition | n | Pearson r | Notes |
|---|---|---|---|
| slow_tilt | 7,608 | **+0.852** | Best — hard dynamic case |
| static:0.5:1.0 (ceiling) | 9,144 | +0.786 | Highest mean disagreement (7.3) |
| combined_slow | 7,056 | +0.765 | |
| static:0.0:0.5 (pan left) | 4,431 | +0.726 | |
| static:1.0:0.5 (pan right) | 2,771 | +0.707 | |
| static:0.5:0.0 (floor) | 4,942 | +0.635 | |
| slow_pan | 9,579 | +0.542 | Weakest dynamic case |
| static:0.5:0.5 (centered) | 3,978 | +0.440 | Low signal — disagreement is uniformly small |

The low correlation in the centered condition is expected: mean true disagreement is 0.85 m/s with low variance, so there is little to predict. Per-frame MAE in that condition is correspondingly small.

### Run-level sanity check

Mean of the model's per-frame predictions, aggregated over each test run, versus the manifest's `estimated_lce_m` (the run's overall VO error):

**Pearson r = +0.771**

Runs the model flags as high-disagreement at the frame level are the same runs whose VO pose estimate ended up far from the physical loop-closure measurement. This confirms the frame-level disagreement signal we constructed is measuring real VO error, not an artifact of acceleration ramps or coordinate-frame mismatch.

---

## Conclusion

**Yes — VO degradation 1 second into the future is predictable from past sensor signals**, with substantial margin over both the naive persistence baseline and VO's own self-reported uncertainty. The result is consistent across conditions and is corroborated by an independent run-level correlation with measured VO error.

This establishes the baseline for the remaining three questions:
- **Q2** — Do image features add value beyond robot + VO-health signals? Same labels, same split, same evaluation; the comparison is whether adding a CNN over the camera frame beats this MLP, particularly under VO-health signal dropout/delay.
- **Q3** — Is the risk prediction calibrated, and can uncertainty flag unreliable predictions? Requires a second head producing predictive variance (e.g. Gaussian NLL or ensemble).
- **Q4** — Can uncertainty improve the robot's trust decision (safe / risky / uncertain)? Selective-prediction policy evaluated by EKF replay.

---

## Known limitations

- **The label is a proxy.** Disagreement vs command, not vs ground truth. Reliable for relative ranking; not a metrology-grade error measure.
- **Hand-rolled features.** We compress a 0.5 s history into 29 numbers manually; a temporal model (TCN/GRU/transformer) on raw sequences could do strictly more.
- **Single regression head.** No spike-classification head, no uncertainty, no calibration analysis yet — those come in Q3.
- **Lead-time not yet swept.** We trained at a fixed 1-second horizon. We have not yet characterized how performance degrades at longer horizons or how much earlier the model warns relative to the current-frame baseline.

---

*Code: `src/research_question_1.py`. Trained model: `analysis/q1/q1_mlp.pt`. Predictions: `analysis/q1/q1_test_predictions.npz`. Data split: `Data/comprehensive_dataset/split.json`.*
