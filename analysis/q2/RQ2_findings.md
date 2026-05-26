> **⚠ Superseded numbers** (pre VO-scaling fix). Null holds: vision **+0.839** vs
> signals **+0.840**. See [`../CORRECTED_RESULTS.md`](../CORRECTED_RESULTS.md).

# Research Question 2 — When Do Visual Features Help Beyond Robot + VO-Health Signals?

## Goal

Determine whether adding the camera image as an input improves the 1-second-ahead VO degradation predictor from Research Question 1. The deployment-relevant version of the question: when robot and VO-health signals are missing, delayed, or unreliable, can images compensate?

---

## Setup

**Identical to Q1**: same dataset (60 B+C runs), same per-frame disagreement label (VO twist in m/s vs commanded twist), same 1-second future-mean target, same standardized 29-dimensional tabular feature vector, and the same committed 80/20 run-level split (`Data/comprehensive_dataset/split.json`):

- Train: 42 runs
- Validation: 6 runs (16, 44, 47, 49, 51, 57)
- Test: 12 runs (13, 22, 24, 32, 34, 37, 43, 48, 53, 58, 63, 68)

The only difference from Q1 is the model. Everything else is held constant so the comparison is fair.

---

## Model architecture

A two-branch network fused at a final head:

```
                                ┌──────────────────────────────────────────────────────┐
image (3, 120, 160)  ─────────► │ CNN: 4 conv blocks (stride-2 downsampling) → AvgPool │ ─► 64-d image embedding
                                └──────────────────────────────────────────────────────┘                          \
                                                                                                                   concat → Linear(128→64) → ReLU → Dropout(0.2) → Linear(64→1) → ŷ
                                ┌──────────────────────────────────────────────────────┐                          /
tabular features (29) ────────► │ MLP: Linear(29→64) → ReLU → Linear(64→64) → ReLU     │ ─► 64-d tabular embedding
                                └──────────────────────────────────────────────────────┘
```

- **CNN branch.** 4 conv layers (3→16→32→64→64 channels) with stride-2 downsampling at each step, then adaptive average pooling to a 64-vector, then a linear layer. Roughly 25 k parameters.
- **Tabular branch.** Two-layer MLP, 29-dim input → 64-dim embedding. Roughly 6 k parameters.
- **Fusion head.** Concatenated 128-dim → 64 → 1. Roughly 8 k parameters.
- **Total:** ~40 k parameters.
- **Training:** PyTorch, AdamW (lr 1e-3, wd 1e-4), batch size 512, 8 epochs, MSE loss against the future-badness label. Best validation checkpoint retained. Single CUDA device (RTX 3080).

A single current frame is used as the visual input — no temporal stacking, no augmentation, no pretrained backbone.

---

## Evaluations

Three preregistered evaluations on the 12 held-out test runs:

1. **Full-info Spearman**, compared directly to the Q1 signals-only model.
2. **VO-health-masked Spearman.** At test time we zero out 11 of the 29 feature columns (recent disagreement statistics, VO feature counts, VO covariance diagonals, dropout fraction) — leaving pan/tilt, command, and instantaneous VO velocity. Image input is unchanged. This simulates loss of VO telemetry while the camera keeps working.
3. **Per-condition Spearman delta** vs Q1, broken down by the 8 B/C conditions. Tells us *where* (if anywhere) vision helps.

---

## Results

### Overall (12 test runs, ~50k frames)

| Predictor | Spearman ρ | RMSE | MAE |
|---|---|---|---|
| Q1 signals-only (reference) | +0.762 | 4.70 | 1.52 |
| **Q2 vision (full info)** | **+0.750** | **4.42** | 1.60 |
| Q2 vision (VO-health masked) | +0.587 | 7.20 | 2.51 |

**Vision did not improve the headline metric.** Spearman changes by −0.012 — within noise. RMSE drops slightly (4.70 → 4.42), suggesting marginal calibration improvement on magnitudes, but rank ordering is unchanged.

### Per-condition Spearman

| Condition | n | Q1 ρ | Q2-full ρ | Q2-mask ρ | Δ vision | Δ mask |
|---|---|---|---|---|---|---|
| combined_slow | 7056 | +0.663 | +0.678 | +0.463 | +0.015 | −0.215 |
| slow_pan | 9579 | +0.584 | +0.586 | +0.418 | +0.002 | −0.167 |
| slow_tilt | 7608 | +0.780 | +0.768 | +0.477 | −0.012 | −0.291 |
| static:0.0:0.5 (pan left) | 4431 | +0.696 | +0.631 | +0.556 | −0.065 | −0.075 |
| static:0.5:0.0 (floor) | 4942 | +0.657 | +0.674 | +0.511 | +0.017 | −0.162 |
| static:0.5:0.5 (centered) | 3978 | +0.602 | +0.569 | +0.530 | −0.034 | −0.039 |
| static:0.5:1.0 (ceiling) | 9144 | +0.813 | +0.811 | +0.515 | −0.002 | −0.296 |
| static:1.0:0.5 (pan right) | 2771 | +0.762 | +0.741 | +0.588 | −0.020 | −0.153 |

No condition shows a meaningful positive lift from vision. The largest positive delta is +0.017 (effectively zero); the largest negative is −0.065. Notably, `slow_pan` — our hypothesized winner for image-based motion-blur detection — moved by +0.002.

### Run-level sanity check vs `estimated_lce_m`

Mean predicted disagreement per test run, correlated with the manifest's per-run VO error:

- Q1 signals-only: Pearson r = **+0.771**
- Q2 vision (full info): Pearson r = **+0.805**
- Q2 vision (VO-health masked): Pearson r = **+0.767**

At the *run* level (not the *frame* level), the vision-augmented model is modestly better correlated with measured VO error. The masking experiment barely degrades this run-level correlation — meaning that for coarse run-level risk assessment, vision and signals carry overlapping information.

---

## Interpretation

### The vision branch was not load-bearing

The masked-info experiment is the key diagnostic. When we removed VO-health features at test time, Spearman dropped from **0.750 → 0.587** (Δ = −0.163). If the CNN had been carrying real information, the masked drop should have been small. It wasn't. The model learned to lean almost entirely on the tabular VO-health features and barely used the image branch.

This is gradient descent doing what it does: with a strong tabular signal and a weaker visual signal, it took the path of least resistance.

### The tabular features already cover the cause

Pan/tilt position and pan/tilt rate are essentially a high-level summary of what the image would contain: *where* the camera is pointing and *how fast* it is moving. The CNN, given only a single 120×160 frame and no temporal context, evidently could not extract anything beyond what those scalars already represent.

### What this means for the research question

At this image resolution (120×160), with a single frame per prediction, and given the full set of robot + VO-health tabular signals, **camera images do not provide additional predictive value for 1-second-ahead VO degradation.**

That is a substantive answer to Q2, not a null result. It says: a deployment-time monitor for VO degradation can rely on tabular signals alone; running a CNN on every frame adds compute cost without prediction gain.

---

## Known limitations and recommended follow-ups

The negative result is conditional on three choices, any of which could change the verdict:

1. **Single frame, no temporal context.** Motion blur is a between-frame phenomenon; a 3–5 frame stack would let the CNN see it directly.
2. **Low image resolution.** 120×160 limits the texture and feature cues the CNN can resolve. The full-resolution frames may not be available in this dataset.
3. **No training-time signal masking.** With every tabular feature present at every training step, the model is never *forced* to use the image. A random-masking training regime would directly test whether the CNN *can* learn to compensate when asked to.

Of these, **(3)** is by far the highest-value follow-up. It directly answers the deployment-relevant question: *"if VO telemetry pipeline drops out, can a vision-aware monitor keep working?"* Same data, same model, same compute — only the training procedure changes.

---

## Conclusion

Adding camera images on top of robot + VO-health signals produced **no meaningful improvement** in 1-second-ahead VO degradation prediction at the frame level: overall Spearman dropped from +0.762 to +0.750. Per-condition deltas are all within noise. The model preferred the tabular features and the vision branch did not become load-bearing under standard training. At the run level, vision adds a small lift in correlation with measured VO error (+0.771 → +0.805), suggesting it captures a coarse signal that doesn't surface at the frame level.

The cleanest practical implication is that **for deployment with intact VO telemetry, vision is unnecessary**. The open follow-up is whether **training-time masking** can promote the vision branch into a working fallback for telemetry dropout scenarios — which would change the deployment story substantially.

---

*Code: `src/research_question_2.py`. Trained model: `analysis/q2/q2_vision_model.pt`. Predictions: `analysis/q2/q2_test_predictions.npz`. Data split: `Data/comprehensive_dataset/split.json`.*
