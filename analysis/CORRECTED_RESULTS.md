# ⚠ SUPERSEDED — measured-ground-truth validation (2026-05-27, "Option A")

The conclusions below are framed against `fused_pose`, which is an onboard
wheel+IMU+VO complementary filter, **not** independent ground truth. Validated
against the dataset's only external measured GT (per-run tape-measured
`physical_lce_m`), the headline story changes:

- Out-of-fold over all 60 camera-on runs, mean predicted risk correlates **+0.64**
  with the filter self-estimate (`estimated_lce_m`) but only **+0.06** with the
  measured `physical_lce_m`; the filter estimate itself is only **+0.14** correlated
  with measured accuracy.
- **RETRACTED:** the "risk-gated fusion beats wheel-only by ~10% ATE" claim (Q4 below).
  ATE is measured against `fused_pose` (relative consistency, not accuracy) and no
  per-step measured GT exists for replayed policies — no localization improvement is
  claimed.
- **RE-SCOPED:** Q1 predicts the VO–fused *divergence* (still Spearman +0.84), not
  measured degradation. Q3 calibration/uncertainty stand as statements about that
  divergence. Q2's vision null is unaffected.

Central finding: an onboard fused-pose estimate is a seductive but invalid label for
VO degradation — a model trained on it predicts the estimate, not reality. See
[`measured_gt_validation.json`](measured_gt_validation.json) and
`fig_measured_gt_gap.png`.

---

# Corrected Results — All Four Questions (2026-05-26)

This supersedes the original per-question numbers. A VO-scaling bug was found and
fixed, the prediction target was re-grounded on ground truth, and Q4's fusion was
corrected. The pipeline is now consistent across all four questions.

## The correction (affects everything)

**Bug:** `research_question_1.py::vo_twist_to_mps` treated `vo_twist` as a
per-update *displacement* and divided by the inter-update period. This inflated
VO ~15× (linear p95 0.32 → 4.7 m/s on a 0.2 m/s robot) and produced the spurious
"heavy-tailed ±30 m/s VO outliers."

**Evidence it was wrong:** raw `vo_twist` already matches true velocity
(differentiated `fused_pose`): linear p95 0.32 vs ground-truth 0.27; angular 0.17
vs 0.34. The parent project's independent headroom analysis also integrates
`vo_twist` raw.

**Fix:** use `vo_twist` directly as a body velocity. One-line source change; it
propagates to all four questions.

**Two further upgrades:**
1. **Ground-truth label.** Degradation is now `‖vo − gt_twist‖` (gt_twist =
   windowed derivative of `fused_pose`, the LiDAR/SLAM reference) instead of
   `‖vo − wheel_command‖`. `wheel_twist` is only a *command*, and using it as both
   a feature and the label was mildly circular. The wheel-based disagreement stays
   as a runtime-observable feature; ground truth is the target only.
2. **Per-dimension fusion** in Q4 (separate linear/angular VO weights).

## Headline numbers (12 held-out test runs)

| Q | Metric | Original | **Corrected** |
|---|---|---|---|
| Q1 | risk Spearman | +0.762 | **+0.840** |
| Q1 | run-level corr w/ `estimated_lce_m` | +0.771 | **+0.794** |
| Q2 | vision vs signals Spearman | +0.750 vs +0.762 | **+0.839 vs +0.840** (still no gain) |
| Q3 | calibration error (raw → recalibrated) | 0.187 → 0.036 | **0.221 → 0.016** |
| Q3 | AUSE / Spearman(σ,\|err\|) | 0.134 / +0.56 | **0.171 / +0.53** |
| Q4 | best policy ATE vs wheel-only | matched (2.35 vs 2.28) | **2.041 vs 2.276 (−10.3%)** |

## Q1 — prediction works, better than before
Spearman **+0.840** on a now-physically-meaningful target (true VO velocity error,
median 0.12 m/s). Run-level correlation with the independent per-run VO error
**+0.794**. The naive "current disagreement" baseline is +0.537.

## Q2 — vision still doesn't help (robust null)
Vision-augmented Spearman **+0.839** vs signals-only **+0.840** — no frame-level
gain, holds under the corrected label. Masking VO-health features drops vision to
+0.657, confirming the CNN never became load-bearing. (Caveat unchanged: single
frame, no temporal stack, no training-time masking — a real test of "can vision
compensate for lost telemetry" needs those.)

## Q3 — calibrated and discriminative (strongest question)
Deep ensemble of 5 Gaussian-NLL MLPs. Point Spearman **+0.854**. Raw ensemble is
under-confident (CE 0.221); a single validation-tuned variance scale (×0.45) gives
**CE 0.016** — essentially calibrated. Uncertainty is discriminative: keeping the
25% least-uncertain frames cuts MAE from 0.115 to **0.032**; AUSE 0.171;
Spearman(σ, |error|) +0.53. This result is robust to the label change.

## Q4 — risk prediction improves localization; uncertainty's added value is not robust

Trajectory ATE vs `fused_pose`, per-dimension robust blend, all knobs tuned on
validation only. Tuned weights `[a_lin=0.25, a_ang=0.0]` — i.e. **use VO linear,
ignore VO angular** (VO angular is noisy/under-scaled).

| policy | test ATE (m) | vs wheel |
|---|---|---|
| wheel-only | 2.276 | — |
| const blend (VO-linear only) | 2.086 | −8.3% |
| **risk-gated blend** | **2.041** | **−10.3%** |
| three-way (+ uncertainty gate) | 2.150 | −5.5% |
| learned per-step weight (twist-MSE) | 6.454 | +184% (worse) |
| learned per-step weight + uncertainty | 6.494 | worse |

Findings:
1. **Fusion beats wheel-only** once VO is correctly scaled (const blend −8%).
2. **Anticipatory risk gating is the best policy** (−10%): trusting VO only when
   the monitor predicts low future degradation beats both wheel-only and naive
   always-on fusion.
3. **Uncertainty gating does *not* robustly help ATE.** The three-way policy is
   worse than risk-only on test — the uncertainty threshold overfit the 6 val
   runs. (The earlier "+5% from uncertainty" was an artifact of the inflated-VO
   bug.) Uncertainty still *describes* unreliable frames well — the UNCERTAIN
   state isolates the worst VO (true error 0.49 vs 0.09 for SAFE) and the worst
   risk-prediction error (0.22 vs 0.03) — but acting on it as a hard gate did not
   lower trajectory error here.
4. **A learned per-step weight trained on per-frame twist error HURTS badly.** It
   over-trusts angular VO (learns a_ang≈0.46) because instantaneous twist MSE does
   not penalize the way angular error *compounds* into heading drift. The correct
   objective is trajectory ATE itself — which the tuned constant per-dim blend
   optimizes directly. This is a clean methodological point: per-frame accuracy ≠
   trajectory accuracy.

## Honest overall framing for the paper
- Positive: an anticipatory VO-degradation monitor (Q1, +0.84) that is calibrated
  with discriminative uncertainty (Q3), and whose risk prediction drives a gating
  policy that **beats wheel-only and naive fusion by ~10%** (Q4).
- Negative/nuanced (all honest, all interesting): vision adds nothing at frame
  level (Q2); uncertainty's *decision* value beyond risk is not robustly shown
  (Q4); naive per-step learned fusion underperforms a tuned constant blend (Q4).
- Caveats unchanged: one environment, offline replay, 12 test runs (gains are
  run-averaged without tight CIs). These remain the binding limits for CoRL.
