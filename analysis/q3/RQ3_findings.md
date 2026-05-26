> **⚠ Superseded numbers** (pre VO-scaling fix + GT label). Corrected: point
> Spearman **+0.854**, CE 0.221→**0.016**, AUSE **0.171**. Conclusions hold.
> See [`../CORRECTED_RESULTS.md`](../CORRECTED_RESULTS.md).

# Research Question 3 — Is the Risk Prediction Calibrated, and Can Uncertainty Flag Unreliable Predictions?

## Goal

Q1 showed the 1-second-ahead VO-degradation risk is predictable. But a single risk number is not enough for a robot to act on: it needs to know **how much to trust each prediction**. This question asks two things:

1. **Calibration** — when the model reports a risk with some confidence, is that confidence honest? If it emits a 90% interval, does the true future-badness actually land inside 90% of the time?
2. **Uncertainty → reliability** — can the model's own uncertainty pick out the frames where its prediction is wrong? An uncertainty estimate is only useful if discarding high-uncertainty predictions actually removes the errors.

This is the prerequisite for Q4: a three-state trust policy (safe / risky / uncertain) is only meaningful if the uncertainty is both calibrated and discriminative.

---

## Setup

**Identical data to Q1/Q2**: same 60 Phase B+C runs, same per-frame disagreement label (VO twist in m/s vs commanded twist), same 1-second future-mean target, same standardized 29-dimensional tabular feature vector, same committed run-level split (42 train / 6 val / 12 test). Q2 found the camera image adds nothing beyond the tabular VO-health signals at the frame level, so Q3 builds on the **signals-only** model.

The only change is that the model now reports uncertainty.

---

## Model — deep ensemble of Gaussian MLPs

Each ensemble member is the Q1 MLP with a second output head:

```
Input (29) → Linear(29→128) → ReLU → Dropout(0.1)
           → Linear(128→128) → ReLU → Dropout(0.1)
           → { mu_head:  Linear(128→1)        # predicted risk (mean)
             { logvar_head: Linear(128→1)      # predicted log-variance
```

- **Loss:** Gaussian negative log-likelihood. The variance head learns *aleatoric* (data) noise — how noisy the future-badness is at this input, regardless of model.
- **Ensemble:** 5 members, different random seeds, 30 epochs each, AdamW (lr 1e-3, wd 1e-4), batch 4096. Best-validation-NLL checkpoint kept per member. Training is ~seconds per member on a single GPU.
- **Predictive distribution:** the Gaussian mixture over members, summarized by its mean and variance via the law of total variance:

  > `mu = mean_k mu_k`  ·  `var = mean_k(var_k + mu_k²) − mu²`

  The first term is averaged aleatoric noise; the spread of the member means adds *epistemic* (model) uncertainty, which grows where members disagree.

**Point-prediction sanity check:** the ensemble mean reproduces Q1's ranking quality — **Spearman +0.770** on the 12 test runs (Q1 reference +0.762). Adding the uncertainty machinery did not cost predictive accuracy. (RMSE is higher at 5.98 vs Q1's 4.70 because NLL training optimizes likelihood, not squared error, and tolerates larger point errors where it also reports large variance — exactly the behavior we want for calibration.)

---

## (A) Calibration

For nominal central-interval probabilities p ∈ {0.1, …, 0.9}, we measure the empirical fraction of test targets that fall inside the predictive Gaussian's p-interval. A perfectly calibrated model lands on the diagonal. Calibration error (CE) is the mean absolute gap between nominal and observed coverage.

| Model | CE | 90% interval actually covers |
|---|---|---|
| Raw ensemble | **0.187** | 0.96 — **under-confident** (intervals too wide) |
| Recalibrated (σ × 0.61) | **0.036** | ≈ 0.90 |

The raw ensemble is systematically **under-confident**: its predicted intervals are wider than they need to be (a 90% interval catches 96% of targets). This is the expected direction for a deep ensemble whose members each also report aleatoric variance — the two uncertainty sources stack.

**Post-hoc recalibration** fixes it with a single scalar. We tune one variance-scale factor `s` on the **validation runs only**, minimizing validation coverage error, then apply it to test. The tuned `s = 0.61` (narrowing the intervals) cuts test CE from 0.187 to **0.036** — essentially calibrated. See `q3_calibration_sparsification.png` panel A: the recalibrated curve sits on the diagonal.

*(Note: scaling variance to minimize NLL instead gave `s≈1.06` and did **not** improve coverage — NLL and coverage are different objectives. We recalibrate against the metric we report.)*

---

## (B) Uncertainty → reliability (sparsification)

A calibrated uncertainty is still useless if it does not concentrate on the actual errors. The sparsification (error-retention) curve tests this directly: rank test frames from most- to least-uncertain, then keep only the least-uncertain fraction and recompute MAE on what remains.

| Retained (least-uncertain) | MAE |
|---|---|
| 100% (full set) | 1.79 |
| 25% | **0.43** |

- **Discarding the high-uncertainty frames removes the errors.** Keeping the most-confident 25% of predictions drops MAE from 1.79 to 0.43 — a 4× reduction.
- **AUSE = 0.134** (normalized area between our curve and the oracle that ranks by true error). The model's uncertainty ranking is close to the best-possible ranking; see panel B, where the "by uncertainty" curve tracks the oracle and sits far below the random baseline.
- **Spearman(σ, |error|) = +0.559** — predicted uncertainty correlates substantially with realized error at the frame level.

---

## Conclusion

**Yes on both counts.** Out of the box the ensemble is *discriminative but under-confident*: its uncertainty reliably ranks predictions by reliability (AUSE 0.134, 4× error reduction on the confident quartile), while its raw intervals are too wide. A single validation-tuned variance scale makes it well-calibrated (CE 0.187 → 0.036) without touching the point predictions or the ranking.

So the risk monitor can emit an honest confidence with each prediction, and that confidence genuinely separates trustworthy predictions from unreliable ones. This is exactly the signal Q4 needs to add an "uncertain" state to the trust policy.

---

## Known limitations

- **Calibration is corrected globally, not per-condition.** A single scale factor fixes average coverage; per-condition intervals may still be slightly off (e.g. the centered condition, where the label has little variance).
- **The label is still the disagreement proxy** (inherited from Q1), so calibration is against proxy future-badness, not metrology-grade VO error.
- **Aleatoric + epistemic are reported jointly.** We did not separate how much of the useful uncertainty comes from the variance head vs the ensemble spread; an ablation (single-member NLL vs full ensemble) would quantify each.

---

*Code: `src/research_question_3.py` (+ `src/common_q34.py`). Ensemble: `analysis/q3/q3_member_{0..4}.pt`. Predictions + tuned scale: `analysis/q3/q3_test_predictions.npz`. Summary: `analysis/q3/q3_summary.json`. Plot: `analysis/q3/q3_calibration_sparsification.png`. Split: `Data/comprehensive_dataset/split.json`.*
