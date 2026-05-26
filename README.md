# Anticipating Visual-Odometry Degradation: A Controlled Dataset and Trust-Monitoring Benchmark

Visual odometry (VO) is a valuable but unreliable localization source: its accuracy
collapses under motion blur, glare, low texture, and adverse camera geometry, and
its self-reported covariance is a poor trust signal (correlation with true error
≈ −0.03). We ask whether a robot can **anticipate VO degradation early enough to
protect its localization**, and whether it can **know when to trust that
prediction**.

This repository contains the dataset, the benchmark tasks, and reference
baselines for that question.

---

## Dataset

We construct a dataset of **311,808 timestamped samples from 70 repeated figure-8
runs** in a single indoor environment (`window_room`), spanning three regimes:

- **Phase A — camera-off baseline** (10 runs): wheel odometry only, VO inactive.
- **Phase B — five static pan/tilt angles** (30 runs): camera held at center,
  full-left, full-right, full-down, full-up — isolating the effect of *where* the
  camera points.
- **Phase C — dynamic camera sweeps** (30 runs): slow pan, slow tilt, combined —
  isolating the effect of camera *motion*.

Each sample pairs a **120×160 RGB image** with **wheel odometry**, **camera
orientation** (pan/tilt), **VO twist / feature / covariance** outputs, and a
**high-quality fused pose** (LiDAR/SLAM, figure-8 loop-closure ~0.05–0.21 m) used
as ground truth. Per-step VO error is computed against this reference to label
degradation events.

This **controlled manipulation of camera pointing** — rather than incidental
"drive around and see what breaks" logging — is what makes the four questions
below answerable, and is the dataset's main distinguishing feature.

> **The 9.9 GB HDF5 (`training_data.hdf5`) is not stored in git** (GitHub file
> limits). The committed `Data/comprehensive_dataset/` holds the run `manifest.json`
> and the frozen train/val/test `split.json`. See [Data access](#data-access).

### HDF5 layout (shared sample axis)

| dataset | shape | notes |
|---|---|---|
| `run_id`, `phase`, `condition`, `route`, `environment` | (N,) | run metadata |
| `timestamps` | (N,) | Unix seconds (~78 Hz logging) |
| `fused_pose` | (N, 3) | **ground truth** [x, y, θ] |
| `wheel_twist` | (N, 2) | commanded [v, ω] — a command, not a measurement |
| `pan_tilt` | (N, 2) | camera orientation, normalized [0,1] |
| `images` | (N, 120, 160, 3) | uint8 RGB |
| `vo_twist` | (N, 2) | VO body velocity [v, ω]; NaN when camera off |
| `vo_features` | (N, 2) | [inliers, matches] |
| `vo_covariance` | (N, 6) | VO self-reported covariance (poorly calibrated) |

> **Units note / known pitfall.** `vo_twist` is already a **body velocity**
> (m/s, rad/s), on the same scale as the differentiated ground-truth pose. A
> plausible-looking "convert to m/s" reconstruction that divides by the
> inter-update period **inflates VO ~15×** and inverts the fusion conclusion in
> Q4. We document this explicitly because it is an easy trap; see
> [`analysis/CORRECTED_RESULTS.md`](analysis/CORRECTED_RESULTS.md).

---

## The four benchmark questions

> Risk/degradation is the **future VO error** vs ground truth:
> `‖vo_twist − gt_twist‖` averaged over the next ~1 s, where `gt_twist` is the
> windowed derivative of `fused_pose`. The wheel-derived disagreement remains a
> runtime-observable *feature*; ground truth is used only as the prediction target.

**(1) Can a model forecast VO degradation in the next ~1 s** from current state and
history, giving the robot time to down-weight VO before bad estimates propagate?

**(2) Do camera images add predictive value** beyond robot- and VO-health signals,
especially when VO health is missing, delayed, or unreliable?

**(3) Are predicted risk scores calibrated**, and can predictive uncertainty flag
unreliable predictions?

**(4) Can risk plus uncertainty drive a three-way trust decision** — safe, risky,
or uncertain — mapping to using, down-weighting, or falling back from VO?

Together, these test whether VO degradation can be **predicted, explained, and
acted on** reliably.

---

## Reference results (12 held-out test runs)

Full detail and methodology in [`analysis/CORRECTED_RESULTS.md`](analysis/CORRECTED_RESULTS.md)
and the per-question `analysis/q*/RQ*_findings.md`.

| Q | Result |
|---|---|
| **Q1 — prediction** | Risk Spearman **+0.840**; run-level corr. with measured VO error **+0.794**. VO degradation is predictable ~1 s ahead. |
| **Q2 — vision** | Vision-augmented Spearman **+0.839** vs signals-only **+0.840** — **no frame-level gain**; the CNN never becomes load-bearing (a robust null at this resolution / single-frame setting). |
| **Q3 — calibration** | Deep ensemble of Gaussian-NLL MLPs. Under-confident raw (CE 0.221) → **CE 0.016** after a single validation-tuned variance scale. Uncertainty is discriminative: keeping the 25% least-uncertain frames cuts MAE 0.115 → **0.032** (AUSE 0.171). |
| **Q4 — trust control** | Trajectory ATE vs `fused_pose`: wheel-only 2.276 m; constant fusion 2.086 m; **risk-gated fusion 2.041 m (−10% vs wheel, best)**. The three-way *uncertainty* gate does **not** robustly improve ATE (overfits validation); uncertainty isolates unreliable frames descriptively but acting on it as a hard gate did not help here. A learned per-step weight trained on per-frame twist error *hurts* (it over-trusts angular VO, which compounds into heading drift — per-frame accuracy ≠ trajectory accuracy). |

**Takeaway:** an anticipatory VO-degradation monitor (Q1) with calibrated,
discriminative uncertainty (Q3) can drive a risk-gating policy that beats both
wheel-only and naive fusion by ~10% (Q4). Vision adds nothing at the frame level
(Q2), and the *decision* value of uncertainty beyond risk is not robustly
established on this data.

---

## Repository structure

```
src/
  research_question_1.py        # Q1: anticipatory degradation prediction (MLP)
  research_question_2.py        # Q2: + image branch (vision value / masking)
  research_question_3.py        # Q3: deep-ensemble calibration + sparsification
  research_question_4.py        # Q4: 3-state trust policy + trajectory ATE
  research_question_4_learned.py# Q4: learned per-step fusion weight (ablation)
  common_q34.py                 # shared data pipeline for Q3/Q4
analysis/
  CORRECTED_RESULTS.md          # master results + the VO-scaling correction
  q1/ .. q4/                    # per-question findings, models, predictions, plots
Data/comprehensive_dataset/
  manifest.json                 # per-run metadata
  split.json                    # frozen 42/6/12 run-level train/val/test split
```

## Running

```bash
# Python 3.12, torch (CUDA), h5py, numpy, matplotlib
# Place training_data.hdf5 where the code can find it (see Data access), then:
cd src
python research_question_1.py          # Q1
python research_question_2.py          # Q2 (preloads images; needs ~18 GB RAM)
python research_question_3.py          # Q3 (trains the ensemble used by Q4)
python research_question_4.py          # Q4 policy + trajectory ATE
python research_question_4_learned.py  # Q4 learned-weight ablation
```

The split is run-level (whole runs held out) and frozen in `split.json`; all
results are reported on the 12-run test set, touched once.

## Data access

The 9.9 GB `training_data.hdf5` is distributed separately (release asset / archive
to be linked here). `manifest.json` and `split.json` in this repo describe and
partition it.

## Limitations

Single environment (`window_room`), single robot, figure-8 routes; evaluation is
offline replay (no closed-loop robot deployment); test set is 12 runs (gains are
run-averaged without confidence intervals). These bound the generality of the
conclusions and are stated plainly rather than hidden.

## License / citation

TODO: add a license (e.g. CC-BY for the dataset, MIT for the code) and a citation
entry before public release.
