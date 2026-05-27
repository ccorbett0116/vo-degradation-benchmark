# Anticipating Visual-Odometry Degradation: A Controlled Dataset and Trust-Monitoring Benchmark

Visual odometry (VO) is a valuable but unreliable localization source: its accuracy
collapses under motion blur, glare, low texture, and adverse camera geometry, and
its self-reported covariance is a poor trust signal (correlation with observed error
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
orientation** (pan/tilt), and **VO twist / feature / covariance** outputs.

As **ground-truth accuracy** we record, per run, the **tape-measured figure-8
loop-closure error** (`physical_lce_m` in the manifest): because the robot returns
to its start, the measured start–end gap is a direct, externally measured proxy for
accumulated localization error. Camera-off runs close the loop to ~0.12 m
(0.09–0.20 m); with the camera on, measured loop-closure error rises to a mean of
~0.99 m (up to 3.9 m) — an ~8× degradation that is the measured evidence that
naïvely fusing low-cost VO is, on average, harmful.

A dense (~78 Hz) **onboard fused-pose** estimate (`fused_pose`: a complementary
filter of wheel + IMU-yaw + inlier-weighted VO, ≤50% VO) is also logged. It is
**not independent of VO** and is **not** metric ground truth — on camera-on runs it
correlates only weakly (Pearson *r* ≈ 0.14) with the measured loop-closure error.
We therefore use it only for per-step VO-*divergence* signals and validate all
predictive claims at the run level against the **measured** `physical_lce_m`.

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
| `fused_pose` | (N, 3) | onboard reference [x, y, θ] — wheel+IMU+VO, *not* independent GT |
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

> Risk/degradation is the **future VO divergence** from the onboard fused estimate:
> `‖vo_twist − gt_twist‖` averaged over the next ~1 s, where `gt_twist` is the
> windowed derivative of `fused_pose`. Because `fused_pose` is **not** independent of
> VO, this is a *divergence* target, not error-vs-truth; we validate it at the run
> level against the **measured** `physical_lce_m`. The wheel-derived disagreement
> remains a runtime-observable *feature*.

**(1) Can a model forecast VO degradation in the next ~1 s** from current state and
history, giving the robot time to down-weight VO before bad estimates propagate?

**(2) Do camera images add predictive value** beyond robot- and VO-health signals,
especially when VO health is missing, delayed, or unreliable?

**(3) Are predicted risk scores calibrated**, and can predictive uncertainty flag
unreliable predictions?

**(4) Can risk plus uncertainty drive a three-way trust decision** — safe, risky,
or uncertain — mapping to using, down-weighting, or falling back from VO?

Together, these test whether VO degradation can be **predicted, explained, and
acted on** — and, critically, whether the prediction tracks *measured* localization
error (it does not; see the central finding below).

---

## Reference results (12 held-out test runs)

Full detail and methodology in [`analysis/CORRECTED_RESULTS.md`](analysis/CORRECTED_RESULTS.md)
and the per-question `analysis/q*/RQ*_findings.md`.

| Q | Result |
|---|---|
| **Q1 — prediction** | The near-future VO–fused **divergence** is highly predictable ~1 s ahead (Spearman **+0.840**). But this divergence is *not* the same as measured localization error — see the central finding below. |
| **Q2 — vision** | Vision-augmented Spearman **+0.839** vs signals-only **+0.840** — **no frame-level gain**; the CNN never becomes load-bearing (a robust null at this resolution / single-frame setting). |
| **Q3 — calibration** | Deep ensemble of Gaussian-NLL MLPs. Under-confident raw (CE 0.221) → **CE 0.016** after a single validation-tuned variance scale. Uncertainty is discriminative: keeping the 25% least-uncertain frames cuts MAE 0.115 → **0.032** (AUSE 0.171). |
| **Q4 — trust control** | Replay ATE is measured only against the `fused_pose` onboard reference, which contains VO and is **not** independent GT — so this is a *relative-consistency* measure, **not** a localization-accuracy claim, and **no localization improvement is claimed**. (No per-step measured GT exists to score a replayed policy.) The decisive test is the measured-GT validation below. For reference under the relative metric: a learned per-step weight on per-frame twist error is worse than a tuned blend, since per-frame accuracy ≠ trajectory accuracy. |

## Central finding (cautionary)

Out-of-fold over **all 60 camera-on runs**, the predicted risk correlates **+0.64**
with the filter's self-estimated drift (`estimated_lce_m`) but only **+0.06** with
the tape-measured loop-closure error (`physical_lce_m`) — because the onboard fused
estimate is *itself* nearly uncorrelated with measured accuracy (**+0.14**). The
monitor flags the **wrong regime**: static / upward-tilted viewpoints draw the
*highest* predicted risk yet achieve the *best* measured accuracy (~0.15–0.37 m),
while slow-pan motion draws low risk but the *worst* measured error (~1.69 m).

> **An onboard fused-pose estimate is a seductive but invalid label for VO
> degradation — a model trained on it predicts the estimate, not reality.**

So: the VO–fused divergence is predictable (Q1) and its uncertainty is calibrated
and discriminative (Q3), and vision adds nothing at the frame level (Q2) — but none
of that yields a localization-protective trust decision, because the label does not
track measured accuracy. Reliable introspective localization needs **externally
measured ground truth**, which this dataset provides. See
[`analysis/measured_gt_validation.json`](analysis/measured_gt_validation.json) and
`analysis/fig_measured_gt_gap.png`.

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
  measured_gt_validation.py     # out-of-fold risk vs MEASURED loop-closure (central finding)
analysis/
  CORRECTED_RESULTS.md          # master results + the VO-scaling correction
  measured_gt_validation.json   # run-level risk vs physical_lce_m / estimated_lce_m
  make_paper_figures.py         # regenerates the figures below
  fig_measured_gt_gap.png       # centerpiece: risk tracks estimate (+0.64), not GT (+0.06)
  fig_dataset_loopclosure.png   # measured loop-closure by regime (camera-off vs on)
  fig_condition_mismatch.png    # predicted risk vs measured error, per condition
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
run-averaged without confidence intervals). **No per-timestep metric ground truth:**
the only externally measured accuracy is the per-run tape-measured loop-closure
error; the dense `fused_pose` reference contains VO and is used only for relative
per-step divergence signals. These bound the generality of the conclusions and are
stated plainly rather than hidden.

## License / citation

TODO: add a license (e.g. CC-BY for the dataset, MIT for the code) and a citation
entry before public release.
