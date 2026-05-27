> **⚠ SUPERSEDED — Option A (2026-05-27). Localization claims below are RETRACTED.**
> ATE here is measured against `fused_pose` — an onboard wheel+IMU+VO complementary
> filter, **not** independent ground truth — so it is a relative-consistency metric,
> not accuracy, and no per-step measured GT exists to score a replayed policy. Against
> the dataset's only measured GT, predicted risk is uncorrelated with the loop-closure
> error (`physical_lce_m` ≈ +0.06 over 60 runs). **No localization improvement is
> claimed.** See [`../CORRECTED_RESULTS.md`](../CORRECTED_RESULTS.md) and
> `../measured_gt_validation.json`.
>
> **⚠ Superseded — conclusions CHANGED by the VO-scaling fix.** Corrected story:
> risk-gated fusion beats wheel-only by **10%** (2.04 vs 2.28 m), but the
> uncertainty gate does **not** robustly help ATE (overfit val). The earlier
> "+5% from uncertainty" was a bug artifact. See
> [`../CORRECTED_RESULTS.md`](../CORRECTED_RESULTS.md).

# Research Question 4 — Can Uncertainty Improve the Robot's VO-Trust Decision?

## Goal

Q3 showed the risk model can emit a calibrated, discriminative uncertainty. This question asks whether that uncertainty is *actionable*: does giving the robot a third "I don't know" state — on top of "VO is safe" and "VO is risky" — produce better trust decisions than reasoning about risk alone?

We define a three-state policy from Q3's recalibrated outputs (risk `mu`, uncertainty `sigma`):

| State | Condition | Action |
|---|---|---|
| **SAFE** | low risk, low uncertainty | trust VO (blend it into the pose estimate) |
| **RISKY** | high predicted risk | reject VO, fall back to wheel odometry |
| **UNCERTAIN** | high predictive uncertainty | conservative wheel fallback |

We test it two ways, both offline replays of the 12 held-out test runs (nothing runs on the robot). All thresholds are tuned on the validation runs only.

---

## (A) What the three states actually capture

At an operating point fixed on validation (RISKY above median validation risk; UNCERTAIN above the 80th-percentile validation uncertainty, taking precedence), we look at what each state contains on the test set:

| State | frames | fraction | mean *true* future VO error | mean risk-prediction error |
|---|---|---|---|---|
| SAFE | 20,439 | 41% | **1.04** | 0.44 |
| RISKY | 9,843 | 20% | 2.32 | 0.62 |
| UNCERTAIN | 19,227 | 39% | **6.34** | **3.83** |

This is the central result (see `q4_trust_policy.png` panel A). The UNCERTAIN state isolates a population that is **simultaneously**:

- the *worst actual VO* (true future-badness 6.34, vs 1.04 in SAFE), and
- where the *risk model itself is least reliable* (its prediction is off by 3.83 on average, vs 0.44 in SAFE).

That second point is what risk alone cannot see. A risk-only policy reads `mu` and acts on it; but precisely on these frames `mu` is untrustworthy. The uncertainty head flags them so the robot can abstain instead of acting on a bad risk estimate. This is the qualitative case that uncertainty adds something a risk score cannot.

**A caveat we state plainly.** When we *rank* frames purely to decide which VO to accept, an uncertainty-penalized score `mu + z·sigma` did **not** beat ranking by `mu` alone — validation tuning selected `z = 0`. Risk and uncertainty are correlated (the model is most uncertain where it predicts high risk), so for the *ordering* of frames, risk already does the job. Uncertainty's distinct value shows up not in re-ranking, but in (i) flagging where the risk score is unreliable (above) and (ii) the trajectory policy below.

---

## (B) Does it improve localization? Trajectory ATE vs ground truth

We replay each test run, integrating the body twist chosen by each policy, and measure absolute trajectory error (ATE) against `fused_pose` — the dataset's high-quality LiDAR/SLAM reference (figure-8 loop-closure ~0.05–0.21 m).

**Robust fusion, not hard switching.** Raw VO twist is heavy-tailed (the parent dataset analysis found VO linear estimates up to ±30 m/s on a 0.2 m/s robot); naively dead-reckoning it explodes (ATE ≈ 16 m). So a trusted frame contributes a *clipped, blended* twist `blend·VO + (1−blend)·wheel`, with the blend weight tuned per policy on validation. This matches the project's standing finding that the localization win comes from *fusion*, not from switching to raw VO.

| Policy | tuned knobs (on val) | test ATE (m) |
|---|---|---|
| wheel-only | — | **2.28** |
| const blend (trust all VO) | blend 0.75 | 2.92 |
| risk-gated | blend 0.25, risk<1.12 | 2.47 |
| **three-way (risk + uncertainty)** | blend 0.25, risk<1.12, **unc<0.50** | **2.35** |

Reading the table (and `q4_trust_policy.png` panel B):

1. **Trusting VO indiscriminately hurts** — const blend (2.92) is worse than ignoring VO entirely (2.28). On these runs VO is, on average, a liability.
2. **Risk gating recovers most of the damage** — only trusting low-risk frames brings ATE back to 2.47.
3. **Adding the uncertainty gate is a clean, controlled win.** The tuned three-way policy is *identical* to the risk-only policy (same blend 0.25, same risk threshold 1.12) **plus** the uncertainty screen. That single addition cuts ATE from 2.47 to **2.35 — a 5.0% improvement** attributable entirely to the uncertainty state. It removes exactly the catastrophic, model-unreliable frames identified in part (A).

---

## Conclusion

**Yes — uncertainty improves the trust decision, with two honest qualifications.**

- The uncertainty state earns its keep by isolating frames that are both catastrophic for VO and where the risk model is itself unreliable (the 39% UNCERTAIN bucket: true error 6.3, prediction error 3.8). Holding the risk gate fixed, adding the uncertainty screen improves trajectory ATE by 5%.
- It does *not* help for the narrower task of merely ranking which VO frames to accept — risk alone suffices there, because risk and uncertainty are correlated.
- On these particular runs the best VO-trusting policy only *matches* wheel-only odometry (2.35 vs 2.28); it does not beat it. Wheel odometry is already strong here and VO is mostly harmful, so the best a trust policy can do is learn to ignore VO on the right frames — and uncertainty helps it pick those frames. (The parent project's separate headroom analysis shows larger fusion gains are reachable with per-step optimal weighting; matching that with a *causal* learned weight, rather than a gated global blend, is the natural next step.)

The deployment takeaway: a calibrated risk-plus-uncertainty monitor lets the robot do better than acting on risk alone — chiefly by knowing when to *not* act on its own risk estimate.

---

## Known limitations

- **Coarse action model.** The policy uses a single global blend weight and hard thresholds. A learned *per-step* blend weight (regressed from risk + uncertainty) would likely extract more of the available headroom and is the obvious follow-up.
- **Proxy label.** Risk/uncertainty are trained against the disagreement proxy (inherited from Q1), not metrology-grade VO error. The ATE evaluation, however, is against true `fused_pose`, so the localization numbers are real.
- **12 test runs.** The 5% ATE gain is a modest, run-averaged effect on a small held-out set; it is a controlled comparison (identical except the uncertainty gate) but would benefit from more runs to tighten the confidence interval.
- **VO pipeline.** The m/s reconstruction of VO twist (change-point detection from the logger) is noisier than a native VO velocity stream would be; some of const-blend's poor showing reflects that reconstruction noise, not only VO itself.

---

*Code: `src/research_question_4.py` (+ `src/common_q34.py`). Reuses the Q3 ensemble (`analysis/q3/q3_member_*.pt`) and recalibration scale. Summary: `analysis/q4/q4_summary.json`. Plot: `analysis/q4/q4_trust_policy.png`. Split: `Data/comprehensive_dataset/split.json`.*
