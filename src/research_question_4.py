"""
Research Question 4
-------------------
Can the model's uncertainty improve the robot's VO-trust decision?

We turn the calibrated risk + uncertainty from Q3 into a three-state policy:

    SAFE      low predicted risk, low uncertainty   -> trust VO
    RISKY     high predicted risk (confidently bad) -> reject VO, use wheel odom
    UNCERTAIN high predictive uncertainty           -> conservative wheel fallback

The claim under test: adding the UNCERTAIN state (using Q3's uncertainty) beats a
risk-only policy that has no notion of "I don't know".

Two evaluations on the 12 held-out test runs:

 (A) SELECTIVE-TRUST cost curve, in Jose's disagreement framework.
     Sweep how much VO we accept ("VO usage") and measure the mean realized
     future-badness on the accepted frames ("accepted VO error"). Compare:
       - risk-only:        rank frames by predicted risk mu
       - risk+uncertainty: rank by an uncertainty-penalized score mu + z*sigma
     Lower accepted error at matched usage = uncertainty improves the decision.

 (B) TRAJECTORY ATE vs fused_pose ground truth (the dataset's high-quality
     LiDAR/SLAM pose, loop-closure ~0.05-0.21 m). We replay each run, integrate
     the body twist chosen by each policy, and measure absolute trajectory error
     against fused_pose. Thresholds are tuned on the validation runs only.

Everything is offline replay on logged data; nothing runs on the robot.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py
import torch

import common_q34 as c
import research_question_1 as rq1
from research_question_3 import GaussianMLP, ensemble_predict, N_MEMBERS

Q3_DIR = c.REPO / "analysis" / "q3"
OUT_DIR = c.REPO / "analysis" / "q4"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------
# Load the Q3 ensemble + recalibration scale
# --------------------------------------------------------------------------
def load_ensemble(in_dim):
    models = []
    for k in range(N_MEMBERS):
        m = GaussianMLP(in_dim).to(DEVICE)
        m.load_state_dict(torch.load(Q3_DIR / f"q3_member_{k}.pt", map_location=DEVICE))
        m.eval()
        models.append(m)
    s = json.loads((Q3_DIR / "q3_summary.json").read_text())["variance_scale"]
    return models, float(s)


# --------------------------------------------------------------------------
# (A) selective-trust cost curve
# --------------------------------------------------------------------------
def accepted_error_curve(score, y, usages):
    """Rank frames by `score` ascending (low score = more trustworthy). For each
    target VO-usage fraction, return mean realized future-badness y over the
    accepted (lowest-score) frames."""
    order = np.argsort(score)
    y_sorted = y[order]
    out = []
    n = len(y)
    for u in usages:
        k = max(1, int(round(u * n)))
        out.append(float(y_sorted[:k].mean()))
    return np.array(out)


def pick_z(mu_va, sigma_va, y_va, usages):
    """Choose the uncertainty penalty z (in score = mu + z*sigma) that minimizes
    the area under the validation accepted-error curve."""
    best_z, best_area = 0.0, np.inf
    for z in np.linspace(0.0, 3.0, 31):
        curve = accepted_error_curve(mu_va + z * sigma_va, y_va, usages)
        area = float(np.trapezoid(curve, usages))
        if area < best_area:
            best_area, best_z = area, z
    return best_z


# --------------------------------------------------------------------------
# (B) trajectory replay vs fused_pose
# --------------------------------------------------------------------------
def integrate(ts, lin, ang, p0):
    """Vectorized midpoint dead-reckoning of a body twist [lin (m/s), ang (rad/s)]
    from an initial pose p0=[x,y,theta]. Returns (x[], y[])."""
    dts = np.diff(ts)
    v, w = lin[:-1], ang[:-1]
    th0 = float(p0[2])
    # heading just before each step, then midpoint heading during the step
    th_before = th0 + np.concatenate([[0.0], np.cumsum(w * dts)[:-1]])
    th_mid = th_before + 0.5 * w * dts
    dx = v * np.cos(th_mid) * dts
    dy = v * np.sin(th_mid) * dts
    xs = float(p0[0]) + np.concatenate([[0.0], np.cumsum(dx)])
    ys = float(p0[1]) + np.concatenate([[0.0], np.cumsum(dy)])
    return xs, ys


def ate(xs, ys, gx, gy):
    """Absolute trajectory error: RMS position error vs ground-truth path."""
    return float(np.sqrt(np.mean((xs - gx) ** 2 + (ys - gy) ** 2)))


def run_frames(f, run_id, models, scale, mu_stats, sd_stats):
    """For one run, return the full per-frame arrays needed for replay, plus
    per-frame predicted risk mu and uncertainty sigma (NaN where no prediction
    is available, e.g. first 0.5 s before the past window fills)."""
    sig = rq1.compute_run_signals(f, run_id)
    rid = f["run_id"][:]
    idx = np.where(rid == run_id)[0]
    fused = f["fused_pose"][idx[0]:idx[-1] + 1]  # contiguous per run

    X, valid = rq1.build_features(sig, rq1.PAST_WINDOW_S)
    mu = np.full(len(X), np.nan); sigma = np.full(len(X), np.nan)
    if valid.any():
        Xs = (X[valid] - mu_stats) / sd_stats
        m, v = ensemble_predict(models, Xs.astype(np.float32))
        mu[valid] = m
        sigma[valid] = np.sqrt(v) * scale
    return dict(
        ts=sig["ts"], vo=sig["vo_mps"], wheel=sig["wheel_cmd"],
        fused=fused, mu=mu, sigma=sigma, cond=None,
    )


# VO twist is heavy-tailed (parent project: linear estimates up to +/-30 m/s on a
# 0.2 m/s robot). Raw integration is wrecked by these outliers, so robust fusion
# clips VO to a generous physical envelope before blending with wheel odometry.
VO_LIN_CLIP = 0.5   # m/s   (robot tops out ~0.2)
VO_ANG_CLIP = 1.0   # rad/s (robot tops out ~0.35)


def policy_twist(fr, policy, tau_risk, tau_unc, blend):
    """Per-frame fused body twist. `blend=(a_lin, a_ang)` are PER-DIMENSION VO
    weights (VO linear is reliable, VO angular is noisy, so they differ). On
    *trusted* frames twist = a*vo_clipped + (1-a)*wheel per axis; elsewhere
    wheel-only. Returns (lin[], ang[], trust_mask)."""
    vo, wheel = fr["vo"], fr["wheel"]
    mu, sigma = fr["mu"], fr["sigma"]
    a_lin, a_ang = blend
    vo_ok = ~np.isnan(vo).any(axis=1)
    have_pred = ~np.isnan(mu)

    if policy == "wheel":
        trust = np.zeros(len(mu), dtype=bool)
    elif policy == "const_blend":           # trust all valid VO (no gating)
        trust = vo_ok
    elif policy == "risk_only":             # gate by predicted risk only
        trust = vo_ok & have_pred & (mu < tau_risk)
    elif policy == "three_way":             # gate by risk AND uncertainty
        safe = have_pred & (mu < tau_risk) & (sigma < tau_unc)
        trust = vo_ok & safe
    else:
        raise ValueError(policy)

    vo_lin = np.clip(vo[:, 0], -VO_LIN_CLIP, VO_LIN_CLIP)
    vo_ang = np.clip(vo[:, 1], -VO_ANG_CLIP, VO_ANG_CLIP)
    lin = np.where(trust, a_lin * vo_lin + (1 - a_lin) * wheel[:, 0], wheel[:, 0])
    ang = np.where(trust, a_ang * vo_ang + (1 - a_ang) * wheel[:, 1], wheel[:, 1])
    return lin, ang, trust


def mean_ate(frames, policy, tau_risk, tau_unc, blend):
    vals = []
    for fr in frames:
        lin, ang, _ = policy_twist(fr, policy, tau_risk, tau_unc, blend)
        xs, ys = integrate(fr["ts"], lin, ang, fr["fused"][0])
        vals.append(ate(xs, ys, fr["fused"][:, 0], fr["fused"][:, 1]))
    return float(np.mean(vals)), vals


# Per-dimension VO blend weights to search (linear, angular searched separately).
BLEND_GRID = np.array([0.0, 0.25, 0.5, 0.75, 1.0])


def tune_policy(frames, policy, mu_grid, unc_grid):
    """Grid-search the policy's free knobs (per-dim blend weights, thresholds
    where applicable) to minimize mean validation ATE. Returns (best_ate, params)."""
    blends = [(bl, ba) for bl in BLEND_GRID for ba in BLEND_GRID]
    best, params = np.inf, {}
    if policy == "const_blend":
        for b in blends:
            a, _ = mean_ate(frames, policy, np.inf, np.inf, b)
            if a < best:
                best, params = a, dict(blend=list(b))
    elif policy == "risk_only":
        for b in blends:
            for tr in mu_grid:
                a, _ = mean_ate(frames, policy, tr, np.inf, b)
                if a < best:
                    best, params = a, dict(blend=list(b), tau_risk=float(tr))
    elif policy == "three_way":
        for b in blends:
            for tr in mu_grid:
                for tu in unc_grid:
                    a, _ = mean_ate(frames, policy, tr, tu, b)
                    if a < best:
                        best, params = a, dict(blend=list(b), tau_risk=float(tr),
                                               tau_unc=float(tu))
    return best, params


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    print(f"device={DEVICE}")
    d = c.build_dataset()
    models, scale = load_ensemble(d["Xtr"].shape[1])
    print(f"loaded {len(models)} ensemble members; variance scale={scale:.2f}")

    # Recalibrated predictions on the valid/labeled frames (for the cost curve).
    mu_va, var_va = ensemble_predict(models, d["Xva"]); sig_va = np.sqrt(var_va) * scale
    mu_te, var_te = ensemble_predict(models, d["Xte"]); sig_te = np.sqrt(var_te) * scale
    yva, yte = d["yva"], d["yte"]

    # ---- (A) Selective-trust cost curve ---------------------------------
    usages = np.linspace(0.1, 1.0, 19)
    z = pick_z(mu_va, sig_va, yva, usages)          # tuned on validation only
    curve_risk = accepted_error_curve(mu_te, yte, usages)
    curve_unc = accepted_error_curve(mu_te + z * sig_te, yte, usages)
    area_risk = float(np.trapezoid(curve_risk, usages))
    area_unc = float(np.trapezoid(curve_unc, usages))
    print(f"\n(A) Selective-trust  (uncertainty penalty z={z:.2f}, tuned on val)")
    print(f"    accepted-error area  risk-only        = {area_risk:.3f}")
    print(f"    accepted-error area  risk+uncertainty = {area_unc:.3f}"
          f"  ({100*(area_risk-area_unc)/area_risk:+.1f}% )")
    for u in (0.5, 0.7):
        i = np.argmin(np.abs(usages - u))
        print(f"    @ {usages[i]*100:.0f}% VO usage:  risk-only={curve_risk[i]:.3f}"
              f"   risk+unc={curve_unc[i]:.3f}")

    # ---- 3-state breakdown at a concrete operating point ----------------
    # Thresholds from validation: RISKY above val-median risk; UNCERTAIN above
    # the 80th-pct val uncertainty.
    tau_risk_op = float(np.median(mu_va))
    tau_unc_op = float(np.percentile(sig_va, 80))
    state = np.full(len(mu_te), "SAFE", dtype=object)
    state[mu_te >= tau_risk_op] = "RISKY"
    state[sig_te >= tau_unc_op] = "UNCERTAIN"   # uncertainty takes precedence
    abs_err = np.abs(mu_te - yte)
    print(f"\n3-state breakdown (op point: risk>={tau_risk_op:.2f}, unc>={tau_unc_op:.2f}):")
    print(f"    {'state':<10} {'n':>7} {'frac':>6} {'mean true y':>12} {'mean |pred err|':>16}")
    for s_name in ("SAFE", "RISKY", "UNCERTAIN"):
        m = state == s_name
        if m.any():
            print(f"    {s_name:<10} {m.sum():>7} {m.mean():>6.2f} "
                  f"{yte[m].mean():>12.3f} {abs_err[m].mean():>16.3f}")

    # ---- Conditional value of uncertainty (the crux) -------------------
    # Among frames a risk-only policy would TRUST (mu below the operating
    # threshold), does uncertainty still separate good from bad VO? If yes, the
    # UNCERTAIN state catches failures that risk alone waves through.
    risk_safe = mu_te < tau_risk_op
    hi_unc = risk_safe & (sig_te >= tau_unc_op)
    lo_unc = risk_safe & (sig_te < tau_unc_op)
    print("\nWithin the risk-'safe' set (mu < op threshold):")
    print(f"    low-uncertainty   n={lo_unc.sum():>6}  mean true y={yte[lo_unc].mean():.3f}")
    print(f"    high-uncertainty  n={hi_unc.sum():>6}  mean true y={yte[hi_unc].mean():.3f}"
          f"   <- risk-only trusts these blindly")

    # ---- (B) Trajectory ATE vs fused_pose -------------------------------
    f = h5py.File(d["hdf5_path"], "r")
    print("\n(B) Trajectory replay vs fused_pose ground truth (robust blend, VO clipped)")
    print("    loading val + test run frames ...")
    val_frames = [run_frames(f, r, models, scale, d["mu"], d["sd"]) for r in d["val_ids"]]
    test_frames = [run_frames(f, r, models, scale, d["mu"], d["sd"]) for r in d["test_ids"]]

    # Tune each policy's knobs on validation ATE only.
    mu_grid = np.quantile(mu_va, np.linspace(0.2, 0.95, 12))
    unc_grid = np.quantile(sig_va, np.linspace(0.3, 0.97, 10))
    tuned = {}
    for pol in ("const_blend", "risk_only", "three_way"):
        _, params = tune_policy(val_frames, pol, mu_grid, unc_grid)
        tuned[pol] = params
    print(f"    tuned on val: {json.dumps(tuned)}")

    def ate_of(pol):
        p = tuned.get(pol, {})
        return mean_ate(test_frames, pol,
                        p.get("tau_risk", np.inf), p.get("tau_unc", np.inf),
                        tuple(p.get("blend", (1.0, 1.0))))[0]

    results = {
        "wheel_only":  mean_ate(test_frames, "wheel", 0, 0, (0.0, 0.0))[0],
        "const_blend": ate_of("const_blend"),
        "risk_only":   ate_of("risk_only"),
        "three_way":   ate_of("three_way"),
    }
    print(f"\n    mean test ATE (m) vs fused_pose:")
    for k in ("wheel_only", "const_blend", "risk_only", "three_way"):
        print(f"      {k:<12} {results[k]:.3f}")
    imp_vs_wheel = 100 * (results["wheel_only"] - results["three_way"]) / results["wheel_only"]
    imp_vs_risk = 100 * (results["risk_only"] - results["three_way"]) / results["risk_only"]
    print(f"    three_way vs wheel_only: {imp_vs_wheel:+.1f}% ATE")
    print(f"    three_way vs risk_only:  {imp_vs_risk:+.1f}% ATE")

    # ---- Plots ----------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    states = ("SAFE", "RISKY", "UNCERTAIN")
    sy = [float(yte[state == s].mean()) for s in states]
    sfrac = [float((state == s).mean()) for s in states]
    bars = ax[0].bar(states, sy, color=["#2a2", "#c84", "#a33"])
    for b, s, fr in zip(bars, sy, sfrac):
        ax[0].text(b.get_x() + b.get_width() / 2, s,
                   f"{s:.1f}\n({fr*100:.0f}%)", ha="center", va="bottom")
    ax[0].set_ylabel("mean true future VO error (disagreement)")
    ax[0].set_title("(A) Actual VO badness by trust state")
    ax[0].grid(alpha=.3, axis="y")

    labels = ["wheel\nonly", "const\nblend", "risk\ngated", "three\nway"]
    vals = [results[k] for k in ("wheel_only", "const_blend", "risk_only", "three_way")]
    colors = ["#888", "#c84", "#48a", "#2a2"]
    ax[1].bar(labels, vals, color=colors)
    for i, v in enumerate(vals):
        ax[1].text(i, v, f"{v:.2f}", ha="center", va="bottom")
    ax[1].set_ylabel("mean trajectory ATE (m) vs fused_pose")
    ax[1].set_title("(B) Localization error by trust policy")
    ax[1].grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "q4_trust_policy.png", dpi=130)
    print(f"\nSaved plot -> {OUT_DIR/'q4_trust_policy.png'}")

    # ---- Save artifacts -------------------------------------------------
    summary = dict(
        cost_curve=dict(z=float(z), usages=usages.tolist(),
                        accepted_error_risk_only=curve_risk.tolist(),
                        accepted_error_risk_unc=curve_unc.tolist(),
                        area_risk_only=area_risk, area_risk_unc=area_unc),
        operating_point=dict(tau_risk=tau_risk_op, tau_unc=tau_unc_op,
                             state_counts={s: int((state == s).sum())
                                           for s in ("SAFE", "RISKY", "UNCERTAIN")},
                             state_mean_true_y={s: float(yte[state == s].mean())
                                                for s in ("SAFE", "RISKY", "UNCERTAIN")},
                             risk_safe_lo_unc_y=float(yte[lo_unc].mean()),
                             risk_safe_hi_unc_y=float(yte[hi_unc].mean())),
        trajectory_ate=dict(tuned=tuned, **{k: float(v) for k, v in results.items()},
                            improvement_vs_wheel_pct=imp_vs_wheel,
                            improvement_vs_risk_pct=imp_vs_risk),
    )
    (OUT_DIR / "q4_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved summary -> {OUT_DIR/'q4_summary.json'}")


if __name__ == "__main__":
    main()
