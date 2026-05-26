"""
Research Question 3
-------------------
Is the 1-second-ahead VO-degradation risk prediction calibrated, and can the
model's own uncertainty identify when its prediction is unreliable?

Approach
  * Make the Q1 risk model *probabilistic*: each MLP outputs a mean mu(x) AND a
    variance sigma2(x), trained with the Gaussian negative-log-likelihood. This
    captures heteroscedastic data noise (aleatoric uncertainty).
  * Train a DEEP ENSEMBLE of 5 such MLPs (different seeds). The spread of the
    members' means adds model uncertainty (epistemic). The predictive
    distribution is the Gaussian mixture, summarized by its mean and variance
    (Lakshminarayanan et al. 2017).

Two evaluations on the 12 held-out test runs:
  (A) CALIBRATION. Using the predictive Gaussian, does a nominal p% central
      interval actually contain p% of the targets? We report the reliability
      curve and a calibration error, before and after a post-hoc variance
      rescaling tuned on the validation runs (Kuleshov-style recalibration).
  (B) UNCERTAINTY -> RELIABILITY. Sparsification / error-retention curve: drop
      the most-uncertain frames and watch the error on what remains. Compare to
      an oracle that drops by true error, and to random dropping. AUSE = area
      between our curve and the oracle (smaller = uncertainty tracks error well).

Everything is offline replay on logged data; nothing runs on the robot.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import common_q34 as c

OUT_DIR = c.REPO / "analysis" / "q3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_MEMBERS = 5
EPOCHS = 30
BATCH = 4096
LR = 1e-3
MIN_VAR = 1e-4  # floor on predicted variance for numerical stability


# --------------------------------------------------------------------------
# Probabilistic MLP: outputs (mu, log_var)
# --------------------------------------------------------------------------
class GaussianMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
        )
        self.mu_head = nn.Linear(hidden, 1)
        self.logvar_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        mu = self.mu_head(h).squeeze(-1)
        log_var = self.logvar_head(h).squeeze(-1)
        log_var = torch.clamp(log_var, math.log(MIN_VAR), 10.0)
        return mu, log_var


def gaussian_nll(mu, log_var, y):
    """Mean Gaussian negative log-likelihood (up to a constant)."""
    inv_var = torch.exp(-log_var)
    return 0.5 * (inv_var * (y - mu) ** 2 + log_var).mean()


def train_member(Xtr, ytr, Xva, yva, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = GaussianMLP(Xtr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
        batch_size=BATCH, shuffle=True,
    )
    Xva_t = torch.from_numpy(Xva).to(DEVICE)
    yva_t = torch.from_numpy(yva).to(DEVICE)
    best_val, best_state = math.inf, None
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            mu, lv = model(xb)
            loss = gaussian_nll(mu, lv, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            mu, lv = model(Xva_t)
            v = gaussian_nll(mu, lv, yva_t).item()
        if v < best_val:
            best_val = v
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_val


@torch.no_grad()
def member_predict(model, X):
    """Return (mu, var) numpy arrays for one member (eval mode)."""
    model.eval()
    mu, lv = model(torch.from_numpy(X).to(DEVICE))
    return mu.cpu().numpy(), np.exp(lv.cpu().numpy())


def ensemble_predict(models, X):
    """Gaussian-mixture mean and variance over ensemble members.

    mu_bar = mean_k mu_k
    var    = mean_k (var_k + mu_k^2) - mu_bar^2   (law of total variance)
    """
    mus, vars_ = [], []
    for m in models:
        mu, var = member_predict(m, X)
        mus.append(mu); vars_.append(var)
    mus = np.stack(mus); vars_ = np.stack(vars_)
    mu_bar = mus.mean(axis=0)
    var = (vars_ + mus ** 2).mean(axis=0) - mu_bar ** 2
    var = np.clip(var, MIN_VAR, None)
    return mu_bar, var


# --------------------------------------------------------------------------
# (A) Calibration
# --------------------------------------------------------------------------
NORM = torch.distributions.Normal(0.0, 1.0)


def coverage_curve(mu, sigma, y, levels):
    """For each nominal central-interval probability p, the empirical fraction
    of targets that fall inside the predicted interval."""
    z = np.array([NORM.icdf(torch.tensor(0.5 + p / 2)).item() for p in levels])
    obs = []
    for p, zz in zip(levels, z):
        lo, hi = mu - zz * sigma, mu + zz * sigma
        obs.append(float(np.mean((y >= lo) & (y <= hi))))
    return np.array(obs)


def calibration_error(levels, observed):
    """Mean absolute deviation between nominal and observed coverage."""
    return float(np.mean(np.abs(observed - levels)))


def fit_variance_scale(mu, sigma, y, levels, grid=None):
    """Find scalar s that minimizes the *coverage* calibration error on the
    validation set with sigma' = s * sigma. A single positive scale factor
    corrects systematic over-confidence (s>1 widens) or under-confidence (s<1).
    We optimize the same coverage metric we report, rather than NLL, so the
    recalibration directly targets calibration."""
    if grid is None:
        grid = np.geomspace(0.2, 5.0, 120)
    best_s, best_ce = 1.0, math.inf
    for s in grid:
        ce = calibration_error(levels, coverage_curve(mu, s * sigma, y, levels))
        if ce < best_ce:
            best_ce, best_s = ce, s
    return best_s


# --------------------------------------------------------------------------
# (B) Sparsification / error-retention
# --------------------------------------------------------------------------
def retention_curve(err, order, fractions):
    """Mean error over the best-retained fraction, where `order` ranks frames
    from MOST to LEAST uncertain (we keep the least-uncertain tail)."""
    n = len(err)
    err_sorted = err[order]            # most-uncertain first
    out = []
    for fr in fractions:
        k = max(1, int(round(fr * n)))
        out.append(float(err_sorted[n - k:].mean()))  # keep least-uncertain k
    return np.array(out)


def main():
    print(f"device={DEVICE}  members={N_MEMBERS}")
    d = c.build_dataset()
    Xtr, ytr = d["Xtr"], d["ytr"]
    Xva, yva = d["Xva"], d["yva"]
    Xte, yte = d["Xte"], d["yte"]

    # -- Train the deep ensemble ------------------------------------------
    models = []
    for k in range(N_MEMBERS):
        m, vnll = train_member(Xtr, ytr, Xva, yva, seed=k)
        models.append(m)
        print(f"  member {k+1}/{N_MEMBERS}  val_nll={vnll:.4f}")

    # -- Predictions ------------------------------------------------------
    mu_te, var_te = ensemble_predict(models, Xte)
    mu_va, var_va = ensemble_predict(models, Xva)
    sigma_te = np.sqrt(var_te)
    sigma_va = np.sqrt(var_va)

    # Sanity: point-prediction quality should match Q1's signals-only MLP.
    def spearman(a, b):
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])
    rho = spearman(mu_te, yte)
    rmse = float(np.sqrt(np.mean((mu_te - yte) ** 2)))
    print(f"\npoint metrics  Spearman={rho:+.3f}  RMSE={rmse:.3f}  (Q1 ref +0.762/4.70)")

    # -- (A) Calibration ---------------------------------------------------
    levels = np.linspace(0.1, 0.9, 9)
    obs_raw = coverage_curve(mu_te, sigma_te, yte, levels)
    s = fit_variance_scale(mu_va, sigma_va, yva, levels)  # tuned on validation only
    obs_cal = coverage_curve(mu_te, s * sigma_te, yte, levels)
    ce_raw = calibration_error(levels, obs_raw)
    ce_cal = calibration_error(levels, obs_cal)
    # Direction of miscalibration: at the 90% nominal interval, is observed
    # coverage below (over-confident) or above (under-confident) nominal?
    cov90 = obs_raw[-1]
    direction = "over-confident (intervals too narrow)" if cov90 < 0.9 else \
                "under-confident (intervals too wide)"
    print(f"\nCalibration error (mean |nominal-observed| coverage):")
    print(f"  raw                  = {ce_raw:.3f}   90% interval covers {cov90:.2f} -> {direction}")
    print(f"  after variance x{s:.2f}  = {ce_cal:.3f}   (scale tuned on val coverage)")

    # -- (B) Sparsification ------------------------------------------------
    abs_err = np.abs(mu_te - yte)
    fractions = np.linspace(0.05, 1.0, 20)
    order_unc = np.argsort(sigma_te)[::-1]               # most-uncertain first
    order_oracle = np.argsort(abs_err)[::-1]             # largest-error first
    curve_unc = retention_curve(abs_err, order_unc, fractions)
    curve_oracle = retention_curve(abs_err, order_oracle, fractions)
    curve_random = np.full_like(fractions, abs_err.mean())
    # AUSE: area between our sparsification curve and the oracle (lower=better),
    # normalized by the full-set MAE so it is unit-free.
    ause = float(np.trapezoid(curve_unc - curve_oracle, fractions) / abs_err.mean())
    print(f"\nSparsification:")
    print(f"  MAE keeping 25% least-uncertain = {curve_unc[fractions <= 0.25][-1]:.3f}")
    print(f"  MAE on full set                 = {abs_err.mean():.3f}")
    print(f"  AUSE (vs oracle, normalized)    = {ause:.3f}  (0 = perfect)")

    # Does the predicted sigma actually correlate with realized error?
    rho_unc_err = spearman(sigma_te, abs_err)
    print(f"  Spearman(sigma, |error|)        = {rho_unc_err:+.3f}")

    # -- Plots -------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    ax[0].plot(levels, obs_raw, "o-", label=f"raw (CE={ce_raw:.3f})")
    ax[0].plot(levels, obs_cal, "s-", label=f"recalibrated x{s:.2f} (CE={ce_cal:.3f})")
    ax[0].set_xlabel("nominal coverage"); ax[0].set_ylabel("observed coverage")
    ax[0].set_title("(A) Calibration reliability"); ax[0].legend(); ax[0].grid(alpha=.3)

    ax[1].plot(fractions, curve_random, ":", color="gray", label="random")
    ax[1].plot(fractions, curve_unc, "o-", label=f"by uncertainty (AUSE={ause:.3f})")
    ax[1].plot(fractions, curve_oracle, "s-", label="oracle (by true error)")
    ax[1].set_xlabel("fraction of frames retained (least-uncertain)")
    ax[1].set_ylabel("MAE on retained frames")
    ax[1].set_title("(B) Sparsification / error-retention")
    ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "q3_calibration_sparsification.png", dpi=130)
    print(f"\nSaved plot -> {OUT_DIR/'q3_calibration_sparsification.png'}")

    # -- Save artifacts ----------------------------------------------------
    np.savez(
        OUT_DIR / "q3_test_predictions.npz",
        y_true=yte, mu=mu_te, sigma=sigma_te, run_id=d["rte"],
        variance_scale=s,
    )
    for k, m in enumerate(models):
        torch.save(m.state_dict(), OUT_DIR / f"q3_member_{k}.pt")
    summary = dict(
        point_spearman=rho, point_rmse=rmse,
        calib_error_raw=ce_raw, calib_error_recalibrated=ce_cal,
        variance_scale=float(s),
        ause=ause, spearman_sigma_abserr=rho_unc_err,
        nominal_levels=levels.tolist(),
        observed_raw=obs_raw.tolist(), observed_recalibrated=obs_cal.tolist(),
    )
    (OUT_DIR / "q3_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved summary -> {OUT_DIR/'q3_summary.json'}")


if __name__ == "__main__":
    main()
