"""
Research Question 1
-------------------
Can we predict VO degradation ~1 second before it happens, from past sensor signals?

Pipeline:
  1. Load B+C frames from training_data.hdf5
  2. Convert vo_twist (per-update displacement) to m/s using detected VO update periods
  3. Build the per-frame "disagreement" signal: || vo_mps - wheel_command ||
  4. Build the future-badness label: mean disagreement over the next ~1 s
  5. Build a flat feature vector per frame from a ~0.5 s past window
  6. Train a small MLP (torch) on 42 train runs, validate on 6, test on 12
  7. Report metrics + per-condition breakdown + run-level sanity check vs estimated_lce_m
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# --------------------------------------------------------------------------
# Logging helpers
# --------------------------------------------------------------------------
class Stage:
    """Context manager that prints a stage banner with elapsed time on exit."""
    _t0_global = time.perf_counter()

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.t0 = time.perf_counter()
        since_start = self.t0 - Stage._t0_global
        print(f"\n[{since_start:7.1f}s] >>> {self.name}", flush=True)
        return self

    def __exit__(self, *exc):
        dt = time.perf_counter() - self.t0
        print(f"[{time.perf_counter()-Stage._t0_global:7.1f}s] <<< {self.name} "
              f"(took {dt:.2f}s)", flush=True)


def progress(prefix: str, i: int, n: int, t_start: float, every: int = 1):
    """Print 'prefix  i/n  pct  elapsed  ETA' with an ETA estimate. Inline."""
    if i % every and i != n:
        return
    elapsed = time.perf_counter() - t_start
    pct = 100.0 * i / n
    eta = elapsed * (n - i) / max(i, 1)
    sys.stdout.write(
        f"\r    {prefix}: {i:>4}/{n}  ({pct:5.1f}%)  "
        f"elapsed={elapsed:6.1f}s  ETA={eta:6.1f}s   "
    )
    sys.stdout.flush()
    if i == n:
        sys.stdout.write("\n")


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
# Paths are resolved relative to the repo root so the code is portable across
# machines. common_q34.py overrides TRAIN_HDF5 to point at the local 9.9 GB file
# (which is not committed to git).
_REPO       = Path(__file__).resolve().parent.parent
DATA_DIR    = _REPO / "Data" / "comprehensive_dataset"
TRAIN_HDF5  = DATA_DIR / "training_data.hdf5"
if not TRAIN_HDF5.exists():       # 9.9 GB file is not in git; use the local copy
    TRAIN_HDF5 = Path("/home/coding/VO_Project/VO_Research/comprehensive_dataset/training_data.hdf5")
MANIFEST    = DATA_DIR / "manifest.json"
SPLIT_FILE  = DATA_DIR / "split.json"
OUT_DIR     = _REPO / "analysis" / "q1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAST_WINDOW_S    = 0.5     # how much past the model sees
FUTURE_HORIZON_S = 1.0     # how far ahead we predict
DROPOUT_PENALTY  = None    # set in __main__ after we see the disagreement distribution
N_VAL_RUNS       = 6       # validation runs carved out of the 48 train runs
LABEL_SOURCE     = "gt"    # "gt": label = ||vo - gt_twist|| (ground truth from
                           # fused_pose); "wheel": Jose's original ||vo - wheel||.
                           # GT is the prediction target; the wheel-derived
                           # disagreement remains a (runtime-observable) feature.

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
np.random.seed(0)


# --------------------------------------------------------------------------
# 1. Load split + manifest
# --------------------------------------------------------------------------
def load_split():
    split = json.loads(SPLIT_FILE.read_text())
    return split["train_run_ids"], split["test_run_ids"]


def load_manifest():
    return {m["run_id"]: m for m in json.loads(MANIFEST.read_text())}


# --------------------------------------------------------------------------
# 2. vo_twist is already a body velocity estimate [lin m/s, ang rad/s]
# --------------------------------------------------------------------------
def vo_twist_to_mps(vo_r: np.ndarray, ts_r: np.ndarray) -> np.ndarray:
    """Return vo_twist as a body velocity, marking dropouts (NaN / all-zero) NaN.

    CORRECTION (2026-05-26): the original version treated vo_twist as a per-update
    *displacement* and divided by the inter-update period. That inflated VO ~15x
    (linear p95 0.32 -> 4.7 m/s on a 0.2 m/s robot) and was the source of the
    "heavy-tailed VO outliers". Validation against differentiated fused_pose shows
    raw vo_twist is already on the true-velocity scale (lin p95 0.32 vs GT 0.27),
    and the parent project's headroom analysis integrates it raw. We therefore use
    it directly; a constant wheel+VO blend now beats wheel-only (1.92 vs 2.28 m
    ATE), which the inflated signal could not. ts_r is kept for signature compat."""
    out = vo_r.astype(np.float32).copy()
    bad = np.isnan(vo_r).any(axis=1) | (np.abs(vo_r).sum(axis=1) == 0)
    out[bad] = np.nan
    return out


# --------------------------------------------------------------------------
# 3-4. Compute disagreement + future-badness label per run
# --------------------------------------------------------------------------
def compute_run_signals(f: h5py.File, run_id: int):
    """For one run: return (frames dict, valid mask).
    `frames` contains every input signal we need plus the disagreement series."""
    rid = f["run_id"][:]
    mask = rid == run_id
    if not mask.any():
        return None

    out = {
        "ts":         f["timestamps"][mask],
        "pan_tilt":   f["pan_tilt"][mask],
        "wheel_cmd":  f["wheel_twist"][mask],
        "vo_twist":   f["vo_twist"][mask],
        "vo_feat":    f["vo_features"][mask],
        "vo_cov":     f["vo_covariance"][mask],
    }
    vo_mps = vo_twist_to_mps(out["vo_twist"], out["ts"])
    out["vo_mps"] = vo_mps

    # Disagreement signal (NaN where VO dropped out)
    diff = vo_mps - out["wheel_cmd"]
    out["disagreement"] = np.linalg.norm(diff, axis=1).astype(np.float32)

    # Pan/tilt derivative (forward diff; first frame = 0)
    dpt = np.zeros_like(out["pan_tilt"])
    dt = np.diff(out["ts"])
    dt = np.where(dt > 1e-6, dt, 1e-6)
    dpt[1:] = (out["pan_tilt"][1:] - out["pan_tilt"][:-1]) / dt[:, None]
    out["pan_tilt_rate"] = dpt.astype(np.float32)
    return out


def gt_twist(fused: np.ndarray, ts: np.ndarray, win_s: float = 0.15) -> np.ndarray:
    """Ground-truth body twist [linear m/s, angular rad/s] from fused_pose
    (LiDAR/SLAM reference, loop-closure ~0.05-0.21 m). Single-step differencing
    of the ~78 Hz pose is too noisy, so we use a centered displacement over a
    ~win_s window (linear p95 ~0.27, angular ~0.40 -- physically sane). Used only
    as a prediction target; fused_pose is not available to the robot at runtime."""
    n = len(ts)
    lin = np.zeros(n, dtype=np.float32)
    ang = np.zeros(n, dtype=np.float32)
    j0 = j1 = 0
    for i in range(n):
        while j0 < i and ts[i] - ts[j0] > win_s / 2:
            j0 += 1
        while j1 < n - 1 and ts[j1] - ts[i] < win_s / 2:
            j1 += 1
        dt = ts[j1] - ts[j0]
        if dt <= 1e-6:
            continue
        dx = fused[j1, 0] - fused[j0, 0]
        dy = fused[j1, 1] - fused[j0, 1]
        dth = np.arctan2(np.sin(fused[j1, 2] - fused[j0, 2]),
                         np.cos(fused[j1, 2] - fused[j0, 2]))
        h = fused[i, 2]
        lin[i] = (dx * np.cos(h) + dy * np.sin(h)) / dt
        ang[i] = dth / dt
    return np.stack([lin, ang], axis=1)


def label_disagreement(f: h5py.File, run_id: int, sig: dict) -> np.ndarray:
    """The series the future-badness LABEL is built from. With LABEL_SOURCE='gt'
    it is ||vo - gt_twist|| (VO error vs ground truth); with 'wheel' it is the
    original ||vo - wheel_command||. Either way `sig['disagreement']` (wheel-based)
    is untouched so it can still serve as a runtime feature."""
    if LABEL_SOURCE == "wheel":
        return sig["disagreement"]
    rid = f["run_id"][:]
    idx = np.where(rid == run_id)[0]
    fused = f["fused_pose"][idx[0]:idx[-1] + 1]
    gt = gt_twist(fused, sig["ts"])
    sig["gt_twist"] = gt
    return np.linalg.norm(sig["vo_mps"] - gt, axis=1).astype(np.float32)


def build_future_label(disagreement: np.ndarray, ts: np.ndarray,
                       horizon_s: float, dropout_value: float) -> np.ndarray:
    """For every frame, return the mean disagreement over the next horizon_s seconds.
    Dropout frames inside the window are counted at `dropout_value` (worst-case).
    The last horizon_s of the run gets NaN (no future to look at)."""
    n = len(ts)
    # Replace NaN with dropout penalty in a copy
    series = np.where(np.isnan(disagreement), dropout_value, disagreement).astype(np.float32)
    label = np.full(n, np.nan, dtype=np.float32)
    # For each frame, find the end-index of the future window
    # Walk a two-pointer: end advances until ts[end] - ts[t] > horizon_s
    end = 0
    for t in range(n):
        if end <= t:
            end = t + 1
        while end < n and (ts[end] - ts[t]) < horizon_s:
            end += 1
        if end >= n:
            break  # not enough future left
        # window is [t+1, end]
        if end > t + 1:
            label[t] = series[t + 1:end + 1].mean()
    return label


# --------------------------------------------------------------------------
# 5. Build flat feature vector from a past window
# --------------------------------------------------------------------------
def build_features(signals: dict, past_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame feature vector summarizing the last `past_s` seconds.
    Returns (X[n, F], valid_mask[n]) — valid_mask is False for frames that lack
    enough past history."""
    ts        = signals["ts"]
    pt        = signals["pan_tilt"]
    pt_rate   = signals["pan_tilt_rate"]
    wcmd      = signals["wheel_cmd"]
    vo_mps    = signals["vo_mps"]
    vo_feat   = signals["vo_feat"]
    vo_cov    = signals["vo_cov"]
    disagr    = signals["disagreement"]

    n = len(ts)
    feats = []
    valid = np.zeros(n, dtype=bool)

    # Pre-fill nan-disagreement with a high value so the model sees dropouts as bad
    disagr_filled = np.where(np.isnan(disagr), 0.0, disagr)
    dropout_flag  = np.isnan(disagr).astype(np.float32)
    # Similarly fill VO m/s
    vo_filled = np.where(np.isnan(vo_mps), 0.0, vo_mps)

    # For each frame, find the start index of the past window
    # (we use a two-pointer; runs are sorted by time)
    start = 0
    for t in range(n):
        while ts[t] - ts[start] > past_s and start < t:
            start += 1
        if t - start < 2:                # need at least a couple of frames
            feats.append(np.zeros(N_FEATS, dtype=np.float32))
            continue
        sl = slice(start, t + 1)
        f = [
            # pan/tilt current value + range over window
            pt[t, 0], pt[t, 1],
            pt[sl, 0].std(), pt[sl, 1].std(),
            # pan/tilt rate stats (motion blur driver)
            np.abs(pt_rate[sl, 0]).mean(), np.abs(pt_rate[sl, 1]).mean(),
            np.abs(pt_rate[sl, 0]).max(),  np.abs(pt_rate[sl, 1]).max(),
            # commanded twist current + over window
            wcmd[t, 0], wcmd[t, 1],
            wcmd[sl, 0].mean(), wcmd[sl, 1].mean(),
            np.abs(np.diff(wcmd[sl, 0])).sum(),    # command transitions in window
            np.abs(np.diff(wcmd[sl, 1])).sum(),
            # VO velocity current + stats
            vo_filled[t, 0], vo_filled[t, 1],
            vo_filled[sl, 0].std(), vo_filled[sl, 1].std(),
            # VO feature count (low = struggling)
            float(vo_feat[t, 0]) if not np.isnan(vo_feat[t, 0]) else 0.0,
            float(vo_feat[t, 1]) if not np.isnan(vo_feat[t, 1]) else 0.0,
            np.nan_to_num(vo_feat[sl, 0], nan=0.0).mean(),
            np.nan_to_num(vo_feat[sl, 1], nan=0.0).mean(),
            # VO self-covariance diagonal terms (model can learn it's miscalibrated)
            float(vo_cov[t, 0]) if not np.isnan(vo_cov[t, 0]) else 0.0,
            float(vo_cov[t, 3]) if not np.isnan(vo_cov[t, 3]) else 0.0,
            # disagreement: current, mean, max, last derivative — STRICTLY PAST
            disagr_filled[t],
            disagr_filled[sl].mean(),
            disagr_filled[sl].max(),
            disagr_filled[t] - disagr_filled[start],
            # dropout fraction in window
            dropout_flag[sl].mean(),
        ]
        feats.append(np.asarray(f, dtype=np.float32))
        valid[t] = True

    return np.stack(feats), valid


N_FEATS = 29  # keep in sync with build_features above


# --------------------------------------------------------------------------
# 6. Model
# --------------------------------------------------------------------------
class RiskMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# --------------------------------------------------------------------------
# 7. Train + evaluate
# --------------------------------------------------------------------------
def train(model, train_loader, val_loader, epochs=20, lr=1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_val = math.inf
    best_state = None
    t_start = time.perf_counter()
    for ep in range(epochs):
        ep_t0 = time.perf_counter()
        model.train()
        tr_loss = 0.0; n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * len(xb); n += len(xb)
        tr_loss /= n
        val_loss = eval_loss(model, val_loader)
        flag = ""
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            flag = "  <-- best"
        ep_dt = time.perf_counter() - ep_t0
        eta = ep_dt * (epochs - ep - 1)
        print(f"    epoch {ep+1:2d}/{epochs}  "
              f"train={tr_loss:.4f}  val={val_loss:.4f}  "
              f"ep_time={ep_dt:5.1f}s  eta={eta:6.1f}s{flag}", flush=True)
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def eval_loss(model, loader):
    model.eval()
    loss_fn = nn.MSELoss(reduction="sum")
    s = 0.0; n = 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        s += loss_fn(model(xb), yb).item()
        n += len(xb)
    return s / max(n, 1)


@torch.no_grad()
def predict(model, X: np.ndarray) -> np.ndarray:
    model.eval()
    t = torch.from_numpy(X).to(DEVICE)
    return model(t).cpu().numpy()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    print(f"[{0.0:7.1f}s] PID={Path('/proc/self').resolve().name}  device={DEVICE}", flush=True)
    train_ids, test_ids = load_split()
    manifest = load_manifest()

    # Carve a fixed validation set out of the train runs, stratified across phases
    rng = np.random.RandomState(0)
    train_ids_shuffled = train_ids.copy()
    rng.shuffle(train_ids_shuffled)
    val_ids   = sorted(train_ids_shuffled[:N_VAL_RUNS])
    train_ids = sorted(train_ids_shuffled[N_VAL_RUNS:])
    print(f"train runs ({len(train_ids)}): {train_ids}")
    print(f"val   runs ({len(val_ids)}): {val_ids}")
    print(f"test  runs ({len(test_ids)}): {test_ids}")

    all_ids = train_ids + val_ids + test_ids

    # -- Step 1+2+3: build per-run signals --------------------------------
    with Stage(f"Compute per-run signals  ({len(all_ids)} runs)"):
        f = h5py.File(TRAIN_HDF5, "r")
        all_signals = {}
        all_disagr  = []
        t0 = time.perf_counter()
        for i, r in enumerate(all_ids, 1):
            sig = compute_run_signals(f, r)
            sig["label_disagreement"] = label_disagreement(f, r, sig)
            all_signals[r] = sig
            d = sig["label_disagreement"]
            all_disagr.append(d[~np.isnan(d)])
            progress("signals", i, len(all_ids), t0)
        all_disagr = np.concatenate(all_disagr)
        dropout_value = float(np.percentile(all_disagr, 99))
        print(f"    LABEL_SOURCE={LABEL_SOURCE}  pooled non-NaN label-disagreement: "
              f"median={np.median(all_disagr):.3f}  "
              f"p95={np.percentile(all_disagr,95):.3f}  p99={dropout_value:.3f}")

    # -- Step 4: build labels --------------------------------------------
    with Stage(f"Build future-badness labels  (horizon = {FUTURE_HORIZON_S}s)"):
        t0 = time.perf_counter()
        for i, (r, sig) in enumerate(all_signals.items(), 1):
            sig["label"] = build_future_label(sig["label_disagreement"], sig["ts"],
                                              FUTURE_HORIZON_S, dropout_value)
            progress("labels", i, len(all_signals), t0)

    # -- Step 5: build feature vectors ----------------------------------
    with Stage(f"Build feature vectors  (past window = {PAST_WINDOW_S}s)"):
        def collect(ids, label):
            Xs, ys, run_track = [], [], []
            t0 = time.perf_counter()
            for i, r in enumerate(ids, 1):
                sig = all_signals[r]
                X, valid = build_features(sig, PAST_WINDOW_S)
                y = sig["label"]
                keep = valid & ~np.isnan(y)
                Xs.append(X[keep]); ys.append(y[keep])
                run_track.append(np.full(keep.sum(), r, dtype=np.int32))
                progress(f"features-{label}", i, len(ids), t0)
            return np.concatenate(Xs), np.concatenate(ys), np.concatenate(run_track)

        Xtr, ytr, _   = collect(train_ids, "train")
        Xva, yva, _   = collect(val_ids,   "val  ")
        Xte, yte, rte = collect(test_ids,  "test ")
        print(f"    shapes -> train: {Xtr.shape}  val: {Xva.shape}  test: {Xte.shape}")

    with Stage("Standardize features (using train statistics)"):
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-6
        Xtr_s = (Xtr - mu) / sd
        Xva_s = (Xva - mu) / sd
        Xte_s = (Xte - mu) / sd

    # -- Step 6: train ---------------------------------------------------
    with Stage(f"Train MLP  (device={DEVICE})"):
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(Xtr_s), torch.from_numpy(ytr)),
            batch_size=4096, shuffle=True, drop_last=False)
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(Xva_s), torch.from_numpy(yva)),
            batch_size=8192, shuffle=False)
        model = RiskMLP(in_dim=Xtr_s.shape[1]).to(DEVICE)
        model = train(model, train_loader, val_loader, epochs=20, lr=1e-3)

    # -- Step 7: evaluate ------------------------------------------------
    with Stage("Evaluate on test set"):
        yhat = predict(model, Xte_s)

        # Baseline 1: current-frame disagreement (use the UNSTANDARDIZED Xte)
        current_disagr_col = 24
        baseline_now = Xte[:, current_disagr_col]

        # Baseline 2: VO's own covariance magnitude as predictor.
        # Per-frame trace of the translational+rotational diagonals from vo_cov.
        # vo_cov columns 0,3 are already in the feature vector (cols 22,23).
        # Use their UNSTANDARDIZED sum as the predictor.
        baseline_cov = Xte[:, 22] + Xte[:, 23]

    def metrics(name, pred, truth):
        # Spearman correlation (rank-based, robust to heavy tails)
        from numpy import argsort
        def rank(x):
            r = np.empty_like(x); r[argsort(x)] = np.arange(len(x)); return r
            return r
        ra, rb = rank(pred), rank(truth)
        rho = np.corrcoef(ra, rb)[0, 1]
        rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
        mae  = float(np.mean(np.abs(pred - truth)))
        print(f"  {name:24s}  Spearman={rho:+.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}")
        return {"name": name, "spearman": float(rho), "rmse": rmse, "mae": mae}

    overall = []
    overall.append(metrics("MLP",                  yhat,         yte))
    overall.append(metrics("baseline=current_dis", baseline_now, yte))
    overall.append(metrics("baseline=vo_cov_sum",  baseline_cov, yte))

    # Per-condition breakdown
    print("\nPer-condition (test set):")
    cond_field = h5py.File(TRAIN_HDF5, "r")["condition"]
    rid_all    = h5py.File(TRAIN_HDF5, "r")["run_id"][:]
    cond_by_run = {}
    for r in test_ids:
        # first frame's condition is fine — it's constant per run
        idx = np.where(rid_all == r)[0][0]
        c = cond_field[idx]
        cond_by_run[r] = c.decode() if isinstance(c, bytes) else c
    for c_name in sorted(set(cond_by_run.values())):
        runs_in_c = [r for r, c in cond_by_run.items() if c == c_name]
        sel = np.isin(rte, runs_in_c)
        if sel.sum() < 100: continue
        rho = np.corrcoef(yhat[sel], yte[sel])[0, 1]
        print(f"  {c_name:<22}  n={sel.sum():>6}  Pearson r={rho:+.3f}  "
              f"mean_pred={yhat[sel].mean():.2f}  mean_true={yte[sel].mean():.2f}")

    # -- Step 8: run-level sanity check vs estimated_lce_m ---------------
    print("\nRun-level sanity check (mean predicted disagreement vs estimated_lce_m):")
    run_means = {}
    for r in test_ids:
        sel = rte == r
        if sel.any():
            run_means[r] = (float(yhat[sel].mean()), float(yte[sel].mean()),
                            manifest[r]["estimated_lce_m"],
                            manifest[r]["physical_lce_m"],
                            cond_by_run[r])
    print(f"  {'run':>4}  {'mean_pred':>10} {'mean_true':>10} {'est_lce':>8} {'phys_lce':>9}  condition")
    for r, (mp, mt, el, pl, c) in sorted(run_means.items()):
        print(f"  {r:>4}  {mp:>10.3f} {mt:>10.3f} {el:>8.2f} {pl:>9.2f}  {c}")
    pred_arr = np.array([v[0] for v in run_means.values()])
    est_arr  = np.array([v[2] for v in run_means.values()])
    print(f"\n  Pearson(mean_predicted, estimated_lce_m) = "
          f"{np.corrcoef(pred_arr, est_arr)[0,1]:+.3f}")

    # Save predictions for later analysis
    np.savez(OUT_DIR / "q1_test_predictions.npz",
             test_run_ids=np.array(test_ids),
             y_true=yte, y_pred=yhat, baseline=baseline_now, run_id=rte)
    torch.save(model.state_dict(), OUT_DIR / "q1_mlp.pt")
    print(f"\nSaved predictions -> {OUT_DIR / 'q1_test_predictions.npz'}")
    print(f"Saved model       -> {OUT_DIR / 'q1_mlp.pt'}")


if __name__ == "__main__":
    main()
