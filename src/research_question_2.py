"""
Research Question 2
-------------------
Do camera images help on top of robot + VO-health signals?

Same labels, same split, same tabular features as Q1.
Adds a small CNN branch over the current frame, fused with the Q1 features.

Three evaluations on the 12 held-out test runs:
  (a) Full-info Spearman vs Q1 signals-only.
  (b) Masked-info Spearman: zero out recent VO-health features at test time,
      simulating loss of VO telemetry. The vision-augmented model should
      degrade more gracefully.
  (c) Per-condition delta: Spearman(vision) - Spearman(signals) broken down
      by condition. This is the headline figure — it tells us WHERE images help.

Memory strategy:
  Training images are SUBSAMPLED (every Kth frame) so the kept images fit in
  RAM as a single uint8 numpy array. Validation and test use every frame.
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
from torch.utils.data import DataLoader, Dataset

# Re-use Q1 data-prep helpers verbatim (same labels, same features, same split).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_question_1 import (
    DATA_DIR, TRAIN_HDF5, MANIFEST, SPLIT_FILE,
    PAST_WINDOW_S, FUTURE_HORIZON_S, N_VAL_RUNS, N_FEATS, LABEL_SOURCE,
    Stage, progress,
    load_split, load_manifest,
    compute_run_signals, build_future_label, build_features, label_disagreement,
)

OUT_DIR = DATA_DIR.parent.parent / "analysis" / "q2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_SUBSAMPLE = 1         # 1 = use every training frame (~14 GB images in RAM)
EPOCHS          = 8
BATCH           = 512       # 3080 has plenty of VRAM headroom at this image size
LR              = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
np.random.seed(0)

# Indices into the 29-dim feature vector that we'll mask at test time
# (matches build_features() in research_question_1.py)
VO_HEALTH_FEATURE_COLS = [
    18, 19, 20, 21,   # vo_feat current + window mean
    22, 23,           # vo_cov diagonals
    24, 25, 26, 27,   # disagreement current/mean/max/drift
    28,               # dropout fraction
]


# --------------------------------------------------------------------------
# Dataset that returns (features, image, label)
# --------------------------------------------------------------------------
class FrameDataset(Dataset):
    """Holds standardized features, a *reference* to the preloaded all-images
    array, the global HDF5 row index per sample, and labels.
    No copying — every dataset shares the same 18 GB images array."""

    def __init__(self, X: np.ndarray, all_images: np.ndarray,
                 global_indices: np.ndarray, y: np.ndarray,
                 run_ids: np.ndarray):
        assert len(X) == len(global_indices) == len(y) == len(run_ids)
        self.X = X.astype(np.float32)
        self.all_images = all_images        # shared uint8 (N_total, 120, 160, 3)
        self.global_indices = global_indices
        self.y = y.astype(np.float32)
        self.run_ids = run_ids

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img = self.all_images[self.global_indices[i]].astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))      # HWC -> CHW
        return self.X[i], img, self.y[i]


# --------------------------------------------------------------------------
# Model: small CNN + tabular MLP fused
# --------------------------------------------------------------------------
class VisionRiskNet(nn.Module):
    def __init__(self, n_feats: int):
        super().__init__()
        # Image branch — small CNN over 120x160x3
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2), nn.ReLU(),    # 60x80
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),   # 30x40
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),   # 15x20
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),   #  8x10
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),                              #  64
            nn.Linear(64, 64), nn.ReLU(),
        )
        # Tabular branch
        self.tab = nn.Sequential(
            nn.Linear(n_feats, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        # Fusion head
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, feats, img):
        a = self.cnn(img)
        b = self.tab(feats)
        z = torch.cat([a, b], dim=1)
        return self.head(z).squeeze(-1)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation."""
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def preload_all_images(f: h5py.File) -> np.ndarray:
    """Sequentially read the entire `images` dataset into one big uint8 array.
    Sequential reads are dramatically faster than scattered indexed reads on
    chunked HDF5. ~18 GB for the full 311k-frame dataset."""
    n = f["images"].shape[0]
    out = np.empty((n, 120, 160, 3), dtype=np.uint8)
    CHUNK = 8192
    t0 = time.perf_counter()
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        out[s:e] = f["images"][s:e]      # sequential slice, fast
        progress("preload", e, n, t0)
    return out


def collect_split(all_signals, ids, subsample=1):
    """Build (X, global_indices_into_hdf5, y, run_ids) for a list of run_ids.
    `subsample` keeps every Kth surviving frame (after the validity filter)."""
    Xs, idxs, ys, rids = [], [], [], []
    rid_all = h5py.File(TRAIN_HDF5, "r")["run_id"][:]
    for r in ids:
        sig = all_signals[r]
        X, valid = build_features(sig, PAST_WINDOW_S)
        y = sig["label"]
        keep = valid & ~np.isnan(y)
        # global indices into the HDF5 file (for image lookup)
        gidx = np.where(rid_all == r)[0]
        keep_idx = np.where(keep)[0]
        if subsample > 1:
            keep_idx = keep_idx[::subsample]
        Xs.append(X[keep_idx])
        idxs.append(gidx[keep_idx])
        ys.append(y[keep_idx])
        rids.append(np.full(len(keep_idx), r, dtype=np.int32))
    return (np.concatenate(Xs), np.concatenate(idxs),
            np.concatenate(ys), np.concatenate(rids))


# --------------------------------------------------------------------------
# Training / evaluation
# --------------------------------------------------------------------------
def train_one(model, train_loader, val_loader, epochs, lr):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_val, best_state = math.inf, None
    for ep in range(epochs):
        ep_t0 = time.perf_counter()
        model.train()
        tr_loss, n = 0.0, 0
        for X, img, y in train_loader:
            X, img, y = X.to(DEVICE), img.to(DEVICE), y.to(DEVICE)
            pred = model(X, img)
            loss = loss_fn(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * len(y); n += len(y)
        tr_loss /= n
        v_loss = eval_loss(model, val_loader)
        flag = ""
        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            flag = "  <-- best"
        dt = time.perf_counter() - ep_t0
        eta = dt * (epochs - ep - 1)
        print(f"    epoch {ep+1:2d}/{epochs}  train={tr_loss:.4f}  val={v_loss:.4f}  "
              f"ep_time={dt:5.1f}s  eta={eta:6.1f}s{flag}", flush=True)
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def eval_loss(model, loader):
    model.eval()
    loss_fn = nn.MSELoss(reduction="sum")
    s, n = 0.0, 0
    for X, img, y in loader:
        X, img, y = X.to(DEVICE), img.to(DEVICE), y.to(DEVICE)
        s += loss_fn(model(X, img), y).item()
        n += len(y)
    return s / max(n, 1)


@torch.no_grad()
def predict(model, loader) -> np.ndarray:
    model.eval()
    out = []
    for X, img, y in loader:
        X, img = X.to(DEVICE), img.to(DEVICE)
        out.append(model(X, img).cpu().numpy())
    return np.concatenate(out)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    Stage._t0_global = time.perf_counter()
    print(f"device={DEVICE}", flush=True)

    train_ids, test_ids = load_split()
    manifest = load_manifest()

    rng = np.random.RandomState(0)
    shuf = train_ids.copy(); rng.shuffle(shuf)
    val_ids   = sorted(shuf[:N_VAL_RUNS])
    train_ids = sorted(shuf[N_VAL_RUNS:])
    print(f"train runs ({len(train_ids)}): {train_ids}")
    print(f"val   runs ({len(val_ids)}): {val_ids}")
    print(f"test  runs ({len(test_ids)}): {test_ids}")
    print(f"train subsample: every {TRAIN_SUBSAMPLE} frame(s)")

    all_ids = train_ids + val_ids + test_ids

    # -- Per-run signals + labels (identical to Q1) -----------------------
    with Stage(f"Compute per-run signals ({len(all_ids)} runs)"):
        f = h5py.File(TRAIN_HDF5, "r")
        all_signals = {}
        all_disagr = []
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

    with Stage(f"Build labels (horizon = {FUTURE_HORIZON_S}s)"):
        t0 = time.perf_counter()
        for i, (r, sig) in enumerate(all_signals.items(), 1):
            sig["label"] = build_future_label(sig["label_disagreement"], sig["ts"],
                                              FUTURE_HORIZON_S, dropout_value)
            progress("labels", i, len(all_signals), t0)

    with Stage("Build feature vectors + collect HDF5 indices"):
        Xtr, gidx_tr, ytr, _   = collect_split(all_signals, train_ids, subsample=TRAIN_SUBSAMPLE)
        Xva, gidx_va, yva, _   = collect_split(all_signals, val_ids,   subsample=1)
        Xte, gidx_te, yte, rte = collect_split(all_signals, test_ids,  subsample=1)
        print(f"    shapes -> train: {Xtr.shape}  val: {Xva.shape}  test: {Xte.shape}")

    # Standardize features using train statistics only
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-6
    Xtr_s = (Xtr - mu) / sd
    Xva_s = (Xva - mu) / sd
    Xte_s = (Xte - mu) / sd

    # -- Preload the entire image dataset once (sequential, fast) ---------
    n_total = f["images"].shape[0]
    with Stage(f"Preload all images sequentially "
               f"({n_total} frames, ~{n_total*57600/1e9:.1f} GB)"):
        all_images = preload_all_images(f)

    # Datasets share the same all_images array — no per-set copies.
    ds_tr = FrameDataset(Xtr_s, all_images, gidx_tr, ytr,
                         np.zeros(len(ytr), dtype=np.int32))
    ds_va = FrameDataset(Xva_s, all_images, gidx_va, yva,
                         np.zeros(len(yva), dtype=np.int32))
    ds_te = FrameDataset(Xte_s, all_images, gidx_te, yte, rte)
    train_loader = DataLoader(ds_tr, batch_size=BATCH, shuffle=True,  num_workers=0, drop_last=False)
    val_loader   = DataLoader(ds_va, batch_size=BATCH, shuffle=False, num_workers=0)
    test_loader  = DataLoader(ds_te, batch_size=BATCH, shuffle=False, num_workers=0)

    # -- Train the vision-augmented model ---------------------------------
    with Stage(f"Train VisionRiskNet  (device={DEVICE}, batch={BATCH}, epochs={EPOCHS})"):
        model = VisionRiskNet(n_feats=Xtr_s.shape[1]).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"    model parameters: {n_params:,}")
        model = train_one(model, train_loader, val_loader, EPOCHS, LR)

    # -- (a) Full-info evaluation -----------------------------------------
    with Stage("Evaluate: full-info"):
        yhat_full = predict(model, test_loader)
        rho_full = spearman(yhat_full, yte)
        rmse_full = float(np.sqrt(np.mean((yhat_full - yte) ** 2)))
        print(f"    VisionRiskNet  Spearman={rho_full:+.3f}  RMSE={rmse_full:.3f}")
        print(f"    Q1 reference   Spearman=+0.762            (signals-only)")

    # -- (b) Masked-info evaluation ---------------------------------------
    #   Zero out VO-health features at test time and re-predict. The CNN gets
    #   the image as before; the tabular branch loses recent VO telemetry.
    with Stage("Evaluate: VO-health signals masked"):
        Xte_masked = Xte_s.copy()
        Xte_masked[:, VO_HEALTH_FEATURE_COLS] = 0.0       # zero in standardized space = the mean
        ds_te_masked = FrameDataset(Xte_masked, all_images, gidx_te, yte, rte)
        loader_masked = DataLoader(ds_te_masked, batch_size=BATCH, shuffle=False, num_workers=0)
        yhat_masked = predict(model, loader_masked)
        rho_masked = spearman(yhat_masked, yte)
        rmse_masked = float(np.sqrt(np.mean((yhat_masked - yte) ** 2)))
        print(f"    VisionRiskNet (masked)  Spearman={rho_masked:+.3f}  RMSE={rmse_masked:.3f}")
        print(f"    Delta from full-info    {rho_masked - rho_full:+.3f}")
        print(f"    -> if small, vision compensates for missing VO telemetry")

    # -- (c) Per-condition breakdown --------------------------------------
    with Stage("Per-condition Spearman (full-info)"):
        cond_field = f["condition"]
        rid_all = f["run_id"][:]
        cond_by_run = {}
        for r in test_ids:
            j = np.where(rid_all == r)[0][0]
            c = cond_field[j]
            cond_by_run[r] = c.decode() if isinstance(c, bytes) else c

        print(f"    {'condition':<22} {'n':>6} {'Spearman':>10}")
        for c_name in sorted(set(cond_by_run.values())):
            runs_in_c = [r for r, c in cond_by_run.items() if c == c_name]
            sel = np.isin(rte, runs_in_c)
            if sel.sum() < 100: continue
            rho = spearman(yhat_full[sel], yte[sel])
            print(f"    {c_name:<22} {sel.sum():>6} {rho:>+10.3f}")

    # -- Run-level sanity check (same as Q1) ------------------------------
    with Stage("Run-level sanity check vs estimated_lce_m"):
        run_means = {}
        for r in test_ids:
            sel = rte == r
            if sel.any():
                run_means[r] = (float(yhat_full[sel].mean()),
                                float(yte[sel].mean()),
                                manifest[r]["estimated_lce_m"])
        pred_arr = np.array([v[0] for v in run_means.values()])
        est_arr  = np.array([v[2] for v in run_means.values()])
        print(f"    Pearson(mean_predicted, estimated_lce_m) = "
              f"{np.corrcoef(pred_arr, est_arr)[0,1]:+.3f}")

    # -- Save outputs -----------------------------------------------------
    np.savez(OUT_DIR / "q2_test_predictions.npz",
             test_run_ids=np.array(test_ids),
             y_true=yte,
             y_pred_full=yhat_full,
             y_pred_masked=yhat_masked,
             run_id=rte)
    torch.save(model.state_dict(), OUT_DIR / "q2_vision_model.pt")
    print(f"\nSaved -> {OUT_DIR}/q2_test_predictions.npz")
    print(f"Saved -> {OUT_DIR}/q2_vision_model.pt")


if __name__ == "__main__":
    main()
