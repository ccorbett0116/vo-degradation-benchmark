"""
Shared data-prep for Research Questions 3 and 4.

Reuses Q1's exact signal/label/feature pipeline (so the proxy label, the run-level
split, and the 29-D feature vector are identical across all four questions), but
points the file paths at this machine's local copy of the dataset.

Q2 showed the camera image adds nothing beyond the tabular VO-health signals at
the frame level, so Q3/Q4 build on the signals-only model from Q1.
"""

from __future__ import annotations

from pathlib import Path
import sys

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import research_question_1 as rq1  # noqa: E402

# --- Point Jose's pipeline at the local data on this machine -----------------
# His committed manifest/split (with estimated_lce_m + physical_lce_m) live in
# the repo; the 9.9 GB HDF5 is not in git, so use the local copy.
LOCAL_HDF5 = Path("/home/coding/VO_Project/VO_Research/comprehensive_dataset/training_data.hdf5")
rq1.TRAIN_HDF5 = LOCAL_HDF5
rq1.DATA_DIR = REPO / "Data" / "comprehensive_dataset"
rq1.MANIFEST = rq1.DATA_DIR / "manifest.json"
rq1.SPLIT_FILE = rq1.DATA_DIR / "split.json"


def gt_twist(fused: np.ndarray, ts: np.ndarray, win_s: float = 0.15) -> np.ndarray:
    """Ground-truth body twist [linear (m/s), angular (rad/s)] from the
    fused_pose reference (LiDAR/SLAM, loop-closure ~0.05-0.21 m).

    Per-frame single-step differencing of the high-rate (~78 Hz) pose is too
    noisy (linear p95 ~0.64 m/s on a 0.2 m/s robot), so we take a centered
    displacement over a ~win_s window. The result is physically sane
    (linear p95 ~0.27, angular p95 ~0.40 -- close to the robot's command limits).

    Used ONLY as the prediction target. fused_pose is not available to the robot
    at runtime, so it never enters the feature vector; the wheel-derived
    disagreement remains a (legitimately observable) input feature."""
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


def build_dataset(verbose: bool = True, label_source: str = "gt") -> dict:
    """Replicate Q1's data prep and return standardized train/val/test arrays.

    Returns a dict with the standardized feature matrices, labels, per-frame run
    ids, per-frame global HDF5 row indices (for fetching raw twist / fused_pose
    in Q4), standardization stats, the split, the manifest, and the raw per-run
    signal dict (needed for Q4 trajectory replay)."""
    train_ids, test_ids = rq1.load_split()
    manifest = rq1.load_manifest()

    # Same fixed validation carve-out Jose uses (seed 0).
    rng = np.random.RandomState(0)
    shuf = train_ids.copy()
    rng.shuffle(shuf)
    val_ids = sorted(shuf[: rq1.N_VAL_RUNS])
    train_ids = sorted(shuf[rq1.N_VAL_RUNS :])
    all_ids = train_ids + val_ids + test_ids
    if verbose:
        print(f"train={len(train_ids)} val={len(val_ids)} test={len(test_ids)} runs")

    f = h5py.File(LOCAL_HDF5, "r")
    rid_all = f["run_id"][:]

    # Per-run signals. The wheel-based disagreement stays in `sig["disagreement"]`
    # (it feeds the feature vector). When label_source="gt" we additionally build
    # a ground-truth disagreement ||vo - gt_twist|| used only for the LABEL.
    all_signals, pooled = {}, []
    for r in all_ids:
        sig = rq1.compute_run_signals(f, r)
        if label_source == "gt":
            idx = np.where(rid_all == r)[0]
            fused = f["fused_pose"][idx[0]:idx[-1] + 1]
            gt = gt_twist(fused, sig["ts"])
            sig["gt_twist"] = gt
            sig["label_disagreement"] = np.linalg.norm(sig["vo_mps"] - gt, axis=1).astype(np.float32)
        elif label_source == "wheel":
            sig["label_disagreement"] = sig["disagreement"]
        else:
            raise ValueError(label_source)
        all_signals[r] = sig
        d = sig["label_disagreement"]
        pooled.append(d[~np.isnan(d)])
    dropout_value = float(np.percentile(np.concatenate(pooled), 99))

    # Future-badness labels: mean (GT or wheel) disagreement over the next 1 s.
    for sig in all_signals.values():
        sig["label"] = rq1.build_future_label(
            sig["label_disagreement"], sig["ts"], rq1.FUTURE_HORIZON_S, dropout_value
        )

    def collect(ids):
        Xs, ys, rids, gidxs = [], [], [], []
        for r in ids:
            sig = all_signals[r]
            X, valid = rq1.build_features(sig, rq1.PAST_WINDOW_S)
            y = sig["label"]
            keep = valid & ~np.isnan(y)
            gidx = np.where(rid_all == r)[0]
            ki = np.where(keep)[0]
            Xs.append(X[ki])
            ys.append(y[ki])
            rids.append(np.full(len(ki), r, dtype=np.int32))
            gidxs.append(gidx[ki])
        return (
            np.concatenate(Xs),
            np.concatenate(ys),
            np.concatenate(rids),
            np.concatenate(gidxs),
        )

    Xtr, ytr, rtr, gtr = collect(train_ids)
    Xva, yva, rva, gva = collect(val_ids)
    Xte, yte, rte, gte = collect(test_ids)

    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-6
    if verbose:
        print(f"frames -> train {Xtr.shape} val {Xva.shape} test {Xte.shape}")

    return dict(
        Xtr=(Xtr - mu) / sd, ytr=ytr, rtr=rtr, gtr=gtr,
        Xva=(Xva - mu) / sd, yva=yva, rva=rva, gva=gva,
        Xte=(Xte - mu) / sd, yte=yte, rte=rte, gte=gte,
        Xtr_raw=Xtr, Xte_raw=Xte,
        mu=mu, sd=sd,
        train_ids=train_ids, val_ids=val_ids, test_ids=test_ids,
        manifest=manifest, all_signals=all_signals,
        hdf5_path=str(LOCAL_HDF5), dropout_value=dropout_value,
        label_source=label_source,
    )


def condition_by_run(test_ids) -> dict:
    """Map run_id -> condition string (constant within a run)."""
    f = h5py.File(LOCAL_HDF5, "r")
    cond = f["condition"]
    rid = f["run_id"][:]
    out = {}
    for r in test_ids:
        j = np.where(rid == r)[0][0]
        c = cond[j]
        out[r] = c.decode() if isinstance(c, bytes) else c
    return out
