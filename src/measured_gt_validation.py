"""
Measured-ground-truth validation (Option A central result).

Trains the Q1 risk MLP under run-level cross-validation so that EVERY one of the
60 camera-on runs receives an out-of-fold (held-out) predicted risk. We then ask
the question that matters: does the predicted risk track the *measured* figure-8
loop-closure error (`physical_lce_m`, tape-measured) -- or only the onboard
filter's self-estimate (`estimated_lce_m`)?

Uses the exact Q1 signal/label/feature pipeline via common_q34 / research_question_1,
so the divergence label and 29-D features are identical to the headline models.
Nothing is retrained beyond these CV folds; no robot involved.
"""
from __future__ import annotations

import json
import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader, TensorDataset

import common_q34 as c
import research_question_1 as rq1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_FOLDS = 6
EPOCHS = 20
torch.manual_seed(0)
np.random.seed(0)


def build_all_runs():
    """Replicate common_q34's per-run label/feature build for ALL 60 camera-on
    runs, returning pooled raw features, labels, and per-frame run ids."""
    train_ids, test_ids = rq1.load_split()
    all_ids = sorted(train_ids + test_ids)
    f = h5py.File(c.LOCAL_HDF5, "r")
    rid_all = f["run_id"][:]

    signals, pooled = {}, []
    for r in all_ids:
        sig = rq1.compute_run_signals(f, r)
        idx = np.where(rid_all == r)[0]
        fused = f["fused_pose"][idx[0]:idx[-1] + 1]
        gt = c.gt_twist(fused, sig["ts"])
        sig["label_disagreement"] = np.linalg.norm(sig["vo_mps"] - gt, axis=1).astype(np.float32)
        signals[r] = sig
        d = sig["label_disagreement"]
        pooled.append(d[~np.isnan(d)])
    dropout_value = float(np.percentile(np.concatenate(pooled), 99))

    Xs, ys, rids = [], [], []
    for r in all_ids:
        sig = signals[r]
        sig["label"] = rq1.build_future_label(
            sig["label_disagreement"], sig["ts"], rq1.FUTURE_HORIZON_S, dropout_value)
        X, valid = rq1.build_features(sig, rq1.PAST_WINDOW_S)
        y = sig["label"]
        keep = valid & ~np.isnan(y)
        ki = np.where(keep)[0]
        Xs.append(X[ki]); ys.append(y[ki])
        rids.append(np.full(len(ki), r, dtype=np.int32))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(rids), all_ids


def train_fold(Xtr, ytr):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    model = rq1.RiskMLP(in_dim=Xtr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(((Xtr - mu) / sd).astype(np.float32)),
                                      torch.from_numpy(ytr.astype(np.float32))),
                        batch_size=4096, shuffle=True)
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    return model, mu, sd


@torch.no_grad()
def predict(model, X, mu, sd):
    model.eval()
    t = torch.from_numpy(((X - mu) / sd).astype(np.float32)).to(DEVICE)
    return model(t).cpu().numpy()


def main():
    print(f"device={DEVICE}  building features for all 60 runs ...")
    X, y, rid, all_ids = build_all_runs()
    print(f"  pooled frames: {X.shape}  runs: {len(all_ids)}")

    # Run-level CV: partition runs into N_FOLDS groups; predict each held-out group.
    rng = np.random.RandomState(0)
    shuffled = np.array(all_ids); rng.shuffle(shuffled)
    folds = np.array_split(shuffled, N_FOLDS)

    oof = np.full(len(y), np.nan, dtype=np.float32)
    for i, held in enumerate(folds, 1):
        tr = ~np.isin(rid, held)
        te = np.isin(rid, held)
        model, mu, sd = train_fold(X[tr], y[tr])
        oof[te] = predict(model, X[te], mu, sd)
        print(f"  fold {i}/{N_FOLDS}: held-out runs {sorted(held.tolist())}")

    man = rq1.load_manifest()
    f = h5py.File(c.LOCAL_HDF5, "r")
    cond_all = f["condition"][:]; rid_all = f["run_id"][:]
    cond_of = {r: (lambda v: v.decode() if isinstance(v, bytes) else v)(
        cond_all[np.where(rid_all == r)[0][0]]) for r in all_ids}

    rows = []
    for r in all_ids:
        sel = rid == r
        rows.append((r, cond_of[r], float(oof[sel].mean()),
                     float(y[sel].mean()),
                     man[r]["physical_lce_m"], man[r]["estimated_lce_m"]))

    pred = np.array([x[2] for x in rows])
    phys = np.array([x[4] for x in rows])
    est = np.array([x[5] for x in rows])

    def pear(a, b): return float(np.corrcoef(a, b)[0, 1])
    print("\n================ RUN-LEVEL (all 60, out-of-fold) ================")
    print(f"  Pearson(pred_risk, MEASURED physical_lce) = {pear(pred, phys):+.3f}")
    print(f"  Pearson(pred_risk, filter estimated_lce)  = {pear(pred, est):+.3f}")
    print(f"  Pearson(physical_lce, estimated_lce)      = {pear(phys, est):+.3f}")

    # Per-condition: where divergence is high vs where real error is high.
    print("\n  per-condition means (sorted by measured physical_lce):")
    conds = {}
    for r, cnd, mp, mt, ph, es in rows:
        conds.setdefault(cnd, []).append((mp, ph, es))
    table = []
    for cnd, vals in conds.items():
        v = np.array(vals)
        table.append((cnd, v[:, 0].mean(), v[:, 1].mean(), v[:, 2].mean(), len(vals)))
    print(f"    {'condition':<18} {'pred_risk':>10} {'phys_lce':>9} {'est_lce':>8} {'n':>3}")
    for cnd, pr, ph, es, n in sorted(table, key=lambda x: x[2]):
        print(f"    {cnd:<18} {pr:>10.3f} {ph:>9.3f} {es:>8.3f} {n:>3}")

    out = dict(
        n_runs=len(all_ids), n_folds=N_FOLDS,
        pearson_pred_physical=pear(pred, phys),
        pearson_pred_estimated=pear(pred, est),
        pearson_physical_estimated=pear(phys, est),
        per_run=[dict(run_id=r, condition=cnd, pred_risk=mp, mean_true_divergence=mt,
                      physical_lce_m=ph, estimated_lce_m=es)
                 for r, cnd, mp, mt, ph, es in rows],
    )
    outpath = c.REPO / "analysis" / "measured_gt_validation.json"
    outpath.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {outpath}")


if __name__ == "__main__":
    main()
