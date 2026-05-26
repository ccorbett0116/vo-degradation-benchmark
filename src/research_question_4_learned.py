"""
Research Question 4 (extended) — a LEARNED per-step fusion weight.

The Q4 policy used hard thresholds + a single global blend weight, and it only
*matched* wheel-only on these runs. The parent project's headroom analysis says
the localization win needs a *per-step* trust weight, not a global one. So here
we learn a continuous per-frame fusion weight a(x) in [0,1]:

    fused_twist = a * vo_clipped + (1 - a) * wheel

trained to make the fused twist match the GROUND-TRUTH twist (differentiated
fused_pose). The loss is fully differentiable in a, so no oracle target is
needed:

    L = mean || a*vo + (1-a)*wheel  -  gt_twist ||^2

Two variants test whether Q3's uncertainty helps the *controller* (not just a
discrete gate):
    - features only            (29 runtime features -> a)
    - features + uncertainty   (29 features + risk mu + uncertainty sigma -> a)

Evaluation: trajectory ATE vs fused_pose on the 12 held-out test runs, against
wheel-only, best global blend, and the Q4 gated policies. Offline replay only.
"""

from __future__ import annotations

import json
import math

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import common_q34 as c
import research_question_1 as rq1
from research_question_3 import GaussianMLP, ensemble_predict, N_MEMBERS
from research_question_4 import (
    integrate, ate, VO_LIN_CLIP, VO_ANG_CLIP, mean_ate, tune_policy,
)

OUT_DIR = c.REPO / "analysis" / "q4"
Q3_DIR = c.REPO / "analysis" / "q3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
np.random.seed(0)


def load_ensemble(in_dim):
    models = []
    for k in range(N_MEMBERS):
        m = GaussianMLP(in_dim).to(DEVICE)
        m.load_state_dict(torch.load(Q3_DIR / f"q3_member_{k}.pt", map_location=DEVICE))
        m.eval()
        models.append(m)
    s = json.loads((Q3_DIR / "q3_summary.json").read_text())["variance_scale"]
    return models, float(s)


class WeightNet(nn.Module):
    """Maps per-frame features to PER-DIMENSION fusion weights [a_lin, a_ang] in
    [0,1]. Separate linear/angular weights matter here because VO linear velocity
    is reliable (~true scale) while VO angular is noisy/under-scaled -- a single
    shared weight cannot down-weight one without the other."""
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))      # (N, 2)


def clip_vo(vo):
    out = vo.copy()
    out[:, 0] = np.clip(out[:, 0], -VO_LIN_CLIP, VO_LIN_CLIP)
    out[:, 1] = np.clip(out[:, 1], -VO_ANG_CLIP, VO_ANG_CLIP)
    return out


def collect_fusion(d, ids, models, scale, use_unc):
    """Per-frame (feat, vo_clip, wheel, gt) over valid frames with usable VO."""
    Xs, vos, whs, gts = [], [], [], []
    for r in ids:
        sig = d["all_signals"][r]
        X, valid = rq1.build_features(sig, rq1.PAST_WINDOW_S)
        y = sig["label"]
        vo, wheel, gt = sig["vo_mps"], sig["wheel_cmd"], sig["gt_twist"]
        vo_ok = ~np.isnan(vo).any(axis=1)
        keep = valid & ~np.isnan(y) & vo_ok
        ki = np.where(keep)[0]
        Xs.append(X[ki]); vos.append(clip_vo(vo[ki]))
        whs.append(wheel[ki]); gts.append(gt[ki])
    X = np.concatenate(Xs); vo = np.concatenate(vos)
    wheel = np.concatenate(whs); gt = np.concatenate(gts)
    Xs_std = (X - d["mu"]) / d["sd"]
    feat = Xs_std.astype(np.float32)
    if use_unc:
        mu, var = ensemble_predict(models, feat)
        sig_ = np.sqrt(var) * scale
        feat = np.concatenate([feat, _z(mu)[:, None], _z(sig_)[:, None]], axis=1).astype(np.float32)
    return feat, vo.astype(np.float32), wheel.astype(np.float32), gt.astype(np.float32)


_Z = {}
def _z(a):
    """Standardize using train-fit stats stored on first call per array role."""
    return (a - a.mean()) / (a.std() + 1e-6)


def fusion_loss(a, vo, wheel, gt):
    fused = a * vo + (1 - a) * wheel          # a is (N,2) per-dimension
    return ((fused - gt) ** 2).sum(dim=1).mean()


def train_weight(feat, vo, wheel, gt, feat_va, vo_va, wheel_va, gt_va, epochs=40):
    model = WeightNet(feat.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(
        torch.from_numpy(feat), torch.from_numpy(vo),
        torch.from_numpy(wheel), torch.from_numpy(gt)),
        batch_size=4096, shuffle=True)
    tva = [torch.from_numpy(x).to(DEVICE) for x in (feat_va, vo_va, wheel_va, gt_va)]
    best, best_state = math.inf, None
    for ep in range(epochs):
        model.train()
        for xb, vb, wb, gb in loader:
            xb, vb, wb, gb = xb.to(DEVICE), vb.to(DEVICE), wb.to(DEVICE), gb.to(DEVICE)
            loss = fusion_loss(model(xb), vb, wb, gb)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = fusion_loss(model(tva[0]), tva[1], tva[2], tva[3]).item()
        if v < best:
            best, best_state = v, {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_weight(model, feat):
    return model(torch.from_numpy(feat).to(DEVICE)).cpu().numpy()


def learned_run_ate(d, model, models, scale, use_unc, run_ids):
    """Integrate each run with the learned per-frame weight; return mean ATE."""
    f = h5py.File(d["hdf5_path"], "r")
    rid = f["run_id"][:]
    vals = []
    for r in run_ids:
        sig = d["all_signals"][r]
        idx = np.where(rid == r)[0]
        fused = f["fused_pose"][idx[0]:idx[-1] + 1]
        X, valid = rq1.build_features(sig, rq1.PAST_WINDOW_S)
        vo, wheel = sig["vo_mps"], sig["wheel_cmd"]
        vo_ok = ~np.isnan(vo).any(axis=1)
        a = np.zeros((len(X), 2), dtype=np.float32)   # default: wheel-only
        usable = valid & vo_ok
        if usable.any():
            feat = ((X[usable] - d["mu"]) / d["sd"]).astype(np.float32)
            if use_unc:
                mu, var = ensemble_predict(models, feat)
                sg = np.sqrt(var) * scale
                feat = np.concatenate([feat, _z(mu)[:, None], _z(sg)[:, None]], axis=1).astype(np.float32)
            a[usable] = predict_weight(model, feat)
        vo_c = clip_vo(np.nan_to_num(vo))
        lin = a[:, 0] * vo_c[:, 0] + (1 - a[:, 0]) * wheel[:, 0]
        ang = a[:, 1] * vo_c[:, 1] + (1 - a[:, 1]) * wheel[:, 1]
        xs, ys = integrate(sig["ts"], lin, ang, fused[0])
        vals.append(ate(xs, ys, fused[:, 0], fused[:, 1]))
    return float(np.mean(vals)), vals


def main():
    print(f"device={DEVICE}  label=GT")
    d = c.build_dataset(label_source="gt")
    models, scale = load_ensemble(d["Xtr"].shape[1])

    # --- baselines / Q4 gated policies (recompute on GT label) -----------
    f = h5py.File(d["hdf5_path"], "r")
    from research_question_4 import run_frames
    val_fr = [run_frames(f, r, models, scale, d["mu"], d["sd"]) for r in d["val_ids"]]
    test_fr = [run_frames(f, r, models, scale, d["mu"], d["sd"]) for r in d["test_ids"]]
    mu_va, var_va = ensemble_predict(models, d["Xva"]); sig_va = np.sqrt(var_va) * scale
    mu_grid = np.quantile(mu_va, np.linspace(0.2, 0.95, 12))
    unc_grid = np.quantile(sig_va, np.linspace(0.3, 0.97, 10))
    tuned = {p: tune_policy(val_fr, p, mu_grid, unc_grid)[1]
             for p in ("const_blend", "risk_only", "three_way")}

    def ate_of(pol):
        p = tuned.get(pol, {})
        return mean_ate(test_fr, pol, p.get("tau_risk", np.inf),
                        p.get("tau_unc", np.inf), tuple(p.get("blend", (1.0, 1.0))))[0]

    results = {
        "wheel_only":  mean_ate(test_fr, "wheel", 0, 0, (0.0, 0.0))[0],
        "const_blend": ate_of("const_blend"),
        "risk_gated":  ate_of("risk_only"),
        "three_way":   ate_of("three_way"),
    }

    # --- learned per-step weight (two variants) --------------------------
    for use_unc, name in [(False, "learned_weight"), (True, "learned_weight+unc")]:
        feat, vo, wheel, gt = collect_fusion(d, d["train_ids"], models, scale, use_unc)
        fva, vva, wva, gva = collect_fusion(d, d["val_ids"], models, scale, use_unc)
        model = train_weight(feat, vo, wheel, gt, fva, vva, wva, gva)
        a_tr = predict_weight(model, feat)
        results[name], _ = learned_run_ate(d, model, models, scale, use_unc, d["test_ids"])
        print(f"  {name}: mean weight a_lin={a_tr[:,0].mean():.2f} a_ang={a_tr[:,1].mean():.2f}"
              f"  test ATE={results[name]:.3f}")

    print("\nmean test ATE (m) vs fused_pose:")
    for k in ("wheel_only", "const_blend", "risk_gated", "three_way",
              "learned_weight", "learned_weight+unc"):
        flag = "  <-- beats wheel" if results[k] < results["wheel_only"] else ""
        print(f"  {k:<20} {results[k]:.3f}{flag}")
    best = min(results, key=results.get)
    print(f"\nbest policy: {best} ({results[best]:.3f} m), "
          f"{100*(results['wheel_only']-results[best])/results['wheel_only']:+.1f}% vs wheel-only")

    (OUT_DIR / "q4_learned_summary.json").write_text(
        json.dumps({k: float(v) for k, v in results.items()}
                   | {"tuned": tuned, "label": "gt"}, indent=2))
    print(f"Saved -> {OUT_DIR/'q4_learned_summary.json'}")


if __name__ == "__main__":
    main()
