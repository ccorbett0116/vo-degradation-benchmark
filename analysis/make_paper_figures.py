"""Generate the Option-A paper figures from already-computed results
(manifest + analysis/measured_gt_validation.json). No HDF5, no training."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
man = {m["run_id"]: m for m in json.load(open(HERE.parent / "Data/comprehensive_dataset/manifest.json"))}
val = json.load(open(HERE / "measured_gt_validation.json"))
rows = val["per_run"]


# ── Figure 1: dataset motivation — measured loop-closure by regime ───────────
fig, ax = plt.subplots(figsize=(5.2, 3.6))
phases = {"A (camera off)": [], "B (static)": [], "C (dynamic)": []}
for m in man.values():
    p = m["phase"]
    key = "A (camera off)" if p == "A" else "B (static)" if p == "B" else "C (dynamic)"
    phases[key].append(m["physical_lce_m"])
labels = list(phases)
data = [phases[k] for k in labels]
bp = ax.boxplot(data, labels=labels, showmeans=True, widths=0.5)
for i, d in enumerate(data, 1):
    ax.scatter(np.full(len(d), i) + np.random.uniform(-.08, .08, len(d)), d,
               s=14, alpha=.5, color="#377eb8", zorder=3)
ax.set_ylabel("measured figure-8 loop-closure error (m)")
ax.set_title("Fusing low-cost VO degrades measured accuracy ~8×")
ax.grid(alpha=.3, axis="y")
fig.tight_layout(); fig.savefig(HERE / "fig_dataset_loopclosure.png", dpi=150)
print("wrote fig_dataset_loopclosure.png")


# ── Figure 2 (centerpiece): predicted risk vs estimated vs measured ──────────
conds = sorted({r["condition"] for r in rows})
cmap = plt.cm.tab10(np.linspace(0, 1, len(conds)))
cof = {c: cmap[i] for i, c in enumerate(conds)}
pred = np.array([r["pred_risk"] for r in rows])
phys = np.array([r["physical_lce_m"] for r in rows])
est = np.array([r["estimated_lce_m"] for r in rows])

fig, ax = plt.subplots(1, 2, figsize=(9.4, 4.2), sharey=True)
for a, x, name, rr in [
    (ax[0], est, "filter self-estimate  estimated_lce_m (m)", val["pearson_pred_estimated"]),
    (ax[1], phys, "MEASURED ground truth  physical_lce_m (m)", val["pearson_pred_physical"])]:
    a.scatter(x, pred, c=[cof[r["condition"]] for r in rows], s=34, alpha=.85,
              edgecolor="k", linewidth=.3)
    # least-squares trend line
    slope, intercept = np.polyfit(x, pred, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    a.plot(xs, intercept + slope * xs, "k--", lw=1, alpha=.7)
    a.set_xlabel(name); a.set_title(f"Pearson r = {rr:+.2f}")
    a.grid(alpha=.3)
ax[0].set_ylabel("mean predicted risk (out-of-fold)")
handles = [plt.Line2D([], [], marker="o", ls="", color=cof[c], label=c) for c in conds]
ax[1].legend(handles=handles, fontsize=6.5, loc="upper right", framealpha=.9)
fig.suptitle("Predicted risk tracks the onboard estimate, not measured accuracy", y=1.0)
fig.tight_layout(); fig.savefig(HERE / "fig_measured_gt_gap.png", dpi=150)
print("wrote fig_measured_gt_gap.png")


# ── Figure 3 (mechanism): per-condition predicted risk vs measured error ─────
by = {}
for r in rows:
    by.setdefault(r["condition"], []).append((r["pred_risk"], r["physical_lce_m"]))
items = sorted(by.items(), key=lambda kv: np.mean([v[1] for v in kv[1]]))
names = [k for k, _ in items]
mpred = np.array([np.mean([v[0] for v in vs]) for _, vs in items])
mphys = np.array([np.mean([v[1] for v in vs]) for _, vs in items])
norm = lambda z: (z - z.min()) / (z.max() - z.min())
x = np.arange(len(names)); w = 0.38
fig, ax = plt.subplots(figsize=(7.6, 3.8))
ax.bar(x - w/2, norm(mpred), w, label="predicted risk (norm.)", color="#e41a1c")
ax.bar(x + w/2, norm(mphys), w, label="measured error (norm.)", color="#4daf4a")
ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7.5)
ax.set_ylabel("min-max normalized")
ax.set_title("The monitor flags the wrong regime\n(conditions sorted by measured error →)")
ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
fig.tight_layout(); fig.savefig(HERE / "fig_condition_mismatch.png", dpi=150)
print("wrote fig_condition_mismatch.png")
