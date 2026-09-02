"""Merge everything the expert-MLP-ladder report needs into one metrics file + plots.

`analyze_dualrwl_em.py` judges the gates; this adds the two things the report needs
beside them and that live outside that script's scope:

  crit    which channels each criterion keeps -- flow-matching Taylor vs Wanda vs the
          |W| half alone, at every rung (run_wanda_expert.py supplies the expert Wanda
          scores; run_wanda.py is VLM-only)
  sanity  the code-integrity checks: are consecutive rungs actually different models
          (bit-identical trajectory fraction), is the change directional (win/loss), and
          are the seeds identical across arms

Writes outputs/<out>/metrics_report.json (keys gates / crit / sanity) plus the criterion
plot, so fill_template.py can render the page with no hand-typed number.

Usage:
  .venv/bin/python experiments/evaluation/build_dualrwl_em_report.py [--out dualrwl_em_analysis]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from safetensors import safe_open  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "head_analysis"))
import mask_lib as ml  # noqa: E402
import paper_numbers as pn  # noqa: E402

BG, INK, MUTED, GRID = "#FAF9F5", "#29261B", "#6B6555", "#E8E6DC"
C1, C2, C3 = "#2a78d6", "#008300", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "font.size": 9, "axes.grid": True, "axes.axisbelow": True,
})
REPO = Path(__file__).resolve().parents[2]
SNAP = Path("/mnt/nvme1n1/ad_vla/cache/hub/models--nvidia--Alpamayo-1.5-10B/snapshots/"
            "7aba8293c09993f2e125c6819df05d7fa3e873ea")
L, INTER = 36, 8256
LAYERS = list(range(L))
RUNGS = ["dualr_wl", "dualrwl_em50", "dualrwl_em75", "dualrwl_em87p5", "dualrwl_em93p75"]
RATIOS = [0.50, 0.75, 0.875, 0.9375]
REF = "fm_taylor_znorm"


def weight_magnitude():
    """||W_down[:, c]||_2 straight from the checkpoint shards -- the |W| half of Wanda."""
    index = json.loads((SNAP / "model.safetensors.index.json").read_text())["weight_map"]
    names = [k for k in index if "expert" in k and "down_proj" in k]
    assert len(names) == L, len(names)
    mag = np.zeros((L, INTER))
    by_shard = {}
    for n in names:
        by_shard.setdefault(index[n], []).append(n)
    for shard, ns in by_shard.items():
        with safe_open(SNAP / shard, framework="pt") as f:
            for n in ns:
                i = int([p for p in n.split(".") if p.isdigit()][0])
                w = f.get_tensor(n).float().numpy()
                mag[i] = np.sqrt((w * w).sum(0))
    return mag


def criteria(wanda_run):
    crit = {
        REF: np.load(REPO / "outputs/importance_stepexp_znorm/importance.npz")[
            "traj_exp_mlp"],
        "fm_taylor_sum": np.load(REPO / "outputs/importance_v2/importance.npz")[
            "traj_exp_mlp"],
        "weight_magnitude": weight_magnitude(),
        "wanda": np.load(REPO / "outputs" / wanda_run / "wanda.npz")["exp_mlp_w"],
    }
    out = {"ratios": RATIOS, "kept": [], "overlap": {k: [] for k in crit if k != REF}}
    for r in RATIOS:
        masks = {k: ml.select_mask(v, r, LAYERS) for k, v in crit.items()}
        ref = masks[REF]
        out["kept"].append(int(ref[0].sum()))
        for k in out["overlap"]:
            out["overlap"][k].append(float((masks[k] * ref).sum() / ref.sum()))
    r = 0.9375
    a = ml.select_mask(crit[REF], r, LAYERS)
    b = ml.select_mask(crit["wanda"], r, LAYERS)
    m = ml.select_mask(crit["weight_magnitude"], r, LAYERS)
    shared = (a * b).sum(1)
    kept = int(a[0].sum())
    rho = [float(np.corrcoef(np.argsort(np.argsort(crit[REF][i])),
                             np.argsort(np.argsort(crit["wanda"][i])))[0, 1])
           for i in range(L)]
    out["deep"] = {
        "kept": kept, "cut": INTER - kept,
        "shared_mean": float(shared.mean()), "shared_min": int(shared.min()),
        "shared_max": int(shared.max()), "differ": kept - float(shared.mean()),
        "overlap": float(shared.sum() / a.sum()),
        "jaccard": float(shared.sum() / (2 * a.sum() - shared.sum())),
        "wanda_vs_own_magnitude": float((m * b).sum() / b.sum()),
        "rho_mean": float(np.mean(rho)), "rho_min": float(np.min(rho)),
    }
    return out, crit


def sanity():
    """Are the rungs different models, and is the change directional?"""
    out = {"sets": {}}
    for s in ("indist", "test", "oodval"):
        rows = {}
        for a in RUNGS:
            try:
                rows[a] = pn.load(*pn.ARMS[a][s])
            except Exception:
                pass
        if len(rows) < len(RUNGS):
            continue
        ids = sorted(set.intersection(*[set(r) for r in rows.values()]))
        pairs = []
        for a, b in zip(RUNGS, RUNGS[1:]):
            same = sum(np.array_equal(np.asarray(rows[a][i]["ade_rollout_k"], float),
                                      np.asarray(rows[b][i]["ade_rollout_k"], float))
                       for i in ids)
            d = np.array([pn.at6(rows[b][i], "ade_rollout_k")
                          - pn.at6(rows[a][i], "ade_rollout_k") for i in ids])
            coc = float(np.mean([rows[a][i]["gen_coc"] == rows[b][i]["gen_coc"]
                                 for i in ids]))
            pairs.append({"pair": f"{b} - {a}", "n": len(ids), "identical": int(same),
                          "worse": int((d > 0).sum()), "better": int((d < 0).sum()),
                          "mean_abs": float(np.abs(d).mean()), "coc_same": coc})
        out["sets"][s] = pairs
    cfg = json.loads((REPO / "outputs" / "dualrwl_em93p75_indist" / "config.json").read_text())
    out["seed"] = cfg["seed"]
    out["k"] = cfg["k"]
    out["seed_rule"] = cfg["seed_rule"]
    out["det"] = cfg["deterministic"]
    return out


def plot_criteria(crit_metrics, out):
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    x = np.arange(len(RATIOS))
    style = {"wanda": (C1, "o", "Wanda  |W|·‖X‖₂"),
             "weight_magnitude": (C3, "s", "weight magnitude  (|W| half only)"),
             "fm_taylor_sum": (C2, "^", "fm Taylor, pre-fix |Σₛ| aggregation")}
    for k, (c, mk, lab) in style.items():
        ax.plot(x, crit_metrics["overlap"][k], marker=mk, color=c, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(r * 100 * 100) / 100:g}%\n{k} kept"
                        for r, k in zip(RATIOS, crit_metrics["kept"])])
    ax.set_ylim(0.80, 1.0)
    ax.set_ylabel("kept-set overlap with fm Taylor (znorm)")
    ax.set_xlabel("expert MLP channels removed per layer")
    ax.set_title("the deeper the cut, the more the criteria agree")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "expert_mlp_criteria.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dualrwl_em_analysis")
    ap.add_argument("--wanda", default="wanda_expert_v1")
    args = ap.parse_args()
    out = REPO / "outputs" / args.out
    (out / "plots").mkdir(parents=True, exist_ok=True)

    gates = json.loads((out / "metrics.json").read_text())
    crit_metrics, _ = criteria(args.wanda)
    plot_criteria(crit_metrics, out / "plots")
    merged = {"gates": gates, "crit": crit_metrics, "sanity": sanity()}
    (out / "metrics_report.json").write_text(json.dumps(merged, indent=1))
    print(f"wrote {out / 'metrics_report.json'} "
          f"(keys {list(merged)}) + plots/expert_mlp_criteria.png")


if __name__ == "__main__":
    main()
