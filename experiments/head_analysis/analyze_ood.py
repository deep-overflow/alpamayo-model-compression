"""Aggregate the OOD evaluation across models (baseline + slims).

Reads outputs/ood_eval/{model}.json (per-clip minADE_tf/rollout, NLL of GT CoC, generated
CoC text) and produces: the headline table (per model: minADE tf/rollout, NLL, deltas vs
baseline over the shared clip set), the reasoning-propagation gap (rollout - tf), a
per-cluster breakdown, and a handful of qualitative GT-vs-generated CoC samples.

Usage: python analyze_ood.py
"""

import json
from pathlib import Path

import numpy as np

REPO = Path("/workspace/alpamayo-model-compression")
OOD = REPO / "outputs" / "ood_eval"
MODELS = ["baseline", "slim_integrated_mag", "slim_cocsafe_r20", "slim_cocsafe_r30"]
LABEL = {"baseline": "baseline (full)", "slim_integrated_mag": "slim integrated (-29.3%)",
         "slim_cocsafe_r20": "slim cocsafe_r20 (-18.1%)",
         "slim_cocsafe_r30": "slim cocsafe_r30 (-24.1%)"}


def load(model):
    p = OOD / f"{model}.json"
    if not p.exists():
        return None
    return {r["clip_id"]: r for r in json.loads(p.read_text())}


def main():
    data = {m: load(m) for m in MODELS}
    have = [m for m in MODELS if data[m]]
    if "baseline" not in have:
        print("baseline not ready yet; have:", have)
        return
    # clip set shared by all available models (paired)
    shared = set(data[have[0]])
    for m in have:
        shared &= set(data[m])
    shared = sorted(shared)
    n = len(shared)
    base = data["baseline"]

    out_dir = OOD
    lines = [f"OOD evaluation ({n} clips shared, GT-CoC context; in-dist baseline "
             f"minADE~0.79 / NLL~0.21 for reference)\n"]
    lines.append(f"{'model':28s} {'minADE_tf':>9s} {'minADE_ro':>9s} {'ro-tf':>6s} "
                 f"{'NLL_gt':>7s} {'d_tf':>7s} {'d_ro':>7s}")
    rows_out = {}
    for m in have:
        d = data[m]
        tf = np.array([d[c]["minADE_tf"] for c in shared])
        ro = np.array([d[c]["minADE_rollout"] for c in shared])
        nll = np.array([d[c]["nll_gtcoc"] for c in shared])
        b_tf = np.array([base[c]["minADE_tf"] for c in shared])
        b_ro = np.array([base[c]["minADE_rollout"] for c in shared])
        d_tf = float((tf - b_tf).mean())
        d_ro = float((ro - b_ro).mean())
        rows_out[m] = {"minADE_tf": float(tf.mean()), "minADE_rollout": float(ro.mean()),
                       "gap_ro_tf": float((ro - tf).mean()), "nll_gtcoc": float(nll.mean()),
                       "d_tf_vs_base": d_tf, "d_ro_vs_base": d_ro}
        lines.append(f"{LABEL[m]:28s} {tf.mean():9.4f} {ro.mean():9.4f} "
                     f"{(ro - tf).mean():6.3f} {nll.mean():7.3f} "
                     f"{d_tf:+7.4f} {d_ro:+7.4f}")

    # per-cluster minADE_tf delta vs baseline
    lines.append("\n=== per-cluster minADE_tf Δ vs baseline ===")
    clusters = sorted({base[c]["cluster"] for c in shared},
                      key=lambda cl: -sum(base[c]["cluster"] == cl for c in shared))
    hdr = f"{'cluster':40s} {'n':>3s} " + " ".join(f"{m.split('_')[1][:6]:>7s}"
                                                   for m in have if m != "baseline")
    lines.append(hdr)
    for cl in clusters:
        cc = [c for c in shared if base[c]["cluster"] == cl]
        cells = []
        for m in have:
            if m == "baseline":
                continue
            d = np.mean([data[m][c]["minADE_tf"] - base[c]["minADE_tf"] for c in cc])
            cells.append(f"{d:+7.3f}")
        lines.append(f"{cl:40s} {len(cc):3d} " + " ".join(cells))

    txt = "\n".join(lines)
    print(txt)
    (out_dir / "summary.txt").write_text(txt + "\n")
    (out_dir / "aggregate.json").write_text(json.dumps(
        {"n_shared": n, "models": rows_out}, indent=2))

    # qualitative: GT vs each model's generated CoC on a few diverse clusters
    qual = []
    seen = set()
    for c in shared:
        cl = base[c]["cluster"]
        if cl in seen:
            continue
        seen.add(cl)
        qual.append({"clip": c, "cluster": cl, "gt_coc": base[c]["gt_coc"],
                     "gen": {m: data[m][c]["gen_coc"] for m in have}})
        if len(qual) >= 8:
            break
    (out_dir / "qualitative.json").write_text(json.dumps(qual, indent=2))
    print(f"\nqualitative samples ({len(qual)} clusters) -> {out_dir / 'qualitative.json'}")


if __name__ == "__main__":
    main()
