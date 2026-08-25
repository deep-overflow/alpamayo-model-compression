"""Publish the expert-tower step aggregations as drop-in `importance.npz` files.

The D stage measured its arms as runtime masks (`run_expert_agg.py` installs PruneMasks and
never writes a checkpoint), which is enough for open-loop but not for closed-loop: alpasim
drivers load a real `slim_state.pt`. This turns the winning aggregation into something
`make_slim.py --importance` can consume, so the checkpoint is built from the same numbers
that produced the open-loop result.

`zscore_layers` is imported from run_expert_agg rather than reimplemented, so the published
score is byte-identical to the arm that was evaluated.

Usage:
  python make_stepexp_importance.py --stepimp stepimp_fm_perstep_v2 --ref importance_v2_ada
"""

import argparse
import json
from pathlib import Path

import numpy as np

from run_expert_agg import trimmed_mean, zscore_layers

REPO = Path(__file__).resolve().parents[2]
AGGS = ("sum", "znorm", "sumabs", "trimclip_znorm")


def aggregate(agg, z, pc, unit):
    if agg == "sum":
        return z[f"{unit}_shipped"].astype(np.float64)
    if agg == "sumabs":
        return z[f"{unit}_abs_step"].astype(np.float64).sum(0)
    if agg == "znorm":
        return np.mean([zscore_layers(a) for a in z[f"{unit}_abs_step"].astype(np.float64)],
                       axis=0)
    if agg == "trimclip_znorm":
        return np.mean([zscore_layers(a.astype(np.float64))
                        for a in trimmed_mean(pc[f"{unit}_abs_step"])], axis=0)
    raise ValueError(agg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stepimp", default="stepimp_fm_perstep_v2")
    ap.add_argument("--ref", default="importance_v2_ada",
                    help="same-architecture run_importance output; every key except "
                         "traj_exp_* is copied from it verbatim")
    ap.add_argument("--aggs", nargs="+", default=list(AGGS))
    ap.add_argument("--prefix", default="importance_stepexp")
    args = ap.parse_args()

    src = REPO / "outputs" / args.stepimp
    z = dict(np.load(src / "step_importance.npz"))
    pc = dict(np.load(src / "step_importance_perclip.npz"))
    ref_dir = REPO / "outputs" / args.ref
    ref = dict(np.load(ref_dir / "importance.npz"))
    src_cfg = json.loads((src / "config.json").read_text())
    ref_cfg = json.loads((ref_dir / "config.json").read_text())
    if src_cfg["clip_ids"] != ref_cfg["clip_ids"]:
        raise ValueError("step run and reference cover different clips")

    for agg in args.aggs:
        out_dir = REPO / "outputs" / f"{args.prefix}_{agg}"
        out_dir.mkdir(parents=True, exist_ok=True)
        arrays = dict(ref)
        arrays["traj_exp_q"] = aggregate(agg, z, pc, "q")
        arrays["traj_exp_mlp"] = aggregate(agg, z, pc, "mlp")
        np.savez(out_dir / "importance.npz", **arrays)
        (out_dir / "config.json").write_text(json.dumps({
            "derived_from": {"step_run": args.stepimp, "reference": args.ref},
            "aggregation": agg, "tower": "expert",
            "replaced_keys": ["traj_exp_q", "traj_exp_mlp"],
            "num_clips": src_cfg["num_clips"], "fm_steps": src_cfg["fm_steps"],
            "model_revision": src_cfg["model_revision"],
            "note": "drop-in for make_slim.py --importance; only the expert trajectory "
                    "score is re-aggregated over the denoising-step axis",
        }, indent=2))
        print(f"{out_dir.name}: traj_exp_* replaced", flush=True)


if __name__ == "__main__":
    main()
