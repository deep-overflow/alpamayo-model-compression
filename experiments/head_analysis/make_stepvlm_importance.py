"""Re-aggregate the per-step VLM measurement into drop-in `importance.npz` files.

`make_slim.py` is deliberately NOT modified. The u40_v2 family already takes an
`--importance <exp_id>` argument (that is how `dual_u40_v2 --importance importance_v1`
reproduces the shipped `slim_dual_uniform` bit-identically), so a new aggregation only has
to be published in the same format under a new exp id:

    python make_stepvlm_importance.py --stepvlm importance_stepvlm_v1 --ref importance_v2_ada
    python make_slim.py --config dual_u40_v2 --importance importance_stepvlm_znorm --no-state

Each output copies every key from `--ref` and replaces only `traj_vlm_q` / `traj_vlm_mlp`
(and, with --with-kv, `traj_kv_k` / `traj_kv_v`). `coc_*` is copied untouched: the CoC
objective is a single NLL backward with no step axis, so it cannot have this defect, and
leaving it alone keeps the arm a one-factor change to max(rank traj, rank coc).

The reference must be measured on the SAME GPU architecture as the step run: Ada-vs-
Blackwell drift moves the u40_v2 selection by 2-3% (kept-overlap 0.97-0.98), which is a
quarter of the effect size this track is chasing.

Aggregations:
  sum     mean_clips |sum_s g|         the shipped rule; must reproduce --ref (gate V0)
  znorm   mean_s of within-layer z-scored |g_s|   the expert tower's winner
  trimz   znorm computed on a 10%-trimmed clip mean (both axes)
  seedz   the one-backward approximation measured in the same pass
  sumabs  mean_clips sum_s |g_s|       removes sign cancellation while LEAVING the step
          mass ordering intact. Added after znorm regressed on this tower: the regression
          was explained by znorm flattening a monotone mass profile that turned out to be
          real signal, and sumabs is the one aggregation that does not flatten it.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

REPO = Path(__file__).resolve().parents[2]
AGGS = ("sum", "znorm", "trimz", "seedz", "sumabs")


def zscore_layers(a):
    """Within-layer z-score of a (L, U) map, so steps of different mass weigh equally."""
    m = a.mean(1, keepdims=True)
    s = a.std(1, keepdims=True)
    return (a - m) / np.where(s > 0, s, 1.0)


def trimmed_mean(a, frac=0.10):
    """Per-entry mean over axis 0 with the largest `frac` of clips dropped."""
    keep = max(a.shape[0] - round(a.shape[0] * frac), 1)
    return np.sort(a, axis=0)[:keep].mean(0)


def aggregate(z, pc, agg, unit):
    """(L, U) score for one aggregation. `z` is the clip-mean npz, `pc` the per-clip one."""
    if agg == "sum":
        return z[f"{unit}_shipped"].astype(np.float64)
    if agg == "sumabs":
        # the step axis aggregated by the clip axis's own rule: |.| first, then add
        return z[f"{unit}_abs_step"].astype(np.float64).sum(0)
    if agg == "seedz":
        return z[f"seedz_{unit}"].astype(np.float64)
    if agg == "znorm":
        return np.mean([zscore_layers(a) for a in z[f"{unit}_abs_step"].astype(np.float64)],
                       axis=0)
    if agg == "trimz":
        # trim on the clip axis first, then normalise the step axis -- both defects at once
        trimmed = trimmed_mean(pc[f"{unit}_abs_step"])          # (S, L, U)
        return np.mean([zscore_layers(a.astype(np.float64)) for a in trimmed], axis=0)
    raise ValueError(agg)


def kv_aggregate(z, agg):
    """KV groups are scored on the cache tensors, so only the step axis applies."""
    if agg in ("sum", "seedz"):
        return z["kv_k_shipped"].astype(np.float64), z["kv_v_shipped"].astype(np.float64)
    return (np.mean([zscore_layers(a) for a in z["kv_k_abs_step"].astype(np.float64)], axis=0),
            np.mean([zscore_layers(a) for a in z["kv_v_abs_step"].astype(np.float64)], axis=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stepvlm", default="importance_stepvlm_v1")
    ap.add_argument("--ref", default="importance_v2_ada",
                    help="same-architecture run_importance output; everything but "
                         "traj_vlm_* is copied from it verbatim")
    ap.add_argument("--aggs", nargs="+", default=list(AGGS))
    ap.add_argument("--prefix", default="importance_stepvlm")
    ap.add_argument("--with-kv", action="store_true",
                    help="also replace traj_kv_*; off by default because the u40_v2 family "
                         "does not touch KV (VLM only) and leaving it keeps the arms "
                         "one-factor. Turn on for cocsafe / j_traj / integrated_mag.")
    args = ap.parse_args()

    src = REPO / "outputs" / args.stepvlm
    z = dict(np.load(src / "step_importance_vlm.npz"))
    pc = dict(np.load(src / "step_importance_vlm_perclip.npz"))
    ref_dir = REPO / "outputs" / args.ref
    ref = dict(np.load(ref_dir / "importance.npz"))
    src_cfg = json.loads((src / "config.json").read_text())
    ref_cfg = json.loads((ref_dir / "config.json").read_text())

    # the two runs must describe the same clips, or "replace only traj" is not one-factor
    if src_cfg["clip_ids"] != ref_cfg["clip_ids"]:
        raise ValueError("step run and reference cover different clips; "
                         "the copied coc_* would not match the replaced traj_*")

    for agg in args.aggs:
        out_dir = REPO / "outputs" / f"{args.prefix}_{agg}"
        out_dir.mkdir(parents=True, exist_ok=True)
        arrays = dict(ref)
        arrays["traj_vlm_q"] = aggregate(z, pc, agg, "q")
        arrays["traj_vlm_mlp"] = aggregate(z, pc, agg, "mlp")
        if args.with_kv:
            arrays["traj_kv_k"], arrays["traj_kv_v"] = kv_aggregate(z, agg)
        np.savez(out_dir / "importance.npz", **arrays)
        # make_slim reads importance.npz only, but keep the provenance next to it
        (out_dir / "config.json").write_text(json.dumps({
            "derived_from": {"step_run": args.stepvlm, "reference": args.ref},
            "aggregation": agg, "replaced_keys": sorted(
                k for k in arrays if not np.array_equal(arrays[k], ref[k])),
            "with_kv": args.with_kv,
            "num_clips": src_cfg["num_clips"], "fm_steps": src_cfg["fm_steps"],
            "model_revision": src_cfg["model_revision"], "gpu": src_cfg["gpu"],
            "note": "drop-in for make_slim.py --importance; coc_* copied from the reference "
                    "because the CoC objective has no denoising-step axis",
        }, indent=2))
        if (ref_dir / "importance_perclip.npz").exists() and agg == "sum":
            # analyze_cvar and friends expect it beside a full importance run
            shutil.copy(ref_dir / "importance_perclip.npz",
                        out_dir / "importance_perclip.npz")
        print(f"{out_dir.name}: replaced {len([k for k in arrays if not np.array_equal(arrays[k], ref[k])])} keys",
              flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
