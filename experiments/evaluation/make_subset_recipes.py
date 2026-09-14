"""Calibration-subset recipes for the per-clip influence study.

plans/2026-09-14_calibration-clip-influence.md. `dual_u40_v2` is built from the mean of
the per-clip Taylor arrays, and `outputs/importance_v2/importance_perclip.npz` stores
those arrays for all 100 calibration clips -- so the criterion for ANY subset of
calib_100 is exact numpy, with no GPU and no re-measurement. Verified: the per-clip mean
reproduces the shipped `importance.npz` to 1.9e-08 absolute (1.3e-07 relative), which is
float32 accumulation order, not a difference.

What still costs something is the build, and that is what this batches: one model load
serves every subset, because `build_masks` only reads `imp` and the two configs while
`apply_surgery` is the only thing that mutates the model. Skipping the surgery turns 32
builds from 48 GPU-minutes into 5.

That shortcut is exactly where a silent divergence from the shipped recipe would live, so
--verify is a gate rather than a courtesy:

  V0a  the all-100 subset must reproduce `outputs/slim_dual_u40_v2/slim_meta.json` unit
       for unit -- the same check that validated the u40 recipe in the first place
  V0b  one subset built here must equal what `make_slim.py --no-state` writes for the
       same subset, key for key

Usage:
  python experiments/evaluation/make_subset_recipes.py --design          # no GPU
  python experiments/evaluation/make_subset_recipes.py --verify --gpu 4
  python experiments/evaluation/make_subset_recipes.py --build --arms 0 32 --gpu 4
  python experiments/evaluation/make_subset_recipes.py --dump-importance 0 --out imp_sub00
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))
sys.path.insert(0, str(Path(__file__).parent))

import analysis_lib as lib
import make_slim as ms
import slim_lib as sl
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from expert_per_clip import reserve_gpu  # also installs the gated-repo hub patch

PERCLIP = "importance_v2"
CONFIG = "dual_u40_v2"
DESIGN = REPO / "outputs" / "clipinfluence" / "design.json"


def make_design(n_clips, size, m, seed):
    """M subsets of exactly `size` clips, in COMPLEMENTARY PAIRS when 2*size == n_clips.

    Fixed size, not Bernoulli(0.5): under Bernoulli the subset size itself varies by +-5
    and its effect on minADE is confounded with membership, which is the quantity being
    estimated. With size held, tau_i is a clean in-vs-out contrast.

    Each pair is one random permutation split at `size`, so its two halves partition the
    pool and every clip lands in exactly one of them. Three things follow, and all three
    matter here:

      * over M = 2p subsets each clip appears exactly p times, so every tau_i has the
        same minimal variance -- independent draws left the SE multiplier spread over
        0.354..0.393 at M=32, and the widest-variance clip is the one that decides a
        max|tau| threshold
      * every even-length PREFIX is itself balanced, so M can be extended later without
        rebalancing or discarding what already ran
      * the pair is antithetic: y_A - y_B is a contrast in which subset-level nuisance
        (how many clips of some bucket happened to be drawn at all) cancels
    """
    rng = np.random.default_rng(seed)
    if 2 * size != n_clips or m % 2:
        return [sorted(rng.choice(n_clips, size=size, replace=False).tolist())
                for _ in range(m)]
    subsets = []
    for _ in range(m // 2):
        p = rng.permutation(n_clips)
        subsets.append(sorted(p[:size].tolist()))
        subsets.append(sorted(p[size:].tolist()))
    return subsets


def subset_importance(per, idx):
    """Mean over the selected clips, every key. idx=None means all clips."""
    return {k: (v.mean(0) if idx is None else v[idx].mean(0)) for k, v in per.items()}


def build_meta(model, imp, arm, subset, full_total):
    vq, vm, eq, em, kvonly = ms.build_masks(CONFIG, imp, model)
    meta = {"vlm": [], "expert": [], "kvonly_layers": list(kvonly)}
    for li in range(vq.shape[0]):
        meta["vlm"].append({"q": np.nonzero(vq[li])[0].tolist(),
                            "mlp": np.nonzero(vm[li])[0].tolist()})
    for li in range(eq.shape[0]):
        meta["expert"].append({"q": np.nonzero(eq[li])[0].tolist(),
                               "mlp": np.nonzero(em[li])[0].tolist()})
    removed = ms.expected_removed(model, vq, vm, eq, em, kvonly)
    meta["config"] = CONFIG
    meta["importance_from"] = f"{PERCLIP}:subset"
    meta["params"] = {"full": full_total, "slim": full_total - removed, "removed": removed}
    meta["subset"] = {"arm": arm, "n": len(subset), "clips": subset}
    return meta


def kept_counts(meta):
    q = sum(len(m["q"]) for m in meta["vlm"])
    mlp = sum(len(m["mlp"]) for m in meta["vlm"])
    return q, mlp


def same_selection(a, b):
    """Compare two metas on everything load_slim() actually reads."""
    for tower in ("vlm", "expert"):
        if len(a[tower]) != len(b[tower]):
            return f"{tower}: {len(a[tower])} vs {len(b[tower])} layers"
        for li, (x, y) in enumerate(zip(a[tower], b[tower])):
            for key in ("q", "mlp"):
                if x[key] != y[key]:
                    return f"{tower}[{li}].{key} differs ({len(x[key])} vs {len(y[key])})"
    if list(a["kvonly_layers"]) != list(b["kvonly_layers"]):
        return "kvonly_layers differ"
    return None


def load_model(gpu, reserve_gb):
    reserve_gpu(reserve_gb, devices=None if gpu is None else [gpu])
    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B", revision=ms.MODEL_REV, dtype=torch.bfloat16).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    lib.set_vlm_attn_impl(model, "sdpa")
    lib.set_expert_attn_impl(model, "sdpa")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", action="store_true", help="write design.json and exit")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--verify", action="store_true", help="gate V0a/V0b")
    ap.add_argument("--dump-importance", type=int, default=None,
                    help="arm index; writes that subset's importance.npz for make_slim.py")
    ap.add_argument("--out", type=str, default=None, help="exp-id for --dump-importance")
    ap.add_argument("--arms", type=int, nargs=2, default=None, metavar=("LO", "HI"))
    ap.add_argument("--m", type=int, default=64, help="subsets in the design")
    ap.add_argument("--size", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--prefix", type=str, default="slim_subinf")
    ap.add_argument("--drop", type=int, nargs="+", default=None,
                    help="build ONE arm from calib_100 minus these clip indices")
    ap.add_argument("--tag", type=str, default=None, help="name for the --drop arm")
    ap.add_argument("--reserve-gb", type=float, default=26.0)
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()

    per = dict(np.load(REPO / "outputs" / PERCLIP / "importance_perclip.npz"))
    n_clips = per["coc_vlm_q"].shape[0]
    man = json.loads((REPO / "outputs" / PERCLIP / "config.json").read_text())["clip_ids"]
    assert len(man) == n_clips, (len(man), n_clips)

    if args.design:
        subsets = make_design(n_clips, args.size, args.m, args.seed)
        DESIGN.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pool": PERCLIP, "n_clips": n_clips, "clip_ids": man,
            "size": args.size, "m": args.m, "seed": args.seed,
            "config": CONFIG, "subsets": subsets,
            "sha": hashlib.sha256(json.dumps(subsets).encode()).hexdigest()[:16],
        }
        DESIGN.write_text(json.dumps(payload, indent=1))
        cnt = np.zeros(n_clips, dtype=int)
        for s in subsets:
            cnt[s] += 1
        print(f"{args.m} subsets of {args.size}/{n_clips}, seed {args.seed}, "
              f"sha {payload['sha']}")
        print(f"per-clip inclusions: min {cnt.min()} max {cnt.max()} mean {cnt.mean():.1f}")
        print(f"-> {DESIGN}")
        return

    if args.dump_importance is not None:
        d = json.loads(DESIGN.read_text())
        subset = d["subsets"][args.dump_importance]
        out = REPO / "outputs" / (args.out or f"imp_sub{args.dump_importance:02d}")
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "importance.npz", **subset_importance(per, subset))
        (out / "config.json").write_text(json.dumps(
            {"source": PERCLIP, "subset": subset, "n": len(subset)}, indent=1))
        print(f"wrote {out}/importance.npz ({len(subset)} clips)")
        return

    model = load_model(args.gpu, args.reserve_gb)
    full_total = sl.n_params(model)
    print(f"full model {full_total:,} params", flush=True)

    if args.verify:
        # V0a: the all-100 subset IS the shipped criterion
        meta = build_meta(model, subset_importance(per, None), "all100",
                          list(range(n_clips)), full_total)
        shipped = json.loads(
            (REPO / "outputs" / f"slim_{CONFIG}" / "slim_meta.json").read_text())
        bad = same_selection(meta, shipped)
        print(f"V0a all-100 vs shipped slim_{CONFIG}: "
              f"{'FAIL ' + bad if bad else 'identical'}")
        print(f"    removed {meta['params']['removed']:,} vs "
              f"{shipped['params']['removed']:,}")
        # V0b: one subset, batch path vs make_slim.py's own output
        ref = REPO / "outputs" / "slim_subinf_v0ref"
        if (ref / "slim_meta.json").exists():
            d = json.loads(DESIGN.read_text())
            sub = build_meta(model, subset_importance(per, d["subsets"][0]), 0,
                             d["subsets"][0], full_total)
            bad = same_selection(sub, json.loads((ref / "slim_meta.json").read_text()))
            print(f"V0b arm 0 batch vs make_slim.py: "
                  f"{'FAIL ' + bad if bad else 'identical'}")
        else:
            print(f"V0b skipped: build {ref} with make_slim.py first "
                  f"(--dump-importance 0 --out imp_sub00, then make_slim --importance "
                  f"imp_sub00 --out outputs/slim_subinf_v0ref --no-state)")
        return

    if args.drop is not None:
        assert args.tag, "--drop needs --tag"
        drop = set(args.drop)
        jobs = [(args.tag, [i for i in range(n_clips) if i not in drop])]
    else:
        d = json.loads(DESIGN.read_text())
        lo, hi = args.arms if args.arms else (0, len(d["subsets"]))
        jobs = [(f"{i:02d}", d["subsets"][i]) for i in range(lo, hi)]

    for arm, subset in jobs:
        out = REPO / "outputs" / f"{args.prefix}_{arm}"
        if (out / "slim_meta.json").exists():
            print(f"skip {out.name} (exists)", flush=True)
            continue
        t0 = time.time()
        meta = build_meta(model, subset_importance(per, subset), arm, subset, full_total)
        out.mkdir(parents=True, exist_ok=True)
        (out / "slim_meta.json").write_text(json.dumps(meta))
        q, mlp = kept_counts(meta)
        print(f"{out.name}: n={len(subset)} kept q={q} mlp={mlp} "
              f"removed={meta['params']['removed']:,} ({time.time() - t0:.0f}s)",
              flush=True)


if __name__ == "__main__":
    main()
