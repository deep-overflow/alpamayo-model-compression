"""Build --no-state slim recipes (slim_meta.json) on CPU, for when every GPU is busy.

A --no-state recipe is only the kept-unit indices and parameter counts; evaluation rebuilds the model from the base weights
on the GPU (slim_lib.load_slim). So the selection and the surgery bookkeeping need no GPU. This calls make_slim.py's own
build_masks / expected_removed and slim_lib's apply_surgery / check_slim / save_slim on a CPU copy of the base model, with
make_slim's command-line defaults and --importance importance_v2. The only step left out is make_slim's one-clip GPU smoke
test; the evaluation that follows exercises the same model. Each config gets its own deep copy of the base model, because
the surgery slices weights in place.

Used for iclr27-rebuttal/plans/2026-11-08_comb-rules-pre-l0keep-test500.md (amended 2026-11-12): the three pre-l0keep rule
builds, checked afterwards by check_comb_pre_l0keep_builds.py (B0: dualsum rebuilt unit for unit as the September build).
Usage (repo root):
  .venv/bin/python iclr27-rebuttal/code/build_recipes_cpu.py \
      dualsum_u40_v2:outputs/iclr27-rebuttal/comb_pre_l0keep/slim_dualsum_u40_v2 rawabsavg_u40_v2:... rawabsmean_u40_v2:...
"""

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))
import make_slim as ms
import slim_lib as sl

IMPORTANCE = "importance_v2"


def main():
    jobs = [a.split(":", 1) for a in sys.argv[1:]]
    assert jobs and all(len(j) == 2 for j in jobs), "usage: config:out_dir ..."
    torch.set_num_threads(16)
    t0 = time.time()
    # flash_attention_2 cannot be set up without CUDA; make_slim switches to sdpa right after loading anyway, and the
    # surgery only slices weights
    base = ms.Alpamayo1_5.from_pretrained("nvidia/Alpamayo-1.5-10B", revision=ms.MODEL_REV, dtype=torch.bfloat16,
                                          attn_implementation="sdpa")
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    print(f"base model on CPU in {time.time() - t0:.0f}s", flush=True)
    imp = dict(np.load(REPO / "outputs" / IMPORTANCE / "importance.npz"))
    for cfg, out in jobs:
        out_dir = REPO / out
        if (out_dir / "slim_meta.json").exists():
            print(f"skip {cfg}: {out_dir}/slim_meta.json exists", flush=True)
            continue
        model = copy.deepcopy(base)
        vq, vm, eq, em, kvonly = ms.build_masks(cfg, imp, model, imp_run=IMPORTANCE)
        full_total = sl.n_params(model)
        meta = sl.apply_surgery(model, vq, vm, eq, em, kvonly_layers=kvonly)
        meta["config"] = cfg
        meta["importance_from"] = IMPORTANCE
        slim_total = sl.n_params(model)
        removed = ms.expected_removed(model, vq, vm, eq, em, kvonly)
        assert slim_total == full_total - removed, (slim_total, full_total - removed)
        sl.check_slim(model)
        meta["params"] = {"full": full_total, "slim": slim_total, "removed": removed}
        sl.save_slim(model, meta, out_dir, write_state=False)
        (out_dir / "config.json").write_text(json.dumps({
            "model": "nvidia/Alpamayo-1.5-10B", "model_revision": ms.MODEL_REV, "config": cfg,
            "importance_from": IMPORTANCE, "stepvlm_from": "importance_stepvlm_v1", "params": meta["params"],
            "kvonly_layers": list(kvonly),
            "kept_q_per_layer": {"vlm": [len(m["q"]) for m in meta["vlm"]],
                                 "expert": [len(m["q"]) for m in meta["expert"]]},
            "built_with": "iclr27-rebuttal/code/build_recipes_cpu.py (CPU, no smoke test)", "gpu": None}, indent=2))
        (out_dir / "summary.txt").write_text(
            f"slim recipe {cfg} (CPU build, no smoke test)\n"
            f"params {full_total:,} -> {slim_total:,} (-{removed:,}, {removed / full_total * 100:.1f}%)\n")
        print(f"{cfg}: {full_total:,} -> {slim_total:,} (-{removed:,}, {removed / full_total * 100:.1f}%) -> {out_dir}",
              flush=True)
        del model


if __name__ == "__main__":
    main()
