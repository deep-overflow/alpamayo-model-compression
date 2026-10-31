"""Look for blown-up OSSCAR solves in a tyr_off supernet (CPU only).

For every module and level, ||W_level||_F against the module's dense weight (base checkpoint). A
reconstruction on a near-singular Hessian (damping 1e-2 x mean diag, the official value) can inflate
the kept weights; such a level shows up as a norm ratio far above its neighbours.

  .venv/bin/python cvpr27/code/tyr_off/check_level_norms.py cvpr27/tyr_off/supernet_r2 [search config]
"""

import json
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))
from slim_lib import MODEL_REV


def dense_weights(names):
    snap = Path(snapshot_download("nvidia/Alpamayo-1.5-10B", revision=MODEL_REV, local_files_only=True))
    index = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    out = {}
    for n in names:
        _, i, rest = n.split(".", 2)                       # supernet names zero-pad the layer index
        key = next(k for k in index if k.endswith(f"language_model.layers.{int(i)}.{rest}.weight"))
        with safe_open(snap / index[key], "pt") as f:
            out[n] = f.get_tensor(key).float()
    return out


def main():
    sup = REPO / "outputs" / sys.argv[1]
    cfg = json.loads((REPO / "outputs" / sys.argv[2]).read_text()) if len(sys.argv) > 2 else None
    meta = json.loads((sup / "metadata.json").read_text())
    dense = dense_weights(meta["layer_names"])
    rows = []
    for n in meta["layer_names"]:
        dn = float(dense[n].norm())
        for lv in sorted(int(k) for k in meta["levels"][n]):
            w = torch.load(sup / n / f"{lv}.pth").float()
            rows.append((n, lv, meta["levels"][n][str(lv)]["keep"], float(w.norm()) / dn))
    worst = sorted(rows, key=lambda r: -r[3])[:15]
    print("largest ||W_level|| / ||W_dense||:")
    for n, lv, k, r in worst:
        print(f"  {n:28s} level {lv:+d} keep {k:6d}  ratio {r:8.2f}")
    n_big = sum(r[3] > 2.0 for r in rows)
    print(f"{n_big} of {len(rows)} (module, level) weights have norm ratio > 2")
    for name, lvls in (("centre (level 0)", {n: 0 for n in meta["layer_names"]}),
                       ("search config", cfg)):
        if lvls is None:
            continue
        sel = [r for r in rows if lvls.get(r[0]) == r[1]]
        top = sorted(sel, key=lambda r: -r[3])[:5]
        print(f"{name}: max ratio {top[0][3]:.2f}; top {[(r[0], r[1], round(r[3], 2)) for r in top]}")


if __name__ == "__main__":
    main()
