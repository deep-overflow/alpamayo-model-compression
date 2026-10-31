"""G0 (cvpr27/plans/2026-10-31_tyr-a100.md): the port's per-level OSSCAR solve equals the official one.

Runs the same random layer + Hessian through Tyr-the-Pruner's LocalPruner.prune (official, src/local_pruner.py)
and through experiments/head_analysis/tyr_lib.prune_levels, for o_proj-like head groups (1 removal round) and
down_proj-like channels (16 rounds), at sparsities that include the 0.0 and 1.0 branches. CPU, fp32.

  .venv/bin/python cvpr27/code/tyr_off/check_osscar_vs_official.py
"""

import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
OFFICIAL = Path("/home/cvlab21/project/chan/Tyr-the-Pruner")
sys.path.insert(0, str(OFFICIAL))
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))

import tyr_lib as tyr
from src.local_pruner import LocalPruner

torch.backends.cuda.matmul.allow_tf32 = False


def case(name, d_in, d_out, n_groups, upd, sparsities, n_tok=4096, seed=0):
    g = torch.Generator().manual_seed(seed)
    layer = torch.nn.Linear(d_in, d_out, bias=False)
    layer.weight.data = torch.randn(d_out, d_in, generator=g) / d_in ** 0.5
    # correlated inputs with a few dead channels, like real activations
    x = torch.randn(n_tok, d_in, generator=g) @ (torch.randn(d_in, d_in, generator=g) / d_in ** 0.5)
    x[:, :3] = 0
    # the official step() picks head grouping only for names containing "self_attn.o_proj"
    off = LocalPruner(layer, layername="self_attn.o_proj" if upd == 1 else "mlp.down_proj",
                      num_heads=n_groups)
    off.update(x)
    H = off.H.clone()
    ref = off.prune(sparsities, expected_sparsity=0, error_accumulation=False)  # list of (out, in)
    keeps = [int(n_groups * (1 - s)) for s in sparsities]
    port = tyr.prune_levels(layer, H, n_groups, keeps, update_iter=upd, damp=1e-2)
    rows = []
    for s, k, w_ref in zip(sparsities, keeps, ref):
        w = port[k].to(w_ref.dtype)
        rows.append({"sparsity": s, "keep": k, "max_abs_diff": float((w - w_ref).abs().max()),
                     "bitwise": bool(torch.equal(w, w_ref))})
    return {"case": name, "levels": rows}


def main():
    out = [
        case("heads (group 64 x 8, 1 round)", 512, 256, 8, 1, [0.0, 0.125, 0.375, 0.625, 1.0]),
        case("channels (group 1, 16 rounds)", 384, 128, 384, 16, [0.0, 0.1, 0.39856, 0.6, 1.0]),
    ]
    ok = all(r["bitwise"] for c in out for r in c["levels"])
    print(json.dumps(out, indent=1))
    print("G0", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
