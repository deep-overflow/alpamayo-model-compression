"""Pre-build verification for the VLM axis arms (gate V0, no GPU, no model load).

make_slim's `expected_removed` derives per-unit parameter costs from config fields; this
checks those costs against the shipped weights themselves (safetensors headers) and then
against a checkpoint whose removed count was already verified by a real build
(`slim_dual_u40_v2`). The axis split is only meaningful if

    dualq_u40_v2  U  dualm_u40_v2  ==  dual_u40_v2      (bit-identical kept sets)
    removed(dualq) + removed(dualm) == removed(dual_u40_v2)

both hold exactly, which is what makes the existing dual_u40_v2 runs a free additivity arm.

Usage:
  .venv/bin/python experiments/head_analysis/check_vlm_axis.py
"""

import glob
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import mask_lib as ml  # noqa: E402
from run_cocsafe import rank_norm  # noqa: E402
from run_grid import allocations  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
REV = "7aba8293c09993f2e125c6819df05d7fa3e873ea"


def weight_shapes(rev=REV):
    """{tensor name: shape} from the safetensors headers -- no weights are read."""
    import huggingface_hub.constants as hc

    snap = Path(hc.HF_HUB_CACHE) / f"models--nvidia--Alpamayo-1.5-10B/snapshots/{rev}"
    shapes = {}
    for p in sorted(glob.glob(str(snap / "model-*.safetensors"))):
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            head = json.loads(f.read(n))
        for k, v in head.items():
            if k != "__metadata__":
                shapes[k] = tuple(v["shape"])
    return shapes


def unit_costs(shapes, prefix):
    """(params per Q head, params per MLP channel, n_heads, intermediate) from the weights.

    A Q head owns its q_proj row block and its o_proj column block. q_norm/k_norm are
    (head_dim,) shared across heads, so removing a head frees none of them -- which is why
    make_slim's expected_removed adds no norm term outside kv-only layers.
    """
    q = shapes[f"{prefix}self_attn.q_proj.weight"]  # (H*D, hidden)
    o = shapes[f"{prefix}self_attn.o_proj.weight"]  # (hidden, H*D)
    g = shapes[f"{prefix}mlp.gate_proj.weight"]  # (I, hidden)
    u = shapes[f"{prefix}mlp.up_proj.weight"]  # (I, hidden)
    d = shapes[f"{prefix}mlp.down_proj.weight"]  # (hidden, I)
    head_dim = shapes[f"{prefix}self_attn.q_norm.weight"][0]
    hidden = q[1]
    assert o == (hidden, q[0]) and g == u == (d[1], hidden) and d[0] == hidden
    n_heads = q[0] // head_dim
    p_head = head_dim * hidden + hidden * head_dim  # q rows + o cols
    p_chan = hidden + hidden + hidden  # gate row + up row + down col
    return p_head, p_chan, n_heads, d[1], hidden, head_dim


def main():
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    shapes = weight_shapes()
    total = sum(int(np.prod(s)) for s in shapes.values())
    say(f"weights: {len(shapes)} tensors, {total:,} params (rev {REV[:7]})")

    towers = {}
    for name, pre in (("vlm", "vlm.model.language_model.layers.0."),
                      ("expert", "expert.layers.0.")):
        p_head, p_chan, n_heads, inter, hidden, head_dim = unit_costs(shapes, pre)
        n_layers = 1 + max(int(k.split(".")[-4]) for k in shapes
                           if k.startswith(pre[: pre.rindex("0.")]) and k.endswith("q_proj.weight"))
        towers[name] = {"p_head": p_head, "p_chan": p_chan, "n_heads": n_heads,
                        "inter": inter, "hidden": hidden, "head_dim": head_dim,
                        "n_layers": n_layers}
        say(f"{name:6s} hidden {hidden:5d}  heads {n_heads:2d} x {head_dim}  inter {inter:5d}  "
            f"layers {n_layers}  |  per head {p_head:9,}  per channel {p_chan:6,}  "
            f"ratio {p_head / p_chan:.1f}x")
        say(f"       axis totals over {n_layers} layers: Q {n_layers * n_heads * p_head:,}  "
            f"MLP {n_layers * inter * p_chan:,}")
    say()

    v = towers["vlm"]
    imp = dict(np.load(REPO / "outputs" / "importance_v2" / "importance.npz"))
    ref_meta = json.loads((REPO / "outputs" / "slim_integrated_mag" / "slim_meta.json").read_text())
    allocs, info = allocations(imp, ref_meta, v["n_layers"], v["n_heads"], v["inter"], 0.5)
    rq, rm = allocs["uniform"]
    say(f"uniform ratio = {rq[0]:.10f}  (u40 family's matched budget, not 0.40)")
    say(f"  -> per layer: Q cut {round(rq[0] * v['n_heads'])}/{v['n_heads']}, "
        f"MLP cut {round(rm[0] * v['inter'])}/{v['inter']}")

    sq = np.maximum(rank_norm(imp["traj_vlm_q"]), rank_norm(imp["coc_vlm_q"]))
    sm = np.maximum(rank_norm(imp["traj_vlm_mlp"]), rank_norm(imp["coc_vlm_mlp"]))
    vq = ml.select_mask_ratios(sq, rq)  # (36, 32)
    vm = ml.select_mask_ratios(sm, rm)  # (36, 12288)

    # the param-matched MLP arm: as many channels as dualq's heads cost, per layer
    cut_q = int((vq[0] == 0).sum())
    c_pm = round(cut_q * v["p_head"] / v["p_chan"])
    vm_pm = ml.select_mask_ratios(sm, np.full(v["n_layers"], c_pm / v["inter"]))
    say(f"  -> param-matched MLP arm: {cut_q} heads x {v['p_head']:,} / {v['p_chan']:,} "
        f"= {cut_q * v['p_head'] / v['p_chan']:.2f} -> {c_pm} channels/layer")
    say()

    dual_meta = json.loads((REPO / "outputs" / "slim_dual_u40_v2" / "slim_meta.json").read_text())
    ok = True

    # V0.1 union == dual_u40_v2, kept index for kept index
    for li, layer in enumerate(dual_meta["vlm"]):
        kq = np.flatnonzero(vq[li])
        km = np.flatnonzero(vm[li])
        if list(kq) != layer["q"] or list(km) != layer["mlp"]:
            ok = False
            say(f"MISMATCH layer {li}: q {len(kq)} vs {len(layer['q'])}, "
                f"mlp {len(km)} vs {len(layer['mlp'])}")
    say(f"V0.1 union(dualq, dualm) == slim_dual_u40_v2 kept sets, all {v['n_layers']} layers: "
        f"{'PASS' if ok else 'FAIL'}")
    for li, layer in enumerate(dual_meta["expert"]):
        if len(layer["q"]) != towers["expert"]["n_heads"] or \
                len(layer["mlp"]) != towers["expert"]["inter"]:
            ok = False
    say(f"V0.2 dual_u40_v2 leaves the expert whole (16 heads / 8256 ch every layer), "
        f"kv-only layers {len(dual_meta['kvonly_layers'])}: "
        f"{'PASS' if ok else 'FAIL'}")

    # V0.3 removed-parameter arithmetic against the built checkpoint
    r_q = int((vq == 0).sum()) * v["p_head"]
    r_m = int((vm == 0).sum()) * v["p_chan"]
    r_pm = int((vm_pm == 0).sum()) * v["p_chan"]
    ref_removed = dual_meta["params"]["removed"]
    say()
    say(f"{'arm':16s} {'cut/layer':>12s} {'removed':>16s} {'of full':>9s} {'of dual':>9s}")
    for name, cut, rem in (("dualq_u40_v2", f"{cut_q}/{v['n_heads']} heads", r_q),
                           ("dualm_u40_v2", f"{int((vm[0] == 0).sum())}/{v['inter']} ch", r_m),
                           (f"dualm_c{c_pm}", f"{c_pm}/{v['inter']} ch", r_pm),
                           ("dual_u40_v2 (both)", "13 + 4898", r_q + r_m)):
        say(f"{name:16s} {cut:>12s} {rem:16,} {rem / total:8.2%} {rem / ref_removed:8.2%}")
    say(f"checkpoint slim_dual_u40_v2: full {dual_meta['params']['full']:,} "
        f"slim {dual_meta['params']['slim']:,} removed {ref_removed:,}")
    add_ok = (r_q + r_m == ref_removed)
    say(f"V0.3 removed(dualq) + removed(dualm) == removed(dual_u40_v2): "
        f"{'PASS' if add_ok else 'FAIL'} ({r_q:,} + {r_m:,} = {r_q + r_m:,})")
    full_ok = (total == dual_meta["params"]["full"])
    say(f"V0.4 safetensors total == slim_meta full: {'PASS' if full_ok else 'FAIL'}")
    say(f"V0.5 param match dualq vs dualm_c{c_pm}: {r_q - r_pm:+,} "
        f"({(r_pm - r_q) / r_q:+.3%} of dualq)")
    ok = ok and add_ok and full_ok

    out = REPO / "outputs" / "vlm_axis_check"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    json.dump({"towers": towers, "uniform_ratio": float(rq[0]), "cut_q": cut_q,
               "cut_m": int((vm[0] == 0).sum()), "cut_m_pm": c_pm,
               "removed": {"dualq_u40_v2": r_q, "dualm_u40_v2": r_m, f"dualm_c{c_pm}": r_pm,
                           "dual_u40_v2": ref_removed},
               "total_params": total, "alloc_info": info, "pass": bool(ok)},
              (out / "metrics.json").open("w"), indent=1)
    say(f"\nwrote {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
