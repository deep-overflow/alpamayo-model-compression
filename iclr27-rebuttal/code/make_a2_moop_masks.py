"""VLM keep-masks for Alpamayo 2 Super under the ICLR27 MoOP recipe (with and without the layer-0 exemption) and the paper's
single-criterion rows FM-only (`traj_u40_v2`) and CE-only (`coc_u40_v2`), added 2026-11-09 at the user's request.

Plan: iclr27-rebuttal/plans/2026-11-09_a2-moop-test500.md (approved 2026-11-09: D1 = same per-layer ratio, D3 = l0keep arm too).

The selection is make_slim.py's `dual_u40_v2` / `dual_u40_l0keep_v2` path with the model's sizes swapped in:
  score  = max(rank_norm(I_traj), rank_norm(I_CoC)) per layer (run_cocsafe.rank_norm; the last layer's all-zero I_traj
           ranks in index order, as in the shipped A1.5 arm)
  cut    = mask_lib.select_mask_ratios with ratio 0.3985632694 for Q heads and MLP channels in every layer
  l0keep = layer 0 keeps all Q heads; their parameters are paid back in MLP channels, floored, spread evenly over the
           layers (first layers +1) and taken from each layer's own ranking (make_slim.py l0keep block)
Expert, KV groups, vision tower and embeddings are untouched.

G0 regression: the same function on A1.5's importance_v2 with A1.5's sizes must reproduce the shipped
outputs/slim_dual_u40_v2 and outputs/slim_dual_u40_l0keep_v2 kept sets unit for unit.

Writes outputs/iclr27-rebuttal/a2_moop/masks_{moop,moop_l0keep}.npz (q_mask (64, 64), mlp_mask (64, 25600); 1 = keep) and
masks_config.json.

--pct N (added 2026-11-09, plans/2026-11-09_a2-u50-test500.md): any N other than 40 uses make_slim's literal ratio N/100 for
moop / fm_only / ce_only (no l0keep) and writes masks_<arm>_u<N>.npz + masks_config_u<N>.json. G0' then also checks the
literal-ratio path: ratio 0.20 / 0.30 on A1.5 must reproduce outputs/slim_dual_u20_v2 / slim_dual_u30_v2 unit for unit.

Usage: .venv/bin/python iclr27-rebuttal/code/make_a2_moop_masks.py [--pct 50]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))
from mask_lib import select_mask_ratios
from run_cocsafe import rank_norm

RATIO = 0.3985632694  # make_slim u40: the matched target, never 0.40
A2_IMP = Path("/home/cvlab21/project/chan/alpamayo2-distill/outputs/a2_importance_v1/importance.npz")
A2 = {"layers": 64, "heads": 64, "inter": 25600, "hidden": 5120, "head_dim": 128}
A15 = {"layers": 36, "heads": 32, "inter": 12288, "hidden": 4096, "head_dim": 128}


def scores(imp, crit):
    """make_slim.py stems: dual = max of the two within-layer ranks; traj / coc = one rank alone (FM-only / CE-only;
    the last layer's all-zero I_traj ranks in index order, as in the shipped traj_u40_v2)."""
    if crit == "dual":
        return (np.maximum(rank_norm(imp["traj_vlm_q"]), rank_norm(imp["coc_vlm_q"])),   # (L, H)
                np.maximum(rank_norm(imp["traj_vlm_mlp"]), rank_norm(imp["coc_vlm_mlp"])))  # (L, I)
    return rank_norm(imp[f"{crit}_vlm_q"]), rank_norm(imp[f"{crit}_vlm_mlp"])


def moop_masks(imp, dims, l0keep, crit="dual", ratio=RATIO):
    """imp: npz-like with traj/coc_vlm_{q,mlp}; returns q_mask (L, H), mlp_mask (L, I), info dict."""
    L, H, I = dims["layers"], dims["heads"], dims["inter"]
    sq, sm = scores(imp, crit)
    rq, rm = np.full(L, ratio), np.full(L, ratio)
    info = {}
    if l0keep:
        attn_cost = 2 * dims["head_dim"] * dims["hidden"]
        mlp_cost = 3 * dims["hidden"]
        n_q0 = round(float(rq[0]) * H)
        n_extra, short = divmod(n_q0 * attn_cost, mlp_cost)
        extra = np.full(L, n_extra // L)
        extra[: n_extra - extra.sum()] += 1
        rq[0] = 0.0
        rm = (np.round(rm * I) + extra) / I
        assert (rm < 1).all()
        info = {"layer0_heads_restored": n_q0, "extra_mlp_channels": int(n_extra),
                "extra_per_layer_max_min": [int(extra.max()), int(extra.min())], "params_short": int(short)}
    q = select_mask_ratios(sq, rq)
    m = select_mask_ratios(sm, rm)
    return q, m, info


def removed_params(q, m, dims):
    attn_cost = 2 * dims["head_dim"] * dims["hidden"]
    mlp_cost = 3 * dims["hidden"]
    return int((q == 0).sum() * attn_cost + (m == 0).sum() * mlp_cost)


def g0_regression(literal=False):
    """Same code on A1.5 importance_v2 vs the shipped kept sets; literal=True adds G0' (make_slim's u20 / u30 = 0.20 / 0.30)."""
    imp = np.load(REPO / "outputs" / "importance_v2" / "importance.npz")
    out = {}
    refs = [("dual", False, "slim_dual_u40_v2", RATIO), ("dual", True, "slim_dual_u40_l0keep_v2", RATIO),
            ("traj", False, "slim_traj_u40_v2", RATIO), ("coc", False, "slim_coc_u40_v2", RATIO)]
    if literal:
        refs += [("dual", False, "slim_dual_u20_v2", 0.20), ("dual", False, "slim_dual_u30_v2", 0.30)]
    for crit, l0keep, ref, ratio in refs:
        q, m, _ = moop_masks(imp, A15, l0keep, crit, ratio)
        meta = json.loads((REPO / "outputs" / ref / "slim_meta.json").read_text())
        same_q = all(set(np.flatnonzero(q[li])) == set(meta["vlm"][li]["q"]) for li in range(A15["layers"]))
        same_m = all(set(np.flatnonzero(m[li])) == set(meta["vlm"][li]["mlp"]) for li in range(A15["layers"]))
        same_p = removed_params(q, m, A15) == meta["params"]["removed"]
        out[ref] = {"q_identical": bool(same_q), "mlp_identical": bool(same_m), "removed_matches": bool(same_p)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pct", type=int, default=40, help="40 = the matched u40 (0.3985632694); else the literal N/100")
    args = ap.parse_args()
    ratio = RATIO if args.pct == 40 else args.pct / 100
    sfx = "" if args.pct == 40 else f"_u{args.pct}"
    arms = ((("moop", "dual", False), ("moop_l0keep", "dual", True), ("fm_only", "traj", False), ("ce_only", "coc", False))
            if args.pct == 40 else (("moop", "dual", False), ("fm_only", "traj", False), ("ce_only", "coc", False)))
    out = REPO / "outputs" / "iclr27-rebuttal" / "a2_moop"
    out.mkdir(parents=True, exist_ok=True)
    g0 = g0_regression(literal=args.pct != 40)
    print("G0 regression on A1.5 (same code -> shipped kept sets):", json.dumps(g0))
    assert all(all(v.values()) for v in g0.values()), "G0 FAILED: the port does not reproduce A1.5's shipped masks"

    imp = np.load(A2_IMP)
    total = 35_810_000_000  # A2 Super, safetensors header (alpamayo2-distill plan 2026-10-15)
    cfg = {"pct": args.pct, "ratio": ratio, "importance": str(A2_IMP), "dims": A2, "g0_regression": g0, "arms": {}}
    for name, crit, l0keep in arms:
        name = f"{name}{sfx}"
        q, m, info = moop_masks(imp, A2, l0keep, crit, ratio)
        # consistency: in every layer the kept units outrank the cut ones under the score used
        sq, sm = scores(imp, crit)
        for s, mk in ((sq, q), (sm, m)):
            for li in range(A2["layers"]):
                if (mk[li] == 0).any():
                    assert s[li][mk[li] == 1].min() >= s[li][mk[li] == 0].max()
        rem = removed_params(q, m, A2)
        q_cut, m_cut = (q == 0).sum(1), (m == 0).sum(1)
        cfg["arms"][name] = {
            "criterion": crit, "l0keep": l0keep, **info, "removed_params": rem,
            "removed_share_of_35.81B": round(rem / total, 4),
            "q_cut_per_layer": sorted(set(q_cut.tolist())), "mlp_cut_per_layer": sorted(set(m_cut.tolist())),
            "layer0_q_kept": np.flatnonzero(q[0]).tolist()}
        path = out / f"masks_{name}.npz"
        if path.exists():  # a running job may hold it: never rewrite, only confirm it is the same selection
            z = np.load(path)
            assert (z["q_mask"] == q).all() and (z["mlp_mask"] == m).all(), f"{path} differs from a rebuild"
        else:
            np.savez(path, q_mask=q.astype(np.float32), mlp_mask=m.astype(np.float32))
        print(f"{name}: Q cut/layer {sorted(set(q_cut.tolist()))} (layer 0: {int(q_cut[0])}), "
              f"MLP cut/layer {sorted(set(m_cut.tolist()))}, removed {rem:,} = {rem / total:.1%} of 35.81B "
              f"{info if info else ''}")
    qa = np.load(out / f"masks_moop{sfx}.npz")["q_mask"]
    if args.pct == 40:
        qb = np.load(out / "masks_moop_l0keep.npz")["q_mask"]
        assert (qa[1:] == qb[1:]).all(), "Q masks of layers 1-63 must match between the two arms"
    cfg["layer0_q_cut_moop"] = np.flatnonzero(qa[0] == 0).tolist()
    (out / f"masks_config{sfx}.json").write_text(json.dumps(cfg, indent=2))
    print(f"layer-0 Q heads cut by moop{sfx}: {cfg['layer0_q_cut_moop']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
