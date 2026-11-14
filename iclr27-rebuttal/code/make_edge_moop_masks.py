"""VLM keep-masks for Alpamayo 2 Edge: MoOP / FM-only / CE-only (plans/2026-11-14_a2-edge-moop-u40.md, step 1).

The selection is make_a2_moop_masks.moop_masks, imported unchanged (make_slim's dual / traj / coc path):
  MoOP    = max(rank_norm(I_FM), rank_norm(I_CE)) within each layer
  FM-only = rank_norm(I_FM), CE-only = rank_norm(I_CE)
  cut     = the same ratio in every layer: u40 = make_slim's matched 0.3985632694, any other --pct = the literal N/100
Importance: outputs/iclr27-rebuttal/edge_importance_v1 (calib_100, no-nav prompt). Edge at u40: 6 of 16 Q heads and 3,673 of
9,216 MLP channels per layer. Expert, KV heads, vision tower and embeddings are untouched.

Parameter bookkeeping is Edge's own: a Q head costs 2 x head_dim x hidden (q_proj rows + o_proj columns), an MLP channel
2 x hidden (relu^2 MLP: fc1 row + fc2 column, no gate). Total parameters from the checkpoint (2,651,163,378).

Gates: G0 = make_a2_moop_masks.g0_regression (the imported function on A1.5 importance_v2 reproduces the shipped dual / dual_l0keep /
traj / coc u40 kept sets unit for unit); G0e = every Edge mask removes exactly round(ratio x units) units in every layer.

Writes outputs/iclr27-rebuttal/edge_moop/masks_{moop,fm_only,ce_only}[_u<N>].npz (q_mask (28, 16), mlp_mask (28, 9216); 1 = keep)
and masks_config[_u<N>].json.

--l0keep (2026-11-14, plans/2026-11-14_a2-edge-l0keep-u40.md): the make_slim l0keep rule for all three criteria -- layer 0 keeps
its 16 Q heads, their 6 x 524,288 parameters are paid back as 768 extra MLP channels (Edge cost 2 x hidden; A2's code assumes
3 x hidden), spread 28 / 27 per layer. G0l: the parameterised rule with 3 x hidden reproduces make_a2_moop_masks' l0keep masks on
A1.5 unit for unit; G0p: each l0keep mask removes exactly the non-l0keep parameter count. Writes masks_<arm>_l0keep[_u<N>].npz.

--q-cut N [N ...] (2026-11-15, plans/2026-11-15_a2-edge-qcut-u40.md): make_slim's _qcut rule at the --pct budget with Edge's MLP
cost -- N Q heads cut per layer, the rest of the per-layer budget in MLP channels (one Edge Q head = 128 channels, so u40-qN cuts
3,673 + (6 - N) x 128 channels). G0q: the parameterised rule with 3 x hidden reproduces make_a2_moop_masks' q-cut 4 on A1.5;
G0p: exact parameter parity with the base allocation; G0n: the q-cut sets nest with it. Writes masks_<arm>_q<N>.npz.

Usage: .venv/bin/python iclr27-rebuttal/code/make_edge_moop_masks.py [--pct 50] [--l0keep | --q-cut 2 4]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import make_a2_moop_masks as mam

REPO = Path(__file__).resolve().parents[2]
EDGE_IMP = REPO / "outputs" / "iclr27-rebuttal" / "edge_importance_v1" / "importance.npz"
EDGE = {"layers": 28, "heads": 16, "inter": 9216, "hidden": 2048, "head_dim": 128}
TOTAL = 2_651_163_378  # Alpamayo2-Edge model card / checkpoint
TEXT_LAYERS = 28 * (2048 * 2048 * 2 + 2048 * 1024 * 2 + 2 * 2048 * 9216 + 2 * 2048)  # q, o, k, v, fc1, fc2, 2 norms


def edge_removed(q, m):
    return int((q == 0).sum() * 2 * EDGE["head_dim"] * EDGE["hidden"] + (m == 0).sum() * 2 * EDGE["hidden"])


def l0keep_masks(imp, dims, crit, ratio, mlp_cost):
    """make_a2_moop_masks.moop_masks(l0keep=True) with the MLP channel cost as a parameter (A1.5/A2 SwiGLU = 3 x hidden,
    Edge relu^2 = 2 x hidden): layer 0 keeps all Q heads, their parameters are paid back in MLP channels, floored, spread
    evenly over the layers (first layers +1) and taken from each layer's own ranking. Returns q (L, H), m (L, I), info."""
    L, H, I = dims["layers"], dims["heads"], dims["inter"]
    sq, sm = mam.scores(imp, crit)
    rq, rm = np.full(L, ratio), np.full(L, ratio)
    attn_cost = 2 * dims["head_dim"] * dims["hidden"]
    n_q0 = round(float(rq[0]) * H)
    n_extra, short = divmod(n_q0 * attn_cost, mlp_cost)
    extra = np.full(L, n_extra // L)
    extra[: n_extra - extra.sum()] += 1
    rq[0] = 0.0
    rm = (np.round(rm * I) + extra) / I
    assert (rm < 1).all()
    info = {"layer0_heads_restored": n_q0, "extra_mlp_channels": int(n_extra),
            "extra_per_layer_max_min": [int(extra.max()), int(extra.min())], "params_short": int(short),
            "extra_per_layer": [int(x) for x in extra]}
    return mam.select_mask_ratios(sq, rq), mam.select_mask_ratios(sm, rm), info


def make_l0keep(imp, ratio, sfx, out, g0):
    """--l0keep: masks_{moop,fm_only,ce_only}_l0keep[_u<N>].npz with Edge's MLP cost (2 x hidden)."""
    imp15 = np.load(REPO / "outputs" / "importance_v2" / "importance.npz")
    g0l = {crit: bool(all(np.array_equal(a, b) for a, b in zip(
        l0keep_masks(imp15, mam.A15, crit, mam.RATIO, 3 * mam.A15["hidden"])[:2],
        mam.moop_masks(imp15, mam.A15, True, crit, mam.RATIO)[:2]))) for crit in ("dual", "traj", "coc")}
    print("G0l (parameterised l0keep == make_a2_moop_masks l0keep on A1.5):", g0l)
    assert all(g0l.values()), "G0l FAILED"
    kq, km = round(ratio * EDGE["heads"]), round(ratio * EDGE["inter"])
    cfg = {"pct": args_pct(ratio), "ratio": ratio, "l0keep": True, "importance": str(EDGE_IMP), "dims": EDGE,
           "g0_regression": g0, "g0l_parameterised_l0keep": g0l, "mlp_cost": 2 * EDGE["hidden"], "total_params": TOTAL,
           "text_decoder_params": TEXT_LAYERS, "arms": {}}
    masks, base_removed = {}, None
    for name, crit in (("moop", "dual"), ("fm_only", "traj"), ("ce_only", "coc")):
        q, m, info = l0keep_masks(imp, EDGE, crit, ratio, 2 * EDGE["hidden"])
        extra = np.array(info["extra_per_layer"])
        assert (q[0] == 1).all() and ((q[1:] == 0).sum(1) == kq).all(), f"G0e (Q) failed for {name}"
        assert ((m == 0).sum(1) == km + extra).all(), f"G0e (MLP) failed for {name}"
        rem = edge_removed(q, m)
        base = edge_removed(*mam.moop_masks(imp, EDGE, False, crit, ratio)[:2])
        assert rem == base - info["params_short"], f"G0p failed for {name}: {rem} vs {base}"
        base_removed = base
        np.savez(out / f"masks_{name}_l0keep{sfx}.npz", q_mask=q.astype(np.float32), mlp_mask=m.astype(np.float32))
        masks[name] = (q, m)
        cfg["arms"][f"{name}_l0keep"] = {"criterion": crit, "removed_params": rem, "removed_frac_total": rem / TOTAL,
                                         "removed_frac_text_decoder": rem / TEXT_LAYERS, **info,
                                         "file": str(out / f"masks_{name}_l0keep{sfx}.npz")}
    ov = {}
    for a, b in (("moop", "fm_only"), ("moop", "ce_only"), ("fm_only", "ce_only")):
        for ax, i in (("q", 0), ("mlp", 1)):
            ra, rb = masks[a][i] == 0, masks[b][i] == 0
            ov[f"{a}_l0keep/{b}_l0keep/{ax}"] = float((ra & rb).sum() / ra.sum())
    cfg["removed_set_overlap"] = ov
    cfg["non_l0keep_removed_params"] = base_removed
    (out / f"masks_config_l0keep{sfx}.json").write_text(json.dumps(cfg, indent=2))
    print(json.dumps({k: v for k, v in cfg.items() if k != "arms"}, indent=1))
    for n, a in cfg["arms"].items():
        print(f"{n}: removed {a['removed_params'] / 1e6:.3f} M = {100 * a['removed_frac_total']:.2f}% of total; layer 0 "
              f"heads restored {a['layer0_heads_restored']}, +{a['extra_mlp_channels']} MLP channels "
              f"({a['extra_per_layer_max_min']}/layer), short {a['params_short']}")


def qcut_masks(imp, dims, crit, ratio, q_cut, mlp_cost):
    """make_a2_moop_masks.moop_masks(q_cut=N) with the MLP channel cost as a parameter: the per-layer budget of the ratio
    allocation is kept, N Q heads are cut per layer and the MLP channel count is derived from the budget."""
    L, H, I = dims["layers"], dims["heads"], dims["inter"]
    sq, sm = mam.scores(imp, crit)
    attn_cost = 2 * dims["head_dim"] * dims["hidden"]
    per_layer = round(ratio * H) * attn_cost + round(ratio * I) * mlp_cost
    n_m, rem = divmod(per_layer - q_cut * attn_cost, mlp_cost)
    assert rem == 0, f"q-cut {q_cut} leaves {rem} parameters over"
    assert 0 <= q_cut < H and 0 < n_m < I
    q = mam.select_mask_ratios(sq, np.full(L, q_cut / H))
    m = mam.select_mask_ratios(sm, np.full(L, n_m / I))
    return q, m, {"q_cut_per_layer": q_cut, "mlp_cut_per_layer": int(n_m), "per_layer_budget": int(per_layer)}


def make_qcut(imp, ratio, sfx, out, q_cuts):
    """--q-cut N [N ...]: masks_{moop,fm_only,ce_only}_q<N>[_u<P>].npz with Edge's MLP cost (2 x hidden)."""
    g0 = mam.g0_regression(qcut=True)
    print("G0 regression on A1.5 (incl. slim_dual_u40_qcut4_v2):", json.dumps(g0))
    assert all(all(v.values()) for v in g0.values()), "G0 FAILED"
    imp15 = np.load(REPO / "outputs" / "importance_v2" / "importance.npz")
    g0q = {crit: bool(all(np.array_equal(a, b) for a, b in zip(
        qcut_masks(imp15, mam.A15, crit, mam.RATIO, 4, 3 * mam.A15["hidden"])[:2],
        mam.moop_masks(imp15, mam.A15, False, crit, mam.RATIO, 4)[:2]))) for crit in ("dual", "traj", "coc")}
    print("G0q (parameterised q-cut == make_a2_moop_masks q-cut 4 on A1.5):", g0q)
    assert all(g0q.values()), "G0q FAILED"
    for n in q_cuts:
        cfg = {"pct": args_pct(ratio), "ratio": ratio, "q_cut": n, "importance": str(EDGE_IMP), "dims": EDGE,
               "g0_regression": g0, "g0q_parameterised_qcut": g0q, "mlp_cost": 2 * EDGE["hidden"], "total_params": TOTAL,
               "text_decoder_params": TEXT_LAYERS, "arms": {}}
        masks = {}
        for name, crit in (("moop", "dual"), ("fm_only", "traj"), ("ce_only", "coc")):
            q, m, info = qcut_masks(imp, EDGE, crit, ratio, n, 2 * EDGE["hidden"])
            qb, mb, _ = mam.moop_masks(imp, EDGE, False, crit, ratio)
            assert ((q == 0).sum(1) == n).all() and ((m == 0).sum(1) == info["mlp_cut_per_layer"]).all(), f"G0e {name}"
            assert edge_removed(q, m) == edge_removed(qb, mb), f"G0p failed for {name}"
            nested = bool(((q == 0) <= (qb == 0)).all() and ((mb == 0) <= (m == 0)).all())
            assert nested, f"G0n failed for {name}: q-cut sets do not nest with the base allocation"
            np.savez(out / f"masks_{name}_q{n}{sfx}.npz", q_mask=q.astype(np.float32), mlp_mask=m.astype(np.float32))
            masks[name] = (q, m)
            rem = edge_removed(q, m)
            cfg["arms"][f"{name}_q{n}"] = {"criterion": crit, "removed_params": rem, "removed_frac_total": rem / TOTAL,
                                           "removed_frac_text_decoder": rem / TEXT_LAYERS, "nested_with_base": nested,
                                           **info, "file": str(out / f"masks_{name}_q{n}{sfx}.npz")}
        ov = {}
        for a, b in (("moop", "fm_only"), ("moop", "ce_only"), ("fm_only", "ce_only")):
            for ax, i in (("q", 0), ("mlp", 1)):
                ra, rb = masks[a][i] == 0, masks[b][i] == 0
                ov[f"{a}_q{n}/{b}_q{n}/{ax}"] = float((ra & rb).sum() / ra.sum())
        cfg["removed_set_overlap"] = ov
        cfg["layer0_q_removed"] = {k: [int(h) for h in np.flatnonzero(v[0][0] == 0)] for k, v in masks.items()}
        (out / f"masks_config_q{n}{sfx}.json").write_text(json.dumps(cfg, indent=2))
        print(json.dumps({k: v for k, v in cfg.items() if k not in ("arms", "g0_regression")}, indent=1))
        for k, a in cfg["arms"].items():
            print(f"{k}: {a['q_cut_per_layer']} Q heads + {a['mlp_cut_per_layer']} MLP channels per layer, removed "
                  f"{a['removed_params']:,} = {100 * a['removed_frac_total']:.2f}% of total")


def args_pct(ratio):
    return 40 if ratio == mam.RATIO else round(ratio * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pct", type=int, default=40, help="40 = the matched u40 (0.3985632694); else the literal N/100")
    ap.add_argument("--l0keep", action="store_true", help="write the _l0keep masks only (Edge MLP cost)")
    ap.add_argument("--q-cut", type=int, nargs="+", default=None,
                    help="Q heads cut per layer at the --pct budget (make_slim _qcut, Edge MLP cost); writes _q<N> masks only")
    args = ap.parse_args()
    ratio = mam.RATIO if args.pct == 40 else args.pct / 100
    sfx = "" if args.pct == 40 else f"_u{args.pct}"
    out = REPO / "outputs" / "iclr27-rebuttal" / "edge_moop"
    out.mkdir(parents=True, exist_ok=True)

    g0 = mam.g0_regression()
    print("G0 regression on A1.5:", json.dumps(g0))
    assert all(all(v.values()) for v in g0.values()), "G0 FAILED"

    imp = np.load(EDGE_IMP)
    assert imp["traj_vlm_q"].shape == (28, 16) and imp["coc_vlm_mlp"].shape == (28, 9216)
    if args.l0keep:
        make_l0keep(imp, ratio, sfx, out, g0)
        return
    if args.q_cut:
        make_qcut(imp, ratio, sfx, out, args.q_cut)
        return
    kq, km = round(ratio * EDGE["heads"]), round(ratio * EDGE["inter"])
    cfg = {"pct": args.pct, "ratio": ratio, "importance": str(EDGE_IMP), "dims": EDGE, "g0_regression": g0,
           "cut_per_layer": {"q_heads": kq, "mlp_channels": km}, "total_params": TOTAL,
           "text_decoder_params": TEXT_LAYERS, "arms": {}}
    masks = {}
    for name, crit in (("moop", "dual"), ("fm_only", "traj"), ("ce_only", "coc")):
        q, m, _ = mam.moop_masks(imp, EDGE, False, crit, ratio)
        assert ((q == 0).sum(1) == kq).all() and ((m == 0).sum(1) == km).all(), f"G0e failed for {name}"
        np.savez(out / f"masks_{name}{sfx}.npz", q_mask=q.astype(np.float32), mlp_mask=m.astype(np.float32))
        rem = edge_removed(q, m)
        masks[name] = (q, m)
        cfg["arms"][name] = {"criterion": crit, "removed_params": rem, "removed_frac_total": rem / TOTAL,
                             "removed_frac_text_decoder": rem / TEXT_LAYERS,
                             "q_removed_per_layer": [int(x) for x in (q == 0).sum(1)],
                             "file": str(out / f"masks_{name}{sfx}.npz")}
    # overlap of the removed sets between arms (Q heads and MLP channels, all layers)
    ov = {}
    for a, b in (("moop", "fm_only"), ("moop", "ce_only"), ("fm_only", "ce_only")):
        for ax, i in (("q", 0), ("mlp", 1)):
            ra, rb = masks[a][i] == 0, masks[b][i] == 0
            ov[f"{a}/{b}/{ax}"] = float((ra & rb).sum() / ra.sum())
    cfg["removed_set_overlap"] = ov
    cfg["layer0_q_removed"] = {n: [int(h) for h in np.flatnonzero(masks[n][0][0] == 0)] for n in masks}
    (out / f"masks_config{sfx}.json").write_text(json.dumps(cfg, indent=2))
    print(json.dumps({k: v for k, v in cfg.items() if k not in ("arms",)}, indent=1))
    for n, a in cfg["arms"].items():
        print(f"{n}: removed {a['removed_params'] / 1e6:.1f} M = {100 * a['removed_frac_total']:.1f}% of total, "
              f"{100 * a['removed_frac_text_decoder']:.1f}% of text decoder")


if __name__ == "__main__":
    main()
