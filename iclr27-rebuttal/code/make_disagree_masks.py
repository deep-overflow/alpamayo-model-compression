"""VLM keep-masks for MoOP with the layer budget set by I_traj / I_CoC top-set disagreement (Alpamayo 1.5, u40, l0keep).

Plan: iclr27-rebuttal/plans/2026-11-10_moop-disagreement-allocation-test500.md (approved 2026-11-10; D1 top-set overlap,
D2 strengths 0.5 and 1.0, D3 no depth control, D4 layer 0 = MoOP l0keep, D5 no layer loses more than half its units).

  disagreement  d_l = 1 - |T & C| / |T | C| per layer and per axis (Q heads, MLP channels), T / C = the top-k units by
                I_traj / I_CoC, k = the units MoOP l0keep keeps in that layer (importance_v2). Layer 35 has no I_traj
                gradient: d_35 = mean of layers 30-34 (the July grid's rule).
  cuts          layers 1-35: cut_l = mean cut * (1 - s * z_l), z_l = d_l centred over 1-35 and scaled so max |z| = 1; per-axis
                totals = MoOP l0keep's (455 Q heads, 172,508 MLP channels on layers 1-35). Cap: no layer loses more than half
                its units (16 Q heads, 6,144 channels); the excess goes to the uncapped layers in proportion to their cuts.
                Integers by largest remainder (ties: earlier layer first), never above the cap.
  layer 0       identical to MoOP l0keep: all 32 Q heads kept, its 4,929 MLP channels cut.
  selection     MoOP's within-layer score max(rank_norm I_traj, rank_norm I_CoC), lowest cut first (mask_lib.select_mask_ratios),
                via make_a2_moop_masks.scores (that code reproduces the shipped MoOP kept sets unit for unit).

G0: strength 0 must reproduce outputs/slim_dual_u40_l0keep_v2 unit for unit; every arm must remove exactly MoOP l0keep's
parameters, keep layer 0 as MoOP does, respect the cap, and keep higher-scoring units than it cuts in every layer.
Writes outputs/iclr27-rebuttal/disagree_alloc/<arm>/masks.npz (vq (36, 32), vm (36, 12288); 1 = keep, the keys
make_slim's maskfile_ config reads) and alloc.json, plus alloc_summary.json.
Usage (repo root): .venv/bin/python iclr27-rebuttal/code/make_disagree_masks.py
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))
sys.path.insert(0, str(REPO / "iclr27-rebuttal" / "code"))
from make_a2_moop_masks import A15, removed_params, scores
from mask_lib import select_mask_ratios

OUT = REPO / "outputs" / "iclr27-rebuttal" / "disagree_alloc"
REF = REPO / "outputs" / "slim_dual_u40_l0keep_v2" / "slim_meta.json"
ARMS = {"s000": 0.0, "dis050": 0.5, "dis100": 1.0}
CAP_FRAC = 0.5                 # D5 (user, 2026-11-10): early layers must not be cut too much
LAYERS = np.arange(1, 36)      # layer 0 is fixed (D4)


def top_set_disagreement(c, t, keep):
    """c, t (L, n) importances; keep (L,) kept counts -> d (L,), nan where an objective has no gradient."""
    d = np.full(len(c), np.nan)
    for li in range(len(c)):
        if np.allclose(c[li], 0) or np.allclose(t[li], 0):
            continue
        k = int(keep[li])
        tc = set(np.argsort(-c[li], kind="stable")[:k].tolist())
        tt = set(np.argsort(-t[li], kind="stable")[:k].tolist())
        d[li] = 1 - len(tc & tt) / len(tc | tt)
    assert np.isnan(d[35]) and not np.isnan(d[:35]).any(), "only layer 35 should lack an I_traj gradient"
    d[35] = np.nanmean(d[30:35])
    return d


def allocate(d, total, s, cap):
    """d (36,) disagreement -> integer cuts (35,) for layers 1-35 summing to total, none above cap."""
    dd = d[LAYERS]
    z = (dd - dd.mean()) / np.abs(dd - dd.mean()).max()   # (35,), max |z| = 1
    c = total / len(LAYERS) * (1 - s * z)                   # (35,) real-valued, sums to total
    raw = c.copy()
    capped = np.zeros(len(c), dtype=bool)
    while True:
        over = (c > cap + 1e-9) & ~capped
        if not over.any():
            break
        capped |= over
        c[capped] = cap
        free = ~capped
        c[free] *= (total - cap * capped.sum()) / c[free].sum()
    base = np.floor(c + 1e-9).astype(int)
    rem = int(total - base.sum())
    for i in np.argsort(-(c - base), kind="stable"):
        if rem == 0:
            break
        if base[i] < cap:
            base[i] += 1
            rem -= 1
    assert base.sum() == total and base.min() >= 0 and base.max() <= cap, (base.sum(), total, base.max(), cap)
    return base, raw, capped


def main():
    imp = np.load(REPO / "outputs" / "importance_v2" / "importance.npz")
    ref = json.loads(REF.read_text())
    H, Ii = A15["heads"], A15["inter"]
    ref_q = np.zeros((36, H))
    ref_m = np.zeros((36, Ii))
    for li, layer in enumerate(ref["vlm"]):
        ref_q[li, layer["q"]] = 1
        ref_m[li, layer["mlp"]] = 1
    keep_q, keep_m = ref_q.sum(1).astype(int), ref_m.sum(1).astype(int)          # (36,) MoOP l0keep kept counts
    cut_q_ref, cut_m_ref = H - keep_q, Ii - keep_m
    tot_q, tot_m = int(cut_q_ref[1:].sum()), int(cut_m_ref[1:].sum())
    assert cut_q_ref[0] == 0 and tot_q == 455 and tot_m == 172_508 and cut_m_ref[0] == 4929
    ref_removed = int(ref["params"]["removed"])

    d_q = top_set_disagreement(imp["coc_vlm_q"], imp["traj_vlm_q"], keep_q)
    d_m = top_set_disagreement(imp["coc_vlm_mlp"], imp["traj_vlm_mlp"], keep_m)
    sq, sm = scores(imp, "dual")                                                  # (36, 32), (36, 12288)
    cap_q, cap_m = int(CAP_FRAC * H), int(CAP_FRAC * Ii)

    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"plan": "iclr27-rebuttal/plans/2026-11-10_moop-disagreement-allocation-test500.md",
               "importance": "importance_v2", "reference": str(REF.relative_to(REPO)), "cap_frac": CAP_FRAC,
               "cap_q": cap_q, "cap_mlp": cap_m, "total_cut_q_layers1_35": tot_q, "total_cut_mlp_layers1_35": tot_m,
               "reference_removed_params": ref_removed, "d_q": d_q.tolist(), "d_mlp": d_m.tolist(), "arms": {}}
    for arm, s in ARMS.items():
        cq, rq_raw, capq = allocate(d_q, tot_q, s, cap_q)
        cm, rm_raw, capm = allocate(d_m, tot_m, s, cap_m)
        cut_q = np.concatenate([[0], cq])                                         # (36,)
        cut_m = np.concatenate([[cut_m_ref[0]], cm])                              # (36,)
        q = select_mask_ratios(sq, cut_q / H)                                     # (36, 32), 1 = keep
        m = select_mask_ratios(sm, cut_m / Ii)                                    # (36, 12288)
        assert ((q == 0).sum(1) == cut_q).all() and ((m == 0).sum(1) == cut_m).all()
        # G0
        rem = removed_params(q, m, A15)
        assert rem == ref_removed, (arm, rem, ref_removed)
        assert (q[0] == ref_q[0]).all() and (m[0] == ref_m[0]).all(), f"{arm}: layer 0 differs from MoOP l0keep"
        for sc, mk in ((sq, q), (sm, m)):
            for li in range(36):
                if (mk[li] == 0).any():
                    assert sc[li][mk[li] == 1].min() >= sc[li][mk[li] == 0].max(), (arm, li)
        if s == 0.0:
            assert (q == ref_q).all() and (m == ref_m).all(), "G0 FAILED: strength 0 does not reproduce MoOP l0keep"
        dst = OUT / arm
        dst.mkdir(exist_ok=True)
        path = dst / "masks.npz"
        if path.exists():   # a running build may read it: never rewrite, only confirm it is the same selection
            z = np.load(path)
            assert (z["vq"] == q).all() and (z["vm"] == m).all(), f"{path} differs from a rebuild"
        else:
            np.savez(path, vq=q.astype(np.float32), vm=m.astype(np.float32))
        info = {"strength": s, "cut_q": cut_q.tolist(), "cut_mlp": cut_m.tolist(),
                "cut_q_before_cap": [0.0, *rq_raw.round(3).tolist()],
                "cut_mlp_before_cap": [float(cut_m_ref[0]), *rm_raw.round(1).tolist()],
                "capped_q_layers": (np.flatnonzero(capq) + 1).tolist(),
                "capped_mlp_layers": (np.flatnonzero(capm) + 1).tolist(),
                "removed_params": rem, "same_as_moop_l0keep": bool((q == ref_q).all() and (m == ref_m).all())}
        (dst / "alloc.json").write_text(json.dumps(info, indent=2))
        summary["arms"][arm] = info
        bands = [(1, 5), (6, 17), (18, 29), (30, 35)]
        print(f"{arm} (s={s}): removed {rem:,} (= MoOP l0keep) | Q cut/layer {cut_q[1:].min()}-{cut_q[1:].max()}, "
              f"MLP {cut_m[1:].min()}-{cut_m[1:].max()} | capped Q layers {info['capped_q_layers']}, "
              f"MLP layers {info['capped_mlp_layers']}")
        print("   band means " + "; ".join(f"L{a}-{b} Q {cut_q[a:b + 1].mean():.1f} MLP {cut_m[a:b + 1].mean():.0f}"
                                         for a, b in bands))
    (OUT / "alloc_summary.json").write_text(json.dumps(summary, indent=2))
    print("G0 PASS: strength 0 reproduces MoOP l0keep; every arm removes exactly its parameters; layer 0 unchanged; "
          f"cap Q {cap_q} / MLP {cap_m} respected -> {OUT}")


if __name__ == "__main__":
    main()
