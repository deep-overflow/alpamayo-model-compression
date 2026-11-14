"""Reasoning fidelity on existing A2 rows: how often an arm's generated CoC equals dense's (same seed, same clip), and a token-level
similarity, per set / allocation. Paired bootstrap over clips for MoOP - FM-only and MoOP - CE-only."""
import difflib
import glob
import json
import sys

import numpy as np

O = "/mnt/nvme1n1/ad_vla/outputs/chan/iclr27-rebuttal"
SETS = {
    "test500 u40 (Ada)": ("a2_test500_dense", {"MoOP": "a2_test500_moop", "FM-only": "a2_test500_fm_only", "CE-only": "a2_test500_ce_only"}),
    "test500 u50 (Blackwell)": ("a2_test500_dense_bw", {"MoOP": "a2_test500_moop_u50_bw", "FM-only": "a2_test500_fm_only_u50_bw",
                                                        "CE-only": "a2_test500_ce_only_u50_bw"}),
    "OOD-Val u40 (GH200)": ("a2_gh200/a2_oodval_dense", {"MoOP": "a2_gh200/a2_oodval_moop", "FM-only": "a2_gh200/a2_oodval_fm_only",
                                                          "CE-only": "a2_gh200/a2_oodval_ce_only"}),
    "OOD-train u40 (GH200)": ("a2_gh200/a2_oodtrain_dense", {"MoOP": "a2_gh200/a2_oodtrain_moop",
                                                              "FM-only": "a2_gh200/a2_oodtrain_fm_only",
                                                              "CE-only": "a2_gh200/a2_oodtrain_ce_only"}),
}


def rows(d):
    out = {}
    for f in glob.glob(f"{O}/{d}/alpamayo2_s*of*.json"):
        with open(f) as fh:
            out.update({r["clip_id"]: r for r in json.load(fh)})
    return out


def boot(x, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


for name, (dense_dir, arms) in SETS.items():
    dr = rows(dense_dir)
    ar = {k: rows(v) for k, v in arms.items()}
    ids = sorted(set(dr).intersection(*[set(v) for v in ar.values()]))
    if not ids:
        print(name, "missing"); continue
    em, sim = {}, {}
    for k, r in ar.items():
        em[k] = np.array([float(r[c]["gen_coc"].strip() == dr[c]["gen_coc"].strip()) for c in ids])
        sim[k] = np.array([difflib.SequenceMatcher(None, r[c]["gen_coc"].split(), dr[c]["gen_coc"].split()).ratio() for c in ids])
    print(f"\n{name} (n={len(ids)})")
    for k in arms:
        print(f"  {k:<8} exact-match with dense {em[k].mean() * 100:5.1f}%   word-sim {sim[k].mean():.3f}")
    for b in ("FM-only", "CE-only"):
        d1, d2 = em["MoOP"] - em[b], sim["MoOP"] - sim[b]
        print(f"  MoOP - {b:<8} exact-match {d1.mean() * 100:+.1f}pp {tuple(round(v * 100, 1) for v in boot(d1))}   "
              f"word-sim {d2.mean():+.3f} {tuple(round(v, 3) for v in boot(d2))}")
sys.exit(0)
