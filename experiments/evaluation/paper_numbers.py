"""Every open-loop cell of the paper tables under the frozen protocol.

Protocol (user-fixed 2026-08-19): rollout condition only (never teacher-forced),
sets = indist val 500 / official test 500 / OOD-val 262, metric = minADE@6 and
minFDE@6 with the MEAN as the headline (median beside it). All rows come from k=8
runs that stored per-sample arrays; the first six samples are exactly what a
6-sample run would have drawn (seeds are base+k). Arms evaluated on the full OOD
set are reduced to OOD-val by the stored `split` field, which pairs clip-for-clip
with the ood_val manifest runs.

Prints one block per paper table; nothing is written, the .tex is edited by hand
from this output so every number in the paper has a single recomputable source.

Usage:
  .venv/bin/python experiments/evaluation/paper_numbers.py
"""

import glob
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
K = 6
BOOT = 5000

# arm -> set -> (rows dir, filter_ood_val)
SETS = ("indist", "test", "oodval")
ARMS = {
    "baseline": {"indist": ("baseline_ada_ps_indist", False),
                 "test": ("baseline_ada_ps_test", False),
                 "oodval": ("baseline_ada_ps_oodval", False)},
    "dual": {"indist": ("dual_u40_v2_ps_indist", False),
             "test": ("dual_u40_v2_ps_test", False),
             "oodval": ("dual_u40_v2_ps_ood", True)},
    "jtraj": {"indist": ("jtraj_u40_v2_ps_indist", False),
              "test": ("jtraj_u40_v2_ps_test", False),
              "oodval": ("jtraj_u40_v2_ps_oodval", False)},
    "traj": {"indist": ("traj_u40_v2_indist", False),
             "test": ("traj_u40_v2_test", False),
             "oodval": ("traj_u40_v2_ood", True)},
    "coc": {"indist": ("coc_u40_v2_indist", False),
            "test": ("coc_u40_v2_test", False),
            "oodval": ("coc_u40_v2_ood", True)},
    "j": {"indist": ("j_u40_v2_indist", False),
          "test": ("j_u40_v2_test", False),
          "oodval": ("j_u40_v2_ood", True)},
    "wanda": {"indist": ("wanda_u40_v2_indist", False),
              "test": ("wanda_u40_v2_test", False),
              "oodval": ("wanda_u40_v2_oodval", False)},
    "wandatxt": {"indist": ("wandatxt_u40_v2_indist", False),
                 "test": ("wandatxt_u40_v2_test", False),
                 "oodval": ("wandatxt_u40_v2_oodval", False)},
    # Tyr: *_sel_* = OSSCAR selection with ORIGINAL weights (the --no-state builds
    # evaluate as selection-only); *_d1r = selection + reconstruction, damp 1.0, state
    "tyr_sel_uniform": {"indist": ("tyr_uniform_u40_indist", False),
                        "test": ("tyr_uniform_u40_test", False),
                        "oodval": ("tyr_uniform_u40_oodval", False)},
    "tyr_sel_search": {"indist": ("tyr_u40_indist", False),
                       "test": ("tyr_u40_test", False),
                       "oodval": ("tyr_u40_oodval", False)},
    "tyr_uniform_d1r": {"indist": ("tyr_uniform_u40_d1r_indist", False),
                        "test": ("tyr_uniform_u40_d1r_test", False),
                        "oodval": ("tyr_uniform_u40_d1r_oodval", False)},
    "tyr_d1r": {"indist": ("tyr_u40_d1r_indist", False),
                "test": ("tyr_u40_d1r_test", False),
                "oodval": ("tyr_u40_d1r_oodval", False)},
    # *_r = upstream damping 1e-2 + reconstruction (the faithful Tyr arms)
    "tyr_uniform_r": {"indist": ("tyr_uniform_u40_r_indist", False),
                      "test": ("tyr_uniform_u40_r_test", False),
                      "oodval": ("tyr_uniform_u40_r_oodval", False)},
    "tyr_r": {"indist": ("tyr_u40_r_indist", False),
              "test": ("tyr_u40_r_test", False),
              "oodval": ("tyr_u40_r_oodval", False)},
    # dual x Tyr factorial (plans/2026-08-21_dual-global.md): g = searched allocation,
    # r = OSSCAR reconstruction, tyralloc = dual selection on tyr_r's allocation
    "dualg": {"indist": ("dualg_u40_indist", False), "test": ("dualg_u40_test", False),
              "oodval": ("dualg_u40_oodval", False)},
    "dualr": {"indist": ("dualr_u40_indist", False), "test": ("dualr_u40_test", False),
              "oodval": ("dualr_u40_oodval", False)},
    "dualgr": {"indist": ("dualgr_u40_indist", False), "test": ("dualgr_u40_test", False),
               "oodval": ("dualgr_u40_oodval", False)},
    "dualscope": {"indist": ("dualscope_u40_indist", False),
                  "test": ("dualscope_u40_test", False),
                  "oodval": ("dualscope_u40_oodval", False)},
    "dualg_tyralloc": {"indist": ("dualg_tyralloc_u40_indist", False),
                       "test": ("dualg_tyralloc_u40_test", False),
                       "oodval": ("dualg_tyralloc_u40_oodval", False)},
    # expert axis decomposition (plans/2026-08-28_expert-axis-ablation.md): one expert
    # axis at a time, znorm traj_exp_*; q25/m25 ratio-matched, m_pm = expertm_c341
    # parameter-matched to q25 (75.4M vs 75.5M)
    "expert_q25": {"indist": ("expertq_u25_indist", False),
                   "test": ("expertq_u25_test", False),
                   "oodval": ("expertq_u25_oodval", False)},
    "expert_m25": {"indist": ("expertm_u25_indist", False),
                   "test": ("expertm_u25_test", False),
                   "oodval": ("expertm_u25_oodval", False)},
    "expert_m_pm": {"indist": ("expertm_c341_indist", False),
                    "test": ("expertm_c341_test", False),
                    "oodval": ("expertm_c341_oodval", False)},
    "expert_q50": {"indist": ("expertq_u50_indist", False),
                   "test": ("expertq_u50_test", False),
                   "oodval": ("expertq_u50_oodval", False)},
    "expert_m50": {"indist": ("expertm_u50_indist", False),
                   "test": ("expertm_u50_test", False),
                   "oodval": ("expertm_u50_oodval", False)},
    "expert_both25": {"indist": ("expert_znorm_r25_ps_indist", False)},
    # cache-targeted reconstruction (plans/2026-08-29_cache-targeted-reconstruction.md):
    # dual selection + expert-weighted OSSCAR refit of layers >= 16 / >= 24 only
    "dualrc_s16": {"indist": ("dualrc_u40_s16_indist", False),
                   "test": ("dualrc_u40_s16_test", False),
                   "oodval": ("dualrc_u40_s16_oodval", False)},
    "dualrc_s24": {"indist": ("dualrc_u40_s24_indist", False),
                   "test": ("dualrc_u40_s24_test", False),
                   "oodval": ("dualrc_u40_s24_oodval", False)},
    # dualr with a different Hessian only (plans/2026-08-30_dualr-weighted-hessian.md):
    # d = own-CoC at decode share 0.16, e = expert-attention-weighted prefill, w = both
    "dualr_rep": {"indist": ("dualr_rep_u40_indist", False),
                  "test": ("dualr_rep_u40_test", False),
                  "oodval": ("dualr_rep_u40_oodval", False)},
    # cache-Jacobian criterion (plans/2026-08-30_cache-jlens-criterion.md): dual with
    # I_traj replaced by I_cache (max(rank I_cache, rank I_CoC)), and I_cache alone
    "cachedual": {"indist": ("cachedual_u40_v2_indist", False),
                  "test": ("cachedual_u40_v2_test", False),
                  "oodval": ("cachedual_u40_v2_oodval", False)},
    "cacheonly": {"indist": ("cacheonly_u40_v2_indist", False),
                  "test": ("cacheonly_u40_v2_test", False),
                  "oodval": ("cacheonly_u40_v2_oodval", False)},
    "dualr_d": {"indist": ("dualr_d_u40_indist", False),
                "test": ("dualr_d_u40_test", False),
                "oodval": ("dualr_d_u40_oodval", False)},
    "dualr_e": {"indist": ("dualr_e_u40_indist", False),
                "test": ("dualr_e_u40_test", False),
                "oodval": ("dualr_e_u40_oodval", False)},
    "dualr_w": {"indist": ("dualr_w_u40_indist", False),
                "test": ("dualr_w_u40_test", False),
                "oodval": ("dualr_w_u40_oodval", False)},
    # wl = w's Hessian plus LingoQA train (plans/2026-08-30_dualr-w-lingo.md), and the
    # two-tower configs built on it: the same refitted VLM plus expert MLP-only pruning
    # at 50% / 75%, expert Q heads and KV untouched
    # (plans/2026-08-31_dualrwl-expert-mlp.md)
    "dualr_wl": {"indist": ("dualr_wl_u40_indist", False),
                 "test": ("dualr_wl_u40_test", False),
                 "oodval": ("dualr_wl_u40_oodval", False)},
    "dualrwl_em50": {"indist": ("dualrwl_em50_indist", False),
                     "test": ("dualrwl_em50_test", False),
                     "oodval": ("dualrwl_em50_oodval", False)},
    "dualrwl_em75": {"indist": ("dualrwl_em75_indist", False),
                     "test": ("dualrwl_em75_test", False),
                     "oodval": ("dualrwl_em75_oodval", False)},
    "dualrwl_em87p5": {"indist": ("dualrwl_em87p5_indist", False),
                      "test": ("dualrwl_em87p5_test", False),
                      "oodval": ("dualrwl_em87p5_oodval", False)},
    "dualrwl_em93p75": {"indist": ("dualrwl_em93p75_indist", False),
                        "test": ("dualrwl_em93p75_test", False),
                        "oodval": ("dualrwl_em93p75_oodval", False)},
    "dualexp_em93p75": {"indist": ("dualexp_em93p75_indist", False),
                        "test": ("dualexp_em93p75_test", False),
                        "oodval": ("dualexp_em93p75_oodval", False)},
    "dualrwl_em96p875": {"indist": ("dualrwl_em96p875_indist", False),
                         "test": ("dualrwl_em96p875_test", False),
                         "oodval": ("dualrwl_em96p875_oodval", False)},
    "dualrwl_em98p4375": {"indist": ("dualrwl_em98p4375_indist", False),
                          "test": ("dualrwl_em98p4375_test", False),
                          "oodval": ("dualrwl_em98p4375_oodval", False)},
    "dualrwl_em100": {"indist": ("dualrwl_em100_indist", False),
                      "test": ("dualrwl_em100_test", False),
                      "oodval": ("dualrwl_em100_oodval", False)},
    # unpruned model re-measured on Blackwell WITH per-sample arrays (baseline_* has only
    # minADE@8): the same-architecture anchor for the Blackwell-evaluated expert_q50/m50
    "baseline_bw": {"indist": ("baseline_bw_ps_indist", False),
                    "test": ("baseline_bw_ps_test", False),
                    "oodval": ("baseline_bw_ps_oodval", False)},
    "it3": {"indist": ("iter_dual_indist", False),
            "test": ("iter_dual_test", False),
            "oodval": ("iter_dual_ood", True)},
    "uniform_w8": {"indist": ("uniform_w8_indist", False),
                   "test": ("uniform_w8_test", False),
                   "oodval": ("uniform_w8_ood", True)},
    "qvla_coc_b8": {"indist": ("qvla_coc_b8_indist", False),
                    "test": ("qvla_coc_b8_test", False),
                    "oodval": ("qvla_coc_b8_ood", True)},
    "uniform_w4": {"indist": ("uniform_w4_indist", False),
                   "test": ("uniform_w4_test", False),
                   "oodval": ("uniform_w4_ood", True)},
    "qvla_coc_b4": {"indist": ("qvla_coc_b4_indist", False),
                    "test": ("qvla_coc_b4_test", False),
                    "oodval": ("qvla_coc_b4_ood", True)},
    "dualsum": {"test": ("dualsum_u40_v2_test", False)},
    "dualprod": {"test": ("dualprod_u40_v2_test", False)},
    "dual_u55": {"test": ("dual_u55_test", False)},
    "jtraj_u55": {"test": ("jtraj_u55_test", False)},
    "w8_all": {"test": ("w8_all_test", False)},
    "w4_all": {"test": ("w4_all_test", False)},
    "prune_w8": {"test": ("dual_u40_w8_test", False)},
    "prune_w4": {"test": ("dual_u40_w4_test", False)},
}


def load(dirname, ood_val_only):
    rows = []
    for f in sorted(glob.glob(str(REPO / "outputs" / dirname / "*_s*of*.json"))):
        rows.extend(json.loads(Path(f).read_text()))
    if ood_val_only:
        rows = [r for r in rows if r.get("split") == "val"]
    out = {}
    for r in rows:
        if "ade_rollout_k" not in r:
            raise SystemExit(f"{dirname}: {r['clip_id']} has no per-sample arrays")
        out[r["clip_id"]] = r
    return out


def at6(r, key):
    return float(np.min(np.asarray(r[key], dtype=float)[:K]))


def stats(rows):
    a = np.array([at6(r, "ade_rollout_k") for r in rows.values()])
    f = np.array([at6(r, "fde_rollout_k") for r in rows.values()])
    d = float(np.mean([r["coc_degenerate"] for r in rows.values()]))
    return a, f, d


def paired(base, arm):
    ids = sorted(set(base) & set(arm))
    da = np.array([at6(arm[i], "ade_rollout_k") - at6(base[i], "ade_rollout_k")
                   for i in ids])
    rng = np.random.default_rng(0)
    meds = [np.median(da[rng.integers(0, len(da), len(da))]) for _ in range(BOOT)]
    lo, hi = np.percentile(meds, [2.5, 97.5])
    star = "*" if lo > 0 or hi < 0 else " "
    return len(ids), float(np.median(da)), float(lo), float(hi), star


def main():
    cache = {}
    for arm, sets in ARMS.items():
        cache[arm] = {}
        for s, (d, fv) in sets.items():
            try:
                cache[arm][s] = load(d, fv)
            except SystemExit as e:
                print(f"!! {e}")
    print(f"== minADE@{K} / minFDE@{K} mean (median) | degen | paired dADE med [CI] "
          f"vs baseline ==")
    for arm in ARMS:
        for s in SETS:
            if s not in cache.get(arm, {}) or not cache[arm][s]:
                continue
            a, f, dg = stats(cache[arm][s])
            line = (f"{arm:12s} {s:6s} n={len(a):4d}  "
                    f"ADE {a.mean():.4f} ({np.median(a):.4f})  "
                    f"FDE {f.mean():.4f} ({np.median(f):.4f})  degen {dg:.3f}")
            if arm != "baseline" and s in cache["baseline"]:
                _n, m, lo, hi, st = paired(cache["baseline"][s], cache[arm][s])
                line += f"  d_med {m:+.4f} [{lo:+.4f},{hi:+.4f}]{st}"
            print(line)
        print()


if __name__ == "__main__":
    main()
