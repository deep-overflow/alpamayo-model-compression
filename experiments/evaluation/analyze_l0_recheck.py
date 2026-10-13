"""Do earlier conclusions survive once layer 0's Q heads are kept? Re-measurement.

`analyze_l0_census.py` flagged conclusions whose arms differed in which layer-0 heads they
dropped. Each arm below is re-evaluated with layer 0's 32 Q heads restored and nothing
else changed (`make_l0_exempt.py`, RECHECK; 0.51% fewer parameters removed), paired clip
by clip with its own stored evaluation, the shipped arm and the unpruned model.

  self-anchored target   dualself, dualbestn6     rejected at +0.054..+0.164 vs dual
  calibration size       dual_u40_st2000          +0.116 vs calib_100 at n=2,000
  draw rule              rd a-c, se a-c           se - rd = -0.153 (3-seed per-clip means)
  clip weighting         dualsafe (val500)        rejected at +0.19 vs dual

Reading: a conclusion "survives" if the contrast it rested on is still there between the
exempt variants; it was the layer-0 lottery if the contrast closes.

Usage:
  python experiments/evaluation/analyze_l0_recheck.py
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
O = REPO / "outputs"
BOOT, K = 20000, 6
# name -> (own evaluation, set)
ARMS = {"self": ("dualself_u40_v2_ps_test", "test"), "bestn6": ("dualbestn6_u40_v2_ps_test", "test"),
        "st2000": ("dual_u40_st2000_test", "test"),
        **{f"rd{x}": (f"dual_rd_{x}_test", "test") for x in "abc"},
        **{f"se{x}": (f"dual_se_{x}_test", "test") for x in "abc"},
        "safe": ("dualsafe_u40_v2_indist", "indist")}
REF = {"test": ("baseline_ada_ps_test", "dual_u40_v2_ps_test", "swap_ship_L0full_500", "500"),
       "indist": ("baseline_ada_ps_indist", "dual_u40_v2_ps_indist", "swap_ship_L0full_val500",
                  "val500")}


def load_run(exp):
    rows = {}
    for p in sorted((O / exp).glob("*.json")):
        if not p.name.startswith("config"):
            for r in json.loads(p.read_text()):
                rows[r["clip_id"]] = r
    return rows


def main():
    out = O / "swap_masks"
    rng = np.random.default_rng(0)
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    def ci(d):
        idx = rng.integers(0, len(d), (BOOT, len(d)))
        lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
        nz = d[d != 0]
        return {"mean": float(d.mean()), "ci": [float(lo), float(hi)],
                "median": float(np.median(d)),
                "p": float(wilcoxon(nz).pvalue) if len(nz) > 10 else 1.0}

    def f(s):
        star = "*" if s["ci"][0] > 0 or s["ci"][1] < 0 else " "
        return f"{s['mean']:+.4f} [{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}]{star}"

    metrics = {"arms": {}, "contrasts": {}}
    ctx, own, ex = {}, {}, {}
    for st, (b, s, se, suf) in REF.items():
        base = load_run(b)
        ids = sorted(base)

        def vec(rows, ids=ids):
            return np.array([min(rows[c]["ade_rollout_k"][:K]) for c in ids])

        ctx[st] = {"ids": ids, "vec": vec, "b": vec(base), "ship": vec(load_run(s)),
                   "bucket": np.array([base[c]["bucket"] for c in ids]), "suf": suf}
        r = load_run(se) if (O / se).exists() else {}
        ctx[st]["ship_ex"] = vec(r) if all(c in r for c in ids) else None

    say(f"{'arm':8s}{'set':>8s}{'own':>8s}{'exempt':>8s}{'shipped':>9s}   exempt - own [95% CI]            "
        f"own - shipped [95% CI]           exempt - shipped [95% CI]        closed")
    for n, (own_exp, st) in ARMS.items():
        c = ctx[st]
        ro = load_run(own_exp)
        re = load_run(f"swap_{n}_L0full_{c['suf']}") if (O / f"swap_{n}_L0full_{c['suf']}").exists() else {}
        if not all(i in ro for i in c["ids"]) or not all(i in re for i in c["ids"]):
            say(f"{n:8s}{st:>8s}   pending (own {sum(i in ro for i in c['ids'])}, "
                f"exempt {sum(i in re for i in c['ids'])} of {len(c['ids'])})")
            continue
        own[n], ex[n] = c["vec"](ro), c["vec"](re)
        d1, d2, d3 = ci(ex[n] - own[n]), ci(own[n] - c["ship"]), ci(ex[n] - c["ship"])
        closed = 1 - d3["mean"] / d2["mean"] if abs(d2["mean"]) > 1e-9 else float("nan")
        cr = (c["bucket"] == "cruise")
        metrics["arms"][n] = {"set": st, "own": float(own[n].mean()), "exempt": float(ex[n].mean()),
                              "exempt_minus_own": d1, "own_minus_shipped": d2,
                              "exempt_minus_shipped": d3, "gap_closed": float(closed),
                              "cruise_own": float((own[n] - c["b"])[cr].mean()),
                              "cruise_exempt": float((ex[n] - c["b"])[cr].mean()),
                              "degen_exempt": float(np.mean([re[i]["coc_degenerate"] for i in c["ids"]]))}
        say(f"{n:8s}{st:>8s}{own[n].mean():8.4f}{ex[n].mean():8.4f}{c['ship'].mean():9.4f}   "
            f"{f(d1)}   {f(d2)}   {f(d3)}   {closed:+.2f}")
    say("closed: share of (own - shipped) that keeping layer 0 removes")
    say("\ncruise, minus unpruned: own -> exempt")
    for n, r in metrics["arms"].items():
        say(f"  {n:8s} {r['cruise_own']:+.3f} -> {r['cruise_exempt']:+.3f}")

    # draw rule: se - rd on per-clip 3-seed means, before and after
    for lab, src in (("as built", own), ("layer 0 kept", ex)):
        if all(f"{r}{x}" in src for r in ("rd", "se") for x in "abc"):
            rd = np.mean([src[f"rd{x}"] for x in "abc"], axis=0)
            se = np.mean([src[f"se{x}"] for x in "abc"], axis=0)
            d = ci(se - rd)
            metrics["contrasts"][f"se_minus_rd ({lab})"] = d
            sd = {r: float(np.std([src[f"{r}{x}"].mean() for x in "abc"], ddof=1)) for r in ("rd", "se")}
            say(f"\nse - rd, {lab}: {f(d)} median {d['median']:+.4f} p={d['p']:.1e}; "
                f"arm means rd {' '.join(f'{src[f'rd{x}'].mean():.3f}' for x in 'abc')} "
                f"(SD {sd['rd']:.3f}), se {' '.join(f'{src[f'se{x}'].mean():.3f}' for x in 'abc')} "
                f"(SD {sd['se']:.3f})")
    # against the exempt shipped arm, the matched-condition reference
    se_ = ctx["test"]["ship_ex"]
    if se_ is not None:
        say("\nexempt arm minus exempt shipped arm (both with layer 0 kept), test500:")
        for n in list(ex):
            if ARMS[n][1] == "test":
                d = ci(ex[n] - se_)
                metrics["arms"][n]["exempt_minus_exempt_shipped"] = d
                say(f"  {n:8s} {f(d)} median {d['median']:+.4f} p={d['p']:.1e}")
    sv = ctx["indist"]["ship_ex"]
    if sv is not None and "safe" in ex:
        d = ci(ex["safe"] - sv)
        metrics["arms"]["safe"]["exempt_minus_exempt_shipped"] = d
        say(f"val500: safe exempt - shipped exempt {f(d)} median {d['median']:+.4f} p={d['p']:.1e}")

    (out / "metrics_recheck.json").write_text(json.dumps(metrics, indent=2))
    (out / "summary_recheck.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
