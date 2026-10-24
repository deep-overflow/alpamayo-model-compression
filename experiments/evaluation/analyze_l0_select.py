"""Step 3 of plans/2026-10-14_l0-functional-redundancy.md: does the P2 selector replace the exemption?

Four arms (shipped dual_u40_v2, and the collapse cases seed n1 / blocks nt_c / nt_d) with layer 0
chosen by run_l0_select.py's greedy P2 search inside the arm's own 13-head band, everything else
the arm's own. test500 minADE@6, each paired per clip with:

  as built   the arm itself (layer 0 cut by its own score)
  twin       its _l0keep variant (layer 0 kept, budget paid in MLP): swap_<arm>_L0fullM_500

Gates (registered before the run, 2026-10-16):
  G3         SD of the four arm means <= 0.05 and no arm above 1.05
  G3-twin    each arm's 95% CI upper bound vs its twin below +0.03
  G3-recover n1 / nt_c / nt_d close >= 80% of the gap between as built and twin
Also checked: the cruise120 runs agree with the same clips of the test500 runs (clip-derived
seeds, so they must), and each selector's P2 against the as-built set's.

Usage:
  python experiments/evaluation/analyze_l0_select.py
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
O = REPO / "outputs"
K, BOOT = 6, 20000
ARMS = {  # name: (selector run, as-built test500 run, twin)
    "ship": ("l0sel_ship", "dual_u40_v2_ps_test", "swap_ship_L0fullM_500"),
    "n1": ("l0sel_n1", "dual_seed_n1_test", "swap_n1_L0fullM_500"),
    "ntc": ("l0sel_ntc", "dual_nt_c_test", "swap_ntc_L0fullM_500"),
    "ntd": ("l0sel_ntd", "dual_nt_d_test", "swap_ntd_L0fullM_500"),
}


def load_run(exp):
    rows = {}
    for p in sorted((O / exp).glob("*.json")):
        if not p.name.startswith("config"):
            for r in json.loads(p.read_text()):
                rows[r["clip_id"]] = r
    return rows


def main():
    out = O / "l0_redundancy"
    rng = np.random.default_rng(0)
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    base = load_run("baseline_ada_ps_test")
    ids = sorted(base)
    assert len(ids) == 500

    def vec(rows):
        return np.array([min(rows[c]["ade_rollout_k"][:K]) for c in ids])

    def ci(d):
        idx = rng.integers(0, len(d), (BOOT, len(d)))
        lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
        nz = d[d != 0]
        return {"mean": float(d.mean()), "ci": [float(lo), float(hi)], "median": float(np.median(d)),
                "p": float(wilcoxon(nz).pvalue) if len(nz) > 10 else 1.0}

    def f(s):
        return f"{s['mean']:+.4f} [{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}] med {s['median']:+.4f} p={s['p']:.2g}"

    metrics = {"plan": "plans/2026-10-14_l0-functional-redundancy.md (step 3)", "arms": {}}
    say(f"test500, minADE@{K}, Ada; unpruned {vec(base).mean():.4f}")
    means = {}
    for name, (sel, built, twin) in ARMS.items():
        s = json.loads((O / sel / "metrics.json").read_text())
        new_rows = load_run(f"swap_{name}_L0p2_500")
        new, b, t = vec(new_rows), vec(load_run(built)), vec(load_run(twin))
        d_twin, d_built = ci(new - t), ci(new - b)
        gap = b.mean() - t.mean()
        rec = (b.mean() - new.mean()) / gap if abs(gap) > 1e-9 else None
        # cruise120 consistency: the same clips, the same seeds
        c120 = load_run(f"swap_{name}_L0p2_cruise120")
        agree = max(abs(min(c120[c]["ade_rollout_k"][:K]) - min(new_rows[c]["ade_rollout_k"][:K]))
                    for c in c120)
        deg = float(np.mean([new_rows[c]["coc_degenerate"] for c in ids]))
        means[name] = float(new.mean())
        metrics["arms"][name] = {
            "selected_drop": s["selected_drop"], "as_built_drop": s["as_built_drop"],
            "p2_selected": s["selected_fm"], "p2_as_built": s["as_built_fm"],
            "mean": float(new.mean()), "as_built": float(b.mean()), "twin": float(t.mean()),
            "vs_twin": d_twin, "vs_as_built": d_built, "recovered_frac": rec, "degen": deg,
            "cruise120_max_abs_diff_vs_test500": float(agree), "cruise120_n": len(c120)}
        say(f"\n{name}: layer-0 drops {s['as_built_drop']} -> {s['selected_drop']}")
        say(f"  P2 (calib FM loss) as built {s['as_built_fm']:.5f} -> selected {s['selected_fm']:.5f}")
        say(f"  test500 as built {b.mean():.4f} | selected {new.mean():.4f} | twin {t.mean():.4f}; "
            f"degen {deg:.3f}")
        say(f"  selected - twin      {f(d_twin)}")
        say(f"  selected - as built  {f(d_built)}"
            + (f"; closes {rec:.0%} of the as-built - twin gap ({gap:+.4f})" if rec is not None else ""))
        say(f"  cruise120 vs the same clips of test500: max |diff| {agree:.2e} over {len(c120)} clips")
    sd = float(np.std(list(means.values()), ddof=1))
    worst = max(means.values())
    g3 = bool(sd <= 0.05 and worst <= 1.05)
    g3_twin = {n: bool(metrics["arms"][n]["vs_twin"]["ci"][1] < 0.03) for n in ARMS}
    g3_rec = {n: bool((metrics["arms"][n]["recovered_frac"] or 0) >= 0.8) for n in ("n1", "ntc", "ntd")}
    metrics.update({"sd_arm_means": sd, "worst_arm": worst, "G3": g3, "G3_twin": g3_twin,
                    "G3_recover": g3_rec})
    say(f"\nG3: SD of the four arm means {sd:.4f} (<= 0.05), worst {worst:.4f} (<= 1.05) -> "
        f"{'PASS' if g3 else 'FAIL'}")
    say("G3-twin (CI upper < +0.03): " + ", ".join(f"{n} {'PASS' if v else 'FAIL'}" for n, v in g3_twin.items()))
    say("G3-recover (>= 80%): " + ", ".join(f"{n} {'PASS' if v else 'FAIL'}" for n, v in g3_rec.items()))
    (out / "metrics_select.json").write_text(json.dumps(metrics, indent=2))
    (out / "summary_select.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
