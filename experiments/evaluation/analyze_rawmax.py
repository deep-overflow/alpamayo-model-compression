"""Raw-value max vs dual, every arm with layer 0's Q heads exempt (plans/2026-10-14_raw-max-criterion.md).

Arms (all `<crit>_u40_l0keep_v2`, same budget, VLM only):
  R    dual, importance_v2           reference; its test500 run is swap_ship_L0fullM_500
  A1   rawmean = max(I_traj/10, I_CoC)                 proposal 1
  A2   rawabsmean = max(mean_s |g_s|, I_CoC)           proposal 2, paired with R'
  R'   dual, importance_v2_ada        A2's reference (same importance files)
  C1   lnorm = max(T / mean_l T, C / mean_l C)         scale removed, shape kept
  C2   dualv3 = dual trunk + I_CoC alone from layer 22 A1's late half on its own
  E1   traj, E2 coc                                    single-criterion endpoints

Step 2 (decision), test500 minADE@6, paired per clip with the arm's reference:
  better      95% CI upper < 0 AND mean <= -0.03 (then val500 must replicate)
  worse       95% CI lower > 0
  equivalent  CI inside +/-0.03 (the post-exemption arm-to-arm floor)
  otherwise   inconclusive
Plus median, Wilcoxon, minFDE@6, per manoeuvre bucket, and CoC degeneracy (exact McNemar).
Step 1 (mechanism, not a gate): held-out 100 indist_500 clips, dense text, FM loss / CoC NLL /
minADE of each mask paired with `dual`'s (run_token_ablation.py --arm-masks).

Usage:
  python experiments/evaluation/analyze_rawmax.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, wilcoxon

REPO = Path(__file__).resolve().parents[2]
O = REPO / "outputs"
BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
C1, C3, C4 = "#2a78d6", "#e87ba4", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})
BOOT, K, MARGIN = 20000, 6, 0.03
REF = {"R": "swap_ship_L0fullM_500", "R'": "dual_ada_u40_l0keep_v2_test"}
# arm -> (reference, label)
ARMS = {"rawmean": ("R", "A1 rawmean"), "rawabsmean": ("R'", "A2 rawabsmean"),
        "lnorm": ("R", "C1 lnorm"), "dualv3": ("R", "C2 dualv3"),
        "traj": ("R", "E1 traj"), "coc": ("R", "E2 coc"), "dual_ada": ("R", "R' dual_ada")}
BASE = "baseline_ada_ps_test"
AS_BUILT = {"dual": "slim_dual_u40_v2", "traj": "slim_traj_u40_v2", "coc": "slim_coc_u40_v2"}
BUCKETS = ["cruise", "decel_stop", "accel", "turn"]
HELDOUT = [f"rawmax_heldout_s{s}" for s in range(4)]


def load_run(exp):
    rows = {}
    p = O / exp
    if not p.exists():
        return rows
    for f in sorted(p.glob("*.json")):
        if not f.name.startswith("config"):
            for r in json.loads(f.read_text()):
                rows[r["clip_id"]] = r
    return rows


def find_as_built(model):
    """The Ada test500 run of an unexempted arm, found through config.json's model field."""
    for cfgp in sorted(O.glob("*/config.json")):
        try:
            c = json.loads(cfgp.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if (str(c.get("model", "")).rstrip("/").endswith(model) and c.get("set") == "test"
                and not c.get("quant") and "Ada" in str(c.get("gpu", ""))
                and c.get("manifest") in (None, "test_500")):
            rows = load_run(cfgp.parent.name)
            if len(rows) == 500 and all(len(r.get("ade_rollout_k", [])) >= K for r in rows.values()):
                return cfgp.parent.name, rows
    return None, None


def main():
    out = O / "rawmax_compare"
    (out / "plots").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    base = load_run(BASE)
    ids = sorted(base)
    assert len(ids) == 500, len(ids)
    bucket = np.array([base[c]["bucket"] for c in ids])

    def vec(rows, key="ade_rollout_k"):
        return np.array([min(rows[c][key][:K]) for c in ids])

    def ci(d):
        idx = rng.integers(0, len(d), (BOOT, len(d)))
        lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
        nz = d[d != 0]
        return {"mean": float(d.mean()), "ci": [float(lo), float(hi)],
                "median": float(np.median(d)),
                "p": float(wilcoxon(nz).pvalue) if len(nz) > 10 else 1.0}

    def verdict(s):
        lo, hi = s["ci"]
        if hi < 0 and s["mean"] <= -MARGIN:
            return "BETTER (needs val500)"
        if lo > 0:
            return "WORSE"
        if lo > -MARGIN and hi < MARGIN:
            return "equivalent"
        return "inconclusive"

    def f(s):
        return f"{s['mean']:+.4f} [{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}] med {s['median']:+.4f} p={s['p']:.2g}"

    def complete(exp):
        rows = load_run(exp)
        return rows if all(c in rows for c in ids) else None

    b = vec(base)
    refs = {k: complete(v) for k, v in REF.items()}
    metrics = {"set": "test500", "n_clips": 500, "margin": MARGIN, "boot": BOOT,
               "unpruned": float(b.mean()), "references": REF, "arms": {}}
    say(f"test500 (n=500, minADE@{K}, Ada): unpruned {b.mean():.4f}")
    for k, r in refs.items():
        if r is not None:
            v = vec(r)
            say(f"reference {k} = {REF[k]}: {v.mean():.4f} (median {np.median(v):.4f}), "
                f"degen {np.mean([r[c]['coc_degenerate'] for c in ids]):.3f}")
    say(f"\n== step 2: arm - its reference, minADE@{K} (gate margin {MARGIN})")
    rows_plot = []
    for arm, (rk, label) in ARMS.items():
        r = complete(f"{arm}_u40_l0keep_v2_test")
        if r is None or refs[rk] is None:
            say(f"{label:16s} not complete yet")
            continue
        a, ref = vec(r), vec(refs[rk])
        d = ci(a - ref)
        dfde = ci(vec(r, "fde_rollout_k") - vec(refs[rk], "fde_rollout_k"))
        deg_a = np.array([r[c]["coc_degenerate"] for c in ids])
        deg_r = np.array([refs[rk][c]["coc_degenerate"] for c in ids])
        n10, n01 = int((deg_a & ~deg_r).sum()), int((~deg_a & deg_r).sum())
        p_mc = binomtest(min(n10, n01), n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
        per_b = {bk: float((a - ref)[bucket == bk].mean()) for bk in BUCKETS}
        rec = {"reference": rk, "mean": float(a.mean()), "median": float(np.median(a)),
               "delta": d, "delta_fde": dfde, "verdict": verdict(d), "vs_unpruned": ci(a - b),
               "degen": float(deg_a.mean()), "degen_ref": float(deg_r.mean()),
               "degen_mcnemar": {"arm_only": n10, "ref_only": n01, "p": float(p_mc)},
               "per_bucket_delta": per_b}
        if arm == "rawabsmean" and refs["R"] is not None:
            rec["delta_vs_R"] = ci(a - vec(refs["R"]))
        metrics["arms"][arm] = rec
        rows_plot.append((label, rk, d))
        say(f"{label:16s} {a.mean():.4f}  vs {rk:2s} {f(d)}  -> {rec['verdict']}")
        say(f"{'':16s} minFDE {f(dfde)}; degen {deg_a.mean():.3f} vs {deg_r.mean():.3f} "
            f"(arm-only {n10}, ref-only {n01}, McNemar p={p_mc:.2g})")
        say(f"{'':16s} per bucket: " + "  ".join(f"{bk} {v:+.3f}" for bk, v in per_b.items()))
        if "delta_vs_R" in rec:
            say(f"{'':16s} vs R (other importance file): {f(rec['delta_vs_R'])}")

    # ---- step 3: val500 replication, same pairing against the exempt shipped arm ----
    vbase, vref = load_run("baseline_ada_ps_indist"), load_run("swap_ship_L0fullM_val500")
    vids = sorted(vbase)
    if len(vids) == 500 and all(c in vref for c in vids):
        def vvec(rows):
            return np.array([min(rows[c]["ade_rollout_k"][:K]) for c in vids])

        r0 = vvec(vref)
        deg_r = np.array([vref[c]["coc_degenerate"] for c in vids])
        say(f"\n== step 3: val500 replication vs swap_ship_L0fullM_val500 ({r0.mean():.4f}, "
            f"degen {deg_r.mean():.3f}; unpruned {vvec(vbase).mean():.4f})")
        for arm, (_, label) in ARMS.items():
            rows = load_run(f"{arm}_u40_l0keep_v2_indist")
            if not all(c in rows for c in vids):
                continue
            a = vvec(rows)
            d = ci(a - r0)
            deg_a = np.array([rows[c]["coc_degenerate"] for c in vids])
            n10, n01 = int((deg_a & ~deg_r).sum()), int((~deg_a & deg_r).sum())
            p_mc = binomtest(min(n10, n01), n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
            metrics.setdefault("val500", {})[arm] = {
                "mean": float(a.mean()), "delta": d, "verdict": verdict(d),
                "degen": float(deg_a.mean()), "degen_ref": float(deg_r.mean()),
                "degen_mcnemar": {"arm_only": n10, "ref_only": n01, "p": float(p_mc)}}
            say(f"{label:16s} {a.mean():.4f}  vs R {f(d)}  -> {verdict(d)}; degen "
                f"{deg_a.mean():.3f} (arm-only {n10}, ref-only {n01}, McNemar p={p_mc:.2g})")

    # what the exemption did to the single-criterion endpoints
    say("\n== as built (layer 0 cut) -> layer 0 kept, test500 mean")
    for arm, model in AS_BUILT.items():
        exp, rows = find_as_built(model)
        new = complete(REF["R"] if arm == "dual" else f"{arm}_u40_l0keep_v2_test")
        if rows is None or new is None:
            continue
        old_v, new_v = vec(rows), vec(new)
        metrics.setdefault("as_built", {})[arm] = {"run": exp, "as_built": float(old_v.mean()),
                                                  "l0keep": float(new_v.mean()),
                                                  "change": ci(new_v - old_v)}
        say(f"{arm:6s} {old_v.mean():.4f} ({exp}) -> {new_v.mean():.4f}   change {f(ci(new_v - old_v))}")

    # ---- step 1: held-out mechanism ----
    shards = [json.loads((O / s / "metrics.json").read_text()) for s in HELDOUT
              if (O / s / "metrics.json").exists()]
    # shards checkpoint as they go: read them only once all 100 clips are in
    if len(shards) == 4 and sum(len(s["per_clip"]["dense"]["fm"]) for s in shards) == 100:
        names = shards[0]["configs"]
        names = [n if isinstance(n, str) else n["name"] for n in names]
        pc = {n: {q: np.concatenate([np.asarray(s["per_clip"][n][q], float) for s in shards])
                  for q in ("fm", "nll", "ade")} for n in names}
        n_clips = len(pc["dense"]["fm"])
        say(f"\n== step 1: held-out, {n_clips} indist_500 clips, dense text (paired with dual)")
        say(f"dense: FM {pc['dense']['fm'].mean():.4f}, NLL {pc['dense']['nll'].mean():.4f}, "
            f"minADE {pc['dense']['ade'].mean():.4f}")
        ho = {}
        for n in names:
            if n == "dense":
                continue
            rec = {}
            for q in ("fm", "nll", "ade"):
                dd = pc[n][q] - pc["dense"][q]
                rec[q] = {"vs_dense": ci(dd), "vs_dense_rel": float(dd.mean() / pc["dense"][q].mean())}
                if n != "arm_dual":
                    rec[q]["vs_dual"] = ci(pc[n][q] - pc["arm_dual"][q])
            ho[n] = rec
            say(f"{n:16s} vs dense: FM {rec['fm']['vs_dense_rel']:+.1%}  NLL {rec['nll']['vs_dense_rel']:+.1%}"
                f"  minADE {rec['ade']['vs_dense']['mean']:+.3f}")
            if n != "arm_dual":
                say(f"{'':16s} vs dual: FM {f(rec['fm']['vs_dual'])}")
                say(f"{'':16s}          NLL {f(rec['nll']['vs_dual'])}")
        metrics["heldout"] = {"n_clips": n_clips, "arms": ho}
        # registered predictions
        def one_sided(x, y):  # H1: x < y
            dd = x - y
            nz = dd[dd != 0]
            return float(wilcoxon(nz, alternative="less").pvalue) if len(nz) > 10 else 1.0
        preds = {}
        for n in ("arm_rawmean", "arm_dualv3"):
            if n in pc:
                preds[f"NLL({n}) < NLL(dual)"] = one_sided(pc[n]["nll"], pc["arm_dual"]["nll"])
        if "arm_rawmean" in pc and "arm_rawabsmean" in pc:
            preds["FM(dual) < FM(rawmean)"] = one_sided(pc["arm_dual"]["fm"], pc["arm_rawmean"]["fm"])
            preds["FM(rawmean) < FM(rawabsmean)"] = one_sided(pc["arm_rawmean"]["fm"],
                                                              pc["arm_rawabsmean"]["fm"])
        metrics["heldout"]["predictions_p_one_sided"] = preds
        say("registered predictions (one-sided Wilcoxon p):")
        for k, v in preds.items():
            say(f"  {k:32s} p = {v:.3g}")
    else:
        say(f"\n== step 1: {len(shards)}/4 held-out shards present")

    # ---- plot ----
    if rows_plot:
        fig, ax = plt.subplots(figsize=(7.2, 0.5 * len(rows_plot) + 1.4))
        ax.axvspan(-MARGIN, MARGIN, color=MUTED, alpha=0.12, lw=0)
        ax.axvline(0, color=MUTED, lw=1)
        for i, (label, rk, d) in enumerate(rows_plot):
            y = len(rows_plot) - 1 - i
            ax.plot(d["ci"], [y, y], color=C1, lw=2, solid_capstyle="round")
            ax.plot(d["mean"], y, "o", color=C1, ms=8, mec=BG, mew=2)
            ax.text(d["ci"][1] + 0.004, y, f"{d['mean']:+.3f}", va="center", color=INK, fontsize=9)
        ax.set_yticks(range(len(rows_plot)))
        ax.set_yticklabels([f"{lab} (vs {rk})" for lab, rk, _ in rows_plot][::-1])
        ax.set_xlabel(f"minADE@{K} difference vs reference, test500 (m), 95% bootstrap CI")
        ax.set_title(f"Layer 0 kept in every arm; shaded band = +/-{MARGIN} m equivalence margin")
        fig.tight_layout()
        fig.savefig(out / "plots" / "rawmax_test500.png", dpi=150)
        plt.close(fig)

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    (out / "config.json").write_text(json.dumps({
        "plan": "plans/2026-10-14_raw-max-criterion.md", "references": REF, "arms": ARMS,
        "base": BASE, "heldout_shards": HELDOUT, "k": K, "boot": BOOT, "margin": MARGIN}, indent=2))


if __name__ == "__main__":
    main()
