"""Does the pruning criterion need the ten diffusion steps scored separately, and does
the layer-35 index-order defect in the shipped `dual` checkpoint matter?

Two questions, one matched-budget family. Every arm removes exactly 2,657,452,032
params (uniform 0.3985632694, VLM only, expert and KV untouched), so only the
within-layer score differs:

  dual_ada  max(rank I_traj, rank I_CoC)               <- reference
  znorm11   mean of 11 within-layer z-scores: CoC NLL + one per diffusion step
  dualfix   dual, with a layer whose half is constant contributing -inf to the max

`dual_ada` exists because `znorm11`'s per-step file was measured on Ada while the
shipped `dual_u40_v2` was built from a Blackwell importance run. Comparing against the
shipped checkpoint would move the importance run and the criterion together, so the
reference is rebuilt from the same Ada file the other two arms use; `dual` (shipped) is
kept in the tables as the second control that says what that rebuild cost.

Gates (plans/2026-08-31_znorm11-criterion.md):
  A1  znorm11 - dual_ada, paired minADE@6: |median| < 0.05 m on all three sets ->
      the step axis is a free refinement of the criterion.
  A2  dualfix - dual_ada, same reading: |median| < 0.05 m -> the shipped checkpoint
      needs no rebuild, and the guard is a correctness fix with no measurable price.

minADE is heavy-tailed, so the median and the Wilcoxon are primary and the mean is
reported beside them. minADE@6 is the frozen protocol: the first six of the eight
stored samples, which are exactly what a 6-sample run would have drawn (seeds base+k).

Usage:
  .venv/bin/python experiments/evaluation/analyze_criterion_agg.py
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
BG, CARD, TEXT, MUTED = "#FAF9F5", "#FFFFFF", "#29261B", "#6B6555"
ACCENT, GOOD, WARN, GREY = "#D97757", "#008300", "#eda100", "#8C8878"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": CARD,
                     "axes.edgecolor": "#E8E6DC", "text.color": TEXT,
                     "axes.labelcolor": TEXT, "xtick.color": MUTED,
                     "ytick.color": MUTED, "font.size": 9, "axes.titlesize": 10})

SETS = ["indist", "test", "oodval"]
SET_LABEL = {"indist": "val500", "test": "test500", "oodval": "OOD-val 262"}
# (directory, keep only the OOD-val rows) -- `dual` was evaluated on the full 1,533-clip
# OOD set and is reduced to the 262 OOD-val clips by its stored `split`, exactly as
# paper_numbers.py does; the other arms ran the ood_val manifest directly
ARMS = {"baseline": ("baseline_ada_ps_{s}", False), "dual": ("dual_u40_v2_ps_{s}", False),
        "dual_ada": ("dual_ada_u40_v2_{s}", False), "znorm11": ("znorm11_u40_v2_{s}", False),
        "dualfix": ("dualfix_u40_v2_{s}", False)}
ARMS["dual"] = ("dual_u40_v2_ps_ood", True)
# the union/operator 2x2 (plans/2026-09-03_union-step-criterion.md). An arm whose rows are
# not on disk yet is dropped per set rather than emptying every intersection, so this runs
# while only part of the extension has landed.
ARMS["maxstep11"] = ("maxstep11_u40_v2_{s}", False)
ARMS["meandual"] = ("meandual_u40_v2_{s}", False)
BOOT = 10000


def load(arm, s):
    pat, val_only = ARMS[arm]
    if arm == "dual" and s != "oodval":
        pat, val_only = "dual_u40_v2_ps_{s}", False
    rows = {}
    d = REPO / "outputs" / pat.format(s=s)
    if not d.is_dir():
        return rows
    for p in sorted(d.glob("*_s*of*.json")):
        for r in json.loads(p.read_text()):
            if val_only and r.get("split") != "val":
                continue
            rows[r["clip_id"]] = r
    return rows


def at_k(r, key, k):
    return float(np.min(np.asarray(r[key], dtype=float)[:k]))


def boot_med(d, seed=0):
    g = np.random.default_rng(seed)
    m = np.median(d[g.integers(0, len(d), (BOOT, len(d)))], axis=1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def kept(name):
    meta = json.loads((REPO / "outputs" / name / "slim_meta.json").read_text())
    q = np.zeros((36, 32), bool)
    m = np.zeros((36, 12288), bool)
    for l, d in enumerate(meta["vlm"]):
        q[l, d["q"]] = True
        m[l, d["mlp"]] = True
    return q, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--out", default="criterion_agg")
    args = ap.parse_args()
    K = args.k
    out = REPO / "outputs" / args.out
    (out / "plots").mkdir(parents=True, exist_ok=True)

    data = {a: {s: load(a, s) for s in SETS} for a in ARMS}
    M = {"k": K, "sets": {}, "paired": {}, "kept": {}}

    # ---- absolute rows -------------------------------------------------------
    print(f"== minADE@{K} / minFDE@{K} mean (median) | CoC degen ==")
    for s in SETS:
        have = [a for a in ARMS if data[a][s]]
        missing = [a for a in ARMS if a not in have]
        ids = sorted(set.intersection(*[set(data[a][s]) for a in have]))
        M["sets"][s] = {"n": len(ids), "missing": missing}
        print(f"-- {SET_LABEL[s]} (n={len(ids)})"
              + (f"  [not run: {', '.join(missing)}]" if missing else ""))
        for a in have:
            ade = np.array([at_k(data[a][s][c], "ade_rollout_k", K) for c in ids])
            fde = np.array([at_k(data[a][s][c], "fde_rollout_k", K) for c in ids])
            dg = np.array([data[a][s][c]["coc_degenerate"] for c in ids], float)
            M["sets"][s][a] = {"ade_mean": ade.mean(), "ade_med": float(np.median(ade)),
                               "fde_mean": fde.mean(), "fde_med": float(np.median(fde)),
                               "degen": dg.mean()}
            print(f"   {a:10s} ADE {ade.mean():.4f} ({np.median(ade):.4f})  "
                  f"FDE {fde.mean():.4f} ({np.median(fde):.4f})  degen {dg.mean():.3f}")

    # ---- paired deltas -------------------------------------------------------
    pairs = [("dual", "baseline"), ("dual_ada", "baseline"), ("znorm11", "baseline"),
             ("dualfix", "baseline"), ("maxstep11", "baseline"), ("meandual", "baseline"),
             ("dual_ada", "dual"), ("znorm11", "dual_ada"), ("dualfix", "dual_ada"),
             ("maxstep11", "dualfix"), ("meandual", "dualfix"), ("znorm11", "dualfix"),
             ("maxstep11", "znorm11")]
    print(f"\n== paired minADE@{K} delta ==")
    for arm, ref in pairs:
        M["paired"][f"{arm}-{ref}"] = {}
        for s in SETS:
            ids = sorted(set(data[arm][s]) & set(data[ref][s]))
            if not ids:
                continue
            d = np.array([at_k(data[arm][s][c], "ade_rollout_k", K)
                          - at_k(data[ref][s][c], "ade_rollout_k", K) for c in ids])
            lo, hi = boot_med(d)
            p = float(stats.wilcoxon(d).pvalue) if np.any(d) else 1.0
            M["paired"][f"{arm}-{ref}"][s] = {"n": len(ids), "med": float(np.median(d)),
                                              "mean": float(d.mean()), "lo": lo, "hi": hi,
                                              "p": p}
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"   {arm+' - '+ref:22s} {SET_LABEL[s]:12s} med {np.median(d):+.4f} "
                  f"mean {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}]{star} p={p:.3g}")

    # ---- how much of the model actually moved --------------------------------
    ref_q, ref_m = kept("slim_dual_ada_u40_v2")
    for name in ("slim_dual_u40_v2", "slim_znorm11_u40_v2", "slim_dualfix_u40_v2",
                 "slim_maxstep11_u40_v2", "slim_meandual_u40_v2"):
        q, m = kept(name)
        aq = (q & ref_q).sum(1) / ref_q.sum(1)
        am = (m & ref_m).sum(1) / ref_m.sum(1)
        diff = [int(l) for l in range(36) if aq[l] < 1 or am[l] < 1]
        M["kept"][name] = {"q_agree": float(aq.mean()), "mlp_agree": float(am.mean()),
                           "q_per_layer": aq.tolist(), "mlp_per_layer": am.tolist(),
                           "layers_differing": diff}
        print(f"\n{name} vs slim_dual_ada_u40_v2: Q {aq.mean():.4f} MLP {am.mean():.4f} "
              f"({len(diff)} layers differ)")

    # identical-output rate: how often the layer-35 guard changes nothing at all
    for arm in ("dualfix", "znorm11", "maxstep11", "meandual"):
        row = {}
        for s in SETS:
            ids = sorted(set(data[arm][s]) & set(data["dual_ada"][s]))
            if not ids:
                continue
            same_ade = sum(at_k(data[arm][s][c], "ade_rollout_k", K)
                           == at_k(data["dual_ada"][s][c], "ade_rollout_k", K) for c in ids)
            same_txt = sum(data[arm][s][c]["gen_coc"] == data["dual_ada"][s][c]["gen_coc"]
                           for c in ids)
            row[s] = {"n": len(ids), "same_ade": same_ade / len(ids),
                      "same_coc": same_txt / len(ids)}
        M[f"identical_{arm}"] = row
        print(f"{arm} vs dual_ada identical: " + "  ".join(
            f"{SET_LABEL[s]} ADE {100*row[s]['same_ade']:.1f}% / CoC {100*row[s]['same_coc']:.1f}%"
            for s in row))

    # ---- plots ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    arms4 = ["dual", "dual_ada", "dualfix", "znorm11", "maxstep11", "meandual"]
    cols = {"dual": GREY, "dual_ada": MUTED, "dualfix": GOOD, "znorm11": ACCENT,
            "maxstep11": "#2F6FBF", "meandual": "#9B59B6"}
    for ax, ref, ttl in zip(axes, ("baseline", "dual_ada"),
                            ("vs unpruned baseline", "vs dual_ada (one factor)")):
        w, xs = 0.14, np.arange(len(SETS))
        shown = [a for a in arms4 if a != ref]
        for i, a in enumerate(shown):
            key = f"{a}-{ref}"
            if key not in M["paired"] or not M["paired"][key]:
                continue
            r = M["paired"][key]
            med = [r[s]["med"] if s in r else np.nan for s in SETS]
            lo = [med[j] - r[SETS[j]]["lo"] if SETS[j] in r else 0 for j in range(3)]
            hi = [r[SETS[j]]["hi"] - med[j] if SETS[j] in r else 0 for j in range(3)]
            ax.bar(xs + (i - (len(shown) - 1) / 2) * w, med, w, color=cols[a], label=a,
                   yerr=[lo, hi], capsize=3, ecolor=MUTED, error_kw={"lw": 1})
        ax.axhline(0, color=TEXT, lw=0.8)
        if ref == "dual_ada":
            ax.axhline(0.05, color=WARN, lw=0.9, ls="--")
            # dualfix is exactly 0.0000 on all three sets, so its bars are invisible
            for j in range(len(SETS)):
                ax.text(xs[j] - w / 2, 0.004, "0.0000", ha="center", fontsize=7.5,
                        color=GOOD, rotation=90)
        ax.set_xticks(xs, [SET_LABEL[s] for s in SETS])
        ax.set_ylabel(f"paired median ΔminADE@{K} (m)")
        ax.set_title(ttl)
        ax.legend(frameon=False, fontsize=8)
    axes[1].text(0.02, 0.93, "dashed = 0.05 m gate", transform=axes[1].transAxes,
                 fontsize=7.5, color=WARN)
    fig.tight_layout()
    fig.savefig(out / "plots" / "deltas.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    for ax, axis in zip(axes, ("q", "mlp")):
        for name, c, lab in (("slim_znorm11_u40_v2", ACCENT, "znorm11"),
                             ("slim_dualfix_u40_v2", GOOD, "dualfix"),
                             ("slim_dual_u40_v2", GREY, "dual (BW importance)")):
            ax.plot(M["kept"][name][f"{axis}_per_layer"], color=c, lw=1.4, label=lab)
        ax.set_xlabel("VLM layer")
        ax.set_ylabel("agreement with dual_ada kept set")
        ax.set_title("Q head" if axis == "q" else "MLP channel")
        ax.set_ylim(0.7, 1.02)
        ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "plots" / "overlap.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    ids = sorted(set(data["dualfix"]["indist"]) & set(data["dual_ada"]["indist"]))
    for ax, arm, c in zip(axes, ("maxstep11", "znorm11"), ("#2F6FBF", ACCENT)):
        d = np.array([at_k(data[arm]["indist"][c_], "ade_rollout_k", K)
                      - at_k(data["dual_ada"]["indist"][c_], "ade_rollout_k", K)
                      for c_ in ids])
        ax.hist(np.clip(d, -1, 1), bins=61, color=c, alpha=0.85)
        ax.axvline(0, color=TEXT, lw=0.8)
        ax.axvline(np.median(d), color=WARN, lw=1.2, ls="--")
        ax.set_yscale("log")
        ax.set_xlabel(f"per-clip ΔminADE@{K} vs dual_ada (m, clipped to ±1)")
        ax.set_ylabel("clips (log)")
        ax.set_title(f"{arm}: median {np.median(d):+.4f}, "
                     f"unchanged {100 * np.mean(d == 0):.1f}%")
    fig.tight_layout()
    fig.savefig(out / "plots" / "delta_dist.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    w, xs = 0.12, np.arange(len(SETS))
    dcol = {"baseline": "#B8B3A3", "dual": GREY, "dual_ada": MUTED, "dualfix": GOOD,
            "znorm11": ACCENT, "maxstep11": "#2F6FBF", "meandual": "#9B59B6"}
    for i, a in enumerate(ARMS):
        v = [100 * M["sets"][s][a]["degen"] if a in M["sets"][s] else np.nan for s in SETS]
        ax.bar(xs + (i - len(ARMS) / 2) * w, v, w, label=a, color=dcol[a])
    ax.set_xticks(xs, [SET_LABEL[s] for s in SETS])
    ax.set_ylabel("CoC degeneracy (%)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "plots" / "degen.png", dpi=150)
    plt.close(fig)

    # the 2x2 itself: operator (max / mean) x arity (2 losses / 11 losses), read as
    # paired median delta against dualfix. Neither factor moves alone; only the corner
    # where both change does, which is what makes znorm11's damage an interaction.
    cells = {("max", "2-way"): "dualfix", ("max", "11-way"): "maxstep11",
             ("mean", "2-way"): "meandual", ("mean", "11-way"): "znorm11"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    ax = axes[0]
    for op, style in (("max", "-o"), ("mean", "--s")):
        ys = []
        for arity in ("2-way", "11-way"):
            arm = cells[(op, arity)]
            r = M["paired"].get(f"{arm}-dualfix", {})
            ys.append(0.0 if arm == "dualfix" else r.get("indist", {}).get("med", np.nan))
        ax.plot([0, 1], ys, style, color=GOOD if op == "max" else ACCENT, lw=1.8,
                label=f"{op} (union)" if op == "max" else f"{op} (average)")
        for x, y in zip((0, 1), ys):
            ax.annotate(f"{y:+.4f}", (x, y), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8)
    ax.axhline(0, color=TEXT, lw=0.8)
    ax.axhline(0.05, color=WARN, lw=0.9, ls=":")
    ax.set_xticks([0, 1], ["2 losses\n(summed traj + CoC)", "11 losses\n(10 FM steps + CoC)"])
    ax.set_ylabel(f"paired median ΔminADE@{K} vs dualfix (m)")
    ax.set_title("val500: only the both-changed corner moves")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    w, xs = 0.35, np.arange(len(SETS))
    for i, arm in enumerate(("maxstep11", "znorm11")):
        r = M["paired"].get(f"{arm}-dualfix", {})
        ys = [r[s]["med"] if s in r else np.nan for s in SETS]
        ax.bar(xs + (i - 0.5) * w, ys, w, color="#2F6FBF" if arm == "maxstep11" else ACCENT,
               label=f"{arm} - dualfix")
    ax.axhline(0, color=TEXT, lw=0.8)
    ax.axhline(0.05, color=WARN, lw=0.9, ls=":")
    ax.set_xticks(xs, [SET_LABEL[s] for s in SETS])
    ax.set_ylabel(f"paired median ΔminADE@{K} (m)")
    ax.set_title("same 11 losses, union vs average")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "plots" / "interaction.png", dpi=150)
    plt.close(fig)

    # where maxstep11's difference lives: nowhere typical, only in the hard tail
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    w, xs = 0.35, np.arange(len(SETS))
    tail = {}
    for j, s in enumerate(SETS):
        a, b = data["maxstep11"][s], data["dualfix"][s]
        ids = sorted(set(a) & set(b))
        if not ids:
            continue
        da = np.array([at_k(a[c], "ade_rollout_k", K) for c in ids])
        db = np.array([at_k(b[c], "ade_rollout_k", K) for c in ids])
        hard = db >= np.quantile(db, 0.9)
        tail[s] = {"easy_mean": float((da - db)[~hard].mean()),
                   "hard_mean": float((da - db)[hard].mean()),
                   "hard_med": float(np.median((da - db)[hard]))}
        ax.bar(xs[j] - w / 2, tail[s]["easy_mean"], w, color=GREY)
        ax.bar(xs[j] + w / 2, tail[s]["hard_mean"], w, color="#2F6FBF")
    M["tail"] = tail
    ax.axhline(0, color=TEXT, lw=0.8)
    ax.set_xticks(xs, [SET_LABEL[s] for s in SETS])
    ax.set_ylabel(f"mean ΔminADE@{K} vs dualfix (m)")
    ax.set_title("maxstep11: easiest 90% (grey) vs hardest 10% (blue)")
    fig.tight_layout()
    fig.savefig(out / "plots" / "tail.png", dpi=150)
    plt.close(fig)

    (out / "metrics.json").write_text(json.dumps(M, indent=2))
    def gate(key):
        r = M["paired"][key]
        return bool(r) and all(abs(v["med"]) < 0.05 for v in r.values())

    g1, g2 = gate("znorm11-dual_ada"), gate("dualfix-dual_ada")
    b1, b2 = gate("maxstep11-dualfix"), gate("meandual-dualfix")
    lines = [f"criterion aggregation, minADE@{K}",
             f"A1 znorm11   == dual_ada : {'PASS' if g1 else 'REJECT'}",
             f"A2 dualfix   == dual_ada : {'PASS' if g2 else 'REJECT'}",
             f"B1 maxstep11 == dualfix  : {'PASS' if b1 else 'REJECT'}",
             f"B2 meandual  == dualfix  : {'PASS' if b2 else 'REJECT'}", ""]
    for arm, ref in pairs:
        for s, r in M["paired"][f"{arm}-{ref}"].items():
            lines.append(f"{arm + ' - ' + ref:22s} {SET_LABEL[s]:12s} n={r['n']:4d} "
                         f"med {r['med']:+.4f} mean {r['mean']:+.4f} "
                         f"[{r['lo']:+.4f},{r['hi']:+.4f}] p={r['p']:.3g}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    (out / "config.json").write_text(json.dumps(
        {"k": K, "arms": ARMS, "sets": SETS, "boot": BOOT}, indent=2))
    print(f"\nA1 (znorm11   == dual_ada): {'PASS' if g1 else 'REJECT'}")
    print(f"A2 (dualfix   == dual_ada): {'PASS' if g2 else 'REJECT'}")
    print(f"B1 (maxstep11 == dualfix) : {'PASS' if b1 else 'REJECT'}")
    print(f"B2 (meandual  == dualfix) : {'PASS' if b2 else 'REJECT'}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
