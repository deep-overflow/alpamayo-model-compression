"""Does the GPU architecture change what pruning costs?

Every published arm here was measured on Ada, and the rule since 2026-08-07 has been
that paired comparisons are only valid within one architecture -- the same clip and seed
gave 0.286 on Ada and 0.291 on Blackwell, and 3-4% of clips produce different CoC text
across architectures. That rule was inferred from a handful of clips and then applied as
a blanket caution; it was never measured at the scale a result is reported at.

The NEURON transfer produced the missing measurement. `baseline` and `dual_u40_v2` were
re-run on A100 over the same three sets, the same manifests and the same clip-derived
seeds, so this compares two complete evaluations rather than two clips.

Three questions, in the order they matter:

  Q1  Is the PRUNING COST architecture-dependent? dual - baseline is paired within each
      architecture, so a kernel difference that shifts both arms equally cancels. This is
      the number every result in this repo is actually built on.
  Q2  Do the ABSOLUTE numbers agree? They have no right to be paired across machines,
      but if the 500-clip means agree anyway, the blanket caution is stronger than it
      needs to be.
  Q3  Where does the architecture actually show up? Per-clip, and in the discrete
      metrics (CoC degeneracy) rather than the continuous ones.

Usage:
  .venv/bin/python experiments/evaluation/analyze_arch.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
BG, CARD, TEXT, MUTED = "#FAF9F5", "#FFFFFF", "#29261B", "#6B6555"
ACCENT, GOOD, WARN, BLUE = "#D97757", "#008300", "#eda100", "#2F6FBF"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": CARD,
                     "axes.edgecolor": "#E8E6DC", "text.color": TEXT,
                     "axes.labelcolor": TEXT, "xtick.color": MUTED,
                     "ytick.color": MUTED, "font.size": 9, "axes.titlesize": 10})
K, BOOT = 6, 10000
SETS = [("indist", "val500"), ("test", "test500"), ("oodval", "OOD-val")]
# (directory, keep only OOD-val rows) -- Ada's dual ran the full 1,533-clip OOD set and
# is reduced by its stored `split`, exactly as paper_numbers.py does
ADA = {("baseline", "indist"): ("baseline_ada_ps_indist", False),
       ("baseline", "test"): ("baseline_ada_ps_test", False),
       ("baseline", "oodval"): ("baseline_ada_ps_oodval", False),
       ("dual", "indist"): ("dual_u40_v2_ps_indist", False),
       ("dual", "test"): ("dual_u40_v2_ps_test", False),
       ("dual", "oodval"): ("dual_u40_v2_ps_ood", True)}


def rows(d, val_only=False):
    out = {}
    p = REPO / "outputs" / d
    for f in sorted(p.glob("*_s*of*.json")):
        for r in json.loads(f.read_text()):
            if val_only and r.get("split") != "val":
                continue
            out[r["clip_id"]] = r
    return out


def at6(r, key="ade_rollout_k"):
    return float(np.min(np.asarray(r[key], float)[:K]))


def boot(x, fn=np.median, seed=0):
    g = np.random.default_rng(seed)
    return tuple(np.percentile(fn(x[g.integers(0, len(x), (BOOT, len(x)))], axis=1),
                               [2.5, 97.5]))


def main():
    out = REPO / "outputs" / "arch_invariance"
    (out / "plots").mkdir(parents=True, exist_ok=True)
    data, M = {}, {"k": K, "sets": {}, "delta": {}, "perclip": {}}
    for arm in ("baseline", "dual"):
        for s, _ in SETS:
            data[("ada", arm, s)] = rows(*ADA[(arm, s)])
            data[("a100", arm, s)] = rows(f"neuron_pull/{arm}_neuron_{s}")

    print(f"{'set':10s} {'arch':6s} {'arm':9s} {'minADE@6':>18s} {'degen':>7s} {'n':>5s}")
    for s, lab in SETS:
        M["sets"][s] = {}
        ids = sorted(set.intersection(*[set(data[(a, m, s)])
                                        for a in ("ada", "a100") for m in ("baseline", "dual")]))
        M["sets"][s]["n"] = len(ids)
        for arch in ("ada", "a100"):
            for arm in ("baseline", "dual"):
                A = data[(arch, arm, s)]
                v = np.array([at6(A[c]) for c in ids])
                dg = float(np.mean([A[c]["coc_degenerate"] for c in ids]))
                M["sets"][s][f"{arch}_{arm}"] = {"mean": float(v.mean()),
                                                 "med": float(np.median(v)), "degen": dg}
                print(f"{lab:10s} {arch:6s} {arm:9s} {v.mean():9.4f} ({np.median(v):.4f}) "
                      f"{100*dg:6.1f}% {len(ids):5d}")

    # Q1 -- the pruning cost, paired inside each architecture
    print("\n== Q1 dual - baseline, paired within each architecture ==")
    print(f"{'set':10s} {'Ada med':>10s} {'A100 med':>10s} {'diff':>9s} {'A100 95% CI':>22s}")
    for s, lab in SETS:
        ids = sorted(set.intersection(*[set(data[(a, m, s)])
                                        for a in ("ada", "a100") for m in ("baseline", "dual")]))
        d = {}
        for arch in ("ada", "a100"):
            d[arch] = np.array([at6(data[(arch, "dual", s)][c])
                                - at6(data[(arch, "baseline", s)][c]) for c in ids])
        lo, hi = boot(d["a100"])
        inside = lo <= np.median(d["ada"]) <= hi
        M["delta"][s] = {"ada_med": float(np.median(d["ada"])), "a100_med": float(np.median(d["a100"])),
                         "ada_mean": float(d["ada"].mean()), "a100_mean": float(d["a100"].mean()),
                         "lo": lo, "hi": hi, "ada_inside_a100_ci": bool(inside),
                         "p_a100": float(stats.wilcoxon(d["a100"]).pvalue)}
        print(f"{lab:10s} {np.median(d['ada']):+10.4f} {np.median(d['a100']):+10.4f} "
              f"{np.median(d['a100']) - np.median(d['ada']):+9.4f}  [{lo:+.4f},{hi:+.4f}]"
              f"  Ada inside: {inside}")

    # Q3 -- per-clip, where the architecture is not invisible at all
    print("\n== Q3 per-clip A100 - Ada (same clip, same seed, different card) ==")
    print(f"{'set':10s} {'arm':9s} {'med |Δ|':>9s} {'mean Δ':>9s} {'p':>9s} {'CoC 동일':>9s} {'|Δ|>0.05':>9s}")
    for s, lab in SETS:
        M["perclip"][s] = {}
        for arm in ("baseline", "dual"):
            ids = sorted(set(data[("ada", arm, s)]) & set(data[("a100", arm, s)]))
            a = np.array([at6(data[("ada", arm, s)][c]) for c in ids])
            b = np.array([at6(data[("a100", arm, s)][c]) for c in ids])
            same = np.mean([data[("ada", arm, s)][c]["gen_coc"]
                            == data[("a100", arm, s)][c]["gen_coc"] for c in ids])
            d = b - a
            M["perclip"][s][arm] = {"med_abs": float(np.median(np.abs(d))),
                                    "mean": float(d.mean()), "same_coc": float(same),
                                    "frac_big": float(np.mean(np.abs(d) > 0.05)),
                                    "p": float(stats.wilcoxon(d).pvalue) if np.any(d) else 1.0}
            r = M["perclip"][s][arm]
            print(f"{lab:10s} {arm:9s} {r['med_abs']:9.4f} {r['mean']:+9.4f} {r['p']:9.3g} "
                  f"{100*same:8.1f}% {100*r['frac_big']:8.1f}%")

    # ---- plots ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    ax, w, xs = axes[0], 0.35, np.arange(len(SETS))
    for i, (arch, col, lab) in enumerate((("ada", MUTED, "Ada (RTX 5880)"),
                                          ("a100", BLUE, "A100 (NEURON)"))):
        med = [M["delta"][s][f"{arch}_med"] for s, _ in SETS]
        err = None
        if arch == "a100":
            err = [[med[j] - M["delta"][SETS[j][0]]["lo"] for j in range(3)],
                   [M["delta"][SETS[j][0]]["hi"] - med[j] for j in range(3)]]
        ax.bar(xs + (i - 0.5) * w, med, w, color=col, label=lab, yerr=err, capsize=3,
               ecolor=TEXT, error_kw={"lw": 1})
    ax.axhline(0, color=TEXT, lw=0.8)
    ax.set_xticks(xs, [l for _, l in SETS])
    ax.set_ylabel(f"paired median ΔminADE@{K} (m)")
    ax.set_title("Q1  pruning cost (dual − baseline), paired within each card")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    for i, arm in enumerate(("baseline", "dual")):
        for j, (arch, col) in enumerate((("ada", MUTED), ("a100", BLUE))):
            v = [M["sets"][s][f"{arch}_{arm}"]["mean"] for s, _ in SETS]
            ax.bar(xs + (2 * i + j - 1.5) * 0.2, v, 0.2, color=col,
                   alpha=1.0 if arm == "dual" else 0.55,
                   label=f"{arm} {arch}" if True else None)
    ax.set_xticks(xs, [l for _, l in SETS])
    ax.set_ylabel(f"minADE@{K} mean (m)")
    ax.set_title("Q2  absolute values (light = baseline, solid = dual)")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "plots" / "deltas.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    ids = sorted(set(data[("ada", "baseline", "indist")]) & set(data[("a100", "baseline", "indist")]))
    a = np.array([at6(data[("ada", "baseline", "indist")][c]) for c in ids])
    b = np.array([at6(data[("a100", "baseline", "indist")][c]) for c in ids])
    ax = axes[0]
    ax.scatter(a, b, s=7, alpha=0.45, color=BLUE, edgecolors="none")
    lim = [0, max(a.max(), b.max()) * 1.02]
    ax.plot(lim, lim, color=TEXT, lw=0.8, ls="--")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(f"Ada minADE@{K} (m)"); ax.set_ylabel(f"A100 minADE@{K} (m)")
    ax.set_title(f"Q3  val500 baseline, per clip (r={np.corrcoef(a, b)[0, 1]:.4f})")
    ax = axes[1]
    d = b - a
    ax.hist(np.clip(d, -0.5, 0.5), bins=61, color=BLUE, alpha=0.85)
    ax.axvline(0, color=TEXT, lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel(f"per-clip A100 − Ada ΔminADE@{K} (m, clipped to ±0.5)")
    ax.set_ylabel("clips (log)")
    ax.set_title(f"identical on {100 * np.mean(d == 0):.1f}% of clips, "
                 f"|Δ|>0.05 on {100 * np.mean(np.abs(d) > 0.05):.1f}%")
    fig.tight_layout()
    fig.savefig(out / "plots" / "perclip.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    for i, arm in enumerate(("baseline", "dual")):
        for j, (arch, col) in enumerate((("ada", MUTED), ("a100", BLUE))):
            v = [100 * M["sets"][s][f"{arch}_{arm}"]["degen"] for s, _ in SETS]
            ax.bar(xs + (2 * i + j - 1.5) * 0.2, v, 0.2, color=col,
                   alpha=1.0 if arm == "dual" else 0.55, label=f"{arm} {arch}")
    ax.set_xticks(xs, [l for _, l in SETS])
    ax.set_ylabel("CoC degeneracy (%)")
    ax.set_title("the discrete metric does move")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "plots" / "degen.png", dpi=150)
    plt.close(fig)

    (out / "metrics.json").write_text(json.dumps(M, indent=2))
    ok = all(M["delta"][s]["ada_inside_a100_ci"] for s, _ in SETS)
    (out / "summary.txt").write_text(
        f"architecture invariance of the pruning cost, minADE@{K}\n"
        f"Ada median inside the A100 bootstrap CI on all three sets: {ok}\n"
        + "\n".join(f"{l:10s} Ada {M['delta'][s]['ada_med']:+.4f}  "
                    f"A100 {M['delta'][s]['a100_med']:+.4f}  "
                    f"[{M['delta'][s]['lo']:+.4f},{M['delta'][s]['hi']:+.4f}]"
                    for s, l in SETS) + "\n")
    print(f"\nAda median inside the A100 CI on all three sets: {ok}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
