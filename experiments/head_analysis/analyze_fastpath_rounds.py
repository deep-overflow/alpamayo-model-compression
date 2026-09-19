"""Graphed fast path on ONE card, arms x rounds: the run-to-run noise floor and the ladder.

plans/2026-09-19_fastpath-single-card-rounds.md. The 2026-09-16 fast-path table came from one
12-clip run per arm, each on whichever Ada card was free, and its e2e ordering of the expert-MLP
ladder is not monotone (em75 976.0 / em87p5 964.1 / em93p75 984.5 ms). Inside a run the clips
barely differ (prompt_len is 3086 for every clip; denoise spread 0.3-1.2%), but the four arms
that share the dual VLM came back with prefill medians 4-5% apart -- a per-run offset, not a
clip-sampling error. `launch_fastpath_rounds.sh` re-measures all five arms on one card, ROUNDS
times, arm order rotated per round (cyclic Latin square). This script reads those runs.

Three levels, kept apart on purpose:
  within a run      median over the live clips (same definition as `analyze_profile_arms.load_fast`,
                    e2e normalised to REF_STEPS CoC tokens)
  across rounds     mean / SD / CV / range of the run-level values -- the noise floor (G4)
  between arms      ratio a/b paired on (round, clip); hierarchical bootstrap (rounds, then clips
                    within round) so the CI carries the run-to-run term and not only clip noise

G1 is a negative control: dual / em75 / em87p5 / em93p75 have the same VLM, so their prefill and
decode/tok must agree. Whatever they disagree by is what this measurement cannot resolve.

Usage:
  python experiments/head_analysis/analyze_fastpath_rounds.py \
      --arms base dual em75 em87p5 em93p75 --tag ada1c --rounds 5 \
      --prior base=fastpipe_base_ada2 dual=fastpipe_dual_ada2 em75=fastpipe_em75_ada2 \
              em87p5=fastpipe_em87p5_ada2 em93p75=fastpipe_em93p75_ada2 \
      [--out outputs/fastpath_rounds]
"""

import argparse
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
C1, C2, C3, C4, C5 = "#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})
BOOT = 10000
STAGES = ["prefill", "tok", "denoise", "e2e"]
LABEL = {"prefill": "ViT+prefill (ms)", "tok": "decode / token (ms)",
         "denoise": "expert denoise (ms)", "e2e": "e2e, normalised (ms)"}


def load_run(name):
    """per-clip stage values of one bench_fastpipeline run, both paths; None if absent."""
    d = REPO / "outputs" / name
    if not (d / "metrics.json").exists():
        return None
    m = json.loads((d / "metrics.json").read_text())
    cfg = json.loads((d / "config.json").read_text())
    ref = m["ref_steps"]
    clips = {}
    for r in m["per_clip"]:
        if r["warmup"] or r.get("denoise_capture"):
            continue
        c = {"fast_steps": r["fast_steps"], "stock_steps": r["stock_steps"]}
        for p, tok in (("fast", r["fast_decode_ms"] / max(r["fast_steps"] - 1, 1)),
                       ("stock", r["stock_decode_ms"] / r["stock_steps"])):
            c[p] = {"prefill": r[f"{p}_prefill_ms"], "tok": tok, "denoise": r[f"{p}_denoise_ms"]}
            c[p]["e2e"] = c[p]["prefill"] + tok * ref + c[p]["denoise"]
        clips[r["clip_id"]] = c
    run = {"name": name, "clips": clips, "ref_steps": ref, "peak_gb": m.get("peak_gb"),
           "gpu": m.get("gpu"), "gpu_index": cfg.get("gpu_index"), "cotenant_max": None}
    for p in ("fast", "stock"):
        med = {s: float(np.median([c[p][s] for c in clips.values()])) for s in STAGES[:3]}
        # run-level e2e is built from the stage medians, as the 2026-09-16 table was
        med["e2e"] = med["prefill"] + med["tok"] * ref + med["denoise"]
        run[p] = med
    ct = d / "cotenant.log"
    if ct.exists():
        ns = [int(ln.split(" n=")[1].split()[0]) for ln in ct.read_text().splitlines() if " n=" in ln]
        run["cotenant_max"] = max(ns) if ns else 0
    return run


def ratio_matrix(runs_a, runs_b, path, stage):
    """a/b per (round, clip), NaN where either side lacks the clip.  -> (R, C)"""
    rounds = sorted(set(runs_a) & set(runs_b))
    ids = sorted(set().union(*[set(runs_a[r]["clips"]) & set(runs_b[r]["clips"]) for r in rounds]))
    M = np.full((len(rounds), len(ids)), np.nan)  # (R, C)
    for i, r in enumerate(rounds):
        ca, cb = runs_a[r]["clips"], runs_b[r]["clips"]
        for j, c in enumerate(ids):
            if c in ca and c in cb:
                M[i, j] = ca[c][path][stage] / cb[c][path][stage]
    return M


def paired(runs_a, runs_b, path, stage, rng):
    """median a/b over (round, clip) pairs; CI resamples rounds, then clips within each round."""
    M = ratio_matrix(runs_a, runs_b, path, stage)  # (R, C)
    R, C = M.shape
    if R < 2 or C < 3:
        return None
    boot = np.empty(BOOT)
    for b in range(BOOT):
        rr = rng.integers(0, R, R)  # (R,)
        cc = rng.integers(0, C, (R, C))  # (R, C)
        boot[b] = np.nanmedian(M[rr[:, None], cc])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    per_round = [float(np.nanmedian(M[i])) for i in range(R)]
    return {"n_rounds": R, "n_clips": C, "median": float(np.nanmedian(M)),
            "lo": float(lo), "hi": float(hi), "per_round": per_round}


def fmt_ratio(p):
    return f"{p['median']:.4f} [{p['lo']:.4f},{p['hi']:.4f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["base", "dual", "em75", "em87p5", "em93p75"])
    ap.add_argument("--tag", default="ada1c")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--baseline", default="base")
    ap.add_argument("--same-vlm", nargs="+", default=["dual", "em75", "em87p5", "em93p75"],
                    help="arms sharing one VLM: the negative control, and the ladder in order")
    ap.add_argument("--prior", nargs="*", default=[], help="name=earlier single fastpipe run")
    ap.add_argument("--out", default="outputs/fastpath_rounds")
    args = ap.parse_args()
    out = REPO / args.out
    (out / "plots").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    arms, base, ladder = args.arms, args.baseline, args.same_vlm

    RUNS = {a: {} for a in arms}  # arm -> round -> run
    for a in arms:
        for r in range(args.rounds):
            run = load_run(f"fastpipe_{a}_{args.tag}_r{r}")
            if run:
                RUNS[a][r] = run
    prior = {}
    for it in args.prior:
        k, v = it.split("=", 1)
        prior[k] = load_run(v)

    # ---- across rounds -------------------------------------------------------------------
    ACROSS = {}
    for a in arms:
        ACROSS[a] = {}
        for p in ("fast", "stock"):
            for s in STAGES:
                v = np.array([RUNS[a][r][p][s] for r in sorted(RUNS[a])])  # (R,)
                sd = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
                ACROSS[a][f"{p}_{s}"] = {
                    "values": v.tolist(), "mean": float(v.mean()), "sd": sd,
                    "cv_pct": 100 * sd / float(v.mean()), "min": float(v.min()),
                    "max": float(v.max())}
        ACROSS[a]["peak_gb"] = [RUNS[a][r]["peak_gb"] for r in sorted(RUNS[a])]

    # position / round effects: deviation of each run's fast e2e from its arm mean
    pos_dev, round_dev = {}, {}
    for i, a in enumerate(arms):
        mu = ACROSS[a]["fast_e2e"]["mean"]
        for r, run in RUNS[a].items():
            d = 100 * (run["fast"]["e2e"] / mu - 1)
            pos_dev.setdefault((i - r) % len(arms), []).append(d)
            round_dev.setdefault(r, []).append(d)
    drift = {"by_position_pct": {k: float(np.mean(v)) for k, v in sorted(pos_dev.items())},
             "by_round_pct": {k: float(np.mean(v)) for k, v in sorted(round_dev.items())}}

    # ---- between arms --------------------------------------------------------------------
    PAIRS = {}

    def pair(a, b, path, stage):
        key = f"{a}/{b}:{path}_{stage}"
        if key not in PAIRS and RUNS.get(a) and RUNS.get(b):
            PAIRS[key] = paired(RUNS[a], RUNS[b], path, stage, rng)
        return PAIRS.get(key)

    for a in arms:
        if a != base:
            for s in STAGES:
                pair(a, base, "fast", s)
            pair(a, base, "stock", "e2e")
    adjacent = list(zip(ladder[1:], ladder[:-1]))  # (em75, dual), (em87p5, em75), ...
    for a, b in adjacent:
        for s in STAGES:
            pair(a, b, "fast", s)
    control = list(itertools.combinations(ladder, 2))
    for b, a in control:
        for s in ("prefill", "tok"):
            pair(a, b, "fast", s)

    # ---- gates ---------------------------------------------------------------------------
    allruns = [run for a in arms for run in RUNS[a].values()]
    g0 = {"runs_found": len(allruns), "runs_expected": len(arms) * args.rounds,
          "gpu_index": sorted({str(r["gpu_index"]) for r in allruns}),
          "gpu": sorted({str(r["gpu"]) for r in allruns}),
          "min_live_clips": min(len(r["clips"]) for r in allruns),
          "cotenant_max": max((r["cotenant_max"] or 0) for r in allruns),
          "cotenant_missing": sum(r["cotenant_max"] is None for r in allruns),
          "live_set_stable": {}, "fast_steps_stable": {}, "peak_gb_range": {}}
    for a in arms:
        rs = [RUNS[a][r] for r in sorted(RUNS[a])]
        g0["live_set_stable"][a] = len({frozenset(r["clips"]) for r in rs}) == 1
        shared = set.intersection(*[set(r["clips"]) for r in rs])
        g0["fast_steps_stable"][a] = all(
            len({r["clips"][c]["fast_steps"] for r in rs}) == 1 for c in shared)
        pk = [r["peak_gb"] for r in rs]
        g0["peak_gb_range"][a] = float(max(pk) - min(pk))
    g0["pass"] = (g0["runs_found"] == g0["runs_expected"] and len(g0["gpu_index"]) == 1
                  and len(g0["gpu"]) == 1 and g0["min_live_clips"] >= 22
                  and g0["cotenant_max"] <= 1 and g0["cotenant_missing"] == 0
                  and all(g0["live_set_stable"].values()) and all(g0["fast_steps_stable"].values())
                  and all(v <= 0.01 for v in g0["peak_gb_range"].values()))

    g1 = {"pairs": {}, "verdict": "PASS"}
    for b, a in control:
        for s in ("prefill", "tok"):
            p = pair(a, b, "fast", s)
            off = abs(p["median"] - 1)
            has1 = p["lo"] <= 1 <= p["hi"]
            v = "PASS" if off < 0.01 and has1 else ("FAIL" if off > 0.02 and not has1 else "MARGINAL")
            g1["pairs"][f"{a}/{b}:{s}"] = {**p, "verdict": v}
            if v == "FAIL" or (v == "MARGINAL" and g1["verdict"] == "PASS"):
                g1["verdict"] = v
    g1["max_abs_offset_pct"] = 100 * max(abs(p["median"] - 1) for p in g1["pairs"].values())

    g2 = {f"{a}/{b}": pair(a, b, "fast", "denoise") for a, b in adjacent}
    g2_pass = all(p["hi"] < 1 for p in g2.values())
    top = ladder[-1]
    g3 = {f"{top}/{base}": pair(top, base, "fast", "e2e")}
    for a, b in adjacent:
        g3[f"{a}/{b}"] = pair(a, b, "fast", "e2e")
    for p in g3.values():
        p["resolved"] = not (p["lo"] <= 1 <= p["hi"])
    g4 = {a: ACROSS[a]["fast_e2e"]["cv_pct"] for a in arms}
    gates = {"G0": g0, "G1": g1, "G2": {"pairs": g2, "pass": g2_pass}, "G3": g3,
             "G4": {"cv_pct": g4, "pass": all(v < 1 for v in g4.values())}}

    metrics = {"across_rounds": ACROSS, "pairs": PAIRS, "drift": drift, "gates": gates,
               "prior": {k: {"fast": v["fast"], "stock": v["stock"], "peak_gb": v["peak_gb"],
                             "n": len(v["clips"])} for k, v in prior.items() if v}}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "config.json").write_text(json.dumps({
        "arms": arms, "tag": args.tag, "rounds": args.rounds, "baseline": base,
        "same_vlm": ladder, "prior": dict(it.split("=", 1) for it in args.prior), "boot": BOOT,
        "runs": {a: [RUNS[a][r]["name"] for r in sorted(RUNS[a])] for a in arms}}, indent=2))

    # ---- summary -------------------------------------------------------------------------
    gpu = f"{g0['gpu'][0]} (index {g0['gpu_index'][0]})" if allruns else "?"
    head = (f"== graphed fast path, one card: {gpu}; {args.rounds} rounds x {len(arms)} arms, "
            f"run-level medians over the live clips, e2e normalised to "
            f"{allruns[0]['ref_steps']} CoC tokens ==")
    cols_hdr = (f"{'arm':8s} {'rounds':>6s} {'clips':>5s} {'prefill':>16s} {'dec/tok':>15s} "
                f"{'denoise':>15s} {'e2e fast':>16s} {'CV%':>5s} {'speedup':>8s} {'peak GiB':>9s}")
    L = [head, cols_hdr]
    bmu = ACROSS[base]["fast_e2e"]["mean"]

    def ms(A, k, w=1):
        return f"{A[k]['mean']:.{w}f}+-{A[k]['sd']:.{w}f}"

    for a in arms:
        A = ACROSS[a]
        L.append(f"{a:8s} {len(RUNS[a]):6d} {min(len(r['clips']) for r in RUNS[a].values()):5d} "
                 f"{ms(A, 'fast_prefill'):>16s} {ms(A, 'fast_tok', 2):>15s} "
                 f"{ms(A, 'fast_denoise'):>15s} {ms(A, 'fast_e2e'):>16s} "
                 f"{A['fast_e2e']['cv_pct']:5.2f} "
                 f"{bmu / A['fast_e2e']['mean']:7.3f}x {np.mean(A['peak_gb']):9.2f}")
    L += ["", "per-round e2e fast (ms), round 0.." + str(args.rounds - 1)]
    for a in arms:
        L.append(f"  {a:8s} " + "  ".join(f"{v:7.1f}" for v in ACROSS[a]["fast_e2e"]["values"]))
    L += ["", "stock path in the same runs (e2e, mean+-sd over rounds)"]
    for a in arms:
        A = ACROSS[a]["stock_e2e"]
        L.append(f"  {a:8s} {A['mean']:7.1f}+-{A['sd']:.1f}  CV {A['cv_pct']:.2f}%")
    if metrics["prior"]:
        L += ["", "against the 2026-09-16 single runs (one 12-clip run per arm, mixed cards)",
              f"{'arm':8s} {'prefill old/new':>18s} {'denoise old/new':>18s} {'e2e old/new':>18s}"]
        for a in arms:
            if a in metrics["prior"]:
                o, n = metrics["prior"][a]["fast"], ACROSS[a]
                L.append(f"{a:8s} {o['prefill']:8.1f}/{n['fast_prefill']['mean']:8.1f} "
                         f"{o['denoise']:9.1f}/{n['fast_denoise']['mean']:8.1f} "
                         f"{o['e2e']:9.1f}/{n['fast_e2e']['mean']:8.1f}")
    L += ["", "== paired ratios, median over (round, clip), 95% hierarchical bootstrap =="]
    for a in arms:
        if a != base:
            L.append(f"  {a + '/' + base:16s} " + "  ".join(
                f"{s} {fmt_ratio(pair(a, base, 'fast', s))}" for s in STAGES))
    for a, b in adjacent:
        L.append(f"  {a + '/' + b:16s} " + "  ".join(
            f"{s} {fmt_ratio(pair(a, b, 'fast', s))}" for s in STAGES))
    L += ["", (f"== G1 negative control (same VLM): {g1['verdict']}, "
               f"largest offset {g1['max_abs_offset_pct']:.2f}% ==")]
    for k, p in g1["pairs"].items():
        L.append(f"  {k:24s} {fmt_ratio(p)}  {p['verdict']}")
    L += ["", ("drift, mean deviation of e2e fast from the arm mean (%): by position in the round "
               + " ".join(f"{k}:{v:+.2f}" for k, v in drift["by_position_pct"].items())
               + "   by round "
               + " ".join(f"{k}:{v:+.2f}" for k, v in drift["by_round_pct"].items()))]
    L += ["", (f"G0 {'PASS' if g0['pass'] else 'FAIL'}: {g0['runs_found']}/{g0['runs_expected']} "
               f"runs, gpu_index {g0['gpu_index']}, min live clips {g0['min_live_clips']}, "
               f"max concurrent compute procs {g0['cotenant_max']}, "
               f"peak range {max(g0['peak_gb_range'].values()):.3f} GiB"),
          (f"G2 {'PASS' if g2_pass else 'FAIL'} (denoise ladder): "
           + "  ".join(f"{k} {fmt_ratio(p)}" for k, p in g2.items())),
          "G3 (e2e): " + "  ".join(
              f"{k} {fmt_ratio(p)} {'resolved' if p['resolved'] else 'not resolved'}"
              for k, p in g3.items()),
          (f"G4 {'PASS' if gates['G4']['pass'] else 'FAIL'} (run-to-run CV of e2e fast, %): "
           + "  ".join(f"{a} {v:.2f}" for a, v in g4.items()))]
    text = "\n".join(L)
    (out / "summary.txt").write_text(text + "\n")
    print(text)

    # ---- plots ---------------------------------------------------------------------------
    cols = dict(zip(arms, [MUTED, C1, C2, C4, C3]))
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, s in zip(axes, STAGES):
        for i, a in enumerate(arms):
            v = ACROSS[a][f"fast_{s}"]["values"]
            ax.scatter(i + np.linspace(-0.18, 0.18, len(v)), v, s=22, color=cols[a], zorder=3)
            ax.hlines(np.mean(v), i - 0.3, i + 0.3, color=cols[a], lw=1.5)
            if a in metrics["prior"]:
                ax.scatter(i, metrics["prior"][a]["fast"][s], marker="x", s=40, color=INK, zorder=4)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels(arms, rotation=30, ha="right")
        ax.set_title(LABEL[s])
    axes[0].scatter([], [], marker="x", color=INK, label="2026-09-16 single run")
    axes[0].scatter([], [], s=22, color=MUTED, label="one round (this run)")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"graphed fast path, one card ({gpu}): run-level medians per round", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "plots" / "rounds_stages.png", dpi=150)
    plt.close(fig)

    def forest(ax, rows, title, band=None):
        for y, (lab, p, col) in enumerate(rows[::-1]):
            ax.plot([p["lo"], p["hi"]], [y, y], color=col, lw=2)
            ax.scatter(p["per_round"], [y] * len(p["per_round"]), s=10, color=col, alpha=0.45)
            ax.scatter([p["median"]], [y], s=34, color=col, zorder=3, edgecolor=BG)
        ax.axvline(1, color=MUTED, lw=1, ls="--")
        if band:
            ax.axvspan(1 - band, 1 + band, color=MUTED, alpha=0.12)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=8)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.set_title(title)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    forest(axes[0], [(f"{a}/{b}", pair(a, b, "fast", "denoise"), C3) for a, b in adjacent],
           "expert denoise, adjacent ladder pairs (ratio)")
    forest(axes[1], [(f"{a}/{b}", pair(a, b, "fast", "e2e"), C1) for a, b in adjacent]
           + [(f"{a}/{base}", pair(a, base, "fast", "e2e"), C2) for a in arms if a != base],
           "e2e fast (ratio); small dots = per-round medians")
    fig.tight_layout()
    fig.savefig(out / "plots" / "ladder_ratios.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, s in zip(axes, ("prefill", "tok")):
        forest(ax, [(f"{a}/{b}", g1["pairs"][f"{a}/{b}:{s}"], C1) for b, a in control],
               f"same-VLM control: {LABEL[s].split(' (')[0]} ratio (band +-1%)", band=0.01)
    fig.tight_layout()
    fig.savefig(out / "plots" / "negative_control.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
