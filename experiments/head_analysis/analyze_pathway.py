"""Analyze the Stage-1 pathway map: expert <- cache-span knockout.

Merges shards, computes clip-paired deltas against the unblocked config, and judges the
four pre-registered gates from plans/2026-08-25_pathway-map.md.

The size-matched random controls are the point of the design: for a large span (vision is
93% of the prompt) a random block of the same size hits mostly the same tokens, so the
control is only informative for the small spans -- which is exactly where the interesting
claim (CoC) lives.

Usage:
  python analyze_pathway.py --shards pathway_x_s0 pathway_x_s13 pathway_x_s26 pathway_x_s38 \
      --out pathway_x_v1 --outputs-root /home/.../outputs
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr, wilcoxon  # noqa: E402

import eval_lib as el  # noqa: E402

BG, INK, MUTED, GRID = "#FAF9F5", "#29261B", "#6B6555", "#E8E6DC"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "font.size": 9, "axes.grid": True, "axes.axisbelow": True,
})

BASE = "X0_none"
SPAN_OF = {"X1_vision": "vision", "X2_hist": "traj_history", "X3_coc": "generated_coc",
           "X4_text": "prompt_text", "X5_sink": "sink_pos0"}
# load_physical_aiavdataset sorts cameras by index, and the default feature set is
# indices {0,1,2,6}, so prompt order is fixed.
CAM_REAL = {"X1c0_cam0": "cross_left_120", "X1c1_cam1": "front_wide_120",
            "X1c2_cam2": "cross_right_120", "X1c3_cam3": "front_tele_30"}


def merge(shard_dirs):
    cfgs, meta, per_clip, buckets, clips, nll = None, None, {}, [], [], []
    for d in shard_dirs:
        m = json.loads((d / "metrics.json").read_text())
        if cfgs is None:
            cfgs, meta = m["configs"], m["meta"]
            per_clip = {c: {"ade": [], "fde": []} for c in cfgs}
        elif m["configs"] != cfgs:
            raise SystemExit(f"config mismatch in {d}")
        for c in cfgs:
            per_clip[c]["ade"] += m["per_clip"][c]["ade"]
            per_clip[c]["fde"] += m["per_clip"][c]["fde"]
        buckets += m["buckets"]
        clips += m["clip_ids"]
        nll += m["coc_nll"]
    return cfgs, meta, per_clip, buckets, clips, nll


def median_ci(d, n_boot=10000, seed=0, alpha=0.05):
    """Bootstrap CI for the median of a paired difference vector."""
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(d), size=(n_boot, len(d)))
    boots = np.median(d[idx], axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.median(d)), float(lo), float(hi)


def paired(a, b):
    """a - b as a paired vector.

    minADE deltas are heavy-tailed here (per-clip baselines span 0.21 - 7.2 m, and a
    perturbation can land closer to GT on a clip the baseline already failed), so the
    median and its bootstrap CI are the primary read and the mean is reported beside
    it -- the convention this repo settled on in the 2026-07-29 grid.
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    mean, mlo, mhi = el.paired_bootstrap_ci(d)
    med, lo, hi = median_ci(d)
    try:
        p = float(wilcoxon(d).pvalue)
    except ValueError:
        p = float("nan")
    return d, med, lo, hi, mean, mlo, mhi, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", default="pathway_x_v1")
    ap.add_argument("--outputs-root", required=True)
    args = ap.parse_args()

    root = Path(args.outputs_root)
    cfgs, meta, per_clip, buckets, clips, nll = merge([root / s for s in args.shards])
    out_dir = root / args.out
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    n = len(clips)
    base = per_clip[BASE]["ade"]

    rows, L = {}, []
    L.append(f"Pathway map Stage 1 -- expert <- cache-span knockout   (n={n} clips, K=8)")
    L.append(f"unblocked minADE: mean {np.mean(base):.4f}  median {np.median(base):.4f}")
    L.append(f"CoC NLL (identical across configs by construction): mean {np.mean(nll):.4f}")
    L.append("")
    L.append("primary = paired median delta with bootstrap CI; mean shown beside it")
    L.append(f"{'config':22s} {'span':22s} {'n_tok':>6s} {'medADE':>8s} "
             f"{'med d':>9s} {'95% CI (median)':>20s} {'mean d':>9s} {'p':>9s} {'worse':>7s}")
    for c in cfgs:
        d, med, lo, hi, mean, mlo, mhi, p = paired(per_clip[c]["ade"], base)
        rows[c] = {"med": med, "lo": lo, "hi": hi, "mean": mean, "mlo": mlo, "mhi": mhi,
                   "p": p, "n_tok": meta[c]["n"], "span": meta[c]["span"],
                   "abs_med": float(np.median(per_clip[c]["ade"])),
                   "abs_mean": float(np.mean(per_clip[c]["ade"])),
                   "worse": int((d > 0).sum()),
                   "sig": bool((lo > 0 or hi < 0) and p < 0.05)}
        r = rows[c]
        L.append(f"{c:22s} {r['span']:22s} {r['n_tok']:6d} {r['abs_med']:8.4f} "
                 f"{med:+9.4f} [{lo:+8.4f},{hi:+8.4f}] {mean:+9.4f} {p:9.2e} "
                 f"{r['worse']:3d}/{n}")

    # ---- G0: span vs size-matched random control -------------------------------
    # A size-matched random draw is only a meaningful null for a span that is a small
    # fraction of the cache. vision is ~93% of the prompt, so "2,880 random positions"
    # blocks essentially everything and is strictly more destructive than vision itself
    # -- degenerate, not evidence. Those spans are reported but excluded from the gate.
    seq_len = meta["X1_vision"]["n"] + meta["X6_all_but_vision"]["n"]
    L.append("")
    L.append(f"G0  span block vs size-matched random block (cache length {seq_len})")
    L.append("    positive = the span is worse than an equal number of arbitrary positions")
    g0 = {}
    for c in [k for k in SPAN_OF if f"{k}_rand" in per_clip]:
        frac = meta[c]["n"] / seq_len
        d, med, lo, hi, mean, mlo, mhi, p = paired(per_clip[c]["ade"],
                                                   per_clip[f"{c}_rand"]["ade"])
        degenerate = frac > 0.5
        sig = bool(lo > 0 and p < 0.05)
        g0[c] = {"med": med, "lo": lo, "hi": hi, "mean": mean, "p": p, "frac": frac,
                 "sig": sig, "gated": not degenerate}
        verdict = ("n/a (span is %.0f%% of cache -- random control degenerate)" % (100 * frac)
                   if degenerate else ("PASS" if sig else "not separated"))
        L.append(f"  {c:22s} med {med:+9.4f} [{lo:+8.4f},{hi:+8.4f}] mean {mean:+9.4f} "
                 f"p={p:.2e}  {verdict}")

    # ---- G1: damage ranking vs attention-mass ranking ---------------------------
    cfg_json = json.loads((Path(root / args.shards[0]) / "config.json").read_text())
    mass = cfg_json["attention_mass_reference"]
    keys = [k for k in SPAN_OF if SPAN_OF[k] in mass]
    dmg = [rows[k]["med"] for k in keys]
    msk = [mass[SPAN_OF[k]] for k in keys]
    rho = float(spearmanr(dmg, msk).statistic)
    L.append("")
    L.append(f"G1  Spearman(damage, attention mass) over {len(keys)} spans = {rho:+.3f}"
             f"   {'PASS (rankings differ)' if rho < 0.9 else 'FAIL (mass already told us)'}")
    for k in keys:
        L.append(f"    {SPAN_OF[k]:16s} mass {mass[SPAN_OF[k]]:.4f}   "
                 f"damage {rows[k]['med']:+.4f}")

    # ---- G2 / G3 ---------------------------------------------------------------
    g2, g3 = rows["X3_coc"], rows["X1_vision"]
    L.append("")
    L.append(f"G3  (positive control) vision block delta {g3['med']:+.4f} "
             f"[{g3['lo']:+.4f},{g3['hi']:+.4f}]  "
             f"{'PASS' if g3['sig'] and g3['med'] > 1.0 else 'FAIL -- no power, do not read G2'}")
    L.append(f"G2  (main) generated-CoC block delta {g2['med']:+.4f} "
             f"[{g2['lo']:+.4f},{g2['hi']:+.4f}] median {g2['med']:+.4f} p={g2['p']:.2e}")
    if g2["sig"]:
        L.append("    -> CI excludes 0: the CoC contributes causally to the trajectory.")
    else:
        L.append("    -> CI includes 0. Because VLM prefill is causal the prompt cache cannot be")
        L.append("       updated by the CoC, so this is the whole channel: no detectable causal")
        L.append("       contribution of the reasoning text to the trajectory at this power.")

    # ---- per-token damage ------------------------------------------------------
    # The spans differ in size by two orders of magnitude, so the raw delta conflates
    # "this span matters" with "this span is big". n_tok is taken from the first clip
    # of the first shard; only the CoC span varies clip to clip (9-19 tokens observed).
    L.append("")
    L.append(f"per-token damage   ({'span':16s} {'n_tok':>6s} {'median d':>10s} "
             f"{'per token':>11s} {'mass':>7s} {'mass/token':>11s})")
    for k in keys:
        r, sp = rows[k], SPAN_OF[k]
        L.append(f"                   {sp:16s} {r['n_tok']:6d} {r['med']:+10.4f} "
                 f"{r['med'] / r['n_tok']:11.3e} {mass[sp]:7.4f} "
                 f"{mass[sp] / r['n_tok']:11.3e}")
    pt = {k: rows[k]["med"] / rows[k]["n_tok"] for k in keys}
    L.append(f"    per-token damage ratio  CoC/prompt_text = "
             f"{pt['X3_coc'] / pt['X4_text']:.1f}x   CoC/vision = "
             f"{pt['X3_coc'] / pt['X1_vision']:.1f}x   CoC/traj_history = "
             f"{pt['X3_coc'] / pt['X2_hist']:.1f}x")

    # ---- per-camera ------------------------------------------------------------
    cams = sorted([c for c in cfgs if c.startswith("X1c")])
    if cams:
        L.append("")
        L.append("per-camera vision blocks (prompt order = camera index ascending)")
        for c in cams:
            r = rows[c]
            L.append(f"  {CAM_REAL.get(c, c):18s} n={r['n_tok']:5d} {r['med']:+9.4f} "
                     f"[{r['lo']:+8.4f},{r['hi']:+8.4f}] p={r['p']:.2e}"
                     f"{'  *' if r['sig'] else ''}")

    # ---- plots -----------------------------------------------------------------
    main_cfgs = [c for c in cfgs if c in SPAN_OF or c == "X6_all_but_vision"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    y = np.arange(len(main_cfgs))
    mm = [rows[c]["med"] for c in main_cfgs]
    err = [[rows[c]["med"] - rows[c]["lo"] for c in main_cfgs],
           [rows[c]["hi"] - rows[c]["med"] for c in main_cfgs]]
    ax.barh(y, mm, xerr=err, color=[C3 if rows[c]["sig"] else C2 for c in main_cfgs],
            height=0.6, error_kw={"ecolor": MUTED, "lw": 1})
    ax.set_yticks(y); ax.set_yticklabels([rows[c]["span"] for c in main_cfgs])
    ax.set_xscale("symlog", linthresh=0.05)
    ax.axvline(0, color=INK, lw=1)
    ax.set_xlabel("paired ΔminADE vs unblocked (m, symlog)")
    ax.set_title("which cache span does the trajectory need", fontsize=10)

    ax = axes[1]
    ks = [k for k in SPAN_OF if k in g0 and g0[k]["gated"]]
    x = np.arange(len(ks)); w = 0.38
    ax.bar(x - w / 2, [rows[k]["med"] for k in ks], w, color=C1, label="span")
    ax.bar(x + w / 2, [rows[f"{k}_rand"]["med"] for k in ks], w, color=MUTED,
           label="size-matched random")
    ax.set_xticks(x); ax.set_xticklabels([SPAN_OF[k] for k in ks], rotation=20, ha="right")
    ax.set_yscale("symlog", linthresh=0.05)
    ax.set_ylabel("ΔminADE (m, symlog)"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("G0 - span vs size-matched random", fontsize=10)

    ax = axes[2]
    for k in keys:
        ax.scatter(mass[SPAN_OF[k]], max(rows[k]["med"], 1e-4), s=60, color=C1, zorder=3)
        ax.annotate(SPAN_OF[k], (mass[SPAN_OF[k]], max(rows[k]["med"], 1e-4)),
                    textcoords="offset points", xytext=(6, 4), fontsize=8, color=MUTED)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("expert attention mass (07-20)"); ax.set_ylabel("ΔminADE (m)")
    ax.set_title(f"G1 - attention mass vs causal damage (rho={rho:+.2f})", fontsize=10)
    fig.tight_layout(); fig.savefig(out_dir / "plots" / "pathway_x.png", dpi=150)
    plt.close(fig)

    (out_dir / "summary.txt").write_text("\n".join(L) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps({
        "n_clips": n, "clip_ids": clips, "buckets": buckets, "shards": args.shards,
        "baseline_minADE_mean": float(np.mean(base)),
        "coc_nll_mean": float(np.mean(nll)),
        "rows": rows, "g0": g0, "g1_spearman_damage_vs_mass": rho,
        "gates": {
            "G0": {k: v["sig"] for k, v in g0.items() if v["gated"]},
            "G1": bool(rho < 0.9),
            "G2": bool(rows["X3_coc"]["sig"]),
            "G3": bool(rows["X1_vision"]["sig"] and rows["X1_vision"]["med"] > 1.0),
        },
    }, indent=2))
    print("\n".join(L))
    print("\nsaved ->", out_dir)


if __name__ == "__main__":
    main()
