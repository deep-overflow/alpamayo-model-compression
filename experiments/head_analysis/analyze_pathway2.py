"""Analyze the Stage-2 pathway map: VLM-internal edges x layer bands, two readouts.

Every config yields both a language readout (teacher-forced CoC NLL) and an action
readout (minADE@8) from the same masked forward, so each cell of the edge x band grid
can be classified by *which channel it damages*. That classification is the point:
this repo's thesis is that reasoning and trajectory live in different units, and if it
transfers to the token/pathway axis there must be edges that damage one channel and
not the other.

Integrity first: E0_causalonly injects a mask with no blocks, i.e. the plain causal
mask, and must reproduce E0_none. It will not match bitwise -- supplying a mask makes
sdpa take the non-flash kernel and materialize repeat_kv instead of enable_gqa -- so
the gate is the bf16 noise floor this repo established elsewhere (|dNLL| < 0.01).

40 cells x 2 readouts is a lot of tests, so raw p is reported next to a
Benjamini-Hochberg FDR. Only H3 was pre-registered; individual cells are exploratory.

Usage:
  python analyze_pathway2.py --shards pathway_e_s0 pathway_e_s13 pathway_e_s26 \
      pathway_e_s38 --out pathway_e_v1 --outputs-root /home/.../outputs
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr, wilcoxon  # noqa: E402

BG, INK, MUTED, GRID = "#FAF9F5", "#29261B", "#6B6555", "#E8E6DC"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "font.size": 9, "axes.grid": True, "axes.axisbelow": True,
})

BASE, INTEG = "E0_none", "E0_causalonly"
NOISE_FLOOR = 0.01          # bf16 noise floor for |dNLL|, from the 2026-07-23 slim verify
MEANINGFUL_REL = 0.05       # a channel effect inside +-5% of its baseline median is practically null
EDGES = ["E1_crossframe", "E2_crosscam", "E3_hist_vision", "E4_instr_vision",
         "E5_coc_vision", "E6_coc_hist", "E7_coc_instr", "E8_all_sink"]
BANDS = ["L0-8", "L9-17", "L18-26", "L27-35", "all"]
PRETTY = {"E1_crossframe": "vision <- same-cam earlier frames",
          "E2_crosscam": "vision <- other cameras",
          "E3_hist_vision": "traj-history <- vision",
          "E4_instr_vision": "instruction <- vision",
          "E5_coc_vision": "CoC <- vision",
          "E6_coc_hist": "CoC <- traj-history",
          "E7_coc_instr": "CoC <- instruction",
          "E8_all_sink": "everything <- sink"}


def merge(shard_dirs):
    cfgs, meta, per_clip, buckets, clips = None, None, {}, [], []
    for d in shard_dirs:
        m = json.loads((d / "metrics.json").read_text())
        if cfgs is None:
            cfgs, meta = m["configs"], m["meta"]
            per_clip = {c: {"ade": [], "fde": [], "nll": []} for c in cfgs}
        elif m["configs"] != cfgs:
            raise SystemExit(f"config mismatch in {d}")
        for c in cfgs:
            for f in ("ade", "fde", "nll"):
                per_clip[c][f] += m["per_clip"][c][f]
        buckets += m["buckets"]
        clips += m["clip_ids"]
    return cfgs, meta, per_clip, buckets, clips


def median_ci(d, n_boot=10000, seed=0, alpha=0.05):
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(d), size=(n_boot, len(d)))
    boots = np.median(d[idx], axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.median(d)), float(lo), float(hi)


def paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    med, lo, hi = median_ci(d)
    try:
        p = float(wilcoxon(d).pvalue)
    except ValueError:
        p = float("nan")
    return {"med": med, "lo": lo, "hi": hi, "mean": float(d.mean()), "p": p,
            "sig": bool((lo > 0 or hi < 0) and p < 0.05)}


def bh_fdr(pvals, q=0.05):
    """Benjamini-Hochberg: returns the boolean reject vector in the input order."""
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    order = np.argsort(np.where(ok, p, 2.0))
    m = int(ok.sum())
    rej = np.zeros(len(p), bool)
    thresh = 0
    for rank, i in enumerate(order[:m], start=1):
        if p[i] <= q * rank / m:
            thresh = rank
    for rank, i in enumerate(order[:m], start=1):
        rej[i] = rank <= thresh
    return rej


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", default="pathway_e_v1")
    ap.add_argument("--outputs-root", required=True)
    args = ap.parse_args()

    root = Path(args.outputs_root)
    _, meta, per_clip, buckets, clips = merge([root / s for s in args.shards])
    out_dir = root / args.out
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    n = len(clips)
    b_ade, b_nll = per_clip[BASE]["ade"], per_clip[BASE]["nll"]

    L = [f"Pathway map Stage 2 -- VLM-internal edges x layer bands   (n={n} clips, K=8)",
         (f"unblocked  minADE median {np.median(b_ade):.4f} mean {np.mean(b_ade):.4f}   "
          f"CoC NLL median {np.median(b_nll):.4f} mean {np.mean(b_nll):.4f}"), ""]

    # ---- integrity --------------------------------------------------------------
    i_ade = paired(per_clip[INTEG]["ade"], b_ade)
    i_nll = paired(per_clip[INTEG]["nll"], b_nll)
    ok = abs(i_nll["med"]) < NOISE_FLOOR
    verdict = f"PASS (within bf16 noise floor {NOISE_FLOOR:.2f})" if ok else "FAIL"
    L.append(f"INTEGRITY  causal-mask-only vs no-mask: dNLL med {i_nll['med']:+.2e} "
             f"dADE med {i_ade['med']:+.2e}   {verdict}")
    if not ok:
        L.append("  -> mask construction is wrong; every row below is meaningless.")
    L.append("")

    # ---- grid -------------------------------------------------------------------
    rows, pv_ade, pv_nll, keys = {}, [], [], []
    for e in EDGES:
        for bn in BANDS:
            c = f"{e}@{bn}"
            if c not in per_clip:
                continue
            r = {"ade": paired(per_clip[c]["ade"], b_ade),
                 "nll": paired(per_clip[c]["nll"], b_nll),
                 "edge": e, "band": bn, "n_pairs": meta[c].get("n_pairs", 0)}
            rows[c] = r
            keys.append(c)
            pv_ade.append(r["ade"]["p"])
            pv_nll.append(r["nll"]["p"])
    rej_a, rej_n = bh_fdr(pv_ade), bh_fdr(pv_nll)
    for i, c in enumerate(keys):
        rows[c]["ade"]["fdr"] = bool(rej_a[i])
        rows[c]["nll"]["fdr"] = bool(rej_n[i])

    # The two channels are in different units (metres vs nats) and, more importantly,
    # have very different clip-to-clip spread, so a raw side-by-side is not a fair
    # comparison of "which channel got hit harder". Normalise each delta by its own
    # channel's unblocked median, and record each channel's resolution -- the typical
    # bootstrap CI half-width across cells, i.e. the smallest effect this n can resolve.
    m_ade, m_nll = float(np.median(b_ade)), float(np.median(b_nll))
    for c in keys:
        rows[c]["ade"]["rel"] = rows[c]["ade"]["med"] / m_ade
        rows[c]["nll"]["rel"] = rows[c]["nll"]["med"] / m_nll
    res = {ch: float(np.median([(rows[c][ch]["hi"] - rows[c][ch]["lo"]) / 2 for c in keys]))
           for ch in ("ade", "nll")}
    # Per-cell equivalence, not a global floor: CI half-widths vary ~10x across cells, so
    # a single "resolution" number would flatter the cells that actually matter. A cell is
    # called practically null on a channel only if its whole CI sits inside +-5% of that
    # channel's unblocked median.
    for c in keys:
        for ch, base_m in (("ade", m_ade), ("nll", m_nll)):
            band = MEANINGFUL_REL * abs(base_m)
            r = rows[c][ch]
            r["null_eq"] = bool(r["lo"] > -band and r["hi"] < band)
    L.append(f"channel spread (median CI half-width over {len(keys)} cells): "
             f"action {res['ade']:.4f} m = {res['ade'] / m_ade:.1%} of baseline; "
             f"language {res['nll']:.4f} = {res['nll'] / m_nll:.1%} of baseline. "
             f"Practical-null band = +-{MEANINGFUL_REL:.0%} of baseline per channel.")
    L.append("")

    def chan(r):
        a, nl = r["ade"]["sig"] and r["ade"]["fdr"], r["nll"]["sig"] and r["nll"]["fdr"]
        return "both" if (a and nl) else "action" if a else "language" if nl else "-"

    L.append("edge x band grid   (paired median delta; * = significant after BH-FDR q=0.05)")
    L.append(f"{'edge':32s} {'band':8s} {'dminADE':>9s} {'rel':>7s} {'':1s} "
             f"{'dCoC NLL':>10s} {'rel':>7s} {'':1s} {'channel':>9s}")
    for e in EDGES:
        for bn in BANDS:
            c = f"{e}@{bn}"
            if c not in rows:
                continue
            r = rows[c]
            L.append(f"{PRETTY[e]:32s} {bn:8s} {r['ade']['med']:+9.4f} "
                     f"{r['ade']['rel']:+6.1%} "
                     f"{'*' if r['ade']['sig'] and r['ade']['fdr'] else ' '} "
                     f"{r['nll']['med']:+10.4f} {r['nll']['rel']:+6.1%} "
                     f"{'*' if r['nll']['sig'] and r['nll']['fdr'] else ' '} "
                     f"{chan(r):>9s}")
        L.append("")

    # ---- H3: are the two channels dissociable? ----------------------------------
    a_vec = [rows[c]["ade"]["med"] for c in keys]
    n_vec = [rows[c]["nll"]["med"] for c in keys]
    rho = float(spearmanr(a_vec, n_vec).statistic)
    cls = {k: [] for k in ["action", "language", "both", "-"]}
    for c in keys:
        cls[chan(rows[c])].append(c)
    L.append(f"H3  Spearman(dminADE, dCoC NLL) over {len(keys)} cells = {rho:+.3f}")
    L.append(f"    action-only  {len(cls['action']):2d} cells: "
             f"{', '.join(cls['action'][:6])}{' ...' if len(cls['action']) > 6 else ''}")
    L.append(f"    language-only{len(cls['language']):2d} cells: "
             f"{', '.join(cls['language'][:6])}{' ...' if len(cls['language']) > 6 else ''}")
    L.append(f"    both         {len(cls['both']):2d} cells")
    L.append(f"    neither      {len(cls['-']):2d} cells")
    h3 = bool(cls["action"] and cls["language"])
    h3_msg = ("ACCEPT -- both single-channel classes are non-empty, so the two readouts "
              "are dissociable at the pathway level" if h3 else
              "not supported -- no edge damages exactly one channel")
    L.append(f"    H3 {h3_msg}")
    # A single-channel cell is evidence of dissociation only if the *other* channel is
    # positively shown to be small, not merely non-significant.
    lang_eq = [c for c in cls["language"] if rows[c]["ade"]["null_eq"]]
    act_eq = [c for c in cls["action"] if rows[c]["nll"]["null_eq"]]
    L.append(f"    of {len(cls['language'])} language-only cells, {len(lang_eq)} are also "
             f"practically null on the action channel (CI inside +-{MEANINGFUL_REL:.0%}) "
             f"-> genuine dissociation; the rest are merely undetermined.")
    L.append(f"    of {len(cls['action']):2d} action-only cells, {len(act_eq)} are also "
             f"practically null on the language channel.")
    if lang_eq:
        L.append(f"      e.g. {', '.join(lang_eq[:5])}")

    # ---- plots -------------------------------------------------------------------
    A = np.array([[rows[f"{e}@{b}"]["ade"]["med"] if f"{e}@{b}" in rows else np.nan
                   for b in BANDS] for e in EDGES])
    N = np.array([[rows[f"{e}@{b}"]["nll"]["med"] if f"{e}@{b}" in rows else np.nan
                   for b in BANDS] for e in EDGES])
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
    for ax, M, title in [(axes[0], A, "action channel  (ΔminADE, m)"),
                         (axes[1], N, "language channel  (ΔCoC NLL)")]:
        # One cell (CoC<-instruction) is ~10x the next largest, so a max-scaled colormap
        # washes out every other cell. Clip the scale to the 90th percentile; the printed
        # numbers carry the true value for the saturated cells.
        v = float(np.nanpercentile(np.abs(M), 90)) or float(np.nanmax(np.abs(M)))
        im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
        ax.set_xticks(range(len(BANDS))); ax.set_xticklabels(BANDS, rotation=30, ha="right")
        ax.set_yticks(range(len(EDGES))); ax.set_yticklabels([PRETTY[e] for e in EDGES], fontsize=8)
        ax.set_title(title, fontsize=10); ax.grid(False)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    # saturated cells get white text or they vanish into the colormap
                    ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=7,
                            color="#FFFFFF" if abs(M[i, j]) > 0.65 * v else INK)
        fig.colorbar(im, ax=ax, fraction=0.046)
    ax = axes[2]
    col = {"action": C3, "language": C1, "both": C4, "-": MUTED}
    for c in keys:
        r = rows[c]
        ax.scatter(r["nll"]["med"], r["ade"]["med"], s=42, color=col[chan(r)], zorder=3,
                   edgecolors="none")
    ax.axhline(0, color=INK, lw=1); ax.axvline(0, color=INK, lw=1)
    ax.set_xlabel("Δ CoC NLL (language)"); ax.set_ylabel("Δ minADE (action, m)")
    ax.set_title(f"channel dissociation (rho={rho:+.2f})", fontsize=10)
    for lab, cc in [("action-only", C3), ("language-only", C1), ("both", C4), ("neither", MUTED)]:
        ax.scatter([], [], color=cc, label=lab, s=42)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout(); fig.savefig(out_dir / "plots" / "pathway_e.png", dpi=150)
    plt.close(fig)

    (out_dir / "summary.txt").write_text("\n".join(L) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps({
        "n_clips": n, "clip_ids": clips, "buckets": buckets, "shards": args.shards,
        "baseline": {"minADE_median": float(np.median(b_ade)),
                     "coc_nll_median": float(np.median(b_nll))},
        "integrity": {"ade": i_ade, "nll": i_nll, "pass": ok, "noise_floor": NOISE_FLOOR},
        "resolution": res,
        "rows": rows, "h3": {"spearman": rho, "classes": dict(cls), "accept": h3,
                             "language_only_action_equivalent": lang_eq,
                             "action_only_language_equivalent": act_eq},
    }, indent=2))
    print("\n".join(L))
    print("\nsaved ->", out_dir)


if __name__ == "__main__":
    main()
