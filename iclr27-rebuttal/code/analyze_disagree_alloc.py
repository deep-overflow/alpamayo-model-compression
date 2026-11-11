"""Test500 open loop of MoOP with the layer budget set by I_traj / I_CoC top-set disagreement (dis050, dis100) vs MoOP l0keep.

Plan: iclr27-rebuttal/plans/2026-11-10_moop-disagreement-allocation-test500.md. Rows: outputs/iclr27-rebuttal/disagree_alloc/
<arm>_test500 (Ada 4-7, 2026-11-11); references, not re-run: MoOP l0keep = outputs/swap_ship_L0fullM_500 (masks identical to
dual_u40_l0keep_v2), Dense = outputs/baseline_pred_test. All Ada.

  every arm   minADE@6 / minFDE@6 at 1.6 / 3.2 / 6.4 s (mean, clip-bootstrap CI, median), CoC degeneration
  paired      arm - MoOP l0keep, arm - Dense, dis100 - dis050 (200,000 draws, seed 0, seed 1 re-check)
  verdict     pre-registered on 6.4 s minADE@6 vs MoOP l0keep: better (CI < 0), equivalent (CI inside +-0.03 m),
              worse (CI > 0), else inconclusive
  buckets     6.4 s minADE@6 arm - MoOP l0keep per driving situation (cruise / decel_stop / accel / turn)
  figure      per-layer Q / MLP cuts of the three allocations and the 6.4 s paired deltas
Usage (repo root): .venv/bin/python iclr27-rebuttal/code/analyze_disagree_alloc.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "cvpr27" / "code"))
sys.path.insert(0, str(REPO / "iclr27-rebuttal" / "code"))
from analyze_a100 import HORIZONS, N_BOOT, K, load_rows, metric
from ci_openloop_lingoqa import flips, fmt_ci, pfmt, star, summarise

OUT = REPO / "outputs"
D = OUT / "iclr27-rebuttal" / "disagree_alloc"
KINDS = ("ade", "fde")
EQ_MARGIN = 0.03
# key, label, run dir (under outputs), tag
ARMS = [("dense", "Dense", "baseline_pred_test", "baseline"),
        ("moop", "MoOP l0keep (uniform)", "swap_ship_L0fullM_500", "slim_swap_ship_L0fullM"),
        ("dis050", "dis050 (strength 0.5)", "iclr27-rebuttal/disagree_alloc/dis050_test500", "slim_dis050"),
        ("dis100", "dis100 (strength 1.0)", "iclr27-rebuttal/disagree_alloc/dis100_test500", "slim_dis100")]
PAIRS = [("dis050", "moop"), ("dis100", "moop"), ("dis100", "dis050"), ("moop", "dense"), ("dis050", "dense"),
         ("dis100", "dense")]
BUCKETS = ["cruise", "decel_stop", "accel", "turn"]
COLORS = {"moop": "#8c8c8c", "dis050": "#3b6fb6", "dis100": "#e07b28"}


def verdict(ci):
    lo, hi = ci
    if hi < 0:
        return "better"
    if lo > 0:
        return "worse"
    if -EQ_MARGIN < lo and hi < EQ_MARGIN:
        return "equivalent"
    return "inconclusive"


def main():
    ids = list(pd.read_parquet(OUT / "eval_sets" / "test_500.parquet")["clip_id"])
    rows, gpus = {}, {}
    for k, _, d, tag in ARMS:
        gpus[k] = json.loads((OUT / d / "config.json").read_text()).get("gpu")
        rows[k] = load_rows(OUT / d, tag)
        missing = [c for c in ids if c not in rows[k]]
        assert not missing, f"{k}: {len(missing)} of 500 clips missing"
    assert all("Ada" in str(g) for g in gpus.values()), gpus
    lab = {k: lab for k, lab, *_ in ARMS}

    series = {f"abs|{k}|{kind}|{h}": [metric(rows[k][c], kind, h) for c in ids]
              for k, *_ in ARMS for kind in KINDS for h in HORIZONS}  # (500,) each
    series.update({f"d|{a}|{b}|{kind}|{h}": [metric(rows[a][c], kind, h) - metric(rows[b][c], kind, h) for c in ids]
                   for a, b in PAIRS for kind in KINDS for h in HORIZONS})
    S = summarise(series)
    bucket = [rows["moop"][c]["bucket"] for c in ids]
    for bk in BUCKETS:   # one bootstrap per situation: its clips differ in number from the other situations'
        S.update(summarise({f"b|{a}|{bk}": [metric(rows[a][c], "ade") - metric(rows["moop"][c], "ade")
                                            for c, b in zip(ids, bucket) if b == bk] for a in ("dis050", "dis100")}))
    coc = {k: float(np.mean([rows[k][c]["coc_degenerate"] for c in ids])) for k, *_ in ARMS}
    worse = {f"{a}|{b}": float(np.mean(np.array(series[f"d|{a}|{b}|ade|6.4s"]) > 0)) for a, b in PAIRS}
    verdicts = {a: verdict(S[f"d|{a}|moop|ade|6.4s"]["ci"]) for a in ("dis050", "dis100")}
    alloc = json.loads((D / "alloc_summary.json").read_text())

    md = ["# MoOP layer budget by importance disagreement -- Test500 open loop (Ada)", "",
          (f"n = {len(ids)} clips; minADE@6 / minFDE@6 = min over the first {K} of 8 samples, rollout only; bootstrap "
           f"{N_BOOT:,} draws, seed 0; * = CI excludes 0."), "",
          "| arm | minADE@6 1.6 s | 3.2 s | 6.4 s | minFDE@6 1.6 s | 3.2 s | 6.4 s | median ADE / FDE 6.4 s | CoC degenerate |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, *_ in ARMS:
        md.append(f"| {lab[k]} | " + " | ".join(fmt_ci(S[f"abs|{k}|{kind}|{h}"]) for kind in KINDS for h in HORIZONS)
                  + f" | {S[f'abs|{k}|ade|6.4s']['median']:.3f} / {S[f'abs|{k}|fde|6.4s']['median']:.3f} | {coc[k]:.1%} |")
    md += ["", ("| pair | dADE 1.6 s | 3.2 s | 6.4 s | dFDE 1.6 s | 3.2 s | 6.4 s | median dADE / dFDE 6.4 s "
                "| p ADE / FDE 6.4 s | clips worse (6.4 s ADE) |"), "|---|---|---|---|---|---|---|---|---|---|"]
    for a, b in PAIRS:
        cells = [fmt_ci(S[f"d|{a}|{b}|{kind}|{h}"], sign=True) + star(S[f"d|{a}|{b}|{kind}|{h}"])
                 for kind in KINDS for h in HORIZONS]
        ea, ef = S[f"d|{a}|{b}|ade|6.4s"], S[f"d|{a}|{b}|fde|6.4s"]
        md.append(f"| {lab[a]} - {lab[b]} | " + " | ".join(cells) + f" | {ea['median']:+.3f} / {ef['median']:+.3f} | "
                  f"{pfmt(ea['p'])} / {pfmt(ef['p'])} | {worse[f'{a}|{b}']:.1%} |")
    md += ["", "Pre-registered verdict (6.4 s minADE@6 vs MoOP l0keep; equivalence margin +-0.03 m):", ""]
    md += [f"- {lab[a]}: {fmt_ci(S[f'd|{a}|moop|ade|6.4s'], sign=True)} -> **{verdicts[a]}**" for a in verdicts]
    md += ["", "6.4 s minADE@6, arm - MoOP l0keep by driving situation:", "",
           "| arm | " + " | ".join(f"{bk} (n = {bucket.count(bk)})" for bk in BUCKETS) + " |",
           "|---|" + "---|" * len(BUCKETS)]
    for a in ("dis050", "dis100"):
        md.append(f"| {lab[a]} | " + " | ".join(fmt_ci(S[f"b|{a}|{bk}"], sign=True) + star(S[f"b|{a}|{bk}"])
                                                for bk in BUCKETS) + " |")
    flipped = [k for k in flips(S) if not k.startswith("abs|")]
    md += ["", f"Seed check (paired cells whose 'excludes 0' call flips at seed 1): {flipped if flipped else 'none'}"]

    dst = D / "analysis"
    dst.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0), gridspec_kw={"width_ratios": [1.2, 1.2, 1.0]})
    layers = np.arange(36)
    ref_q = [32 - 19 if li else 0 for li in layers]
    for ax, key, ref, title in ((axes[0], "cut_q", ref_q, "Q heads cut per layer (of 32)"),
                                (axes[1], "cut_mlp", None, "MLP channels cut per layer (of 12,288)")):
        moop = ref if ref is not None else alloc["arms"]["s000"][key]
        ax.plot(layers, moop, color=COLORS["moop"], lw=2, label="MoOP l0keep (uniform)")
        for a in ("dis050", "dis100"):
            ax.plot(layers, alloc["arms"][a][key], color=COLORS[a], lw=2, marker="o", ms=3, label=lab[a])
        cap = alloc["cap_q"] if key == "cut_q" else alloc["cap_mlp"]
        ax.axhline(cap, color="#bbbbbb", lw=1, ls="--")
        ax.text(35, cap, " cap", va="bottom", ha="right", fontsize=8, color="#666666")
        ax.set_xlabel("layer")
        ax.set_title(title, fontsize=10)
        ax.grid(color="#eeeeee", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    ax = axes[2]
    hs = list(HORIZONS)
    for i, a in enumerate(("dis050", "dis100")):
        for j, h in enumerate(hs):
            e = S[f"d|{a}|moop|ade|{h}"]
            y = j + (i - 0.5) * 0.25
            ax.plot(e["ci"], [y, y], color=COLORS[a], lw=2, solid_capstyle="round")
            ax.plot([e["mean"]], [y], "o", color=COLORS[a], ms=6, mec="white", mew=1.5,
                    label=lab[a] if j == 0 else None)
    ax.axvline(0, color="#999999", lw=1)
    ax.set_yticks(range(len(hs)), [f"minADE@6 {h}" for h in hs])
    ax.set_xlabel("arm - MoOP l0keep (m), 95% CI")
    ax.set_title("Paired difference vs MoOP l0keep", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(dst / "cuts_and_deltas.png", dpi=150)
    plt.close(fig)

    (dst / "config.json").write_text(json.dumps({
        "plan": "iclr27-rebuttal/plans/2026-11-10_moop-disagreement-allocation-test500.md",
        "arms": {k: {"label": lab_, "dir": d, "tag": t, "gpu": gpus[k]} for k, lab_, d, t in ARMS},
        "pairs": PAIRS, "k": K, "horizons": HORIZONS, "n_boot": N_BOOT, "seeds": [0, 1],
        "equivalence_margin_m": EQ_MARGIN, "alloc": str((D / "alloc_summary.json").relative_to(REPO))}, indent=2))
    (dst / "metrics.json").write_text(json.dumps({"stats": S, "coc_degenerate": coc, "worse_frac": worse,
                                                  "verdicts": verdicts, "seed_flips": flipped}, indent=2))
    (dst / "summary.txt").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
