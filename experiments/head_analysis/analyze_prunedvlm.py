"""Gates and figures for the pruned-VLM internals run.

plans/2026-09-01_pruned-vlm-internals.md. Reads outputs/<exp>/{residual,attn}.npz plus the
per-clip metrics.json and judges:

  G0  alignment -- layer-0 cache K identical, layer-0 h_in identical between the two passes
  G1  residual decomposition bitwise (only present when the run used --verify-taps)
  G2  attention rows sum to 1 before renormalisation, to the bf16 rounding scale
  G3  two floors: the no-mask re-run floor (must be ~0) and the bf16 rounding floor
  G4  the reading -- span-level mass barely moves (prior work), but does the DISTRIBUTION
      move? TV far above the floor with flat span mass means the expert relocates its
      attention inside the spans rather than between them.

Everything is aggregated with a clip bootstrap; no per-cell claim is made, only marginals.

Usage:
  .venv/bin/python experiments/head_analysis/analyze_prunedvlm.py \
      [--exp prunedvlm_dual] [--floor prunedvlm_nomask] [--smoke prunedvlm_smoke] \
      [--out prunedvlm_analysis] [--template ...]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))

BG, INK, MUTED, GRID = "#FAF9F5", "#29261B", "#6B6555", "#E8E6DC"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "font.size": 9, "axes.grid": True, "axes.axisbelow": True,
})
REPO = Path(__file__).resolve().parents[2]
POINTS = ["h_in", "h_mid", "h_out"]
TRANS = ["attn", "mlp"]


def load(exp):
    d = REPO / "outputs" / exp
    cfg = json.loads((d / "config.json").read_text())
    rows = json.loads((d / "metrics.json").read_text())
    res = {k: v for k, v in np.load(d / "residual.npz").items()}
    attn = {k: v for k, v in np.load(d / "attn.npz").items()}
    return cfg, rows, res, attn


def boot(x, n=4000, seed=0):
    """Median over clips with a clip bootstrap CI. x is (n_clips,) or (n_clips, ...)."""
    x = np.asarray(x, dtype=np.float64)
    med = float(np.median(x, axis=0).mean()) if x.ndim > 1 else float(np.median(x))
    rng = np.random.default_rng(seed)
    b = []
    for _ in range(n):
        pick = rng.integers(0, len(x), len(x))
        s = x[pick]
        b.append(float(np.median(s, axis=0).mean()) if x.ndim > 1 else float(np.median(s)))
    lo, hi = np.percentile(b, [2.5, 97.5])
    return {"med": med, "lo": float(lo), "hi": float(hi), "n": len(x)}


def si(spans, name):
    return spans.index(name)


def gates(cfg, rows, res, attn, floor_rows=None, floor_attn=None, smoke_rows=None):
    g = {}
    g["G0_layer0_dk_max"] = float(max(r["layer0_max_dk"] for r in rows))
    g["G0_layer0_hin_rel_max"] = float(res["cross_rel"][:, 0, 0, :].max())
    g["G0_pass"] = bool(g["G0_layer0_dk_max"] == 0.0
                        and g["G0_layer0_hin_rel_max"] == 0.0)
    if smoke_rows is not None:
        g["G1_bitwise_frac"] = float(np.mean([r.get("g1_bitwise_frac", np.nan)
                                              for r in smoke_rows]))
        g["G1_pass"] = bool(g["G1_bitwise_frac"] == 1.0)
    g["G2_rowsum_dev_max"] = float(max(r["rowsum_bf16_max"] for r in rows))
    g["G2_pass"] = bool(g["G2_rowsum_dev_max"] < 1e-2)
    if floor_attn is not None:
        g["G3_rerun_tv_max"] = float(floor_attn["tv"].max())
        g["G3_rerun_js_max"] = float(floor_attn["js"].max())
    if floor_rows is not None:
        g["G3_rerun_tv_mean"] = float(np.mean([r["tv_mean"] for r in floor_rows]))
    if smoke_rows is not None:
        g["G3_bf16_floor_rel"] = float(np.mean([r.get("bf16_floor_rel", np.nan)
                                                for r in smoke_rows]))
    return g


def summarise(cfg, res, attn):
    spans = cfg["spans"]
    keep = [s for s in spans if s in ("all", "vision", "text", "hist", "sink", "coc",
                                      "special", "sys_text", "cam_text", "instr")]
    # the runner stores spans in the order it built them; recover it from the arrays
    order = cfg.get("_span_order") or spans
    out = {"spans": order}
    ai = si(order, "all") if "all" in order else 0

    out["tv_overall"] = boot(attn["tv"].reshape(len(attn["tv"]), -1).mean(1))
    out["js_overall"] = boot(attn["js"].reshape(len(attn["js"]), -1).mean(1))
    out["kl_pq"] = boot(attn["kl_pq"].reshape(len(attn["kl_pq"]), -1).mean(1))
    out["kl_qp"] = boot(attn["kl_qp"].reshape(len(attn["kl_qp"]), -1).mean(1))
    out["kl_asym"] = boot((attn["kl_pq"] - attn["kl_qp"]).reshape(
        len(attn["kl_pq"]), -1).mean(1))
    # span mass change, in percentage points, per span (clip-mean over l,h,s)
    dmass = (attn["mass_p"] - attn["mass_d"]) * 100
    out["dmass_pp"] = {n: boot(dmass[..., j].reshape(len(dmass), -1).mean(1))
                       for j, n in enumerate(order) if n in keep}
    out["dmass_abs_max_pp"] = float(max(abs(v["med"]) for v in out["dmass_pp"].values()))

    # The decomposition the whole analysis turns on. TV over the full key axis counts every
    # relocation; the part of it that a span-level readout could ever see is
    # 0.5 * sum_span |mass_p - mass_d| over the DISJOINT spans (the coarse five plus the
    # expert's own 64 diffusion tokens). Whatever is left moved inside a span, where prior
    # work -- which only ever reported span mass -- was blind to it.
    part = [order.index(n) for n in ("vision", "text", "hist", "sink", "coc") if n in order]
    dm = attn["mass_p"][..., part] - attn["mass_d"][..., part]
    dwn = (attn["own_p"] - attn["own_d"])[..., None]
    between = 0.5 * (np.abs(dm).sum(-1) + np.abs(dwn).sum(-1))  # (n, L, H, S)
    tv_q = attn["tv"].mean(-1)  # TV averaged over the 64 queries -> (n, L, H, S)
    out["tv_between"] = boot(between.reshape(len(between), -1).mean(1))
    out["tv_within"] = boot((tv_q - between).reshape(len(between), -1).mean(1))
    out["within_share"] = float(out["tv_within"]["med"]
                                / max(out["tv_within"]["med"] + out["tv_between"]["med"],
                                      1e-12))
    out["dent"] = boot((attn["ent_p"] - attn["ent_d"]).reshape(len(attn["ent_p"]), -1).mean(1))

    for pi, p in enumerate(POINTS):
        out[f"rel_{p}"] = boot(res["cross_rel"][:, :, pi, ai].mean(1))
        out[f"cos_{p}"] = boot(res["cross_cos"][:, :, pi, ai].mean(1))
    out["d_attn"] = boot((res["cross_rel"][:, :, 1, ai]
                          - res["cross_rel"][:, :, 0, ai]).mean(1))
    out["d_mlp"] = boot((res["cross_rel"][:, :, 2, ai]
                         - res["cross_rel"][:, :, 1, ai]).mean(1))
    for ti, t in enumerate(TRANS):
        for tag in ("d", "p"):
            out[f"tcos_{t}_{tag}"] = boot(res[f"trans_cos_{tag}"][:, :, ti, ai].mean(1))
            out[f"tmag_{t}_{tag}"] = boot(res[f"trans_mag_{tag}"][:, :, ti, ai].mean(1))
    return out


def plot_residual(cfg, res, out):
    order = cfg.get("_span_order") or cfg["spans"]
    ai = order.index("all")
    L = res["cross_rel"].shape[1]
    x = np.arange(L)

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.8))
    for pi, (p, c) in enumerate(zip(POINTS, (C1, C4, C2))):
        m = res["cross_rel"][:, :, pi, ai]
        axes[0].plot(x, m.mean(0), color=c, lw=2, label=p)
        axes[0].fill_between(x, np.percentile(m, 25, 0), np.percentile(m, 75, 0),
                             color=c, alpha=0.15)
    axes[0].set_xlabel("VLM text layer")
    axes[0].set_ylabel(r"$\|h_p-h_d\|/\|h_d\|$")
    axes[0].set_title("residual divergence at the three points")
    axes[0].legend(fontsize=8)

    da = res["cross_rel"][:, :, 1, ai] - res["cross_rel"][:, :, 0, ai]
    dm = res["cross_rel"][:, :, 2, ai] - res["cross_rel"][:, :, 1, ai]
    axes[1].plot(x, da.mean(0), color=C1, lw=2, label=r"$\Delta_{attn}$")
    axes[1].plot(x, dm.mean(0), color=C2, lw=2, label=r"$\Delta_{mlp}$")
    axes[1].axhline(0, color=MUTED, lw=0.8)
    axes[1].set_xlabel("VLM text layer")
    axes[1].set_ylabel("rel added inside the layer")
    axes[1].set_title("which sub-block opens the gap")
    axes[1].legend(fontsize=8)

    for ti, (t, c) in enumerate(zip(TRANS, (C1, C2))):
        axes[2].plot(x, res["trans_cos_d"][:, :, ti, ai].mean(0), color=c, lw=2,
                     label=f"{t} dense")
        axes[2].plot(x, res["trans_cos_p"][:, :, ti, ai].mean(0), color=c, lw=1.6,
                     ls="--", label=f"{t} pruned")
    axes[2].set_xlabel("VLM text layer")
    axes[2].set_ylabel(r"$\cos$(before, after)")
    axes[2].set_title("how much each block turns the stream")
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "resid_layers.png", dpi=150)
    plt.close(fig)

    show = [s for s in ("vision", "instr", "cam_text", "sys_text", "special", "hist",
                        "sink", "coc") if s in order]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.0))
    for ax, arr, ttl in ((axes[0], res["cross_rel"][:, :, 1, :] - res["cross_rel"][:, :, 0, :],
                          r"$\Delta_{attn}$ by token type"),
                         (axes[1], res["cross_rel"][:, :, 2, :] - res["cross_rel"][:, :, 1, :],
                          r"$\Delta_{mlp}$ by token type")):
        m = np.stack([arr[:, :, order.index(s)].mean(0) for s in show])
        im = ax.imshow(m, aspect="auto", cmap="magma", origin="lower")
        ax.set_yticks(range(len(show)))
        ax.set_yticklabels(show, fontsize=8)
        ax.set_xlabel("VLM text layer")
        ax.set_title(ttl)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out / "resid_spans.png", dpi=150)
    plt.close(fig)


def plot_detail(cfg, attn, out):
    """The three axes the headline figures fold away: depth, head, and direction.

    Left   -- the between/within split resolved by layer. The 71% within-span share is a
              global number; this asks whether shallow layers move mass between spans while
              deep layers reshuffle inside them.
    Middle -- TV per (layer, head). The expert's 16 Q heads read 8 KV groups two-to-one
              (head h reads group h//2), so a head-resolved map also says whether the
              relocation follows the GQA grouping.
    Right  -- the KL asymmetry per layer. Positive means the pruned side drops keys the
              dense side kept; the scalar hides where that happens.
    """
    order = cfg.get("_span_order") or cfg["spans"]
    part = [order.index(n) for n in ("vision", "text", "hist", "sink", "coc") if n in order]
    dm = attn["mass_p"][..., part] - attn["mass_d"][..., part]
    dwn = (attn["own_p"] - attn["own_d"])[..., None]
    between = 0.5 * (np.abs(dm).sum(-1) + np.abs(dwn).sum(-1))  # (n, L, H, S)
    tv_q = attn["tv"].mean(-1)
    L, H = tv_q.shape[1], tv_q.shape[2]

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.8))
    b = between.mean((0, 2, 3))
    t = tv_q.mean((0, 2, 3))
    axes[0].fill_between(np.arange(L), 0, b, color=C4, alpha=0.55, label="between spans")
    axes[0].fill_between(np.arange(L), b, t, color=C1, alpha=0.55, label="within spans")
    axes[0].plot(np.arange(L), t, color=INK, lw=1.2)
    axes[0].set_xlabel("expert layer")
    axes[0].set_ylabel("TV")
    axes[0].set_title("where the relocation is, and whether it crosses a span")
    axes[0].legend(fontsize=8, loc="upper left")

    m = tv_q.mean((0, 3))  # (L, H)
    im = axes[1].imshow(m.T, aspect="auto", cmap="magma", origin="lower")
    for g in range(1, H // 2):
        axes[1].axhline(2 * g - 0.5, color="white", lw=0.4, alpha=0.5)
    axes[1].set_xlabel("expert layer")
    axes[1].set_ylabel("expert Q head")
    axes[1].set_title("TV per head (white lines = KV groups)")
    axes[1].grid(False)
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    asym = (attn["kl_pq"] - attn["kl_qp"]).mean((0, 2, 3, 4))
    axes[2].plot(np.arange(L), asym, color=C3, lw=2)
    axes[2].axhline(0, color=MUTED, lw=0.8)
    axes[2].set_xlabel("expert layer")
    axes[2].set_ylabel(r"KL$(d\|p)$ − KL$(p\|d)$")
    axes[2].set_title("positive = the pruned side drops keys")
    fig.tight_layout()
    fig.savefig(out / "attn_detail.png", dpi=150)
    plt.close(fig)


def plot_cross_tower(cfg, res, attn, out):
    """The two towers share a layer index -- expert layer l reads VLM cache layer l.

    So "which VLM layers were disturbed most" and "which expert layers read differently"
    can be put on the same axis. The cache at layer l is built by that layer's k/v_proj
    from its own input, so the natural pair is rel(h_in) against the expert's TV.
    """
    order = cfg.get("_span_order") or cfg["spans"]
    ai = order.index("all")
    L = res["cross_rel"].shape[1]
    show = [s for s in ("vision", "instr", "cam_text", "sys_text", "special", "hist",
                        "sink", "coc") if s in order]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.0))
    m = np.stack([res["cross_rel"][:, :, 2, order.index(s)].mean(0) for s in show])
    im = axes[0].imshow(m, aspect="auto", cmap="magma", origin="lower")
    axes[0].set_yticks(range(len(show)))
    axes[0].set_yticklabels(show, fontsize=8)
    axes[0].set_xlabel("VLM text layer")
    axes[0].set_title(r"how far each token type has drifted at $h_{out}$")
    axes[0].grid(False)
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    x = res["cross_rel"][:, :, 0, ai].mean(0)          # VLM residual divergence at h_in
    y = attn["tv"].mean((0, 2, 3, 4))                   # expert TV, same layer index
    sc = axes[1].scatter(x, y, c=np.arange(L), cmap="viridis", s=28)
    r = float(np.corrcoef(x, y)[0, 1])
    rho = float(spearmanr(x, y).statistic)
    axes[1].set_xlabel(r"VLM residual divergence at $h_{in}$ (rel)")
    axes[1].set_ylabel("expert TV at the same layer")
    axes[1].set_title(f"cache disturbance vs how differently it is read\n"
                      f"Pearson {r:+.2f}, Spearman {rho:+.2f}")
    fig.colorbar(sc, ax=axes[1], fraction=0.046, label="layer")
    fig.tight_layout()
    fig.savefig(out / "cross_tower.png", dpi=150)
    plt.close(fig)
    return {"pearson": r, "spearman": rho}


def plot_attn(cfg, attn, out):
    order = cfg.get("_span_order") or cfg["spans"]
    tv = attn["tv"]  # (n, L, H, S, Q)
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.8))
    m = tv.mean((0, 2, 4))  # (L, S)
    im = axes[0].imshow(m.T, aspect="auto", cmap="magma", origin="lower")
    axes[0].set_xlabel("expert layer")
    axes[0].set_ylabel("denoising step")
    axes[0].set_title("TV: fraction of attention mass relocated")
    axes[0].grid(False)
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    q = tv.mean((0, 1, 2, 3))  # (Q,)
    axes[1].plot(np.arange(len(q)), q, color=C1, lw=1.6)
    axes[1].set_xlabel("action (diffusion) token")
    axes[1].set_ylabel("TV")
    axes[1].set_title("relocation across the 64 query tokens")

    dm = (attn["mass_p"] - attn["mass_d"]) * 100  # (n, L, H, S, span)
    show = [s for s in ("vision", "text", "coc", "hist", "sink") if s in order]
    for s, c in zip(show, (C1, C2, C3, C4, MUTED)):
        axes[2].plot(np.arange(dm.shape[1]), dm[..., order.index(s)].mean((0, 2, 3)),
                     color=c, lw=1.6, label=s)
    axes[2].axhline(0, color=MUTED, lw=0.8)
    axes[2].set_xlabel("expert layer")
    axes[2].set_ylabel("Δ span mass (pp)")
    axes[2].set_title("mass BETWEEN spans barely moves")
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "attn_maps.png", dpi=150)
    plt.close(fig)

    # the judgement figure: coarse axis (span mass change) against fine axis (TV)
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    # only the disjoint coarse spans: "all" is the total and the fine spans are subsets of
    # "text", so summing every stored span would double-count the movement
    part = [order.index(s) for s in ("vision", "text", "hist", "sink", "coc") if s in order]
    dm_part = (attn["mass_p"] - attn["mass_d"])[..., part]
    xs = np.abs(dm_part).sum(-1).mean(0).ravel() * 100
    ys = tv.mean(-1).mean(0).ravel()
    ax.scatter(xs, ys, s=5, alpha=0.25, color=C1, edgecolors="none")
    ax.set_xlabel("total |Δ span mass| per (layer, head, step)  (pp)")
    ax.set_ylabel("TV (within-span relocation included)")
    ax.set_title("span mass vs full distribution")
    lim = max(xs.max(), 1e-9)
    ax.plot([0, lim], [0, lim / 100], color=MUTED, lw=1, ls="--",
            label="if all movement were between spans")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "attn_judgement.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="prunedvlm_dual")
    ap.add_argument("--floor", default="prunedvlm_nomask")
    ap.add_argument("--smoke", default="prunedvlm_smoke")
    ap.add_argument("--out", default="prunedvlm_analysis")
    ap.add_argument("--template", default=None)
    args = ap.parse_args()

    out = REPO / "outputs" / args.out
    (out / "plots").mkdir(parents=True, exist_ok=True)
    cfg, rows, res, attn = load(args.exp)
    cfg["_span_order"] = rows[0]["spans_present"]

    floor_rows = floor_attn = smoke_rows = None
    if (REPO / "outputs" / args.floor / "metrics.json").exists():
        _, floor_rows, _, floor_attn = load(args.floor)
    if (REPO / "outputs" / args.smoke / "metrics.json").exists():
        _, smoke_rows, _, _ = load(args.smoke)

    g = gates(cfg, rows, res, attn, floor_rows, floor_attn, smoke_rows)
    s = summarise(cfg, res, attn)
    plot_residual(cfg, res, out / "plots")
    plot_attn(cfg, attn, out / "plots")
    plot_detail(cfg, attn, out / "plots")
    xt = plot_cross_tower(cfg, res, attn, out / "plots")
    s["xt_pearson"], s["xt_spearman"] = xt["pearson"], xt["spearman"]

    def f(d):
        return f"{d['med']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]"

    lines = [(f"pruned VLM internals -- {cfg['config']} vs dense, {len(rows)} clips "
              f"({cfg['manifest']}[{cfg['clip_offset']}:])"),
             f"instrument: {cfg['instrument']}, gpu {cfg['gpu']}", "",
             "gates:",
             (f"  G0 layer0 cache dK max {g['G0_layer0_dk_max']:.1e}, "
              f"layer0 h_in rel max {g['G0_layer0_hin_rel_max']:.1e} -> "
              f"{'PASS' if g['G0_pass'] else 'FAIL'}")]
    if "G1_bitwise_frac" in g:
        lines.append(f"  G1 residual decomposition bitwise {g['G1_bitwise_frac']:.3f} -> "
                     f"{'PASS' if g['G1_pass'] else 'FAIL'} (from --verify-taps run)")
    lines.append(f"  G2 attention rowsum dev max {g['G2_rowsum_dev_max']:.2e} -> "
                 f"{'PASS' if g['G2_pass'] else 'FAIL'}")
    if "G3_rerun_tv_max" in g:
        lines.append(f"  G3-1 re-run floor: TV max {g['G3_rerun_tv_max']:.2e}")
    if "G3_bf16_floor_rel" in g:
        lines.append(f"  G3-2 bf16 rounding floor: residual rel {g['G3_bf16_floor_rel']:.2e}")

    lines += ["", "(B) expert attention over the cache -- clip median [95% CI]:",
              f"  TV                    {f(s['tv_overall'])}",
              f"  JS                    {f(s['js_overall'])}",
              f"  KL(dense||pruned)     {f(s['kl_pq'])}",
              f"  KL(pruned||dense)     {f(s['kl_qp'])}",
              f"  KL asymmetry          {f(s['kl_asym'])}",
              f"  d normalised entropy  {f(s['dent'])}", "",
              "  TV decomposition (the reading):",
              f"    between spans       {f(s['tv_between'])}",
              f"    within spans        {f(s['tv_within'])}",
              f"    within share        {s['within_share']:.1%}", "",
              ("  layer-matched link (VLM residual rel at h_in vs expert TV): "
               f"Pearson {s['xt_pearson']:+.2f}, Spearman {s['xt_spearman']:+.2f}"), "",
              "  span mass change (pp):"]
    for n, v in s["dmass_pp"].items():
        lines.append(f"    {n:9s} {f(v)}")
    lines += ["", "(A) VLM residual stream -- clip median over layers [95% CI]:"]
    for p in POINTS:
        lines.append(f"  rel({p:5s})  {f(s[f'rel_{p}'])}    cos {f(s[f'cos_{p}'])}")
    lines += [f"  gap opened by attention  {f(s['d_attn'])}",
              f"  gap opened by MLP        {f(s['d_mlp'])}", "",
              "  within-model transitions (dense / pruned):"]
    for t in TRANS:
        lines.append(f"    cos {t:4s}  {f(s[f'tcos_{t}_d'])}  /  {f(s[f'tcos_{t}_p'])}")
        lines.append(f"    mag {t:4s}  {f(s[f'tmag_{t}_d'])}  /  {f(s[f'tmag_{t}_p'])}")

    tv, fl = s["tv_overall"]["med"], g.get("G3_rerun_tv_max", 0.0)
    verdict = ("mass stays where it was but is REDISTRIBUTED inside the spans"
               if tv > 20 * max(fl, 1e-12) else
               "the expert reads the shifted cache the same way")
    lines += ["", "G4 reading:",
              (f"  span-level mass moves at most {s['dmass_abs_max_pp']:.3f} pp "
               f"(prior work: < 1 pp) while TV = {tv:.4f}, "
               f"{tv / max(fl, 1e-12):.0f}x the re-run floor"),
              f"  -> {verdict}"]
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    (out / "metrics.json").write_text(json.dumps({"gates": g, "summary": s,
                                                  "config": cfg}, indent=1, default=float))
    print("\n".join(lines))

    if args.template:
        t = Path(args.template)
        text = t.read_text()

        def lookup(tok):
            p = tok.strip().split(".")
            if p[0] == "g":
                v = g[p[1]]
                if not isinstance(v, float):
                    return str(v)
                return f"{v:.2e}" if v and abs(v) < 1e-2 else f"{v:.4f}"
            if p[0] == "n":
                return str(len(rows))
            d = s["dmass_pp"][p[1]] if p[0] == "dmass" else s[p[0]]
            if not isinstance(d, dict):
                if p[0].startswith("xt_"):
                    return f"{d:+.2f}"
                return f"{d:.1%}" if abs(d) <= 1 else f"{d:.4f}"
            k = p[-1] if len(p) > 1 else "ci"
            return {"med": f"{d['med']:+.4f}", "abs": f"{abs(d['med']):.4f}",
                    "ci": f(d)}.get(k, f(d))

        text = re.sub(r"\{\{([^}]+)\}\}", lambda mo: lookup(mo.group(1)), text)
        t.with_name(t.stem + "_filled.html").write_text(text)
        print("rendered ->", t.with_name(t.stem + "_filled.html"))
    print("saved ->", out)


if __name__ == "__main__":
    main()
