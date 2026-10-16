"""Can a functional-redundancy signal see which layer-0 head sets collapse? No GPU.

plans/2026-10-14_l0-functional-redundancy.md. Inputs: the layer-0 o_proj input Hessian per
token stream from run_racfit.py --dump-hessian (calib_100), the layer-0 o_proj weight, and
the arms / outcomes / sets of make_l0_redundancy_masks.py. No outcome is used to build any
signal.

  G_hk = tr(W_h H_hk W_k^T)   the heads' contributions to the layer output, as a Gram matrix
  P1   e_mask(D)  = sum_{h,k in D} G_hk / sum G     output energy D removes, cross terms in
  S1   e_refit(D) = output energy left after the best linear refit from the kept heads
  S2   cluster coverage: average linkage on 1 - cos(G), 6 clusters; the largest
       energy-weighted fraction of one cluster D removes
  S3   e_mask on the vision stream alone (S3v) and on the model's own CoC alone (S3d)
  B0   sum over D of the shipped importance file's dual score (what the criterion believes)

Gates:
  G0   tr(H_V), tr(H_T) match racfit_g0h's stored layer-0 traces to a relative 1e-3
  G1   Spearman(P1, outcome) >= +0.45 with p < 0.01 on group A AND >= +0.20 with p < 0.05
       on group B
  G1b  P1 beats B0 on both groups (arm bootstrap of the rho difference, 95% CI above 0)
  S1-S3 are secondary, Holm-corrected within each group.
Step 2 (if its shards exist): P2 = mean FM loss on calib_100 with that layer-0 set under the
shipped mask elsewhere, same thresholds (G2).

Usage:
  python experiments/evaluation/analyze_l0_redundancy.py [--hessian-run l0red_h_calib100]
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from safetensors import safe_open
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

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
MODEL_REV = "7aba8293c09993f2e125c6819df05d7fa3e873ea"
W_KEY = "vlm.model.language_model.layers.0.self_attn.o_proj.weight"
N_HEADS, HD, BOOT, DAMP, N_CLUSTERS = 32, 128, 5000, 1e-2, 6
MARKED = (4, 7, 10, 24, 26, 30)     # heads the census / single-head map flagged


def o_proj_weight():
    from huggingface_hub import constants
    snap = (Path(constants.HF_HUB_CACHE) / "models--nvidia--Alpamayo-1.5-10B" / "snapshots"
            / MODEL_REV)
    shard = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"][W_KEY]
    with safe_open(snap / shard, framework="pt") as f:
        return f.get_tensor(W_KEY).double()                      # (4096 out, 4096 in)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hessian-run", default="l0red_h_calib100")
    ap.add_argument("--prop-shards", nargs="+",
                    default=[f"l0red_prop_s{s}" for s in range(4)])
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    out = O / "l0_redundancy"
    (out / "plots").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    lines, metrics = [], {"plan": "plans/2026-10-14_l0-functional-redundancy.md"}

    def say(s=""):
        print(s)
        lines.append(s)

    # ---- inputs ----
    dump = torch.load(O / args.hessian_run / "sol" / "L00_o_H.pt", map_location="cpu")
    Hs = {s: dump[s].double() for s in ("V", "T", "D")}
    n = {s: int(dump[f"n_{s}"]) for s in ("V", "T", "D")}
    W = o_proj_weight()
    sj = json.loads((O / "l0_redundancy" / "sets.json").read_text())
    arms, sets = sj["arms"], sj["sets"]

    # ---- G0 ----
    g0h = np.load(O / "racfit_g0h" / "racfit.npz")
    g0 = {}
    for s in ("V", "T", "D"):
        ours, ref = float(torch.trace(Hs[s])), float(g0h[f"diag_trace_{s}_o"][0])
        g0[s] = {"ours": ours, "racfit_g0h": ref, "rel": abs(ours - ref) / ref, "n_tokens": n[s]}
    g0_pass = g0["V"]["rel"] < 1e-3 and g0["T"]["rel"] < 1e-3
    metrics["G0"] = {"streams": g0, "pass": g0_pass}
    say("G0: layer-0 o_proj stream traces vs racfit_g0h (same clips; V and T do not depend on K)")
    for s, r in g0.items():
        say(f"  {s}: ours {r['ours']:.6g} vs {r['racfit_g0h']:.6g} (rel {r['rel']:.2e}), "
            f"{r['n_tokens']:,} tokens")
    say(f"  -> {'PASS' if g0_pass else 'FAIL'} (D is K=4 here vs K=1 there, so it is not gated)")

    # ---- the Gram matrix of head contributions ----
    H = Hs["V"] + Hs["T"] + Hs["D"]                          # the forward's own token mixture

    def gram(Hm):
        G = torch.zeros(N_HEADS, N_HEADS, dtype=torch.float64)
        for h in range(N_HEADS):
            left = W[:, h * HD:(h + 1) * HD].T @ W              # (128, 4096) = W_h^T W
            for k in range(N_HEADS):
                # G_hk = tr(W_h H_hk W_k^T) = sum_ij (W_h^T W_k)_ij (H_hk)_ij
                M = left[:, k * HD:(k + 1) * HD]                # (128, 128)
                G[h, k] = (M * Hm[h * HD:(h + 1) * HD, k * HD:(k + 1) * HD]).sum()
        return G.numpy()

    WH = W @ H                                                  # (4096, 4096)
    E_tot = float((WH * W).sum())                               # tr(W H W^T)
    G = gram(H)
    Gv, Gd = gram(Hs["V"]), gram(Hs["D"])
    tot = G.sum()
    assert abs(tot - E_tot) / E_tot < 1e-8, (tot, E_tot)        # the blocks tile the trace
    cos = G / np.sqrt(np.outer(np.diag(G), np.diag(G)))
    Z = linkage(squareform(1 - cos, checks=False), "average")
    clus = fcluster(Z, N_CLUSTERS, "maxclust")
    order = leaves_list(Z)

    # dual score of the shipped importance file at layer 0 (B0)
    imp = np.load(O / "importance_v2" / "importance.npz")

    def rn(x):
        return np.argsort(np.argsort(x)) / (len(x) - 1)

    dual0 = np.maximum(rn(imp["traj_vlm_q"][0]), rn(imp["coc_vlm_q"][0]))

    # e_refit: energy left after refitting the layer output from the kept heads' inputs
    def e_refit(drop):
        keep = [h for h in range(N_HEADS) if h not in drop]
        idx = torch.cat([torch.arange(h * HD, (h + 1) * HD) for h in keep])
        Hkk = H[idx][:, idx]
        Hkk = Hkk + DAMP * Hkk.diagonal().mean() * torch.eye(len(idx), dtype=torch.float64)
        B = WH[:, idx]                                          # (4096, |K|) = W H_{:,K}
        L = torch.linalg.cholesky(Hkk)
        X = torch.cholesky_solve(B.T, L)                        # (|K|, 4096)
        return float((E_tot - (B * X.T).sum()) / E_tot)

    def e_mask(Gm, drop):
        d = list(drop)
        return float(Gm[np.ix_(d, d)].sum() / Gm.sum())

    def cover(drop):
        en = np.diag(G)
        return float(max(en[[h for h in drop if clus[h] == c]].sum() / en[clus == c].sum()
                         for c in np.unique(clus)))

    sig = {}
    for name, s in sets.items():
        d = s["drop"]
        sig[name] = {"P1": e_mask(G, d), "S1": e_refit(d), "S2": cover(d),
                     "S3v": e_mask(Gv, d), "S3d": e_mask(Gd, d), "B0": float(dual0[d].sum())}

    # P2 from step 2's shards, if present: mean FM loss (and NLL) with that set in layer 0
    # a shard is an exp id (outputs/<id>/metrics.json) or a saved metrics file; the first
    # attempt was killed at clip 20-21 of 25, so its checkpoints are combined with a run of
    # the 20 missing clips (seeds are clip-derived, so the split does not change any value)
    prop = [O / s if s.endswith(".json") else O / s / "metrics.json" for s in args.prop_shards]
    ms = [json.loads(p.read_text()) for p in prop] if all(p.exists() for p in prop) else []
    clip_ids = [c for m in ms for c in m["clip_ids"][: len(m["per_clip"]["dense"]["fm"])]]
    # shards checkpoint as they go: read them only once all 100 clips are in, each once
    if ms and len(clip_ids) == 100 and len(set(clip_ids)) == 100:
        for q, key in (("fm", "P2"), ("nll", "P2nll")):
            for name in sets:
                cfg = f"arm_{name}"
                if all(cfg in m["per_clip"] for m in ms):
                    v = np.concatenate([np.asarray(m["per_clip"][cfg][q], float) for m in ms])
                    sig[name][key] = float(v.mean())
        dense_fm = np.concatenate([np.asarray(m["per_clip"]["dense"]["fm"], float) for m in ms])
        metrics["P2_dense_fm"] = float(dense_fm.mean())

    # ---- per-head view ----
    say(f"\nlayer-0 head contributions (natural mixture: V {n['V']:,} / T {n['T']:,} / "
        f"D {n['D']:,} tokens); {N_CLUSTERS} clusters by output direction")
    say("  head  energy share  cluster  dual score  most-aligned other head (cos)")
    for h in range(N_HEADS):
        c2 = cos[h].copy()
        c2[h] = -np.inf
        j = int(np.argmax(c2))
        flag = " *" if h in MARKED else ""
        say(f"  {h:4d}{G[h, h] / np.trace(G):13.3f}{clus[h]:9d}{dual0[h]:12.3f}      "
            f"{j:2d} ({cos[h, j]:+.2f}){flag}")
    say(f"  off-diagonal share of output energy: {(tot - np.trace(G)) / tot:+.3f} "
        f"(positive = heads write overlapping directions)")
    metrics["heads"] = {"energy_share": (np.diag(G) / np.trace(G)).tolist(),
                        "cluster": clus.tolist(), "dual0": dual0.tolist(),
                        "offdiag_share": float((tot - np.trace(G)) / tot)}

    # ---- rank tests ----
    names = ["P1", "S1", "S2", "S3v", "S3d", "B0"] + [k for k in ("P2", "P2nll")
                                                       if any(k in v for v in sig.values())]

    def boot_rho(x, y, x2=None):
        idx = rng.integers(0, len(y), (BOOT, len(y)))
        r = np.array([spearmanr(x[i], y[i])[0] for i in idx])
        if x2 is None:
            return np.percentile(r, [2.5, 97.5]).tolist()
        r2 = np.array([spearmanr(x2[i], y[i])[0] for i in idx])
        return np.percentile(r - r2, [2.5, 97.5]).tolist()

    res = {}
    for g, label in (("A", "A: 50 dual-family arms, test500"),
                     ("B", "B: 64 half-split arms, val 150 delta"),
                     ("C", "C: single-head map, cruise60")):
        ga = [a for a in arms if a["group"] == g]
        y = np.array([a["y"] for a in ga])
        say(f"\n== group {label}  ({len(ga)} arms, {len({a['set'] for a in ga})} distinct sets)")
        say(f"  {'signal':6s}{'rho arms':>10s}{'p':>10s}{'95% CI':>18s}{'rho sets':>10s}{'p':>10s}")
        res[g] = {}
        for s in names:
            if g == "C" and s.startswith("P2"):
                continue
            x = np.array([sig[a["set"]][s] for a in ga])
            if np.ptp(x) == 0:
                # S2: head 23 is a cluster of its own and every arm drops it, so the largest
                # fraction of one cluster removed is 1 for every set -- the signal is undefined
                res[g][s] = {"degenerate": True, "value": float(x[0])}
                say(f"  {s:6s}  constant {x[0]:.3f} across the group: degenerate, not tested")
                continue
            r, p = spearmanr(x, y)
            # distinct-set level: outcome averaged over arms sharing a set
            us = sorted({a["set"] for a in ga})
            ys = np.array([np.mean([a["y"] for a in ga if a["set"] == u]) for u in us])
            xs = np.array([sig[u][s] for u in us])
            rs, ps = spearmanr(xs, ys)
            lo_hi = boot_rho(x, y) if g != "C" else [np.nan, np.nan]
            res[g][s] = {"rho": float(r), "p": float(p), "ci": lo_hi, "rho_sets": float(rs),
                         "p_sets": float(ps), "n": len(ga)}
            say(f"  {s:6s}{r:+10.3f}{p:10.2g}   [{lo_hi[0]:+.2f}, {lo_hi[1]:+.2f}]{rs:+10.3f}{ps:10.2g}")
        if g in ("A", "B"):
            xP = np.array([sig[a["set"]]["P1"] for a in ga])
            xB = np.array([sig[a["set"]]["B0"] for a in ga])
            res[g]["P1_minus_B0_ci"] = boot_rho(xP, y, xB)
            say(f"  rho(P1) - rho(B0), 95% CI {res[g]['P1_minus_B0_ci'][0]:+.2f} to "
                f"{res[g]['P1_minus_B0_ci'][1]:+.2f}; rho(P1, B0) across arms "
                f"{spearmanr(xP, xB)[0]:+.3f}")
            if "P2" in res[g]:
                x2 = np.array([sig[a["set"]]["P2"] for a in ga])
                res[g]["P2_minus_B0_ci"] = boot_rho(x2, y, xB)
                say(f"  rho(P2) - rho(B0), 95% CI {res[g]['P2_minus_B0_ci'][0]:+.2f} to "
                    f"{res[g]['P2_minus_B0_ci'][1]:+.2f}; rho(P2, B0) {spearmanr(x2, xB)[0]:+.3f}, "
                    f"rho(P2, P1) {spearmanr(x2, xP)[0]:+.3f}")
            # Holm over the secondaries that are defined in this group
            sec = sorted((res[g][s]["p"], s) for s in ("S1", "S2", "S3v", "S3d")
                         if not res[g][s].get("degenerate"))
            m = len(sec)
            holm, run = {}, 0.0
            for i, (p, s) in enumerate(sec):
                run = max(run, min(1.0, (m - i) * p))
                holm[s] = run
            res[g]["holm_secondary"] = holm
            say("  Holm-adjusted p, secondaries: " + ", ".join(f"{s} {v:.2g}" for s, v in holm.items()))
    cen = json.loads((O / "swap_masks" / "metrics_census.json").read_text())
    say(f"\ncontext (outcome-informed, census): n_off rho A {cen['rho_noff_test'][0]:+.3f}, "
        f"B {cen['rho_noff_halfsplit'][0]:+.3f}; head-indicator ridge B->A "
        f"{cen['transfer']['halfsplit_to_test'][0]:+.3f}, A->B {cen['transfer']['test_to_halfsplit'][0]:+.3f}")

    def gate(s):
        a, b = res["A"][s], res["B"][s]
        return a["rho"] >= 0.45 and a["p"] < 0.01 and b["rho"] >= 0.20 and b["p"] < 0.05

    g1 = gate("P1")
    g1b = res["A"]["P1_minus_B0_ci"][0] > 0 and res["B"]["P1_minus_B0_ci"][0] > 0
    metrics.update({"signals": sig, "results": res, "G1_pass": g1, "G1b_pass": g1b})
    say(f"\nG1 (P1 = e_mask): {'PASS' if g1 else 'FAIL'}   G1b (P1 beats B0 on A and B): "
        f"{'PASS' if g1b else 'FAIL'}")
    if "P2" in res["A"]:
        metrics["G2_pass"] = gate("P2")
        say(f"G2 (P2 = propagated FM loss): {'PASS' if metrics['G2_pass'] else 'FAIL'}")

    # ---- plots ----
    cmap = LinearSegmentedColormap.from_list("div", [C1, "#D9D6CC", C3])
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(cos[np.ix_(order, order)], cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(N_HEADS))
    ax.set_yticks(range(N_HEADS))
    lab = [f"{h}{'*' if h in MARKED else ''}" for h in order]
    ax.set_xticklabels(lab, fontsize=7, rotation=90)
    ax.set_yticklabels(lab, fontsize=7)
    ax.set_title("Layer 0: cosine between heads' output contributions (cluster order)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="cos(G)")
    fig.tight_layout()
    fig.savefig(out / "plots" / "l0_head_cosine.png", dpi=150)
    plt.close(fig)

    rows = [("P1", "P1: share of layer-0 output energy removed")]
    if "P2" in res["A"]:
        rows.append(("P2", "P2: mean FM loss on calib_100, this set in layer 0"))
    fig, axs = plt.subplots(len(rows), 2, figsize=(10, 4 * len(rows)), squeeze=False)
    for r, (s, xlab) in enumerate(rows):
        for c, (g, ylab) in enumerate((("A", "test500 minADE@6"), ("B", "val150 delta minADE@6"))):
            ax = axs[r, c]
            ga = [a for a in arms if a["group"] == g]
            x = [sig[a["set"]][s] for a in ga]
            ax.plot(x, [a["y"] for a in ga], "o", color=C1, ms=8, mec=BG, mew=1.5, alpha=0.85)
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
            ax.set_title(f"group {g}, {s}: rho {res[g][s]['rho']:+.2f} (p={res[g][s]['p']:.2g})")
    fig.tight_layout()
    fig.savefig(out / "plots" / "l0_signals_vs_outcome.png", dpi=150)
    plt.close(fig)

    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    (out / "config.json").write_text(json.dumps({
        "plan": "plans/2026-10-14_l0-functional-redundancy.md", "hessian_run": args.hessian_run,
        "prop_shards": args.prop_shards, "weight": W_KEY, "model_revision": MODEL_REV,
        "damp": DAMP, "n_clusters": N_CLUSTERS, "boot_rho": BOOT}, indent=2))


if __name__ == "__main__":
    main()
