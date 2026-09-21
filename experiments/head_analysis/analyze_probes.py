"""Gates D1-D4 of plans/2026-09-21_importance-causal-validation.md: random-readout probes.

run_gradient_anatomy.py --probes measures, on the dense model and the same clips, the
token-resolved gate importance of five "losses": the CoC NLL and the FM loss (the shipped
scores) and three random linear readouts with no language or driving content in them --
through the LM head's input (head), through the expert's output (expert), and straight off
every cache entry (cache).

  D1  the expert-door probe reproduces I_traj's layer profile (Pearson >= 0.95) and the
      log-ratio expert-door / head-door has its change point at layer 23
  D2  the head-door probe reproduces I_CoC's layer profile (Pearson >= 0.90)
  D3  in layers 6-21 each probe ranks Q heads like the real loss through its door
      (split-half ceiling corrected rho >= 0.8)
  D4  the ports are the expert's: the cache-side gradient profile of the expert-door probe
      follows the FM loss's over cache layers 16-35 (Pearson >= 0.9), while the bare cache
      door, which has no expert in it, shows no step at 22-23 on the unit side.
      The plan worded D4 on the unit-side port profile; that needs 72 backwards per clip per
      probe and was not run, so the cache-side profile (what arrives at each cache layer) is
      the quantity evaluated here, and its relation to the unit-side port share of
      portmap_v1 is reported next to it.

Usage:
  python experiments/head_analysis/analyze_probes.py --exp gradanat_probes_v1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from analyze_gradient_anatomy import (  # noqa: E402
    C1, C2, C3, C4, LAST, LATE, MUTED, VIS, corrected, plt, shares, split_half,
)
from analyze_pruned_anatomy import step_fit  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
TRUNK = slice(6, 22)
SCORES = {"I_traj": "fm", "I_CoC": "ce", "head probe": "pr_head", "expert probe": "pr_expert",
          "cache probe": "pr_cache"}
COL = {"I_traj": C1, "I_CoC": C3, "head probe": C3, "expert probe": C1, "cache probe": C4}


def full(z, key, ax):
    return z[f"{key}_full_{ax}"]  # (L, U) E|sum over types|


def typed(z, key, ax):
    return z[{"fm": "fm_type", "ce": "ce"}.get(key, key) + f"_{ax}"]  # (5, L, U) E|G_type|


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="gradanat_probes_v1")
    ap.add_argument("--portmap", default="portmap_v1")
    ap.add_argument("--n-split", type=int, default=50)
    args = ap.parse_args()

    d = REPO / "outputs" / args.exp
    (d / "plots").mkdir(exist_ok=True)
    z = dict(np.load(d / "anatomy.npz"))
    pq = dict(np.load(d / "anatomy_perclip_q.npz"))
    n = json.loads((d / "metrics.json").read_text())["n_clips"]
    rng = np.random.default_rng(0)
    out, lines = {"n_clips": n}, [f"random-readout probes -- {args.exp}, {n} clips", ""]

    # ------------------------------------------------------------------ layer profiles
    prof = {ax: {name: full(z, key, ax).sum(1)[:LAST] for name, key in SCORES.items()} for ax in ("q", "mlp")}
    lines.append("layer profiles (sum over units of E|G|, layers 0-34): Pearson between scores")
    out["profile_pearson"] = {}
    for ax in ("q", "mlp"):
        names = list(SCORES)
        m = np.corrcoef(np.stack([prof[ax][k] for k in names]))
        out["profile_pearson"][ax] = {a: {b: float(m[i, j]) for j, b in enumerate(names)} for i, a in enumerate(names)}
        lines.append(f"  {ax}: " + " ".join(f"{k[:12]:>13s}" for k in names))
        for i, a in enumerate(names):
            lines.append(f"  {a:13s}" + " ".join(f"{m[i, j]:13.3f}" for j in range(len(names))))
    lines.append("")

    lines.append("change point of log-ratios by depth (two-level least-squares step; tau, step, R2)")
    out["steps"] = {}
    pairs = (("I_traj", "I_CoC"), ("expert probe", "head probe"), ("cache probe", "head probe"),
             ("I_traj", "expert probe"), ("expert probe", "cache probe"), ("I_CoC", "head probe"))
    for ax in ("q", "mlp"):
        for a, b in pairs:
            tau, step, r2 = step_fit(np.log(prof[ax][a] / prof[ax][b]))
            out["steps"][f"{ax}|{a}/{b}"] = {"tau": tau, "step": step, "r2": r2}
            lines.append(f"  {ax:3s} {a:12s} / {b:12s}: tau {tau:2d}  step {step:6.2f}x  R2 {r2:.2f}")
    lines.append("")

    d1 = {ax: {"pearson": out["profile_pearson"][ax]["expert probe"]["I_traj"],
               "tau": out["steps"][f"{ax}|expert probe/head probe"]["tau"]} for ax in ("q", "mlp")}
    d1["pass"] = bool(all(v["pearson"] >= 0.95 and v["tau"] == 23 for v in d1.values()))
    d2 = {ax: out["profile_pearson"][ax]["head probe"]["I_CoC"] for ax in ("q", "mlp")}
    d2["pass"] = bool(min(d2["q"], d2["mlp"]) >= 0.90)
    out["D1"], out["D2"] = d1, d2
    lines += [f"D1  expert-door probe vs I_traj profile: Pearson Q {d1['q']['pearson']:.3f} MLP {d1['mlp']['pearson']:.3f} "
              f"(>= 0.95); change point of expert/head probe ratio Q {d1['q']['tau']} MLP {d1['mlp']['tau']} (= 23) "
              f"-> {'PASS' if d1['pass'] else 'FAIL'}",
              f"D2  head-door probe vs I_CoC profile: Pearson Q {d2['q']:.3f} MLP {d2['mlp']:.3f} (>= 0.90) "
              f"-> {'PASS' if d2['pass'] else 'FAIL'}", ""]

    # ------------------------------------------------------------------ token hand-over of the probes
    lines.append("share of E|G| at vision tokens (L6-17 -> L27-34); the probes hand over too?")
    out["vision_share"] = {}
    for ax in ("q", "mlp"):
        for name, key in SCORES.items():
            sh = shares(typed(z, key, ax))[VIS]  # (36,)
            out["vision_share"][f"{ax}|{name}"] = sh.tolist()
            lines.append(f"  {ax:3s} {name:13s} {sh[6:18].mean():.3f} -> {sh[27:LAST].mean():.3f}")
    lines.append("")

    # ------------------------------------------------------------------ D3 within-layer ranks
    lines.append("D3  does a probe rank Q heads like the real loss through its door? "
                 "(split-half ceiling corrected; trunk = 6-21, late = 22-34)")
    d3 = {}
    real = {"expert": np.abs(pq["fm_full"].sum(1)), "head": np.abs(pq["ce"].sum(1)),
            "cache": np.abs(pq["fm_full"].sum(1))}  # (N, 36, 32)
    for name in ("head", "expert", "cache"):
        p = np.abs(pq[f"pr_{name}"].sum(1))  # (N, 36, 32)
        st, sc_, cr = split_half(p[:, :LAST], real[name][:, :LAST], args.n_split, rng)
        d3[name] = {"trunk": corrected(st, sc_, cr, TRUNK), "late": corrected(st, sc_, cr, LATE)}
        t, lt = d3[name]["trunk"], d3[name]["late"]
        lines.append(f"  {name:6s} probe vs {'I_CoC' if name == 'head' else 'I_traj'}: trunk self "
                     f"{t['self_traj']:.2f}/{t['self_coc']:.2f} cross {t['cross']:.2f} corrected {t['corrected']:.3f} | "
                     f"late corrected {lt['corrected']:.3f}")
    st, sc_, cr = split_half(np.abs(pq["pr_expert"].sum(1))[:, :LAST], np.abs(pq["pr_head"].sum(1))[:, :LAST],
                             args.n_split, rng)
    d3["expert_vs_head_probe"] = {"trunk": corrected(st, sc_, cr, TRUNK), "late": corrected(st, sc_, cr, LATE)}
    lines.append(f"  expert probe vs head probe (the two doors, no objective): trunk corrected "
                 f"{d3['expert_vs_head_probe']['trunk']['corrected']:.3f} | late "
                 f"{d3['expert_vs_head_probe']['late']['corrected']:.3f}")
    d3["pass"] = bool(d3["head"]["trunk"]["corrected"] >= 0.8 and d3["expert"]["trunk"]["corrected"] >= 0.8)
    lines += [f"    -> {'PASS' if d3['pass'] else 'FAIL'} (head and expert probes, trunk, >= 0.8)", ""]
    out["D3"] = d3

    # ------------------------------------------------------------------ D4 cache-side profiles
    lines.append("D4  what arrives at each cache layer (share over cache layers; FM loss vs expert-door probe)")
    d4 = {}
    for key_fm, key_pr, label in (("direct", "direct_probe", "|k dL/dk| + |v dL/dv|"),
                                  ("cg_fm", "cg_probe", "gradient energy")):
        a, b = z[key_fm].sum(1), z[key_pr].sum(1)  # (36,) over position types
        a, b = a / a.sum(), b / b.sum()
        d4[label] = {"fm": a.tolist(), "probe": b.tolist(),
                     "pearson_16_35": float(np.corrcoef(a[16:], b[16:])[0, 1]),
                     "pearson_all": float(np.corrcoef(a, b)[0, 1])}
        lines.append(f"  {label:24s} Pearson over cache 16-35 {d4[label]['pearson_16_35']:.3f} (all 36: "
                     f"{d4[label]['pearson_all']:.3f}); top FM layers {np.argsort(-a)[:5].tolist()} "
                     f"probe {np.argsort(-b)[:5].tolist()}")
    ppath = REPO / "outputs" / args.portmap / "port_map.npz"
    if ppath.exists():
        pm = np.load(ppath)
        for ax in ("q", "mlp"):
            port = pm[f"{ax}_signed"].sum((1, 2, 3))
            port = port / port.sum()
            a = np.array(d4["|k dL/dk| + |v dL/dv|"]["fm"])
            d4[f"unit_side_port_{ax}_vs_cache_side"] = float(np.corrcoef(port[16:], a[16:])[0, 1])
            lines.append(f"  unit-side port share ({args.portmap}, {ax}) vs cache-side FM profile, cache 16-35: "
                         f"Pearson {d4[f'unit_side_port_{ax}_vs_cache_side']:.3f}")
    bare = {ax: out["steps"][f"{ax}|cache probe/head probe"] for ax in ("q", "mlp")}
    expd = {ax: out["steps"][f"{ax}|expert probe/cache probe"] for ax in ("q", "mlp")}
    d4["bare_cache_step"] = bare
    d4["expert_over_bare_step"] = expd
    smooth = {}
    for ax in ("q", "mlp"):
        p = prof[ax]["cache probe"]
        smooth[ax] = float(max(p[i] / (0.5 * (p[i - 1] + p[i + 1])) for i in range(1, LAST - 1)))
    d4["bare_cache_max_over_neighbours"] = smooth
    d4["pass"] = bool(d4["|k dL/dk| + |v dL/dv|"]["pearson_16_35"] >= 0.9
                      and all(v < 2 for v in smooth.values()))
    lines += [f"  bare cache door, unit side: largest layer / mean of its neighbours Q {smooth['q']:.2f} MLP "
              f"{smooth['mlp']:.2f} (< 2 = smooth); ratio to head probe steps at Q {bare['q']['tau']} "
              f"({bare['q']['step']:.2f}x) MLP {bare['mlp']['tau']} ({bare['mlp']['step']:.2f}x); expert/bare steps at "
              f"Q {expd['q']['tau']} ({expd['q']['step']:.2f}x) MLP {expd['mlp']['tau']} ({expd['mlp']['step']:.2f}x)",
              f"    -> {'PASS' if d4['pass'] else 'FAIL'} (cache-side Pearson >= 0.9 and a smooth bare-cache profile)", ""]
    out["D4"] = d4

    # ------------------------------------------------------------------ plots
    L = np.arange(LAST)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9))
    for ax_i, ax in enumerate(("q", "mlp")):
        for name in SCORES:
            p = prof[ax][name]
            axs[ax_i].plot(L, p / p.max(), color=COL[name], lw=1.6 if name.startswith("I_") else 1.1,
                           ls="-" if name.startswith("I_") else (0, (4, 2)), label=name)
        axs[ax_i].set_title(f"layer profile / its maximum, {'Q heads' if ax == 'q' else 'MLP channels'}")
        axs[ax_i].set_xlabel("VLM layer")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(d / "plots" / "probe_profiles.png", dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9))
    for ax_i, ax in enumerate(("q", "mlp")):
        for (a, b), col in zip(pairs[:3], ("black", C1, C4)):
            axs[ax_i].plot(L, prof[ax][a] / prof[ax][b] / (prof[ax][a] / prof[ax][b])[:22].mean(),
                           color=col, lw=1.4, label=f"{a} / {b}")
        axs[ax_i].set_yscale("log")
        axs[ax_i].axvline(22.5, color=MUTED, lw=0.7)
        axs[ax_i].set_title(f"ratio by depth, scaled to its layer 0-21 mean, {'Q heads' if ax == 'q' else 'MLP'}")
        axs[ax_i].set_xlabel("VLM layer")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(d / "plots" / "probe_ratios.png", dpi=150)
    plt.close(fig)

    fig, ax_ = plt.subplots(figsize=(6.5, 3.6))
    r = d4["|k dL/dk| + |v dL/dv|"]
    ax_.plot(np.arange(36), r["fm"], color=C1, lw=1.5, label="FM loss")
    ax_.plot(np.arange(36), r["probe"], color=C2, lw=1.2, ls=(0, (4, 2)), label="expert-door probe")
    ax_.set_xlabel("cache layer")
    ax_.set_title("share of the cache-side first-order read, by cache layer")
    ax_.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(d / "plots" / "cache_side_profile.png", dpi=150)
    plt.close(fig)

    (d / "summary.txt").write_text("\n".join(lines) + "\n")
    (d / "metrics_analysis.json").write_text(json.dumps(out, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
