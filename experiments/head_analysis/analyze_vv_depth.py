"""Gates C1-C4 of plans/2026-09-21_importance-causal-validation.md: vision-vision interaction by depth.

Causal half (run_pathway2.py --cuts): an edge is blocked in the nested windows [l, 36)
("from l on": until what depth is the interaction still needed?) and, for VV, in [0, l)
("up to l": how much can later layers make up?). Damage = paired median difference to the
unblocked forward on the same clip, text and seeds, as a fraction of the baseline median --
the statistic of analyze_pathway2.py, so the curves continue the stored 9-layer map.
The depth profile of an interaction is the difference between successive cut points.

Observational half (run_vision_census.py): per head, the attention mass of vision queries
on {sink, text, own image, same-camera earlier frames, other cameras}.

  C1  the VV action-damage profile over the twelve 3-layer windows follows I_traj at vision
      tokens (Spearman >= 0.6, both axes)
  C2  I_traj at vision tokens peaks at layers 16-20. VV[18,36) >= 25% of VV[0,36) on minADE
      would put vision-vision interaction under that peak; < 10% (my prediction) says the
      interaction is finished before it; in between is undecided
  C3  every from-l-on action curve is inside +-5% of baseline for l >= 24
  C4  shared substrate: VV's NLL profile follows its action profile over the windows below
      layer 24 (Spearman >= 0.6); and cross-image attention mass predicts I_traj and I_CoC at
      vision tokens alike at head level (band-mean within-layer rho, layers 6-17, within 0.1)

Usage:
  python experiments/head_analysis/analyze_vv_depth.py --shards vvdepth_s0 vvdepth_s13 \
      vvdepth_s26 vvdepth_s38 --census vision_census_v1_s0 vision_census_v1_s13 \
      vision_census_v1_s26 vision_census_v1_s38 --out vvdepth_v1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))

from analyze_pathway2 import C1, C2, C3, C4, MUTED, NOISE_FLOOR, median_ci, merge, plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
EDGES = {"VV_allvision": "vision <- any other vision token", "E1_crossframe": "<- same-camera earlier frames",
         "E2_crosscam": "<- other cameras", "V3_ownimage": "<- own image"}
COL = {"VV_allvision": "black", "E1_crossframe": C1, "E2_crosscam": C2, "V3_ownimage": C4}
VIS = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--census", nargs="+", default=["vision_census_v1"], help="census run(s); shards are clip-weighted")
    ap.add_argument("--anatomy", default="gradanat_v1")
    ap.add_argument("--stored", default="pathway_e_v1", help="the 9-layer map, for the reproduction check")
    ap.add_argument("--stored-shards", nargs="+",
                    default=["pathway_e_s0", "pathway_e_s13", "pathway_e_s26", "pathway_e_s38"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    outs = REPO / "outputs"
    cfgs, _, per, _, clips = merge([outs / s for s in args.shards])
    cuts = sorted({int(c.split("@from")[1]) for c in cfgs if "@from" in c})
    n = len(clips)
    base = {f: np.array(per["E0_none"][f], float) for f in ("ade", "nll")}
    bmed = {f: float(np.median(base[f])) for f in base}
    out = {"n_clips": n, "cuts": cuts, "baseline_median": bmed}
    lines = [f"vision-vision interaction by depth -- {n} clips, cuts {cuts}",
             f"unblocked: minADE median {bmed['ade']:.4f}  CoC NLL median {bmed['nll']:.4f}", ""]

    # ------------------------------------------------------------------ integrity
    integ = np.array(per["E0_causalonly"]["nll"]) - base["nll"]
    out["integrity"] = {"causal_only_dnll_median": float(np.median(integ)),
                        "pass": bool(abs(np.median(integ)) < NOISE_FLOOR)}
    lines.append(f"INTEGRITY  causal-mask-only vs no mask: dNLL median {np.median(integ):+.2e} "
                 f"(< {NOISE_FLOOR}) -> {'PASS' if out['integrity']['pass'] else 'FAIL'}")
    try:
        _, _, old, _, old_clips = merge([outs / s for s in args.stored_shards])
        if old_clips == clips:
            rep = {}
            for new, stored in (("E0_none", "E0_none"), ("E1_crossframe@from0", "E1_crossframe@all"),
                                ("E2_crosscam@from0", "E2_crosscam@all")):
                if new in per:
                    rep[new] = {f: float(np.max(np.abs(np.array(per[new][f]) - np.array(old[stored][f]))))
                                for f in ("ade", "nll")}
            out["integrity"]["reproduces_stored_max_abs_diff"] = rep
            lines.append("           against the stored 9-layer map, same clips and seeds, max |diff| per clip: "
                         + "; ".join(f"{k} ade {v['ade']:.3f} nll {v['nll']:.4f}" for k, v in rep.items()))
        else:
            lines.append("           stored map has different clips: reproduction check skipped")
    except FileNotFoundError:
        lines.append("           stored shards not found: reproduction check skipped")
    lines.append("")

    # ------------------------------------------------------------------ nested curves
    def damage(cfg, f):
        d = np.array(per[cfg][f], float) - base[f]
        med, lo, hi = median_ci(d)
        return {"med": med, "lo": lo, "hi": hi, "rel": med / bmed[f], "rel_lo": lo / bmed[f],
                "rel_hi": hi / bmed[f], "mean_rel": float(d.mean() / base[f].mean())}

    curves = {}
    for e in EDGES:
        for fam in ("from", "upto"):
            pts = [c for c in cuts if f"{e}@{fam}{c}" in per]
            if pts:
                curves[f"{e}|{fam}"] = {c: {f: damage(f"{e}@{fam}{c}", f) for f in ("ade", "nll")} for c in pts}
    out["curves"] = {k: {str(c): v for c, v in d.items()} for k, d in curves.items()}
    for key, d in curves.items():
        e, fam = key.split("|")
        lines.append(f"{EDGES[e]}  blocked in " + ("[l, 36)" if fam == "from" else "[0, l)")
                     + "   (paired median / baseline median)")
        lines.append("   l      " + " ".join(f"{c:>7d}" for c in d))
        lines.append("   minADE " + " ".join(f"{v['ade']['rel']:+7.1%}" for v in d.values()))
        lines.append("   NLL    " + " ".join(f"{v['nll']['rel']:+7.1%}" for v in d.values()))
        lines.append("")

    # window profile: what each 3-layer window adds, given everything after it is already blocked
    def profile(e, f):
        vals = []
        for i, c in enumerate(cuts):
            a = np.array(per[f"{e}@from{c}"][f], float)
            b = np.array(per[f"{e}@from{cuts[i + 1]}"][f], float) if i + 1 < len(cuts) else base[f]
            vals.append(float(np.median(a - b)) / bmed[f])
        return np.array(vals)

    prof = {e: {f: profile(e, f) for f in ("ade", "nll")} for e in EDGES if f"{e}|from" in curves}
    out["window_profile"] = {e: {f: v.tolist() for f, v in d.items()} for e, d in prof.items()}
    z = np.load(outs / args.anatomy / "anatomy.npz")
    edges_w = cuts + [36]
    imp = {}
    for ax in ("q", "mlp"):
        for name, arr in (("I_traj@vision", z[f"fm_type_{ax}"][VIS]), ("I_CoC@vision", z[f"ce_{ax}"][VIS]),
                          ("I_traj", z[f"fm_full_{ax}"]), ("I_CoC", z[f"ce_full_{ax}"])):
            lay = arr.sum(1)  # (36,)
            imp[f"{ax}|{name}"] = np.array([lay[a:b].mean() for a, b in zip(edges_w[:-1], edges_w[1:])])

    vv = prof["VV_allvision"]
    c1 = {ax: float(spearmanr(vv["ade"], imp[f"{ax}|I_traj@vision"])[0]) for ax in ("q", "mlp")}
    c1["pass"] = bool(min(c1["q"], c1["mlp"]) >= 0.6)
    out["C1"] = c1
    lines += ["window profile of VV (what each 3-layer window adds to the damage, given all later layers blocked)",
              "   window  " + " ".join(f"{a:>3d}-{b - 1:<3d}" for a, b in zip(edges_w[:-1], edges_w[1:])),
              "   minADE  " + " ".join(f"{v:+7.1%}" for v in vv["ade"]),
              "   NLL     " + " ".join(f"{v:+7.1%}" for v in vv["nll"]),
              "   I_traj@vision (Q, / max)  " + " ".join(f"{v:7.2f}" for v in imp["q|I_traj@vision"] / imp["q|I_traj@vision"].max()),
              "   I_traj@vision (MLP, / max)" + " ".join(f"{v:7.2f}" for v in imp["mlp|I_traj@vision"] / imp["mlp|I_traj@vision"].max()),
              "",
              f"C1  Spearman(VV action profile, I_traj at vision tokens) over {len(cuts)} windows: Q {c1['q']:+.2f} "
              f"MLP {c1['mlp']:+.2f} (>= 0.6) -> {'PASS' if c1['pass'] else 'FAIL'}"]
    out["profile_spearman"] = {
        f"{e}|{f}|{k}": float(spearmanr(prof[e][f], v)[0]) for e in prof for f in ("ade", "nll") for k, v in imp.items()}

    full, late = curves["VV_allvision|from"][0]["ade"], curves["VV_allvision|from"][18]["ade"]
    frac = late["med"] / full["med"] if full["med"] > 0 else float("nan")
    verdict = ("interaction reaches under the importance peak (the hypothesis holds to the peak)" if frac >= 0.25
               else "interaction is finished before the importance peak (as predicted)" if frac < 0.10
               else "undecided")
    out["C2"] = {"vv_from18_rel": late["rel"], "vv_from0_rel": full["rel"], "fraction": float(frac), "verdict": verdict}
    lines.append(f"C2  VV[18,36) {late['rel']:+.1%} [{late['rel_lo']:+.1%},{late['rel_hi']:+.1%}] vs VV[0,36) {full['rel']:+.1%} "
                 f"[{full['rel_lo']:+.1%},{full['rel_hi']:+.1%}]: fraction {frac:.2f} -> {verdict}")

    c3 = {}
    for e in prof:
        pts = {c: curves[f"{e}|from"][c]["ade"] for c in cuts if c >= 24}
        c3[e] = {"max_abs_rel": float(max(abs(v["rel"]) for v in pts.values())),
                 "ci_inside": bool(all(v["rel_lo"] > -0.05 and v["rel_hi"] < 0.05 for v in pts.values()))}
    c3["pass"] = bool(all(v["max_abs_rel"] < 0.05 for v in c3.values()))
    out["C3"] = c3
    lines.append("C3  from-l-on action damage for l >= 24, largest |median| / baseline: "
                 + "; ".join(f"{e} {v['max_abs_rel']:.1%}{'' if v['ci_inside'] else ' (CI leaves the band)'}"
                             for e, v in c3.items() if e != "pass")
                 + f" -> {'PASS' if c3['pass'] else 'FAIL'}")

    below = [i for i, c in enumerate(cuts) if c < 24]
    c4 = {"nll_vs_action_profile": float(spearmanr(vv["ade"][below], vv["nll"][below])[0])}

    # ------------------------------------------------------------------ census
    cdirs = [outs / c for c in args.census if (outs / c / "census.npz").exists()]
    have_census = bool(cdirs)
    if have_census:
        # shards are means over their own clips: weight by clip count
        w = np.array([json.loads((c / "metrics.json").read_text())["n_clips"] for c in cdirs], float)
        mass = sum(wi * np.load(c / "census.npz")["mass"] for wi, c in zip(w, cdirs)) / w.sum()
        out["census_clips"] = int(w.sum())  # mass (36, 32, 5): sink, text, own image, cross frame, cross camera
        lay = mass.mean(1)  # (36, 5)
        cross = mass[..., 3] + mass[..., 4]  # (36, 32)
        out["census_layer_mean"] = lay.tolist()
        lines += ["", "census: attention mass of vision queries by layer band (mean over heads)",
                  f"   {'layers':8s} {'sink':>7s} {'text':>7s} {'own img':>8s} {'x-frame':>8s} {'x-camera':>9s}"]
        for a, b in ((0, 6), (6, 12), (12, 18), (18, 22), (22, 28), (28, 36)):
            lines.append(f"   {a:2d}-{b - 1:<5d} " + " ".join(f"{v:8.3f}" for v in lay[a:b].mean(0)))
        head = {}
        for name, s in (("I_traj@vision", z["fm_type_q"][VIS]), ("I_CoC@vision", z["ce_q"][VIS])):
            rho = np.array([spearmanr(cross[li], s[li])[0] for li in range(35)])
            head[name] = {"L6_17": float(rho[6:18].mean()), "L18_21": float(rho[18:22].mean()),
                          "L22_34": float(rho[22:35].mean()), "by_layer": rho.tolist()}
        c4["head_level_cross_image_mass_vs_importance"] = head
        c4["head_level_gap_L6_17"] = abs(head["I_traj@vision"]["L6_17"] - head["I_CoC@vision"]["L6_17"])
        lines.append("   within-layer Spearman over the 32 heads, cross-image mass vs importance at vision tokens "
                     "(L6-17 / L18-21 / L22-34): "
                     + "; ".join(f"{k} {v['L6_17']:+.2f} / {v['L18_21']:+.2f} / {v['L22_34']:+.2f}" for k, v in head.items()))
        lay_cross = cross.mean(1)[:35]
        out["census_layer_vs_importance"] = {
            k: float(spearmanr(lay_cross, z[key][VIS].sum(1)[:35])[0])
            for k, key in (("q|I_traj@vision", "fm_type_q"), ("mlp|I_traj@vision", "fm_type_mlp"),
                           ("q|I_CoC@vision", "ce_q"), ("mlp|I_CoC@vision", "ce_mlp"))}
        lines.append("   layer level, Spearman(cross-image mass, importance at vision tokens) over layers 0-34: "
                     + "; ".join(f"{k} {v:+.2f}" for k, v in out["census_layer_vs_importance"].items()))
    c4["pass"] = bool(c4["nll_vs_action_profile"] >= 0.6 and c4.get("head_level_gap_L6_17", 1.0) < 0.1)
    out["C4"] = c4
    lines += ["", f"C4  VV NLL profile vs action profile over the windows below layer 24: Spearman "
              f"{c4['nll_vs_action_profile']:+.2f} (>= 0.6); head-level gap between the two scores, L6-17: "
              f"{c4.get('head_level_gap_L6_17', float('nan')):.2f} (< 0.1) -> {'PASS' if c4['pass'] else 'FAIL'}"]

    # ------------------------------------------------------------------ plots
    out_dir = outs / args.out
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9))
    for ax_, f, title in ((axs[0], "ade", "minADE"), (axs[1], "nll", "CoC NLL")):
        for e in prof:
            d = curves[f"{e}|from"]
            x = list(d)
            ax_.plot(x, [d[c][f]["rel"] for c in x], color=COL[e], lw=1.5, marker="o", ms=3, label=EDGES[e])
            ax_.fill_between(x, [d[c][f]["rel_lo"] for c in x], [d[c][f]["rel_hi"] for c in x], color=COL[e], alpha=0.12)
        ax_.axhspan(-0.05, 0.05, color=MUTED, alpha=0.15)
        ax_.set_xlabel("edge blocked from layer l to the last")
        ax_.set_title(f"{title}: paired median change / baseline")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_dir / "plots" / "nested_from.png", dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9))
    centers = [(a + b - 1) / 2 for a, b in zip(edges_w[:-1], edges_w[1:])]
    for ax_, ax in ((axs[0], "q"), (axs[1], "mlp")):
        ax_.bar(centers, vv["ade"] / max(vv["ade"].max(), 1e-9), width=2.6, color=MUTED, alpha=0.5,
                label="VV action damage added by the window")
        for k, col in (("I_traj@vision", C1), ("I_CoC@vision", C3)):
            v = imp[f"{ax}|{k}"]
            ax_.plot(centers, v / v.max(), color=col, lw=1.6, marker="o", ms=3, label=k)
        ax_.set_title(f"window profiles / their maximum, {'Q heads' if ax == 'q' else 'MLP channels'}")
        ax_.set_xlabel("VLM layer")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_dir / "plots" / "window_profile_vs_importance.png", dpi=150)
    plt.close(fig)

    if "VV_allvision|upto" in curves:
        fig, ax_ = plt.subplots(figsize=(6.5, 3.8))
        for fam, col, lab in (("from", "black", "blocked in [l, 36)"), ("upto", C1, "blocked in [0, l)")):
            d = curves[f"VV_allvision|{fam}"]
            ax_.plot(list(d), [v["ade"]["rel"] for v in d.values()], color=col, lw=1.5, marker="o", ms=3, label=lab)
        ax_.axhspan(-0.05, 0.05, color=MUTED, alpha=0.15)
        ax_.set_xlabel("cut layer l")
        ax_.set_title("VV: minADE change / baseline")
        ax_.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / "plots" / "vv_from_vs_upto.png", dpi=150)
        plt.close(fig)

    if have_census:
        fig, ax_ = plt.subplots(figsize=(7, 3.8))
        ax_.stackplot(np.arange(36), lay.T, labels=("sink", "text", "own image", "same-camera earlier frames",
                                                    "other cameras"),
                      colors=(MUTED, "#c9c5b5", C4, C1, C2), alpha=0.85)
        ax_.set_xlabel("VLM layer")
        ax_.set_title("where vision queries put their attention (mean over heads)")
        ax_.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=3, frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / "plots" / "vision_query_census.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps(out, indent=2))
    (out_dir / "config.json").write_text(json.dumps({
        "plan": "plans/2026-09-21_importance-causal-validation.md (part C)", "shards": args.shards,
        "census": args.census, "anatomy": args.anatomy, "n_clips": n, "clip_ids": clips}, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
