"""Does compression cost concentrate on the scenes Claude calls long-tail? Stored open-loop rows only.

Splits the A100 per-clip open-loop records (outputs/cvpr27/a100/<arm>_{test2000,oodval}, k=8, seed 42,
`run_baseline.py --no-tf`) by the Claude label from `label_longtail.py`, and per arm reports

  - minADE@6 / minFDE@6 on long-tail (category != NOMINAL) and NOMINAL clips, absolute values first
  - d = arm - dense, paired per clip, with a bootstrap CI, on each subset
  - DiD = d_longtail - d_nominal: is the cost larger on long-tail clips?
  - DiD_matched: the same, but compared within GT manoeuvre bucket (eval_lib.bucket, stored per row)
    and weighted by the long-tail clips' bucket mix, because long-tail scenes may simply contain
    more stops and turns, which cost more for every arm
  - CoC degeneracy rate on each subset
  - the same split into event long-tail (NVIDIA's eight named clusters) and OTHER_LONGTAIL, which on
    test_2000 is mostly conditions (snow, night rain, glare, fog) rather than events

No GPU, no model: the arms were all evaluated on these clips already.

  .venv/bin/python cvpr27/long-tail-detection/analyze_longtail_openloop.py --labels label_v1 --out openloop_v1
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
A100 = REPO / "outputs" / "cvpr27" / "a100"
OUT_ROOT = REPO / "outputs" / "cvpr27" / "long_tail_detection"

ARMS = {  # name -> A100 dir prefix; the table order of 2026-11-02_a100-results-summary.md
    "dense": "dense", "MoOP": "moop", "MoOP l0keep": "moop_l0keep",
    "l0keep u35": "l0keep_u35", "l0keep u45": "l0keep_u45", "l0keep u50": "l0keep_u50",
    "FM-only l0keep": "fm_only_l0keep", "CE-only l0keep": "ce_only_l0keep",
    "L1-4 MLP 95%": "l0keep_l14m95", "L1-4 MLP 90%": "l0keep_l14m90", "L1-4 MLP 80%": "l0keep_l14m80",
    "LLM-Pruner lp_m24": "lp_m24", "LLM-Pruner lp_r50": "lp_r50",
    "Tyr tyr_off": "tyr_off", "Tyr tyr_u40_r": "tyr_u40_r",
}
SETS = {"test_2000": "test2000", "ood_val": "oodval"}
# NVIDIA's eight named event clusters; OTHER_LONGTAIL is kept apart because on test_2000 Claude
# fills it mostly with conditions (snow, night rain, glare, fog), not events
EVENT = {"WORK_ZONES_TEMP_TRAFFIC_CONTROL", "PEDESTRIAN_DENSITY_OR_CLOSE_PROXIMITY",
         "SPECIAL_OR_UNCOMMON_VEHICLE_BEHAVIOR", "CYCLISTS_AND_MICROMOBILITY_COMPLEX",
         "COMPLEX_INTERSECTION_INTERACTION", "EMERGENCY_INCIDENT_SCENE", "ROAD_DEBRIS_OR_SAFETY_TRACES",
         "ANIMALS_BIRDS_ROADKILL"}
LT_GROUPS = ("longtail", "lt_event", "lt_other")
GROUP_NAME = {"nominal": "nominal", "lt_event": "long-tail: event", "lt_other": "long-tail: other/condition"}
# validated with dataviz validate_palette (light surface): CVD PASS; aqua contrast WARN -> legend + tables
COLOR = {"nominal": "#2a78d6", "lt_event": "#eb6834", "lt_other": "#1baf7a"}


def load_arm(prefix, suffix):
    rows = []
    for f in sorted(glob.glob(str(A100 / f"{prefix}_{suffix}" / "*_s*of*.json"))):
        rows += json.loads(Path(f).read_text())
    df = pd.DataFrame(rows).drop_duplicates("clip_id").set_index("clip_id")
    return pd.DataFrame({
        "ade6": [min(k[:6]) for k in df.ade_rollout_k],     # minADE@6 = min over the first 6 of 8 seeds
        "fde6": [min(k[:6]) for k in df.fde_rollout_k],
        "degen": df.coc_degenerate.astype(float), "bucket": df.bucket}, index=df.index)


def boot_means(x, n_boot, rng, chunk=5000):
    """(n_boot,) bootstrap means of x."""
    x = np.asarray(x, dtype=float)
    out = np.empty(n_boot)
    for s in range(0, n_boot, chunk):
        e = min(s + chunk, n_boot)
        out[s:e] = x[rng.integers(0, len(x), size=(e - s, len(x)))].mean(axis=1)
    return out


def ci(b):
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def did_matched(d_lt, b_lt, d_nom, b_nom, n_boot, rng):
    """sum_b p_LT(b) * (mean d_LT|b - mean d_NOM|b), over buckets present in both; resampled within cells."""
    buckets = [b for b in pd.unique(b_lt) if (b_nom == b).any()]
    w = np.array([(b_lt == b).sum() for b in buckets], dtype=float)
    w /= w.sum()
    point = sum(wi * (d_lt[b_lt == b].mean() - d_nom[b_nom == b].mean()) for wi, b in zip(w, buckets))
    boot = sum(wi * (boot_means(d_lt[b_lt == b], n_boot, rng) - boot_means(d_nom[b_nom == b], n_boot, rng))
               for wi, b in zip(w, buckets))
    return float(point), ci(boot), {b: float(wi) for b, wi in zip(buckets, w)}


def analyze_set(set_name, labels, n_boot, rng):
    lab = labels[labels.set == set_name].set_index("clip_id")
    arms = {a: load_arm(p, SETS[set_name]) for a, p in ARMS.items()}
    ids = lab.index.intersection(arms["dense"].index)
    assert len(ids) == len(lab), f"{set_name}: {len(lab) - len(ids)} labelled clips have no dense row"
    cat = lab.loc[ids, "category"]
    groups = {"longtail": ids[(cat != "NOMINAL").values], "lt_event": ids[cat.isin(EVENT).values],
              "lt_other": ids[(cat == "OTHER_LONGTAIL").values], "nominal": ids[(cat == "NOMINAL").values]}
    dense = arms["dense"].loc[ids]
    res = {"n": {g: len(v) for g, v in groups.items()}, "arms": {}}
    for a, df in arms.items():
        df = df.loc[ids]
        r = {}
        for g, gi in groups.items():
            r[g] = {"ade6": float(df.loc[gi, "ade6"].mean()), "fde6": float(df.loc[gi, "fde6"].mean()),
                    "coc_degen": float(df.loc[gi, "degen"].mean())}
        r["all"] = {"ade6": float(df.ade6.mean()), "fde6": float(df.fde6.mean())}
        if a != "dense":
            d = df.ade6 - dense.ade6                                  # (n_clips,) paired per clip
            bt = {}
            for g, gi in groups.items():
                bt[g] = boot_means(d.loc[gi].values, n_boot, rng)
                r[g].update({"d_ade6": float(d.loc[gi].mean()), "d_ade6_ci": ci(bt[g]),
                             "d_fde6": float((df.fde6 - dense.fde6).loc[gi].mean())})
            gn = groups["nominal"]
            r["did"] = {}
            for g in LT_GROUPS:
                gl = groups[g]
                mp, mci, w = did_matched(d.loc[gl].values, df.loc[gl, "bucket"].values, d.loc[gn].values,
                                         df.loc[gn, "bucket"].values, n_boot, rng)
                r["did"][g] = {"point": float(r[g]["d_ade6"] - r["nominal"]["d_ade6"]),
                               "ci": ci(bt[g] - bt["nominal"]), "matched": mp, "matched_ci": mci,
                               "bucket_weights": w}
        res["arms"][a] = r
    res["bucket_mix"] = {g: dense.loc[gi, "bucket"].value_counts(normalize=True).round(3).to_dict()
                         for g, gi in groups.items()}
    # per category and per difficulty, for the dense model and the shipped MoOP l0keep
    d_m = arms["MoOP l0keep"].loc[ids, "ade6"] - dense.ade6
    lab = lab.loc[ids]
    res["by_category"] = {c: {"n": len(g), "dense_ade6": float(dense.loc[g.index, "ade6"].mean()),
                              "moop_l0keep_d_ade6": float(d_m.loc[g.index].mean())}
                          for c, g in lab.groupby("category")}
    res["by_difficulty"] = {int(k): {"n": len(g), "dense_ade6": float(dense.loc[g.index, "ade6"].mean()),
                                     "moop_l0keep_d_ade6": float(d_m.loc[g.index].mean())}
                            for k, g in lab.groupby("difficulty")}
    if lab.gt_cluster.notna().any():
        res["confusion_vs_nvidia"] = pd.crosstab(lab.gt_cluster, lab.category).to_dict(orient="index")
    return res


def fmt_ci(c):
    return f"[{c[0]:+.3f}, {c[1]:+.3f}]"


def table_lines(set_name, res):
    n = res["n"]
    out = [(f"\n## {set_name}: long-tail {n['longtail']} (event {n['lt_event']}, other {n['lt_other']}) "
           f"/ nominal {n['nominal']} clips")]
    out += [f"bucket mix {g:9s} {mix}" for g, mix in res["bucket_mix"].items()]
    out += ["", "### long-tail (all) vs nominal", "",
            ("| arm | minADE@6 LT | minADE@6 NOM | d LT [95% CI] | d NOM [95% CI] | DiD [95% CI] | "
            "DiD bucket-matched [95% CI] | CoC degen LT / NOM |"),
            "|---|---|---|---|---|---|---|---|"]
    for a, r in res["arms"].items():
        lt, nm = r["longtail"], r["nominal"]
        if a == "dense":
            out.append(f"| {a} | {lt['ade6']:.3f} | {nm['ade6']:.3f} | — | — | — | — | "
                       f"{lt['coc_degen']:.1%} / {nm['coc_degen']:.1%} |")
            continue
        dd = r["did"]["longtail"]
        out.append(f"| {a} | {lt['ade6']:.3f} | {nm['ade6']:.3f} | {lt['d_ade6']:+.3f} {fmt_ci(lt['d_ade6_ci'])} | "
                   f"{nm['d_ade6']:+.3f} {fmt_ci(nm['d_ade6_ci'])} | {dd['point']:+.3f} {fmt_ci(dd['ci'])} | "
                   f"{dd['matched']:+.3f} {fmt_ci(dd['matched_ci'])} | "
                   f"{lt['coc_degen']:.1%} / {nm['coc_degen']:.1%} |")
    out += ["", "### split: event clusters vs other/condition (OTHER_LONGTAIL), each against nominal", "",
            ("| arm | minADE@6 event | minADE@6 other | d event [95% CI] | d other [95% CI] | "
            "DiD event, matched [95% CI] | DiD other, matched [95% CI] |"),
            "|---|---|---|---|---|---|---|"]
    for a, r in res["arms"].items():
        ev, ot = r["lt_event"], r["lt_other"]
        if a == "dense":
            out.append(f"| {a} | {ev['ade6']:.3f} | {ot['ade6']:.3f} | — | — | — | — |")
            continue
        de, do = r["did"]["lt_event"], r["did"]["lt_other"]
        out.append(f"| {a} | {ev['ade6']:.3f} | {ot['ade6']:.3f} | {ev['d_ade6']:+.3f} {fmt_ci(ev['d_ade6_ci'])} | "
                   f"{ot['d_ade6']:+.3f} {fmt_ci(ot['d_ade6_ci'])} | {de['matched']:+.3f} {fmt_ci(de['matched_ci'])} | "
                   f"{do['matched']:+.3f} {fmt_ci(do['matched_ci'])} |")
    out.append("\nby Claude category (dense minADE@6, MoOP l0keep d):")
    for c, v in sorted(res["by_category"].items(), key=lambda kv: -kv[1]["n"]):
        out.append(f"  {c:40s} n={v['n']:4d}  dense {v['dense_ade6']:.3f}  d {v['moop_l0keep_d_ade6']:+.3f}")
    out.append("by Claude difficulty:")
    for k, v in res["by_difficulty"].items():
        out.append(f"  {k}  n={v['n']:4d}  dense {v['dense_ade6']:.3f}  d {v['moop_l0keep_d_ade6']:+.3f}")
    return out


def plot(results, path):
    arms = [a for a in ARMS if a != "dense"]
    fig, axes = plt.subplots(1, len(results), figsize=(5.2 * len(results), 0.42 * len(arms) + 1.6),
                             sharey=True, squeeze=False)
    y = np.arange(len(arms))[::-1]
    for ax, (s, res) in zip(axes[0], results.items()):
        for g, off in (("nominal", -0.2), ("lt_event", 0.0), ("lt_other", 0.2)):
            m = np.array([res["arms"][a][g]["d_ade6"] for a in arms])
            lo = np.array([res["arms"][a][g]["d_ade6_ci"][0] for a in arms])
            hi = np.array([res["arms"][a][g]["d_ade6_ci"][1] for a in arms])
            ax.hlines(y + off, lo, hi, color=COLOR[g], lw=2)
            ax.plot(m, y + off, "o", ms=6, color=COLOR[g], mec="white", mew=1,
                    label=f"{GROUP_NAME[g]} (n={res['n'][g]})")
        ax.axvline(0, color="#52514e", lw=1)
        ax.set_title(f"{s}", loc="left", fontsize=11)
        ax.set_xlabel("d minADE@6 vs dense (m), 95% bootstrap CI")
        ax.grid(axis="x", color="#e6e5e0", lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(loc="lower right", frameon=False, fontsize=9)
    axes[0][0].set_yticks(y, arms)
    fig.suptitle("Compression cost on Claude-flagged long-tail vs nominal clips (A100, k=8, seed 42)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="label_v1")
    ap.add_argument("--out", default="openloop_v1")
    ap.add_argument("--n-boot", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = OUT_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    labels = pd.read_parquet(OUT_ROOT / args.labels / "labels.parquet")
    labels = labels[(labels.status == "ok") & (labels.rep == 0)]
    rng = np.random.default_rng(args.seed)
    results = {s: analyze_set(s, labels, args.n_boot, rng) for s in SETS if (labels.set == s).any()}

    (out / "config.json").write_text(json.dumps({
        "labels": args.labels, "n_boot": args.n_boot, "seed": args.seed, "arms": ARMS,
        "source": "outputs/cvpr27/a100/<arm>_{test2000,oodval} (A100, run_baseline.py --no-tf, k=8, seed 42)",
        "metric": "minADE@6 = min over the first 6 of 8 trajectory samples, 6.4 s horizon",
        "longtail": "Claude category != NOMINAL (rep 0)"}, indent=2))
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    lines = []
    for s, res in results.items():
        lines += table_lines(s, res)
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    plot(results, out / "d_ade6_longtail_vs_nominal.png")


if __name__ == "__main__":
    main()
