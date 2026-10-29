"""A100 (KISTI NEURON) open-loop track: the gate, then the A100 table.

Plan and gates: plans/2026-10-29_kisti-a100-openloop.md.

Every arm of an A100 table is measured on A100. Same-card reruns are bitwise identical, but
A100 vs Ada moves dense by <=0.003 m and MoOP by up to 0.018 m minADE@6, and ~5% of clips
change CoC text (2026-10-29), so these rows are never paired with, or merged into, Ada tables.

  G0  every A100 card the jobs saw is an A100-SXM4-80GB, every config.json names model
      revision 7aba829, and alpamayo1_5 is commit f42e594
  G1  the rerun of dense test2000 shard 0 is bitwise identical to the first pass -- all eight
      samples at 1.6/3.2/6.4 s and the CoC text -- on all 500 clips
  G2  dense test2000 on A100 vs the stored Ada dense: |mean delta minADE@6| <= 0.005 m and
      the same CoC text on >= 95% of clips. A version/cache check, not a pairing.

Usage (after experiments/transfer/a100_pull.sh):
  python experiments/evaluation/analyze_a100.py --stage gate
  python experiments/evaluation/analyze_a100.py --stage table
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]

BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
COLORS = {"dense": "#6B6555", "moop": "#2a78d6", "moop_l0keep": "#008300"}
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})

# arm -> rows-file tag (run_baseline names rows after the model dir)
ARMS = {"dense": "baseline", "moop": "slim_dual_u40_v2",
        "moop_l0keep": "slim_dual_u40_l0keep_v2"}
SETS = {"test2000": ("test_2000.parquet", 2000), "oodval": ("ood_val.parquet", 262)}
# the stored Ada dense test2000: test500 + the 1,500 clips added to it
ADA_DENSE = [("baseline_ada_ps_test", "baseline"), ("minitron/test2000x_dense", "baseline")]
GPU = "NVIDIA A100-SXM4-80GB"
MODEL_REV = "7aba8293c09993f2e125c6819df05d7fa3e873ea"
ALPAMAYO_COMMIT = "f42e594aaf8b50dcd2cbb359d62e3ffc7b12fcf8"
G1_N = 500
G2_TOL, G2_SAME = 0.005, 0.95
HORIZONS = {"1.6s": "_h16", "3.2s": "_h32", "6.4s": ""}
SAME_FIELDS = [f"{m}_rollout_k{h}" for h in HORIZONS.values() for m in ("ade", "fde")] + ["gen_coc"]
K = 6            # minADE@6: the first 6 of the 8 stored samples (seeds base+k)
N_BOOT = 200_000


def load_rows(exp_dir, tag):
    """Merge every shard file for one model; last write per clip wins."""
    rows = {}
    for p in sorted(exp_dir.glob(f"{tag}_s*of*.json")):
        for r in json.loads(p.read_text()):
            rows[r["clip_id"]] = r
    return rows


def manifest_ids(out, set_name):
    return list(pd.read_parquet(out / "eval_sets" / SETS[set_name][0])["clip_id"])


def metric(r, kind, h="6.4s"):
    return float(min(r[f"{kind}_rollout_k{HORIZONS[h]}"][:K]))


def boot_ci(d, n_boot=N_BOOT, seed=0, chunk=10_000):
    """Bootstrap CI of the mean, chunked so 200k resamples fit in memory."""
    d = np.asarray(d, dtype=float)
    rng = np.random.RandomState(seed)
    means = []
    for s in range(0, n_boot, chunk):
        idx = rng.randint(0, len(d), size=(min(chunk, n_boot - s), len(d)))
        means.append(d[idx].mean(1))
    lo, hi = np.percentile(np.concatenate(means), [2.5, 97.5])
    return float(lo), float(hi)


def cluster_boot_ci(d, groups, n_boot=20_000, seed=0):
    """Bootstrap CI of a mean difference resampling whole groups (LingoQA segments)."""
    d, groups = np.asarray(d, dtype=float), np.asarray(groups)
    keys = np.unique(groups)
    sums = np.array([d[groups == g].sum() for g in keys])
    counts = np.array([(groups == g).sum() for g in keys])
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(keys), size=(n_boot, len(keys)))
    means = sums[idx].sum(1) / counts[idx].sum(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def parse_env(path):
    """GPU names and the alpamayo1_5 commit from a job's env_<jobid>.txt."""
    text = path.read_text()
    gpus = re.findall(r"^GPU \d+: (.+?) \(UUID", text, flags=re.MULTILINE)
    commit = re.search(r'"commit_id":\s*"([0-9a-f]+)"', text)
    return {"file": path.name, "gpus": gpus, "alpamayo1_5_commit": commit.group(1) if commit else None}


def open_loop_dirs(a100):
    names = [f"{a}_{s}" for a in ARMS for s in SETS] + ["dense_test2000_rerun"]
    return [a100 / n for n in names if (a100 / n / "config.json").exists()]


def gate(out, a100):
    ids = manifest_ids(out, "test2000")
    first = load_rows(a100 / "dense_test2000", "baseline")
    rerun = load_rows(a100 / "dense_test2000_rerun", "baseline")
    ada = {}
    for d, tag in ADA_DENSE:
        ada.update(load_rows(out / d, tag))

    # G0: the cards, the model revision, the package
    envs = [parse_env(p) for p in sorted((a100 / "queue").glob("env_*.txt"))]
    cfgs = {d.name: json.loads((d / "config.json").read_text()) for d in open_loop_dirs(a100)}
    gpus_seen = sorted({g for e in envs for g in e["gpus"]}
                       | {str(c.get("gpu")) for c in cfgs.values()})
    g0 = {
        "env_files": envs,
        "configs": {k: {"gpu": c.get("gpu"), "model_revision": c.get("model_revision")}
                    for k, c in cfgs.items()},
        "gpus_seen": gpus_seen,
        "gpu_ok": bool(envs) and gpus_seen == [GPU],
        "revision_ok": bool(cfgs) and all(c.get("model_revision") == MODEL_REV for c in cfgs.values()),
        "alpamayo_ok": bool(envs) and all(e["alpamayo1_5_commit"] == ALPAMAYO_COMMIT for e in envs),
    }
    g0["pass"] = g0["gpu_ok"] and g0["revision_ok"] and g0["alpamayo_ok"]

    # G1: rerun vs first pass, field by field at stored precision
    common = sorted(set(rerun) & set(first))
    same = [all(rerun[c][f] == first[c][f] for f in SAME_FIELDS) for c in common]
    diff_clips = [c for c, s in zip(common, same) if not s]
    g1 = {"n_rerun": len(rerun), "n_common": len(common), "n_identical": int(sum(same)),
          "first_differing": diff_clips[:5]}
    g1["pass"] = len(common) == G1_N and g1["n_identical"] == G1_N

    # G2: A100 dense vs Ada dense, on clips both have
    both = [c for c in ids if c in first and c in ada]
    a = np.array([metric(first[c], "ade") for c in both])
    b = np.array([metric(ada[c], "ade") for c in both])
    fa = np.array([metric(first[c], "fde") for c in both])
    fb = np.array([metric(ada[c], "fde") for c in both])
    d = a - b
    lo, hi = boot_ci(d) if len(d) else (float("nan"), float("nan"))
    coc_same = float(np.mean([first[c]["gen_coc"] == ada[c]["gen_coc"] for c in both])) if both else 0.0
    g2 = {"n_a100": len(first), "n_ada": len(ada), "n_both": len(both),
          "minADE6_a100": float(a.mean()) if len(a) else None,
          "minADE6_ada": float(b.mean()) if len(b) else None,
          "minFDE6_a100": float(fa.mean()) if len(fa) else None,
          "minFDE6_ada": float(fb.mean()) if len(fb) else None,
          "delta_minADE6": float(d.mean()) if len(d) else None, "delta_ci": [lo, hi],
          "coc_same": coc_same}
    g2["pass"] = (len(both) == SETS["test2000"][1] and abs(g2["delta_minADE6"]) <= G2_TOL
                  and coc_same >= G2_SAME)

    verdict = "PASS" if g0["pass"] and g1["pass"] and g2["pass"] else "FAIL"
    lines = [
        f"A100 track gate -- {verdict}",
        "",
        (f"G0 {'PASS' if g0['pass'] else 'FAIL'}: cards seen {g0['gpus_seen']} | "
        f"revision {'ok' if g0['revision_ok'] else 'MISMATCH'} | "
        f"alpamayo1_5 {'ok' if g0['alpamayo_ok'] else 'MISMATCH'} ({len(envs)} env file(s))"),
        (f"G1 {'PASS' if g1['pass'] else 'FAIL'}: rerun {g1['n_identical']}/{g1['n_common']} clips "
        f"bitwise identical (need {G1_N}/{G1_N})"),
    ]
    if g2["n_both"]:
        lines.append(
            f"G2 {'PASS' if g2['pass'] else 'FAIL'}: n={g2['n_both']} minADE@6 A100 {g2['minADE6_a100']:.3f} "
            f"vs Ada {g2['minADE6_ada']:.3f} (delta {g2['delta_minADE6']:+.4f} [{lo:+.4f}, {hi:+.4f}], "
            f"need |delta| <= {G2_TOL}) | minFDE@6 {g2['minFDE6_a100']:.3f} vs {g2['minFDE6_ada']:.3f} | "
            f"same CoC {coc_same:.1%} (need >= {G2_SAME:.0%})")
    else:
        lines.append("G2 FAIL: no clip in both the A100 and the Ada rows")
    if diff_clips:
        lines.append(f"   first differing rerun clips: {diff_clips[:5]}")

    dst = a100 / "gate"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "config.json").write_text(json.dumps({
        "a100_dir": str(a100), "ada_dense": ADA_DENSE, "gpu": GPU, "model_revision": MODEL_REV,
        "alpamayo1_5_commit": ALPAMAYO_COMMIT, "g1_n": G1_N, "g2_tol": G2_TOL,
        "g2_coc_same": G2_SAME, "k": K, "n_boot": N_BOOT}, indent=2))
    (dst / "metrics.json").write_text(json.dumps(
        {"verdict": verdict, "G0": g0, "G1": g1, "G2": g2}, indent=2))
    (dst / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"-> {dst}")


def lingo(a100, arm):
    m = a100 / f"lingo_vqa_scores_{arm}" / "metrics.json"
    s = a100 / f"lingo_vqa_{arm}" / "scored.json"
    if not (m.exists() and s.exists()):
        return None
    acc = json.loads(m.read_text())["runs"][0]["accuracy"]
    rows = {r["question_id"]: r for r in json.loads(s.read_text())}
    return {"accuracy": float(acc), "rows": rows}


def table(out, a100):
    res = {"open_loop": {}, "paired_vs_dense": {}, "lingoqa": {}}
    rows = {}
    for set_name, (_, n_exp) in SETS.items():
        ids = manifest_ids(out, set_name)
        for arm, tag in ARMS.items():
            r = load_rows(a100 / f"{arm}_{set_name}", tag)
            have = [c for c in ids if c in r]
            if not have:
                continue
            rows[(arm, set_name)] = r
            e = {"n": len(have), "complete": len(have) == n_exp,
                 "coc_degenerate": float(np.mean([r[c]["coc_degenerate"] for c in have]))}
            for kind in ("ade", "fde"):
                for h in HORIZONS:
                    e[f"min{kind.upper()}6_{h}"] = float(np.mean([metric(r[c], kind, h) for c in have]))
                e[f"min{kind.upper()}6_6.4s_median"] = float(np.median([metric(r[c], kind) for c in have]))
            res["open_loop"][f"{arm}/{set_name}"] = e

        base = rows.get(("dense", set_name))
        for arm in ARMS:
            if arm == "dense" or base is None or (arm, set_name) not in rows:
                continue
            r = rows[(arm, set_name)]
            both = [c for c in ids if c in base and c in r]
            p = {"n": len(both)}
            for kind in ("ade", "fde"):
                d = np.array([metric(r[c], kind) - metric(base[c], kind) for c in both])
                p[f"delta_min{kind.upper()}6"] = float(d.mean())
                p[f"delta_min{kind.upper()}6_ci"] = boot_ci(d)
                p[f"delta_min{kind.upper()}6_median"] = float(np.median(d))
            res["paired_vs_dense"][f"{arm}/{set_name}"] = p

    lq = {arm: lingo(a100, arm) for arm in ARMS}
    for arm, v in lq.items():
        if v is None:
            continue
        e = {"accuracy": v["accuracy"]}
        if arm != "dense" and lq["dense"] is not None:
            qs = sorted(set(v["rows"]) & set(lq["dense"]["rows"]))
            d = [float(v["rows"][q]["correct"]) - float(lq["dense"]["rows"][q]["correct"]) for q in qs]
            seg = [v["rows"][q]["segment_id"] for q in qs]
            e.update({"n_paired": len(qs), "delta": float(np.mean(d)), "delta_ci": cluster_boot_ci(d, seg)})
        res["lingoqa"][arm] = e

    md = ["# A100 (KISTI NEURON) open-loop table", "",
          ("All rows on NVIDIA A100-SXM4-80GB; do not merge with Ada tables. "
          "k=8, per-clip seed 42, max_gen 256, rollout only; @6 = min over the first 6 samples."), "",
          "| set | arm | n | minADE@6 1.6 s | 3.2 s | 6.4 s | minFDE@6 1.6 s | 3.2 s | 6.4 s | CoC degenerate |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for key, e in res["open_loop"].items():
        arm, set_name = key.split("/")
        n = f"{e['n']}" + ("" if e["complete"] else " (incomplete)")
        md.append(f"| {set_name} | {arm} | {n} | " + " | ".join(
            f"{e[f'minADE6_{h}']:.3f}" for h in HORIZONS) + " | " + " | ".join(
            f"{e[f'minFDE6_{h}']:.3f}" for h in HORIZONS) + f" | {e['coc_degenerate']:.1%} |")
    if res["paired_vs_dense"]:
        md += ["", "Paired difference vs dense (6.4 s, mean [95% bootstrap CI], median):", "",
               "| set | arm | n | delta minADE@6 | median | delta minFDE@6 | median |",
               "|---|---|---|---|---|---|---|"]
        for key, p in res["paired_vs_dense"].items():
            arm, set_name = key.split("/")
            a_lo, a_hi = p["delta_minADE6_ci"]
            f_lo, f_hi = p["delta_minFDE6_ci"]
            md.append(f"| {set_name} | {arm} | {p['n']} | {p['delta_minADE6']:+.3f} [{a_lo:+.3f}, {a_hi:+.3f}] | "
                      f"{p['delta_minADE6_median']:+.3f} | {p['delta_minFDE6']:+.3f} [{f_lo:+.3f}, {f_hi:+.3f}] | "
                      f"{p['delta_minFDE6_median']:+.3f} |")
    if res["lingoqa"]:
        md += ["", "LingoQA (500 questions, Lingo-Judge; delta vs dense with a segment-clustered CI):", "",
               "| arm | accuracy | delta vs dense |", "|---|---|---|"]
        for arm, e in res["lingoqa"].items():
            dlt = (f"{e['delta'] * 100:+.1f} pp [{e['delta_ci'][0] * 100:+.1f}, {e['delta_ci'][1] * 100:+.1f}]"
                   if "delta" in e else "--")
            md.append(f"| {arm} | {e['accuracy']:.1%} | {dlt} |")

    dst = a100 / "table"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "config.json").write_text(json.dumps({
        "a100_dir": str(a100), "arms": ARMS, "sets": {k: v[0] for k, v in SETS.items()},
        "k": K, "n_boot": N_BOOT, "horizons": HORIZONS}, indent=2))
    (dst / "metrics.json").write_text(json.dumps(res, indent=2))
    (dst / "summary.txt").write_text("\n".join(md) + "\n")
    (dst / "table.md").write_text("\n".join(md) + "\n")

    fig, axes = plt.subplots(1, len(SETS), figsize=(4.2 * len(SETS), 3.4))
    for ax, set_name in zip(np.atleast_1d(axes), SETS):
        arms = [a for a in ARMS if f"{a}/{set_name}" in res["open_loop"]]
        vals = [res["open_loop"][f"{a}/{set_name}"]["minADE6_6.4s"] for a in arms]
        ax.bar(arms, vals, color=[COLORS[a] for a in arms], width=0.6)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(f"{set_name} (A100)")
        ax.set_ylabel("minADE@6, 6.4 s (m)")
    fig.tight_layout()
    fig.savefig(dst / "minade6.png", dpi=150)
    print("\n".join(md))
    print(f"-> {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["gate", "table"])
    ap.add_argument("--outputs", type=Path, default=REPO / "outputs")
    ap.add_argument("--a100", default="a100", help="track dir under --outputs")
    args = ap.parse_args()
    a100 = args.outputs / args.a100
    if args.stage == "gate":
        gate(args.outputs, a100)
    else:
        table(args.outputs, a100)


if __name__ == "__main__":
    main()
