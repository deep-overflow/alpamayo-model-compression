"""95% CIs for the ICLR27 arms: open loop (Test500, OOD-Val262), LingoQA, and the Table 4 CoC NLL column.

Rows are the Ada runs that reproduce paper Table 1 / Table 4 (checked against PAPER below). Nothing is re-run.

  open loop  minADE@6 / minFDE@6 at 1.6 / 3.2 / 6.4 s: mean with a clip bootstrap CI, median, CoC degeneration;
             paired arm - dense and arm - MoOP per clip (mean, CI, median, two-sided bootstrap p)
  LingoQA    Lingo-Judge accuracy over 500 questions; the 5 questions of a segment share a clip, so CIs
             resample the 100 segments; paired deltas vs dense, vs MoOP and vs the always-"yes" floor
  NLL        Table 4: the dense model's own Test500 CoC teacher-forced through each arm (nats/token, clip mean)

All bootstraps: 200,000 draws, seed 0, shared draws across arms (common random numbers); every paired CI is
recomputed with seed 1 and cells whose "excludes 0" call flips are listed.

LLM-Pruner has no complete Ada open-loop or LingoQA rows (its Ada outputs were deleted with soowon's tree), so
it appears only in the NLL column and in a separate A100 block (dense / MoOP / LLM-Pruner, outputs/cvpr27/a100;
Test500 = the test_500 clips inside test_2000). The A100 and Ada numbers are never put in one table.

Usage (repo root): .venv/bin/python iclr27-rebuttal/code/ci_openloop_lingoqa.py
"""

import argparse
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
from analyze_a100 import HORIZONS, N_BOOT, K, load_rows, metric

OUT = REPO / "outputs"
A100 = OUT / "cvpr27" / "a100"
SETS = {"test500": ("test_500.parquet", 500), "oodval262": ("ood_val.parquet", 262)}
KINDS = ("ade", "fde")
CHUNK = 10_000

# key, label, run tag, Ada test dir, Ada OOD dir, LingoQA dir, NLL dir (None = no Ada rows)
ARMS = [
    ("dense", "Dense", "baseline", "baseline_pred_test", "baseline_pred_oodval", "lingo_vqa_baseline",
     "nll_dense_baseline_test"),
    ("wanda", "Wanda", "slim_wanda_u40_v2", "wanda_u40_v2_test", "wanda_u40_v2_oodval",
     "lingo_vqa_slim_wanda_u40_v2", "nll_dense_slim_wanda_u40_v2_test"),
    ("lp_r50", "LLM-Pruner", "lp_r50", None, None, None, "nll_dense_lp_r50_test"),
    ("tyr", "Tyr", "slim_tyr_u40_r", "tyr_u40_r_pred_test", "tyr_u40_r_pred_oodval", "lingo_vqa_slim_tyr_u40_r",
     "nll_dense_slim_tyr_u40_r_test"),
    ("fm_only", "FM-only", "slim_traj_u40_v2", "traj_u40_v2_test", "traj_u40_v2_ood", "lingo_vqa_slim_traj_u40_v2",
     "nll_dense_slim_traj_u40_v2_test"),
    ("ce_only", "CE-only", "slim_coc_u40_v2", "coc_u40_v2_test", "coc_u40_v2_ood", "lingo_vqa_slim_coc_u40_v2",
     "nll_dense_slim_coc_u40_v2_test"),
    ("moop", "MoOP", "slim_dual_u40_v2", "dual_u40_v2_pred_test", "dual_u40_v2_pred_oodval",
     "lingo_vqa_slim_dual_u40_v2", "nll_dense_slim_dual_u40_v2_test"),
    ("moop_e", "MoOP+E", "slim_dualexp_u40_em93p75", "dualexp_em93p75_test", "dualexp_em93p75_oodval", None, None),
]
LABEL = {a[0]: a[1] for a in ARMS}
# A100 block: key -> (dir prefix, tag)
A100_ARMS = {"dense": ("dense", "baseline"), "moop": ("moop", "slim_dual_u40_v2"), "lp_r50": ("lp_r50", "slim_lp_r50")}
A100_SET_DIR = {"test500": "test2000", "oodval262": "oodval"}
FLOOR = "lingo_vqa_const_yes"   # always answer "yes": 37.6%, the higher of the two constant floors

# paper Table 1 (6.4 s minADE@6, minFDE@6; Test500, OOD-Val262) and Table 4 (LingoQA %, NLL)
PAPER_T1 = {"dense": (0.842, 2.267, 1.000, 2.496), "wanda": (2.975, 8.269, 2.790, 6.884),
            "lp_r50": (1.451, 3.889, 1.662, 4.153), "tyr": (0.950, 2.570, 1.177, 2.990),
            "fm_only": (1.008, 2.720, 1.178, 2.960), "ce_only": (1.584, 4.500, 1.546, 4.010),
            "moop": (0.950, 2.546, 1.119, 2.851), "moop_e": (0.951, 2.539, 1.118, 2.844)}
PAPER_T4 = {"dense": (73.2, 0.175), "wanda": (9.2, 2.183), "lp_r50": (44.8, 0.462), "tyr": (34.2, 0.516),
            "fm_only": (37.0, 0.586), "ce_only": (30.2, 0.313), "moop": (68.8, 0.252)}


def boot_many(series, n_boot=N_BOOT, seed=0, groups=None):
    """Bootstrap the mean of aligned series with one set of draws -> {name: (lo, hi, p, draws)}.

    series: {name: (N,)}. groups=None resamples the N items (same draws as analyze_a100.boot_ci); otherwise
    groups (N,) labels clusters and whole clusters are resampled (ratio of resampled sums to resampled counts).
    p is the two-sided bootstrap p of mean = 0 (meaningful only for paired differences).
    """
    names = list(series)
    D = np.stack([np.asarray(series[k], dtype=float) for k in names], 1)  # (N, M)
    if groups is not None:
        keys, inv = np.unique(np.asarray(groups), return_inverse=True)
        S = np.zeros((len(keys), D.shape[1]))  # (G, M) per-cluster sums
        np.add.at(S, inv, D)
        cnt = np.bincount(inv, minlength=len(keys)).astype(float)  # (G,)
    else:
        S, cnt = D, np.ones(len(D))
    n = len(S)
    rng = np.random.RandomState(seed)
    means = []
    for s in range(0, n_boot, CHUNK):
        c = min(CHUNK, n_boot - s)
        idx = rng.randint(0, n, size=(c, n))  # (c, n)
        w = np.bincount((idx + n * np.arange(c)[:, None]).ravel(), minlength=c * n).reshape(c, n)  # (c, n)
        means.append((w @ S) / (w @ cnt)[:, None])  # (c, M)
    m = np.concatenate(means)  # (n_boot, M)
    lo, hi = np.percentile(m, [2.5, 97.5], axis=0)
    p = np.minimum(1.0, 2 * np.minimum((m <= 0).mean(0), (m >= 0).mean(0)))
    return {k: (float(lo[i]), float(hi[i]), float(p[i])) for i, k in enumerate(names)}


def summarise(series, groups=None):
    """Mean, median, and seed-0 / seed-1 bootstrap CIs for each series."""
    b0 = boot_many(series, groups=groups)
    b1 = boot_many(series, seed=1, groups=groups)
    out = {}
    for k, v in series.items():
        v = np.asarray(v, dtype=float)
        lo, hi, p = b0[k]
        out[k] = {"mean": float(v.mean()), "median": float(np.median(v)), "ci": [lo, hi], "p": p,
                  "ci_seed1": list(b1[k][:2]), "p_seed1": b1[k][2]}
    return out


def ids_of(set_name):
    return list(pd.read_parquet(OUT / "eval_sets" / SETS[set_name][0])["clip_id"])


def open_loop_block(runs, ref_keys):
    """runs: {arm: {set: rows}} -> absolute and paired summaries per set. ref_keys: arms to pair against."""
    res = {}
    for set_name, (_, n_exp) in SETS.items():
        ids = ids_of(set_name)
        arms = [a for a in runs if set_name in runs[a]]
        for a in arms:
            missing = [c for c in ids if c not in runs[a][set_name]]
            assert not missing, f"{a}/{set_name}: {len(missing)} of {n_exp} clips missing"
        hz = {a: [h for h in HORIZONS if all(f"ade_rollout_k{HORIZONS[h]}" in runs[a][set_name][c] for c in ids)]
              for a in arms}
        absolute, paired = {}, {}
        for a in arms:
            r = runs[a][set_name]
            for kind in KINDS:
                for h in hz[a]:
                    absolute[f"{a}|{kind}|{h}"] = [metric(r[c], kind, h) for c in ids]  # (N,)
                    for ref in ref_keys:
                        if ref != a and ref in runs and h in hz.get(ref, []):
                            rb = runs[ref][set_name]
                            paired[f"{a}|{ref}|{kind}|{h}"] = [metric(r[c], kind, h) - metric(rb[c], kind, h)
                                                               for c in ids]  # (N,)
        coc = {a: float(np.mean([runs[a][set_name][c]["coc_degenerate"] for c in ids])) for a in arms}
        res[set_name] = {"n": len(ids), "horizons": hz, "coc_degenerate": coc,
                         "absolute": summarise(absolute), "paired": summarise(paired)}
    return res


def lingo_rows(d):
    return {r["question_id"]: r for r in json.loads((d / "scored.json").read_text())}


def lingo_block(dirs, ref_keys):
    """dirs: {arm: dir} -> accuracy with a segment CI and paired deltas vs refs and vs the always-yes floor."""
    rows = {a: lingo_rows(d) for a, d in dirs.items()}
    floor = lingo_rows(OUT / FLOOR)
    qs = sorted(set.intersection(*(set(r) for r in rows.values())) & set(floor))
    assert len(qs) == 500, f"LingoQA: {len(qs)} common questions"
    seg = [floor[q]["segment_id"] for q in qs]
    for r in rows.values():
        assert all(r[q]["segment_id"] == floor[q]["segment_id"] for q in qs)
    acc = {a: [float(r[q]["correct"]) for q in qs] for a, r in rows.items()}  # (500,)
    yes = [float(floor[q]["correct"]) for q in qs]
    series = {f"{a}|abs": v for a, v in acc.items()}
    for a, x in acc.items():
        for ref in ref_keys:
            if ref != a and ref in acc:
                series[f"{a}|{ref}"] = list(np.subtract(x, acc[ref]))
        series[f"{a}|floor_yes"] = list(np.subtract(x, yes))
    return {"n_questions": len(qs), "n_segments": len(set(seg)), "floor_yes_accuracy": float(np.mean(yes)),
            "stats": summarise(series, groups=seg)}


def nll_block(dirs):
    rows = {a: load_rows(d, d.name.removeprefix("nll_dense_").removesuffix("_test")) for a, d in dirs.items()}
    ids = ids_of("test500")
    for a, r in rows.items():
        assert all(c in r for c in ids), f"NLL {a}: clips missing"
    v = {a: [r[c]["nll_dense"] for c in ids] for a, r in rows.items()}  # (500,)
    series = {f"{a}|abs": x for a, x in v.items()}
    for a, x in v.items():
        for ref in ("dense", "moop"):
            if ref != a:
                series[f"{a}|{ref}"] = list(np.subtract(x, v[ref]))
    return {"n": len(ids), "stats": summarise(series)}


def flips(stats):
    """Paired cells whose 'CI excludes 0' call differs between seed 0 and seed 1."""
    bad = []
    for k, e in stats.items():
        ex0 = e["ci"][0] > 0 or e["ci"][1] < 0
        ex1 = e["ci_seed1"][0] > 0 or e["ci_seed1"][1] < 0
        if ex0 != ex1:
            bad.append(k)
    return bad


def fmt_ci(e, nd=3, scale=1.0, sign=False):
    f = f"{{:{'+' if sign else ''}.{nd}f}}"
    m, (lo, hi) = e["mean"] * scale, (e["ci"][0] * scale, e["ci"][1] * scale)
    return f"{f.format(m)} [{f.format(lo)}, {f.format(hi)}]"


def pfmt(p):
    return f"< {1 / N_BOOT:.0e}" if p == 0 else f"{p:.2g}"


def star(e):
    return "*" if e["ci"][0] > 0 or e["ci"][1] < 0 else ""


def md_open_loop(res, title, arms):
    md = [f"### {title}", ""]
    for set_name, blk in res.items():
        md += [f"**{set_name}** (n = {blk['n']} clips; mean [95% clip-bootstrap CI]; median in the last column)", "",
               ("| arm | minADE@6 1.6 s | 3.2 s | 6.4 s | minFDE@6 1.6 s | 3.2 s | 6.4 s | median ADE / FDE 6.4 s "
                "| CoC degenerate |"), "|---|---|---|---|---|---|---|---|---|"]
        A = blk["absolute"]
        for a in arms:
            if a not in blk["coc_degenerate"]:
                continue
            cells = [fmt_ci(A[f"{a}|{k}|{h}"]) if f"{a}|{k}|{h}" in A else "n/a" for k in KINDS for h in HORIZONS]
            med = f"{A[f'{a}|ade|6.4s']['median']:.3f} / {A[f'{a}|fde|6.4s']['median']:.3f}"
            md.append(f"| {LABEL[a]} | " + " | ".join(cells) + f" | {med} | {blk['coc_degenerate'][a]:.1%} |")
        md.append("")
        P = blk["paired"]
        for ref in ("dense", "moop"):
            md += [(f"Paired arm - {LABEL[ref]} on {set_name} (mean [95% CI], * = CI excludes 0; median; "
                    "bootstrap p at 6.4 s):"), "",
                   ("| arm | dADE 1.6 s | 3.2 s | 6.4 s | dFDE 1.6 s | 3.2 s | 6.4 s | median dADE / dFDE 6.4 s | "
                    "p ADE / FDE 6.4 s |"), "|---|---|---|---|---|---|---|---|---|"]
            for a in arms:
                key = f"{a}|{ref}|ade|6.4s"
                if key not in P:
                    continue
                cells = [(fmt_ci(P[f"{a}|{ref}|{k}|{h}"], sign=True) + star(P[f"{a}|{ref}|{k}|{h}"]))
                         if f"{a}|{ref}|{k}|{h}" in P else "n/a" for k in KINDS for h in HORIZONS]
                ea, ef = P[f"{a}|{ref}|ade|6.4s"], P[f"{a}|{ref}|fde|6.4s"]
                md.append(f"| {LABEL[a]} | " + " | ".join(cells)
                          + f" | {ea['median']:+.3f} / {ef['median']:+.3f} | {pfmt(ea['p'])} / {pfmt(ef['p'])} |")
            md.append("")
    return md


def md_lingo(blk, title, arms):
    S = blk["stats"]
    md = [f"### {title}", "",
          (f"{blk['n_questions']} questions in {blk['n_segments']} segments; CIs resample segments. "
           f"Always-'yes' floor {blk['floor_yes_accuracy']:.1%}."), "",
          "| arm | accuracy (%) | d vs Dense (pp) | d vs MoOP (pp) | d vs always-yes floor (pp) |",
          "|---|---|---|---|---|"]
    for a in arms:
        if f"{a}|abs" not in S:
            continue
        cells = []
        for ref in ("dense", "moop", "floor_yes"):
            k = f"{a}|{ref}"
            cells.append(fmt_ci(S[k], nd=1, scale=100, sign=True) + star(S[k]) if k in S else "--")
        md.append(f"| {LABEL[a]} | {fmt_ci(S[f'{a}|abs'], nd=1, scale=100)} | " + " | ".join(cells) + " |")
    return [*md, ""]


def md_nll(blk, arms):
    S = blk["stats"]
    md = ["### Table 4 NLL column (dense Test500 CoC through each arm, nats/token)", "",
          f"n = {blk['n']} clips; mean [95% clip-bootstrap CI]; median.", "",
          "| arm | NLL | median | d vs Dense | d vs MoOP |", "|---|---|---|---|---|"]
    for a in arms:
        if f"{a}|abs" not in S:
            continue
        cells = [fmt_ci(S[f"{a}|{r}"], sign=True) + star(S[f"{a}|{r}"]) if f"{a}|{r}" in S else "--"
                 for r in ("dense", "moop")]
        md.append(f"| {LABEL[a]} | {fmt_ci(S[f'{a}|abs'])} | {S[f'{a}|abs']['median']:.3f} | " + " | ".join(cells)
                  + " |")
    return [*md, ""]


def paper_check(ol, lq, nll):
    lines = []
    for a, (ta, tf, oa, of) in PAPER_T1.items():
        for set_name, want in (("test500", (ta, tf)), ("oodval262", (oa, of))):
            A = ol.get(set_name, {}).get("absolute", {})
            if f"{a}|ade|6.4s" not in A:
                continue
            got = (A[f"{a}|ade|6.4s"]["mean"], A[f"{a}|fde|6.4s"]["mean"])
            ok = all(abs(g - w) <= 0.0005 + 1e-9 for g, w in zip(got, want))
            lines.append(f"T1 {a:8s} {set_name:9s} ours {got[0]:.4f} / {got[1]:.4f}  paper {want[0]:.3f} / "
                         f"{want[1]:.3f}  {'ok' if ok else 'DIFF'}")
    for a, (acc, nl) in PAPER_T4.items():
        if f"{a}|abs" in lq["stats"]:
            g = lq["stats"][f"{a}|abs"]["mean"] * 100
            lines.append(f"T4 {a:8s} LingoQA ours {g:.1f}  paper {acc:.1f}  {'ok' if abs(g - acc) < 0.05 else 'DIFF'}")
        if f"{a}|abs" in nll["stats"]:
            g = nll["stats"][f"{a}|abs"]["mean"]
            lines.append(f"T4 {a:8s} NLL     ours {g:.4f} paper {nl:.3f}  {'ok' if abs(g - nl) <= 0.0005 + 1e-9 else 'DIFF'}")
    return lines


def plot(ol, dst):
    arms = [a for a, *_ in ARMS if a != "dense"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    for ax, set_name in zip(axes, SETS):
        P = ol[set_name]["paired"]
        have = [a for a in arms if f"{a}|dense|ade|6.4s" in P]
        y = np.arange(len(have))[::-1]
        for yi, a in zip(y, have):
            e = P[f"{a}|dense|ade|6.4s"]
            ax.plot(e["ci"], [yi, yi], color="#3b6fb6", lw=2, solid_capstyle="round")
            ax.plot([e["mean"]], [yi], "o", color="#3b6fb6", ms=6, mec="white", mew=1.5)
            ax.text(e["ci"][1], yi + 0.18, f"{e['mean']:+.3f}", fontsize=8, color="#333333", ha="right")
        ax.axvline(0, color="#999999", lw=1)
        ax.set_xscale("symlog", linthresh=0.5)
        ticks = [0, 0.1, 0.2, 0.5, 1, 2]
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.set_yticks(y, [LABEL[a] for a in have])
        ax.set_title(f"{set_name} (Ada)", fontsize=10)
        ax.set_xlabel("arm - Dense, minADE@6 6.4 s (m), 95% CI")
        ax.grid(axis="x", color="#e6e6e6", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(dst / "paired_minade6_vs_dense.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="ci_openloop_lingoqa")
    args = ap.parse_args()
    dst = OUT / "iclr27-rebuttal" / args.exp_id
    dst.mkdir(parents=True, exist_ok=True)
    order = [a for a, *_ in ARMS]

    ada_runs, gpus = {}, {}
    for a, _, tag, td, od, *_ in ARMS:
        for set_name, d in (("test500", td), ("oodval262", od)):
            if d is None:
                continue
            gpus[d] = json.loads((OUT / d / "config.json").read_text()).get("gpu")
            ada_runs.setdefault(a, {})[set_name] = load_rows(OUT / d, tag)
    assert all("Ada" in str(g) for g in gpus.values()), gpus
    ol = open_loop_block(ada_runs, ("dense", "moop"))

    lq_dirs = {a: OUT / d for a, *_, d, _ in ARMS if d is not None}
    for d in lq_dirs.values():
        assert "Ada" in json.loads((d / "config.json").read_text())["gpu"], d
    lq = lingo_block(lq_dirs, ("dense", "moop"))
    nll = nll_block({a: OUT / d for a, *_, d in ARMS if d is not None})

    a100_runs = {a: {s: load_rows(A100 / f"{p}_{A100_SET_DIR[s]}", tag) for s in SETS}
                 for a, (p, tag) in A100_ARMS.items()}
    ol_a100 = open_loop_block(a100_runs, ("dense", "moop"))
    lq_a100 = lingo_block({a: A100 / f"lingo_vqa_{p}" for a, (p, _) in A100_ARMS.items()}, ("dense", "moop"))

    check = paper_check(ol, lq, nll)
    flipped = {f"ada/{s}": flips(ol[s]["paired"]) for s in SETS}
    flipped.update({f"a100/{s}": flips(ol_a100[s]["paired"]) for s in SETS})
    flipped.update({"lingoqa": flips(lq["stats"]), "lingoqa_a100": flips(lq_a100["stats"]),
                    "nll": flips(nll["stats"])})

    md = ["# ICLR27 arms -- 95% CIs: open loop, LingoQA, CoC NLL", "",
          (f"Ada (RTX 5880) rows unless marked A100. minADE@6 / minFDE@6 = min over the first {K} of 8 samples, "
           f"rollout only. Bootstrap {N_BOOT:,} draws, seed 0, clip-level for open loop and NLL, "
           "segment-level for LingoQA. Paired = per-clip (per-question) arm minus reference."), ""]
    md += md_open_loop(ol, "Open loop (Ada)", order)
    md += md_lingo(lq, "LingoQA (Ada, Lingo-Judge)", order)
    md += md_nll(nll, order)
    md += ["## A100 block (LLM-Pruner has no complete Ada rows; do not merge with the Ada tables)", ""]
    md += md_open_loop(ol_a100, "Open loop (A100)", list(A100_ARMS))
    md += md_lingo(lq_a100, "LingoQA (A100)", list(A100_ARMS))
    md += ["## Paper check", "", "```", *check, "```", "",
           "## Seed check (paired cells whose 'excludes 0' call flips between seed 0 and seed 1)", ""]
    md += [f"- {k}: {v if v else 'none'}" for k, v in flipped.items()]

    (dst / "config.json").write_text(json.dumps({
        "arms": {a: {"label": lab, "tag": t, "test500": td, "oodval262": od, "lingoqa": ld, "nll": nd}
                 for a, lab, t, td, od, ld, nd in ARMS},
        "a100_arms": {a: {"dir_prefix": p, "tag": t} for a, (p, t) in A100_ARMS.items()},
        "a100_dir": str(A100.relative_to(REPO)), "gpus": gpus, "sets": {k: v[0] for k, v in SETS.items()},
        "k": K, "horizons": HORIZONS, "n_boot": N_BOOT, "seeds": [0, 1], "ci": "percentile 95%",
        "lingoqa_cluster": "segment_id", "lingoqa_floor": FLOOR}, indent=2))
    (dst / "metrics.json").write_text(json.dumps({
        "open_loop_ada": ol, "lingoqa_ada": lq, "nll_ada": nll, "open_loop_a100": ol_a100,
        "lingoqa_a100": lq_a100, "paper_check": check, "seed_flips": flipped}, indent=2))
    (dst / "summary.txt").write_text("\n".join(md) + "\n")
    plot(ol, dst)
    print("\n".join(md))
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
