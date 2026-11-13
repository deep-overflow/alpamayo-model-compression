"""u50 arms on Blackwell (iclr27-rebuttal/plans/2026-11-12_u50-openloop-blackwell.md): open loop (Test500, OOD-Val262, all
horizons), LingoQA and dense-CoC NLL with 95% CIs, absolute and paired vs MoOP u50 / dense.

Reuses ci_openloop_lingoqa.open_loop_block / lingo_block / nll_block (clip-level bootstrap, LingoQA segment-level,
200,000 draws, seed 0 with a seed-1 check), so the numbers follow the paper's CI report. Arms still missing rows are
skipped per block, so the script can be rerun as the queue finishes.

  .venv/bin/python iclr27-rebuttal/code/u50_openloop_ci.py
writes outputs/iclr27-rebuttal/u50_openloop_bw/analysis/{metrics.json, table.md}
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "iclr27-rebuttal" / "code"))
sys.path.insert(0, str(REPO / "cvpr27" / "code"))
import ci_openloop_lingoqa as ci
from analyze_a100 import load_rows

D = REPO / "outputs" / "iclr27-rebuttal" / "u50_openloop_bw"
ARMS = [("dense", "Baseline (dense)", "baseline"), ("moop", "MoOP u50", "slim_dual_u50_v2"),
        ("tyr", "Týr-the-Pruner u50", "slim_tyr_u50_r"), ("lp", "LLM-Pruner u50", "slim_lp_r60"),
        ("fm", "FM-only u50", "slim_traj_u50_v2"), ("ce", "CE-only u50", "slim_coc_u50_v2"),
        ("wanda", "Wanda u50", "slim_wanda_u50_v2")]
LABEL = {k: lab for k, lab, _ in ARMS}
REFS = ("moop", "dense", "dense_bw")
HZ = ("1.6s", "3.2s", "6.4s")


def complete(d, tag, n):
    return d.exists() and len(load_rows(d, tag)) >= n


def ci_txt(e, sign=False, scale=1.0, nd=3):
    m, (lo, hi) = e["mean"] * scale, e["ci"]
    lo, hi = lo * scale, hi * scale
    star = "*" if sign and (lo > 0 or hi < 0) else ""
    f = f"{{:{'+' if sign else ''}.{nd}f}}"
    return f"{f.format(m)} [{f.format(lo)}, {f.format(hi)}]{star}"


def nll_summary(dirs, refs):
    """ci.nll_block without its fixed dense/moop references: absolute NLL and paired vs whichever refs are present."""
    tags = {k: tag for k, _, tag in ARMS}
    ids = ci.ids_of("test500")
    v = {}
    for a, d in dirs.items():
        r = load_rows(d, tags[a])
        assert all(c in r for c in ids), f"NLL {a}: clips missing"
        v[a] = [r[c]["nll_dense"] for c in ids]  # (500,)
    series = {f"{a}|abs": x for a, x in v.items()}
    for a, x in v.items():
        for ref in refs:
            if ref != a and ref in v:
                series[f"{a}|{ref}"] = [p - q for p, q in zip(x, v[ref])]
    return {"n": len(ids), "stats": ci.summarise(series)}


def main():
    runs = {}
    for k, _, tag in ARMS:
        for s, (_, n) in ci.SETS.items():
            d = D / f"{k}_{s}"
            if complete(d, tag, n):
                runs.setdefault(k, {})[s] = load_rows(d, tag)
    lingo_dirs = {k: D / f"lingo_vqa_{k}" for k, _, _ in ARMS if (D / f"lingo_vqa_{k}" / "scored.json").exists()}
    nll_dirs = {k: D / f"nll_dense_{tag}_test" for k, _, tag in ARMS
                if (D / f"nll_dense_{tag}_test" / "summary_s0of1.txt").exists()}
    if "dense" not in runs:
        # No Blackwell dense (user, 11-12). The user asked (11-13) for pairs with the ICLR paper's dense model:
        # its Ada runs behind paper Tables 1 / 9 -- same clips and per-clip seeds, different GPU type.
        P = REPO / "outputs"
        runs["dense"] = {"test500": load_rows(P / "baseline_pred_test", "baseline"),
                         "oodval262": load_rows(P / "baseline_pred_oodval", "baseline")}
        lingo_dirs["dense"] = P / "lingo_vqa_baseline"
        nll_dirs["dense"] = P / "nll_dense_baseline_test"
        LABEL["dense"] = "Dense (ICLR paper, Ada)"
        ARMS[0] = ("dense", LABEL["dense"], "baseline")
        # Same checkpoint and protocol rerun on this Blackwell type (driving only): same-GPU check of the dense pairs
        runs["dense_bw"] = {"test500": load_rows(P / "baseline_bw_ps_test", "baseline"),
                            "oodval262": load_rows(P / "baseline_bw_ps_oodval", "baseline")}
        LABEL["dense_bw"] = "Dense (ICLR checkpoint, Blackwell rerun)"
        ARMS.insert(1, ("dense_bw", LABEL["dense_bw"], "baseline"))
    refs = tuple(r for r in REFS if r in runs)
    lrefs = tuple(r for r in refs if r in lingo_dirs)
    nrefs = tuple(r for r in refs if r in nll_dirs)
    ol = ci.open_loop_block(runs, refs)
    lingo = ci.lingo_block(lingo_dirs, lrefs) if len(lingo_dirs) >= 1 else None
    nll = nll_summary(nll_dirs, nrefs) if nll_dirs else None

    L = ["# u50 arms -- open loop, LingoQA, CoC NLL (RTX PRO 5000 Blackwell, GPUs 0-3)", "",
         ("Clip-level bootstrap (LingoQA: segment-level), 200,000 draws, seed 0. * = CI excludes 0. "
          "u50 arms ran on Blackwell. 'Dense (ICLR paper, Ada)' is the paper's run (same clips and seeds), so its "
          "pairs cross GPU types; 'Dense (ICLR checkpoint, Blackwell rerun)' is the same checkpoint on this GPU type "
          "(driving only)."), ""]
    for s in ci.SETS:
        if s not in ol:
            continue
        r = ol[s]
        L += [f"## {s} ({r['n']} clips) -- absolute", "",
              "| Method | minADE@6 1.6 s | 3.2 s | 6.4 s | minFDE@6 1.6 s | 3.2 s | 6.4 s | CoC degeneration |",
              "|---|---|---|---|---|---|---|---|"]
        for k, lab, _ in ARMS:
            if k not in runs or s not in runs[k]:
                continue
            cells = [ci_txt(r["absolute"][f"{k}|{kind}|{h}"]) for kind in ci.KINDS for h in HZ]
            L.append(f"| {lab} | " + " | ".join(cells) + f" | {100 * r['coc_degenerate'][k]:.1f}% |")
        for ref in refs:
            L += ["", f"## {s} -- paired, method minus {LABEL[ref]}", "",
                  "| Method | minADE@6 1.6 s | 3.2 s | 6.4 s | minFDE@6 1.6 s | 3.2 s | 6.4 s |", "|---|---|---|---|---|---|---|"]
            for k, lab, _ in ARMS:
                key = f"{k}|{ref}|ade|6.4s"
                if k == ref or key not in r["paired"]:
                    continue
                L.append(f"| {lab} | " + " | ".join(ci_txt(r["paired"][f"{k}|{ref}|{kind}|{h}"], sign=True)
                                                    for kind in ci.KINDS for h in HZ) + " |")
        L.append("")
    if lingo:
        st = lingo["stats"]
        L += [(f"## LingoQA ({lingo['n_questions']} questions, {lingo['n_segments']} segments; always-'yes' "
               f"{100 * lingo['floor_yes_accuracy']:.1f}%)"), "",
              "| Method | accuracy % | " + " | ".join(f"- {LABEL[r]} (pp)" for r in lrefs) + " | - always 'yes' (pp) |",
              "|---|---|" + "---|" * len(lrefs) + "---|"]
        for k, lab, _ in ARMS:
            if f"{k}|abs" not in st:
                continue
            row = [ci_txt(st[f"{k}|abs"], scale=100, nd=1)]
            for ref in lrefs:
                row.append(ci_txt(st[f"{k}|{ref}"], sign=True, scale=100, nd=1) if f"{k}|{ref}" in st else "--")
            row.append(ci_txt(st[f"{k}|floor_yes"], sign=True, scale=100, nd=1))
            L.append(f"| {lab} | " + " | ".join(row) + " |")
        L.append("")
    if nll:
        st = nll["stats"]
        L += [f"## CoC NLL, Test500 dense CoC ({nll['n']} clips; nats/token)", "",
              "| Method | NLL | " + " | ".join(f"- {LABEL[r]}" for r in nrefs) + " |", "|---|---|" + "---|" * len(nrefs)]
        for k, lab, _ in ARMS:
            if f"{k}|abs" not in st:
                continue
            row = [ci_txt(st[f"{k}|abs"])] + [ci_txt(st[f"{k}|{ref}"], sign=True) if f"{k}|{ref}" in st else "--"
                                             for ref in nrefs]
            L.append(f"| {lab} | " + " | ".join(row) + " |")
    out = D / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({"open_loop": ol, "lingoqa": lingo, "nll": nll}, indent=1))
    (out / "table.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
