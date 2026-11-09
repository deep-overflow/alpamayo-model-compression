"""Gates of iclr27-rebuttal/plans/2026-11-09_ada-fill-lp-fm-ce.md (Ada re-runs that fill the CI report's gaps).

  G1  FM-only / CE-only: the re-run's ade_rollout_k, fde_rollout_k and gen_coc equal the stored Ada rows on every
      Test500 and OOD-Val262 clip (762 per arm), and every new row carries the 1.6 / 3.2 s fields
  G2  LLM-Pruner paper checkpoint (data/alpasim/drivers/lp_r50): the 10 OOD-Val262 clips of outputs/lp_r50_oodval
      (Ada, same checkpoint) are bit-identical, and the 6.4 s means reproduce Table 1 and LingoQA 44.8 to the
      printed digit

Usage (repo root): .venv/bin/python iclr27-rebuttal/code/gate_ada_fill.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "cvpr27" / "code"))
from analyze_a100 import load_rows, metric

OUT = REPO / "outputs"
NEW = OUT / "iclr27-rebuttal" / "ada_fill"
SETS = {"test500": "test_500.parquet", "oodval262": "ood_val.parquet"}
FIELDS = ["ade_rollout_k", "fde_rollout_k", "gen_coc"]
HZ_FIELDS = ["ade_rollout_k_h16", "ade_rollout_k_h32", "fde_rollout_k_h16", "fde_rollout_k_h32"]
OLD = {"fm_only": ("slim_traj_u40_v2", {"test500": "traj_u40_v2_test", "oodval262": "traj_u40_v2_ood"}),
       "ce_only": ("slim_coc_u40_v2", {"test500": "coc_u40_v2_test", "oodval262": "coc_u40_v2_ood"})}
LP_TAG, LP_OLD = "lp_r50", "lp_r50_oodval"
PAPER_LP = {"test500": (1.451, 3.889), "oodval262": (1.662, 4.153)}
PAPER_LINGO = 0.448


def ids_of(set_name):
    return list(pd.read_parquet(OUT / "eval_sets" / SETS[set_name])["clip_id"])


def gpus(d):
    return sorted({json.loads(p.read_text()).get("gpu") for p in d.glob("config.json")})


def compare(new, old, ids, fields):
    """Clips of ids in both: how many agree on every field, and the first few that do not."""
    both = [c for c in ids if c in new and c in old]
    diff = [c for c in both if any(new[c][f] != old[c][f] for f in fields)]
    return {"n_both": len(both), "n_same": len(both) - len(diff), "first_diff": diff[:5]}


def main():
    res, lines = {"G1": {}, "G2": {}}, []

    for arm, (tag, old_dirs) in OLD.items():
        for set_name, od in old_dirs.items():
            ids = ids_of(set_name)
            d = NEW / f"{arm}_{set_name}"
            new, old = load_rows(d, tag), load_rows(OUT / od, tag)
            e = {"n_ids": len(ids), "n_new": sum(c in new for c in ids), "gpu": gpus(d),
                 **compare(new, old, ids, FIELDS),
                 "horizons_present": all(all(f in new[c] for f in HZ_FIELDS) for c in ids if c in new)}
            if e["first_diff"]:
                dd = [metric(new[c], "ade") - metric(old[c], "ade") for c in e["first_diff"]]
                e["first_diff_delta_minADE6"] = dd
            e["pass"] = e["n_new"] == e["n_same"] == len(ids) and e["horizons_present"]
            res["G1"][f"{arm}/{set_name}"] = e
            lines.append(f"G1 {arm:8s} {set_name:9s} {'PASS' if e['pass'] else 'FAIL'}: new {e['n_new']}/{len(ids)}, "
                         f"identical {e['n_same']}/{e['n_both']} on {FIELDS}, 1.6/3.2 s fields "
                         f"{'present' if e['horizons_present'] else 'MISSING'}, gpu {e['gpu']}"
                         + (f", first differing {e['first_diff']}" if e["first_diff"] else ""))

    g2 = res["G2"]
    old10 = load_rows(OUT / LP_OLD, LP_TAG)
    for set_name, (pa, pf) in PAPER_LP.items():
        ids = ids_of(set_name)
        d = NEW / f"{LP_TAG}_{set_name}"
        new = load_rows(d, LP_TAG)
        have = [c for c in ids if c in new]
        e = {"n_ids": len(ids), "n_new": len(have), "gpu": gpus(d)}
        if have:
            e["minADE6"] = float(np.mean([metric(new[c], "ade") for c in have]))  # mean over (N,) clips
            e["minFDE6"] = float(np.mean([metric(new[c], "fde") for c in have]))
            e["paper"] = [pa, pf]
            e["matches_paper"] = bool(round(e["minADE6"], 3) == pa and round(e["minFDE6"], 3) == pf)
        if set_name == "oodval262":
            e["old_ada_10"] = compare(new, old10, ids, FIELDS + HZ_FIELDS)
        e["pass"] = len(have) == len(ids) and e.get("matches_paper", False) and (
            set_name != "oodval262" or e["old_ada_10"]["n_same"] == e["old_ada_10"]["n_both"] == 10)
        g2[set_name] = e
        lines.append(f"G2 lp_r50   {set_name:9s} {'PASS' if e['pass'] else 'FAIL'}: new {len(have)}/{len(ids)}"
                     + (f", minADE@6 {e['minADE6']:.4f} / minFDE@6 {e['minFDE6']:.4f} vs paper {pa:.3f} / {pf:.3f}"
                        if have else "")
                     + (f", stored Ada clips identical {e['old_ada_10']['n_same']}/{e['old_ada_10']['n_both']}"
                        if set_name == "oodval262" else "") + f", gpu {e['gpu']}")
    m = NEW / f"lingo_vqa_scores_{LP_TAG}" / "metrics.json"
    if m.exists():
        acc = json.loads(m.read_text())["runs"][0]["accuracy"]
        cfg = json.loads((NEW / f"lingo_vqa_{LP_TAG}" / "config.json").read_text())
        g2["lingoqa"] = {"accuracy": acc, "paper": PAPER_LINGO, "gpu": cfg.get("gpu"),
                         "pass": abs(acc - PAPER_LINGO) < 5e-4}
        lines.append(f"G2 lp_r50   LingoQA   {'PASS' if g2['lingoqa']['pass'] else 'FAIL'}: {acc:.3f} vs paper "
                     f"{PAPER_LINGO:.3f}, gpu {cfg.get('gpu')}")
    else:
        g2["lingoqa"] = {"pass": False, "missing": str(m)}
        lines.append(f"G2 lp_r50   LingoQA   FAIL: {m} missing")

    res["G1_pass"] = all(e["pass"] for e in res["G1"].values())
    res["G2_pass"] = all(e["pass"] for e in g2.values())
    lines = [f"Ada fill gates -- G1 {'PASS' if res['G1_pass'] else 'FAIL'}, G2 {'PASS' if res['G2_pass'] else 'FAIL'}",
             "", *lines]
    dst = NEW / "gate"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "config.json").write_text(json.dumps({
        "new_root": str(NEW.relative_to(REPO)), "old_runs": OLD, "lp_old": LP_OLD, "fields": FIELDS,
        "horizon_fields": HZ_FIELDS, "paper_lp": PAPER_LP, "paper_lingo": PAPER_LINGO}, indent=2))
    (dst / "metrics.json").write_text(json.dumps(res, indent=2))
    (dst / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
