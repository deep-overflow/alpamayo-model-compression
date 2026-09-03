"""Join the three result families on one method key and group them by the question asked.

The collected CSVs are three separate lists that cannot be joined: open loop keys on the
arm name (`dual`), alpasim on the checkpoint (`slim_dual_u40_v2`), LingoQA on the run id
(`lingo_vqa_slim_dual_u40_v2`). This normalises all three to one `method` and emits a
wide table -- one row per method, all three families side by side -- so a method can be
read across trajectory, driving and reasoning at once.

It also supplies the structure a flat dump has no room for:

  track       which experiment the row belongs to, mirroring the groupings in
              `paper_numbers.ARMS` and the named-config table in CLAUDE.md
  role        anchor (unpruned reference) | arm | control | superseded
  vs          the run the paired delta is measured against, so a row states its
              comparison instead of a bare number
  budget      the compression setting, so arms are only read against equal-budget peers

Rows with no track land in `other` rather than being forced into one. `superseded` marks
an open-loop dir whose per-sample twin exists (it can only report minADE@8).

Alpasim rows are deduplicated: the same checkpoint recurs in up to 11 run directories
with identical scores because every matrix re-includes the baseline and the reference
arms. One row per (checkpoint, n_scenes), largest suite kept.

Usage:
  .venv/bin/python experiments/evaluation/make_master.py
  .venv/bin/python experiments/evaluation/make_master.py --index results_index
"""

import argparse
import csv
import datetime
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# method -> track. Mirrors the comment blocks in paper_numbers.ARMS and the named-config
# table in CLAUDE.md; anything absent is reported as `other`, never guessed into a track.
TRACKS = {
    "criterion (u40, one-factor)": ["baseline", "traj", "coc", "j", "dual", "jtraj"],
    "prior work": ["wanda", "wandatxt", "lp_coc", "lp_dual"],
    "budget sweep": ["dual_u55", "dual_u70", "coc_u55", "traj_u55", "jtraj_u55"],
    "recovery (LoRA)": ["slim_recover_dual_u55", "slim_recover_dual_u55_v2",
                        "slim_recover_dual_u55_d5", "slim_recover_coc_u55",
                        "slim_recover_traj_u55", "slim_recover_dual_u70"],
    "Tyr reconstruction": ["tyr_sel_uniform", "tyr_sel_search", "tyr_uniform_d1r",
                           "tyr_d1r", "tyr_uniform_r", "tyr_r"],
    "dual x Tyr factorial": ["dualg", "dualr", "dualgr", "dualscope", "dualg_tyralloc"],
    "dualr Hessian 2x2": ["dualr_rep", "dualr_d", "dualr_e", "dualr_w"],
    "LingoQA Hessian": ["dualr_wl"],
    # expert MLP-only pruning at M%, stacked on a VLM half. dualrwl_* keep dualr_wl's
    # half, dualexp_em93p75 the plain dual half, so the two VLM halves are readable
    # against each other at the one shared rung. reports/evaluation/
    # 2026-09-01_expert-mlp-ladder.html.
    "expert MLP ladder": ["dualrwl_em50", "dualrwl_em75", "dualrwl_em87p5",
                          "dualrwl_em93p75", "dualrwl_em96p875",
                          "dualrwl_em98p4375", "dualrwl_em100",
                          "dualexp_em93p75"],
    "cache-targeted recon": ["dualrc_s16", "dualrc_s24"],
    "cache criterion": ["cachedual_u40_v2", "cacheonly_u40_v2"],
    "expert axis": ["expert_q25", "expert_m25", "expert_m_pm", "expert_q50",
                    "expert_m50", "expert_both25", "dualexp_cond_ps",
                    "dualexp_e10_cond_ps", "dualexp_e15_cond_ps",
                    "dualexp_u40_e25_ps"],
    "VLM axis": ["vlm_q", "vlm_m", "vlm_m_pm", "dual_bw", "baseline_bw",
                 "dualm_c1109_ada", "dualm_u40_v2_ada"],
    "quantization": ["uniform_w8", "uniform_w4", "qvla_coc_b8", "qvla_coc_b4",
                     "w8_all", "w4_all", "prune_w8", "prune_w4"],
    "dual combination rule": ["dualsum", "dualprod", "dual2nd_u40_v2"],
    "calibration source": ["dual_u40_ctl", "dual_u40_mix", "dual_u40_ood"],
    "traj aggregation": ["trajctl_val", "trajsumabs_val", "trajznorm_val"],
    "iterative": ["it3", "iter_coc"],
    "LingoQA-informed criterion": ["trajvqa_u40_v2", "vqa_u40_v2", "coclingo_u40_v2"],
    # plans/2026-08-31_znorm11-criterion.md. All three rebuild the u40 budget from
    # importance_v2_ada, so dual_ada -- the shipped dual recipe on that same importance
    # file -- is the peer to read them against, not the shipped dual_u40_v2.
    "znorm11 criterion": ["dual_ada", "dualfix", "znorm11"],
}
METHOD2TRACK = {m: t for t, ms in TRACKS.items() for m in ms}
# controls exist to say what a factor does alone; they are not candidate configs
CONTROLS = {"dual_u40_ctl", "dual_u40_mix", "dual_u40_ood", "trajctl_val",
            "trajsumabs_val", "trajznorm_val", "dualsum", "dualprod",
            "expert_m_pm", "vlm_m_pm", "dualm_c1109_ada", "dual_ada"}
SETS = [("indist", "val"), ("test", "test"), ("oodval", "ood")]
MASTER_COLS = (
    ["track", "method", "ckpt", "role", "budget", "prune_pct", "arch", "vs"]
    + [f"{s}_{m}" for _, s in SETS for m in ("ade6", "fde6", "degen", "d", "sig")]
    + ["cl_scenes", "cl_score", "cl_dscore", "cl_p", "cl_collision", "cl_offroad",
       "cl_progress", "lingo_pct", "has", "openloop_dir"]
)


def norm(name):
    """LingoQA run id -> the checkpoint it evaluated. `slim_` is NOT stripped: it is part
    of the checkpoint name that open loop and alpasim both use, so removing it would
    break exactly the join this table exists for."""
    return re.sub(r"^lingo_(vqa|judge)_", "", name)


def label(tag):
    """Display name for a checkpoint that open loop never gave a short arm name."""
    return re.sub(r"^slim_", "", tag)


def budget_of(prune_pct, method):
    if method.startswith(("uniform_w", "qvla_", "w8_all", "w4_all")):
        return "quant only"
    if not prune_pct:
        return ""
    # only the three named uniform ratios get a name; everything else stays a number,
    # so a 32.2% expert-plus-VLM config is never mislabelled as the 33.1% u55 budget
    p = float(prune_pct)
    for exact, name in ((23.99, "24% (u40)"), (33.12, "33% (u55)"),
                        (41.84, "42% (u70)")):
        if abs(p - exact) < 0.05:
            return name
    return f"{p:.1f}%"


def load(index_dir, name):
    with (index_dir / f"{name}.csv").open() as fh:
        return list(csv.DictReader(fh))


def fold_openloop(rows):
    """(checkpoint, arch) -> set -> row.

    The checkpoint tag is the only key the three families share -- open loop's short arm
    name (`dual`) never appears in alpasim or LingoQA, but `slim_dual_u40_v2` appears in
    all three. Architecture stays in the key because Ada and Blackwell are not bitwise
    comparable. Within one key the per-sample run wins, so `dual_u40_v2` (minADE@8 only)
    collapses into `dual` instead of occupying a second row."""
    out, dropped = {}, []
    for r in rows:
        key = (r["tag"], r["arch"])
        cur = out.setdefault(key, {}).get(r["set"])
        if cur is None:
            out[key][r["set"]] = r
        elif bool(r["minADE6_mean"]) > bool(cur["minADE6_mean"]):
            out[key][r["set"]] = r
            dropped.append(cur["dir"])
        else:
            dropped.append(r["dir"])
    return out, dropped


def fold_closedloop(rows):
    """checkpoint -> the widest suite it was run on; identical repeats collapse."""
    best = {}
    for r in rows:
        if not r["score"]:
            continue                       # forensics-only dirs carry no aggregate
        m = norm(r["config"])
        n = int(r["n_scenes"])
        if m not in best or n > int(best[m]["n_scenes"]):
            best[m] = r
    return best


def fold_lingoqa(rows):
    """checkpoint -> the standard unprompted VQA score (the number the reports quote)."""
    out = {}
    for r in rows:
        if r["protocol"] != "vqa" or r["style"] not in ("", "unprompted"):
            continue
        m = norm(r["exp_id"])
        if m.endswith(("_concise", "_g256")) or m.startswith("const_"):
            continue
        out.setdefault(m, r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="results_index")
    ap.add_argument("--name", default="master.csv")
    args = ap.parse_args()

    index_dir = REPO / "outputs" / args.index
    ol, dropped = fold_openloop(load(index_dir, "openloop"))
    cl = fold_closedloop(load(index_dir, "closedloop"))
    lq = fold_lingoqa(load(index_dir, "lingoqa"))

    # one entry per (checkpoint, arch); checkpoints seen only in alpasim or LingoQA join
    # on the Ada side, which is where both of those are always measured
    keys = set(ol) | {(t, "Ada") for t in set(cl) | set(lq)}
    rows = []
    for tag, arch in sorted(keys):
        sets = ol.get((tag, arch), {})
        any_ol = next(iter(sets.values()), None)
        m = any_ol["arm"] if any_ol else label(tag)
        c = cl.get(tag) if arch in ("Ada", "") else None
        lg = lq.get(tag) if arch in ("Ada", "") else None
        row = {"method": m, "ckpt": tag,
               "track": METHOD2TRACK.get(m, "other"),
               "arch": arch,
               "prune_pct": any_ol["prune_pct"] if any_ol else "",
               "vs": any_ol["baseline_ref"] if any_ol else "",
               "openloop_dir": any_ol["dir"] if any_ol else ""}
        row["budget"] = budget_of(row["prune_pct"], m)
        if tag == "baseline":
            row["role"] = "anchor"
        elif m in CONTROLS:
            row["role"] = "control"
        elif not sets:
            row["role"] = "no open loop"
        else:
            row["role"] = "arm"
        for key, short in SETS:
            r = sets.get(key)
            if not r:
                continue
            row[f"{short}_ade6"] = r["minADE6_mean"]
            row[f"{short}_fde6"] = r["minFDE6_mean"]
            row[f"{short}_degen"] = r["coc_degen"]
            row[f"{short}_d"] = r["d_minADE6_median"]
            row[f"{short}_sig"] = r["d_sig"]
        if c:
            row.update({"cl_scenes": c["n_scenes"], "cl_score": c["score"],
                        "cl_dscore": c["d_score_mean"], "cl_p": c["wilcoxon_p"],
                        "cl_collision": c["collision_at_fault"],
                        "cl_offroad": c["offroad"],
                        "cl_progress": c["progress_clipped_rel"]})
        if lg:
            row["lingo_pct"] = lg["judge_pct"]
        row["has"] = "".join(x for x, ok in (("O", bool(sets)), ("C", bool(c)),
                                             ("L", bool(lg))) if ok)
        rows.append(row)

    # track first, then budget, then method: equal-budget peers end up adjacent
    order = list(TRACKS) + ["other"]
    rows.sort(key=lambda r: (order.index(r["track"]), r["budget"], r["method"]))

    out = index_dir / args.name
    kst = datetime.timezone(datetime.timedelta(hours=9))
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([("AD VLA compression - one row per method, three evaluation"
                     " families joined. generated "
                     + datetime.datetime.now(tz=kst).strftime("%Y-%m-%d %H:%M KST"))])
        w.writerow([("has: O=open loop, C=closed loop, L=LingoQA. role: anchor="
                     "unpruned reference, arm=candidate, control=isolates one factor,"
                     " superseded=older run of the same checkpoint (minADE@8 only)")])
        w.writerow([("val/test/ood = minADE@6 & minFDE@6 means, degen = CoC degeneracy"
                     " rate, d = paired median delta vs the run in `vs`, sig=* when the"
                     " bootstrap CI excludes zero")])
        w.writerow([("cl_* = alpasim over cl_scenes scenes (widest suite per checkpoint;"
                     " repeats across matrix dirs collapsed). lingo_pct = Lingo-Judge %,"
                     " unprompted VQA")])
        w.writerow([])
        w.writerow(MASTER_COLS)
        for r in rows:
            w.writerow([r.get(c, "") for c in MASTER_COLS])
    n_full = sum(1 for r in rows if r["has"] == "OCL")
    print(f"{out}  {out.stat().st_size / 1024:.1f} KB  {len(rows)} methods  "
          f"({n_full} with all three families, {len(dropped)} superseded open-loop "
          f"dirs folded away)", flush=True)
    for t in order:
        k = [r for r in rows if r["track"] == t]
        if k:
            print(f"  {t:28s} {len(k):3d}  " + ", ".join(r["method"] for r in k[:6])
                  + (" ..." if len(k) > 6 else ""), flush=True)


if __name__ == "__main__":
    main()
