"""Harvest every measured result under `outputs/` into three flat tables.

The numbers live in three different shapes -- per-clip shard rows for open loop, a
per-config dict for alpasim, a runs list for LingoQA -- and each is currently read by a
different `analyze_*`. This walks all three and emits one CSV per family, one row per
(arm x set) or (run x config), so the whole study can be pasted or pushed into a
spreadsheet without re-running an analysis. Nothing is copied from a report: every cell
is recomputed from the stored rows, so the tables cannot drift from the data.

Open-loop numbers follow the frozen protocol (`paper_numbers.py`): rollout condition
only, metric is minADE@6 / minFDE@6 reduced from the stored per-sample arrays, and OOD
runs are cut to the val split (262 clips) by the stored `split` field, so a full-OOD run
and an ood_val-manifest run land in the same row. Runs predating per-sample storage keep
their minADE@8 columns and leave the @6 columns blank rather than being dropped.

Paired deltas are against the unpruned baseline **of the same GPU architecture** -- two
runs of the same clip differ by ~0.005 m across Ada and Blackwell, so a cross-arch delta
would be reading kernel noise. The baseline actually used is named in `baseline_ref`.

Discovery is by directory shape, not a hand-kept registry, so a new run appears in the
table the moment it finishes. `paper_numbers.ARMS` is still consulted for the paper's
short arm names; anything it does not name keeps its directory-derived name.

Usage:
  .venv/bin/python experiments/evaluation/collect_results.py
  .venv/bin/python experiments/evaluation/collect_results.py --out results_index --k 6
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))

from paper_numbers import ARMS

BOOT = 5000
SET_SUFFIX = ("indist", "test", "oodval", "ood")
ALPASIM_RUNS = Path("/home/cvlab21/project/chan/alpasim-runs")
CL_SCENES, CL_ROLLOUTS = 150, 300      # the only closed-loop shape that gets reported

# horizons stored per sample: 16 steps = 1.6 s, 32 = 3.2 s, full = 6.4 s. The min is
# taken independently at each horizon, so `minADE6_h16` is the best-of-6 over the 1.6 s
# window, NOT the 1.6 s prefix of whichever sample won at 6.4 s.
HORIZONS = [("", ""), ("_h16", "_h16"), ("_h32", "_h32")]
OPENLOOP_COLS = (
    ["arm", "set", "tag", "dir", "n_clips", "arch", "gpu", "k_stored",
     "params_full", "params_slim", "prune_pct"]
    + [f"min{m}6{sfx}_{stat}" for _, sfx in HORIZONS for m in ("ADE", "FDE")
       for stat in ("mean", "median")]
    + ["minADE8_mean", "minADE8_median", "minFDE8_mean", "minFDE8_median",
       "nll_self_mean", "coc_degen", "coc_empty", "coc_soup",
       "baseline_ref", "d_minADE6_median", "d_ci_lo", "d_ci_hi", "d_sig", "d_n"]
)
# the 18 per-rollout metrics to report, in the order the spec asks for them. `score` and
# `passed` live on the rollout itself; the rest are under rollout["metrics"].
CL_ROLLOUT_KEYS = ["score", "passed"]
CL_METRIC_KEYS = [
    "collision_at_fault", "offroad", "progress_clipped_rel", "progress_rel",
    "wrong_lane", "dist_to_gt_trajectory", "collision_front", "collision_lateral",
    "collision_rear", "plan_deviation", "min_distance_to_obstacle_m",
    "min_distance_to_lane_boundary_m", "dist_traveled_m", "open_loop_collision",
    "safety_monitor_triggered", "duration_frac_20s",
]
CLOSED_COLS = (["run", "config", "n_scenes", "n_rollouts"]
               + CL_ROLLOUT_KEYS + CL_METRIC_KEYS
               + ["score_ci_lo", "score_ci_hi",
                  "repeat_abs_diff_mean", "repeat_abs_diff_max", "n_failed_rollouts",
                  "coc_degen", "coc_empty", "coc_soup", "coc_len",
                  "d_score_mean", "d_ci_lo", "d_ci_hi", "wilcoxon_p",
                  "wins", "losses", "ties"])
LINGO_COLS = ["exp_id", "arm", "protocol", "style", "n_questions", "n_matched",
              "accuracy", "judge_pct", "mean_logit", "answer_words_median",
              "answer_words_mean", "truncated_frac", "source_dir"]

# paper short name -> run dir, inverted from the registry so discovery can adopt it
DIR2ARM = {d: arm for arm, sets in ARMS.items() for d, _ in sets.values()}


def arch_of(gpu):
    """Ada and Blackwell are not bitwise comparable; everything pairs within one."""
    g = gpu or ""
    for a in ("Blackwell", "Ada", "H100", "A100"):
        if a in g:
            return a
    return g or "unknown"


def split_dirname(name):
    """`dualr_wl_u40_indist` -> (`dualr_wl_u40`, `indist`); unknown suffix -> (name, None)."""
    for s in SET_SUFFIX:
        if name.endswith("_" + s):
            return name[: -len(s) - 1], s
    return name, None


def at_k(row, key, k):
    """minADE@k from the stored per-sample array; seeds are base+k so a prefix is a run."""
    return float(np.min(np.asarray(row[key], dtype=float)[:k]))


def merge_shards(exp_dir, tag):
    """Every shard of one checkpoint, deduped by clip id (last write wins)."""
    rows = {}
    for p in sorted(exp_dir.glob(f"{tag}_s*of*.json")):
        try:
            batch = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue                      # a run still writing its shard
        if not isinstance(batch, list):
            continue
        for r in batch:
            if isinstance(r, dict) and "clip_id" in r:
                rows[r["clip_id"]] = r
    return rows


def slim_params(tag):
    """params.full/slim/removed from the checkpoint that produced these rows, if any."""
    meta = REPO / "outputs" / tag / "slim_meta.json"
    if not meta.exists():
        return None
    try:
        return json.loads(meta.read_text()).get("params")
    except json.JSONDecodeError:
        return None


def boot_ci(d, seed=0):
    """Bootstrap CI of the median paired delta -- median, because minADE is heavy-tailed."""
    rng = np.random.default_rng(seed)
    meds = [np.median(d[rng.integers(0, len(d), len(d))]) for _ in range(BOOT)]
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return float(np.median(d)), float(lo), float(hi)


def describe(rows, k):
    """Every open-loop cell for one merged run."""
    vals = list(rows.values())
    out = {"n_clips": len(vals)}
    has_ps = bool(vals) and "ade_rollout_k" in vals[0]
    out["k_stored"] = len(vals[0]["ade_rollout_k"]) if has_ps else ""
    if has_ps:
        for _, sfx in HORIZONS:
            src = f"ade_rollout_k{sfx}", f"fde_rollout_k{sfx}"
            if src[0] not in vals[0]:
                continue                  # older runs stored the full horizon only
            for key, m in zip(src, ("ADE", "FDE")):
                v = np.array([at_k(r, key, k) for r in vals])
                out[f"min{m}6{sfx}_mean"] = float(v.mean())
                out[f"min{m}6{sfx}_median"] = float(np.median(v))
    for src, dst in (("minADE_rollout", "minADE8"), ("minFDE_rollout", "minFDE8")):
        v = np.array([r[src] for r in vals if src in r], dtype=float)
        if len(v):
            out[dst + "_mean"] = float(v.mean())
            out[dst + "_median"] = float(np.median(v))
    for src, dst in (("nll_self", "nll_self_mean"), ("coc_degenerate", "coc_degen"),
                     ("coc_empty", "coc_empty"), ("coc_soup", "coc_soup")):
        v = [r[src] for r in vals if src in r]
        if v:
            out[dst] = float(np.mean(v))
    return out


def collect_openloop(k, verbose):
    """One entry per (run dir, checkpoint tag); OOD dirs are cut to the val split."""
    entries = []
    for exp_dir in sorted((REPO / "outputs").iterdir()):
        if not exp_dir.is_dir():
            continue
        tags = sorted({re.sub(r"_s\d+of\d+\.json$", "", p.name)
                       for p in exp_dir.glob("*_s*of*.json")})
        if not tags:
            continue
        cfg = {}
        cfg_path = exp_dir / "config.json"
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text())
            except json.JSONDecodeError:
                cfg = {}
        stem, suffix = split_dirname(exp_dir.name)
        which = cfg.get("set") or suffix
        for tag in tags:
            rows = merge_shards(exp_dir, tag)
            if not rows or "minADE_rollout" not in next(iter(rows.values())):
                continue                  # not an open-loop run (bench dirs share the shape)
            if which == "ood":            # frozen protocol: OOD means OOD-val
                n_full = len(rows)
                rows = {i: r for i, r in rows.items() if r.get("split") == "val"}
                if not rows:
                    continue
                which_label, dropped = "oodval", n_full - len(rows)
            else:
                which_label, dropped = which, 0
            e = {"dir": exp_dir.name, "tag": tag, "set": which_label,
                 "arm": DIR2ARM.get(exp_dir.name, stem),
                 "gpu": cfg.get("gpu", ""), "arch": arch_of(cfg.get("gpu")),
                 "dropped_ood_train": dropped, "rows": rows}
            e.update(describe(rows, k))
            p = slim_params(tag)
            if p:
                e["params_full"] = p.get("full", "")
                e["params_slim"] = p.get("slim", "")
                if p.get("full"):
                    e["prune_pct"] = round(p.get("removed", 0) / p["full"] * 100, 2)
            entries.append(e)
            if verbose:
                print(f"  [openloop] {exp_dir.name}/{tag} {which_label} n={len(rows)}",
                      flush=True)
    return entries


def disambiguate(entries):
    """Two run dirs can reduce to one arm name -- `baseline_indist` (minADE@8 only) and
    `baseline_ada_ps_indist` both stem to `baseline`. The registry name wins; the others
    fall back to their directory, so no (arm, set) pair is ever ambiguous."""
    seen = set()
    for e in sorted(entries, key=lambda x: (x["dir"] not in DIR2ARM, x["dir"], x["tag"])):
        for name in (e["arm"], e["dir"], f"{e['dir']}/{e['tag']}"):
            if (name, e["set"]) not in seen:
                e["arm"] = name
                break
        seen.add((e["arm"], e["set"]))


def pick_baseline(entries, e):
    """Unpruned reference for one arm: same set, same architecture, same metric depth."""
    want_ps = e.get("minADE6_mean") is not None
    cand = [b for b in entries
            if b["tag"] == "baseline" and b["set"] == e["set"] and b["arch"] == e["arch"]
            and (b.get("minADE6_mean") is not None) == want_ps]
    if not cand:
        return None
    # prefer the per-sample rerun, then the one covering the most clips
    return min(cand, key=lambda b: (("_ps_" not in b["dir"]), -b["n_clips"]))


def add_paired(entries, k):
    """Paired median dminADE@k vs the architecture-matched baseline, with a bootstrap CI."""
    for e in entries:
        if e["tag"] == "baseline":
            continue
        b = pick_baseline(entries, e)
        if b is None:
            continue
        e["baseline_ref"] = b["dir"]
        ids = sorted(set(b["rows"]) & set(e["rows"]))
        if not ids:
            continue
        key = "ade_rollout_k" if e.get("minADE6_mean") is not None else None
        if key:
            d = np.array([at_k(e["rows"][i], key, k) - at_k(b["rows"][i], key, k)
                          for i in ids])
        else:
            d = np.array([e["rows"][i]["minADE_rollout"] - b["rows"][i]["minADE_rollout"]
                          for i in ids])
        med, lo, hi = boot_ci(d)
        e["d_minADE6_median"] = med
        e["d_ci_lo"] = lo
        e["d_ci_hi"] = hi
        e["d_sig"] = "*" if lo > 0 or hi < 0 else ""
        e["d_n"] = len(ids)


def mean_boot_ci(x, seed=0):
    """Mean and its bootstrap CI -- the closed-loop convention (half the scenes sit at
    the 1.0 ceiling, so the mean CI is the primary reading and Wilcoxon is secondary)."""
    x = np.asarray([v for v in x if v is not None], dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return None, None, None
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(BOOT, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def per_scene(rollouts, key):
    """scene -> mean of key over that scene's rollouts (mirrors analyze_alpasim)."""
    by = {}
    for r in rollouts:
        v = r.get(key)
        if v is not None:
            by.setdefault(r["scene"], []).append(float(v))
    return {s: float(np.mean(v)) for s, v in by.items() if v}


def load_alpasim_run(path):
    """One run's rollouts, flattened. Plain JSON -- no alpasim venv needed (only the CoC
    text lives in the ASL protobuf logs, and that is read from the analysed dirs)."""
    d = json.loads(path.read_text())
    out = []
    for r in d["rollouts"]:
        m = r.get("metrics", {})
        row = {"scene": r["clipgt_id"], "status": r.get("status"),
               "score": r.get("score"), "passed": float(bool(r.get("passed")))}
        row.update({k: m.get(k) for k in CL_METRIC_KEYS})
        out.append(row)
    return out


def collect_coc_stats():
    """checkpoint -> CoC degeneracy, from whichever analysed dir carries it. Closed-loop
    CoC text comes from the ASL logs, so it only exists where analyze_alpasim has run."""
    out = {}
    for exp_dir in sorted((REPO / "outputs").glob("alpasim_*")):
        m = exp_dir / "metrics.json"
        if not m.exists():
            continue
        try:
            d = json.loads(m.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(d, dict) or d.get("n_scenes") != CL_SCENES:
            continue
        for name, c in (d.get("coc") or {}).items():
            out.setdefault(name, c)
    return out


def collect_closedloop(verbose):
    """One row per driver config, read from the raw alpasim runs.

    Only the reported shape is kept: exactly CL_SCENES scenes x CL_ROLLOUTS rollouts.
    That drops the 30-scene pilots (underpowered: sigma ~ 0.32 per scene means 30 scenes
    resolve only 0.179) and the partial shards, and it admits runs that were never fed
    through `analyze_alpasim` -- reading the run JSON directly is what makes those
    visible. Aggregation is per-rollout -> per-scene mean -> config mean, and the paired
    delta is per-scene against the baseline run of the same shape.
    """
    runs = {}
    for f in sorted(ALPASIM_RUNS.glob("*/aggregate/results-summary.json")):
        run = f.parent.parent.name
        try:
            rollouts = load_alpasim_run(f)
        except (json.JSONDecodeError, KeyError):
            continue
        scenes = {r["scene"] for r in rollouts}
        if len(scenes) != CL_SCENES or len(rollouts) != CL_ROLLOUTS:
            continue
        runs[run] = rollouts
        if verbose:
            print(f"  [closedloop] {run} {len(scenes)}x{len(rollouts) // len(scenes)}",
                  flush=True)

    # the config name is the run dir minus its prefix; the baseline run is the one whose
    # config is exactly "baseline"
    def config_of(run):
        """Run dir -> driver checkpoint name (`fmp_G_default_r40_merged` ->
        `G_default_r40`, which is the dir under alpasim's drivers/)."""
        for p in ("m2601_merged_", "m2601_150_", "matrix_", "fmp_"):
            if run.startswith(p):
                run = run[len(p):]
                break
        return run.removesuffix("_merged")

    base = next((r for run, r in runs.items() if config_of(run) == "baseline"), None)
    base_scene = per_scene(base, "score") if base else {}
    coc = collect_coc_stats()

    rows = []
    for run, rollouts in sorted(runs.items(), key=lambda kv: config_of(kv[0])):
        cfg = config_of(run)
        r = {"run": run, "config": cfg, "n_scenes": len({x["scene"] for x in rollouts}),
             "n_rollouts": len(rollouts)}
        for k in CL_ROLLOUT_KEYS + CL_METRIC_KEYS:
            sc = per_scene(rollouts, k)
            if sc:
                r[k] = float(np.mean(list(sc.values())))
        sc = per_scene(rollouts, "score")
        _, lo, hi = mean_boot_ci(list(sc.values()))
        r["score_ci_lo"], r["score_ci_hi"] = lo, hi
        # repeat noise: |score difference| between the two rollouts of one scene
        by = {}
        for x in rollouts:
            if x.get("score") is not None:
                by.setdefault(x["scene"], []).append(float(x["score"]))
        rep = [abs(v[0] - v[1]) for v in by.values() if len(v) == 2]
        if rep:
            r["repeat_abs_diff_mean"] = float(np.mean(rep))
            r["repeat_abs_diff_max"] = float(np.max(rep))
        r["n_failed_rollouts"] = sum(1 for x in rollouts
                                     if x["status"] != "pass" and not x["passed"])
        c = coc.get(cfg) or coc.get(f"slim_{cfg}") or {}
        r["coc_degen"] = c.get("mean_degenerate_frac", "")
        r["coc_empty"] = c.get("mean_empty_frac", "")
        r["coc_soup"] = c.get("mean_soup_frac", "")
        r["coc_len"] = c.get("mean_len", "")
        if base is not None and cfg != "baseline":
            ids = sorted(set(sc) & set(base_scene))
            d = np.array([sc[i] - base_scene[i] for i in ids])
            if len(d):
                mean, lo, hi = mean_boot_ci(d)
                r["d_score_mean"], r["d_ci_lo"], r["d_ci_hi"] = mean, lo, hi
                nz = d[d != 0]
                r["wilcoxon_p"] = float(wilcoxon(nz).pvalue) if len(nz) >= 5 else ""
                r["wins"] = int((d > 0).sum())
                r["losses"] = int((d < 0).sum())
                r["ties"] = int((d == 0).sum())
        rows.append(r)
    return rows


def collect_lingoqa(verbose):
    """VQA accuracy from the scoring dirs, plus the CoC-condition judge runs."""
    seen, rows = {}, []
    for exp_dir in sorted((REPO / "outputs").glob("lingo_vqa_scores*")):
        m = exp_dir / "metrics.json"
        if not m.exists():
            continue
        try:
            d = json.loads(m.read_text())
        except json.JSONDecodeError:
            continue
        for run in d.get("runs", []):
            eid = run.get("exp_id")
            if eid in seen:
                continue
            seen[eid] = exp_dir.name
            rows.append({
                "exp_id": eid, "arm": run.get("arm", ""), "protocol": "vqa",
                "style": run.get("style") or "", "n_questions": run.get("n_predictions", ""),
                "n_matched": run.get("n_matched", ""), "accuracy": run.get("accuracy", ""),
                "judge_pct": round(run["accuracy"] * 100, 1) if "accuracy" in run else "",
                "mean_logit": run.get("mean_logit", ""),
                "answer_words_median": run.get("answer_words_median", ""),
                "answer_words_mean": run.get("answer_words_mean", ""),
                "truncated_frac": run.get("truncated_frac", ""),
                "source_dir": exp_dir.name,
            })
    for exp_dir in sorted((REPO / "outputs").glob("lingo_judge_*")):
        m = exp_dir / "metrics.json"
        if not m.exists():
            continue
        try:
            d = json.loads(m.read_text())
        except json.JSONDecodeError:
            continue
        if "accuracy" not in d:
            continue
        rows.append({
            "exp_id": exp_dir.name, "arm": exp_dir.name[len("lingo_judge_"):],
            "protocol": "coc_judge", "style": d.get("condition", ""),
            "n_questions": d.get("n_questions", ""), "n_matched": "",
            "accuracy": d["accuracy"], "judge_pct": round(d["accuracy"] * 100, 1),
            "mean_logit": d.get("mean_score", ""), "answer_words_median": "",
            "answer_words_mean": "", "truncated_frac": "", "source_dir": exp_dir.name,
        })
    if verbose:
        print(f"  [lingoqa] {len(rows)} runs", flush=True)
    return rows


def cell(e, key, fmt="{:.4f}"):
    """One summary cell; a run predating per-sample storage prints `-`, not a zero."""
    v = e.get(key, "")
    return fmt.format(v) if isinstance(v, float) else "-"


def write_csv(path, cols, rows):
    """Floats go out at 6 significant digits -- full float64 repr triples the file and
    no cell here is meaningful past the 4th decimal."""
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: f"{r[c]:.6g}" if isinstance(r.get(c), float) else r.get(c, "")
                        for c in cols})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_index")
    ap.add_argument("--k", type=int, default=6, help="samples for minADE@k (frozen: 6)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    verbose = not args.quiet

    ol = collect_openloop(args.k, verbose)
    disambiguate(ol)
    add_paired(ol, args.k)
    ol.sort(key=lambda e: (e["arm"], SET_SUFFIX.index(e["set"])
                           if e["set"] in SET_SUFFIX else 9, e["dir"]))
    cl = collect_closedloop(verbose)
    lq = collect_lingoqa(verbose)

    write_csv(out_dir / "openloop.csv", OPENLOOP_COLS, ol)
    write_csv(out_dir / "closedloop.csv", CLOSED_COLS, cl)
    write_csv(out_dir / "lingoqa.csv", LINGO_COLS, lq)

    # rows are the per-clip records; they stay out of the json, only the cells go in
    slim = [{c: e.get(c, "") for c in OPENLOOP_COLS} for e in ol]
    (out_dir / "metrics.json").write_text(json.dumps(
        {"openloop": slim, "closedloop": cl, "lingoqa": lq}, indent=2))
    (out_dir / "config.json").write_text(json.dumps({
        "k": args.k, "protocol": "rollout only; OOD cut to split==val; minADE@k mean",
        "boot": BOOT, "paired_baseline": "same set, same GPU architecture",
        "closedloop_shape": f"{CL_SCENES} scenes x {CL_ROLLOUTS} rollouts only",
        "horizons": "minADE/minFDE@k at 6.4 s (no suffix), 1.6 s (_h16), 3.2 s (_h32); "
                    "the min over samples is taken independently at each horizon",
        "omitted": "teacher-forced columns (minADE_tf / minFDE_tf / nll_gtcoc) are not "
                   "reported: no Ada baseline carries per-sample TF arrays, so the "
                   "column would have arms but no anchor",
        "sources": {"openloop": "outputs/*/<tag>_s*of*.json",
                    "closedloop": f"{ALPASIM_RUNS}/*/aggregate/results-summary.json "
                                  "(CoC degeneracy from outputs/alpasim_*/metrics.json)",
                    "lingoqa": "outputs/lingo_vqa_scores*/metrics.json + "
                               "outputs/lingo_judge_*/metrics.json"},
    }, indent=2))

    per_set = {}
    for e in ol:
        per_set[e["set"]] = per_set.get(e["set"], 0) + 1
    counts = " ".join(f"{s}={n}" for s, n in sorted(per_set.items()))
    n_runs = len({r["run"] for r in cl})
    head = (f"{'arm':22s} {'set':7s} {'n':>5s} {'arch':10s} "
            f"{'minADE@' + str(args.k):>9s} {'minFDE@' + str(args.k):>9s} "
            f"{'degen':>7s} {'d_med':>9s}")
    cl_line = (f"closed loop {len(cl)} configs at "
               f"{CL_SCENES}x{CL_ROLLOUTS // CL_SCENES} ({n_runs} alpasim runs)")
    lines = [f"open loop   {len(ol)} rows  {counts}", cl_line,
             f"lingoqa     {len(lq)} rows", "", head]
    for e in ol:
        lines.append(f"{e['arm'][:22]:22s} {e['set']:7s} {e['n_clips']:5d} "
                     f"{e['arch'][:10]:10s} {cell(e, 'minADE6_mean'):>9s} "
                     f"{cell(e, 'minFDE6_mean'):>9s} "
                     f"{cell(e, 'coc_degen', '{:.3f}'):>7s} "
                     f"{cell(e, 'd_minADE6_median', '{:+.4f}'):>9s}{e.get('d_sig', '')}")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:4]), flush=True)
    print(f"saved -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
