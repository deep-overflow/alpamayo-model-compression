"""Collect every finished alpasim closed-loop run into one arm x metric table.

Two scene suites have been run and they are disjoint by construction
(`make_hard_suite.py` asserts it, re-verified here): the 150-scene prefix of
`public_2601` and the 100-scene `hard100` band drawn from its 80-100th difficulty
percentile. An arm therefore has up to two independent readings and they must never be
pooled -- the 150 is measurably easier than the suite it came from, so its absolute rates
are optimistic, while hard100 was selected to sit near 0.50.

Runs are discovered by directory shape across three roots (ours, soowon's LLM-Pruner runs,
and the fm-expert-pruning arm), so a newly merged run appears without editing a registry.

Aggregation follows `analyze_alpasim.py` exactly so numbers stay comparable with the
shipped reports: score is per-scene mean over rollouts then mean over scenes with a
bootstrap CI, every rate is a fraction of *rollouts*, and the paired delta is per-scene
against that suite's own baseline.

Runs with the alpasim repo's venv (needs alpasim_utils for the CoC logs):

    cd /home/cvlab21/project/chan/alpasim && uv run python \
        <repo>/experiments/head_analysis/collect_alpasim_table.py \
        --out <repo>/outputs/alpasim_table
"""

import argparse
import asyncio
import json
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

CHAN = Path("/home/cvlab21/project/chan/alpasim-runs")
SOOWON = Path("/mnt/nvme1n1/ad_vla/outputs/soowon/alpasim-runs")
DRIVERS = Path("/mnt/nvme1n1/ad_vla/data/alpasim/drivers")

# (run-dir glob, suite, forced label).  baseline has no driver dir -- the wizard gets no
# checkpoint override -- so its param count is 0 by construction.
SOURCES = [
    (CHAN / "m2601_merged_*", "s150", None),
    (CHAN / "fmp_G_default_r40_merged", "s150", "G_default_r40"),
    (SOOWON / "cl150_merged_*", "s150", None),
    (CHAN / "h100_merged_*", "hard100", None),
]

# `offroad_or_collision_at_fault` is the score's own pass/fail gate: score_criteria says
# score = 0 unless collision_at_fault == 0 and offroad == 0, so carrying it lets the table
# check its own accounting (gate rate must equal the share of zero-scored rollouts).
RATE = ["collision_at_fault", "collision_any", "collision_rear", "offroad",
        "offroad_or_collision_at_fault", "wrong_lane", "safety_monitor_triggered",
        "passed"]
MEAN = ["progress_rel", "progress_clipped_rel", "dist_to_gt_trajectory",
        "min_distance_to_obstacle_m", "dist_traveled_m", "plan_deviation"]


def discover():
    """-> {suite: {label: run_dir}}, {label: driver_name}."""
    runs, drivers = {}, {}
    for pat, suite, forced in SOURCES:
        cands = sorted(pat.parent.glob(pat.name)) if "*" in pat.name else [pat]
        for run in cands:
            if not (run / "aggregate" / "results-summary.json").exists():
                continue
            name = run.name
            for p in ("m2601_merged_", "cl150_merged_", "h100_merged_"):
                name = name.removeprefix(p)
            label = (forced or name).removeprefix("slim_")
            runs.setdefault(suite, {})[label] = run
            if label != "baseline":
                drivers[label] = forced or name
    return runs, drivers


def load_rollouts(run_dir):
    d = json.loads((run_dir / "aggregate" / "results-summary.json").read_text())
    rows = []
    for r in d["rollouts"]:
        m = r["metrics"]
        # a rollout whose metrics carry an `error` never produced any metric: alpasim
        # scores it 0.  On hard100 these are all the same route sanity check ("route folds
        # back on itself"), with an identical waypoint index in both rollouts of a scene,
        # so the defect is in the map -- but which arms hit it depends on how far each
        # drives, so it is NOT automatically paired across arms.  main() therefore drops
        # the union over arms, not each arm's own set.
        rows.append({"scene": r["clipgt_id"], "rollout_id": r["rollout_id"],
                     "score": float(r["score"]) if r["score"] is not None else np.nan,
                     "passed": float(bool(r["passed"])),
                     "unscorable": "error" in m,
                     **{k: m.get(k) for k in RATE + MEAN if k in m}})
    return rows


async def _read_coc(asl_path):
    from alpasim_utils.logs import async_read_pb_log

    texts = []
    async for entry in async_read_pb_log(str(asl_path)):
        if entry.HasField("driver_return"):
            blob = entry.driver_return.debug_info.unstructured_debug_info
            if blob:
                try:
                    t = pickle.loads(blob).get("reasoning_text")
                    if t:
                        texts.append(str(t))
                # deliberately as broad as analyze_alpasim's: a driver blob that does not
                # unpickle must count as "no CoC at this step", not abort the arm, or the
                # degeneracy rates stop being comparable with the shipped reports
                except Exception:  # noqa: BLE001, S110
                    pass
    return texts


def coc_stats(texts):
    """Degeneracy heuristics -- thresholds identical to analyze_alpasim.coc_stats."""
    if not texts:
        return {"n_steps": 0, "degenerate_frac": np.nan, "empty_frac": np.nan,
                "soup_frac": np.nan, "mean_len": np.nan}
    empty, soup, lens = [], [], []
    for t in texts:
        s = t.strip()
        if s.startswith("['") and s.endswith("']"):
            s = s[2:-2]
        s = s.strip()
        is_empty = len(s) == 0
        words = s.split()
        ur = len(set(words)) / max(len(words), 1)
        na = sum(ord(ch) > 127 for ch in s) / max(len(s), 1)
        is_soup = (not is_empty) and (na > 0.05 or ur < 0.5 or len(s) > 300)
        empty.append(is_empty)
        soup.append(is_soup)
        if not is_empty:
            lens.append(len(s))
    return {"n_steps": len(texts), "empty_frac": float(np.mean(empty)),
            "soup_frac": float(np.mean(soup)),
            "degenerate_frac": float(np.mean(np.asarray(empty) | np.asarray(soup))),
            "mean_len": float(np.mean(lens)) if lens else np.nan}


def coc_of_rollout(asl_path):
    """Top-level so a process pool can pickle it.  Parsing an ASL is protobuf+pickle work,
    i.e. GIL-bound, so threads would not help -- 7,100 rollouts is 1 h serial, 8 min over
    8 processes."""
    p = Path(asl_path)
    return coc_stats(asyncio.run(_read_coc(p)) if p.exists() else [])


def per_scene(rows, key):
    out = {}
    for r in rows:
        out.setdefault(r["scene"], []).append(r[key])
    return {s: float(np.nanmean(v)) for s, v in out.items()}


def boot_ci(x, n_boot=10000, seed=0, alpha=0.05):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), \
        float(np.quantile(means, 1 - alpha / 2))


def wilcoxon_p(d):
    from scipy.stats import wilcoxon

    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    d = d[d != 0]
    return float(wilcoxon(d).pvalue) if len(d) >= 5 else None


def fisher(a_hit, a_n, b_hit, b_n):
    """Rate vs the baseline's rate.  Two rollouts of one scene are not independent, so
    this is a screening statistic, not a licence to claim significance from it alone."""
    from scipy.stats import fisher_exact

    odds, p = fisher_exact([[a_hit, a_n - a_hit], [b_hit, b_n - b_hit]])
    return float(odds), float(p)


def params_of(driver):
    if driver is None:
        return {"removed": 0, "pct": 0.0}
    meta, cfg = DRIVERS / driver / "slim_meta.json", DRIVERS / driver / "config.json"
    d = json.loads(meta.read_text()) if meta.exists() else {}
    p = dict(d.get("params") or {})
    if not p.get("removed") and cfg.exists():
        c = json.loads(cfg.read_text())
        p = {"full": c.get("params_before"), "removed": c.get("removed_params")}
    if not p.get("removed"):
        return {"removed": None, "pct": None}
    full = p.get("full") or 11078526194
    return {"removed": int(p["removed"]), "pct": 100.0 * p["removed"] / full}


def pooled(keep_rows, metrics):
    """Arms measured on BOTH suites, combined.

    The suites are disjoint, so combining is just a mean over the union of scenes -- but
    there are two defensible ways to weight it and they answer different questions:

    - `score_pooled` (n=250): every scene counts once, so the 150 carries 60% of the
      weight.  This is the highest-powered estimate of an arm's effect and is what the
      paired delta, its CI and the Wilcoxon are computed on.
    - `score_macro`: the unweighted mean of the two suite means, i.e. easy and hard count
      50/50.  Neither is a sample of `public_2601` (one is its easy prefix, the other its
      80-100th difficulty band), so this is the "one number per difficulty regime" reading.

    Neither is an estimate of the arm's score on the full 913-scene suite; say which one a
    quoted number is.  Rates are pooled over rollouts, and CoC over the rollouts that
    actually produced text (n_rollouts - coc_missing), which is not all of them.
    """
    suites = list(keep_rows)
    both = sorted(set.intersection(*(set(k) for k in keep_rows.values())))
    base = {s: per_scene(keep_rows[s]["baseline"], "score") for s in suites}
    out = {"suites": suites, "arms": {},
           "n_scenes": sum(metrics["suites"][s]["n_scenes"] for s in suites)}

    for label in both:
        rows = [r for s in suites for r in keep_rows[s][label]]
        sc, bs = {}, {}
        for s in suites:
            sc.update(per_scene(keep_rows[s][label], "score"))
            bs.update(base[s])
        scenes = sorted(sc)
        per = [metrics["suites"][s]["arms"][label] for s in suites]
        a = {"n_scenes": len(scenes), "n_rollouts": len(rows),
             "params": per[0]["params"],
             "by_suite": {s: metrics["suites"][s]["arms"][label]["score"] for s in suites}}
        m, lo, hi = boot_ci([sc[t] for t in scenes])
        a["score_pooled"], a["score_ci_lo"], a["score_ci_hi"] = m, lo, hi
        a["score_macro"] = float(np.mean([v for v in a["by_suite"].values()]))
        if label != "baseline":
            d = [sc[t] - bs[t] for t in scenes]
            dm, dlo, dhi = boot_ci(d)
            a["d_score"], a["d_lo"], a["d_hi"] = dm, dlo, dhi
            a["d_p"] = wilcoxon_p(d)
            a["wins"] = int(np.sum(np.asarray(d) > 0))
            a["losses"] = int(np.sum(np.asarray(d) < 0))
            a["d_macro"] = float(np.mean(
                [metrics["suites"][s]["arms"][label]["d_score"] for s in suites]))
        for k in RATE:
            v = [r[k] for r in rows if r.get(k) is not None]
            a[k] = float(100 * np.nanmean(v)) if v else None
            a[k + "_n"] = int(np.nansum(v)) if v else None
        for k in MEAN:
            v = [r[k] for r in rows if r.get(k) is not None]
            a[k] = float(np.nanmean(v)) if v else None
        allsc = np.asarray([r["score"] for r in rows], float)
        a["perfect_pct"] = float(100 * np.mean(allsc >= 0.999))
        a["zero_pct"] = float(100 * np.mean(allsc <= 0.001))
        by = {}
        for r in rows:
            by.setdefault(r["scene"], []).append(r["score"])
        rep = [abs(v[0] - v[1]) for v in by.values() if len(v) >= 2]
        a["repeat_abs_diff"] = float(np.nanmean(rep)) if rep else None
        # CoC fractions are means over the rollouts that produced text, so pool by that
        # count -- weighting by n_rollouts would silently include the ones that produced none
        w = [p["n_rollouts"] - p["coc_missing"] for p in per]
        for k in ("coc_degenerate_frac", "coc_empty_frac", "coc_soup_frac", "coc_mean_len"):
            a[k] = float(np.average([p[k] for p in per], weights=w))
        a["coc_missing"] = int(sum(p["coc_missing"] for p in per))
        out["arms"][label] = a

    # the two suites disagreed on the ranking (150: dual > lp > tyr; hard100: dual > tyr >
    # lp), so the question pooling exists to answer is whether n=250 separates any pair at
    # all.  Every pair, not just vs baseline.
    sc_all = {}
    for label in both:
        d = {}
        for s in suites:
            d.update(per_scene(keep_rows[s][label], "score"))
        sc_all[label] = d
    scenes = sorted(sc_all[both[0]])
    out["pairs"] = {}
    for i, x in enumerate(both):
        for y in both[i + 1:]:
            d = [sc_all[y][t] - sc_all[x][t] for t in scenes]
            dm, dlo, dhi = boot_ci(d)
            out["pairs"][f"{y}-{x}"] = {
                "delta": dm, "lo": dlo, "hi": dhi, "p": wilcoxon_p(d),
                "wins": int(np.sum(np.asarray(d) > 0)),
                "losses": int(np.sum(np.asarray(d) < 0))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skip-coc", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--coc-from", type=Path,
                    help="an earlier metrics.json to carry CoC stats over from; the ASL "
                         "logs of a merged run never change, so re-running for the score "
                         "bookkeeping alone need not reparse them")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    runs, drivers = discover()
    all_scenes, metrics = {}, {"suites": {}, "drivers": drivers}
    prev = {}
    if args.coc_from and args.coc_from.exists():
        old = json.loads(args.coc_from.read_text())
        prev = {(s, a): v for s, d in old["suites"].items()
                for a, v in d["arms"].items()}
        print(f"reusing CoC for {len(prev)} arm-suites from {args.coc_from}")

    keep_rows = {}
    for suite, arms in runs.items():
        rows_by = {label: load_rollouts(run) for label, run in sorted(arms.items())}
        keep_rows[suite] = rows_by
        base_rows = rows_by["baseline"]
        base_scene = per_scene(base_rows, "score")
        scenes = sorted(base_scene)
        all_scenes[suite] = set(scenes)
        # a scene the map defect killed for ANY arm is dropped from EVERY arm: which arms
        # reach the bad waypoint depends on how far each drives, so keeping each arm's own
        # survivors would compare different scene sets
        dead = sorted({r["scene"] for rows in rows_by.values()
                       for r in rows if r["unscorable"]})
        live = [s for s in scenes if s not in dead]
        out = {"n_scenes": len(scenes), "unscorable_scenes": dead,
               "n_scenes_clean": len(live), "arms": {}}

        for label, run in sorted(arms.items()):
            rows = rows_by[label]
            sc = per_scene(rows, "score")
            if set(sc) != set(scenes):
                print(f"  WARNING {suite}/{label}: scene set differs, skipped")
                continue
            a = {"run": str(run), "n_rollouts": len(rows),
                 "params": params_of(drivers.get(label))}
            m, lo, hi = boot_ci([sc[s] for s in scenes])
            a["score"], a["score_ci_lo"], a["score_ci_hi"] = m, lo, hi
            a["score_median"] = float(np.median([sc[s] for s in scenes]))
            allsc = np.asarray([r["score"] for r in rows], float)
            a["perfect_pct"] = float(100 * np.mean(allsc >= 0.999))
            a["zero_pct"] = float(100 * np.mean(allsc <= 0.001))
            by = {}
            for r in rows:
                by.setdefault(r["scene"], []).append(r["score"])
            rep = [abs(v[0] - v[1]) for v in by.values() if len(v) >= 2]
            a["repeat_abs_diff"] = float(np.nanmean(rep)) if rep else None
            a["own_unscorable_scenes"] = sorted({r["scene"] for r in rows
                                                 if r["unscorable"]})
            a["score_clean"] = float(np.mean([sc[t] for t in live])) if live else None

            for k in RATE:
                v = [r[k] for r in rows if r.get(k) is not None]
                a[k] = float(100 * np.nanmean(v)) if v else None
                a[k + "_n"] = int(np.nansum(v)) if v else None
            for k in MEAN:
                v = [r[k] for r in rows if r.get(k) is not None]
                a[k] = float(np.nanmean(v)) if v else None

            if label != "baseline":
                d = [sc[s] - base_scene[s] for s in scenes]
                dm, dlo, dhi = boot_ci(d)
                a["d_score"], a["d_lo"], a["d_hi"] = dm, dlo, dhi
                a["d_p"] = wilcoxon_p(d)
                a["wins"] = int(np.sum(np.asarray(d) > 0))
                a["losses"] = int(np.sum(np.asarray(d) < 0))
                dc = [sc[s] - base_scene[s] for s in live]
                a["d_score_clean"] = float(np.mean(dc)) if dc else None
                a["d_p_clean"] = wilcoxon_p(dc) if dc else None
                for k in ("collision_at_fault", "offroad"):
                    bh = int(np.nansum([r[k] for r in base_rows if r.get(k) is not None]))
                    a[k + "_or"], a[k + "_p"] = fisher(a[k + "_n"], len(rows),
                                                       bh, len(base_rows))

            cached = prev.get((suite, label))
            if cached and "coc_degenerate_frac" in cached:
                # CoC is the expensive half and depends only on the ASL logs, which do not
                # change once a run is merged -- so a re-run that only touches the score
                # bookkeeping can carry it over instead of reparsing 7,100 rollouts
                for k in ("coc_degenerate_frac", "coc_empty_frac", "coc_soup_frac",
                          "coc_mean_len", "coc_missing"):
                    a[k] = cached.get(k)
            elif not args.skip_coc:
                asls = [str(run / "rollouts" / r["scene"] / r["rollout_id"] / "rollout.asl")
                        for r in rows]
                with ProcessPoolExecutor(max_workers=args.workers) as pool:
                    per_roll = list(pool.map(coc_of_rollout, asls, chunksize=4))
                for k in ("degenerate_frac", "empty_frac", "soup_frac"):
                    a["coc_" + k] = float(100 * np.nanmean([p[k] for p in per_roll]))
                a["coc_mean_len"] = float(np.nanmean([p["mean_len"] for p in per_roll]))
                a["coc_missing"] = int(sum(p["n_steps"] == 0 for p in per_roll))
            out["arms"][label] = a
            print(f"  {suite:8s} {label:28s} score {a['score']:.3f}  "
                  f"cocdeg {a.get('coc_degenerate_frac', float('nan')):5.1f}%", flush=True)
        metrics["suites"][suite] = out

    ov = all_scenes["s150"] & all_scenes["hard100"]
    metrics["suite_overlap"] = len(ov)
    assert not ov, f"suites overlap in {len(ov)} scenes -- they are not independent"
    metrics["both"] = pooled(keep_rows, metrics)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(f"\nwrote {args.out / 'metrics.json'}  (suite overlap {len(ov)} scenes)")


if __name__ == "__main__":
    main()
