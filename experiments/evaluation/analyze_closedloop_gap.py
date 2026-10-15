"""Why does the dense model score below the pruned ones in closed loop? Stored logs only, no GPU.

Table 2 has the dense model at 0.750 against 0.78-0.83 for every non-collapsed pruned arm, all
of it in `progress`. Three explanations were never separated
(plans/2026-10-07_closedloop-gap-decomposition.md):

  H-route  the dense model takes a valid route other than the recorded one more often, and
           scoring stops 4 m from the recorded path
  H-stop   pruned models stop and hesitate less and so cover more of the route in 20 s
  H-gate   the difference is in off-road / collision events after all

alpasim's chain is reimplemented exactly (`processing.DEFAULT_MODIFIERS` +
`main._run_aggregation`): drop timesteps before `eval_relevant`; keep up to the first
`collision_any | offroad`; keep up to the first `dist_to_gt_trajectory >= 4.0`;
`progress_clipped_rel = last(progress_rel_to_total)`; `progress_score = min(clamp(p,0,1)/0.8, 1)`
(1.0 when `gt_dist_traveled_m < 5`); score 0 on an at-fault collision or off-road. With P(t)
the progress score at time t, each rollout's loss splits exactly into

    1 - score = [1 - P(T_end)]            slow    route not covered even with no truncation
              + [P(T_end) - P(t_gate)]    frozen  progress frozen at a collision / off-road
              + [P(t_gate) - P(t_ship)]   cut4m   progress frozen by the 4 m rule
              + [P(t_ship) - score]       zeroed  score set to 0 by a gate

so `score_arm - score_dense` is four scene-paired differences. A rollout cut at 4 m with P < 1
is followed into the unscored remainder: no gate and >= 10 m more driven = alternative route
(the "multi-answer" case); a later off-road / at-fault collision = hidden failure; back inside
4 m = excursion.

Gates, fixed before running:
  G0  the recomputed score equals the shipped one on every rollout with a parquet (< 1e-6)
  G1  a term explains MoOP - dense if its scene-paired CI excludes 0 and it is >= half the gap
      on Origin150, same sign on Hard100
  G2  H-route: the dense alternative-route rate exceeds MoOP's, CI excluding 0 on both sets
  G3  H-stop: the dense `slow` term or stop-time share exceeds MoOP's, CI excluding 0 on both

Added after the first run, so NOT pre-registered (sections marked "post hoc"):

  * the same loss split by the one rule that ended the scored window (`gate` / `rear` / `cut` /
    `full`). The telescoping split above spreads a zeroed rollout's loss over three terms using
    whatever the car did after the event; this one gives every rollout to exactly one cause.
  * the human driver as the yardstick. The usdz holds the recorded ego track and it shares the
    sim's clock (GT arc length rebuilt from the parquet agrees to a median 1 mm), so each
    timestep has the human's position along the route. That gives pace relative to the human,
    metres ahead / behind, stops the human did not make, and what the model did at each place
    the human stood still.

Traffic is replayed, not reactive (`trafficsim.skip: true`), which is what makes a rear-end
collision a symptom of being slower than the recording.

    .venv/bin/python experiments/evaluation/analyze_closedloop_gap.py --exp-id closedloop_gap_v1
"""

import argparse
import json
import re
import zipfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, INK, MUTED, GRID = "#FAF9F5", "#29261B", "#6B6555", "#E6E3DA"
# categorical slots in fixed order and a three-step ordinal blue; both pass the dataviz validator on #FAF9F5
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
ORD = ["#86b6ef", "#3987e5", "#104281"]
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})

RUNS = Path("/home/cvlab21/project/chan/alpasim-runs")
SOOWON = Path("/mnt/nvme1n1/ad_vla/outputs/soowon/alpasim-analysis/runs_root")
SANGOH = Path("/mnt/nvme1n1/ad_vla/results/sangoh/alpasim_runs/eval2601_a1_5")
USDZ = Path("/mnt/nvme1n1/ad_vla/data/nre-artifacts/all-usdzs")
O = Path("/mnt/nvme1n1/ad_vla/outputs/chan")
SCENES = O / "scene_difficulty/scene_difficulty_public2601.csv"   # scene_id -> artifact uuid

ARMS = [
    ("baseline", "baseline", "Baseline"),
    ("dual", "slim_dual_u40_v2", "MoOP"),
    ("dual+E", "slim_dualexp_u40_em93p75", "MoOP (+expert)"),
    ("traj", "slim_traj_u40_v2", "FM-only"),
    ("coc", "slim_coc_u40_v2", "CE-only"),
    ("tyr", "slim_tyr_u40_r", "Tyr"),
    ("lp", "lp_r50", "LLM-Pruner"),
    ("wanda", "slim_wanda_u40_v2", "Wanda"),
]
LABEL = {k: lab for k, _, lab in ARMS}
SETS = [("origin150", "m2601_merged_", 150), ("hard100", "h100_merged_", 100)]
NEED = ["eval_relevant", "collision_any", "collision_front", "collision_lateral", "collision_rear",
        "offroad", "dist_to_gt_trajectory", "progress_rel_to_total", "dist_traveled_m",
        "gt_dist_traveled_m"]
MAX_DEV, SAT, MIN_GT = 4.0, 0.8, 5.0   # base_config.yaml: aggregation_modifiers / scene_score
STOP_V, ALT_M = 0.5, 10.0              # stopped below 0.5 m/s; "kept driving" = 10 m after the cut
ROUTE_END = 0.999                      # progress_rel_to_total at which the recorded route is done
MIN_STOP_S, AT_M = 1.0, 15.0           # a stop lasts >= 1 s; "the same place" = within 15 m
BAND_IN, BAND_OUT = 15.0, 10.0         # the place the human stood: from 15 m before it to 10 m past it
AHEAD_S, EARLY_S, SLOW_V, LEAD_M = 2.0, 1.0, 2.0, 5.0
# post hoc (2026-10-12): a human stop that begins within START_S of the scored clip's start is under way at
# the hand-over, so the car is handed over standing - "stopped" is automatic there and a drive-through
# impossible. Those places only pad the denominator of the drive-through rate.
START_S = 2.0
TERMS = ["slow", "frozen", "cut4m", "zeroed"]   # pre-registered telescoping split
WHY = ["gate", "rear", "cut", "full"]           # post hoc: the rule that ended the scored window
CUTS = ["route_end", "saturated", "alt_route", "excursion", "hidden_failure", "stalled"]
STOPS = ["stopped", "stopped_on", "left_early", "crawled", "drove_through", "crashed", "unresolved", "not_reached"]
CROSSED = ["stopped", "stopped_on", "left_early", "crawled", "drove_through"]   # dealt with the place, one way or another
STOP_HEAD = (f"  {'arm':15s}    n  stopped  ...within 30 m on  left early  crawled  drove through = before + during"
             f" + after the human stood  crashed  unresolved  not reached | crossed  drove through, of crossed"
             f" [95% CI]  ... while the human stood")


def run_dir(prefix, cfg, n):
    d = RUNS / f"{prefix}{cfg}"
    if d.exists():
        return d
    if cfg == "lp_r50" and n == 150:
        return SOOWON / "lp_r50"   # the 150-scene LLM-Pruner run lives in soowon's runs_root
    return None


def gt_track(uuid):
    """Recorded ego track from the usdz (a ZIP; only the 68 KB rig file is read).

    Returns (t_us (N,), s (N,)): absolute time and arc length along the recorded path.
    Same parse as outputs/scene_difficulty/extract_scene_feats.py.
    """
    z = zipfile.ZipFile(USDZ / f"{uuid}.usdz")
    txt = z.read("rig_trajectories.usda").decode()
    off = int(re.search(r"absoluteTimeOffsetMicroSec = (\d+)", txt).group(1))
    end_tc = float(re.search(r"endTimeCode = ([\d.]+)", txt).group(1))
    blk = txt.split('def Xform "sensor_rig_0"')[1]
    tc, pos = [], []
    for t, body in re.findall(r"^\s*([\d.]+):\s*\(\s*(.*?)\s*\)\s*,?\s*$", blk, re.MULTILINE):
        n = [float(x) for x in re.findall(r"-?\d+\.?\d*(?:e-?\d+)?", body)]
        if len(n) == 16:
            tc.append(float(t))
            pos.append(n[12:14])
    di = json.loads(z.read("data_info.json"))["pose-range"]
    fps = end_tc / ((di["end-timestamp_us"] - di["start-timestamp_us"]) / 1e6)
    xy = np.array(pos)                                                            # (N, 2)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])   # (N,)
    return off + np.array(tc) / fps * 1e6, s


def runs(mask):
    """Maximal True runs of an interval mask (T-1,) as (a, b): intervals a..b-1, time t[a]..t[b]."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def anatomy(path, gt):
    """One rollout's metrics.parquet -> (rollout fields, one record per place the human stood still)."""
    df = pd.read_parquet(path, columns=["name", "timestamps_us", "values"])
    df = df[df["name"].isin(NEED)]
    w = df.pivot_table(index="timestamps_us", columns="name", values="values",
                       aggfunc="first").sort_index().ffill()
    for c in NEED:
        if c not in w.columns:
            w[c] = 0.0
    w = w.fillna(0.0)
    ev = w["eval_relevant"].to_numpy() > 0                      # (T0,)
    if not ev.any():
        return None
    w = w.iloc[int(np.argmax(ev)):]                             # RemoveTimestepsBeforeEvent
    ts = w.index.to_numpy(float)                                # (T,) absolute microseconds
    t = ts / 1e6                                                # (T,) seconds
    off = w["offroad"].to_numpy() > 0                           # (T,)
    caf = (w["collision_front"].to_numpy() > 0) | (w["collision_lateral"].to_numpy() > 0)  # (T,)
    rear = w["collision_rear"].to_numpy() > 0                   # (T,)
    any_gate = (w["collision_any"].to_numpy() > 0) | off        # (T,) offroad_or_collision
    dev = w["dist_to_gt_trajectory"].to_numpy()                 # (T,)
    prog = w["progress_rel_to_total"].to_numpy()                # (T,)
    dist = w["dist_traveled_m"].to_numpy()                      # (T,)
    gtd = float(w["gt_dist_traveled_m"].to_numpy()[-1])
    n = len(t)

    def pscore(i):
        if gtd < MIN_GT:
            return 1.0
        return float(min(min(max(prog[i], 0.0), 1.0) / SAT, 1.0))

    i_end = n - 1
    g = np.flatnonzero(any_gate)
    i_gate = int(g[0]) if len(g) else None
    i_g = i_gate if i_gate is not None else i_end               # last kept index, gate rule only
    c = np.flatnonzero(dev[:i_g + 1] >= MAX_DEV)
    i_4 = int(c[0]) if len(c) else None
    i_ship = i_4 if i_4 is not None else i_g                    # last kept index, shipped chain
    k_re = np.flatnonzero(prog >= ROUTE_END)
    i_re = int(k_re[0]) if len(k_re) else None                  # recorded route finished
    i_dev = min(i_g, i_re) if i_re is not None else i_g         # 4 m rule off, but stop at the route end

    p_end, p_gate, p_ship = pscore(i_end), pscore(i_g), pscore(i_ship)
    hit_ship = bool(caf[:i_ship + 1].any() or off[:i_ship + 1].any())
    hit_g = bool(caf[:i_g + 1].any() or off[:i_g + 1].any())
    hit_dev = bool(caf[:i_dev + 1].any() or off[:i_dev + 1].any())
    score = 0.0 if hit_ship else p_ship
    loss = 1.0 - score

    if i_4 is not None:
        end = "cut4m"
    elif i_gate is not None:
        end = "atfault" if caf[i_gate] else ("offroad" if off[i_gate] else "rear")
    else:
        end = "full"
    why = "gate" if hit_ship else {"cut4m": "cut", "rear": "rear", "full": "full"}[end]
    cut = ""
    if i_4 is not None:
        if prog[i_4] >= ROUTE_END:
            cut = "route_end"                                   # ran past the end of the recording
        elif p_ship >= 1.0:
            cut = "saturated"                                   # left the path with 80% already done
        else:
            later = (caf[i_4 + 1:] | off[i_4 + 1:]).any()
            back = (dev[i_4 + 1:] < MAX_DEV).any()
            more = float(dist[-1] - dist[i_4])
            cut = ("hidden_failure" if later else "excursion" if back
                   else "alt_route" if more >= ALT_M else "stalled")

    # ---- the human as the yardstick (post hoc) ---------------------------------------------
    s_gt = np.interp(ts, gt[0], gt[1])                          # (T,) human's arc length, same clock
    s_e = prog * gtd                                            # (T,) ego position along the route
    if i_re is not None:
        s_e[i_re:] = gtd + dist[i_re:] - dist[i_re]             # odometry past the end of the route
    lead = s_e - s_gt                                           # (T,) metres ahead of the human
    dt = np.diff(t)                                             # (T-1,)
    v = np.diff(dist) / np.maximum(dt, 1e-6)                    # (T-1,) ego speed
    vg = np.diff(s_gt) / np.maximum(dt, 1e-6)                   # (T-1,) human speed
    win = np.arange(n - 1) < i_ship                             # (T-1,) interval inside the scored window
    dur, dur_w = float(t[-1] - t[0]), float(t[i_ship] - t[0])
    e_stops = [(a, b) for a, b in runs((v < STOP_V) & win) if t[b] - t[a] >= MIN_STOP_S]
    g_stops = [(a, b) for a, b in runs(vg < STOP_V) if t[b] - t[a] >= MIN_STOP_S]
    sg_mid = 0.5 * (s_gt[1:] + s_gt[:-1])                       # (T-1,) where the human was, per interval
    extra = []                                                  # ego stops where the human kept moving
    for a, b in e_stops:
        band = np.abs(sg_mid - s_e[a]) <= AT_M
        if band.any() and vg[band].min() >= SLOW_V:
            extra.append(float(t[b] - t[a]))

    # Places where the human stood still. The car is followed past the end of the route (odometry
    # is exact there) but never past a mid-route 4 m cut, where its route position is a guess.
    i_w = i_g if (i_4 is None or prog[i_4] >= ROUTE_END) else i_ship
    at_all = [(a, b) for a, b in runs((v < STOP_V) & (np.arange(n - 1) < i_w)) if t[b] - t[a] >= MIN_STOP_S]
    stops = []
    for a, b in g_stops:
        sp, ta, tb = float(s_gt[a]), float(t[a]), float(t[b])
        pos = s_e[:i_w + 1]                                     # (i_w+1,) route position while the car is followed
        ent, at, out = (np.flatnonzero(pos >= sp + d) for d in (-BAND_IN, 0.0, BAND_OUT))
        i0 = int(ent[0]) if len(ent) else None                  # enters the place
        i1 = max(int(out[0]) if len(out) else i_w, (i0 or 0) + 1)   # leaves it, or the window ends inside
        stops.append({
            "s_pos": sp, "t_a": ta - t[0], "t_b": tb - t[0], "dur": tb - ta, "open_end": bool(b >= n - 3),
            "entered": i0 is not None, "passed": bool(len(out)),
            "stood": bool(i0 is not None and any(x < i1 and y > i0 for x, y in at_all)),
            "stood_on": bool(len(out) and any(x >= out[0] and s_e[x] <= sp + 30.0 for x, _ in at_all)),
            "vmin": float(v[i0:i1].min()) if i0 is not None and i0 < n - 1 else np.nan,
            "crash": bool(i0 is not None and not len(out) and i_gate is not None and i_w == i_gate and caf[i_gate]),
            "cross_rel": float(t[at[0]] - ta) if len(at) else np.nan,     # < 0: the car got there first
            "cross_left": float(tb - t[at[0]]) if len(at) else np.nan,    # how long the human still stood
            "pass_left": float(tb - t[out[0]]) if len(out) else np.nan,
        })

    def lead_at(sec):
        k = np.flatnonzero(t >= t[0] + sec)
        return float(lead[k[0]]) if len(k) and k[0] <= i_ship else np.nan

    gate_kind = "none" if i_gate is None else ("atfault" if caf[i_gate] else "offroad" if off[i_gate] else "rear")
    d1 = np.flatnonzero(np.array([pscore(i) for i in range(n)]) >= 1.0)
    last = slice(max(i_ship - 10, 0), max(i_ship, 1))           # the last second of the scored window
    sp_all = np.array([s_gt[a] for a, _ in g_stops])            # (n_human_stops,) where the human stood
    roll = {
        "score": score, "p_end": p_end, "p_gate": p_gate, "p_ship": p_ship,
        "slow": 1.0 - p_end, "frozen": p_end - p_gate, "cut4m": p_gate - p_ship, "zeroed": p_ship - score,
        "end": end, "cut": cut, "why": why,
        **{f"L_{x}": loss if why == x else 0.0 for x in WHY},
        "score_no4m": 0.0 if hit_g else p_gate,                 # the modifier literally switched off
        "score_dev": 0.0 if hit_dev else pscore(i_dev),         # ... but scoring still ends with the route
        "frac_scored": dur_w / max(dur, 1e-6),
        "km_ship": float(dist[i_ship] - dist[0]) / 1000.0, "km_dev": float(dist[i_dev] - dist[0]) / 1000.0,
        "km_all": float(dist[-1] - dist[0]) / 1000.0,
        "off_ship": float(off[:i_ship + 1].any()), "caf_ship": float(caf[:i_ship + 1].any()),
        "gate_dev": float(hit_dev), "off_all": float(off.any()), "caf_all": float(caf.any()),
        "rear_all": float(rear.any()), "gate_kind": gate_kind,
        "lead_at_gate": float(lead[i_gate]) if i_gate is not None else np.nan,
        "stop_frac": float(dt[v < STOP_V].sum() / max(dur, 1e-6)),      # whole 20 s (pre-registered G3)
        "v_mean": float((dist[-1] - dist[0]) / max(dur, 1e-6)),
        "stop_s": float(dt[win & (v < STOP_V)].sum()), "gt_stop_s": float(dt[win & (vg < STOP_V)].sum()),
        "dur_w": dur_w, "ego_d": float(dist[i_ship] - dist[0]), "gt_d": float(s_gt[i_ship] - s_gt[0]),
        "lead_end": float(lead[i_ship]), "lead_mean": float(lead[:i_ship + 1].mean()),
        "lead_0": float(lead[0]), "lead_2": lead_at(2.0), "lead_5": lead_at(5.0), "lead_10": lead_at(10.0),
        "v0_ratio": float(v[0] / vg[0]) if vg[0] > 2.0 else np.nan,     # speed in the first scored 0.1 s
        "behind": float((lead[:i_ship + 1] < -LEAD_M).mean()), "ahead": float((lead[:i_ship + 1] > LEAD_M).mean()),
        "n_extra": float(len(extra)), "extra_s": float(sum(extra)), "has_extra": float(len(extra) > 0),
        "n_stood": float(len(e_stops)), "stood_s": float(sum(t[b] - t[a] for a, b in e_stops)),
        "gt_stood": float(len(g_stops) > 0), "v_end": float(v[last].mean()), "vg_end": float(vg[last].mean()),
        "near_stop_end": float(len(sp_all) > 0 and np.abs(sp_all - s_e[i_ship]).min() <= 30.0),
        "gt_dist": gtd, "reached_sat": float(len(d1) > 0),
        "t_to_sat": float(t[d1[0]] - t[0]) if len(d1) else np.nan,
    }
    return roll, stops


def build(rebuild, out):
    cache, cache_s = out / "rollouts.parquet", out / "stop_events.parquet"
    if cache.exists() and cache_s.exists() and not rebuild:
        return pd.read_parquet(cache), pd.read_parquet(cache_s)
    uuid = pd.read_csv(SCENES).set_index("scene_id")["uuid"]
    tracks, rows, stops = {}, [], []
    for label, prefix, n in SETS:
        for key, cfg, _ in ARMS:
            d = run_dir(prefix, cfg, n)
            if d is None:
                print(f"  {label:10s} {key:9s} no run", flush=True)
                continue
            j = json.loads((d / "aggregate/results-summary.json").read_text())
            n_ok = n_ab = 0
            for r in j["rollouts"]:
                sc, rid = r["clipgt_id"], r["rollout_id"]
                base = {"set": label, "arm": key, "scene": sc, "rollout": rid, "shipped": r.get("score"),
                        "failure_reason": str(r.get("failure_reason"))}
                p = d / "rollouts" / sc / rid / "metrics.parquet"
                a = None
                if p.exists():
                    if sc not in tracks:
                        tracks[sc] = gt_track(uuid[sc])
                    a = anatomy(p, tracks[sc])
                if a is None:
                    n_ab += 1
                    rows.append({**base, "end": "aborted", "cut": "", "why": ""})
                    continue
                n_ok += 1
                rows.append({**base, **a[0]})
                stops += [{"set": label, "arm": key, "scene": sc, "rollout": rid, **e} for e in a[1]]
            print(f"  {label:10s} {key:9s} {n_ok} rollouts" + (f"  (no parquet {n_ab})" if n_ab else ""),
                  flush=True)
    df, st = pd.DataFrame(rows), pd.DataFrame(stops)
    df.to_parquet(cache)
    st.to_parquet(cache_s)
    return df, st


def boot(x, rng, n=5000):
    """mean and percentile CI of a (N,) vector of scene-paired values (NaN scenes dropped)."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    m = x[rng.integers(0, len(x), size=(n, len(x)))].mean(1)    # (n,)
    return float(x.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))


def boot_fn(cols, fn, rng, n=5000):
    """fn of per-scene column sums with a scene-resampled CI; cols is (N, C), fn maps (C, ...) -> (...)."""
    cols = np.asarray(cols, float)
    s = cols[rng.integers(0, len(cols), size=(n, len(cols)))].sum(1)   # (n, C)
    b = fn(s.T)                                                         # (n,)
    return float(fn(cols.sum(0))), float(np.quantile(b, .025)), float(np.quantile(b, .975))


def boot_ratio(num, den, rng):
    """sum(num) / sum(den) over scenes; num, den are (N,) per scene."""
    return boot_fn(np.c_[num, den], lambda s: s[0] / np.maximum(s[1], 1e-9), rng)


def fmt(c, d=4):
    return f"{c[0]:+.{d}f} [{c[1]:+.{d}f},{c[2]:+.{d}f}]" + ("*" if c[1] > 0 or c[2] < 0 else " ")


def pace_table(dd, arms, ref, say):
    """Rear-ended and ran-short rollouts, what they cost, and route completion; own generator, so adding the
    table leaves every number printed after it unchanged."""
    rb, out, per = np.random.default_rng(7), {}, {}
    key = dd["set"] + "|" + dd["scene"] if "set" in dd else dd["scene"]
    say("\n[2c] post hoc - the two endings that only being slower than the recording produces, and route completion")
    say(f"  {'arm':15s}    n | rear-ended    %  score lost then | ran all 20 s, short of 80%    %  score lost then"
        f" | both, mean loss | drove past the route end  full marks")
    for k in arms:
        m = dd["arm"] == k
        a = dd[m]
        rear, short = a["end"] == "rear", (a["end"] == "full") & (a["score"] < 1.0)
        per[k] = (a["L_rear"] + a["L_full"]).groupby(key[m]).mean()
        out[k] = {"n": len(a), "rear_n": int(rear.sum()), "short_n": int(short.sum()),
                  "rear_loss_then": float(a.loc[rear, "L_rear"].mean()) if rear.any() else 0.0,
                  "short_loss_then": float(a.loc[short, "L_full"].mean()) if short.any() else 0.0,
                  "loss": float((a["L_rear"] + a["L_full"]).mean()),
                  "route_end": float((a["cut"] == "route_end").mean()), "full_marks": float((a["score"] >= 1.0).mean())}
        o = out[k]
        say(f"  {LABEL[k]:15s} {o['n']:4d} |    {o['rear_n']:3d}    {100 * o['rear_n'] / o['n']:5.1f}       {o['rear_loss_then']:.3f}"
            f"      |          {o['short_n']:3d}           {100 * o['short_n'] / o['n']:5.1f}       {o['short_loss_then']:.3f}"
            f"      |      {o['loss']:.3f}      |         {100 * o['route_end']:5.1f}%          {100 * o['full_marks']:5.1f}%")
    say(f"  score gained over {LABEL[ref]} through the two together (scene-paired), and its share of the score difference:")
    sc = dd.groupby([key, dd["arm"]])["score"].mean().unstack()
    for k in arms:
        if k == ref:
            continue
        j = per[k].index.intersection(per[ref].index)
        out[k]["vs_ref"] = boot(-(per[k][j] - per[ref][j]).to_numpy(), rb)
        gap = float((sc[k] - sc[ref]).mean())
        out[k]["share_of_gap"] = out[k]["vs_ref"][0] / gap if gap else float("nan")
        say(f"    {LABEL[k]:15s} {fmt(out[k]['vs_ref'])}   score difference {gap:+.4f}  ->  {100 * out[k]['share_of_gap']:+.0f}%")
    return out


def stop_class(e, slow_v=SLOW_V):
    """What the car did at a place where the human stood still; one row per (stop x rollout)."""
    left = e["passed"] & (e["pass_left"] >= EARLY_S)             # gone at least 1 s before the human was
    went = np.where(e["stood_on"], "stopped_on", np.where(e["vmin"] >= slow_v, "drove_through", "crawled"))
    return np.where(~e["entered"], "not_reached",
                    np.where(e["stood"], np.where(left, "left_early", "stopped"),
                             np.where(e["passed"], went, np.where(e["crash"], "crashed", "unresolved"))))


def stop_when(e):
    """When the car crossed the point, against the interval the human stood there."""
    return np.where(e["cross_rel"].isna(), "", np.where(e["cross_rel"] < -AHEAD_S, "before",
                                                       np.where(e["cross_left"] < EARLY_S, "after", "during")))


def stop_counts(a):
    """stop events -> per-scene counts: crossed the place, drove through, drove through while the human stood."""
    d = a["cls"] == "drove_through"
    x = pd.DataFrame({"scene": a["scene"], "crossed": a["cls"].isin(CROSSED).astype(float), "thr": d.astype(float),
                      "dur": (d & (a["when"] == "during")).astype(float)})
    return x.groupby("scene").sum()


def stops_table(se, arms, ref, rng, say):
    """Class shares at the human's stops, the drove-through rate, its paired difference and sensitivity."""
    def rate(s):
        return s[0] / np.maximum(s[1], 1e-9)

    def rate_diff(s):
        return s[0] / np.maximum(s[1], 1e-9) - s[2] / np.maximum(s[3], 1e-9)

    out, cr = {}, stop_counts(se[se["arm"] == ref])
    for k in arms:
        a = se[se["arm"] == k]
        c = a["cls"].value_counts(normalize=True)
        w = a.loc[a["cls"] == "drove_through", "when"].value_counts() / max(len(a), 1)
        sg = stop_counts(a)
        thr, dur = boot_fn(sg[["thr", "crossed"]], rate, rng), boot_fn(sg[["dur", "crossed"]], rate, rng)
        out[k] = {"n": len(a), "share": {x: float(c.get(x, 0.0)) for x in STOPS},
                  "when": {x: float(w.get(x, 0.0)) for x in ["before", "during", "after"]},
                  "n_crossed": int(sg["crossed"].sum()), "thr_of_crossed": thr, "during_of_crossed": dur, "sens": {}}
        say(f"  {LABEL[k]:15s} {len(a):4d}   {100 * c.get('stopped', 0):5.1f}              {100 * c.get('stopped_on', 0):5.1f}"
            f"       {100 * c.get('left_early', 0):5.1f}    {100 * c.get('crawled', 0):5.1f}   {100 * c.get('drove_through', 0):5.1f} ="
            f" {100 * w.get('before', 0):4.1f} + {100 * w.get('during', 0):4.1f} + {100 * w.get('after', 0):4.1f}"
            f"      {100 * c.get('crashed', 0):5.1f}       {100 * c.get('unresolved', 0):5.1f}        "
            f"{100 * c.get('not_reached', 0):5.1f} |  {int(sg['crossed'].sum()):3d}   {100 * thr[0]:5.1f}% [{100 * thr[1]:4.1f},"
            f"{100 * thr[2]:4.1f}]          {100 * dur[0]:5.1f}% [{100 * dur[1]:4.1f},{100 * dur[2]:4.1f}]")
        for name, sub in [("slow limit 1 m/s", a.assign(cls=stop_class(a, 1.0))),
                          ("slow limit 3 m/s", a.assign(cls=stop_class(a, 3.0))),
                          ("human went on again", a[~a["open_end"]]), ("human still standing", a[a["open_end"]])]:
            x = stop_counts(sub).sum() if len(sub) else {"thr": 0.0, "crossed": 0.0}
            out[k]["sens"][name] = [float(x["thr"] / max(x["crossed"], 1e-9)), int(x["crossed"])]
        if k != ref:
            v = sg.join(cr, rsuffix="_r", how="inner")
            out[k]["thr_vs_ref"] = boot_fn(v[["thr", "crossed", "thr_r", "crossed_r"]], rate_diff, rng)
            out[k]["during_vs_ref"] = boot_fn(v[["dur", "crossed", "dur_r", "crossed_r"]], rate_diff, rng)
    say(f"  drove-through rate, arm - {LABEL[ref]}:        " + "  ".join(
        f"{LABEL[k]} {fmt(out[k]['thr_vs_ref'], 3)}" for k in arms if k != ref))
    say(f"  ... while the human stood, arm - {LABEL[ref]}: " + "  ".join(
        f"{LABEL[k]} {fmt(out[k]['during_vs_ref'], 3)}" for k in arms if k != ref))
    for name in out[arms[0]]["sens"]:
        say(f"    drove-through rate, {name:21s}: " + "  ".join(
            f"{LABEL[k]} {100 * out[k]['sens'][name][0]:.1f}% of {out[k]['sens'][name][1]}" for k in arms))
    return out


def stopping(dd, ref, arm, rng, say):
    """Does the gap between two arms live where somebody stood still? Scene strata that add up to the gap."""
    a, r = dd[dd["arm"] == arm], dd[dd["arm"] == ref]
    rules = [f"L_{x}" for x in WHY]
    sa, sr = a.groupby("scene")[["score", *rules]].mean(), r.groupby("scene")[["score", *rules]].mean()
    d = (sa["score"] - sr["score"]).to_numpy()                     # (scenes,) paired gap
    idx = sa.index
    stood_a = (a.groupby("scene")["n_stood"].max() > 0).reindex(idx).to_numpy()   # (scenes,) in any rollout
    stood_r = (r.groupby("scene")["n_stood"].max() > 0).reindex(idx).to_numpy()
    human = (r.groupby("scene")["gt_stood"].max() > 0).reindex(idx).to_numpy()
    out = {"gap": boot(d, rng), "strata": {}}

    def line(name, m):
        inside = boot(d[m], rng) if m.sum() > 1 else (float("nan"),) * 3
        part = boot(d * m, rng)                                    # the parts of a partition add up to the gap
        out["strata"][name] = {"n": int(m.sum()), "gap": inside, "part": part}
        say(f"      {name:40s} {m.sum():3d} scenes   gap inside {fmt(inside)}   adds {fmt(part)}"
            f" = {100 * part[0] / out['gap'][0]:3.0f}% of the gap")

    say(f"\n[9] post hoc - is it the stopping? {LABEL[arm]} - {LABEL[ref]} = {fmt(out['gap'])}, split over scenes")
    say("    by what the human did in the clip:")
    line("the human stood still (>= 1 s) somewhere", human)
    line("the human never stood", ~human)
    say("    by what the two models did inside their scored windows (either rollout):")
    line("neither model stood still", ~stood_a & ~stood_r)
    line(f"{LABEL[ref]} stood, {LABEL[arm]} did not", stood_r & ~stood_a)
    line("both stood", stood_r & stood_a)
    line(f"{LABEL[arm]} stood, {LABEL[ref]} did not", stood_a & ~stood_r)
    m0 = ~stood_a & ~stood_r
    dl = -(sa[rules] - sr[rules]).to_numpy()                       # (scenes, 4) by ending rule
    out["no_stop_by_rule"] = {x: boot(dl[m0, i], rng) for i, x in enumerate(WHY)}
    say("    inside 'neither model stood still', by ending rule: " + "  ".join(
        f"{x} {fmt(out['no_stop_by_rule'][x])}" for x in WHY))

    out["pace"] = {}
    for k, x in [(ref, r), (arm, a)]:
        mov, gmov = (x["dur_w"] - x["stop_s"]).sum(), (x["dur_w"] - x["gt_stop_s"]).sum()
        out["pace"][k] = {"pace": float(x["ego_d"].sum() / x["gt_d"].sum()),
                          "speed": float((x["ego_d"].sum() / mov) / (x["gt_d"].sum() / gmov)), "time": float(mov / gmov)}
    pr, pa = out["pace"][ref], out["pace"][arm]
    share = np.log(pa["speed"] / pr["speed"]) / np.log(pa["pace"] / pr["pace"])
    out["pace"]["share_speed"] = float(share)
    say(f"    distance against the human = speed while moving x time spent moving: {LABEL[ref]} {pr['pace']:.3f} ="
        f" {pr['speed']:.3f} x {pr['time']:.3f}; {LABEL[arm]} {pa['pace']:.3f} = {pa['speed']:.3f} x {pa['time']:.3f}"
        f"  ->  {100 * share:.0f}% of the pace gap is speed while moving, {100 * (1 - share):.0f}% is standing less")

    out["rear"], out["short"] = {}, {}
    say("    rollouts whose window a rear-end collision closed, in the last second before it:")
    for k, x in [(LABEL[ref], r), (LABEL[arm], a), ("all eight arms", dd)]:
        e = x[x["end"] == "rear"]
        out["rear"][k] = {"n": len(e), "standing": float((e["v_end"] < STOP_V).mean()),
                          "under_2": float((e["v_end"] < SLOW_V).mean()), "human_standing": float((e["vg_end"] < STOP_V).mean()),
                          "near_human_stop": float(e["near_stop_end"].mean()), "v": float(e["v_end"].median()),
                          "vg": float(e["vg_end"].median()), "lead": float(e["lead_at_gate"].median())}
        o = out["rear"][k]
        say(f"      {k:15s} n={len(e):3d}: car standing {100 * o['standing']:3.0f}%, under 2 m/s {100 * o['under_2']:3.0f}%;"
            f" human standing {100 * o['human_standing']:3.0f}%; within 30 m of a place the human stood"
            f" {100 * o['near_human_stop']:3.0f}%; median speed {o['v']:.1f} vs the human's {o['vg']:.1f} m/s,"
            f" {o['lead']:+.1f} m from the human")
    say("    rollouts that ran all 20 s and still covered under 80%:")
    for k, x in [(LABEL[ref], r), (LABEL[arm], a), ("all eight arms", dd)]:
        e = x[(x["why"] == "full") & (x["score"] < 1.0)]
        mov, gmov = (e["dur_w"] - e["stop_s"]).sum(), (e["dur_w"] - e["gt_stop_s"]).sum()
        out["short"][k] = {"n": len(e), "stood": float((e["n_stood"] > 0).mean()) if len(e) else float("nan"),
                           "stop_s": float(e["stop_s"].mean()) if len(e) else float("nan"),
                           "gt_stop_s": float(e["gt_stop_s"].mean()) if len(e) else float("nan"),
                           "speed": float((e["ego_d"].sum() / mov) / (e["gt_d"].sum() / gmov)) if len(e) else float("nan")}
        o = out["short"][k]
        say(f"      {k:15s} n={len(e):3d}: stood still at some point {100 * o['stood']:3.0f}%; stopped {o['stop_s']:.1f} s"
            f" against the human's {o['gt_stop_s']:.1f} s; speed while moving {o['speed']:.2f}x the human's")
    return out


def plots(res, out):
    sets = [("origin150", "Origin150"), ("hard100", "Hard100")]
    arms = [k for k, _, _ in ARMS]
    rule = {"gate": "off-road / at-fault collision (score zeroed)", "rear": "rear-ended by replayed traffic",
            "cut": "left the recorded path by 4 m", "full": "ran all 20 s, covered < 80% of the route"}

    # 1. where the score is lost: part-to-whole, one bar per arm
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=True)
    y = np.arange(len(arms))[::-1]                                # (8,) first arm on top
    for ax, (key, title) in zip(axes, sets):
        A = res[key]["arms"]
        left = np.zeros(len(arms))                                # (8,)
        for x, c in zip(WHY, CAT):
            val = np.array([A[k][f"L_{x}"] for k in arms])        # (8,)
            ax.barh(y, val, left=left, height=0.56, color=c, edgecolor=BG, linewidth=1.6, label=rule[x])
            left += val
        for yi, k, tot in zip(y, arms, left):
            ax.text(tot + 0.008, yi, f"score {A[k]['score']:.3f}", va="center", fontsize=9)
        ax.set_yticks(y, [LABEL[k] for k in arms])
        ax.set_xlim(0, left.max() * 1.24)
        ax.set_title(f"{title}, {res[key]['n_scenes']} scenes", loc="left", fontsize=10.5)
        ax.set_xlabel("score lost, 1 - score (mean over rollouts)")
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
    h, lab = axes[0].get_legend_handles_labels()
    fig.suptitle("Where the closed-loop score is lost, by the one rule that ended each rollout's scored window",
                 x=0.012, ha="left", fontsize=12, fontweight="bold")
    fig.legend(h, lab, loc="upper left", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.008, 0.945))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "plots/loss_by_rule.png", dpi=150)
    plt.close(fig)

    # 2. MoOP - dense, split by rule, with scene-paired CIs
    rows = [("score", "score, MoOP - dense"), ("L_gate", "fewer rollouts zeroed by a gate"),
            ("L_rear", "rear-ended less often"), ("L_cut", "lost less to the 4 m rule"),
            ("L_full", "ran out of time less often")]
    ser = [("origin150", f"Origin150 ({res['origin150']['n_scenes']} scenes)", CAT[0], "o"),
           ("hard100", f"Hard100 ({res['hard100']['n_scenes']})", CAT[1], "s"),
           ("pooled", f"both ({res['pooled']['n_scenes']})", INK, "D")]
    fig, ax = plt.subplots(figsize=(9.8, 4.7))
    hi = max(res[key]["paired"]["dual"][c][2] for key, _, _, _ in ser for c, _ in rows)
    for i, (col, _) in enumerate(rows):
        for j, (key, name, c, m) in enumerate(ser):
            ci = res[key]["paired"]["dual"][col]
            yy = -i + (1 - j) * 0.23
            ax.plot(ci[1:], [yy, yy], color=c, linewidth=2, solid_capstyle="round")
            ax.plot(ci[0], yy, m, color=c, markersize=7.5, markeredgecolor=BG, markeredgewidth=1.5,
                    label=name if i == 0 else None)
        p = res["pooled"]["paired"]["dual"][col]
        ax.text(hi + 0.012, -i - 0.23, f"{p[0]:+.3f}  [{p[1]:+.3f}, {p[2]:+.3f}]", va="center", fontsize=9)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.axhline(-0.5, color=GRID, linewidth=0.8)
    ax.set_yticks(-np.arange(len(rows)), [r[1] for r in rows])
    ax.set_xlim(right=hi + 0.085)
    ax.set_xlabel("difference in score: scene-paired mean, 95% bootstrap CI (numbers: both sets)")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    h, lab = ax.get_legend_handles_labels()
    fig.suptitle("What MoOP gains over the dense model in closed loop; the four lower rows add up to the first",
                 x=0.012, ha="left", fontsize=11.5, fontweight="bold")
    fig.legend(h, lab, loc="upper left", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.008, 0.945))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out / "plots/moop_vs_dense_by_rule.png", dpi=150)
    plt.close(fig)

    # 3. position against the human driver, 5 s and 10 s into the scored window
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=True)
    for ax, (key, title) in zip(axes, sets):
        A = res[key]["arms"]
        for yi, k in zip(y, arms):
            a5, a10 = A[k]["lead_5"], A[k]["lead_10"]
            ax.plot([a5, a10], [yi, yi], color=ORD[1], linewidth=2, solid_capstyle="round", zorder=2)
            ax.plot(a5, yi, "o", color=ORD[0], markersize=8.5, markeredgecolor=BG, markeredgewidth=1.5, zorder=3,
                    label="5 s after hand-over" if k == arms[0] else None)
            ax.plot(a10, yi, "o", color=ORD[2], markersize=8.5, markeredgecolor=BG, markeredgewidth=1.5, zorder=3,
                    label="10 s" if k == arms[0] else None)
            ax.annotate(f"{a10:+.1f} m", (a10, yi), textcoords="offset points",
                        xytext=(9 if a10 >= a5 else -9, 0), ha="left" if a10 >= a5 else "right", va="center", fontsize=9)
        ax.axvline(0, color=MUTED, linewidth=1)
        ax.set_yticks(y, [LABEL[k] for k in arms])
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo - 1.6, hi + 1.6)
        ax.set_title(f"{title}", loc="left", fontsize=10.5)
        ax.set_xlabel("metres ahead of the human driver (negative = behind)")
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
    h, lab = axes[0].get_legend_handles_labels()
    fig.suptitle("Position along the route against the human driver, inside the scored window",
                 x=0.012, ha="left", fontsize=12, fontweight="bold")
    fig.legend(h, lab, loc="upper left", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.008, 0.945))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "plots/lead_vs_human.png", dpi=150)
    plt.close(fig)

    # 4. what each model did where the human stood still (Origin150 has 61 such places, Hard100 only 9)
    R0 = res["origin150"]
    grp = [("stopped", ["stopped", "stopped_on", "left_early"]), ("slowed below 2 m/s, never stood", ["crawled"]),
           ("drove through at 2 m/s or more", ["drove_through"])]
    for S, title, name in [
            (R0["stops"], f"At the {R0['stops_n']} places on Origin150 where the human driver stood still",
             "stops_where_human_stood"),
            (R0["stops_moving"], f"At the {R0['stops_moving_n']} of them that the car reaches while moving (Origin150)",
             "stops_arriving_moving")]:
        order = sorted(arms, key=lambda k: S[k]["thr_of_crossed"][0])
        yy = np.arange(len(order))[::-1]
        fig, ax = plt.subplots(figsize=(9.6, 4.3))
        left = np.zeros(len(order))
        for (lab, cls), c in zip(grp, ORD):
            val = np.array([sum(S[k]["share"][x] for x in cls) / sum(S[k]["share"][x] for x in CROSSED) for k in order])
            ax.barh(yy, 100 * val, left=left, height=0.56, color=c, edgecolor=BG, linewidth=1.6, label=lab)
            left += 100 * val
        for yi, k in zip(yy, order):
            t = S[k]["thr_of_crossed"]
            ax.text(101.5, yi, f"{100 * t[0]:.0f}%  [{100 * t[1]:.0f}, {100 * t[2]:.0f}]   n={S[k]['n_crossed']}",
                    va="center", fontsize=9)
        ax.set_yticks(yy, [LABEL[k] for k in order])
        ax.set_xlim(0, 131)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("% of crossings (a stop x a rollout in which the car reached the place); right: drove through,"
                      " 95% CI")
        ax.tick_params(length=0)
        h, lab = ax.get_legend_handles_labels()
        fig.suptitle(title, x=0.012, ha="left", fontsize=12, fontweight="bold")
        fig.legend(h, lab, loc="upper left", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.008, 0.945))
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        fig.savefig(out / f"plots/{name}.png", dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="closedloop_gap_v1")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--reference", default="baseline")
    args = ap.parse_args()
    out = O / args.exp_id
    (out / "plots").mkdir(parents=True, exist_ok=True)
    df, st = build(args.rebuild, out)
    st["cls"], st["when"] = stop_class(st), stop_when(st)
    rng = np.random.default_rng(0)
    res, lines, SC, SE, DD = {}, [], {}, {}, {}
    ref = args.reference
    foc = "dual" if ref != "dual" else "baseline"               # the arm the stopping section is written for
    tag = "" if ref == "baseline" else f"_ref-{ref}"            # arm-vs-arm reruns sit beside the main one

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    # ---- G0: does the reimplementation reproduce the shipped scores? ----------------------
    ok = df[df["end"] != "aborted"]
    diff = (ok["score"] - ok["shipped"].astype(float)).abs()
    neg = ok[TERMS].to_numpy()                                   # (rollouts, 4)
    res["G0"] = {"n": len(ok), "max_abs_diff": float(diff.max()),
                 "n_mismatch": int((diff > 1e-6).sum()), "n_aborted": int((df["end"] == "aborted").sum()),
                 "pass": bool(diff.max() < 1e-6), "negative_terms": int((neg < -1e-9).sum()),
                 "negative_mass": float(neg[neg < -1e-9].sum()), "loss_mass": float((1 - ok["score"]).sum())}
    say(f"G0 reproduction: {len(ok)} rollouts with a parquet, max |recomputed - shipped| = "
        f"{diff.max():.2e}, mismatches {int((diff > 1e-6).sum())}, without parquet "
        f"{int((df['end'] == 'aborted').sum())}  ->  {'PASS' if res['G0']['pass'] else 'FAIL'}")
    say(f"   the four terms sum to 1 - score: max error "
        f"{float(np.abs(neg.sum(1) - (1 - ok['score'].to_numpy())).max()):.1e}; negative terms "
        f"{res['G0']['negative_terms']} carrying {res['G0']['negative_mass']:+.2f} of a total loss of "
        f"{res['G0']['loss_mass']:.1f} (progress is not monotone once the car has left the path)")

    for label, _, _ in SETS:
        d = df[df["set"] == label]
        bad = set(d.loc[d["end"] == "aborted", "scene"])            # an aborted rollout in ANY arm
        dd = d[(d["end"] != "aborted") & ~d["scene"].isin(bad)].copy()
        arms = [k for k, _, _ in ARMS if k in set(dd["arm"])]
        R = res[label] = {"n_scenes": int(dd["scene"].nunique()), "dropped_scenes": len(bad)}
        say(f"\n================ {label}: {dd['scene'].nunique()} scenes"
            + (f" (the {len(bad)} with an aborted rollout in some arm are dropped; the paper's means count"
               f" those rollouts as 0)" if bad else ""))
        dd["alt"] = (dd["cut"] == "alt_route").astype(float)
        dd["cut_cost"] = dd["cut"].isin(["alt_route", "excursion", "hidden_failure", "stalled"]).astype(float)
        dd["gate_ship"] = ((dd["off_ship"] > 0) | (dd["caf_ship"] > 0)).astype(float)
        dd["rear_end"] = (dd["end"] == "rear").astype(float)
        dd["stop_w"] = dd["stop_s"] / dd["dur_w"].clip(lower=1e-6)
        dd["stop_excess_s"] = dd["stop_s"] - dd["gt_stop_s"]

        # ---- [1] per-arm means, pre-registered split -------------------------------------
        say("\n[1] where each arm's score is lost (rollout means; the four terms sum to 1 - score)")
        say(f"  {'arm':15s} paper  score   slow  frozen  cut4m zeroed | 4 m rule off: literal  route-end-aware"
            f" | scored time")
        R["arms"] = {}
        for k in arms:
            a = dd[dd["arm"] == k]
            A = R["arms"][k] = {c: float(a[c].mean()) for c in
                                ["score", "score_no4m", "score_dev", "frac_scored", "stop_frac", "v_mean",
                                 "stop_w", "stop_excess_s", "lead_end", "lead_mean", "lead_0", "lead_2", "lead_5",
                                 "lead_10", "v0_ratio", "behind", "ahead", "has_extra", "extra_s", "reached_sat"]
                                + TERMS + [f"L_{x}" for x in WHY]}
            A["paper"] = float(d[d["arm"] == k]["shipped"].astype(float).mean())
            say(f"  {LABEL[k]:15s} {A['paper']:.3f}  {A['score']:.3f}  {A['slow']:.3f}  {A['frozen']:.3f}"
                f"  {A['cut4m']:.3f}  {A['zeroed']:.3f} |               {A['score_no4m']:.3f}"
                f"       {A['score_dev']:.3f}      |   {100 * A['frac_scored']:.1f}%")

        # ---- [2] how the scored window ended ------------------------------------------------
        say("\n[2] how the scored window ended (% of rollouts)")
        say(f"  {'arm':15s} full 20 s  off-road  at-fault  rear-ended | 4 m rule: route end  >=80% done"
            f"  alt-route  excursion  hidden-fail  stalled")
        for k in arms:
            a = dd[dd["arm"] == k]
            e, c = a["end"].value_counts(normalize=True), a["cut"].value_counts(normalize=True)
            R["arms"][k]["end"] = {x: float(e.get(x, 0.0)) for x in ["full", "offroad", "atfault", "rear", "cut4m"]}
            R["arms"][k]["cut"] = {x: float(c.get(x, 0.0)) for x in CUTS}
            say(f"  {LABEL[k]:15s}   {100 * e.get('full', 0):5.1f}     {100 * e.get('offroad', 0):5.1f}"
                f"     {100 * e.get('atfault', 0):5.1f}       {100 * e.get('rear', 0):5.1f} |"
                + "".join(f"     {100 * c.get(x, 0):6.1f}" for x in CUTS))

        # ---- [2b] post hoc: loss by the rule that ended the window ----------------------------
        say("\n[2b] post hoc - the same loss, each rollout given to the one rule that ended its window")
        say(f"  {'arm':15s} 1-score = gate (zeroed)  + rear-ended  + 4 m rule  + ran 20 s short of 80%")
        for k in arms:
            A = R["arms"][k]
            say(f"  {LABEL[k]:15s}  {1 - A['score']:.3f}  =     {A['L_gate']:.3f}           {A['L_rear']:.3f}"
                f"        {A['L_cut']:.3f}           {A['L_full']:.3f}")

        # ---- [2c] post hoc (2026-10-12): the two endings that only slowness produces, in rollouts ----
        R["pace"] = pace_table(dd, arms, ref, say)

        # ---- [3] scene-paired decomposition against the reference ---------------------------
        cols = (["score", "score_no4m", "score_dev", "alt", "cut_cost", "stop_frac", "v_mean", "gate_ship",
                 "gate_dev", "rear_end", "frac_scored", "stop_w", "stop_excess_s", "lead_end", "lead_mean",
                 "lead_2", "lead_5", "lead_10", "behind", "ahead", "has_extra", "extra_s", "reached_sat"]
                + TERMS + [f"L_{x}" for x in WHY])
        sc = SC[label] = dd.groupby(["scene", "arm"])[cols].mean().unstack("arm")   # (scenes, cols x arms)
        say(f"\n[3] score difference vs {LABEL[ref]} (scene-paired, 95% bootstrap CI); every term is"
            f" -(loss_arm - loss_ref), so each row of terms adds up to the score difference")
        R["paired"] = {}
        for k in arms:
            if k == ref:
                continue
            row = {"n": len(sc)}
            for x in cols:
                sign = -1.0 if (x in TERMS or x.startswith("L_")) else 1.0
                row[x] = boot(sign * (sc[x][k] - sc[x][ref]), rng)
            R["paired"][k] = row
            say(f"  {LABEL[k]:15s} score {fmt(row['score'])}")
            say("  " + " " * 15 + " pre-registered: " + "  ".join(f"{tm} {fmt(row[tm])}" for tm in TERMS))
            say("  " + " " * 15 + " by ending rule: " + "  ".join(f"{x} {fmt(row['L_' + x])}" for x in WHY))
            say("  " + " " * 15 + f" 4 m rule off: literal {fmt(row['score_no4m'])}   route-end-aware"
                f" {fmt(row['score_dev'])}")

        # ---- [4] behaviour, with the human as the yardstick ---------------------------------
        say("\n[4] post hoc - pace and stopping inside the scored window, against the human driver in the"
            " same seconds")
        say(f"  {'arm':15s} distance, ego/human   speed while moving, ego/human   stopped: ego  human |"
            f" m ahead of the human at 2 s    5 s   10 s   window end | time >5 m behind  ahead |"
            f" stops the human did not make: rollouts  s/rollout")
        R["pace"] = {}
        for k in arms:
            a = dd[dd["arm"] == k]
            sg = a.groupby("scene")[["ego_d", "gt_d", "stop_s", "gt_stop_s", "dur_w"]].sum()   # (scenes, 5)
            pace = boot_ratio(sg["ego_d"], sg["gt_d"], rng)
            mov = boot_fn(sg, lambda s: (s[0] / (s[4] - s[2])) / (s[1] / (s[4] - s[3])), rng)
            es, hs = boot_ratio(sg["stop_s"], sg["dur_w"], rng), boot_ratio(sg["gt_stop_s"], sg["dur_w"], rng)
            A = R["arms"][k]
            R["pace"][k] = {"pace": pace, "moving_speed": mov, "ego_stop": es, "human_stop": hs}
            say(f"  {LABEL[k]:15s} {pace[0]:.3f} [{pace[1]:.3f},{pace[2]:.3f}]      {mov[0]:.3f} [{mov[1]:.3f},{mov[2]:.3f}]"
                f"            {100 * es[0]:5.1f}% {100 * hs[0]:5.1f}% |            {A['lead_2']:+6.2f} {A['lead_5']:+6.2f}"
                f" {A['lead_10']:+6.2f}     {A['lead_end']:+6.2f}   |      {100 * A['behind']:5.1f}%   {100 * A['ahead']:5.1f}% |"
                f"                     {100 * A['has_extra']:5.1f}%     {A['extra_s']:.2f}")
        say(f"  at the first scored timestep, every arm alike: {dd['lead_0'].mean():+.2f} m from the human"
            f" (sd {dd['lead_0'].std():.2f}), speed {dd['v0_ratio'].mean():.3f}x the human's"
            f" (by arm {dd.groupby('arm')['v0_ratio'].mean().min():.3f}-{dd.groupby('arm')['v0_ratio'].mean().max():.3f})")
        say(f"  scene-paired difference, arm - {LABEL[ref]}:")
        say(f"  {'arm':15s} whole-20 s stop share (G3)   m ahead at 2 s        at 5 s                at 10 s"
            f"               at window end          time >5 m behind            extra-stop rollouts")
        for k in arms:
            if k == ref:
                continue
            r = R["paired"][k]
            say(f"  {LABEL[k]:15s} {fmt(r['stop_frac'])}  {fmt(r['lead_2'], 2)}  {fmt(r['lead_5'], 2)}  "
                f"{fmt(r['lead_10'], 2)}  {fmt(r['lead_end'], 2)}  {fmt(r['behind'])}  {fmt(r['has_extra'])}")

        # ---- [5] what the model did where the human stood still -----------------------------
        se = SE[label] = st[(st["set"] == label) & ~st["scene"].isin(bad)]
        ev0 = se[se["arm"] == ref].drop_duplicates(["scene", "s_pos"])
        say(f"\n[5] post hoc - places where the human stood still for >= {MIN_STOP_S:.0f} s: {len(ev0)} stops in "
            f"{ev0['scene'].nunique()} of {dd['scene'].nunique()} scenes ({int(ev0['open_end'].sum())} still standing"
            f" when the clip ends); % of (stop x rollout)")
        say(STOP_HEAD)
        R["stops"] = stops_table(se, arms, ref, rng, say)
        # its own generator, so that adding this table leaves every number printed after it unchanged
        mv = se[se["t_a"] >= START_S]
        R["stops_n"], R["stops_moving_n"] = len(ev0), int((ev0["t_a"] >= START_S).sum())
        say(f"\n[5b] post hoc - the same, only where the car arrives moving: {R['stops_moving_n']} of the {len(ev0)}"
            f" stops begin >= {START_S:.0f} s into the scored clip; the other {len(ev0) - R['stops_moving_n']} are under"
            f" way at the hand-over, where the car starts standing and cannot drive through")
        say(STOP_HEAD)
        R["stops_moving"] = stops_table(mv, arms, ref, np.random.default_rng(5), say)

        # ---- [6] progress-free view ----------------------------------------------------------
        say("\n[6] progress-free view: gate events (off-road or at-fault collision)")
        say(f"  {'arm':15s} scored window: rollouts  per km [95% CI]          | 4 m rule off, up to the route"
            f" end: rollouts  per km [95% CI]   | rear-ended (ends window, not a gate)")
        R["gates"] = {}
        for k in arms:
            a = dd[dd["arm"] == k]
            sg = a.groupby("scene").agg(ev_s=("gate_ship", "sum"), km_s=("km_ship", "sum"),
                                        ev_d=("gate_dev", "sum"), km_d=("km_dev", "sum"))
            ks, kd = boot_ratio(sg["ev_s"], sg["km_s"], rng), boot_ratio(sg["ev_d"], sg["km_d"], rng)
            R["gates"][k] = {"share_ship": float(a["gate_ship"].mean()), "share_dev": float(a["gate_dev"].mean()),
                             "per_km_ship": ks, "per_km_dev": kd, "km_ship": float(sg["km_s"].sum()),
                             "km_dev": float(sg["km_d"].sum()), "rear": float(a["rear_end"].mean())}
            say(f"  {LABEL[k]:15s}        {100 * a['gate_ship'].mean():5.1f}%     {ks[0]:.2f} [{ks[1]:.2f},{ks[2]:.2f}]"
                f" ({sg['km_s'].sum():.1f} km) |                       {100 * a['gate_dev'].mean():5.1f}%     "
                f"{kd[0]:.2f} [{kd[1]:.2f},{kd[2]:.2f}] ({sg['km_d'].sum():.1f} km) |   "
                f"{100 * a['rear_end'].mean():5.1f}%")
        gk = dd[dd["gate_kind"] != "none"].groupby("gate_kind")["lead_at_gate"]
        R["lead_at_gate"] = {x: {"n": len(v), "median": float(v.median()), "behind": float((v < -LEAD_M).mean()),
                                 "ahead": float((v > LEAD_M).mean())} for x, v in gk}
        say("  where the car was relative to the human when the event happened (all arms pooled): " + "; ".join(
            f"{x} n={v['n']} median {v['median']:+.1f} m, >5 m behind {100 * v['behind']:.0f}%, >5 m ahead"
            f" {100 * v['ahead']:.0f}%" for x, v in R["lead_at_gate"].items()))

        # ---- [7] A/A noise floor -------------------------------------------------------------
        say("\n[7] A/A noise floor: rollout 1 vs rollout 2 of the same arm on the same scene")
        say(f"  {'arm':15s} mean|d|   sd(d)   95% half-width of a 2-vs-2 scene-paired difference under the null")
        R["aa"] = {}
        for k in arms:
            a = dd[dd["arm"] == k].sort_values(["scene", "rollout"])
            pr = a.groupby("scene")["score"].apply(list)
            pr = np.array([x[:2] for x in pr if len(x) >= 2])                           # (scenes, 2)
            dlt = pr[:, 0] - pr[:, 1]                                                    # (scenes,)
            hw = 1.96 * dlt.std(ddof=1) / np.sqrt(2 * len(dlt))
            R["aa"][k] = {"n": len(dlt), "mean_abs": float(np.abs(dlt).mean()),
                          "sd": float(dlt.std(ddof=1)), "half_width_2v2": float(hw)}
            say(f"  {LABEL[k]:15s}  {np.abs(dlt).mean():.3f}    {dlt.std(ddof=1):.3f}    +-{hw:.3f}")

        # ---- [9] post hoc: does the stopping carry the gap? ------------------------------------
        DD[label] = dd
        if foc in arms:
            R["stopping"] = stopping(dd, ref, foc, rng, say)

    # ---- dense vs dense across two independent runs (ours vs sangoh's 913-scene run) -------
    sj = SANGOH / "aggregate/results-summary.json"
    if sj.exists():
        s = pd.DataFrame([{"scene": r["clipgt_id"], "score": r.get("score")}
                          for r in json.loads(sj.read_text())["rollouts"]]).dropna()
        s = s.groupby("scene")["score"].mean()
        ours = df[(df["set"] == "origin150") & (df["arm"] == "baseline")].groupby("scene")["shipped"].mean()
        both = ours.index.intersection(s.index)
        c = boot((ours.loc[both] - s.loc[both]).to_numpy(float), rng)
        res["dense_vs_dense"] = {"n": len(both), "diff": c}
        say(f"\n[8] dense vs dense, two independent runs on the shared Origin150 scenes (ours 2 rollouts, "
            f"other run 1): n={len(both)}  {fmt(c)}")

    # ---- both sets pooled (post hoc; the two scene sets are disjoint) -----------------------
    sc = pd.concat(SC.values())                                   # (246 scenes, cols x arms)
    arms = [k for k, _, _ in ARMS if k in sc["score"].columns]
    P = res["pooled"] = {"n_scenes": len(sc), "paired": {}}
    say(f"\n================ post hoc - both sets pooled: {len(sc)} scenes, difference vs {LABEL[ref]}")
    say(f"  {'arm':15s} score                         = gate                       + rear-ended"
        f"                 + 4 m rule                   + ran 20 s short           | 4 m rule off, route-end-aware")
    for k in arms:
        if k == ref:
            continue
        row = P["paired"][k] = {x: boot((-1.0 if x.startswith("L_") else 1.0) * (sc[x][k] - sc[x][ref]), rng)
                                for x in ["score", "score_dev", "lead_10", "stop_frac"] + [f"L_{x}" for x in WHY]}
        say(f"  {LABEL[k]:15s} {fmt(row['score'])}  " + "  ".join(fmt(row["L_" + x]) for x in WHY)
            + f" | {fmt(row['score_dev'])}")
    P["pace"] = pace_table(pd.concat(DD.values()), arms, ref, say)
    say("  places where the human stood still, both sets:")
    say(STOP_HEAD)
    P["stops"] = stops_table(pd.concat(SE.values()), arms, ref, rng, say)
    say(f"  ... only where the car arrives moving (the stop begins >= {START_S:.0f} s into the scored clip), both sets:")
    say(STOP_HEAD)
    se_all = pd.concat(SE.values())
    P["stops_moving"] = stops_table(se_all[se_all["t_a"] >= START_S], arms, ref, np.random.default_rng(6), say)
    if foc in arms:
        P["stopping"] = stopping(pd.concat(DD.values()), ref, foc, rng, say)

    # ---- gates G1-G3 -----------------------------------------------------------------------
    if ref == "baseline":
        say("\n================ pre-registered gates (MoOP vs Baseline)")
        o, h = res["origin150"]["paired"]["dual"], res["hard100"]["paired"]["dual"]
        g1 = {}
        for tm in TERMS:
            sig = o[tm][1] > 0 or o[tm][2] < 0
            half = abs(o[tm][0]) >= 0.5 * abs(o["score"][0])
            same = np.sign(o[tm][0]) == np.sign(h[tm][0])
            g1[tm] = bool(sig and half and same)
            say(f"  G1 {tm:7s}: Origin150 {fmt(o[tm])} = {100 * o[tm][0] / o['score'][0]:.0f}% of the gap;"
                f" Hard100 {fmt(h[tm])} = {100 * h[tm][0] / h['score'][0]:.0f}%  ->  {'EXPLAINS' if g1[tm] else 'no'}")
        g2 = bool(o["alt"][2] < 0 and h["alt"][2] < 0)       # arm - ref < 0 means dense has MORE
        g3 = bool((o["slow"][1] > 0 and h["slow"][1] > 0) or (o["stop_frac"][2] < 0 and h["stop_frac"][2] < 0))
        say(f"  G2 H-route: alt-route rate MoOP - dense  Origin150 {fmt(o['alt'])}  Hard100 {fmt(h['alt'])}"
            f"  ->  {'SUPPORTED' if g2 else 'not supported'}")
        say(f"  G3 H-stop : slow term {fmt(o['slow'])} / {fmt(h['slow'])};  stop-time share MoOP - dense "
            f"{fmt(o['stop_frac'])} / {fmt(h['stop_frac'])}  ->  {'SUPPORTED' if g3 else 'not supported'}")
        res["gates"] = {"G1": g1, "G2": g2, "G3": g3}
        plots(res, out)

    (out / f"summary{tag}.txt").write_text("\n".join(lines) + "\n")
    (out / f"metrics{tag}.json").write_text(json.dumps(res, indent=1))
    (out / "config.json").write_text(json.dumps({
        "exp_id": args.exp_id, "reference": ref, "arms": {k: c for k, c, _ in ARMS},
        "sets": [s[0] for s in SETS], "max_dev_m": MAX_DEV, "progress_saturation": SAT,
        "min_gt_dist_m": MIN_GT, "stop_speed_mps": STOP_V, "alt_route_min_m": ALT_M,
        "route_end_progress": ROUTE_END, "min_stop_s": MIN_STOP_S, "at_m": AT_M, "band_m": [BAND_IN, BAND_OUT],
        "ahead_s": AHEAD_S, "early_s": EARLY_S, "slow_v_mps": SLOW_V, "lead_m": LEAD_M, "start_s": START_S,
        "n_boot": 5000, "seed": 0, "runs_root": str(RUNS), "lp_r50_origin150": str(SOOWON / "lp_r50"),
        "gt_tracks": str(USDZ), "traffic": "replayed (trafficsim.skip: true)",
    }, indent=1))
    print(f"\nwrote {out}/summary{tag}.txt, metrics{tag}.json, config.json, rollouts.parquet, stop_events.parquet")


if __name__ == "__main__":
    main()
