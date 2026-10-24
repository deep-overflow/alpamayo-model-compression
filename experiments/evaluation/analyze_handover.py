"""Why is the dense model slower while moving in closed loop? Stored rollout logs only, no GPU.

`analyze_closedloop_gap.py` found that MoOP's closed-loop advantage over the dense model is pace,
and that the dense model's deficit builds in the first seconds after the forced-GT hand-over.
This reads every rollout's `rollout.asl` step by step around that hand-over
(plans/2026-10-07_dense-pace-cause.md):

  H-artifact     the speed lost at hand-over comes from the harness, not from a driver model
  H-extrapolate  the dense model reads that deceleration from its ego history and plans to
                 continue it; the pruned models plan back to the speed the scene calls for
  H-caution      (alternative) the dense model slows for reasons its CoC states

Per control step, 1.7 s before hand-over to 10 s after: the car's speed
(`controller_request.state`), the controller's reported acceleration
(`controller_return.dynamic_state`), the driver's plan (`driver_return.trajectory`, 65 poses over
6.4 s; mean planned speed over 0-1, 1-2, 2-3 s) and its CoC text. The human's speed is the usdz
track, which shares the sim clock. Step k = 0 is the hand-over: the first plan, issued on inputs
that are identical in every arm; k = 1 is the first plan after the car has driven a free step.

Gates, fixed after one scene had been read by hand and before the rest:
  A1  moving starts (>= 2 m/s): the speed change over the first free step is the same in every arm
      (per scene, max - min across all arms' rollouts < 0.5 points in >= 95% of scenes), and the
      controller's acceleration at hand-over is saturated (<= -7.9 m/s2) in a majority
  B0  at k = 0, r = planned speed 2-3 s out / current speed - 1 differs between dense and MoOP by
      less than a quarter of the B1 difference
  B1  at k = 1, r is lower for dense than for MoOP, scene-paired CI excluding 0, on both sets
  C1  the larger the drop, the larger dense's shortfall 2.5 s later relative to MoOP's
      (Spearman over scenes, CI excluding 0, on both sets)
  D1  the MoOP - dense score gap is larger in the top tertile of scenes by drop size than in the
      bottom one (pooled CI excluding 0); confounded with speed, so supportive at best
  E1  among dense plans in the first second that ask for a > 10% slowdown within 2-3 s, the share
      whose CoC names a slowdown: > 1/2 supports H-caution, < 1/4 rejects it

Extraction needs alpasim's venv (`alpasim_utils` reads the protobuf logs):

    /mnt/nvme1n1/ad_vla/venvs/alpasim-chan/bin/python experiments/evaluation/analyze_handover.py \
        --exp-id handover_cause_v1
"""

import argparse
import asyncio
import json
import pickle
from multiprocessing import Pool

import analyze_closedloop_gap as G
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

O = G.O
STEP_US, PRE, POST = 100_000, 17, 100      # control step; steps kept before / after hand-over
MOVING = 2.0                                # a moving start: at least 2 m/s at hand-over
SLOW_WORDS = ("decelerat", "slow", "stop", "yield", "brake", "keep distance", "follow")
KS = (0, 1, 2, 3, 5, 10, 20, 30, 50, 100)   # steps the time-course tables read
IDX = ["set", "arm", "scene", "rollout"]
EXAMPLE = "clipgt-127c4f85-a5db-4920-9813-7970dcd6ebdb"


def read_rollout(task):
    """One rollout.asl -> per-step rows from 1.7 s before the hand-over to 10 s after it."""
    from alpasim_utils.logs import async_read_pb_log

    key, asl, stop_k = task

    async def go():
        st, ret, plans = {}, {}, {}
        async for e in async_read_pb_log(str(asl)):
            if e.HasField("controller_request"):
                r = e.controller_request.state
                st[r.timestamp_us] = r.state.linear_velocity.x
                if plans and r.timestamp_us > min(plans) + (stop_k + 1) * STEP_US:
                    break                                               # nothing later is read
            elif e.HasField("controller_return"):
                for s in e.controller_return.states:
                    ret[s.timestamp_us] = (s.dynamic_state.linear_velocity.x, s.dynamic_state.linear_acceleration.x)
            elif e.HasField("driver_return"):
                ps = e.driver_return.trajectory.poses
                if len(ps) < 31:
                    continue
                xy = np.array([[p.pose.vec.x, p.pose.vec.y] for p in ps[:31]])     # (31, 2) first 3 s of the plan
                seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)                  # (30,) metres per 0.1 s
                blob = e.driver_return.debug_info.unstructured_debug_info
                coc = str(pickle.loads(blob).get("reasoning_text") or "") if blob else ""
                plans[ps[0].timestamp_us] = (seg[:10].sum(), seg[10:20].sum(), seg[20:30].sum(), coc[:90])
        return st, ret, plans

    st, ret, plans = asyncio.run(go())
    if not plans:
        return key, []
    t_ho = min(plans)                                  # the first plan is issued at the hand-over
    by_k = {round((t - t_ho) / STEP_US): t for t in st}
    rows = []
    for k in range(-PRE, POST + 1):
        t = by_k.get(k)
        if t is None or abs(t - t_ho - k * STEP_US) > 2000:
            continue
        p, r = plans.get(t), ret.get(t, (np.nan, np.nan))
        rows.append((k, t, st[t], r[0], r[1], *(p if p else (np.nan, np.nan, np.nan, ""))))
    return key, rows


def read_many(tasks, workers):
    """(key, asl, stop_k) tasks -> one row per control step, with the human's speed alongside."""
    rows = []
    with Pool(workers) as pool:
        for i, (key, rr) in enumerate(pool.imap_unordered(read_rollout, tasks, chunksize=4)):
            rows += [(*key, *x) for x in rr]
            if (i + 1) % 500 == 0:
                print(f"  read {i + 1}/{len(tasks)} rollouts", flush=True)
    df = pd.DataFrame(rows, columns=[*IDX, "k", "t_us", "v", "v_ret", "a_ret", "p1", "p12", "p23", "coc"])
    uuid = pd.read_csv(G.SCENES).set_index("scene_id")["uuid"]
    parts = []
    for sc, g in df.groupby("scene"):
        gt_t, gt_s = G.gt_track(uuid[sc])                      # (N,), (N,) the human's arc length against time
        t = g["t_us"].to_numpy(float)                          # (rows,)
        s = {d: np.interp(t + d, gt_t, gt_s) for d in (-5e4, 0.0, 5e4, 1e6, 2e6, 3e6)}   # arc length at t + d
        parts.append(g.assign(vh=(s[5e4] - s[-5e4]) / 0.1,     # human speed now, then over the plan's three seconds
                              h1=np.where(t + 1e6 <= gt_t[-1], s[1e6] - s[0.0], np.nan),   # the track ends with the clip
                              h12=np.where(t + 2e6 <= gt_t[-1], s[2e6] - s[1e6], np.nan),
                              h23=np.where(t + 3e6 <= gt_t[-1], s[3e6] - s[2e6], np.nan)))
    return pd.concat(parts, ignore_index=True)


def build(rebuild, out, workers):
    cache = out / "steps.parquet"
    if cache.exists() and not rebuild:
        return pd.read_parquet(cache)
    roll = pd.read_parquet(O / "closedloop_gap_v1/rollouts.parquet")
    roll = roll[roll["end"] != "aborted"]
    tasks = []
    for label, prefix, n in G.SETS:
        for key, cfg, _ in G.ARMS:
            d = G.run_dir(prefix, cfg, n)
            for r in roll[(roll["set"] == label) & (roll["arm"] == key)].itertuples():
                tasks.append(((label, key, r.scene, r.rollout), d / "rollouts" / r.scene / r.rollout / "rollout.asl",
                              POST))
    df = read_many(tasks, workers)
    df.to_parquet(cache)
    return df


def build_arms(rebuild, out, workers, df):
    """Every Origin150 arm with a merged run: steps to 5.5 s after the hand-over, and the facts of each arm."""
    cache = out / "steps_arms.parquet"
    paper = {cfg: key for key, cfg, _ in G.ARMS}
    runs = {d.name[len("m2601_merged_"):]: d for d in sorted(G.RUNS.glob("m2601_merged_*"))}
    runs["lp_r50"] = G.SOOWON / "lp_r50"
    info, tasks = {}, []
    for cfg, d in runs.items():
        s = d / "aggregate/results-summary.json"
        if not s.exists():
            continue
        rr = json.loads(s.read_text())["rollouts"]
        sc = np.array([r["score"] if r["score"] is not None else np.nan for r in rr], float)   # (rollouts,)
        mp = O / cfg / "slim_meta.json"
        m = json.loads(mp.read_text()) if mp.exists() else None
        kept = ({"q": float(np.mean([len(x["q"]) for x in m["vlm"]])), "mlp": float(np.mean([len(x["mlp"]) for x in m["vlm"]])),
                 "exp_mlp": float(np.mean([len(x["mlp"]) for x in m["expert"]]))} if m
                else {"q": 32.0, "mlp": 12288.0, "exp_mlp": 8256.0} if cfg == "baseline"
                else {"q": float("nan"), "mlp": float("nan"), "exp_mlp": float("nan")})
        info[cfg] = {"score": float(np.nanmean(sc)), "zeroed": float(np.mean(sc == 0)), "n": len(rr), **kept}
        if cfg not in paper:
            tasks += [(("origin150", cfg, r["clipgt_id"], r["rollout_id"]),
                       d / "rollouts" / r["clipgt_id"] / r["rollout_id"] / "rollout.asl", 55) for r in rr]
    if cache.exists() and not rebuild:
        return pd.read_parquet(cache), info
    new = read_many([t for t in tasks if t[1].exists()], workers)
    old = df[(df["set"] == "origin150") & (df["k"] <= 55)]
    old = old.assign(arm=old["arm"].map({key: cfg for cfg, key in paper.items()}))
    steps = pd.concat([old, new], ignore_index=True)
    steps.to_parquet(cache)
    return steps, info


def wide(df, col):
    """(rollouts, steps) table of one column."""
    return df.pivot(index=IDX, columns="k", values=col)


def scene_pair(F, col, a, b, rng):
    """Scene-paired mean of F[col] for arm a minus arm b, with a bootstrap CI; F is per rollout."""
    m = F.groupby(["arm", "scene"])[col].mean().unstack("arm")
    return G.boot((m[a] - m[b]).dropna().to_numpy(), rng)


def boot_spearman(x, y, rng, n=5000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    idx = rng.integers(0, len(x), size=(n, len(x)))
    b = np.array([spearmanr(x[i], y[i])[0] for i in idx])
    return float(spearmanr(x, y)[0]), float(np.nanquantile(b, .025)), float(np.nanquantile(b, .975)), len(x)


def plots(df, F, res, out):
    arms = [k for k, _, _ in G.ARMS]
    sets = [("origin150", "Origin150"), ("hard100", "Hard100")]
    style = {"baseline": (G.INK, 2.4, 3, "Baseline (dense)"), "dual": (G.CAT[0], 2.4, 3, "MoOP")}

    # 1. speed against the human around the hand-over, moving starts
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), sharey=True)
    for ax, (key, title) in zip(axes, sets):
        tc = res[key]["timecourse"]
        ks = np.array(tc["k"])
        for k in arms:
            c, lw, z, lab = style.get(k, (G.MUTED, 1.0, 2, "the other six arms" if k == "dual+E" else None))
            ax.plot(ks / 10, tc["ratio"][k], color=c, linewidth=lw, zorder=z, label=lab,
                    alpha=1.0 if k in style else 0.55, solid_capstyle="round")
        for k in style:
            ax.annotate(f"{tc['ratio'][k][-1]:.2f}", (ks[-1] / 10, tc["ratio"][k][-1]), textcoords="offset points",
                        xytext=(6, 0), va="center", fontsize=9)
        ax.axhline(1.0, color=G.MUTED, linewidth=1)
        ax.axvline(0.0, color=G.MUTED, linewidth=1)
        ax.set_title(f"{title}, moving starts", loc="left", fontsize=10.5)
        ax.set_xlabel("seconds since the hand-over from the recording to the model")
        ax.grid(axis="y", color=G.GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
    axes[0].set_ylabel("speed, car / human driver")
    h, lab = axes[0].get_legend_handles_labels()
    fig.suptitle("Every arm loses the same speed at the hand-over; the dense model then loses more and recovers last",
                 x=0.012, ha="left", fontsize=12, fontweight="bold")
    fig.legend(h, lab, loc="upper left", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.008, 0.945))
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out / "plots/speed_around_handover.png", dpi=150)
    plt.close(fig)

    # 2. where the plan is heading, just before and just after the drop
    y = np.arange(len(arms))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=True)
    for ax, (key, title) in zip(axes, sets):
        P = res[key]["plan"]
        for yi, k in zip(y, arms):
            b, a = 100 * P[k]["head_0"][0], 100 * P[k]["head_1"][0]
            ax.plot([b, a], [yi, yi], color=G.ORD[1], linewidth=2, solid_capstyle="round", zorder=2)
            ax.plot(b, yi, "o", color=G.ORD[0], markersize=8.5, markeredgecolor=G.BG, markeredgewidth=1.5, zorder=3,
                    label="plan issued at the hand-over" if k == arms[0] else None)
            ax.plot(a, yi, "o", color=G.ORD[2], markersize=8.5, markeredgecolor=G.BG, markeredgewidth=1.5, zorder=3,
                    label="first plan after it, 0.1 s later" if k == arms[0] else None)
            ax.annotate(f"{a:+.1f}%", (a, yi), textcoords="offset points", xytext=(-9, 0), ha="right", va="center",
                        fontsize=9)
        ax.axvline(0, color=G.MUTED, linewidth=1)
        ax.set_yticks(y, [G.LABEL[k] for k in arms])
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo - 0.18 * (hi - lo), hi + 0.04 * (hi - lo))
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.set_xlabel("planned speed 2-3 s ahead, against the speed at the hand-over (%)")
        ax.grid(axis="x", color=G.GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
    h, lab = axes[0].get_legend_handles_labels()
    fig.suptitle("What each model plans once it has seen the car slow down", x=0.012, ha="left", fontsize=12,
                 fontweight="bold")
    fig.legend(h, lab, loc="upper left", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.008, 0.945))
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out / "plots/plan_after_drop.png", dpi=150)
    plt.close(fig)

    # 4. every Origin150 arm: how strongly it extrapolates the hand-over deceleration, against its score
    T = pd.DataFrame(res["arms"]["table"]).T
    name = {"baseline": "Baseline (dense)", "slim_dual_u40_v2": "MoOP 24%", "slim_dual_u20_v2": "MoOP 12%",
            "slim_dual_u30_v2": "MoOP 18%", "slim_expert_znorm_r25": "expert-only", "lp_r50": "LLM-Pruner",
            "slim_tyr_u40_r": "Tyr", "slim_wanda_u40_v2": "Wanda", "slim_coc_u40_v2": "CE-only",
            "slim_recover_dual_u55": "MoOP 33% + recovery", "slim_act_h0_l35k": "MLP-only", "slim_j_u40_v2": "J-lens only"}
    sick = T["zeroed"] > 0.15
    full = (T["q"] == 32) & ~sick
    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    for m, c, lab in [(~sick & ~full, G.CAT[0], "VLM Q heads pruned"), (full, G.CAT[1], "all 32 VLM Q heads kept"),
                      (sick, G.MUTED, "zeroes more than 15% of rollouts")]:
        ax.scatter(100 * T.loc[m, "X"], T.loc[m, "score"], s=70, color=c, edgecolor=G.BG, linewidth=1.5, label=lab, zorder=3)
    below = {"slim_dual_u20_v2", "slim_dual_u30_v2", "slim_act_h0_l35k"}   # labels that would run into a neighbour
    for cfg, lab in name.items():
        if cfg in T.index:
            off, ha = ((0, -14), "center") if cfg in below else ((-8, 6), "right") if cfg == "lp_r50" else ((7, 5), "left")
            ax.annotate(lab, (100 * T.loc[cfg, "X"], T.loc[cfg, "score"]), textcoords="offset points", xytext=off,
                        ha=ha, fontsize=8.5)
    ax.set_xlabel("planned speed 2-3 s ahead against speed now, in the plan made 0.2 s after the hand-over (%, moving starts)")
    ax.set_ylabel("closed-loop score, Origin150")
    ax.grid(color=G.GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    h, lab = ax.get_legend_handles_labels()
    fig.suptitle(f"{len(T)} Origin150 arms: how far each continues the hand-over braking, against its score",
                 x=0.012, ha="left", fontsize=12, fontweight="bold")
    fig.legend(h, lab, loc="upper left", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.008, 0.945))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out / "plots/arms_extrapolation_vs_score.png", dpi=150)
    plt.close(fig)

    # 3. one scene, step by step
    ex = df[(df["scene"] == EXAMPLE) & df["arm"].isin(["baseline", "dual"]) & (df["k"] <= 60)]
    if len(ex):
        fig, ax = plt.subplots(figsize=(9.6, 4.4))
        one = ex[ex["arm"] == "baseline"]
        one = one[one["rollout"] == one["rollout"].iloc[0]]
        ax.plot(one["k"] / 10, one["vh"], color=G.MUTED, linewidth=2, label="human driver (recording)")
        for k, (c, _, _, lab) in style.items():
            g = ex[ex["arm"] == k]
            g = g[g["rollout"] == g["rollout"].iloc[0]].sort_values("k")
            ax.plot(g["k"] / 10, g["v"], color=c, linewidth=2.2, label=lab, solid_capstyle="round")
            p = g[g["k"] == 1].iloc[0]                                  # the first plan after the drop
            ax.plot([0.1, 0.6, 1.6, 2.6], [p["v"], p["p1"], p["p12"], p["p23"]], color=c, linewidth=1.0, marker="o",
                    markersize=6, markeredgecolor=G.BG, markeredgewidth=1.2, alpha=0.9)
        ax.axvline(0.0, color=G.MUTED, linewidth=1)
        ax.set_xlabel("seconds since the hand-over")
        ax.set_ylabel("speed (m/s)")
        ax.grid(axis="y", color=G.GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        h, lab = ax.get_legend_handles_labels()
        fig.suptitle("One scene (127c4f85): thick = speed driven, thin with dots = the plan made 0.1 s after the hand-over",
                     x=0.012, ha="left", fontsize=11, fontweight="bold")
        fig.legend(h, lab, loc="upper left", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.008, 0.945))
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        fig.savefig(out / "plots/example_scene.png", dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="handover_cause_v1")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--rebuild-arms", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    out = O / args.exp_id
    (out / "plots").mkdir(parents=True, exist_ok=True)
    df = build(args.rebuild, out, args.workers)
    rng = np.random.default_rng(0)
    res, lines = {}, []
    arms = [k for k, _, _ in G.ARMS]

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    roll = pd.read_parquet(O / "closedloop_gap_v1/rollouts.parquet").set_index(IDX)[
        ["score", "dur_w", "why", "lead_5", "lead_10"]]
    V, VH, A = wide(df, "v"), wide(df, "vh"), wide(df, "a_ret")              # (rollouts, steps)
    P1, P12, P23, H23 = wide(df, "p1"), wide(df, "p12"), wide(df, "p23"), wide(df, "h23")
    F = pd.DataFrame({
        "v0": V[0], "drop": V[1] / V[0] - 1, "drop_abs": V[1] - V[0], "a0": A[0],
        "human_step": VH[1] / VH[0] - 1,
        "r1_0": P1[0] / V[0] - 1, "r12_0": P12[0] / V[0] - 1, "r23_0": P23[0] / V[0] - 1,
        "r1_1": P1[1] / V[1] - 1, "r12_1": P12[1] / V[1] - 1, "r23_1": P23[1] / V[1] - 1,
        "head_0": P23[0] / V[0] - 1, "head_1": P23[1] / V[0] - 1,              # against the speed at hand-over
        "p23_0": P23[0], "p23_1": P23[1], "h23_0": H23[0], "h23_1": H23[1], "vh0": VH[0],
        "x2": P23[2] / V[2] - 1,
        "rel25": V[25] / V[0] - VH[25] / VH[0], "rel50": V[50] / V[0] - VH[50] / VH[0],
    }).join(roll)
    F["moving"] = F["v0"] >= MOVING
    F.loc[F["dur_w"] < 2.5, "rel25"] = np.nan                                 # the scored window had already closed
    F.loc[F["dur_w"] < 5.0, "rel50"] = np.nan
    F = F.reset_index()
    say(f"{len(F)} rollouts, {int(F['moving'].sum())} moving starts (>= {MOVING:.0f} m/s at hand-over);"
        f" {len(df)} control steps read")

    for label, _, _ in G.SETS:
        Fs = F[(F["set"] == label)]
        M = Fs[Fs["moving"]]
        R = res[label] = {"n_rollouts": len(Fs), "n_moving": len(M), "n_scenes_moving": int(M["scene"].nunique())}
        say(f"\n================ {label}: {len(Fs)} rollouts in {Fs['scene'].nunique()} scenes;"
            f" moving starts {len(M)} in {M['scene'].nunique()} scenes")

        # ---- A1: the hand-over drop is the harness's ----------------------------------------
        sp = M.groupby("scene")["drop"].agg(lambda x: 100 * (x.max() - x.min()))    # points, across all arms' rollouts
        R["A1"] = {"drop_mean": float(M["drop"].mean()), "drop_abs_median": float(M["drop_abs"].median()),
                   "human_step": float(M["human_step"].mean()), "spread_median_pp": float(sp.median()),
                   "spread_lt_05": float((sp < 0.5).mean()), "a0_saturated": float((M["a0"] <= -7.9).mean()),
                   "a0_median": float(M["a0"].median()),
                   "by_arm": {k: float(M[M["arm"] == k]["drop"].mean()) for k in arms}}
        a1 = R["A1"]
        R["A1"]["pass"] = bool(a1["spread_lt_05"] >= 0.95 and a1["a0_saturated"] > 0.5)
        say("\n[A] the first free control step (0.1 s after the hand-over), moving starts")
        say(f"  speed change over that step: mean {100 * a1['drop_mean']:+.2f}%, median {a1['drop_abs_median']:+.2f} m/s;"
            f" the human over the same 0.1 s: {100 * a1['human_step']:+.2f}%")
        say("  by arm: " + "  ".join(f"{G.LABEL[k]} {100 * a1['by_arm'][k]:+.2f}%" for k in arms))
        say(f"  per scene, max - min across all arms' rollouts: median {a1['spread_median_pp']:.3f} points;"
            f" below 0.5 points in {100 * a1['spread_lt_05']:.1f}% of scenes")
        say(f"  controller acceleration at the hand-over: median {a1['a0_median']:+.2f} m/s2, at saturation"
            f" (<= -7.9) in {100 * a1['a0_saturated']:.1f}% of moving starts")
        pre = df[(df["set"] == label) & (df["k"] <= 0)].merge(M[IDX], on=IDX)
        ramp = pre.groupby("k")["a_ret"].median()
        say("  the controller's reported acceleration while the car is still being moved along the recording"
            " (median, m/s2): " + "  ".join(f"{k / 10:+.1f}s {ramp[k]:+.1f}" for k in (-15, -12, -9, -6, -3, 0) if k in ramp))
        say(f"  A1 -> {'PASS' if a1['pass'] else 'FAIL'}")
        fit = M[["a0", "drop_abs"]].dropna()
        sl = np.polyfit(fit["a0"], fit["drop_abs"], 1)
        R["A1"]["drop_vs_a0"] = {"slope_s": float(sl[0]), "intercept": float(sl[1]),
                                 "r": float(np.corrcoef(fit["a0"], fit["drop_abs"])[0, 1])}
        say(f"  post hoc: speed lost in that step = {sl[0]:.3f} s x the controller's acceleration at the hand-over"
            f" {sl[1]:+.3f} m/s, r = {R['A1']['drop_vs_a0']['r']:.3f} over {len(fit)} rollouts")

        # ---- B: what each model plans before and after the drop -----------------------------
        say("\n[B] planned mean speed against the car's speed when the plan is made (moving starts; scene means)")
        say(f"  {'arm':15s} at the hand-over (identical inputs): 0-1 s   1-2 s   2-3 s  |  0.1 s later: 0-1 s   1-2 s   2-3 s"
            f"  |  2-3 s out vs speed at hand-over   | plans 0.2 s later asking > 10% less")
        R["plan"] = {}
        for k in arms:
            mk = M[M["arm"] == k]
            m = mk.groupby("scene")[["r1_0", "r12_0", "r23_0", "r1_1", "r12_1", "r23_1", "head_0", "head_1", "x2"]].mean()
            R["plan"][k] = {c: G.boot(m[c].dropna().to_numpy(), rng) for c in m.columns}
            p = R["plan"][k]
            p["x2_lt10"] = float((mk["x2"] < -0.10).mean())
            say(f"  {G.LABEL[k]:15s}                                   {100 * p['r1_0'][0]:+6.1f}% {100 * p['r12_0'][0]:+6.1f}%"
                f" {100 * p['r23_0'][0]:+6.1f}%  |            {100 * p['r1_1'][0]:+6.1f}% {100 * p['r12_1'][0]:+6.1f}%"
                f" {100 * p['r23_1'][0]:+6.1f}%  |          {100 * p['head_1'][0]:+6.1f}% [{100 * p['head_1'][1]:+.1f},"
                f"{100 * p['head_1'][2]:+.1f}]   |        {100 * p['x2_lt10']:5.1f}%")
        hv = M[M["h23_0"].notna()]
        hum = hv[hv["arm"] == "baseline"]
        R["clean_inputs"] = {"human": float(hum["h23_0"].sum() / hum["vh0"].sum()),
                             "plan": {k: float(hv[hv["arm"] == k]["p23_0"].sum() / hv[hv["arm"] == k]["v0"].sum()) for k in arms},
                             "abs_err": {k: float((hv[hv["arm"] == k]["p23_0"] - hv[hv["arm"] == k]["h23_0"]).abs().mean())
                                         for k in arms}}
        ci = R["clean_inputs"]
        say(f"  post hoc, on the identical inputs at the hand-over (ratio of sums): the human's speed 2-3 s later was"
            f" {ci['human']:.3f}x; plans: " + "  ".join(f"{G.LABEL[k]} {ci['plan'][k]:.3f}" for k in arms))
        say("    mean |planned - human| speed 2-3 s out (m/s): " + "  ".join(
            f"{G.LABEL[k]} {ci['abs_err'][k]:.2f}" for k in arms))
        b0, b1 = scene_pair(M, "r23_0", "baseline", "dual", rng), scene_pair(M, "r23_1", "baseline", "dual", rng)
        R["B0"] = {"diff": b0, "pass": bool(abs(b0[0]) < 0.25 * abs(b1[0]))}
        R["B1"] = {"diff": b1, "pass": bool(b1[2] < 0)}
        say(f"  dense - MoOP, 2-3 s out, scene-paired: at the hand-over {G.fmt([100 * x for x in b0], 2)} points;"
            f" 0.1 s later {G.fmt([100 * x for x in b1], 2)} points")
        say(f"  B0 (no standing difference before the drop) -> {'PASS' if R['B0']['pass'] else 'FAIL'};"
            f"  B1 (dense plans lower after it) -> {'PASS' if R['B1']['pass'] else 'FAIL'}")
        R["plan_vs_dense"] = {k: scene_pair(M, "r23_1", k, "baseline", rng) for k in arms if k != "baseline"}
        say("  every arm - dense at 0.1 s, 2-3 s out: " + "  ".join(
            f"{G.LABEL[k]} {G.fmt([100 * x for x in v], 1)}" for k, v in R["plan_vs_dense"].items()))

        # ---- time course ---------------------------------------------------------------------
        ok = df[(df["set"] == label)].merge(M[[*IDX, "dur_w"]], on=IDX)
        ok = ok[(ok["k"] <= 0) | (ok["k"] / 10 <= ok["dur_w"])]                  # only while the rollout is scored
        ks = sorted(k for k in ok["k"].unique() if k >= -15)
        g = ok.groupby(["arm", "k"])[["v", "vh", "p23"]].sum()
        R["timecourse"] = {"k": [int(k) for k in ks],
                           "ratio": {a: [float(g.loc[(a, k), "v"] / g.loc[(a, k), "vh"]) for k in ks] for a in arms}}
        say("\n[C] speed, car / human (ratio of sums over moving starts still inside their scored window)")
        say(f"  {'arm':15s} " + " ".join(f"{k / 10:>6.1f}s" for k in KS))
        for a in arms:
            say(f"  {G.LABEL[a]:15s} " + " ".join(f"{g.loc[(a, k), 'v'] / g.loc[(a, k), 'vh']:7.3f}" for k in KS))
        pl = ok[ok["p23"].notna()]
        gp = pl.groupby(["arm", "k"])[["v", "p23"]].sum()
        say("  planned speed 2-3 s out / speed now, same rollouts:")
        for a in arms:
            say(f"  {G.LABEL[a]:15s} " + " ".join(f"{gp.loc[(a, k), 'p23'] / gp.loc[(a, k), 'v']:7.3f}" for k in KS))

        # ---- C1: dose ------------------------------------------------------------------------
        sc = M.groupby(["arm", "scene"])[["drop", "rel25", "rel50", "score"]].mean().unstack("arm")
        drop_mag = -100 * sc["drop"]["baseline"]                                  # points of speed lost, per scene
        short = -100 * (sc["rel25"]["baseline"] - sc["rel25"]["dual"])            # dense's extra shortfall at 2.5 s
        c1 = boot_spearman(drop_mag, short, rng)
        R["C1"] = {"rho": c1, "pass": bool(c1[1] > 0)}
        say("\n[D] 2.5 s after the hand-over: speed change since then, minus the human's (points; scene means)")
        R["persist"] = {}
        for a in arms:
            x, yv = -100 * sc["drop"][a], -100 * sc["rel25"][a]
            okk = ~(x.isna() | yv.isna())
            xv, yy = x[okk].to_numpy(), yv[okk].to_numpy()
            idx = rng.integers(0, len(xv), size=(2000, len(xv)))
            sl = np.array([np.polyfit(xv[i], yy[i], 1)[0] for i in idx])
            R["persist"][a] = {"mean": float(yy.mean()), "slope": [float(np.polyfit(xv, yy, 1)[0]),
                                                                   float(np.quantile(sl, .025)), float(np.quantile(sl, .975))],
                               "rho": float(spearmanr(xv, yy)[0]), "n": len(xv)}
            p = R["persist"][a]
            say(f"  {G.LABEL[a]:15s} shortfall {p['mean']:+5.1f} points; per point lost at the hand-over"
                f" {p['slope'][0]:+.2f} [{p['slope'][1]:+.2f},{p['slope'][2]:+.2f}] (rho {p['rho']:+.2f}, n={p['n']})")
        say(f"  mean drop at the hand-over over these scenes: {drop_mag.mean():.1f} points")
        say(f"  C1: rho(drop, dense's extra shortfall over MoOP) = {c1[0]:+.3f} [{c1[1]:+.3f},{c1[2]:+.3f}] n={c1[3]}"
            f"  ->  {'PASS' if R['C1']['pass'] else 'FAIL'}")

        # ---- E1: does the reasoning text ask for the slowdown? -----------------------------
        early = df[(df["set"] == label) & df["k"].between(1, 10) & df["p23"].notna()].merge(M[IDX], on=IDX)
        low = early["coc"].str.lower()
        early = early.assign(slowing=early["p23"] / early["v"] - 1 < -0.10,
                             says=low.apply(lambda s: any(w in s for w in SLOW_WORDS)),
                             explicit=low.apply(lambda s: any(w in s for w in SLOW_WORDS[:5])))
        R["E1"] = {}
        say("\n[E] plans in the first second after the hand-over: how many ask for a > 10% slowdown within 2-3 s, and"
            " what their CoC says")
        say("    (post hoc columns: the registered word list counts 'keep distance' / 'follow', which two thirds of all"
            " CoCs contain; 'explicit' keeps only decelerate / slow / stop / yield / brake)")
        for a in arms:
            e = early[early["arm"] == a]
            s_ = e[e["slowing"]]
            R["E1"][a] = {"n": len(e), "slowing": float(e["slowing"].mean()),
                          "says_given_slowing": float(s_["says"].mean()) if len(s_) else float("nan"),
                          "says_overall": float(e["says"].mean()),
                          "explicit_given_slowing": float(s_["explicit"].mean()) if len(s_) else float("nan"),
                          "explicit_overall": float(e["explicit"].mean())}
            p = R["E1"][a]
            say(f"  {G.LABEL[a]:15s} {100 * p['slowing']:5.1f}% of {p['n']} plans slow down; of those, the CoC names a"
                f" slowdown in {100 * p['says_given_slowing']:5.1f}% (in {100 * p['says_overall']:4.1f}% of all plans)"
                f" | explicit: {100 * p['explicit_given_slowing']:5.1f}% (in {100 * p['explicit_overall']:4.1f}% of all)")
        e1 = R["E1"]["baseline"]["says_given_slowing"]
        R["E1"]["verdict"] = "supports H-caution" if e1 > 0.5 else "rejects H-caution" if e1 < 0.25 else "inconclusive"
        top = early[(early["arm"] == "baseline") & early["slowing"]]["coc"].str.slice(2, 40).value_counts().head(4)
        say("  what the dense model's CoC says in those plans, most common: " + "; ".join(
            f"'{t}' {c}" for t, c in top.items()))
        say(f"  E1 -> {R['E1']['verdict']}")

    # ---- D1: does the score gap sit where the drop is large? (pooled, tertiles within set) ----
    sc = F.groupby(["set", "arm", "scene"])[["drop", "score", "v0"]].mean().unstack("arm")
    gap = (sc["score"]["dual"] - sc["score"]["baseline"]).dropna()
    mag = (-100 * sc["drop"]["baseline"]).reindex(gap.index).fillna(0.0)          # a standing start loses nothing
    mag[sc["v0"]["baseline"].reindex(gap.index) < MOVING] = 0.0
    ter = mag.groupby(level="set").transform(lambda x: pd.qcut(x.rank(method="first"), 3, labels=False))
    res["D1"] = {"tertiles": {}}
    say("\n================ [F] the MoOP - dense score gap by how much speed the hand-over took (tertiles within each set)")
    for t, name in enumerate(["smallest drop", "middle", "largest drop"]):
        gg = gap[ter == t]
        res["D1"]["tertiles"][name] = {"n": len(gg), "drop": float(mag[ter == t].mean()), "gap": G.boot(gg.to_numpy(), rng)}
        say(f"  {name:14s} {len(gg):3d} scenes, mean drop {mag[ter == t].mean():5.1f} points:"
            f" gap {G.fmt(res['D1']['tertiles'][name]['gap'])}")
    hi, lo = gap[ter == 2].to_numpy(), gap[ter == 0].to_numpy()
    b = np.array([hi[rng.integers(0, len(hi), len(hi))].mean() - lo[rng.integers(0, len(lo), len(lo))].mean()
                  for _ in range(5000)])
    d1 = [float(hi.mean() - lo.mean()), float(np.quantile(b, .025)), float(np.quantile(b, .975))]
    rho = boot_spearman(mag, gap, rng)
    res["D1"].update({"top_minus_bottom": d1, "rho": rho, "pass": bool(d1[1] > 0)})
    say(f"  largest - smallest: {G.fmt(d1)};  rho(drop, gap) = {rho[0]:+.3f} [{rho[1]:+.3f},{rho[2]:+.3f}] n={rho[3]}"
        f"  ->  D1 {'PASS' if res['D1']['pass'] else 'FAIL'}")

    # ---- post hoc: a standing start has no speed to lose, so the hand-over cannot perturb it ----
    sc = F.groupby(["set", "arm", "scene"])[["score", "lead_5", "lead_10", "v0"]].mean().unstack("arm")
    mv = sc["v0"]["baseline"] >= MOVING
    res["standing"] = {}
    say("\n================ [F2] post hoc - standing starts (< 2 m/s at the hand-over) against moving starts, MoOP - dense")
    for name, sel in [("origin150", sc.index.get_level_values("set") == "origin150"),
                      ("hard100", sc.index.get_level_values("set") == "hard100"), ("both", np.ones(len(sc), bool))]:
        for kind, m in [("moving", mv & sel), ("standing", ~mv & sel)]:
            row = {"n": int(m.sum())}
            for c in ["score", "lead_5", "lead_10"]:
                d = (sc[c]["dual"] - sc[c]["baseline"])[m].dropna().to_numpy()
                row[c] = G.boot(d, rng) if len(d) > 1 else [float("nan")] * 3
                row[f"n_{c}"] = len(d)
            res["standing"][f"{name}_{kind}"] = row
            say(f"  {name:9s} {kind:8s} {row['n']:3d} scenes: score {G.fmt(row['score'])}   lead at 5 s"
                f" {G.fmt(row['lead_5'], 2)} m (n={row['n_lead_5']})   at 10 s {G.fmt(row['lead_10'], 2)} m"
                f" (n={row['n_lead_10']})")

    # ---- across every Origin150 arm (amendment to the plan) --------------------------------
    steps, info = build_arms(args.rebuild_arms, out, args.workers, df)
    Va, VHa, Pa = (steps.pivot(index=IDX, columns="k", values=c) for c in ("v", "vh", "p23"))   # (rollouts, steps)
    Fa = pd.DataFrame({"v0": Va[0], "x": Pa[2] / Va[2] - 1, "v25": Va[25], "vh25": VHa[25]}).reset_index()
    Fa = Fa[Fa["v0"] >= MOVING]
    T = {}
    for cfg, g in Fa.groupby("arm"):
        ok = g.dropna(subset=["v25", "vh25"])
        T[cfg] = {**info[cfg], "X": float(g.groupby("scene")["x"].mean().mean()), "X_lt10": float((g["x"] < -0.10).mean()),
                  "P": float(ok["v25"].sum() / ok["vh25"].sum()), "n_moving": len(g)}
    T = pd.DataFrame(T).T.sort_values("X")
    say(f"\n================ [G] every Origin150 arm with a merged run ({len(T)} arms); X = planned speed 2-3 s out /"
        f" speed now - 1, for the plan made 0.2 s after the hand-over (moving starts); P = car / human speed at 2.5 s")
    say(f"  {'arm':28s}  VLM Q heads  VLM MLP  expert MLP |     X    share < -10%      P    score  zeroed")
    for cfg, r in T.iterrows():
        say(f"  {cfg:28s}     {r['q']:5.1f}     {r['mlp']:6.0f}     {r['exp_mlp']:6.0f} | {100 * r['X']:+6.1f}%     "
            f"{100 * r['X_lt10']:5.1f}%     {r['P']:.3f}   {r['score']:.3f}   {100 * r['zeroed']:4.1f}%")
    healthy = T[T["zeroed"] <= 0.15]
    ga, gh = boot_spearman(T["X"], T["score"], rng), boot_spearman(healthy["X"], healthy["score"], rng)
    gp, gq = boot_spearman(T["X"], T["P"], rng), boot_spearman(T["q"], T["X"], rng)
    lowest = list(T.index[:2])
    res["arms"] = {"table": T.to_dict(orient="index"), "rho_X_score": ga, "rho_X_score_healthy": gh, "rho_X_P": gp,
                   "rho_q_X": gq, "lowest_X": lowest,
                   "G_arms": bool(ga[1] > 0), "G_arms_healthy": bool(gh[1] > 0),
                   "G_dense": bool(set(lowest) == {"baseline", "slim_expert_znorm_r25"}), "G_pace": bool(gp[1] > 0)}
    say(f"  rho(X, score) = {ga[0]:+.3f} [{ga[1]:+.3f},{ga[2]:+.3f}] over {ga[3]} arms; without the {len(T) - len(healthy)}"
        f" arms that zero more than 15% of rollouts: {gh[0]:+.3f} [{gh[1]:+.3f},{gh[2]:+.3f}] over {gh[3]}")
    say(f"  rho(X, P) = {gp[0]:+.3f} [{gp[1]:+.3f},{gp[2]:+.3f}];  rho(VLM Q heads kept, X) = {gq[0]:+.3f}"
        f" [{gq[1]:+.3f},{gq[2]:+.3f}] over {gq[3]} arms")
    say(f"  the two lowest X: {lowest[0]}, {lowest[1]}")
    say(f"  G-arms {'PASS' if res['arms']['G_arms'] else 'FAIL'} (healthy arms only:"
        f" {'PASS' if res['arms']['G_arms_healthy'] else 'FAIL'});  G-dense {'PASS' if res['arms']['G_dense'] else 'FAIL'};"
        f"  G-pace {'PASS' if res['arms']['G_pace'] else 'FAIL'}")

    say("\n================ gates")
    for label, _, _ in G.SETS:
        R = res[label]
        say(f"  {label:10s} A1 {'PASS' if R['A1']['pass'] else 'FAIL'}   B0 {'PASS' if R['B0']['pass'] else 'FAIL'}"
            f"   B1 {'PASS' if R['B1']['pass'] else 'FAIL'}   C1 {'PASS' if R['C1']['pass'] else 'FAIL'}"
            f"   E1 {R['E1']['verdict']}")
    say(f"  pooled     D1 {'PASS' if res['D1']['pass'] else 'FAIL'}")

    plots(df, F, res, out)
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    (out / "config.json").write_text(json.dumps({
        "exp_id": args.exp_id, "arms": {k: c for k, c, _ in G.ARMS}, "sets": [s[0] for s in G.SETS],
        "step_us": STEP_US, "steps_before": PRE, "steps_after": POST, "moving_start_mps": MOVING,
        "slow_words": list(SLOW_WORDS), "n_boot": 5000, "seed": 0, "runs_root": str(G.RUNS),
        "rollouts_from": str(O / "closedloop_gap_v1/rollouts.parquet"),
    }, indent=1))
    print(f"\nwrote {out}/summary.txt, metrics.json, config.json, steps.parquet, plots/")


if __name__ == "__main__":
    main()
