"""Analysis of the open-loop ego-history perturbation (`run_history_perturb.py`).

The model's response to a perturbed history is its **predicted acceleration**: the planned speed
is v0 + cumsum(a dt) and v0 is fitted to the history by the action space, identically in every
arm. So each condition is read as the change in mean predicted acceleration over the first 2 s
against `clean`, per clip (mean over the K samples), with a clip bootstrap.

Gates (plans/2026-10-08_history-perturbation-openloop.md):
  S0  the perturbation touches only the intended history tokens; v0 is the same in every arm
  H1  dense: da(drop2) < 0, CI excluding 0
  H2  da(drop2) more negative for dense than for MoOP (clip-paired), also read for drop1
  H3  da(drop2) orders the arms by how much of the VLM was pruned, at most one adjacent inversion
  H4  dense keeps >= 2/3 of da(drop2) with the clean CoC teacher-forced
  exploratory: rise1 (mirror) and slow10 (a slower history with no braking event)

Addenda (same plan; runs `<exp-id>b` and `<exp-id>c`, each with its own clean): the histories the
simulator actually produces, replayed on recorded frames, read on the **plan slope** (planned speed
over 1-2 s minus over 0-1 s; v0 cancels, and `analyze_handover.py`'s step table holds the same two
numbers for every closed-loop plan):
  R1  dense: dslope(sim2) < 0, CI excluding 0, and more negative than MoOP's (clip-paired)
  R2  dense's dslope(sim2) as a share of the closed-loop change at the same step: >= 2/3 the history
      is sufficient, < 1/3 it is a minor channel
  R3  dslope(sim2) orders dense < MoOP 12% < 18% < 24%, at most one adjacent inversion

    .venv/bin/python experiments/evaluation/analyze_history_perturb.py --exp-id histperturb_v1
"""

import argparse
import json
from collections import Counter
from itertools import pairwise

import analyze_closedloop_gap as G
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import trim_mean, ttest_1samp, wilcoxon

O = G.O
# in the order the dose gate expects: unpruned VLMs first, then by how much of the VLM was pruned
ARMS = [("baseline", "Baseline (dense)", 0.0), ("slim_expert_znorm_r25", "expert-only pruning", 0.0),
        ("slim_dual_u20_v2", "MoOP 12%", 11.9), ("slim_dual_u30_v2", "MoOP 18%", 18.1),
        ("slim_dual_u40_v2", "MoOP 24%", 24.0), ("slim_tyr_u40_r", "Tyr 24%", 24.0)]
DOSE = ["baseline", "slim_expert_znorm_r25", "slim_dual_u20_v2", "slim_dual_u30_v2", "slim_dual_u40_v2"]
CONDS = ["drop1", "drop2", "rise1", "slow10", "drop1@cleanCoC", "drop2@cleanCoC"]
EXPL = ("decelerat", "slow", "stop", "yield", "brake")


def per_clip(rows, cond):
    """rows of one arm -> dict of (clips,) arrays for one condition, each the mean over the K samples."""
    g = [r[cond] for r in rows]
    a01, a12, a23 = (np.array([np.mean(x[k]) for x in g]) for k in ("a01", "a12", "a23"))
    prof = np.array([x["a_prof"] for x in g])                               # (clips, 64)
    return {"a02": (a01 + a12) / 2, "a23": a23, "v0": np.array([x["v0"] for x in g]),
            "v23": np.array([np.mean(x["v23"]) for x in g]), "ade": np.array([np.min(x["ade"]) for x in g]),
            # plan slope: mean planned speed over 1-2 s minus over 0-1 s. v0 cancels, and the closed-loop
            # logs hold the same two numbers, so this is the one quantity both can be compared on.
            "slope": np.array([np.mean(x["v12"]) - np.mean(x["v01"]) for x in g]),
            "a_first": prof[:, :3].mean(1), "prof": prof}


# condition -> control step after the closed-loop hand-over whose ego history it reproduces
STEP = {"drop1": 1, "sim2": 2, "sim3": 3, "sim5": 5}
METRICS = ("a02", "a_first", "slope", "v0")


FLIP = 0.5   # m/s per s: a plan slope moving by more than this is a different plan, not a nudge


def spread(x):
    """(clips,) changes -> what a mean hides: median, share below zero, 10% trimmed mean, signed-rank p,
    and the share of plans that moved by more than FLIP each way."""
    return {"median": float(np.median(x)), "below": float((x < 0).mean()), "trim10": float(trim_mean(x, 0.1)),
            "p": float(wilcoxon(x).pvalue), "down": float((x < -FLIP).mean()), "up": float((x > FLIP).mean())}


def fmt_spread(s):
    return (f"{s['median']:+.3f} ({100 * s['below']:.0f}% < 0, p={s['p']:.2g};"
            f" {100 * s['down']:.0f}% / {100 * s['up']:.0f}% beyond -/+{FLIP})")


def closed_loop_slopes(arms, rng):
    """Plan slope in closed loop (Origin150) by control step after the hand-over.

    The plan at step 0 is made on an untouched history, the one at step j on a history whose last j
    steps the simulator has slowed. Returns (summary, per-scene changes against the step-0 plan).
    """
    p = O / "handover_cause_v1/steps_arms.parquet"
    if not p.exists():
        return {}, {}
    s = pd.read_parquet(p, columns=["arm", "scene", "rollout", "k", "v", "p1", "p12"])
    s = s[s["arm"].isin(arms) & s["k"].between(0, 10)]
    w = s.pivot(index=["arm", "scene", "rollout"], columns="k", values=["v", "p1", "p12"])
    w = w[w[("v", 0)] >= 2.0]                                               # moving starts, as in open loop
    out, raw = {}, {}
    for a in arms:
        if a not in w.index.get_level_values("arm"):
            continue
        g = w.xs(a, level="arm")
        m = (g["p12"] - g["p1"]).groupby("scene").mean()                   # (scenes, 11) by step
        d = m.sub(m[0], axis=0)
        raw[a] = d.assign(v0=g["v"][0].groupby("scene").mean(),            # per scene: steps 0..10, the speed,
                          lost1=(g["v"][1] - g["v"][0]).groupby("scene").mean())   # and what the first step took
        out[a] = {"k0": float(m[0].mean()), "n": len(m), "course": [float(d[j].mean()) for j in range(11)],
                  **{f"d{j}": G.boot(d[j].dropna().to_numpy(), rng) for j in sorted(set(STEP.values()))},
                  **{f"med{j}": spread(d[j].dropna().to_numpy()) for j in sorted(set(STEP.values()))},
                  # median speed lost against the hand-over, j control steps later
                  "lost": [float((g["v"][j] - g["v"][0]).median()) for j in range(11)]}
        # where the plan starts against where the car is: its mean speed over the first second minus the
        # car's speed, on the starts that lost the full amount (the action space fits v0 to the history)
        sat = g[g["v"][1] - g["v"][0] <= -1.0]
        out[a]["start_gap"] = [float((sat["p1"][j] - sat["v"][j]).median()) for j in range(6)]
        out[a]["lost_sat"] = [float((sat["v"][j] - sat["v"][0]).median()) for j in range(6)]
    return out, raw


def boot_fine(x, rng, n=200_000, chunk=20_000):
    """G.boot with enough draws that a bound sitting at zero does not fall on either side of it by chance."""
    m = np.concatenate([x[rng.integers(0, len(x), (chunk, len(x)))].mean(1) for _ in range(n // chunk)])
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


def ratio_ci(x, y, rng, n=5000):
    """mean(x) / mean(y) for two independent samples (clips, scenes), percentile CI."""
    bx = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    by = y[rng.integers(0, len(y), (n, len(y)))].mean(1)
    lo, hi = np.percentile(bx / by, [2.5, 97.5])
    return float(x.mean() / y.mean()), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="histperturb_v1")
    ap.add_argument("--title-replay", default="Fed the ego history the simulator produces, the dense model slows on"
                    " recorded frames too")
    ap.add_argument("--title-sustained", default="A deceleration that lasts in the ego history: the dense model continues"
                    " it, MoOP 24% much less")
    ap.add_argument("--title-arms", default="Same history as 0.2 s after the hand-over: open loop orders the arms as the"
                    " simulator does")
    ap.add_argument("--title-registered", default="Registered conditions (speed held after the drop): only unpruned VLMs lean"
                    " to slowing")
    args = ap.parse_args()
    out = O / args.exp_id
    (out / "plots").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    res, lines = {}, []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    # an arm counts only once its runner has written summary_<arm>.txt: <arm>.json is checkpointed every
    # 10 clips, and reading a partial one shrinks the common clip set for every arm
    raw = {k: json.loads((out / f"{k}.json").read_text()) for k, _, _ in ARMS if (out / f"summary_{k}.txt").exists()}
    arms = [k for k, _, _ in ARMS if k in raw]
    label = {k: lab for k, lab, _ in ARMS}
    running = [k for k, _, _ in ARMS if k not in raw and (out / f"{k}.json").exists()]
    if running:
        say("not finished, left out: " + ", ".join(label[k] for k in running))
    common = sorted(set.intersection(*[{r["clip_id"] for r in raw[k]} for k in arms]))
    rows = {k: sorted((r for r in raw[k] if r["clip_id"] in set(common)), key=lambda r: r["clip_id"]) for k in arms}
    v_last = np.array([r["v_last"] for r in rows[arms[0]]])                   # (clips,)
    delta = np.array([r["delta_m"] for r in rows[arms[0]]]) / 0.1             # (clips,) m/s taken from a step
    say(f"{len(arms)} arms, {len(common)} clips in common; speed at t0 mean {v_last.mean():.2f} m/s,"
        f" taken out of one history step: mean {delta.mean():.3f} m/s (= {100 * (delta / v_last).mean():.1f}%)")
    D = {k: {c: per_clip(rows[k], c) for c in ["clean", *CONDS]} for k in arms}

    # ---- S0: what the perturbation did to the prompt and to v0 -------------------------------
    say("\n[S0] history tokens that differ from clean (index in the history span : clips), dense run")
    res["S0"] = {"tokens": {}}
    for c in ("drop1", "drop2", "rise1", "slow10"):
        cnt = Counter(tuple(r[c]["hist_tokens_changed"]) for r in rows[arms[0]])
        res["S0"]["tokens"][c] = {str(k): v for k, v in cnt.most_common(4)}
        say(f"  {c:7s} " + "  ".join(f"{list(k)} x{v}" for k, v in cnt.most_common(4)))
    say("  v0 the action space fits to the history, change against clean (m/s; the same function in every arm):")
    for c in ("drop1", "drop2", "rise1", "slow10"):
        dv = D[arms[0]][c]["v0"] - D[arms[0]]["clean"]["v0"]
        gap = max(float(np.abs(D[k][c]["v0"] - D[arms[0]][c]["v0"]).max()) for k in arms)
        res["S0"][f"dv0_{c}"] = {"mean": float(dv.mean()), "max_arm_spread": gap}
        say(f"  {c:7s} mean {dv.mean():+.3f}  (largest difference between arms {gap:.1e})")
    res["S0"]["pass"] = bool(all(res["S0"][f"dv0_{c}"]["max_arm_spread"] < 1e-3 for c in ("drop1", "drop2")))

    # ---- clean plans --------------------------------------------------------------------------
    say("\n[1] clean history: what each arm plans (clip means)")
    say(f"  {'arm':22s} accel 0-2 s   2-3 s   planned speed 2-3 s / v0   minADE@{len(rows[arms[0]][0]['clean']['ade'])}")
    res["clean"] = {}
    for k in arms:
        c = D[k]["clean"]
        res["clean"][k] = {"a02": float(c["a02"].mean()), "a23": float(c["a23"].mean()),
                           "v23_ratio": float(c["v23"].sum() / c["v0"].sum()), "ade": float(c["ade"].mean())}
        r = res["clean"][k]
        say(f"  {label[k]:22s}   {r['a02']:+.3f}    {r['a23']:+.3f}          {r['v23_ratio']:.3f}"
            f"                 {r['ade']:.3f}")

    # ---- the response -------------------------------------------------------------------------
    say("\n[2] change in mean predicted acceleration over the first 2 s against clean, m/s2 [95% clip CI]")
    say(f"  {'arm':22s} " + "".join(f"{c:>28s}" for c in CONDS))
    res["da"], res["dv23"] = {}, {}
    for k in arms:
        # 200,000 draws: dense's drop2 bound sits at zero and 5,000 put it on either side by chance
        res["da"][k] = {c: boot_fine(D[k][c]["a02"] - D[k]["clean"]["a02"], rng) for c in CONDS}
        res["dv23"][k] = {c: G.boot(D[k][c]["v23"] - D[k]["clean"]["v23"], rng) for c in CONDS}
        say(f"  {label[k]:22s} " + "".join(f"{G.fmt(res['da'][k][c], 3):>28s}" for c in CONDS))
    say("  change in planned speed 2-3 s out against clean, m/s (v0 moves this in every arm; see S0):")
    for k in arms:
        say(f"  {label[k]:22s} " + "".join(f"{G.fmt(res['dv23'][k][c], 2):>28s}" for c in CONDS))
    say("  planned speed 2-3 s out against the speed the history ends at (the closed-loop X), ratio of sums - 1:")
    res["X"] = {}
    for k in arms:
        x = {"clean": float(D[k]["clean"]["v23"].sum() / v_last.sum() - 1),
             "drop1": float(D[k]["drop1"]["v23"].sum() / (v_last - delta).sum() - 1),
             "drop2": float(D[k]["drop2"]["v23"].sum() / (v_last - delta).sum() - 1)}
        res["X"][k] = x
        say(f"  {label[k]:22s} clean {100 * x['clean']:+6.1f}%   drop1 {100 * x['drop1']:+6.1f}%"
            f"   drop2 {100 * x['drop2']:+6.1f}%")
    # post hoc: v0 falls by more than the speed taken out (the trapezoidal fit overshoots a one-step
    # change), so "holds the lower speed" and "returns to the clean plan" need a common scale
    say("  post hoc - share of the v0 change the plan has undone 2-3 s out: 1 = back on the clean plan,"
        " 0 = keeps the changed speed, < 0 = moves further the same way (ratio of sums)")
    res["undone"] = {}
    for k in arms:
        res["undone"][k] = {}
        for c in ("drop1", "drop2", "rise1", "slow10"):
            dv0 = D[k][c]["v0"] - D[k]["clean"]["v0"]
            dv = D[k][c]["v23"] - D[k]["clean"]["v23"]
            res["undone"][k][c] = G.boot_fn(np.c_[dv, dv0], lambda s: 1 - s[0] / s[1], rng)
        say(f"  {label[k]:22s} " + "   ".join(
            f"{c} {res['undone'][k][c][0]:+.2f} [{res['undone'][k][c][1]:+.2f},{res['undone'][k][c][2]:+.2f}]"
            for c in ("drop1", "drop2", "rise1", "slow10")))

    # ---- gates ---------------------------------------------------------------------------------
    say("\n================ gates")
    say(f"  S0 perturbation as intended, v0 identical across arms -> {'PASS' if res['S0']['pass'] else 'FAIL'}")
    d0 = res["da"]["baseline"]
    # H1's upper bound sits at zero (the table above is on 200,000 draws for that reason); a t-test alongside
    own = {c: D["baseline"][c]["a02"] - D["baseline"]["clean"]["a02"] for c in ("drop2", "drop1")}
    res["H1"] = {"da": d0["drop2"], "da_drop1": d0["drop1"], "pass": bool(d0["drop2"][2] < 0),
                 "t_p": {c: float(ttest_1samp(x, 0).pvalue) for c, x in own.items()}}
    say(f"  H1 dense continues the deceleration: da(drop2) = {G.fmt(d0['drop2'], 4)} m/s2 (200k draws; t-test"
        f" p={res['H1']['t_p']['drop2']:.3f})  ->  {'PASS' if res['H1']['pass'] else 'FAIL'}"
        f"      [secondary, drop1: {G.fmt(d0['drop1'], 4)}, p={res['H1']['t_p']['drop1']:.3f}]")
    if "slim_dual_u40_v2" in arms:
        res["H2"] = {}
        for c in ("drop2", "drop1"):
            dd = ((D["baseline"][c]["a02"] - D["baseline"]["clean"]["a02"])
                  - (D["slim_dual_u40_v2"][c]["a02"] - D["slim_dual_u40_v2"]["clean"]["a02"]))
            res["H2"][c] = boot_fine(dd, rng)
        res["H2"]["pass"] = bool(res["H2"]["drop2"][2] < 0)
        say(f"  H2 more than MoOP: dense - MoOP, drop2 {G.fmt(res['H2']['drop2'], 3)}, drop1"
            f" {G.fmt(res['H2']['drop1'], 3)} m/s2  ->  {'PASS' if res['H2']['pass'] else 'FAIL'}")
    dose = [k for k in DOSE if k in arms]
    vals = [res["da"][k]["drop2"][0] for k in dose]
    # the two unpruned VLMs are one level; an inversion is a later level more negative than an earlier one
    lv = [min(vals[:2]), *vals[2:]] if len(dose) == len(DOSE) else vals
    inv = sum(b < a for a, b in pairwise(lv))
    res["H3"] = {"order": dose, "da": vals, "inversions": int(inv), "pass": bool(inv <= 1)}
    say("  H3 dose: " + "  ".join(f"{label[k]} {v:+.3f}" for k, v in zip(dose, vals))
        + f"  ->  {inv} inversion(s), {'PASS' if res['H3']['pass'] else 'FAIL'}")
    keep = d0["drop2@cleanCoC"][0] / d0["drop2"][0] if d0["drop2"][0] else float("nan")
    keep1 = d0["drop1@cleanCoC"][0] / d0["drop1"][0] if d0["drop1"][0] else float("nan")
    res["H4"] = {"kept": float(keep), "kept_drop1": float(keep1), "pass": bool(keep >= 2 / 3)}
    say(f"  H4 not through the text: with the clean CoC teacher-forced dense keeps {100 * keep:.0f}% of da(drop2)"
        f"  ->  {'PASS' if res['H4']['pass'] else 'FAIL (< 1/3 would mean it runs through the text)'}"
        f"      [secondary, drop1: {100 * keep1:.0f}%; a ratio of two means, and H1 decides whether the first is one]")
    # post hoc: the gates are on clip means; the same quantities as medians with a signed-rank test,
    # and the share of clips moving each way, show whether a few clips carry them
    say("  post hoc - the gate quantities per clip: median, Wilcoxon signed-rank p, clips below / above zero")
    res["robust"] = {}
    for c in ("drop1", "drop2"):
        own = D["baseline"][c]["a02"] - D["baseline"]["clean"]["a02"]
        qs = {"dense": own}
        if "slim_dual_u40_v2" in arms:
            qs["MoOP 24%"] = D["slim_dual_u40_v2"][c]["a02"] - D["slim_dual_u40_v2"]["clean"]["a02"]
            qs["dense - MoOP 24%"] = own - qs["MoOP 24%"]
        for name, x in qs.items():
            p = float(wilcoxon(x).pvalue)
            res["robust"][f"{c}|{name}"] = {"median": float(np.median(x)), "p": p, "below": int((x < 0).sum()),
                                            "above": int((x > 0).sum()), "mean": float(x.mean()),
                                            "trim10": float(trim_mean(x, 0.1))}
            say(f"    {c} {name:17s} median {np.median(x):+.3f}  p={p:.3g}  {int((x < 0).sum())} / {int((x > 0).sum())}"
                f"   mean {x.mean():+.3f}, 10% trimmed {trim_mean(x, 0.1):+.3f}")

    # ---- exploratory ---------------------------------------------------------------------------
    say("\n[3] exploratory")
    say("  rise1 (last step longer by the same amount): change in acceleration over 0-2 s; a mirror image of drop1"
        " would be equal and opposite")
    for k in arms:
        say(f"    {label[k]:22s} rise1 {G.fmt(res['da'][k]['rise1'], 3)}   drop1 {G.fmt(res['da'][k]['drop1'], 3)}")
    say("  slow10 (whole history 10% slower): planned speed 2-3 s out against clean; 0.90 = the plan keeps the lower"
        " speed, 1.00 = it returns to the original one")
    res["slow10"] = {}
    for k in arms:
        r = float(D[k]["slow10"]["v23"].sum() / D[k]["clean"]["v23"].sum())
        res["slow10"][k] = {"v23_ratio": r, "v0_ratio": float(D[k]["slow10"]["v0"].sum() / D[k]["clean"]["v0"].sum())}
        say(f"    {label[k]:22s} {r:.3f}  (v0 {res['slow10'][k]['v0_ratio']:.3f})")
    say("  reasoning text under drop2: clips whose CoC differs from the clean one, and CoCs naming a slowdown")
    res["coc"] = {}
    for k in arms:
        ch = np.mean([r["drop2"]["gen_coc"] != r["clean"]["gen_coc"] for r in rows[k]])
        ex = {c: float(np.mean([any(w in r[c]["gen_coc"].lower() for w in EXPL) for r in rows[k]]))
              for c in ("clean", "drop2")}
        res["coc"][k] = {"changed": float(ch), **ex}
        say(f"    {label[k]:22s} text changed {100 * ch:5.1f}%   names a slowdown: clean {100 * ex['clean']:5.1f}%"
            f"  drop2 {100 * ex['drop2']:5.1f}%")

    # ---- post hoc: the same quantity in closed loop ---------------------------------------------
    cl, cl_raw = closed_loop_slopes(arms, rng)
    res["slope"] = {"closed_loop": cl, "open_loop": {}}
    say("\n[4] post hoc - against closed loop on one quantity: plan slope = planned speed over 1-2 s minus over"
        " 0-1 s (m/s per s; v0 cancels)")
    say(f"  {'arm':22s} first 0.3 s of predicted accel: drop1 - clean | slope: open loop drop1 - clean"
        f"   closed loop 0.1 s - 0 s | open loop drop2 - clean   closed loop 0.2 s - 0 s")
    for k in arms:
        o1 = G.boot(D[k]["drop1"]["slope"] - D[k]["clean"]["slope"], rng)
        o2 = G.boot(D[k]["drop2"]["slope"] - D[k]["clean"]["slope"], rng)
        f1 = G.boot(D[k]["drop1"]["a_first"] - D[k]["clean"]["a_first"], rng)
        res["slope"]["open_loop"][k] = {"drop1": o1, "drop2": o2, "a_first_drop1": f1}
        c = cl.get(k)
        say(f"  {label[k]:22s}          {G.fmt(f1, 3)}          |      {G.fmt(o1, 3)}   "
            + (f"{G.fmt(c['d1'], 3)}" if c else f"{'-':>24s}") + f"  | {G.fmt(o2, 3)}   "
            + (f"{G.fmt(c['d2'], 3)}" if c else f"{'-':>24s}"))
    say("  the same per clip / per scene - median (share below zero, signed-rank p):")
    for k in arms:
        o = {c: spread(D[k][c]["slope"] - D[k]["clean"]["slope"]) for c in ("drop1", "drop2")}
        res["slope"]["open_loop"][k]["spread"] = o
        c = cl.get(k)
        say(f"  {label[k]:22s} open loop drop1 {fmt_spread(o['drop1'])}   closed loop 0.1 s "
            + (fmt_spread(c["med1"]) if c else "-") + f"  | open loop drop2 {fmt_spread(o['drop2'])}   closed loop 0.2 s "
            + (fmt_spread(c["med2"]) if c else "-"))
    if "baseline" in cl:
        o, c = res["slope"]["open_loop"]["baseline"], cl["baseline"]
        res["slope"]["dense_open_over_closed"] = {"step1": o["drop1"][0] / c["d1"][0], "step2": o["drop2"][0] / c["d2"][0]}
        say(f"  dense, open loop as a share of closed loop: {100 * o['drop1'][0] / c['d1'][0]:.0f}% at the first step,"
            f" {100 * o['drop2'][0] / c['d2'][0]:.0f}% at the second (closed loop: Origin150 renders, {c['n']} scenes;"
            f" open loop: Test500 frames, {len(common)} clips)")
        say("  but drop2 holds the lower speed, and the simulator's history at 0.2 s keeps slowing; median speed lost"
            " against the hand-over by control step (dense car): "
            + " ".join(f"{x:+.2f}" for x in c["lost"][1:6]))
        # why the second step is slower in every arm: the plan starts from the action space's fitted v0, and
        # that fit reads a one-step drop as a trend
        dv0 = res["S0"]["dv0_drop1"]["mean"]
        say(f"  the second slow step is mechanical: the action space's v0 moves by {dv0:+.2f} m/s for a"
            f" {-delta.mean():+.2f} m/s step (open loop, every arm), so the plan starts below the car;"
            " in the simulator, plan mean speed over its first second minus the car's speed (median, starts that lost"
            " >= 1.0 m/s), steps 0-5:")
        for k in arms:
            if k in cl:
                say(f"    {label[k]:22s} " + " ".join(f"{x:+.2f}" for x in cl[k]["start_gap"])
                    + "   | car against the hand-over: " + " ".join(f"{x:+.2f}" for x in cl[k]["lost_sat"]))

    # ---- addenda: the simulator's own histories, replayed on recorded frames ---------------------
    # separate runs, each with its own clean; the main run's drop1 / drop2 belong to the same series
    sup, prof = {}, {}                         # (cond, arm) -> clip-indexed changes against clean / (64,) profile
    for k in arms:
        ids = [r["clip_id"] for r in rows[k]]
        for c in ("drop1", "drop2"):
            sup[(c, k)] = {m: pd.Series(D[k][c][m] - D[k]["clean"][m], index=ids) for m in METRICS}
    drift = {}
    for sfx in ("b", "c"):
        for k in arms:
            d = O / f"{args.exp_id}{sfx}"
            if not (d / f"summary_{k}.txt").exists():
                continue
            rr = json.loads((d / f"{k}.json").read_text())
            ids = [r["clip_id"] for r in rr]
            base = per_clip(rr, "clean")
            # same clips, seeds and card type as the main run, so the two clean plans should coincide
            main = pd.Series(D[k]["clean"]["a02"], index=[r["clip_id"] for r in rows[k]])
            both = pd.Series(base["a02"], index=ids).sub(main).dropna()
            drift[f"{sfx}:{k}"] = float(both.abs().max()) if len(both) else float("nan")
            prof[(f"clean:{sfx}", k)] = base["prof"].mean(0)
            for c in [c for c in rr[0] if c != "clean" and isinstance(rr[0][c], dict)]:
                x = per_clip(rr, c)
                sup[(c, k)] = {m: pd.Series(x[m] - base[m], index=ids) for m in METRICS}
                prof[(c, k)] = x["prof"].mean(0)
    order = [c for c in ("drop1", "sim2", "sim2@cleanCoC", "sim3", "sim5", "drop2", "brake5")
             if any((c, k) in sup for k in arms)]
    if len(order) > 2:
        what = {"drop1": "last step -1.06 (as 0.1 s after)", "sim2": "-1.06, -1.28 (as 0.2 s after)",
                "sim2@cleanCoC": "... with the clean CoC kept", "sim3": "-1.06, -1.28, -1.38 (0.3 s, dense car)",
                "sim5": "-1.06 ... -1.80 (0.5 s, dense car)", "drop2": "-1.06, -1.06 (speed held)",
                "brake5": "uniform -3 m/s2 for 0.5 s"}
        res["addenda"] = {"clean_drift_max_abs_a02": drift, "rows": {}}
        say("\n[5] addenda - the simulator's own histories on recorded frames, change against the same run's clean")
        say("  clean plan in the addendum runs against the main run, largest |difference| in accel 0-2 s: "
            + ", ".join(f"{k} {v:.1e}" for k, v in drift.items()))
        say(f"  {'history (m/s off the last steps)':40s} {'arm':18s} {'accel 0-2 s':>24s} {'first 0.3 s':>24s}"
            f" {'plan slope':>24s} {'closed loop, same step':>24s}  open/closed")
        for c in order:
            for k in arms:
                if (c, k) not in sup:
                    continue
                s = sup[(c, k)]
                r = {m: G.boot(s[m].to_numpy(), rng) for m in ("a02", "a_first", "slope")}
                row = {**r, "n": len(s["a02"]), "dv0": float(s["v0"].mean()), "spread": spread(s["slope"].to_numpy())}
                # the closed-loop counterpart exists where the arm really had this history: every arm up to
                # step 2, the dense car alone after that
                j = STEP.get(c.split("@")[0])
                same = j is not None and k in cl and (j <= 2 or k == "baseline")
                tail = f"{'-':>24s}"
                if same:
                    row["closed"] = cl[k][f"d{j}"]
                    tail = f"{G.fmt(cl[k][f'd{j}'], 3):>24s}"
                    if k == "baseline":
                        row["share"] = ratio_ci(s["slope"].to_numpy(), cl_raw[k][j].dropna().to_numpy(), rng)
                        tail += f"  {100 * row['share'][0]:.0f}% [{100 * row['share'][1]:.0f},{100 * row['share'][2]:.0f}]"
                res["addenda"]["rows"][f"{c}|{k}"] = row
                say(f"  {what[c]:40s} {label[k]:18s} {G.fmt(r['a02'], 3):>24s} {G.fmt(r['a_first'], 3):>24s}"
                    f" {G.fmt(r['slope'], 3):>24s} {tail}")
                say(f"  {'':40s} {f'v0 {row["dv0"]:+.2f} m/s':18s} {'plan slope per clip: median':>49s}"
                    f" {fmt_spread(row['spread']):>24s} " + (f"{fmt_spread(cl[k][f'med{j}']):>24s}" if same else ""))
        pair = {}
        for c in order:
            if ("baseline" in arms and "slim_dual_u40_v2" in arms and (c, "baseline") in sup
                    and (c, "slim_dual_u40_v2") in sup):
                pair[c] = {m: G.boot((sup[(c, "baseline")][m] - sup[(c, "slim_dual_u40_v2")][m]).dropna().to_numpy(), rng)
                           for m in ("a02", "a_first", "slope")}
        res["addenda"]["dense_minus_moop"] = pair
        say("  dense - MoOP 24%, clip-paired:")
        for c, p in pair.items():
            say(f"    {what[c]:40s} accel 0-2 s {G.fmt(p['a02'], 3)}   first 0.3 s {G.fmt(p['a_first'], 3)}"
                f"   plan slope {G.fmt(p['slope'], 3)}")
        if ("sim2", "baseline") in sup and ("drop2", "baseline") in sup:
            d = (sup[("sim2", "baseline")]["slope"] - sup[("drop2", "baseline")]["slope"]).dropna().to_numpy()
            res["addenda"]["sim2_minus_drop2_dense"] = G.boot(d, rng)
            say(f"  dense, sim2 - drop2 (the second step still slowing, 0.22 m/s): plan slope"
                f" {G.fmt(res['addenda']['sim2_minus_drop2_dense'], 3)}")
        # exploratory: does the response depend on what the scene calls for? Bucket = what the human did next
        bucket = pd.Series({r["clip_id"]: r["bucket"] for r in rows[arms[0]]})
        res["addenda"]["by_bucket"] = {}
        say("  exploratory - first 0.3 s of predicted acceleration against clean, by what the human did next:"
            " mean (share of clips below zero)")
        for c in [c for c in ("sim2", "sim5", "brake5") if any((c, k) in sup for k in arms)]:
            for k in arms:
                if (c, k) not in sup:
                    continue
                g = sup[(c, k)]["a_first"].groupby(bucket.reindex(sup[(c, k)]["a_first"].index))
                res["addenda"]["by_bucket"][f"{c}|{k}"] = {b: {"mean": float(x.mean()), "below": float((x < 0).mean()),
                                                              "n": len(x)} for b, x in g}
                say(f"    {c:7s} {label[k]:18s} " + "   ".join(
                    f"{b} (n={len(x)}) {x.mean():+.2f} ({100 * (x < 0).mean():.0f}%)" for b, x in g))

        # like for like: the two sides are different scene sets, and the open-loop cut is the full 1.06 m/s on
        # every clip. So: closed-loop scenes that lost at least 1.0 m/s in the first step (none of them starts
        # below 6 m/s), split by the one covariate both sides have - the speed the car starts from
        edges, names = [6, 12, 20, 1e3], ["6-12", "12-20", "20+"]
        res["addenda"]["by_speed"] = {}
        say("  post hoc - like for like, by the speed the car starts from (m/s): change in plan slope, mean (median, n);"
            " closed loop = scenes that lost >= 1.0 m/s in the first step")
        for c in [c for c in ("drop1", "sim2") if any((c, k) in sup for k in arms)]:
            j = STEP[c]
            for k in arms:
                if (c, k) not in sup or k not in cl_raw:
                    continue
                s = sup[(c, k)]["slope"]
                vo = pd.Series(v_last, index=[r["clip_id"] for r in rows[k]]).reindex(s.index)
                cv = cl_raw[k][cl_raw[k]["lost1"] <= -1.0]
                og = s.groupby(pd.cut(vo, edges, labels=names, right=False), observed=False)
                cg = cv[j].groupby(pd.cut(cv["v0"], edges, labels=names, right=False), observed=False)
                rew = float((og.mean() * cg.size() / cg.size().sum()).sum())      # open loop at the closed-loop mix
                closed = float(cv[j].mean())
                res["addenda"]["by_speed"][f"{c}|{k}"] = {
                    "open_mean": og.mean().to_dict(), "open_median": og.median().to_dict(), "n_open": og.size().to_dict(),
                    "closed_mean": cg.mean().to_dict(), "closed_median": cg.median().to_dict(),
                    "n_closed": cg.size().to_dict(), "open_at_closed_mix": rew, "closed": closed}
                say(f"    {c:6s} {label[k]:18s} "
                    + "   ".join(f"{b}: open {og.mean()[b]:+.3f} ({og.median()[b]:+.3f}, {og.size()[b]})"
                                 f" closed {cg.mean()[b]:+.3f} ({cg.median()[b]:+.3f}, {cg.size()[b]})" for b in names)
                    + f"   | open at the closed-loop speed mix {rew:+.3f}, closed {closed:+.3f}"
                    + (f" = {100 * rew / closed:.0f}%" if k == "baseline" else ""))

        say("\n================ addendum gates (registered before the sim2 / sim3 runs)")
        if ("sim2", "baseline") in sup:
            r0 = res["addenda"]["rows"]["sim2|baseline"]
            ok1 = r0["slope"][2] < 0 and "sim2" in pair and pair["sim2"]["slope"][2] < 0
            res["R1"] = {"dense": r0["slope"], "dense_minus_moop": pair.get("sim2", {}).get("slope"), "pass": bool(ok1)}
            say(f"  R1 reproduction at the faithful history: dense dslope(sim2) = {G.fmt(r0['slope'], 3)}, dense - MoOP"
                f" {G.fmt(pair['sim2']['slope'], 3) if 'sim2' in pair else '-'}  ->  {'PASS' if ok1 else 'FAIL'}")
            if "share" in r0:
                sh = r0["share"]
                verdict = ("history sufficient (>= 2/3)" if sh[0] >= 2 / 3 else
                           "minor channel (< 1/3)" if sh[0] < 1 / 3 else "both: between 1/3 and 2/3")
                res["R2"] = {"share": sh, "verdict": verdict}
                say(f"  R2 share of the closed-loop change the history alone carries: {100 * sh[0]:.0f}%"
                    f" [{100 * sh[1]:.0f},{100 * sh[2]:.0f}]  ->  {verdict}")
            # registered on dense and the three MoOP ratios; expert-only, where run, is the same level as dense
            dose2 = [k for k in DOSE if ("sim2", k) in sup]
            v2 = [res["addenda"]["rows"][f"sim2|{k}"]["slope"][0] for k in dose2]
            ready = {"baseline", "slim_dual_u20_v2", "slim_dual_u30_v2", "slim_dual_u40_v2"} <= set(dose2)
            lv2 = [min(v2[:2]), *v2[2:]] if "slim_expert_znorm_r25" in dose2 else v2
            inv2 = sum(b < a for a, b in pairwise(lv2))
            res["R3"] = {"order": dose2, "dslope": v2, "inversions": int(inv2), "pass": bool(ready and inv2 <= 1)}
            say("  R3 dose at the faithful history: " + "  ".join(f"{label[k]} {v:+.3f}" for k, v in zip(dose2, v2))
                + (f"  ->  {inv2} inversion(s), {'PASS' if res['R3']['pass'] else 'FAIL'}" if ready
                   else f"  ->  {len(dose2)} arms run so far, not judged"))

    # ---- plots ---------------------------------------------------------------------------------
    t = (np.arange(64) + 1) * 0.1
    show = [k for k in ("baseline", "slim_dual_u40_v2") if k in arms]
    fig, axes = plt.subplots(1, len(show), figsize=(6.2 * len(show), 4.3), sharey=True, squeeze=False)
    for ax, k in zip(axes[0], show):
        for c, col, lab in [("clean", G.MUTED, "clean history"), ("drop1", G.CAT[0], "last step 1.06 m/s slower"),
                            ("drop2", G.CAT[1], "last two steps slower")]:
            ax.plot(t[:40], D[k][c]["prof"].mean(0)[:40], color=col, linewidth=2.2, label=lab, solid_capstyle="round")
        ax.axhline(0, color=G.MUTED, linewidth=1)
        ax.set_title(label[k], loc="left", fontsize=10.5)
        ax.set_xlabel("seconds ahead in the plan")
        ax.grid(axis="y", color=G.GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
    axes[0][0].set_ylabel("predicted acceleration (m/s²), mean over clips")
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.suptitle("Same scene, same seed: only the last steps of the ego history differ", x=0.012, ha="left",
                 fontsize=12, fontweight="bold")
    fig.legend(h, lab, loc="upper left", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.008, 0.945))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out / "plots/accel_profile.png", dpi=150)
    plt.close(fig)

    y = np.arange(len(arms))[::-1]
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    for j, (c, col, m, lab) in enumerate([("drop1", G.CAT[0], "o", "last step slower"),
                                          ("drop2", G.CAT[1], "s", "last two steps slower"),
                                          ("drop2@cleanCoC", G.CAT[2], "D", "... with the clean CoC kept")]):
        for yi, k in zip(y, arms):
            ci = res["da"][k][c]
            yy = yi + (1 - j) * 0.22
            ax.plot(ci[1:], [yy, yy], color=col, linewidth=2, solid_capstyle="round")
            ax.plot(ci[0], yy, m, color=col, markersize=7.5, markeredgecolor=G.BG, markeredgewidth=1.5,
                    label=lab if k == arms[0] else None)
    for yi, k in zip(y, arms):
        ax.text(ax.get_xlim()[1], yi, f"  {res['da'][k]['drop2'][0]:+.2f}", va="center", fontsize=9)
    ax.axvline(0, color=G.MUTED, linewidth=1)
    ax.set_yticks(y, [label[k] for k in arms])
    ax.set_xlabel("change in predicted acceleration over the first 2 s against the clean history (m/s², 95% CI;"
                  " number: two steps)")
    ax.grid(axis="x", color=G.GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    h, lab = ax.get_legend_handles_labels()
    fig.suptitle(args.title_registered, x=0.012, ha="left", fontsize=12, fontweight="bold")
    fig.legend(h, lab, loc="upper left", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.008, 0.945))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out / "plots/response_by_arm.png", dpi=150)
    plt.close(fig)

    pair_arms = [(k, col) for k, col in (("baseline", G.CAT[0]), ("slim_dual_u40_v2", G.CAT[1])) if k in arms]
    if cl and "addenda" in res:
        # the simulator's histories on recorded frames (points) against the closed-loop plans themselves (lines)
        fig, ax = plt.subplots(figsize=(9.8, 5.0))
        for k, col in pair_arms:
            if k not in cl:
                continue
            ax.plot(np.arange(9) / 10, cl[k]["course"][:9], color=col, linewidth=2.2, solid_capstyle="round")
            ax.text(0.81, cl[k]["course"][8], f" {label[k]}", color=col, va="center", fontsize=9.5, fontweight="bold")
            for c, j in STEP.items():
                row = res["addenda"]["rows"].get(f"{c}|{k}")
                if row is None:
                    continue
                own = j <= 2 or k == "baseline"            # the arm really had this history in closed loop
                ax.plot([j / 10, j / 10], row["slope"][1:], color=col, linewidth=2, solid_capstyle="round")
                ax.plot(j / 10, row["slope"][0], "o", markersize=8.5, markeredgewidth=1.8,
                        color=col if own else G.BG, markeredgecolor=G.BG if own else col)
        ax.axhline(0, color=G.MUTED, linewidth=1)
        ax.plot([], [], color=G.INK, linewidth=2.2, label="closed loop: the plans alpasim logged (Origin150)")
        ax.plot([], [], "o", color=G.INK, markersize=8, markeredgecolor=G.BG, markeredgewidth=1.5,
                label="open loop: that step's ego history on recorded frames (Test500, 95% CI)")
        ax.plot([], [], "o", color=G.BG, markersize=8, markeredgecolor=G.INK, markeredgewidth=1.8,
                label="open loop, given the dense car's history")
        ax.set_xlim(-0.02, 1.02)
        ax.set_xticks(np.arange(9) / 10)
        ax.set_xlabel("time since the hand-over (s) = how many 0.1 s steps of the ego history the simulator has slowed")
        ax.set_ylabel("change in plan slope against the plan\non the untouched history (m/s per s)")
        ax.grid(axis="y", color=G.GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        h, lab = ax.get_legend_handles_labels()
        fig.suptitle(args.title_replay, x=0.012, ha="left", fontsize=12, fontweight="bold")
        fig.legend(h, lab, loc="upper left", ncol=1, frameon=False, fontsize=9, bbox_to_anchor=(0.008, 0.945))
        fig.tight_layout(rect=(0, 0, 1, 0.83))
        fig.savefig(out / "plots/open_vs_closed.png", dpi=150)
        plt.close(fig)

        # every arm at the one step where they all had the same history in the simulator
        ks = [k for k in arms if f"sim2|{k}" in res["addenda"]["rows"]]
        if ks:
            y = np.arange(len(ks))[::-1]
            fig, ax = plt.subplots(figsize=(9.8, 4.9))
            for j, (name, col, m, lab) in enumerate([
                    ("sim2", G.CAT[0], "o", "open loop: that history on recorded frames (Test500)"),
                    ("sim2@cleanCoC", G.CAT[2], "D", "... with the clean CoC kept"),
                    ("closed", G.CAT[1], "s", "closed loop: the plans alpasim logged 0.2 s after the hand-over (Origin150)")]):
                for yi, k in zip(y, ks):
                    ci = (cl[k]["d2"] if k in cl else None) if name == "closed" else \
                        res["addenda"]["rows"].get(f"{name}|{k}", {}).get("slope")
                    if ci is None:
                        continue
                    yy = yi + (1 - j) * 0.22
                    ax.plot(ci[1:], [yy, yy], color=col, linewidth=2, solid_capstyle="round")
                    ax.plot(ci[0], yy, m, color=col, markersize=7.5, markeredgecolor=G.BG, markeredgewidth=1.5,
                            label=lab if k == ks[0] else None)
            ax.axvline(0, color=G.MUTED, linewidth=1)
            ax.set_yticks(y, [label[k] for k in ks])
            ax.set_xlabel("change in plan slope against the plan on the untouched history (m/s per s, 95% CI)")
            ax.grid(axis="x", color=G.GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            ax.tick_params(length=0)
            h, lab = ax.get_legend_handles_labels()
            fig.suptitle(args.title_arms, x=0.012, ha="left", fontsize=12, fontweight="bold")
            fig.legend(h, lab, loc="upper left", ncol=1, frameon=False, fontsize=9, bbox_to_anchor=(0.008, 0.945))
            fig.tight_layout(rect=(0, 0, 1, 0.83))
            fig.savefig(out / "plots/faithful_history_by_arm.png", dpi=150)
            plt.close(fig)

        lines_ = [(c, col, lab) for c, col, lab in [("sim2", G.CAT[0], "two steps slower, as 0.2 s after the hand-over"),
                                                   ("sim5", G.CAT[1], "five steps slower, as 0.5 s after (dense car)"),
                                                   ("brake5", G.CAT[2], "uniform braking at 3 m/s² for 0.5 s")]
                  if any((c, k) in prof for k, _ in pair_arms)]
        fig, axes = plt.subplots(1, len(pair_arms), figsize=(6.2 * len(pair_arms), 4.5), sharey=True, squeeze=False)
        for ax, (k, _) in zip(axes[0], pair_arms):
            ax.plot(t[:40], D[k]["clean"]["prof"].mean(0)[:40], color=G.MUTED, linewidth=2.2, label="clean history",
                    solid_capstyle="round")
            for c, col, lab in lines_:
                if (c, k) in prof:
                    ax.plot(t[:40], prof[(c, k)][:40], color=col, linewidth=2.2, label=lab, solid_capstyle="round")
            ax.axhline(0, color=G.MUTED, linewidth=1)
            ax.set_title(label[k], loc="left", fontsize=10.5)
            ax.set_xlabel("seconds ahead in the plan")
            ax.grid(axis="y", color=G.GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            ax.tick_params(length=0)
        axes[0][0].set_ylabel("predicted acceleration (m/s²), mean over clips")
        h, lab = axes[0][0].get_legend_handles_labels()
        fig.suptitle(args.title_sustained, x=0.012, ha="left", fontsize=12, fontweight="bold")
        fig.legend(h, lab, loc="upper left", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.008, 0.945))
        fig.tight_layout(rect=(0, 0, 1, 0.84))
        fig.savefig(out / "plots/accel_profile_sustained.png", dpi=150)
        plt.close(fig)

    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    (out / "config.json").write_text(json.dumps({
        "exp_id": args.exp_id, "arms": arms, "n_clips": len(common), "conditions": ["clean", *CONDS],
        "primary": "mean predicted acceleration over 0-2 s, perturbed minus clean, clip bootstrap",
        "addenda": {"quantity": "plan slope = planned speed over 1-2 s minus over 0-1 s, perturbed minus clean",
                    "runs": {f"{args.exp_id}{s}": {f.stem.removeprefix("config_"): json.loads(f.read_text())["conditions"]
                                                    for f in sorted((O / f"{args.exp_id}{s}").glob("config_*.json"))}
                             for s in ("b", "c") if (O / f"{args.exp_id}{s}").exists()},
                    "closed_loop_source": "outputs/handover_cause_v1/steps_arms.parquet (analyze_handover.py),"
                                          " Origin150, starts moving at >= 2 m/s"},
        "n_boot": {"main table, H1, H2": 200_000, "elsewhere": 5000}, "flip_threshold": FLIP, "seed": 0,
        "per_arm_config": [f"config_{k}.json" for k in arms],
    }, indent=1))
    print(f"\nwrote {out}/summary.txt, metrics.json, config.json, plots/")


if __name__ == "__main__":
    main()
