"""CoC-action consistency: the Alpamayo-R1 RL reward, measured on stored rollouts.

Alpamayo-R1 (arXiv 2511.00088, S5.3) scores a GRPO sample with `r_consistency`: derive
meta-actions from the predicted trajectory by rule (longitudinal + lateral), parse the
intended behaviour out of the reasoning trace, and give 1 iff both axes match --
unparseable counts as 0. This script asks whether that quantity is worth adding to the
pruning criterion, using only text and metrics that `run_baseline.py` already stores.

Two facts make the reward reproducible here without touching a GPU:

- The released 10B has **no meta-action tokens**. `token_utils.extract_text_tokens`
  looks for `<|meta_action_start|>`/`<|meta_action_end|>`, but `base_model`'s
  SPECIAL_TOKENS_KEYS has no such key (the slots read `_padding_0` .. `_padding_8`) and
  the Cosmos-Reason2 tokenizer has no such token, so the extractor returns "" always.
  A "meta-action segment NLL" objective therefore has nothing to attach to.
- It does not need one: the CoC is a single templated sentence whose HEAD CLAUSE is
  the meta-action ("Keep lane since ...", "Stop for the red light", "Turn right at
  ..."). Over 1,000 stored baseline rollouts the first two words take 36 distinct
  values, 90.1% of them in the top 15 -- the paper's meta-action vocabulary, inlined.

What is *not* stored is the predicted waypoints, so `--mode agree` substitutes the
GT-derived `bucket` (identical for every arm) for the model's own trajectory. That
makes it CoC-vs-GROUND-TRUTH agreement -- reasoning accuracy -- not the paper's
CoC-vs-OWN-TRAJECTORY self-consistency. Since `bucket` is constant per clip, every
arm-to-arm movement is pure CoC drift, which is what the criterion question needs.
For the paper's exact metric, add `el.bucket(pred_k[j])` to the record in
`run_baseline.py` (pred_k is already in scope, so it costs no compute) and re-run.

Only unambiguous heads are mapped; "Keep distance", "Adapt speed" and "Nudge *" are
genuinely ambiguous on the longitudinal axis and are left out rather than forced, so
coverage is ~41% and only the paired differences are meaningful. Scoring unmapped as
a miss (`--score uncond`) follows the paper; `--score cond` conditions on the mapped
subset and is reported alongside because degenerate CoCs leave that subset and would
otherwise flatter a collapsing arm.

`--mode couple` groups clips by whether the arm's CoC text differs from the reference
arm's -- a property of the PAIR, not of either arm's own score, so it is not the
subgroup trap of `plans/2026-09-04_union-step-criterion.md`. Baseline's own minADE is
printed per group because clip difficulty is the confound that remains.

Usage:
  .venv/bin/python experiments/evaluation/coc_action_consistency.py \
      --mode agree --score uncond baseline_ada_ps_indist dual_u40_v2_ps_indist ...
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path("/mnt/nvme1n1/ad_vla/outputs/chan")

# Head clause -> the eval_lib.bucket class it asserts. Ambiguous heads are absent on
# purpose: "Keep distance" is a following behaviour that is cruise or decel depending
# on the lead, and "Nudge left" is a lateral deviation below the 30 deg turn threshold.
HEAD2BUCKET = {
    "Stop to": "decel_stop", "Stop for": "decel_stop", "Stop due": "decel_stop",
    "Stop at": "decel_stop", "Stop behind": "decel_stop",
    "Remain stopped": "decel_stop", "Stay stopped": "decel_stop",
    "Slow down": "decel_stop", "Decelerate to": "decel_stop",
    "Yield to": "decel_stop", "Yield due": "decel_stop", "Wait for": "decel_stop",
    "Turn left": "turn", "Turn right": "turn",
    "Accelerate to": "accel", "Resume speed": "accel",
    "Keep lane": "cruise", "Keep speed": "cruise", "Maintain lane": "cruise",
}


def head(t):
    """The meta-action clause: the first two words of the CoC sentence."""
    return " ".join(str(t).split()[:2])


def rows(d):
    out = {}
    for p in glob.glob(str(OUT / d / "*.json")):
        if Path(p).name in ("config.json", "metrics.json"):
            continue
        with open(p) as fh:
            data = json.load(fh)
        if isinstance(data, list):
            for r in data:
                if "clip_id" in r and "gen_coc" in r:
                    out[r["clip_id"]] = r
    return out


def at6(r, key="ade_rollout_k"):
    return float(np.min(np.asarray(r[key], float)[:6]))


def mode_agree(tab, arms, ids, uncond):
    buckets = {c: tab[arms[0]][c]["bucket"] for c in ids}
    print("GT bucket mix:", {b: sum(v == b for v in buckets.values())
                             for b in sorted(set(buckets.values()))})
    hit, cov = {}, {}
    for a in arms:
        v = tab[a]
        b = [HEAD2BUCKET.get(head(v[c]["gen_coc"])) for c in ids]
        cov[a] = np.array([x is not None for x in b])
        ok = np.array([x == buckets[c] for x, c in zip(b, ids)], float)
        hit[a] = ok if uncond else np.where(cov[a], ok, np.nan)

    lbl = "agree(all)" if uncond else "agree|mapped"
    print(f"\n{'arm':26s} {'coverage':>9s} {lbl:>13s} {'degen':>7s} {'minADE@6':>9s}")
    for a in arms:
        ade = np.array([at6(tab[a][c]) for c in ids])
        dg = np.mean([tab[a][c]["coc_degenerate"] for c in ids])
        val = hit[a].mean() if uncond else np.nanmean(hit[a][cov[a]])
        print(f"{a:26s} {100 * cov[a].mean():8.1f}% {100 * val:12.1f}% "
              f"{100 * dg:6.1f}% {np.median(ade):9.4f}")

    base = arms[0]
    scope = "ALL clips (unparseable = 0, as in the paper)" if uncond else \
            "clips both arms map"
    print(f"\npaired vs {base} -- McNemar, {scope}; exact binomial")
    for a in arms[1:]:
        both = np.ones(len(ids), bool) if uncond else (cov[base] & cov[a])
        x, y = hit[base][both], hit[a][both]
        b01 = int(np.sum((x == 1) & (y == 0)))
        b10 = int(np.sum((x == 0) & (y == 1)))
        p = stats.binomtest(b10, b01 + b10, 0.5).pvalue if b01 + b10 else float("nan")
        same = np.mean([tab[base][c]["gen_coc"] == tab[a][c]["gen_coc"] for c in ids])
        print(f"  {a:26s} n={int(both.sum()):4d}  base-only {b01:3d}  arm-only {b10:3d}  "
              f"d={100 * (y.mean() - x.mean()):+5.1f}pp  p={p:8.3g}  "
              f"identical CoC {100 * same:5.1f}%")


def mode_couple(tab, arms, ids):
    base = arms[0]
    print(f"\ncoupling: is the trajectory damage larger where the CoC changed?  "
          f"reference = {base}")
    print(f"{'arm':26s} {'n_chg':>6s} {'d|CoC changed':>14s} {'d|CoC same':>12s} "
          f"{'p':>9s} {'ref ADE chg/same':>19s}")
    for a in arms[1:]:
        chg = np.array([tab[base][c]["gen_coc"] != tab[a][c]["gen_coc"] for c in ids])
        d = np.array([at6(tab[a][c]) - at6(tab[base][c]) for c in ids])
        b = np.array([at6(tab[base][c]) for c in ids])
        p = (stats.mannwhitneyu(d[chg], d[~chg]).pvalue
             if chg.any() and (~chg).any() else float("nan"))
        print(f"{a:26s} {int(chg.sum()):6d} {np.median(d[chg]):+14.4f} "
              f"{np.median(d[~chg]):+12.4f} {p:9.3g} "
              f"{np.median(b[chg]):8.3f} /{np.median(b[~chg]):8.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="output dirs; the first is the reference")
    ap.add_argument("--mode", choices=("agree", "couple"), default="agree")
    ap.add_argument("--score", choices=("uncond", "cond"), default="uncond")
    args = ap.parse_args()

    tab = {a: rows(a) for a in args.arms}
    arms = [a for a in args.arms if tab.get(a)]
    tab = {a: tab[a] for a in arms}
    ids = sorted(set.intersection(*[set(v) for v in tab.values()]))
    print(f"arms {len(arms)}  clips paired {len(ids)}\n")

    if args.mode == "agree":
        mode_agree(tab, arms, ids, args.score == "uncond")
    else:
        mode_couple(tab, arms, ids)


if __name__ == "__main__":
    main()
