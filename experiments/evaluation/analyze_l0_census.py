"""Which earlier conclusions were riding on the layer-0 head choice? No GPU.

`analyze_l0_exempt.py` showed that which 13 of layer 0's 32 Q heads a dual_u40_v2 arm
drops decides whether it collapses, and that the criterion cannot see it. Every earlier
comparison whose arms had their importance measured separately therefore carries a hidden
factor. This reads it off the stored kept sets:

  1. census    every slim_meta.json of the u40 VLM-only shape (19 Q heads and 7,390 MLP
               channels in all 36 layers, expert untouched), its layer-0 dropped set, and
               its test500 / val500 minADE@6 wherever an evaluation of it is stored
               (found through each evaluation's config.json "model" field).
  2. signal    does the layer-0 set alone predict the score? `n_off` = dropped heads that
               the shipped arm keeps; per-head kept-vs-dropped contrasts; and a ridge
               fit of log(minADE) on the head indicators, trained on one group of arms and
               tested on a disjoint one (the 64 half-split arms of the clip-influence
               study, scored on 150 val clips, are the held-out group).
  3. re-read   the families behind earlier conclusions -- draw rule (rd / se / su / nt),
               calibration size (st2000), hard pool (tr), clip weighting (dualsafe),
               estimator variants -- with their layer-0 sets side by side.

A prediction of a collapse from the layer-0 set is a correlation across arms, not the
exemption test: it says which conclusions need re-measuring, not what they would become.

Usage:
  python experiments/evaluation/analyze_l0_census.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

REPO = Path(__file__).resolve().parents[2]
O = REPO / "outputs"
BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
C1, C3, C4 = "#2a78d6", "#e87ba4", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})
K = 6
# families whose arms share the dual criterion and differ only in what the importance
# was measured on / how -- the comparisons the layer-0 lottery can contaminate
FAMILIES = {
    "seed (calib_100)": ["dual_u40_v2", "dual_ada_u40_v2", "dual_seed_ref"]
                        + [f"dual_seed_{a}" for a in ("c1", "c2", "c3", "c4", "n1", "n2", "n3", "n4")],
    "draw, greedy (nt)": [f"dual_nt_{x}" for x in "abcde"],
    "draw, random (rd)": [f"dual_rd_{x}" for x in "abc"],
    "draw, strat-eval (se)": [f"dual_se_{x}" for x in "abc"],
    "draw, strat-uniform (su)": [f"dual_su_{x}" for x in "abc"],
    "hard pool (tr)": [f"dual_tr_{x}" for x in "abcde"],
    "size ladder": ["dual_nt_c200", "dual_nt_c300", "dual_nt_c500", "dual_tr_c200",
                    "dual_tr_c300", "dual_tr_c500", "dual_st2000", "dual_st2000b",
                    "dual_u40_st2000"],
    "calibration source": ["dual_u40_ctl", "dual_u40_ood", "dual_u40_mix"],
    "estimator / weighting": ["dualsafe_u40_v2", "dualfix_u40_v2", "maxstep11_u40_v2",
                              "meandual_u40_v2", "dualself_u40_v2", "dualbestn6_u40_v2",
                              "dualgtk3_u40_v2", "dualgtk5_u40_v2", "dualgtk10_u40_v2",
                              "dual2nd_u40_v2", "dual_u40_it3", "dualw_lin_u40_v2",
                              "dualw_h19_u40_v2", "dualfm10_u40_v2", "dualfm20_u40_v2"],
}


def u40_meta(d):
    p = d / "slim_meta.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    v, e = m.get("vlm"), m.get("expert")
    if not v or len(v) != 36 or m.get("kvonly_layers"):
        return None
    if not all(len(l["q"]) == 19 and len(l["mlp"]) == 7390 for l in v):
        return None
    if e and not all(len(l["q"]) == 16 for l in e):
        return None
    return v


def main():
    out = O / "swap_masks"
    (out / "plots").mkdir(parents=True, exist_ok=True)
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    # ---- 1. census of kept sets ----
    meta = {}
    for d in sorted(O.glob("slim_*")):
        if d.name.startswith("slim_swap_"):
            continue
        v = u40_meta(d)
        if v is not None:
            meta[d.name[5:]] = v
    ship0 = set(range(32)) - set(meta["dual_u40_v2"][0]["q"])
    drop0 = {n: set(range(32)) - set(v[0]["q"]) for n, v in meta.items()}
    say(f"{len(meta)} kept sets of the u40 VLM-only shape; shipped layer 0 drops {sorted(ship0)}")

    # evaluations, located through their config's model field
    score = {}                                  # name -> {set: (mean, median, n)}
    for cfgp in O.glob("*/config.json"):
        try:
            c = json.loads(cfgp.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        model, st = str(c.get("model", "")), c.get("set")
        if "slim_" not in model or st not in ("test", "indist") or c.get("manifest") or c.get("quant"):
            continue
        name = Path(model).name[5:]
        if name not in meta or "Ada" not in str(c.get("gpu", "")):
            continue
        rows = {}
        for p in cfgp.parent.glob("*.json"):
            if p.name.startswith("config"):
                continue
            try:
                for r in json.loads(p.read_text()):
                    if "ade_rollout_k" in r and len(r["ade_rollout_k"]) >= K:
                        rows[r["clip_id"]] = min(r["ade_rollout_k"][:K])
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
        if len(rows) == 500 and st not in score.get(name, {}):
            v = np.array(list(rows.values()))
            score.setdefault(name, {})[st] = (float(v.mean()), float(np.median(v)), cfgp.parent.name)
    say(f"evaluations found on Ada: test500 for {sum('test' in s for s in score.values())} "
        f"kept sets, val500 for {sum('indist' in s for s in score.values())}")

    # ---- 2. does the layer-0 set predict the score? ----
    def n_off(n):
        return len(drop0[n] - ship0)

    dual_arms = [a for fam in FAMILIES.values() for a in fam if a in meta]
    metrics = {"n_kept_sets": len(meta), "shipped_l0_dropped": sorted(ship0), "families": {}}
    for st, label in (("test", "test500"), ("indist", "val500")):
        arms = [a for a in dual_arms if st in score.get(a, {})]
        if len(arms) < 6:
            continue
        y = np.array([score[a][st][0] for a in arms])
        x = np.array([n_off(a) for a in arms])
        r, p = spearmanr(x, y)
        metrics[f"rho_noff_{st}"] = [float(r), float(p), len(arms)]
        say(f"\n== {label}: {len(arms)} dual-family arms. Spearman(n_off, mean minADE@6) = "
            f"{r:+.3f} (p={p:.1e})")
        for k in sorted(set(x)):
            v = y[x == k]
            say(f"   n_off={k}: {len(v):2d} arms, mean {v.mean():.3f}, median {np.median(v):.3f}, "
                f"range {v.min():.3f}-{v.max():.3f}, collapsed (>1.3) {int((v > 1.3).sum())}")
        say("   per head (dropped vs kept; heads with >= 3 arms each way):")
        rows = []
        for h in range(32):
            dm = np.array([h in drop0[a] for a in arms])
            if 3 <= dm.sum() <= len(arms) - 3:
                pv = mannwhitneyu(y[dm], y[~dm]).pvalue
                rows.append((pv, h, int(dm.sum()), float(y[dm].mean()), float(y[~dm].mean())))
        for pv, h, nd, yd, yk in sorted(rows):
            say(f"   H{h:<3d} dropped in {nd:2d}: {yd:.3f} vs kept {yk:.3f}  "
                f"({yd - yk:+.3f}, p={pv:.3f})")
        metrics[f"per_head_{st}"] = [{"head": h, "n_dropped": nd, "dropped": yd, "kept": yk,
                                      "p": pv} for pv, h, nd, yd, yk in rows]

    # held-out group: the 64 half-split arms, y = delta minADE on 150 val clips
    ci_m = json.loads((O / "clipinfluence" / "metrics.json").read_text())
    # y_ade["values"] is in the order of metrics["arms"] ("00".."63")
    ysub = {f"subinf_{a}": float(v) for a, v in zip(ci_m["arms"], ci_m["y_ade"]["values"])}
    sub = [s for s in sorted(ysub) if s in meta]
    if sub:
        ys = np.array([ysub[s] for s in sub])
        xs = np.array([n_off(s) for s in sub])
        r, p = spearmanr(xs, ys)
        metrics["rho_noff_halfsplit"] = [float(r), float(p), len(sub)]
        say(f"\n== held-out group: {len(sub)} half-split arms of calib_100 (delta minADE on 150 "
            f"val clips). Spearman(n_off, delta) = {r:+.3f} (p={p:.1e})")
        for k in sorted(set(xs)):
            v = ys[xs == k]
            say(f"   n_off={k}: {len(v):2d} arms, mean {v.mean():+.3f}, median {np.median(v):+.3f}, "
                f"range {v.min():+.3f} to {v.max():+.3f}")
        say("   per head (dropped vs kept; heads with >= 5 arms each way):")
        hs = []
        for h in range(32):
            dm = np.array([h in drop0[s] for s in sub])
            if 5 <= dm.sum() <= len(sub) - 5:
                pv = mannwhitneyu(ys[dm], ys[~dm]).pvalue
                hs.append((pv, h, int(dm.sum()), float(ys[dm].mean()), float(ys[~dm].mean())))
        for pv, h, nd, yd, yk in sorted(hs):
            say(f"   H{h:<3d} dropped in {nd:2d}: {yd:+.3f} vs kept {yk:+.3f}  "
                f"({yd - yk:+.3f}, p={pv:.4f})")
        metrics["per_head_halfsplit"] = [{"head": h, "n_dropped": nd, "dropped": yd, "kept": yk,
                                          "p": pv} for pv, h, nd, yd, yk in hs]

        # ridge on head indicators: fit on the half-split arms, test on the test500 arms,
        # and the reverse. Heads are the ones that vary in BOTH groups.
        arms = [a for a in dual_arms if "test" in score.get(a, {})]
        if len(arms) >= 10:
            vary = [h for h in range(32)
                    if 0 < sum(h in drop0[a] for a in arms) < len(arms)
                    and 0 < sum(h in drop0[s] for s in sub) < len(sub)]
            XA = np.array([[h in drop0[a] for h in vary] for a in arms], float)
            XB = np.array([[h in drop0[s] for h in vary] for s in sub], float)
            yA = np.log(np.array([score[a]["test"][0] for a in arms]))
            yB = ys

            def ridge(X, y, lam=1.0):
                Xc, yc = X - X.mean(0), y - y.mean()
                return np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ yc)

            wB, wA = ridge(XB, yB), ridge(XA, yA)
            r1, p1 = spearmanr(XA @ wB, yA)
            r2, p2 = spearmanr(XB @ wA, yB)
            metrics["transfer"] = {"heads": vary, "halfsplit_to_test": [float(r1), float(p1)],
                                   "test_to_halfsplit": [float(r2), float(p2)],
                                   "w_halfsplit": dict(zip(map(str, vary), map(float, wB))),
                                   "w_test": dict(zip(map(str, vary), map(float, wA)))}
            say(f"\n== transfer of a head-indicator ridge (heads {vary})")
            say(f"   fit on {len(sub)} half-split arms -> rank the {len(arms)} test500 arms: "
                f"Spearman {r1:+.3f} (p={p1:.1e})")
            say(f"   fit on {len(arms)} test500 arms -> rank the {len(sub)} half-split arms: "
                f"Spearman {r2:+.3f} (p={p2:.1e})")
            say("   weights (positive = dropping the head hurts): half-split fit / test500 fit")
            for h, a, b in sorted(zip(vary, wB, wA), key=lambda t: -t[1]):
                say(f"   H{h:<3d} {a:+.3f} / {b:+.3f}")

    # ---- 3. the families, side by side ----
    say("\n== families: layer-0 heads dropped beyond the 8 every arm drops; "
        "'*' marks heads the shipped arm keeps")
    common = set.intersection(*[drop0[a] for a in dual_arms])
    say(f"dropped by every dual-family arm: {sorted(common)}")
    for fam, arms in FAMILIES.items():
        say(f"\n{fam}")
        say(f"  {'arm':22s}{'n_off':>6s}{'test500':>9s}{'val500':>8s}  layer-0 extra drops")
        rec = []
        for a in arms:
            if a not in meta:
                continue
            ex = sorted(drop0[a] - common)
            t = score.get(a, {}).get("test", (float("nan"),))[0]
            v = score.get(a, {}).get("indist", (float("nan"),))[0]
            rec.append({"arm": a, "n_off": n_off(a), "test500": t, "val500": v, "extra": ex})
            say(f"  {a:22s}{n_off(a):6d}{t:9.4f}{v:8.4f}  "
                + " ".join(f"{h}{'*' if h not in ship0 else ''}" for h in ex))
        metrics["families"][fam] = rec

    (out / "metrics_census.json").write_text(json.dumps(metrics, indent=2))
    (out / "summary_census.txt").write_text("\n".join(lines) + "\n")

    arms = [a for a in dual_arms if "test" in score.get(a, {})]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    x = np.array([n_off(a) for a in arms])
    y = np.array([score[a]["test"][0] for a in arms])
    rng = np.random.default_rng(0)
    ax.scatter(x + rng.uniform(-0.12, 0.12, len(x)), y, s=22, color=C1, alpha=0.8)
    ax.axhline(score["dual_u40_v2"]["test"][0], color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("layer-0 heads dropped that the shipped arm keeps")
    ax.set_ylabel("test500 minADE@6, mean")
    ax.set_title(f"{len(arms)} dual-family arms")
    if sub:
        ax = axes[1]
        ax.scatter(xs + rng.uniform(-0.12, 0.12, len(xs)), ys, s=22, color=C3, alpha=0.8)
        ax.set_xlabel("layer-0 heads dropped that the shipped arm keeps")
        ax.set_ylabel("delta minADE vs unpruned, 150 val clips")
        ax.set_title(f"{len(sub)} half-split arms of calib_100 (held out)")
    fig.tight_layout()
    fig.savefig(out / "plots" / "l0_census.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
