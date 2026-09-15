"""Per-clip influence of calib_100 on dual_u40_v2, from the subset arms.

plans/2026-09-14_calibration-clip-influence.md. Each arm m is `dual_u40_v2` built from a
50-clip subset S_m of calib_100 and evaluated on the same 150 val500 clips, so

    y_m       = mean over those clips of (arm minADE@6 - baseline minADE@6)
    tau_i     = mean(y_m | i in S_m) - mean(y_m | i not in S_m)

tau_i > 0 means the clip makes things WORSE by being in the calibration set.

The design is complementary pairs -- each pair is one permutation of the 100 clips split
50/50 -- which buys three things the estimator needs:

  * every clip sits in exactly M/2 subsets, so every tau_i has the same variance. Under
    independent draws the SE multiplier ranged over 0.354..0.393 at M=32, and the
    noisiest clip is precisely the one that sets a max|tau| threshold.
  * tau_i reduces to a mean over pair differences, sum_p s_pi (y_A - y_B) * 2/M, in which
    subset-level nuisance cancels.
  * the null test is then EXACT rather than asymptotic: under "no clip matters", which
    half of a pair is called A is arbitrary, so flipping the sign of each pair difference
    independently generates the null distribution of tau. That is a randomization test
    over 2^(M/2) relabellings, and it needs no normality and no variance estimate.

Selection here is recomputed model-free (`tyr.dual_scores` + `mask_lib.select_mask_ratios`
are pure numpy) and asserted against the all-100 recipe, so the LOO churn used for gate C1
costs nothing.

Gates: P0 (is there any spread to work with), A1 (does any clip clear the null),
C1 (does kept-set churn predict influence -- expected NO, this repo has found four
other quantities that do not predict capability).

Usage:
  python experiments/evaluation/analyze_clipinfluence.py --arms 32
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2 as chi2_dist
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))
sys.path.insert(0, str(Path(__file__).parent))

import mask_lib as ml
import paper_numbers as pn
import tyr_lib as tyr

BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})

PERCLIP = "importance_v2"
BASELINE = "baseline_ada_ps_indist"
RATIO = 0.3985632694           # the matched u40 budget, never 0.40
N_PERM = 20000
P0_MIN_SD = 0.03               # pre-registered: below this, no per-clip resolution
BUCKETS = ["cruise", "decel_stop", "accel", "turn_left", "turn_right"]


def load_y(arms, fit_limit, prefix):
    """(clip ids, {arm: {clip: (ade6, fde6)}}) for the arms that have finished."""
    base = pn.load(BASELINE, False)
    got, ids = {}, None
    for a in arms:
        d = REPO / "outputs" / f"{prefix}_{a}_fit{fit_limit}"
        if not list(d.glob("*_s*of*.json")):
            continue
        rows = pn.load(d.name, False)
        if len(rows) < fit_limit:
            continue
        got[a] = rows
        ids = sorted(rows) if ids is None else ids
        assert sorted(rows) == ids, f"{a}: different clips than the first arm"
    assert ids, "no finished arms"
    assert set(ids) <= set(base), "fit clips missing from the baseline run"
    y = {}
    for a, rows in got.items():
        d_ade = [pn.at6(rows[c], "ade_rollout_k") - pn.at6(base[c], "ade_rollout_k")
                 for c in ids]
        d_fde = [pn.at6(rows[c], "fde_rollout_k") - pn.at6(base[c], "fde_rollout_k")
                 for c in ids]
        y[a] = {"ade": np.array(d_ade), "fde": np.array(d_fde),
                "degen": float(np.mean([r["coc_degenerate"] for r in rows.values()]))}
    return ids, y


def trimmed(d, frac=0.1):
    """Mean after dropping frac from each tail -- minADE deltas are heavy-tailed."""
    s = np.sort(d)
    k = int(len(s) * frac)
    return float(s[k:len(s) - k].mean())


def tau_from_pairs(S, y):
    """tau_i and the pair differences it is built from.

    S is (M, n_clips) 0/1 with complementary rows 2p, 2p+1. sign[p, i] is +1 when clip i
    is in the first half of pair p and -1 when it is in the second, so
    tau = (2/M) * sign.T @ diff exactly reproduces the in-minus-out means.
    """
    m = S.shape[0]
    assert m % 2 == 0
    a, b = S[0::2], S[1::2]
    assert np.array_equal(a + b, np.ones_like(a)), "rows are not complementary pairs"
    sign = a * 2.0 - 1.0                      # (M/2, n_clips)
    diff = y[0::2] - y[1::2]                  # (M/2,)
    return (2.0 / m) * (sign.T @ diff), sign, diff


def signflip_null(sign, diff, n_perm, seed=0):
    """Exact randomization null: flip which half of each pair is called 'in'."""
    rng = np.random.default_rng(seed)
    m2 = len(diff)
    flips = rng.choice([-1.0, 1.0], size=(n_perm, m2))
    return (2.0 / (2 * m2)) * (flips * diff) @ sign      # (n_perm, n_clips)


def model_free_masks(imp, n_layers=36, n_heads=32, n_mlp=12288):
    sq, sm = tyr.dual_scores(imp)
    vq = ml.select_mask_ratios(sq, np.full(n_layers, RATIO))
    vm = ml.select_mask_ratios(sm, np.full(n_layers, RATIO))
    return vq, vm


def kept_set(mask):
    return {(li, u) for li, u in zip(*np.nonzero(mask))}


def loo_churn(per, n_clips):
    """Fraction of the all-100 kept set that a single clip's removal displaces."""
    full = {k: v.mean(0) for k, v in per.items()}
    fq, fm = model_free_masks(full)
    kq, km = kept_set(fq), kept_set(fm)
    out = np.zeros((n_clips, 2))
    tot = per["coc_vlm_q"].shape[0]
    for i in range(n_clips):
        sel = [j for j in range(tot) if j != i]
        imp = {k: v[sel].mean(0) for k, v in per.items()}
        vq, vm = model_free_masks(imp)
        out[i, 0] = 1 - len(kq & kept_set(vq)) / len(kq)
        out[i, 1] = 1 - len(km & kept_set(vm)) / len(km)
    return out, (fq, fm)


def atypicality(per):
    """Spearman of each clip's own dual score against the pooled one, per axis."""
    pooled_q, pooled_m = tyr.dual_scores({k: v.mean(0) for k, v in per.items()})
    n = per["coc_vlm_q"].shape[0]
    out = np.zeros((n, 2))
    for i in range(n):
        sq, sm = tyr.dual_scores({k: v[i] for k, v in per.items()})
        out[i, 0] = spearmanr(sq.ravel(), pooled_q.ravel()).statistic
        out[i, 1] = spearmanr(sm.ravel(), pooled_m.ravel()).statistic
    return out


def covariates(per, n_clips, clip_ids):
    cov = {}
    meta = json.loads((REPO / "outputs" / PERCLIP / "metrics.json").read_text())
    rec = {r["clip_id"]: r for r in meta["per_clip"]}
    cov["fm_loss"] = np.array([rec[c]["fm_loss"] for c in clip_ids])
    cov["coc_len"] = np.array([float(rec[c]["coc_len"]) for c in clip_ids])
    # prompt_len is constant 3086 across calib_100 -- a covariate with no variance
    cov["draw_order"] = np.arange(n_clips, dtype=float)
    cov["imp_norm_traj"] = np.linalg.norm(
        per["traj_vlm_mlp"].reshape(n_clips, -1), axis=1)
    cov["imp_norm_coc"] = np.linalg.norm(
        per["coc_vlm_mlp"].reshape(n_clips, -1), axis=1)
    atyp = atypicality(per)
    cov["atypical_q"] = -atyp[:, 0]
    cov["atypical_mlp"] = -atyp[:, 1]
    # GT-path clearance to the nearest labelled obstacle. 93/100 clips carry a value; the
    # other 7 are null in the stored file and take the median, which is what `dualsafe`
    # did rather than dropping them
    cl = REPO / "outputs" / "calib_clearance" / "clearance.json"
    if cl.exists():
        d = json.loads(cl.read_text())
        if all(c in d for c in clip_ids):
            med = float(np.median([v for v in d.values() if v is not None]))
            cov["clearance"] = np.array(
                [med if d[c] is None else float(d[c]) for c in clip_ids])
    return cov


def bucket_of(clip_ids):
    b = pd.read_parquet(REPO / "outputs" / "calib_buckets" / "clip_buckets.parquet")
    b = b[b["set"] == "calib_100"].set_index("clip_id")["bucket"]
    return np.array([b.loc[c] for c in clip_ids])


def bucket_contrast(sign, diff, buckets, n_perm, seed=1):
    """Per-clip effect of a bucket, from the pair differences.

    The regressor is (count of bucket b in half A) - (count in half B), which for
    complementary halves is just sign summed over the bucket's clips. Fitting diff on
    those columns gives an effect PER CLIP of that bucket, and the sign-flip null applies
    unchanged.

    The columns sum to exactly zero -- both halves hold 50 clips, so the bucket counts
    cannot all move up together -- which makes the design rank-deficient by one. That is
    not a defect to patch: an absolute per-bucket level is simply not identified here,
    only contrasts are. lstsq's minimum-norm solution is orthogonal to the null direction
    (1,1,...,1), so the coefficients returned sum to zero and read as **how much better
    or worse than an average calibration clip** a clip of that bucket is. Dropping a
    column and reading effects against a reference bucket would say the same thing in a
    less symmetric way.
    """
    names = [b for b in BUCKETS if (buckets == b).any()]
    x = np.stack([sign[:, buckets == b].sum(1) for b in names], axis=1)  # (M/2, B)
    coef, *_ = np.linalg.lstsq(x, diff, rcond=None)
    rng = np.random.default_rng(seed)
    flips = rng.choice([-1.0, 1.0], size=(n_perm, len(diff)))
    null = np.stack([np.linalg.lstsq(x, f * diff, rcond=None)[0] for f in flips])
    p = (np.abs(null) >= np.abs(coef)).mean(0)
    return names, coef, p, null


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", type=int, default=32)
    ap.add_argument("--fit-limit", type=int, default=150)
    ap.add_argument("--prefix", type=str, default="subinf")
    ap.add_argument("--exp-id", type=str, default="clipinfluence")
    ap.add_argument("--top-k", type=int, default=20, help="Stage B drop-set size")
    args = ap.parse_args()

    design = json.loads(
        (REPO / "outputs" / "clipinfluence" / "design.json").read_text())
    clip_ids = design["clip_ids"]
    n_clips = design["n_clips"]
    tags = [f"{i:02d}" for i in range(args.arms)]
    ids, y = load_y(tags, args.fit_limit, args.prefix)
    # tau is a mean over COMPLEMENTARY PAIRS, so a half-finished pair cannot be used at
    # all -- and the pairs are rows (2p, 2p+1), so what is usable is the longest finished
    # PREFIX rounded down to an even length, not however many arms happen to be done.
    n_done = len([t for t in tags if t in y])
    p = 0
    while p < len(tags) and tags[p] in y:
        p += 1
    done = tags[:(p // 2) * 2]
    print(f"{n_done}/{args.arms} arms finished, {len(done)} usable (whole leading pairs), "
          f"{len(ids)} fit clips")
    assert len(done) >= 4, "need at least two pairs"

    S = np.zeros((len(done), n_clips))
    for m, t in enumerate(done):
        S[m, design["subsets"][int(t)]] = 1.0

    per = dict(np.load(REPO / "outputs" / PERCLIP / "importance_perclip.npz"))
    buckets = bucket_of(clip_ids)
    out = REPO / "outputs" / args.exp_id
    (out / "plots").mkdir(parents=True, exist_ok=True)
    res = {"arms": done, "n_fit_clips": len(ids), "design_sha": design["sha"]}

    # --- y and gate P0 -----------------------------------------------------------
    metrics = {}
    for key in ("ade", "fde"):
        yv = np.array([y[t][key].mean() for t in done])
        yt = np.array([trimmed(y[t][key]) for t in done])
        metrics[key] = {"mean": yv, "trim": yt}
    yv = metrics["ade"]["mean"]
    res["y_ade"] = {"values": yv.tolist(), "sd": float(yv.std(ddof=1)),
                    "min": float(yv.min()), "max": float(yv.max()),
                    "mean": float(yv.mean())}
    res["gates"] = {"P0_sd_y": float(yv.std(ddof=1)),
                    "P0_pass": bool(yv.std(ddof=1) >= P0_MIN_SD),
                    "P0_threshold": P0_MIN_SD}
    print(f"P0 sd(y_ade) = {yv.std(ddof=1):.4f} over [{yv.min():+.4f}, {yv.max():+.4f}] "
          f"-> {'PASS' if yv.std(ddof=1) >= P0_MIN_SD else 'FAIL (bucket level only)'}")

    # --- what KIND of damage: reasoning collapse or trajectory? ---------------------
    # `dual+h4` and `dualsafe` both made the point that CoC health is not a safety proxy,
    # and in opposite directions. Worth knowing which axis a bad calibration subset moves
    # before reading anything else into y.
    degen = np.array([y[t]["degen"] for t in done])
    r_dg = spearmanr(degen, yv)
    res["degeneracy"] = {"values": degen.tolist(), "mean": float(degen.mean()),
                         "max": float(degen.max()),
                         "rho_with_y": float(r_dg.statistic), "p": float(r_dg.pvalue)}
    print(f"CoC degeneracy over arms: mean {degen.mean():.3f} max {degen.max():.3f}; "
          f"rho with y {r_dg.statistic:+.3f} (p={r_dg.pvalue:.3f})")

    # --- gate A2: is y additive in the clips at all? --------------------------------
    # The two halves of a pair partition calib_100, so under ANY additive model
    # y_S = (1/|S|) sum_{i in S} e_i the pair sum is the same constant for every pair.
    # It is not, and that is the first thing to know: if the map S -> y is not additive
    # there is no per-clip decomposition to find, and tau is a projection of a
    # non-additive surface rather than a property of a clip.
    #
    # y on a FIXED clip set is deterministic (Ada, clip-derived seeds, bitwise), so the
    # spread of pair sums is structural with no measurement error to explain it away.
    # The chi2 is the conservative version for generalising past these clips: it treats
    # the 150 as a sample and charges each arm the SE of its own paired deltas.
    se = np.array([y[t]["ade"].std(ddof=1) / np.sqrt(len(ids)) for t in done])
    psum = yv[0::2] + yv[1::2]
    pvar = se[0::2] ** 2 + se[1::2] ** 2
    chi2 = float((((psum - psum.mean()) ** 2) / pvar).sum())
    p_add = float(chi2_dist.sf(chi2, len(psum) - 1))
    res["gates"].update({
        "A2_pair_sums": psum.tolist(), "A2_sd": float(psum.std(ddof=1)),
        "A2_range": float(psum.max() - psum.min()),
        "A2_sd_from_noise": float(np.sqrt(pvar.mean())),
        "A2_chi2": chi2, "A2_df": len(psum) - 1, "A2_p": p_add,
        "A2_additive": bool(p_add >= 0.05)})
    # The same statement in the form that reads without a chi2: additivity forces
    # y_B = C - y_A, so the two halves of a pair would be PERFECTLY anticorrelated.
    # Measuring rho instead of the sum turns "is the sum constant" into "does one half
    # being good make the other bad", which is the question in words.
    r_pair = spearmanr(yv[0::2], yv[1::2])
    zz = np.arctanh(r_pair.statistic)
    se_z = 1 / np.sqrt(len(psum) - 3)
    lo, hi = np.tanh([zz - 1.96 * se_z, zz + 1.96 * se_z])
    res["gates"].update({"A2_rho_pair": float(r_pair.statistic),
                         "A2_rho_lo": float(lo), "A2_rho_hi": float(hi),
                         "A2_rho_required": -1.0})
    print(f"A2 pair sums span [{psum.min():.4f}, {psum.max():.4f}] "
          f"(sd {psum.std(ddof=1):.4f} vs {np.sqrt(pvar.mean()):.4f} from eval noise); "
          f"chi2 {chi2:.1f}/{len(psum) - 1} df p={p_add:.2g} -> "
          f"{'additive' if p_add >= 0.05 else 'NOT additive'}")
    print(f"   rho(y_A, y_B) within a pair = {r_pair.statistic:+.3f} "
          f"[{lo:+.3f}, {hi:+.3f}] -- additivity requires exactly -1.000, so the two "
          f"halves are INDEPENDENT draws, not a split of one quantity")
    res["eval_noise_se"] = {"mean": float(se.mean()), "max": float(se.max())}

    # --- tau and the exact sign-flip null ----------------------------------------
    taus, nulls = {}, {}
    for key in ("ade", "fde"):
        for agg in ("mean", "trim"):
            t, sign, diff = tau_from_pairs(S, metrics[key][agg])
            null = signflip_null(sign, diff, N_PERM)
            taus[(key, agg)] = t
            nulls[(key, agg)] = null
    tau, sign, diff = tau_from_pairs(S, yv)
    null = nulls[("ade", "mean")]
    thresh = float(np.percentile(np.abs(null).max(1), 95))
    pvals = (np.abs(null) >= np.abs(tau)).mean(0)
    order = np.argsort(-tau)
    res["tau_ade"] = tau.tolist()
    res["tau_p"] = pvals.tolist()
    res["gates"]["A1_fwer_threshold"] = thresh
    res["gates"]["A1_max_abs_tau"] = float(np.abs(tau).max())
    res["gates"]["A1_pass"] = bool(np.abs(tau).max() > thresh)
    res["gates"]["A1_n_clear"] = int((np.abs(tau) > thresh).sum())
    print(f"A1 max|tau| = {np.abs(tau).max():.4f} vs FWER-5% null {thresh:.4f} -> "
          f"{res['gates']['A1_n_clear']} of {n_clips} clips clear")

    # A1b: the OMNIBUS version. max|tau| asks whether one clip stands out and pays the
    # full multiplicity price for it; sd(tau) pools all 100 and asks whether there is any
    # per-clip structure at all, which is both the better-powered question and the one
    # that decides whether a per-clip answer exists. The null sd comes from the same
    # exact sign-flip relabelling, so no variance model is assumed, and the difference of
    # the two variances estimates how much of the observed tau spread is real.
    null_sd = float(null.std(axis=1).mean())
    obs_sd = float(tau.std(ddof=1))
    p_omni = float((null.std(axis=1) >= obs_sd).mean())
    excess = obs_sd ** 2 - null_sd ** 2
    res["gates"].update({
        "A1b_sd_tau": obs_sd, "A1b_sd_null": null_sd, "A1b_p": p_omni,
        "A1b_true_sd": float(np.sqrt(max(0.0, excess))),
        "A1b_signal_share": float(max(0.0, excess) / obs_sd ** 2),
        "A1b_pass": bool(p_omni < 0.05)})
    # Turn "nothing found" into a bound: a true per-clip sd of `mde` would have pushed
    # sd(tau) past the null's 95th percentile. Approximate -- it adds the signal variance
    # to the null variance rather than re-deriving the projection -- but it is the
    # difference between an unfalsifiable null and a measured ceiling.
    p95 = float(np.percentile(null.std(axis=1), 95))
    mde = float(np.sqrt(max(0.0, p95 ** 2 - null_sd ** 2)))
    res["gates"]["A1b_p95_null"] = p95
    res["gates"]["A1b_mde_per_clip_sd"] = mde
    print(f"A1b sd(tau) {obs_sd:.4f} vs sign-flip null {null_sd:.4f} (p={p_omni:.3f}) -> "
          f"true per-clip sd {np.sqrt(max(0.0, excess)):.4f}, "
          f"{100 * max(0.0, excess) / obs_sd ** 2:.0f}% of the spread is signal")
    # Express that ceiling in the units the question was asked in. Under the additive
    # model y_S = mean_{i in S} e_i, sampling n of N without replacement gives
    #   sd(tau) = (N/(N-1))/n * sd(e)        and      sd(y) = sd(e) * sqrt((1/n)(1-n/N)(N/(N-1)))
    # so a detectable sd(tau) converts to the sd(y) that such a structure would produce.
    # Set that against the spread actually observed, minus evaluation noise: the
    # comparison says whether the design could have seen an additive explanation of the
    # size that is there to be explained.
    n_sub = len(design["subsets"][0])
    k_tau = (n_clips / (n_clips - 1)) / n_sub
    k_y = np.sqrt((1 / n_sub) * (1 - n_sub / n_clips) * (n_clips / (n_clips - 1)))
    struct_sd = float(np.sqrt(max(0.0, yv.var(ddof=1) - pvar.mean() / 2)))
    res["gates"]["A1b_mde_as_sd_y"] = float(mde / k_tau * k_y)
    res["gates"]["A1b_structural_sd_y"] = struct_sd
    print(f"    detectable per-clip sd at this M: {mde:.4f} "
          f"(null p95 {p95:.4f}); tau's own SE is {null_sd:.4f}")
    print(f"    -> an additive structure that big would make sd(y) "
          f"{mde / k_tau * k_y:.4f}; the structural sd(y) actually present is "
          f"{struct_sd:.4f}")
    print(f"   tau range [{tau.min():+.4f}, {tau.max():+.4f}], "
          f"analytic per-clip SE {2 * yv.std(ddof=1) / np.sqrt(len(done)):.4f}")

    # metric agreement: an estimate that does not survive a change of metric is noise
    for key, agg in (("ade", "trim"), ("fde", "mean")):
        r = spearmanr(tau, taus[(key, agg)]).statistic
        res.setdefault("tau_agreement", {})[f"{key}_{agg}"] = float(r)
        print(f"   rho(tau_ade_mean, tau_{key}_{agg}) = {r:+.3f}")

    # --- buckets ------------------------------------------------------------------
    names, coef, bp, _ = bucket_contrast(sign, diff, buckets, N_PERM)
    res["bucket"] = {"names": names, "per_clip_effect": coef.tolist(),
                     "p": bp.tolist(),
                     "n": [int((buckets == b).sum()) for b in names],
                     "tau_mean": [float(tau[buckets == b].mean()) for b in names]}
    print("\nbucket per-clip effect (sign-flip p):")
    for b, c, p_, nb in zip(names, coef, bp, res["bucket"]["n"]):
        print(f"  {b:12s} n={nb:3d}  {c:+.5f}  p={p_:.4f}{'  *' if p_ < 0.05 else ''}")

    # --- covariates and gate C1 ---------------------------------------------------
    cov = covariates(per, n_clips, clip_ids)
    churn, (fq, _) = loo_churn(per, n_clips)
    cov["loo_churn_q"] = churn[:, 0]
    cov["loo_churn_mlp"] = churn[:, 1]
    shipped = json.loads(
        (REPO / "outputs" / "slim_dual_u40_v2" / "slim_meta.json").read_text())
    ship_q = {(li, h) for li, m in enumerate(shipped["vlm"]) for h in m["q"]}
    assert kept_set(fq) == ship_q, "model-free selection does not reproduce the recipe"
    print("\n  (model-free selection reproduces slim_dual_u40_v2 exactly)")

    res["covariates"] = {}
    print("\nSpearman vs tau (and vs |tau|):")
    for k, v in sorted(cov.items()):
        r1 = spearmanr(v, tau)
        r2 = spearmanr(v, np.abs(tau))
        res["covariates"][k] = {"rho_tau": float(r1.statistic), "p_tau": float(r1.pvalue),
                                "rho_abs": float(r2.statistic), "p_abs": float(r2.pvalue)}
        print(f"  {k:16s} rho={r1.statistic:+.3f} (p={r1.pvalue:.3f})   "
              f"|tau| rho={r2.statistic:+.3f} (p={r2.pvalue:.3f})")
    c1 = res["covariates"]["loo_churn_q"]
    res["gates"]["C1_rho_churn_abs_tau"] = c1["rho_abs"]
    res["gates"]["C1_p"] = c1["p_abs"]

    # --- anchor: what the FULL 100 costs on exactly these clips ---------------------
    full = pn.load("dual_u40_v2_ps_indist", False)
    base = pn.load(BASELINE, False)
    anchor = float(np.mean([pn.at6(full[c], "ade_rollout_k")
                            - pn.at6(base[c], "ade_rollout_k") for c in ids]))
    res["anchor_full100"] = {"delta": anchor,
                             "n_subsets_better": int((yv < anchor).sum()),
                             "n_subsets": len(yv),
                             "ratio_mean": float(yv.mean() / anchor)}
    print(f"\nanchor: full calib_100 costs {anchor:+.4f} on these {len(ids)} clips; "
          f"{(yv < anchor).sum()}/{len(yv)} half-size subsets beat it "
          f"(subset mean is {yv.mean() / anchor:.1f}x)")

    # --- Stage C: set-level predictors, the ones n=6 could not test -----------------
    # plans/2026-09-07_draw-order rejected 21 set-level explanations across SIX draws.
    # Here the same quantities get n = len(done) subsets whose y spans an order of
    # magnitude, so a null means something it could not mean there.
    res["set_level"] = {}
    print("\nStage C -- set-level predictors of y (Spearman over arms):")
    feats = {k: np.array([v[design["subsets"][int(t)]].mean() for t in done])
             for k, v in cov.items()}
    for b in names:
        feats[f"n_{b}"] = np.array(
            [float((buckets[design["subsets"][int(t)]] == b).sum()) for t in done])
    # 15 correlated predictors on n = len(done) arms: a raw p<0.05 is worth 0.75 hits by
    # chance here, and `draw-order` already showed how a list of set-level candidates
    # behaves. Two controls, both needed -- BH for how many survive, and a permutation
    # max|rho| for whether the BEST one is remarkable at all.
    keys = sorted(feats)
    rhos = np.array([spearmanr(feats[k], yv).statistic for k in keys])
    praw = np.array([spearmanr(feats[k], yv).pvalue for k in keys])
    rank = np.argsort(praw)
    q = np.empty_like(praw)
    run = 1.0
    for j in rank[::-1]:
        run = min(run, praw[j] * len(keys) / (1 + list(rank).index(j)))
        q[j] = run
    rngp = np.random.default_rng(2)
    maxnull = np.array([
        np.abs([spearmanr(feats[k], yv[rngp.permutation(len(yv))]).statistic
                for k in keys]).max() for _ in range(2000)])
    fwer = float(np.percentile(maxnull, 95))
    for j, k in enumerate(keys):
        res["set_level"][k] = {"rho": float(rhos[j]), "p": float(praw[j]),
                               "q_bh": float(q[j])}
        print(f"  {k:16s} rho={rhos[j]:+.3f} (p={praw[j]:.3f}, q={q[j]:.3f})"
              f"{'  *' if q[j] < 0.05 else ''}")
    res["set_level_fwer"] = {"max_abs_rho": float(np.abs(rhos).max()),
                             "threshold": fwer,
                             "pass": bool(np.abs(rhos).max() > fwer),
                             "n_q05": int((q < 0.05).sum())}
    print(f"  max|rho| {np.abs(rhos).max():.3f} vs permutation FWER-5% {fwer:.3f}; "
          f"{(q < 0.05).sum()} survive BH")

    # --- POST HOC: does resembling the full-100 selection predict y? ---------------
    # Not in the plan and not in the Stage C multiplicity correction above, because it
    # was asked after seeing those results. Kept separate and labelled for that reason:
    # it is a hypothesis for an independent sample, not a finding. Bonferroni is taken
    # over all 17 predictors actually tested (15 pre-registered + these 2).
    full_meta = json.loads(
        (REPO / "outputs" / "slim_dual_u40_v2" / "slim_meta.json").read_text())
    res["post_hoc_overlap"] = {}
    for key in ("q", "mlp"):
        ref = {(li, u) for li, mm in enumerate(full_meta["vlm"]) for u in mm[key]}
        ov = []
        for t in done:
            am = json.loads((REPO / "outputs" / f"slim_subinf_{t}"
                             / "slim_meta.json").read_text())
            s = {(li, u) for li, mm in enumerate(am["vlm"]) for u in mm[key]}
            ov.append(len(ref & s) / len(ref))
        ov = np.array(ov)
        r = spearmanr(ov, yv)
        res["post_hoc_overlap"][key] = {
            "rho": float(r.statistic), "p": float(r.pvalue),
            "min": float(ov.min()), "max": float(ov.max()), "mean": float(ov.mean()),
            "bonferroni_17": float(min(1.0, r.pvalue * 17))}
        print(f"POST HOC overlap_{key} with the full-100 kept set: "
              f"{ov.min():.3f}..{ov.max():.3f}, rho with y {r.statistic:+.3f} "
              f"(p={r.pvalue:.4f}, Bonferroni/17 {min(1.0, r.pvalue * 17):.4f})")

    # --- Stage B drop sets --------------------------------------------------------
    k = args.top_k
    res["stage_b"] = {"worst": order[:k].tolist(), "best": order[-k:].tolist(),
                      "worst_tau": tau[order[:k]].tolist(),
                      "best_tau": tau[order[-k:]].tolist()}
    rng = np.random.default_rng(20260914)
    for s in "abc":
        res["stage_b"][f"rand_{s}"] = sorted(
            rng.choice(n_clips, size=k, replace=False).tolist())
    print(f"\nStage B: drop-{k} worst = {order[:k].tolist()}")
    print(f"         drop-{k} best  = {sorted(order[-k:].tolist())}")

    # --- plots --------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.axhspan(-thresh, thresh, color=MUTED, alpha=0.15,
               label=f"FWER-5% null (+-{thresh:.3f})")
    cols = {b: c for b, c in zip(BUCKETS, [C1, C2, C3, C4, INK])}
    ax.scatter(range(n_clips), tau[order], s=18,
               c=[cols.get(b, MUTED) for b in buckets[order]])
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("calibration clip, sorted by tau")
    ax.set_ylabel("tau  (minADE@6, m)")
    ax.set_title(f"per-clip marginal contribution, {len(done)} subset arms")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "plots" / "tau_sorted.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    axes[0].hist(yv, bins=12, color=C1, alpha=0.8)
    axes[0].set_xlabel("y = mean paired dminADE@6")
    axes[0].set_title(f"arm spread (sd {yv.std(ddof=1):.3f})")
    for j, b in enumerate(names):
        v = tau[buckets == b]
        axes[1].scatter(np.full(len(v), j) + np.random.uniform(-.12, .12, len(v)), v,
                        s=14, color=cols.get(b, MUTED))
        axes[1].scatter([j], [v.mean()], marker="_", s=400, color=INK)
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels([b.replace("_", "\n") for b in names], fontsize=8)
    axes[1].axhline(0, color=MUTED, lw=0.8)
    axes[1].set_ylabel("tau")
    axes[1].set_title("tau by action bucket")
    axes[2].scatter(cov["loo_churn_q"], np.abs(tau), s=16, color=C3)
    axes[2].set_xlabel("LOO kept-set churn (Q)")
    axes[2].set_ylabel("|tau|")
    axes[2].set_title(f"C1: rho {c1['rho_abs']:+.3f} (p={c1['p_abs']:.2f})")
    fig.tight_layout()
    fig.savefig(out / "plots" / "influence_structure.png", dpi=150)
    plt.close(fig)

    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    lines = [
        f"== per-clip calibration influence, {len(done)} arms x {len(ids)} fit clips ==",
        f"P0 sd(y) {yv.std(ddof=1):.4f}  range [{yv.min():+.4f}, {yv.max():+.4f}]",
        (f"A1 max|tau| {np.abs(tau).max():.4f} vs null {thresh:.4f}: "
         f"{res['gates']['A1_n_clear']} clips clear"),
        "bucket per-clip effect: " + ", ".join(
            f"{b} {c:+.4f} (p={p_:.3f})" for b, c, p_ in zip(names, coef, bp)),
        f"C1 rho(churn, |tau|) {c1['rho_abs']:+.3f} p={c1['p_abs']:.3f}",
    ]
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
