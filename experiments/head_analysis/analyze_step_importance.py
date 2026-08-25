"""Gates G0-G4 for the denoising-step decomposition of the expert's trajectory importance.

Reads the runs produced by run_step_importance.py (--mode fm / --mode infer) and
run_step_mask.py and decides, in the order the plan pre-registered them:

  G0 integrity      the per-step decomposition summed back over steps reproduces
                    importance.npz `traj_exp_*` (median relative error < 1e-3, per-layer
                    Spearman > 0.999). Everything below is void if this fails.
  G1 cancellation   C_u = |sum_s g_s| / sum_s |g_s|. The shipped score is the numerator,
                    the clip axis uses the denominator's convention. median C < 0.5 AND
                    kept-set overlap(sum, sumabs) < 0.90 -> steps really do cancel.
  G2 heterogeneity  step-to-step Spearman vs the clip split-half noise floor at a fixed
                    step. Below the floor -> the step axis carries real structure.
  G3 path mismatch  training-path score vs inference-path score, and which of the two
                    predicts the measured per-step damage.
  G4 damage curve   minADE when the mask hits exactly one step, over s.

Usage:
  python experiments/head_analysis/analyze_step_importance.py \
      --fm-perstep stepimp_fm_perstep --fm-shared stepimp_fm_shared \
      --infer stepimp_infer --stepmask stepmask_v1 --out stepimp_analysis
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr, wilcoxon  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))

import eval_lib as el  # noqa: E402
import mask_lib as ml  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
EPS = 1e-30

BG = "#FAF9F5"
INK = "#29261B"
MUTED = "#6B6555"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def layer_spearman(a, b):
    """Mean over layers of Spearman rho between two (L, U) score maps.

    Per layer, not pooled: selection argsorts within a layer, so a correlation that a
    shared depth trend inflates would not describe the choice actually being made.
    """
    rs = []
    for i in range(a.shape[0]):
        r = spearmanr(a[i], b[i])[0]
        if np.isfinite(r):
            rs.append(r)
    return float(np.mean(rs)) if rs else float("nan")


def keep_overlap(a, b, ratio):
    """Fraction of a's kept units that b also keeps, at a per-layer uniform ratio."""
    layers = list(range(a.shape[0]))
    ka = ml.select_mask(a, ratio, layers) == 1
    kb = ml.select_mask(b, ratio, layers) == 1
    return float((ka & kb).sum() / max(ka.sum(), 1))


def median_ci(d, n_boot=10000, seed=0):
    d = np.asarray(d, dtype=float)
    rng = np.random.RandomState(seed)
    boots = np.median(d[rng.randint(0, len(d), size=(n_boot, len(d)))], axis=1)
    return float(np.median(d)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def split_half_floor(per_clip, n_splits=20, seed=0):
    """Noise floor: correlate two disjoint halves of the clips at the SAME step.

    This is the number the step-to-step correlation has to beat. Without it a low
    step-to-step rho would be indistinguishable from 100 clips simply not being enough
    to pin a ranking down.

    Returned corrected by Spearman-Brown, 2r/(1+r). Each half holds 50 clips while the
    per-step scores being compared each hold all 100, so the raw split-half number is the
    reliability of a half-length measurement and would understate the floor -- which would
    bias the whole analysis toward declaring step heterogeneity that is really sampling
    noise. The raw values are returned alongside so both can be reported.
    """
    n, S = per_clip.shape[0], per_clip.shape[1]
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n_splits):
        idx = rng.permutation(n)
        a, b = idx[: n // 2], idx[n // 2 : 2 * (n // 2)]
        for s in range(S):
            out.append(layer_spearman(per_clip[a, s].mean(0).astype(np.float64),
                                      per_clip[b, s].mean(0).astype(np.float64)))
    raw = np.array(out)
    return 2 * raw / (1 + raw), raw


def perclip_fidelity(z, pc, lines, tol=1e-3):
    """Which per-clip arrays actually reproduce the fp64 aggregate they came from?

    Exists because they once did not: the MLP per-clip array was stored as fp16 while the
    gradients live at ~1e-7, inside fp16's subnormal range, so 75% of it underflowed to
    exact zero and re-aggregating it carried a median relative error of 0.22. Every
    per-clip statistic computed from that file -- split-half floors, clip concentration --
    was meaningless, and nothing in the analysis said so. This makes the file prove itself
    before any of it is used.
    """
    ok = {}
    if pc is None:
        return ok
    lines.append("per-clip fidelity -- does re-aggregating reproduce the fp64 aggregate?")
    for unit in ("q", "mlp", "kv_k"):
        key = f"{unit}_abs_step"
        if key not in pc or key not in z:
            continue
        got = pc[key].astype(np.float64).mean(0)
        ref = z[key]
        rel = np.abs(got - ref) / (np.abs(ref) + EPS)
        good = float(np.median(rel)) < tol
        ok[unit] = good
        lines.append(f"  {unit:4s} dtype {pc[key].dtype}  median rel {np.median(rel):.3e}  "
                     f"p99 {np.percentile(rel, 99):.3e}   "
                     f"{'usable' if good else 'CORRUPT -- per-clip stats skipped'}")
    lines.append("")
    return ok


def load_run(exp_id):
    d = REPO / "outputs" / exp_id
    z = dict(np.load(d / "step_importance.npz"))
    cfg = json.loads((d / "config.json").read_text())
    pc_path = d / "step_importance_perclip.npz"
    pc = np.load(pc_path) if pc_path.exists() else None
    return z, cfg, pc


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


def compare_to_importance(z, imp, label, lines, judge=True):
    """|sum_s g_s| from the decomposition vs what run_importance stored."""
    ok, res = True, {}
    for unit, key in (("q", "traj_exp_q"), ("mlp", "traj_exp_mlp")):
        a, b = z[f"{unit}_shipped"], imp[key]
        rel = np.abs(a - b) / (np.abs(b) + EPS)
        rho = layer_spearman(a, b)
        good = np.median(rel) < 1e-3 and rho > 0.999
        ok = ok and good
        res[unit] = {"median_rel": float(np.median(rel)), "p99_rel": float(np.percentile(rel, 99)),
                     "layer_rho": rho, "pass": bool(good)}
        lines.append(f"  [{label}] {unit:4s} median rel err {np.median(rel):.2e}  "
                     f"p99 {np.percentile(rel, 99):.2e}  per-layer rho {rho:.6f}"
                     + (f"   {'PASS' if good else 'FAIL'}" if judge else ""))
    res["pass"] = bool(ok)
    return res


def gate_g0(z, imp, imp_alt, lines):
    """The reference must come from the SAME GPU architecture as the decomposition.

    3-4% of clips generate different CoC text on Ada than on Blackwell, which changes the
    cache and therefore the gradients; comparing across architectures would charge that
    drift to the decomposition. importance_v2 (Blackwell) is reported as drift only.
    """
    lines.append("G0 integrity -- does the decomposition reproduce run_importance?")
    res = {"same_arch": compare_to_importance(z, imp, "same-arch", lines)}
    if imp_alt is not None:
        res["cross_arch_drift"] = compare_to_importance(z, imp_alt, "cross-arch drift",
                                                        lines, judge=False)
    lines.append(f"  -> G0 {'PASS' if res['same_arch']['pass'] else 'FAIL'}\n")
    return res


def gate_g1(z, ratio, lines):
    lines.append("G1 sign cancellation -- |sum_s g_s| vs sum_s |g_s|")
    res = {}
    for unit in ("q", "mlp"):
        shipped = z[f"{unit}_shipped"]                 # mean_clips |sum_s g_s|
        sumabs = z[f"{unit}_abs_step"].sum(0)          # mean_clips sum_s |g_s|
        canc = shipped / (sumabs + EPS)
        ov = keep_overlap(shipped, sumabs, ratio)
        res[unit] = {"median_C": float(np.median(canc)),
                     "p10_C": float(np.percentile(canc, 10)),
                     "p90_C": float(np.percentile(canc, 90)),
                     "overlap_sum_sumabs": ov}
        lines.append(f"  {unit:4s} C median {np.median(canc):.3f}  "
                     f"[p10 {np.percentile(canc, 10):.3f}, p90 {np.percentile(canc, 90):.3f}]"
                     f"   kept-overlap(sum, sumabs) @r{ratio:.2f} = {ov:.3f}")
    # three states, exactly as pre-registered: ACCEPT needs both conditions on some unit
    # type, REJECT needs C > 0.9 everywhere, anything between is INCONCLUSIVE rather than
    # silently rounded to a REJECT
    accept = any(res[u]["median_C"] < 0.5 and res[u]["overlap_sum_sumabs"] < 0.90
                 for u in ("q", "mlp"))
    reject = all(res[u]["median_C"] > 0.9 for u in ("q", "mlp"))
    verdict = "ACCEPT" if accept else ("REJECT" if reject else "INCONCLUSIVE")
    note = {"ACCEPT": "cancellation is a real mechanism",
            "REJECT": "no meaningful cancellation",
            "INCONCLUSIVE": "cancellation exists but barely moves the selection"}[verdict]
    lines.append(f"  -> G1 {verdict} ({note})\n")
    res["verdict"] = verdict
    return res


def gate_g2(z, pc, ratio, lines, usable=None):
    lines.append("G2 step heterogeneity -- step-to-step rho vs the clip split-half floor")
    res = {}
    for unit in ("q", "mlp"):
        a = z[f"{unit}_abs_step"].astype(np.float64)  # (S, L, U)
        S = a.shape[0]
        M = np.eye(S)
        for i in range(S):
            for j in range(i + 1, S):
                M[i, j] = M[j, i] = layer_spearman(a[i], a[j])
        # rho is the ranking; overlap is what selection actually does with it at the
        # operating budget, and the two can disagree when the tail is what moves
        O = np.eye(S)
        for i in range(S):
            for j in range(S):
                if i != j:
                    O[i, j] = keep_overlap(a[i], a[j], ratio)
        off = M[~np.eye(S, dtype=bool)]
        off_o = O[~np.eye(S, dtype=bool)]
        mass = a.sum((1, 2))
        entry = {"step_rho_matrix": M.tolist(), "step_overlap_matrix": O.tolist(),
                 "step_rho_median": float(np.median(off)),
                 "step_rho_p05": float(np.percentile(off, 5)),
                 "step_overlap_median": float(np.median(off_o)),
                 "step_overlap_min": float(off_o.min()),
                 "mass_per_step": mass.tolist(),
                 "mass_max_over_min": float(mass.max() / max(mass.min(), EPS)),
                 "argmax_mass_step": int(mass.argmax())}
        entry["overlap_shipped_vs_argmax_step"] = keep_overlap(
            z[f"{unit}_shipped"], a[mass.argmax()], ratio)
        # which step is the static mask actually serving? one number per step, so a mask
        # that silently specialises to one t shows up as a peak instead of an average
        entry["overlap_shipped_by_step"] = [keep_overlap(z[f"{unit}_shipped"], a[s], ratio)
                                            for s in range(S)]
        # the budget statement: of the units each step would keep on its own, how many are
        # kept by EVERY step, and how wide would a mask have to be to hold the union
        L, U = a.shape[1], a.shape[2]
        k = U - int(round(U * ratio))   # mirrors select_mask's rounding exactly
        inter, union = [], []
        for li in range(L):
            sets = [set(np.argsort(a[s, li])[-k:].tolist()) for s in range(S)]
            inter.append(len(set.intersection(*sets)) / k)
            union.append(len(set.union(*sets)) / k)
        entry["kept_at_every_step"] = float(np.mean(inter))
        entry["union_over_k"] = float(np.mean(union))
        # against k/U, not 1-ratio: select_mask rounds, and with only 16 Q heads per layer
        # k/U is 0.625 while 1-ratio is 0.600, which understated this by 3.7 points
        entry["kept_fraction"] = k / U
        entry["union_width"] = float(np.mean(union) * k / U)

        key = f"{unit}_abs_step"
        if pc is not None and key in pc and (usable is None or usable.get(unit, True)):
            floor, floor_raw = split_half_floor(pc[key])
            # the pre-registered test is CI vs CI on the medians, not a raw percentile
            s_med, s_lo, s_hi = median_ci(off)
            f_med, f_lo, f_hi = median_ci(floor)
            entry.update({"step_rho_median_ci": [s_med, s_lo, s_hi],
                          "floor_median_ci": [f_med, f_lo, f_hi],
                          "floor_median_raw": float(np.median(floor_raw)),
                          "below_floor": bool(s_hi < f_lo)})
        res[unit] = entry
        lines.append(f"  {unit:4s} step-to-step rho median {entry['step_rho_median']:.3f} "
                     f"(p05 {entry['step_rho_p05']:.3f}), kept-overlap median "
                     f"{entry['step_overlap_median']:.3f} (min {entry['step_overlap_min']:.3f})")
        if "floor_median_ci" in entry:
            s_med, s_lo, s_hi = entry["step_rho_median_ci"]
            f_med, f_lo, f_hi = entry["floor_median_ci"]
            lines.append(f"       step-to-step median {s_med:.3f} [{s_lo:.3f},{s_hi:.3f}]  vs  "
                         f"clip split-half floor {f_med:.3f} [{f_lo:.3f},{f_hi:.3f}] "
                         f"(raw {entry['floor_median_raw']:.3f}, Spearman-Brown corrected)")
            lines.append(f"       -> {'below floor: real structure' if entry['below_floor'] else 'within noise'}")
        lines.append(f"       mass max/min over steps {entry['mass_max_over_min']:.2f} "
                     f"(peak at step {entry['argmax_mass_step']}), "
                     f"kept-overlap(shipped, peak step) {entry['overlap_shipped_vs_argmax_step']:.3f}")
        lines.append("       kept-overlap(shipped, step s) by s: " +
                     " ".join(f"{x:.2f}" for x in entry["overlap_shipped_by_step"]))
        lines.append(f"       of each step's own keep-set, {entry['kept_at_every_step']:.3f} is kept "
                     f"at EVERY step; the union is {entry['union_over_k']:.2f}x, so covering every "
                     f"step needs {entry['union_width'] * 100:.1f}% width "
                     f"(the budget allows {entry['kept_fraction'] * 100:.1f}%)")
    verdict = any(res[u].get("below_floor", False) for u in ("q", "mlp"))
    lines.append(f"  -> G2 {'ACCEPT (steps differ beyond sampling noise)' if verdict else 'REJECT'}\n")
    res["verdict"] = bool(verdict)
    return res


def clip_concentration(pc, ratio, lines, usable=None):
    """How much of the clip mean is one clip? (exploratory, NOT pre-registered)

    Found while checking whether the per-step mass profile was an artefact: it was. The
    clip axis is a mean, and at some steps a single clip out of 100 carries half the total
    |dL/dg|. That is a property of the shipped criterion itself, not of the step
    decomposition -- `traj_exp_*` is the same mean. Whether it matters depends on whether
    the outlier clips rank units differently from the rest, which is what the overlap
    between the mean-based and trimmed-based selections measures.
    """
    lines.append("clip concentration (exploratory, not pre-registered)")
    res = {}
    for unit in ("q", "mlp"):
        key = f"{unit}_abs_step"
        if pc is None or key not in pc or (usable is not None and not usable.get(unit, True)):
            continue
        a = pc[key].astype(np.float64)                    # (N, S, L, U)
        mass = a.sum((2, 3))                              # (N, S)
        share = mass.max(0) / mass.sum(0)
        sh_key = f"{unit}_shipped"
        trim_mass = np.sort(mass, 0)[: int(0.9 * len(mass))].mean(0)
        mean_mass = mass.mean(0)
        entry = {"top_clip_share_by_step": [float(x) for x in share],
                 "mass_mean_norm": [float(x) for x in mean_mass / mean_mass.max()],
                 "mass_trim10_norm": [float(x) for x in trim_mass / trim_mass.max()],
                 "mass_trim10_max_over_min": float(trim_mass.max() / trim_mass.min())}
        if sh_key in pc:
            s = pc[sh_key].astype(np.float64)             # (N, L, U) the shipped per-clip score
            mean_sel = s.mean(0)
            trim = np.sort(s, 0)[: int(0.9 * len(s))].mean(0)
            med = np.median(s, 0)
            entry["overlap_mean_vs_trim10"] = keep_overlap(mean_sel, trim, ratio)
            entry["overlap_mean_vs_median"] = keep_overlap(mean_sel, med, ratio)
            lines.append(f"  {unit:4s} top-clip share by step: " +
                         " ".join(f"{x:.2f}" for x in share))
            lines.append("       mass by step, clip mean:   " +
                         " ".join(f"{x:.2f}" for x in entry["mass_mean_norm"]))
            lines.append("       mass by step, 10%-trimmed: " +
                         " ".join(f"{x:.2f}" for x in entry["mass_trim10_norm"]) +
                         f"   (max/min {entry['mass_trim10_max_over_min']:.2f})")
            lines.append(f"       kept-overlap(clip mean, 10%-trimmed) {entry['overlap_mean_vs_trim10']:.3f}"
                         f"   vs median {entry['overlap_mean_vs_median']:.3f}")
        res[unit] = entry
    lines.append("")
    return res


def gate_g3(fm, inf, ratio, lines):
    lines.append("G3 path mismatch -- training path (FM) vs inference path (Euler)")
    res = {}
    for unit in ("q", "mlp"):
        a = fm[f"{unit}_abs_step"].astype(np.float64)
        b = inf[f"{unit}_abs_step"].astype(np.float64)
        per_step = [layer_spearman(a[s], b[s]) for s in range(min(a.shape[0], b.shape[0]))]
        agg = layer_spearman(a.sum(0), b.sum(0))
        ov = keep_overlap(a.sum(0), b.sum(0), ratio)
        res[unit] = {"per_step_rho": [float(x) for x in per_step],
                     "aggregate_rho": agg, "kept_overlap": ov}
        lines.append(f"  {unit:4s} aggregate rho {agg:.3f}  kept-overlap @r{ratio:.2f} {ov:.3f}")
        lines.append("       per-step rho " + " ".join(f"{x:.2f}" for x in per_step))
    verdict = min(res["q"]["aggregate_rho"], res["mlp"]["aggregate_rho"]) < 0.7
    lines.append(f"  -> G3 {'ACCEPT (paths disagree)' if verdict else 'REJECT (paths agree)'}\n")
    res["verdict"] = bool(verdict)
    return res


def gate_g4(sm, lines):
    """Step-limited masking: paired curves over s, on two readouts.

    `dev` (mean waypoint distance from the unmasked path at the same seed) is the primary
    readout because it measures the perturbation against its own control; `ade` is the task
    metric and is reported alongside. The verdict needs only one criterion's only_s curve to
    be step-dependent -- that is enough to say the steps are not interchangeable.
    """
    lines.append("G4 step-limited masking -- is the damage step-dependent?")
    rows, meta = sm["rows"], sm["meta"]
    names = sm["configs"]
    has_dev = "dev_k" in rows[0]["configs"]["baseline"]
    vals = {"ade": {n: np.array([min(r["configs"][n]["ade_k"]) for r in rows]) for n in names}}
    if has_dev:
        vals["dev"] = {n: np.array([np.mean(r["configs"][n]["dev_k"]) for r in rows])
                       for n in names}
    res = {"n_clips": len(rows), "baseline_minADE": float(vals["ade"]["baseline"].mean()),
           "readouts": list(vals), "configs": {}}

    for n in names:
        if n == "baseline":
            continue
        entry = {"kind": meta[n]["kind"],
                 **{k: v for k, v in meta[n].items() if k != "kind"}}
        for readout, v in vals.items():
            # dev is already a difference from the control, so it is not differenced again
            d = v[n] - v["baseline"] if readout == "ade" else v[n]
            m, lo, hi = el.paired_bootstrap_ci(d)
            entry[readout] = {"mean": m, "lo": lo, "hi": hi, "median": float(np.median(d))}
        res["configs"][n] = entry

    verdicts = []
    for kind in ("only", "except"):
        for crit in sorted({meta[n].get("criterion") for n in names
                            if meta[n]["kind"] == kind}):
            sel = sorted((meta[n]["step"], n) for n in names
                         if meta[n]["kind"] == kind and meta[n].get("criterion") == crit)
            if not sel:
                continue
            entry = {}
            for readout, v in vals.items():
                curve = [res["configs"][n][readout]["mean"] for _, n in sel]
                hi_n, lo_n = sel[int(np.argmax(curve))][1], sel[int(np.argmin(curve))][1]
                base_v = v["baseline"] if readout == "ade" else 0.0
                d = (v[hi_n] - base_v) - (v[lo_n] - base_v)
                m, lo, hi = el.paired_bootstrap_ci(d)
                p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
                # a ratio only means anything for a non-negative curve; dADE crosses zero,
                # so it gets the spread instead of a max/min that would read as 4e5
                ratio_ = float(max(curve) / min(curve)) if min(curve) > 0 else None
                entry[readout] = {
                    "curve": [float(x) for x in curve],
                    "max_over_min": ratio_, "spread": float(max(curve) - min(curve)),
                    "worst_step": int(meta[hi_n]["step"]), "best_step": int(meta[lo_n]["step"]),
                    "worst_minus_best": {"mean": m, "lo": lo, "hi": hi, "wilcoxon_p": p}}
                lines.append(f"  {kind}_{crit} [{readout}]: by step " +
                             " ".join(f"{x:+.3f}" for x in curve))
                lines.append(f"       spread {entry[readout]['spread']:.4f}" +
                             (f", max/min {ratio_:.2f}" if ratio_ is not None else "") +
                             f"  step {meta[hi_n]['step']} - step {meta[lo_n]['step']} = "
                             f"{m:+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.4f}")
            res[f"{kind}_{crit}"] = entry
            if kind == "only":
                primary = entry.get("dev", entry["ade"])
                verdicts.append(primary["max_over_min"] is not None
                                and primary["max_over_min"] >= 2.0
                                and primary["worst_minus_best"]["lo"] > 0)
            full = res["configs"].get(f"full_{crit}")
            if full:
                lines.append(f"       full mask (all steps) ade {full['ade']['mean']:+.4f} "
                             f"[{full['ade']['lo']:+.4f},{full['ade']['hi']:+.4f}]")
    # Is the step curve about WHICH units are masked, or just about where in the Euler
    # chain the perturbation lands? A late perturbation has fewer remaining steps to be
    # corrected by, so some rise with s is structural and shared by any mask. Two criteria
    # that select very different units but trace the same normalised shape point at the
    # chain, not at unit heterogeneity -- G2 is what answers that question, not G4.
    only_keys = [k for k in res if k.startswith("only_")]
    if len(only_keys) >= 2:
        shapes = {}
        for k in only_keys:
            for readout in ("dev", "ade"):
                if readout in res[k]:
                    c = np.array(res[k][readout]["curve"], dtype=float)
                    shapes.setdefault(readout, {})[k] = c / np.abs(c).mean()
        res["shape_agreement"] = {}
        for readout, d in shapes.items():
            ks = sorted(d)
            r = float(np.corrcoef(d[ks[0]], d[ks[1]])[0, 1])
            res["shape_agreement"][readout] = {"pearson": r,
                                               **{k: [float(x) for x in v] for k, v in d.items()}}
            lines.append(f"  shape agreement [{readout}] between {ks[0]} and {ks[1]}: r={r:.3f}")
            for k in ks:
                lines.append(f"       {k} normalised: " + " ".join(f"{x:.2f}" for x in d[k]))
        lines.append("       (high r = the curve follows chain position, which any mask shares;"
                     " unit heterogeneity is G2's question, not this one)")

    verdict = any(verdicts)
    lines.append(f"  -> G4 {'ACCEPT (damage depends on the step)' if verdict else 'REJECT'}\n")
    res["verdict"] = bool(verdict)
    return res


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------


def plot_all(out_dir, z_shared, pc, g2, g4, z_infer, conc=None):
    pd_ = out_dir / "plots"
    pd_.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, unit in zip(axes, ("q", "mlp")):
        M = np.array(g2[unit]["step_rho_matrix"])
        im = ax.imshow(M, cmap="RdYlBu_r", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(f"expert {unit}: step-to-step Spearman")
        ax.set_xlabel("denoising step")
        ax.set_ylabel("denoising step")
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    fig.tight_layout()
    fig.savefig(pd_ / "step_rho.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, unit in zip(axes, ("q", "mlp")):
        mass = np.array(g2[unit]["mass_per_step"])
        ax.plot(mass / mass.max(), "o-", color=C1, label="clip mean")
        if conc and unit in conc and "mass_trim10_norm" in conc[unit]:
            # the mean is dominated by single clips at some steps; the trimmed curve is
            # what the profile looks like without them, and the two disagree qualitatively
            ax.plot(conc[unit]["mass_trim10_norm"], "s--", color=C4, label="10%-trimmed")
        ax.set_title(f"expert {unit}: importance mass by step (normalised)")
        ax.set_xlabel("denoising step")
        ax.set_ylabel("sum_u |dL/dg|")
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(pd_ / "step_mass.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, unit in zip(axes, ("q", "mlp")):
        shipped = z_shared[f"{unit}_shipped"].ravel()
        sumabs = z_shared[f"{unit}_abs_step"].sum(0).ravel()
        ax.hist(shipped / (sumabs + EPS), bins=60, color=C3, edgecolor="none")
        ax.axvline(1.0, color=MUTED, lw=1, ls="--")
        ax.set_title(f"expert {unit}: cancellation ratio |sum g| / sum |g|")
        ax.set_xlabel("C_u")
        ax.set_ylabel("units")
    fig.tight_layout()
    fig.savefig(pd_ / "cancellation.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, unit in zip(axes, ("q", "mlp")):
        a = z_shared[f"{unit}_abs_step"]
        prof = a.sum(2)                                   # (S, L)
        prof = prof / (prof.max(axis=1, keepdims=True) + EPS)
        im = ax.imshow(prof.T, aspect="auto", cmap="Oranges", interpolation="nearest")
        ax.set_title(f"expert {unit}: depth profile per step (row-normalised)")
        ax.set_xlabel("denoising step")
        ax.set_ylabel("layer")
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    fig.tight_layout()
    fig.savefig(pd_ / "depth_by_step.png", dpi=150)
    plt.close(fig)

    if g4:
        readouts = [r for r in ("dev", "ade") if r in g4.get("readouts", ["ade"])]
        fig, axes = plt.subplots(1, len(readouts), figsize=(5.5 * len(readouts), 4),
                                 squeeze=False)
        titles = {"dev": ("path deviation from the unmasked run", "mean waypoint distance (m)"),
                  "ade": ("paired dminADE vs unmasked", "dminADE (m)")}
        for ax, readout in zip(axes[0], readouts):
            for key, color in (("only_traj", C1), ("only_magnitude", C2), ("except_traj", C4)):
                if key in g4 and readout in g4[key]:
                    ax.plot(g4[key][readout]["curve"], "o-", color=color, label=key)
            ax.axhline(0, color=MUTED, lw=1)
            ax.set_title(f"{titles[readout][0]}, mask at one step")
            ax.set_xlabel("denoising step")
            ax.set_ylabel(titles[readout][1])
            ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(pd_ / "step_damage.png", dpi=150)
        plt.close(fig)

    if z_infer is not None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        for ax, unit in zip(axes, ("q", "mlp")):
            for z_, lab, c in ((z_shared, "training path (FM)", C1),
                               (z_infer, "inference path (Euler)", C2)):
                m = z_[f"{unit}_abs_step"].sum((1, 2))
                ax.plot(m / m.max(), "o-", color=c, label=lab)
            ax.set_title(f"expert {unit}: mass by step, both paths")
            ax.set_xlabel("denoising step")
            ax.set_ylabel("normalised mass")
            ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(pd_ / "path_mass.png", dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fm-perstep", default=None, help="run with --noise-mode per_step (G0)")
    ap.add_argument("--fm-shared", required=True, help="run with --noise-mode shared")
    ap.add_argument("--infer", default=None)
    ap.add_argument("--stepmask", default=None)
    ap.add_argument("--importance", default="importance_v2_ada",
                    help="run_importance output measured on the SAME architecture (G0 reference)")
    ap.add_argument("--importance-alt", default=None,
                    help="a second run_importance output, reported as drift, never judged")
    ap.add_argument("--ratio", type=float, default=0.40)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    imp = dict(np.load(REPO / "outputs" / args.importance / "importance.npz"))
    z_shared, cfg_shared, pc_shared = load_run(args.fm_shared)

    lines = [
        f"Denoising-step decomposition -- {args.out}",
        (f"  fm-shared={args.fm_shared} fm-perstep={args.fm_perstep} "
         f"infer={args.infer} stepmask={args.stepmask}"),
        f"  n_clips={cfg_shared['num_clips']}  ratio={args.ratio}",
        "",
    ]
    res = {"args": vars(args)}

    if args.fm_perstep:
        z_ps, _, _ = load_run(args.fm_perstep)
        imp_alt = (dict(np.load(REPO / "outputs" / args.importance_alt / "importance.npz"))
                   if args.importance_alt else None)
        res["G0"] = gate_g0(z_ps, imp, imp_alt, lines)
    else:
        lines.append("G0 skipped (no --fm-perstep run given)\n")

    usable = perclip_fidelity(z_shared, pc_shared, lines)
    res["perclip_usable"] = usable
    res["G1"] = gate_g1(z_shared, args.ratio, lines)
    res["G2"] = gate_g2(z_shared, pc_shared, args.ratio, lines, usable=usable)
    res["clip_concentration"] = clip_concentration(pc_shared, args.ratio, lines, usable=usable)

    z_infer = None
    if args.infer:
        z_infer, _, _ = load_run(args.infer)
        res["G3"] = gate_g3(z_shared, z_infer, args.ratio, lines)

    g4 = None
    if args.stepmask:
        sm = json.loads((REPO / "outputs" / args.stepmask / "metrics.json").read_text())
        g4 = gate_g4(sm, lines)
        res["G4"] = g4

    plot_all(out_dir, z_shared, pc_shared, res["G2"], g4, z_infer,
             conc=res.get("clip_concentration"))
    (out_dir / "summary.txt").write_text("\n".join(lines))
    (out_dir / "metrics.json").write_text(json.dumps(res, indent=2))
    print("\n".join(lines))
    print("saved ->", out_dir)


if __name__ == "__main__":
    main()
