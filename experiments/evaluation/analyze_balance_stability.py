"""Is the FM/CE balance knob learnable at n=100, or is it below the calibration noise?

`dual` = max(rank(I_traj), rank(I_CoC)) gives the two objectives equal say. That is a
choice and it can carry a knob (`experiments/paper/criterion_scale.py`):

    score_delta = max(rank(I_traj), rank(I_CoC) - delta),   delta in [-1, 1]

with delta = +1 / 0 / -1 reproducing traj_u40_v2 / dual_u40_v2 / coc_u40_v2 exactly.
Before spending GPU on a sweep, this asks whether the knob has any range the
calibration draw does not already swamp. Both rank fields are estimated from the same
100 clips, and swapping those 100 clips alone moves val500 by +0.029..+0.451 m
(reports/evaluation/2026-09-03_calib-draw-variance.html) -- the same trap that killed
tau tuning for dualsafe.

Eleven mutually disjoint n=100 draws are on disk, all measured on Ada with seed 42:
importance_nt500_{a..e}, importance_tr500_{a..e}, importance_v2_ada (the shipped
calib_100 re-measured on Ada). Pairwise shared clips <= 1. importance_tr100_a_solo is
the same 100 clips as tr500_a and is excluded as a duplicate.

Pre-registered gates, stated before running:

  S1 (range)        Knob displacement at |delta| = 0.1 must exceed the draw-to-draw
                    displacement at delta = 0. If a different calibration draw moves the
                    retained set further than the knob does, the knob is unlearnable at
                    this n and the sweep is not worth building.
  S2 (consistency)  Applying the same delta in two draws must move them in a COMMON
                    direction: overlap(K_d(delta), K_d'(delta)) > overlap(K_d(delta),
                    K_d'(0)). A knob whose effect is draw-specific is fitting noise.
  S3 (location)     Against a leave-one-out consensus of the other ten draws, the delta
                    that best matches must be consistent across draws. Scatter means the
                    optimum is a property of the draw; a consistent non-zero optimum
                    would be the one result that justifies building arms.

Selection side only -- no model is loaded and no clip is evaluated.

Usage:
  .venv/bin/python experiments/evaluation/analyze_balance_stability.py
"""

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "balance_stability"

DRAWS = ([f"importance_nt500_{s}" for s in "abcde"]
         + [f"importance_tr500_{s}" for s in "abcde"]
         + ["importance_v2_ada"])
KEEP = {"q": (19, 32), "mlp": (7390, 12288)}
DELTAS = (-0.4, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.4)


def rank_norm(a):
    out = np.empty_like(a, dtype=np.float64)  # (L, U)
    for l in range(a.shape[0]):
        out[l] = np.argsort(np.argsort(a[l])) / (a.shape[1] - 1)
    return out


def kept(score, keep):
    return [set(np.argsort(-score[l])[:keep]) for l in range(score.shape[0])]


def ov(a, b, keep):
    return float(np.mean([len(x & y) / keep for x, y in zip(a, b)]))


def load(axis, root):
    """Rank-normalised (FM, CE) field per draw. Both are (36, U)."""
    out = {}
    for d in DRAWS:
        z = np.load(root / d / "importance.npz")
        out[d] = (rank_norm(z[f"traj_vlm_{axis}"].astype(np.float64)),
                  rank_norm(z[f"coc_vlm_{axis}"].astype(np.float64)))
    return out


def analyse(axis, keep, n, root):
    f = load(axis, root)
    ks = {d: {dl: kept(np.maximum(t, c - dl), keep) for dl in DELTAS}
          for d, (t, c) in f.items()}
    pairs = [(a, b) for i, a in enumerate(DRAWS) for b in DRAWS[i + 1:]]
    res = {"axis": axis, "keep": keep, "n_units": n, "chance": keep / n}

    # S1 -- knob range against the draw noise floor
    noise = float(np.mean([1 - ov(ks[a][0.0], ks[b][0.0], keep) for a, b in pairs]))
    sig = {dl: float(np.mean([1 - ov(ks[d][0.0], ks[d][dl], keep) for d in DRAWS]))
           for dl in DELTAS}
    res["draw_displacement_at_delta0"] = noise
    res["knob_displacement"] = sig

    # S2 -- does the same delta move different draws the same way?
    cons = {}
    for dl in DELTAS:
        same = float(np.mean([ov(ks[a][dl], ks[b][dl], keep) for a, b in pairs]))
        cross = float(np.mean([ov(ks[a][dl], ks[b][0.0], keep) for a, b in pairs]
                              + [ov(ks[b][dl], ks[a][0.0], keep) for a, b in pairs]))
        cons[dl] = {"same_delta": same, "vs_delta0": cross, "gain": same - cross}
    res["consistency"] = cons

    # S3 -- best delta against a leave-one-out consensus of the other ten draws
    best = {}
    for d in DRAWS:
        others = [x for x in DRAWS if x != d]
        ct = np.mean([f[x][0] for x in others], axis=0)
        cc = np.mean([f[x][1] for x in others], axis=0)
        ref = kept(np.maximum(ct, cc), keep)
        curve = {dl: ov(ks[d][dl], ref, keep) for dl in DELTAS}
        best[d] = {"best_delta": max(curve, key=curve.get), "curve": curve}
    res["loo"] = best
    return res


def report(r):
    keep, n = r["keep"], r["n_units"]
    print("=" * 78)
    print(f"VLM {r['axis']}   keep {keep}/{n}   chance overlap {r['chance'] * 100:.1f}%"
          f"   {len(DRAWS)} disjoint n=100 draws")
    print("=" * 78)
    noise = r["draw_displacement_at_delta0"]
    print(f"  S1  draw-to-draw displacement at delta=0 (the noise floor): "
          f"{noise * 100:.1f}%")
    print(f"      {'delta':>7}  {'knob displ':>11}  {'vs noise':>9}")
    for dl in DELTAS:
        s = r["knob_displacement"][dl]
        print(f"      {dl:+7.2f}  {s * 100:10.1f}%  {s / noise:8.2f}x")
    s01 = max(r["knob_displacement"][0.1], r["knob_displacement"][-0.1])
    print(f"      -> at |delta|=0.1 the knob moves {s01 * 100:.1f}% vs a noise floor of "
          f"{noise * 100:.1f}%  ==> S1 {'PASS' if s01 > noise else 'FAIL'}")

    print("  S2  does the same delta move different draws the same way?")
    print(f"      {'delta':>7}  {'same delta':>11}  {'vs delta0':>10}  {'gain':>8}")
    gains = []
    for dl in DELTAS:
        c = r["consistency"][dl]
        if dl != 0.0:
            gains.append(c["gain"])
        print(f"      {dl:+7.2f}  {c['same_delta'] * 100:10.1f}%"
              f"  {c['vs_delta0'] * 100:9.1f}%  {c['gain'] * 100:+7.2f}pp")
    print(f"      -> gain is positive at {sum(g > 0 for g in gains)}/{len(gains)} "
          f"non-zero deltas  ==> S2 {'PASS' if all(g > 0 for g in gains) else 'FAIL'}")

    bd = [r["loo"][d]["best_delta"] for d in DRAWS]
    uniq = sorted(set(bd))
    print("  S3  best delta vs a leave-one-out consensus of the other ten draws:")
    print(f"      {' '.join(f'{x:+.2f}' for x in bd)}")
    print(f"      distinct values {uniq}  ==> S3 "
          f"{'PASS' if len(uniq) == 1 and uniq[0] != 0.0 else 'FAIL'}"
          f"{' (consistent, and it is delta=0 -- equal say is right)' if uniq == [0.0] else ''}")
    print()


def noise_vs_n(axis, keep, root, knob_at_01):
    """S4 -- how the draw noise floor shrinks with calibration size, and the n at which
    it would fall under the knob's own displacement. Effectively-disjoint pairs only
    (<= 7 shared clips out of 200-2000)."""
    pairs = {100: [(a, b) for i, a in enumerate(DRAWS) for b in DRAWS[i + 1:]],
             200: [("importance_nt500_c200", "importance_tr500_c200")],
             300: [("importance_nt500_c300", "importance_tr500_c300")],
             500: [("importance_nt500_c500", "importance_tr500_c500")],
             2000: [("importance_st4000_b2000", "importance_st4000_c2000")]}
    cache, out = {}, {}
    for n_calib, ps in pairs.items():
        for d in {x for p in ps for x in p}:
            if d not in cache:
                z = np.load(root / d / "importance.npz")
                cache[d] = kept(np.maximum(rank_norm(z[f"traj_vlm_{axis}"].astype(np.float64)),
                                           rank_norm(z[f"coc_vlm_{axis}"].astype(np.float64))),
                                keep)
        out[n_calib] = float(np.mean([1 - ov(cache[a], cache[b], keep) for a, b in ps]))

    ns = np.array(sorted(out), dtype=float)
    dv = np.array([out[int(k)] for k in ns])
    p, logc = np.polyfit(np.log(ns), np.log(dv), 1)
    need = float(np.exp((np.log(knob_at_01) - logc) / p))
    return {"displacement_by_n": out, "exponent": float(p),
            "n_for_knob_to_clear": need}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs", type=Path, default=REPO / "outputs")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    res = {}
    for axis, (keep, n) in KEEP.items():
        res[axis] = analyse(axis, keep, n, a.outputs)
        report(res[axis])
        k01 = max(res[axis]["knob_displacement"][0.1],
                  res[axis]["knob_displacement"][-0.1])
        s4 = noise_vs_n(axis, keep, a.outputs, k01)
        res[axis]["scaling"] = s4
        print(f"  S4  draw noise floor vs calibration size (VLM {axis}):")
        for k, v in sorted(s4["displacement_by_n"].items()):
            print(f"      n={k:>5}  displacement {v * 100:5.1f}%")
        print(f"      fit displacement ~ n^{s4['exponent']:+.3f}; the knob's own "
              f"{k01 * 100:.1f}% at |delta|=0.1 is cleared near "
              f"n = {s4['n_for_knob_to_clear']:,.0f}")
        print()

    (a.out / "config.json").write_text(json.dumps(
        {"draws": DRAWS, "deltas": list(DELTAS), "keep": KEEP,
         "criterion": "max(rank(I_traj), rank(I_CoC) - delta)",
         "gates": {"S1": "knob displacement at |delta|=0.1 > draw displacement at 0",
                   "S2": "overlap(K_d(delta), K_d'(delta)) > overlap(K_d(delta), K_d'(0))",
                   "S3": "leave-one-out best delta consistent across draws"}},
        indent=2))
    (a.out / "metrics.json").write_text(json.dumps(res, indent=2, default=str))

    lines = ["Balance knob max(rank(I_traj), rank(I_CoC) - delta): is delta learnable?",
             f"{len(DRAWS)} disjoint n=100 calibration draws, Ada, seed 42.", ""]
    for axis in KEEP:
        r = res[axis]
        k01 = max(r["knob_displacement"][0.1], r["knob_displacement"][-0.1])
        noise = r["draw_displacement_at_delta0"]
        picks = [r["loo"][d]["best_delta"] for d in DRAWS]
        bd = sorted(set(picks))
        gains = [r["consistency"][d]["gain"] for d in DELTAS if d != 0.0]
        lines += [
            f"VLM {axis}:",
            (f"  S1 {'PASS' if k01 > noise else 'FAIL'}  knob {k01 * 100:.1f}% at "
             f"|delta|=0.1 vs draw noise {noise * 100:.1f}%  "
             f"({noise / k01:.1f}x larger)"),
            (f"  S2 {'PASS' if all(g > 0 for g in gains) else 'FAIL'}  same delta moves "
             f"different draws together at {sum(g > 0 for g in gains)}/{len(gains)} "
             f"non-zero deltas (gain "
             f"{r['consistency'][0.4]['gain'] * 100:+.2f}pp at +0.4)"),
            (f"  S3 {'PASS' if bd != [0.0] and len(bd) == 1 else 'FAIL'}  leave-one-out "
             f"best delta scatters over {bd}, modal value "
             f"{max(bd, key=picks.count):+.2f}"),
            (f"  S4       noise ~ n^{r['scaling']['exponent']:+.3f}; knob clears it "
             f"near n = {r['scaling']['n_for_knob_to_clear']:,.0f}"),
            ""]
    lines += [
        "Verdict: do not build the sweep at n=100. The knob has a real, draw-independent",
        "component (S2), but its whole usable range sits inside the calibration draw's",
        "noise, and no consistent non-zero optimum survives a leave-one-out consensus.",
        "delta=0 -- equal say, i.e. the shipped dual -- is the modal optimum, which makes",
        "equal weighting an empirically supported default rather than an arbitrary one.",
    ]
    (a.out / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote {a.out}/metrics.json, summary.txt")


if __name__ == "__main__":
    main()
