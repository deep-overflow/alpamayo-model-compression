"""How the two VLM objectives combine: raw scale, and the weight that balances them.

Section 1 -- raw scale.
    The two Taylor scores are |dL/dg| at g=1, so each carries its own loss's unit:
    reasoning NLL in nats, trajectory flow-matching in action-MSE. The ratio between
    them is therefore a scale convention, not a property of the network -- but it is
    large (FM is 6.3-10.5x CE) and, worse, it varies ~9-12x across depth, so no single
    global rescaling can stand in for the per-layer normalisation.

    The consequence is what matters: under max(raw), the trajectory objective decides
    94-96% of the retained units, i.e. `dual` would silently be `traj`. Under
    max(rank_norm) it is ~50/50. rank_norm is not cosmetic normalisation; it is what
    gives the two objectives equal say, which is what makes `max` a union.

Section 2 -- the balance as one hyperparameter.
    `dual` gives the two objectives equal say. That is a choice, not a constant of
    nature, and it can carry a knob. Two parameterisations are swept here; both run
    from traj through dual to coc, and all three endpoints are configs already built
    and evaluated at this budget, which is what makes a sweep interpretable.

    multiplicative:  max((1 - lam) * rank(FM), lam * rank(CE)),  lam in [0, 1]
    additive:        max(rank(FM), rank(CE) - delta),            delta in [-1, 1]

    Because select_mask_ratios only argsorts within a layer, and scaling a layer's
    whole score field by a positive constant is monotone, lam = 0 / 0.5 / 1 and
    delta = +1 / 0 / -1 reproduce traj_u40_v2 / dual_u40_v2 / coc_u40_v2 exactly
    (the module asserts this by overlap, which must print 1.000000).

    **The multiplicative form is a switch, not a dial** -- ranks are uniform on [0, 1],
    so the comparison reduces to r_coc/r_traj against (1-lam)/lam, and over the
    retained set that ratio only spans about [0.4, 2.5]. Everything happens inside
    lam in [0.4, 0.6]; outside it the criterion saturates to a single objective.
    The additive form shifts a uniform field by a constant instead of rescaling it,
    which is why its decision share moves smoothly across delta in [-0.4, +0.4].
    Use the additive one if this ever becomes a real knob.

    This module measures the SELECTION side only -- how far the retained set travels
    and how the decision share moves. Whether an interior setting buys anything needs
    a build and an eval, and see the warning that prints after the sweeps.

Usage:
  .venv/bin/python experiments/paper/criterion_scale.py
"""

import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
IMP = REPO / "outputs" / "importance_v2" / "importance.npz"

# shipped *_u40_v2 budget: uniform 0.3985632694
KEEP = {"q": (19, 32), "mlp": (7390, 12288)}
LAMBDAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
DELTAS = (-1.0, -0.6, -0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4, 0.6, 1.0)


def rank_norm(a):
    """Per-layer rank mapped to [0, 1]. A monotone map, so it is a no-op for a single
    criterion and only bites when two fields are combined."""
    out = np.empty_like(a, dtype=np.float64)  # (L, U)
    for l in range(a.shape[0]):
        out[l] = np.argsort(np.argsort(a[l])) / (a.shape[1] - 1)
    return out


def kept(score, keep):
    """Retained index set per layer, exactly as select_mask_ratios argsorts."""
    return [set(np.argsort(-score[l])[:keep]) for l in range(score.shape[0])]


def overlap(a, b, keep):
    return float(np.mean([len(x & y) / keep for x, y in zip(a, b)]))


def scale_section(t, c, axis, keep, n):
    print("=" * 78)
    print(f"VLM {axis}   (36 layers x {n} units)   keep {keep}/{n}")
    print("=" * 78)
    for name, a in (("FM (I_traj)", t), ("CE (I_CoC )", c)):
        print(f"  {name}: mean {a.mean():.4e}  median {np.median(a):.4e}"
              f"  max {a.max():.4e}  sum {a.sum():.4e}")
    print(f"  FM / CE : mean {t.mean() / c.mean():.2f}x  median "
          f"{np.median(t) / np.median(c):.2f}x  max {t.max() / c.max():.2f}x")

    tl, cl = t.sum(1), c.sum(1)
    nz = tl > 0
    r = tl[nz] / cl[nz]
    lay = np.arange(t.shape[0])[nz]
    print(f"  per-layer FM/CE over the {int(nz.sum())} layers with I_traj > 0:"
          f"  min {r.min():.2f}x (L{lay[r.argmin()]})  median {np.median(r):.2f}x"
          f"  max {r.max():.2f}x (L{lay[r.argmax()]})  -> spread {r.max() / r.min():.1f}x")
    print(f"  layers where CE > FM: {np.arange(t.shape[0])[cl > tl].tolist()}"
          f"   (layer 35's I_traj is structurally zero)")
    k = np.median(r)
    print(f"  after the best single rescale ({k:.2f}x), per-layer FM/(k*CE) still spans"
          f" {r.min() / k:.2f}x .. {r.max() / k:.2f}x -- a global constant cannot do it")

    rt, rc = rank_norm(t), rank_norm(c)
    for label, ft, fc in (("raw ", t, c), ("rank", rt, rc)):
        s = np.maximum(ft, fc)
        won = sum(int((fc[l, list(ix)] > ft[l, list(ix)]).sum())
                  for l, ix in enumerate(kept(s, keep)))
        tot = keep * s.shape[0]
        print(f"  max({label}): CoC decides {won / tot * 100:5.1f}% of retained units,"
              f"  FM decides {100 - won / tot * 100:5.1f}%")
    print(f"  retained-set overlap max(raw) vs max(rank): "
          f"{overlap(kept(np.maximum(t, c), keep), kept(np.maximum(rt, rc), keep), keep) * 100:.1f}%"
          f"   (chance {keep / n * 100:.1f}%)")
    print()


def sweep(name, fields, knobs, mid, keep, n, rt, rc, lo_end="traj"):
    """Print one parameterisation's path from traj through dual to coc.

    `fields(k)` returns the (weighted FM, weighted CE) pair at knob value k, so the
    decision share is read off the same quantities the max compares."""
    ref = {k: kept(np.maximum(*fields(k)), keep) for k in knobs}
    dual = kept(np.maximum(rt, rc), keep)
    traj, coc = kept(rt, keep), kept(rc, keep)
    lo, hi = (traj, coc) if lo_end == "traj" else (coc, traj)
    ln, hn = (lo_end, "coc" if lo_end == "traj" else "traj")
    print(f"  {name}")
    print(f"    identity: {knobs[0]:+.2f} vs {ln} {overlap(ref[knobs[0]], lo, keep):.6f}"
          f"   {mid:+.2f} vs dual {overlap(ref[mid], dual, keep):.6f}"
          f"   {knobs[-1]:+.2f} vs {hn} {overlap(ref[knobs[-1]], hi, keep):.6f}"
          f"   (all must be 1.000000)")
    print(f"    {'knob':>6}  {'vs dual':>8}  {'vs traj':>8}  {'vs coc':>8}  {'CE decides':>11}")
    for k in knobs:
        ft, fc = fields(k)
        won = sum(int((fc[l, list(ix)] > ft[l, list(ix)]).sum())
                  for l, ix in enumerate(ref[k]))
        print(f"    {k:+6.2f}  {overlap(ref[k], dual, keep) * 100:7.1f}%"
              f"  {overlap(ref[k], traj, keep) * 100:7.1f}%"
              f"  {overlap(ref[k], coc, keep) * 100:7.1f}%"
              f"  {won / (keep * rt.shape[0]) * 100:10.1f}%")
    print()


def lambda_section(t, c, axis, keep, n):
    rt, rc = rank_norm(t), rank_norm(c)
    print("=" * 78)
    print(f"balance knob -- VLM {axis}   (chance overlap {keep / n * 100:.1f}%)")
    print("=" * 78)
    sweep("multiplicative:  max((1-lam)*rank(FM), lam*rank(CE))",
          lambda k: ((1 - k) * rt, k * rc), LAMBDAS, 0.5, keep, n, rt, rc)
    sweep("additive:        max(rank(FM), rank(CE) - delta)",
          lambda k: (rt, rc - k), DELTAS, 0.0, keep, n, rt, rc, lo_end="coc")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--importance", type=Path, default=IMP)
    a = ap.parse_args()

    z = np.load(a.importance)
    print(f"importance: {a.importance}\n")
    for axis, (keep, n) in KEEP.items():
        t = z[f"traj_vlm_{axis}"].astype(np.float64)  # (36, U)
        c = z[f"coc_vlm_{axis}"].astype(np.float64)  # (36, U)
        scale_section(t, c, axis, keep, n)
        lambda_section(t, c, axis, keep, n)

    print("Selecting lambda on an evaluation set is the tau-tuning trap: both rank fields")
    print("are estimated from the same 100 calibration clips, whose draw alone moves")
    print("val500 by +0.029..+0.451 m (2026-09-03_calib-draw-variance). Before sweeping,")
    print("check that the lambda -> retained-set map is stable across calibration draws;")
    print("if it is not, the best lambda is a property of the draw, not of the criterion.")


if __name__ == "__main__":
    main()
