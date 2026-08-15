"""Per-row bit allocation under a projected-storage budget.

QVLA solves this with a staged greedy demotion: start every channel at 16 bits, walk the
stages 16->8->8->4->4->2->2->0 in ascending order of the marginal ratio
`rho = (s_lo - s_hi) / (b_hi - b_lo)`, and stop when the average bit budget is met.
That is degenerate at exactly the budget we were asked for. A target of 8 average bits is
reached the instant the 16->8 stage finishes, so every row lands on 8 and the result is
bit-identical to uniform W8 -- no mixed precision at all, and nothing for the criterion
to express.

The fix is to allocate against the budget directly instead of walking fixed stages. With
one bit-width per row drawn from a small menu this is a multiple-choice knapsack, whose
LP relaxation is solved exactly by a Lagrangian sweep: for a price `lam` on storage each
row independently takes `argmin_b (damage[b] + lam * cost[b])`, total cost falls
monotonically in `lam`, so bisecting `lam` lands on the budget. The integrality gap of
MCKP is at most one item and there are 1.86 M rows here, so the relaxation is the answer
for practical purposes. Unlike the staged greedy it can hold a sensitive row at 16 while
paying for it with rows at 4, 2 or 0, which is the whole point of a criterion.

`damage[b]` is the CoC directional sensitivity from `run_qsens_coc.py`, in units of CoC
NLL for every tensor alike, so scores are comparable across modules without any
normalisation (QVLA makes the same observation about its own metric). `damage[16] = 0`
exactly, because `eps = Q_16(W) - W = 0`.
"""

import json

import numpy as np
import quant_lib as ql

# Quantization menu. QVLA includes 0-bit so that allocation and structured pruning share
# one budget, and regularises the 2->0 stage with dual-threshold / L0 constraints because
# over-pruning is a known failure of that design. Here 0-bit is off by default for a
# sharper reason: the CoC NLL gradient reaches an lm_head row only through the tokens the
# calibration clips actually use, so the other ~155 k vocabulary rows score at the noise
# floor (measured median 1.9e-8) and a 0-bit option prunes almost all of them. That does
# not mean they are unimportant -- it means 100 clips never exercised them, and a model
# that cannot emit any token outside the calibration support is a calibration artefact,
# not a compression result. `--allow-prune` restores the full QVLA menu.
FULL_MENU = (0, 2, 4, 8, 16)
QUANT_MENU = (2, 4, 8, 16)
MENU = QUANT_MENU


def load_scores(npz_path, menu=MENU):
    """-> (names, shapes {name: (O, I)}, S {name: (O, len(MENU)) damage})."""
    z = np.load(npz_path, allow_pickle=True)
    names = [str(n) for n in z["_names"]]
    bits = [int(b) for b in z["_bits"]]
    shapes = {n: tuple(int(x) for x in s) for n, s in zip(names, z["_shapes"])}
    S = {}
    for n in names:
        O = shapes[n][0]
        d = np.zeros((O, len(menu)), dtype=np.float64)
        for j, b in enumerate(menu):
            if b in bits:
                d[:, j] = z[f"{n}|{b}"]
            # b == 16 stays 0: eps is identically zero, so the score is too
        S[n] = d
    return names, shapes, S


def cost_table(names, shapes, group, menu=MENU):
    """-> (R, len(MENU)) projected storage cost in bits, row-aligned with damage."""
    per_tensor = []
    for n in names:
        O, I = shapes[n]
        c = np.array([ql.row_bits_cost(I, b, group) for b in menu], dtype=np.float64)
        per_tensor.append(np.broadcast_to(c, (O, len(menu))))
    return np.concatenate(per_tensor, 0)


def uniform_budget(names, shapes, bits, group):
    """Total projected bits if every row in the pool sat at `bits`."""
    return float(sum(O * ql.row_bits_cost(I, bits, group) for O, I in
                     (shapes[n] for n in names)))


def allocate(damage, cost, budget, iters=200, n_menu=None):
    """Lagrangian MCKP. -> (choice index per row, total cost, lam).

    Cost is non-increasing in lam, so bisect in log space. The returned solution is
    feasible (cost <= budget); the residual is spent afterwards by upgrading the rows
    with the best damage-reduction-per-bit still available.
    """
    lo, hi = 1e-30, 1e30                       # lam: damage units per bit of storage
    choice = None
    for _ in range(iters):
        lam = np.sqrt(lo * hi)
        j = np.argmin(damage + lam * cost, axis=1)
        c = cost[np.arange(len(j)), j].sum()
        if c > budget:
            lo = lam                           # storage too expensive -> raise its price
        else:
            hi = lam
            choice = j
    if choice is None:                         # even the cheapest menu exceeds budget
        choice = np.argmin(cost, axis=1)

    # spend what bisection left on the table: single-step upgrades, best ratio first
    total = cost[np.arange(len(choice)), choice].sum()
    slack = budget - total
    if slack > 0:
        cur_c = cost[np.arange(len(choice)), choice]
        cur_d = damage[np.arange(len(choice)), choice]
        up = np.minimum(choice + 1, (n_menu or damage.shape[1]) - 1)
        dc = cost[np.arange(len(choice)), up] - cur_c
        dd = cur_d - damage[np.arange(len(choice)), up]
        gain = np.where(dc > 0, dd / np.maximum(dc, 1e-30), -np.inf)
        for r in np.argsort(-gain):
            if gain[r] <= 0 or dc[r] > slack:
                continue
            choice[r] = up[r]
            slack -= dc[r]
            if slack <= 0:
                break
        total = cost[np.arange(len(choice)), choice].sum()
    return choice, float(total), float(hi)


def spec_from_choice(names, shapes, choice, menu=MENU):
    """Split the flat row-choice vector back into {name: per-row bit array}."""
    spec, off = {}, 0
    for n in names:
        O = shapes[n][0]
        spec[n] = np.array([menu[j] for j in choice[off:off + O]], dtype=np.int64)
        off += O
    assert off == len(choice), (off, len(choice))
    return spec


def summarize(names, shapes, spec, group, menu=MENU):
    """Bit histogram, projected bytes and per-part breakdown."""
    import torch
    eff, tot_bits, tot_par = ql.effective_bits(
        {n: shapes[n] for n in names}, {n: torch.from_numpy(spec[n]) for n in names}, group)
    hist = {b: 0 for b in menu}
    part_bits = {}
    for n in names:
        for b in menu:
            hist[b] += int((spec[n] == b).sum())
        part = ("lm_head" if "lm_head" in n else
                "vit" if ".visual." in n else "vlm_text")
        O, I = shapes[n]
        pb = part_bits.setdefault(part, [0, 0])
        pb[0] += sum(int((spec[n] == b).sum()) * ql.row_bits_cost(I, b, group) for b in menu)
        pb[1] += O * I
    return {
        "effective_bits": eff,
        "projected_pool_gb": tot_bits / 8 / 1e9,
        "pool_bf16_gb": tot_par * 2 / 1e9,
        "row_bit_hist": hist,
        "row_bit_frac": {b: hist[b] / sum(hist.values()) for b in menu},
        "per_part_effective_bits": {k: v[0] / v[1] for k, v in part_bits.items()},
    }


def main():
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument("--qsens", default="qsens_coc_v1", help="outputs/<id>/qsens.npz")
    ap.add_argument("--match-uniform", type=int, default=8,
                    help="budget = what uniform W<this> costs over the same pool")
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--out", default="outputs/quant_specs")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--allow-prune", action="store_true",
                    help="add 0-bit to the menu (full QVLA); see the note on lm_head")
    args = ap.parse_args()

    menu = FULL_MENU if args.allow_prune else QUANT_MENU
    repo = Path(__file__).resolve().parents[2]
    names, shapes, S = load_scores(repo / "outputs" / args.qsens / "qsens.npz", menu)
    damage = np.concatenate([S[n] for n in names], 0)
    cost = cost_table(names, shapes, args.group, menu)
    budget = uniform_budget(names, shapes, args.match_uniform, args.group)

    choice, total, lam = allocate(damage, cost, budget, n_menu=len(menu))
    spec = spec_from_choice(names, shapes, choice, menu)
    tag = args.tag or f"qvla_coc_b{args.match_uniform}"

    out = repo / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{tag}.npz", **spec, _names=np.array(names, dtype=object))

    uni = {n: np.full(shapes[n][0], args.match_uniform, dtype=np.int64) for n in names}
    np.savez(out / f"uniform_w{args.match_uniform}.npz", **uni,
             _names=np.array(names, dtype=object))

    meta = {
        "tag": tag, "qsens": args.qsens, "group": args.group,
        "criterion": "CoC NLL directional sensitivity (no trajectory loss)",
        "menu": list(menu), "budget_bits": budget, "used_bits": total,
        "budget_matched_to": f"uniform W{args.match_uniform}",
        "lambda": lam, "n_tensors": len(names), "n_rows": len(choice),
        "allocated": summarize(names, shapes, spec, args.group, menu),
        "uniform": summarize(names, shapes, uni, args.group, menu),
    }
    (out / f"{tag}.json").write_text(json.dumps(meta, indent=2))

    a, u = meta["allocated"], meta["uniform"]
    print(f"budget {budget / 8 / 1e9:.3f} GB (uniform W{args.match_uniform}), "
          f"used {total / 8 / 1e9:.3f} GB ({100 * total / budget:.2f}%)")
    print(f"  allocated: {a['effective_bits']:.3f} eff bits, "
          f"{a['projected_pool_gb']:.3f} GB  (pool bf16 {a['pool_bf16_gb']:.2f} GB)")
    print(f"  uniform  : {u['effective_bits']:.3f} eff bits, {u['projected_pool_gb']:.3f} GB")
    print("  row bits : " + "  ".join(
        f"{b}b {100 * a['row_bit_frac'][b]:.1f}%" for b in menu))
    print("  per part : " + "  ".join(
        f"{k} {v:.2f}b" for k, v in a["per_part_effective_bits"].items()))
    print(f"-> {out / (tag + '.npz')}")


if __name__ == "__main__":
    main()
