"""Draw the two evaluation sets and record how well each matches the source distribution.

Two sets, disjoint by construction:

  in-distribution  official val, cached, minus every clip that appears anywhere in
                   ood_reasoning.parquet and minus the calibration clips, then N drawn
                   by greedy distribution matching against the full-val clip-level
                   distribution (same six weighted attributes the chunk stage used).
  OOD              whatever build_ood_cache.py managed to cache, carrying each clip's
                   event cluster, curated CoC and official split.

All 1,740 OOD clips are excluded from the in-distribution pool, not just the 290 that
sit in val, so the two sets cannot overlap. Calibration clips are excluded because the
pruning criterion is fitted on them.

The greedy draw adds, at each step, the clip that most reduces the weighted L1 distance
to the target. One clip moves exactly six cells (one per attribute), so every
candidate's delta is evaluated vectorised over the pool. Ties break through a seeded
RNG, so the draw is reproducible.

See SAMPLING.md for the measured quality of each stage and the reasoning behind N.

Usage:
  python experiments/evaluation/make_eval_sets.py --n-indist 500 --seed 42
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
AV = Path("/mnt/nvme1n1/ad_vla/data/physicalai_av")
PRE = AV / "pre_processed"

# same attributes and weights as the chunk-level manifest
ATTR = {"country": 4.0, "platform_class": 2.0, "time_of_day": 2.0,
        "season": 1.5, "month": 1.0, "radar_config": 1.0}

BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})


def derive(df, dc):
    """Attach the six matching attributes; time_of_day and season are derived."""
    d = df.join(dc[["country", "platform_class", "radar_config", "month", "hour_of_day"]])
    d["time_of_day"] = np.where((d.hour_of_day >= 6) & (d.hour_of_day < 18),
                                "daytime", "nighttime")
    d["season"] = d["month"].map(
        lambda m: "winter" if m in (12, 1, 2) else "spring" if m in (3, 4, 5)
        else "summer" if m in (6, 7, 8) else "fall")
    return d


def encode(full, pool):
    """Integer cell index per pool clip, and the target proportion per cell."""
    codes, targets = {}, {}
    for a in ATTR:
        # pyarrow-backed columns keep mixed types through astype(str); map() forces str
        fs, ps = full[a].map(str), pool[a].map(str)
        cats = sorted(set(fs) | set(ps))
        codes[a] = ps.map({c: i for i, c in enumerate(cats)}).to_numpy()
        t = fs.value_counts(normalize=True)
        targets[a] = np.array([t.get(c, 0.0) for c in cats])
    return codes, targets


def l1_per_attr(counts, n, targets):
    return {a: float(np.abs(targets[a] - counts[a] / n).sum()) for a in ATTR}


def jsd_per_attr(counts, n, targets):
    """Jensen-Shannon divergence, base 2, per attribute."""
    out = {}
    for a in ATTR:
        p, q = targets[a], counts[a] / n
        m = 0.5 * (p + q)

        def kl(x, y):
            nz = x > 0
            return float((x[nz] * np.log2(x[nz] / np.where(y[nz] > 0, y[nz], 1e-12))).sum())
        out[a] = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return out


def weighted_l1(counts, n, targets):
    per = l1_per_attr(counts, n, targets)
    return sum(w * per[a] for a, w in ATTR.items()) / sum(ATTR.values())


def greedy(codes, targets, n_pool, n_sel, seed):
    """Pick n_sel indices, each step taking the clip that most cuts the weighted L1."""
    rng = np.random.default_rng(seed)
    counts = {a: np.zeros(len(targets[a])) for a in ATTR}
    chosen = np.zeros(n_pool, dtype=bool)
    order = []
    for k in range(1, n_sel + 1):
        score = np.zeros(n_pool)
        for a, w in ATTR.items():
            cur = counts[a]
            resid = np.abs(targets[a] - cur / k)        # per-cell L1 if nothing added
            base = resid.sum()
            # adding one clip to cell c only changes cell c's term
            after = base - resid[codes[a]] + np.abs(targets[a][codes[a]] - (cur[codes[a]] + 1) / k)
            score += w * after
        score[chosen] = np.inf
        tied = np.flatnonzero(score == score.min())
        pick = int(tied[rng.integers(len(tied))])
        chosen[pick] = True
        order.append(pick)
        for a in ATTR:
            counts[a][codes[a][pick]] += 1
    return np.array(order), counts


def random_reference(codes, targets, n_pool, n_sel, seed, n_draws=20):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_draws):
        idx = rng.choice(n_pool, n_sel, replace=False)
        c = {a: np.bincount(codes[a][idx], minlength=len(targets[a])).astype(float)
             for a in ATTR}
        vals.append(weighted_l1(c, n_sel, targets))
    return float(np.mean(vals)), float(np.std(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-indist", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exp-id", default="eval_sets")
    ap.add_argument("--split", default="val", choices=["val", "test", "train"],
                    help="official split to draw from; train yields the calibration set")
    ap.add_argument("--sweep", type=int, nargs="+", default=None,
                    help="sizes to report in quality.csv (the chosen N is added); "
                         "defaults to eval sizes, or calibration sizes for --split train")
    args = ap.parse_args()

    out_dir = REPO / "outputs" / args.exp_id
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    ci = pd.read_parquet(AV / "clip_index.parquet")
    dc = pd.read_parquet(AV / "metadata" / "data_collection.parquet")
    ood_src = pd.read_parquet(AV / "reasoning" / "ood_reasoning.parquet")
    split = json.loads((REPO / "outputs" / "split.json").read_text())
    cached = {s.split("__t0_")[0]
              for s in json.loads((PRE / "eval" / "index.json").read_text())}

    args.sweep = args.sweep or ([25, 50, 100, 200] if args.split == "train"
                                else [100, 200, 500, 1000])

    full = derive(ci[ci.split == args.split], dc)            # target: that whole official split
    # val clips are already in the per-clip cache, so the pool is what the cache holds;
    # test and train were never fully downloaded, so their pool is the whole split and the
    # chosen clips get streamed into a cache afterwards.
    pool = full[full.index.isin(cached)] if args.split == "val" else full
    n_start = len(pool)
    pool = pool[~pool.index.isin(set(ood_src.index))]        # every OOD clip, not just val's
    n_after_ood = len(pool)
    # train is where the calibration set comes from, so there is no prior calibration set to
    # exclude -- this run is what defines it
    if args.split != "train":
        pool = pool[~pool.index.isin(set(split["calib"]))]
    print(f"target(full {args.split}) {len(full):,}  start {n_start:,}  "
          f"-OOD {n_start - n_after_ood}  -calib {n_after_ood - len(pool)}  "
          f"-> pool {len(pool):,}", flush=True)

    codes, targets = encode(full, pool)

    # ---- quality sweep; the chosen N reuses its prefix of the same greedy order ----
    sizes = sorted(set(args.sweep) | {args.n_indist})
    order, _ = greedy(codes, targets, len(pool), max(sizes), args.seed)
    rows = []
    for n in sizes:
        idx = order[:n]
        counts = {a: np.bincount(codes[a][idx], minlength=len(targets[a])).astype(float)
                  for a in ATTR}
        l1, jsd = l1_per_attr(counts, n, targets), jsd_per_attr(counts, n, targets)
        rmean, rstd = random_reference(codes, targets, len(pool), n, args.seed)
        rows.append({"n": n, "weighted_l1": weighted_l1(counts, n, targets),
                     "countries": int(pool.iloc[idx]["country"].nunique()),
                     "random_l1_mean": rmean, "random_l1_std": rstd,
                     **{f"{a}_l1": l1[a] for a in ATTR},
                     **{f"{a}_jsd": jsd[a] for a in ATTR}})
        print(f"  N={n:5d}  weighted L1 {rows[-1]['weighted_l1']:.4f}  "
              f"(random {rmean:.4f}+-{rstd:.4f})  countries {rows[-1]['countries']}", flush=True)
    quality = pd.DataFrame(rows)
    quality.to_csv(out_dir / f"quality_{args.split}.csv", index=False)

    # ---- in-distribution manifest ----
    sel = pool.iloc[order[:args.n_indist]].copy()
    sel["t0_us"] = 5_100_000
    keep = ["chunk", "country", "platform_class", "radar_config", "month",
            "hour_of_day", "time_of_day", "season", "t0_us"]
    # "indist_" keeps the val name the completed run already points at; the train draw is
    # the calibration set, named for what it is rather than where it came from
    stem = {"val": f"indist_{args.n_indist}", "train": f"calib_{args.n_indist}"}.get(
        args.split, f"{args.split}_{args.n_indist}")
    sel.reset_index(names="clip_id")[["clip_id", *keep]].to_parquet(
        out_dir / f"{stem}.parquet", index=False)

    # ---- OOD manifest, straight off the cache so it matches what can be loaded ----
    ood_man = pd.read_parquet(PRE / "ood" / "manifest.parquet")
    built = {s.split("__t0_")[0]
             for s in json.loads((PRE / "ood" / "index.json").read_text())}
    ood = ood_man[ood_man.clip_id.isin(built)].copy()
    ood.to_parquet(out_dir / "ood.parquet", index=False)

    # ---- plots ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(quality.n, quality.weighted_l1, "o-", color=C1, label="greedy match")
    ax.errorbar(quality.n, quality.random_l1_mean, yerr=quality.random_l1_std,
                fmt="s--", color=MUTED, capsize=3, label="random sample")
    ax.axhline(0.0144, color=C2, lw=1, ls=":", label="chunk stage (200 chunks)")
    ax.axvline(args.n_indist, color=C3, lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N clips")
    ax.set_ylabel("weighted L1 vs full val")
    ax.set_title("distribution match of the in-distribution draw")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    tgt = full["country"].map(str).value_counts(normalize=True)
    got = sel["country"].map(str).value_counts(normalize=True)
    top = tgt.head(10).index
    x = np.arange(len(top))
    ax.bar(x - 0.2, [tgt.get(c, 0) * 100 for c in top], 0.4, color=MUTED, label="full val")
    ax.bar(x + 0.2, [got.get(c, 0) * 100 for c in top], 0.4, color=C1,
           label=f"selected (N={args.n_indist})")
    ax.set_xticks(x)
    ax.set_xticklabels([c[:11] for c in top], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("share of clips (%)")
    ax.set_title("country distribution, top 10")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / f"match_{args.split}.png", dpi=150)
    plt.close(fig)

    # ---- records ----
    chosen = quality[quality.n == args.n_indist].iloc[0]
    (out_dir / f"config_{args.split}.json").write_text(json.dumps({
        "purpose": "in-distribution and OOD evaluation sets for open-loop baseline",
        "seed": args.seed, "n_indist": args.n_indist, "split": args.split,
        "method": "clip-level greedy distribution matching against full-val",
        "attributes": ATTR,
        "target": {"split": "val", "n_clips": len(full)},
        "pool": {"start": n_start, "after_drop_ood": n_after_ood,
                 "after_drop_calib": len(pool)},
        "exclusions": {"ood_clips_total": len(ood_src), "calib_clips": len(split["calib"])},
        "t0_us_indist": 5_100_000, "t0_us_ood": "per-clip event_start_timestamp",
    }, indent=2))
    (out_dir / f"metrics_{args.split}.json").write_text(json.dumps({
        "quality": rows,
        "indist": {"n": len(sel), "countries": int(sel["country"].nunique())},
        "ood": {"n": len(ood),
                "by_split": ood["split"].value_counts().to_dict(),
                "by_cluster": ood["cluster"].value_counts().to_dict()},
    }, indent=2))
    (out_dir / f"summary_{args.split}.txt").write_text(
        f"evaluation sets (seed {args.seed})\n\n"
        f"in-distribution ({args.split})  {len(sel)} clips from {len(pool):,} pooled "
        f"(start {n_start:,} - {n_start - n_after_ood} OOD "
        f"- {n_after_ood - len(pool)} calib)\n"
        f"  weighted L1 vs full val {chosen.weighted_l1:.4f}  "
        f"(random draw {chosen.random_l1_mean:.4f})\n"
        f"  countries {int(chosen.countries)}   t0 5,100,000 us\n\n"
        f"OOD              {len(ood)} clips  "
        f"(val {int((ood['split'] == 'val').sum())} primary, "
        f"train {int((ood['split'] == 'train').sum())} secondary)\n"
        f"  {ood['cluster'].nunique()} event clusters   t0 per-clip event_start_timestamp\n")
    print(f"\nin-dist {len(sel)}  |  OOD {len(ood)} "
          f"(val {int((ood['split'] == 'val').sum())})  -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
