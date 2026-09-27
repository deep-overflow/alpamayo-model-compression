"""Which generated-CoC query positions attend to the ego history (and to the other key groups)?

From run_coc_census.py --per-query shards (census_pos.npz: mass_pos (N, L, PAD, H, 6) f16 in the
group order sink / vision / hist / prompt / coc_prev / self; metrics.json carries the decoded CoC
tokens). Units are words (subwords merged) plus the two special tokens, with the categories and
clauses of analyze_coc_token_importance.classify. For each band (0-21 / 22-34) and head set (all
heads; the census groups: per layer the 6 heads with the largest rank(I_traj@CoC) - rank(I_CoC@CoC)
= FM-favoured and the 6 smallest = CE-favoured):

  1. mean attention mass of a query unit on each key group, by category, by clause, by relative
     quarter of the words, and at the two special tokens (per-clip means, clip-bootstrap CIs)
  2. where the CoC queries' ego-history attention sits: share of the summed ego-history mass by
     category against the category's token share
  3. the words whose queries put the most mass on the ego history (n >= 3)

Usage:
  python experiments/head_analysis/analyze_coc_query_census.py --shards coc_census_pos_v1_s0 ... --out coc_census_pos_v1
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_coc_token_importance import CATS, classify, merge_words

REPO = Path(__file__).resolve().parents[2]
LAST = 35
BANDS = {"0-21": slice(0, 22), "22-34": slice(22, LAST)}
GROUPS = ("sink", "vision", "hist", "prompt", "coc_prev", "self")
CLAUSES = ("decision clause", "connective", "cause clause", "special")
COC, K_GROUP = 3, 6
rng = np.random.default_rng(0)


def ci_of(v, n=2000):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    boot = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)])
    return float(v.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--anatomy", default="gradanat_v1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = REPO / "outputs" / args.out
    (out / "plots").mkdir(parents=True, exist_ok=True)
    clips, n_coc, tokens, mass = [], [], [], []
    for s in args.shards:
        m = json.loads((REPO / "outputs" / s / "metrics.json").read_text())
        clips += m["clip_ids"]
        n_coc += m["n_coc"]
        tokens += m["tokens"]
        mass.append(np.load(REPO / "outputs" / s / "census_pos.npz")["mass_pos"])
    mass = np.concatenate(mass).astype(np.float32)  # (N, L, PAD, H, 6)
    N, L, _pad, H, _ = mass.shape

    pq = np.load(REPO / "outputs" / args.anatomy / "anatomy_perclip_q.npz")
    Tc = np.abs(pq["fm_full"][:, COC].astype(np.float64)).mean(0)
    Cc = np.abs(pq["ce"][:, COC].astype(np.float64)).mean(0)
    rk = lambda x: np.argsort(np.argsort(x, axis=1), axis=1) / (x.shape[1] - 1)
    diff = rk(Tc) - rk(Cc)
    sets = {"all heads": np.ones((L, H), bool), "FM-favoured": np.zeros((L, H), bool), "CE-favoured": np.zeros((L, H), bool)}
    for l in range(LAST):
        order = np.argsort(diff[l])
        sets["FM-favoured"][l, order[-K_GROUP:]] = True
        sets["CE-favoured"][l, order[:K_GROUP]] = True

    # per-unit masses: for every clip, unit -> (L, H, 6) mean over the unit's token positions
    units_all, cats_all, clauses_all, words_all, quarter_all, umass = [], [], [], [], [], []
    for i in range(N):
        units, spans = merge_words(tokens[i])
        cats, norm, clause = classify(units)
        n_words = sum(c not in ("cot_end", "traj_start") for c in cats)
        wi, qs = 0, []
        for c in cats:
            if c in ("cot_end", "traj_start"):
                qs.append(-1)
            else:
                qs.append(min(3, int(4 * wi / n_words)))
                wi += 1
        units_all.append(units); cats_all.append(cats); clauses_all.append(clause); words_all.append(norm); quarter_all.append(qs)
        umass.append(np.stack([np.nanmean(mass[i][:, sp], axis=1) for sp in spans]))  # (U, L, H, 6): mean over the unit's positions

    lines = [f"CoC-query attention by query position -- {N} clips, shards {', '.join(args.shards)}", ""]
    metrics = {"n_clips": N, "bands": {}}
    for b, sl in BANDS.items():
        mb = {}
        lines.append(f"===== band {b} =====")
        for sname, smask in sets.items():
            sel = smask[sl]  # (Lb, H)
            # per unit, mean mass over the set's heads in the band -> (U, 6)
            per_unit = [um[:, sl][:, sel].mean(1) for um in umass]
            # 1. by category / clause / quarter / special: per-clip mean over the units in the class, then mean over clips
            def table(labels_all, labels, title, key, b=b, sname=sname, per_unit=per_unit, mb=mb):
                lines.append(f"[{b}] {sname}: mean attention mass of a CoC query on each key group, by {title}")
                lines.append(f"  {'class':16s} {'n units':>7s} | " + " ".join(f"{g:>9s}" for g in GROUPS) + " |  hist [CI]")
                res = {}
                for lab in labels:
                    per_clip = np.full((N, 6), np.nan)
                    n_units = 0
                    for i in range(N):
                        idx = [j for j, x in enumerate(labels_all[i]) if x == lab]
                        if idx:
                            per_clip[i] = per_unit[i][idx].mean(0)
                            n_units += len(idx)
                    if n_units == 0:
                        continue
                    mean = np.nanmean(per_clip, 0)
                    hm, lo, hi = ci_of(per_clip[:, 2])
                    res[str(lab)] = {"n_units": n_units, "mass": mean.tolist(), "hist_ci": [lo, hi]}
                    lines.append(f"  {lab!s:16s} {n_units:7d} | " + " ".join(f"{v:9.3f}" for v in mean) + f" |  {hm:.4f} [{lo:.4f}, {hi:.4f}]")
                mb[f"{sname}:{key}"] = res
            table(cats_all, CATS, "category", "category")
            table(clauses_all, CLAUSES, "clause", "clause")
            table(quarter_all, [0, 1, 2, 3], "relative quarter of the words (-1 = special tokens)", "quarter")
            # 2. where the ego-history attention sits: share of the summed hist mass by category vs token share
            lines.append(f"[{b}] {sname}: share of the CoC queries' summed ego-history mass by category (mean over clips) vs token share")
            lines.append(f"  {'category':12s} {'token share':>11s} {'hist share':>11s} {'ratio':>6s}")
            res = {}
            for c in CATS:
                sh, ts = [], []
                for i in range(N):
                    cats = np.array(cats_all[i])
                    hist = per_unit[i][:, 2]
                    tot = hist.sum()
                    if tot > 0:
                        sh.append(hist[cats == c].sum() / tot)
                        ts.append((cats == c).mean())
                sh, ts = float(np.mean(sh)), float(np.mean(ts))
                res[c] = {"token_share": ts, "hist_share": sh, "ratio": sh / max(ts, 1e-30)}
                lines.append(f"  {c:12s} {ts:11.3f} {sh:11.3f} {sh / max(ts, 1e-30):6.2f}")
            mb[f"{sname}:hist_share"] = res
            # 3. words whose queries attend the ego history most
            acc = {}
            for i in range(N):
                for j, (u, w, c) in enumerate(zip(units_all[i], words_all[i], cats_all[i])):
                    key = u if c in ("cot_end", "traj_start") else w
                    a = acc.setdefault(key, {"n": 0, "hist": 0.0, "vision": 0.0, "coc_prev": 0.0, "cat": c})
                    a["n"] += 1
                    a["hist"] += float(per_unit[i][j, 2]); a["vision"] += float(per_unit[i][j, 1]); a["coc_prev"] += float(per_unit[i][j, 4])
            items = sorted([(k, a) for k, a in acc.items() if a["n"] >= 3], key=lambda kv: -kv[1]["hist"] / kv[1]["n"])
            mb[f"{sname}:top_words_hist"] = [{"word": k, "cat": a["cat"], "n": a["n"], "hist": a["hist"] / a["n"], "vision": a["vision"] / a["n"], "coc_prev": a["coc_prev"] / a["n"]} for k, a in items]
            lines.append(f"[{b}] {sname}: query units with the most ego-history attention (n >= 3; mass per occurrence)")
            lines.append(f"  {'unit':22s} {'cat':10s} {'n':>4s} | {'hist':>7s} {'vision':>7s} {'prev CoC':>8s}")
            for k, a in items[:20]:
                lines.append(f"  {k:22s} {a['cat']:10s} {a['n']:4d} | {a['hist'] / a['n']:7.4f} {a['vision'] / a['n']:7.4f} {a['coc_prev'] / a['n']:8.4f}")
            lines.append("")
        # FM-favoured minus CE-favoured on the ego history, words vs special tokens
        for lab, pick in (("words", lambda c: c not in ("cot_end", "traj_start")), ("special tokens", lambda c: c in ("cot_end", "traj_start"))):
            d = []
            for i in range(N):
                idx = [j for j, c in enumerate(cats_all[i]) if pick(c)]
                if not idx:
                    continue
                fm = umass[i][idx][:, sl][:, sets["FM-favoured"][sl]].mean(1)[:, 2].mean()
                ce = umass[i][idx][:, sl][:, sets["CE-favoured"][sl]].mean(1)[:, 2].mean()
                d.append(fm - ce)
            m, lo, hi = ci_of(d)
            mb[f"hist_FMfav_minus_CEfav:{lab}"] = {"mean": m, "ci": [lo, hi]}
            lines.append(f"[{b}] ego-history mass, FM-favoured - CE-favoured heads, at {lab}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        lines.append("")
        metrics["bands"][b] = mb

    # plot: ego-history mass by relative position (10 bins) + the two special tokens, per band and head set
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, (b, sl) in zip(axes, BANDS.items()):
        for sname, col in (("all heads", "black"), ("FM-favoured", "#618BC8"), ("CE-favoured", "#ECAE3C")):
            sel = sets[sname][sl]
            bins = [[] for _ in range(12)]
            for i in range(N):
                cats = cats_all[i]
                n_words = sum(c not in ("cot_end", "traj_start") for c in cats)
                hist = umass[i][:, sl][:, sel].mean(1)[:, 2]
                wi = 0
                for j, c in enumerate(cats):
                    if c == "cot_end":
                        bins[10].append(hist[j])
                    elif c == "traj_start":
                        bins[11].append(hist[j])
                    else:
                        bins[min(9, int(10 * wi / n_words))].append(hist[j]); wi += 1
            ax.plot(range(12), [np.mean(v) if v else np.nan for v in bins], marker="o", ms=3, color=col, label=sname)
        ax.set_xticks(range(12)); ax.set_xticklabels([f"{k * 10}%" for k in range(10)] + ["cot_end", "traj_start"], rotation=60, fontsize=7)
        ax.set_title(f"layers {b}", fontsize=9); ax.set_xlabel("CoC query position")
    axes[0].set_ylabel("attention mass on ego-history keys"); axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "plots" / "hist_mass_by_query_position.png", dpi=150); plt.close(fig)

    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    (out / "config.json").write_text(json.dumps({"shards": args.shards, "anatomy": args.anatomy, "k_group": K_GROUP}, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
