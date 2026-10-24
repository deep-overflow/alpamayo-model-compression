"""Sample-level failure analysis of the single-criterion arms, open loop (stored records only).
Sets: test500 and ood (1,533). Arms: baseline_ada / coc_u40_v2 / traj_u40_v2 / dual_u40_v2, paired
per clip (same seeds). d_a = minADE_a - minADE_base. Tail_a = worst decile of d_a."""
import glob, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, mannwhitneyu, fisher_exact

REPO = Path("/home/cvlab21/project/chan/alpamayo-model-compression")
ARMS = {"base": "baseline_ada", "coc": "coc_u40_v2", "traj": "traj_u40_v2", "dual": "dual_u40_v2"}
SETS = {"test": "test_500", "ood": "ood"}
KW = {"stop": r"\bstop|halt|come to a (complete )?stop", "slow": r"slow|decelerat|brak|yield|reduce speed", "wait": r"\bwait|stationary|remain stopped|stay stopped",
      "pedestrian": r"pedestrian|people|person|crosswalk|jaywalk", "light": r"traffic light|red light|green light|signal", "construction": r"construction|cone|work zone|barrier|roadwork",
      "lane_change": r"lane change|change (to the|lanes?)|merge", "turn": r"\bturn|turning", "intersection": r"intersection|junction|roundabout", "accel": r"accelerat|speed up|proceed|continue"}
rng = np.random.default_rng(0)


def records(exp):
    out = []
    for f in sorted(glob.glob(str(REPO / "outputs" / exp / "*_s*of*.json"))):
        out += json.loads(Path(f).read_text())
    return {r["clip_id"]: r for r in out}


def head2(t):
    w = re.sub(r"[^a-z ]", " ", (t or "").lower()).split()
    return " ".join(w[:2])


def load(sname):
    recs = {a: records(f"{e}_{sname}") for a, e in ARMS.items()}
    cids = sorted(set.intersection(*(set(r) for r in recs.values())))
    rows = []
    for cid in cids:
        row = {"clip_id": cid, "bucket": recs["base"][cid]["bucket"]}
        for a, r in recs.items():
            x = r[cid]
            row[f"ade_{a}"] = x["minADE_rollout"]; row[f"deg_{a}"] = bool(x["coc_degenerate"]); row[f"txt_{a}"] = x["gen_coc"]
            row[f"len_{a}"] = x["gen_len"]; row[f"nlls_{a}"] = x["nll_self"]
            if sname == "ood":
                row[f"adetf_{a}"] = x["minADE_tf"]; row[f"nllg_{a}"] = x["nll_gtcoc"]
        if sname == "ood":
            row["cluster"] = recs["base"][cid]["cluster"]; row["gt_coc"] = recs["base"][cid]["gt_coc"]
        rows.append(row)
    df = pd.DataFrame(rows)
    man = pd.read_parquet(REPO / "outputs" / "eval_sets" / f"{SETS[sname]}.parquet")
    df = df.merge(man[[c for c in ("clip_id", "country", "time_of_day", "platform_class") if c in man.columns]], on="clip_id", how="left")
    for a in ("coc", "traj", "dual"):
        df[f"d_{a}"] = df[f"ade_{a}"] - df["ade_base"]
        df[f"chg_{a}"] = df[f"txt_{a}"] != df["txt_base"]
        df[f"hchg_{a}"] = df[f"txt_{a}"].map(head2) != df["txt_base"].map(head2)
        if sname == "ood":
            df[f"dtf_{a}"] = df[f"adetf_{a}"] - df["adetf_base"]; df[f"dnllg_{a}"] = df[f"nllg_{a}"] - df["nllg_base"]
    for k, pat in KW.items():
        df[f"kw_{k}"] = df["txt_base"].str.lower().str.contains(pat, regex=True)
    return df


def boot_mean(x, n=2000):
    x = np.asarray(x, float); m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def enrich(df, mask_tail, col, lines, top=6):
    """share of a categorical covariate in the tail vs all, with Fisher p for each level."""
    out = []
    for lv, n_all in df[col].value_counts().items():
        a = int((df.loc[mask_tail, col] == lv).sum()); n_t = int(mask_tail.sum())
        p_t, p_a = a / n_t, n_all / len(df)
        tab = [[a, n_t - a], [n_all - a, len(df) - n_t - (n_all - a)]]
        out.append((lv, p_t, p_a, p_t / p_a if p_a else np.nan, fisher_exact(tab)[1], a))
    out.sort(key=lambda t: -abs(np.log(t[3])) if t[3] > 0 else 0)
    for lv, p_t, p_a, ratio, p, a in out[:top]:
        lines.append(f"      {str(lv)[:34]:34s} tail {p_t:5.1%} (n={a:3d}) vs all {p_a:5.1%}  x{ratio:4.2f}  Fisher p={p:.3g}")


out, L = {}, []
for sname in SETS:
    df = load(sname); n = len(df); k = max(1, n // 10)
    L += [f"\n=================== {sname} (n={n}) ===================",
          "paired d minADE vs baseline: " + " | ".join(f"{a} mean {df[f'd_{a}'].mean():+.3f} median {df[f'd_{a}'].median():+.3f}" for a in ("coc", "traj", "dual"))]
    tails = {a: df[f"d_{a}"].rank(ascending=False) <= k for a in ("coc", "traj", "dual")}
    base_tail = df["ade_base"].rank(ascending=False) <= k
    jac = lambda x, y: (x & y).sum() / (x | y).sum()
    rho = spearmanr(df["d_traj"], df["d_coc"])[0]
    L += [f"\n[1] overlap of failures: Spearman(d_traj, d_coc) = {rho:+.3f}; worst-decile Jaccard traj/coc = {jac(tails['traj'], tails['coc']):.3f} (chance 0.053); "
          f"traj-tail vs baseline-hard = {jac(tails['traj'], base_tail):.3f}, coc-tail vs baseline-hard = {jac(tails['coc'], base_tail):.3f}"]
    # rank-decile matrix
    dec_t = pd.qcut(df["d_traj"].rank(method="first"), 5, labels=False); dec_c = pd.qcut(df["d_coc"].rank(method="first"), 5, labels=False)
    L.append("    quintile cross-tab (rows: d_traj quintile 0=best..4=worst, cols: d_coc quintile) -- share of clips in the worst quintile of both: "
             f"{((dec_t == 4) & (dec_c == 4)).mean():.1%} (independent: 4.0%)")
    out.setdefault(sname, {})["overlap"] = {"spearman": rho, "jaccard_traj_coc": jac(tails["traj"], tails["coc"]), "jaccard_traj_base": jac(tails["traj"], base_tail), "jaccard_coc_base": jac(tails["coc"], base_tail)}
    # exclusive tails
    excl = {"traj": tails["traj"] & (df["d_coc"] <= df["d_coc"].median()), "coc": tails["coc"] & (df["d_traj"] <= df["d_traj"].median())}
    L.append(f"    exclusive tails: traj-only-fails {int(excl['traj'].sum())} clips, coc-only-fails {int(excl['coc'].sum())} clips (of {k} per tail)")
    # [2] what the dual arm does on each tail
    L.append("\n[2] does dual cover each tail? (median d, share of clips within +0.05 m of baseline)")
    for a in ("traj", "coc"):
        t = tails[a]; oth = "coc" if a == "traj" else "traj"
        L.append(f"    on the {a} tail (n={int(t.sum())}): d_{a} median {df.loc[t, f'd_{a}'].median():+.2f} | d_{oth} median {df.loc[t, f'd_{oth}'].median():+.2f} "
                 f"(within 0.05: {(df.loc[t, f'd_{oth}'] < 0.05).mean():.0%}) | d_dual median {df.loc[t, 'd_dual'].median():+.2f} (within 0.05: {(df.loc[t, 'd_dual'] < 0.05).mean():.0%})")
        out[sname][f"dual_on_{a}_tail"] = {"d_self_median": float(df.loc[t, f"d_{a}"].median()), "d_other_median": float(df.loc[t, f"d_{oth}"].median()), "d_dual_median": float(df.loc[t, "d_dual"].median()),
                                           "dual_within_0.05": float((df.loc[t, "d_dual"] < 0.05).mean()), "other_within_0.05": float((df.loc[t, f"d_{oth}"] < 0.05).mean())}
    # [3] characterize each tail
    for a in ("traj", "coc"):
        t = tails[a]; L.append(f"\n[3] the {a} tail (worst {int(t.sum())} clips of d_{a}):")
        L.append(f"    CoC degenerate in tail {df.loc[t, f'deg_{a}'].mean():.0%} vs all {df[f'deg_{a}'].mean():.1%}; CoC text changed {df.loc[t, f'chg_{a}'].mean():.0%} vs {df[f'chg_{a}'].mean():.0%}; "
                 f"head clause changed {df.loc[t, f'hchg_{a}'].mean():.0%} vs {df[f'hchg_{a}'].mean():.0%}")
        L.append(f"    baseline minADE median tail {df.loc[t, 'ade_base'].median():.2f} vs rest {df.loc[~t, 'ade_base'].median():.2f} (MWU p={mannwhitneyu(df.loc[t, 'ade_base'], df.loc[~t, 'ade_base'])[1]:.2g}); "
                 f"baseline CoC length median tail {df.loc[t, 'len_base'].median():.0f} vs rest {df.loc[~t, 'len_base'].median():.0f}")
        L.append("    manoeuvre:"); enrich(df, t, "bucket", L)
        if sname == "ood":
            L.append("    OOD cluster:"); enrich(df, t, "cluster", L, top=5)
            L.append(f"    teacher-forced (GT CoC) d minADE median in tail {df.loc[t, f'dtf_{a}'].median():+.2f} vs own-rollout {df.loc[t, f'd_{a}'].median():+.2f}; d NLL(GT CoC) tail {df.loc[t, f'dnllg_{a}'].median():+.3f} vs rest {df.loc[~t, f'dnllg_{a}'].median():+.3f}")
        else:
            L.append("    country / time of day:"); enrich(df, t, "country", L, top=3); enrich(df, t, "time_of_day", L, top=2)
        L.append("    baseline-CoC keywords (x = tail share / all share):")
        kws = []
        for kname in KW:
            c = f"kw_{kname}"; p_t, p_a = df.loc[t, c].mean(), df[c].mean()
            if p_a > 0:
                tab = [[int(df.loc[t, c].sum()), int((~df.loc[t, c]).sum())], [int(df.loc[~t, c].sum()), int((~df.loc[~t, c]).sum())]]
                kws.append((kname, p_t, p_a, p_t / p_a, fisher_exact(tab)[1]))
        kws.sort(key=lambda x: -abs(np.log(max(x[3], 1e-3))))
        L.append("      " + "; ".join(f"{kn} {p_t:.0%}/{p_a:.0%} x{r:.2f} (p={p:.2g})" for kn, p_t, p_a, r, p in kws[:6]))
    # [4] reasoning drift vs action damage, healthy clips only
    L.append("\n[4] is action damage tied to reasoning drift? healthy (non-degenerate) clips, d minADE by whether the CoC head clause changed")
    for a in ("traj", "coc", "dual"):
        h = ~df[f"deg_{a}"]
        for b in (None, "decel_stop", "turn", "cruise"):
            m = h if b is None else (h & (df["bucket"] == b))
            same, chg = df.loc[m & ~df[f"hchg_{a}"], f"d_{a}"], df.loc[m & df[f"hchg_{a}"], f"d_{a}"]
            if len(same) > 5 and len(chg) > 5:
                L.append(f"    {a:4s} {str(b or 'all'):10s} head same n={len(same):4d} mean {same.mean():+.3f} med {same.median():+.3f} | head changed n={len(chg):4d} mean {chg.mean():+.3f} med {chg.median():+.3f} | MWU p={mannwhitneyu(same, chg)[1]:.2g} | changed rate {len(chg) / (len(same) + len(chg)):.0%}")
    # [5] damage partition: degenerate / head changed / head same
    L.append("\n[5] share of each arm's total damage (sum of positive d) by language state of that clip")
    for a in ("traj", "coc", "dual"):
        dpos = df[f"d_{a}"].clip(lower=0); tot = dpos.sum()
        st = np.where(df[f"deg_{a}"], "degenerate", np.where(df[f"hchg_{a}"], "head changed", np.where(df[f"chg_{a}"], "wording changed", "same text")))
        parts = {s: float(dpos[st == s].sum() / tot) for s in ("degenerate", "head changed", "wording changed", "same text")}
        cnt = {s: int((st == s).sum()) for s in parts}
        L.append(f"    {a:4s} " + " | ".join(f"{s} {parts[s]:.0%} of damage on {cnt[s] / n:.0%} of clips" for s in parts))
        out[sname][f"damage_partition_{a}"] = parts
    df.to_parquet(f"/home/cvlab21/project/chan/.claude/jobs/48e21e0f/tmp/afc/openloop_{sname}.parquet", index=False)
Path("/home/cvlab21/project/chan/.claude/jobs/48e21e0f/tmp/afc/openloop_summary.txt").write_text("\n".join(L) + "\n")
json.dump(out, open("/home/cvlab21/project/chan/.claude/jobs/48e21e0f/tmp/afc/openloop_metrics.json", "w"), indent=2)
print("\n".join(L))
