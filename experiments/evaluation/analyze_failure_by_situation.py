"""In which driving situations does each single-criterion arm fail? Stored open-loop records only.

coc_u40_v2 (CoC-only) / traj_u40_v2 (trajectory-only) / dual_u40_v2 against baseline_ada, paired
per clip over indist_500 + test_500 + ood (2,533 clips, K = 8, same seeds), split by the GT
manoeuvre: eval_lib.bucket with the turn split by the sign of the net heading change.

  1  action channel    paired d minADE per manoeuvre, its size relative to the dense model's
                       minADE there, and the clip-wise interaction x = d_coc - d_traj
                       (a cross-over -- CoC-only failing on turns, trajectory-only on straight
                       driving -- would make x change sign across manoeuvres)
  2  language channel  CoC degeneracy rate, share of clips whose CoC text changed, and on OOD
                       the GT-CoC teacher-forced NLL
  3  confound          is an arm's action damage carried by the clips whose own CoC collapsed?
  4  which calibration clips make each score (calib_100 per-clip gate gradients of
     outputs/gradanat_v1): share of I_traj / I_CoC mass per manoeuvre

Usage:
  python experiments/evaluation/analyze_failure_by_situation.py --out failure_by_situation_v1
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "head_analysis"))
sys.path.insert(0, str(Path(__file__).parent))

import eval_lib as el  # noqa: E402
import sample_cache as sc  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
ARMS = {"base": "baseline_ada", "coc": "coc_u40_v2", "traj": "traj_u40_v2", "dual": "dual_u40_v2"}
SETS = {"indist": ("indist_500", "eval"), "test": ("test_500", "test"), "ood": ("ood", "ood")}
ORDER = ["cruise", "accel", "decel_stop", "turn_left", "turn_right"]
B4 = ["cruise", "accel", "decel_stop", "turn"]
BG, MUTED = "#FAF9F5", "#6B6555"
COL = {"coc": "#e87ba4", "traj": "#2a78d6", "dual": "#008300"}


def bucket5(xy):
    """eval_lib.bucket with the turn split by the sign of the net heading change. xy (64, 2)."""
    vel = np.diff(xy, axis=0) / el.DT
    speed = np.linalg.norm(vel, axis=1)
    moving = speed > 0.5
    signed = 0.0
    if moving.sum() >= 2:
        head = np.arctan2(vel[moving, 1], vel[moving, 0])
        signed = float(np.rad2deg(np.angle(np.exp(1j * (head[-1] - head[0])))))
    b = el.bucket(xy)
    return (("turn_left" if signed > 0 else "turn_right") if b == "turn" else b), signed


def records(exp):
    out = []
    for f in sorted(glob.glob(str(REPO / "outputs" / exp / "*_s*of*.json"))):
        out += json.loads(Path(f).read_text())
    return {r["clip_id"]: r for r in out}


def load():
    rows = []
    for sname, (manifest, cache) in SETS.items():
        man = dict(sc.calib_samples(REPO, manifest))
        recs = {a: records(f"{e}_{sname}") for a, e in ARMS.items()}
        for cid in sorted(set.intersection(*(set(r) for r in recs.values()))):
            z = np.load(sc.path_for(cache, cid, man[cid]), allow_pickle=True)
            b5, signed = bucket5(np.asarray(z["ego_future_xyz"]).reshape(-1, 3)[:, :2])
            row = {"set": sname, "clip_id": cid, "b5": b5, "b": b5.replace("_left", "").replace("_right", ""),
                   "stored_bucket": recs["base"][cid]["bucket"], "signed": signed}
            for a, r in recs.items():
                row[f"ade_{a}"] = r[cid]["minADE_rollout"]
                row[f"deg_{a}"] = float(r[cid]["coc_degenerate"])
                row[f"txt_{a}"] = r[cid]["gen_coc"]
                row[f"nll_{a}"] = r[cid].get("nll_gtcoc", np.nan)
            rows.append(row)
    df = pd.DataFrame(rows)
    for a in ("coc", "traj", "dual"):
        df[f"d_{a}"] = df[f"ade_{a}"] - df["ade_base"]
        df[f"dn_{a}"] = df[f"nll_{a}"] - df["nll_base"]
        df[f"chg_{a}"] = (df[f"txt_{a}"] != df["txt_base"]).astype(float)
    df["x"] = df["d_coc"] - df["d_traj"]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="failure_by_situation_v1")
    ap.add_argument("--anatomy", default="gradanat_v1")
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    def ci(x):
        x = np.asarray(x, float)
        if len(x) < 5:
            return float("nan"), float("nan")
        b = x[rng.integers(0, len(x), (args.n_boot, len(x)))].mean(1)
        return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

    def contrast(a_, b_):
        a_, b_ = np.asarray(a_, float), np.asarray(b_, float)
        d = (a_[rng.integers(0, len(a_), (args.n_boot, len(a_)))].mean(1)
             - b_[rng.integers(0, len(b_), (args.n_boot, len(b_)))].mean(1))
        return float(a_.mean() - b_.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

    df = load()
    agree = float((df["b"] == df["stored_bucket"]).mean())
    out = {"n_clips": {k: int(v) for k, v in df.groupby("set").size().items()}, "bucket_agreement": agree}
    lines = [f"failure by driving situation -- {len(df)} clips {out['n_clips']}; recomputed bucket == stored bucket: {agree:.4f}",
             "counts: " + json.dumps({s: g.b5.value_counts().reindex(ORDER, fill_value=0).to_dict()
                                      for s, g in df.groupby("set")}), ""]

    # ------------------------------------------------------------------ 1 action, 2 language
    for label, sub in (("ID", df[df.set != "ood"]), ("OOD", df[df.set == "ood"]), ("ALL", df)):
        res = {}
        lines += [f"== {label} (n={len(sub)})",
                  "action: paired d minADE vs dense, mean [95% CI] (relative to the dense mean of that manoeuvre)",
                  f"  {'manoeuvre':11s} {'n':>5s} {'dense':>6s} | {'CoC-only':^32s} | {'trajectory-only':^32s} | {'dual':^32s}"]
        for b in ORDER + ["turn", "all"]:
            s = sub if b == "all" else sub[sub.b == "turn"] if b == "turn" else sub[sub.b5 == b]
            cells, res[b] = [], {"n": len(s), "dense": float(s.ade_base.mean())}
            for a in ("coc", "traj", "dual"):
                lo, hi = ci(s[f"d_{a}"])
                res[b][a] = {"d": float(s[f"d_{a}"].mean()), "ci": [lo, hi],
                             "rel": float(s[f"d_{a}"].mean() / s.ade_base.mean()),
                             "degenerate": float(s[f"deg_{a}"].mean()), "text_changed": float(s[f"chg_{a}"].mean()),
                             "d_nll_gtcoc": float(s[f"dn_{a}"].mean()) if s[f"dn_{a}"].notna().all() else None}
                cells.append(f"{res[b][a]['d']:+.3f} [{lo:+.3f},{hi:+.3f}] {res[b][a]['rel']:+6.1%}")
            lines.append(f"  {b:11s} {len(s):5d} {s.ade_base.mean():6.3f} | " + " | ".join(cells))
        lines.append("interaction x = d_coc - d_traj per clip (a cross-over needs a sign change across manoeuvres)")
        for b in B4:
            s = sub[sub.b == b]
            lo, hi = ci(s.x)
            res[b]["x"] = {"mean": float(s.x.mean()), "ci": [lo, hi], "median": float(s.x.median())}
            lines.append(f"  {b:11s} mean {s.x.mean():+.3f} [{lo:+.3f},{hi:+.3f}]  median {s.x.median():+.4f}")
        res["contrasts"] = {}
        for name, m1, m2 in (("turn - cruise", sub.b == "turn", sub.b == "cruise"),
                             ("decel_stop - cruise", sub.b == "decel_stop", sub.b == "cruise"),
                             ("turn_left - turn_right", sub.b5 == "turn_left", sub.b5 == "turn_right")):
            for col in ("x", "d_coc", "d_traj", "d_dual"):
                est, lo, hi = contrast(sub[m1][col], sub[m2][col])
                res["contrasts"][f"{name}|{col}"] = {"est": est, "ci": [lo, hi]}
                lines.append(f"  contrast {name:22s} on {col:6s}: {est:+.3f} [{lo:+.3f},{hi:+.3f}]")
        lines.append("language: CoC degeneracy rate dense / CoC-only / trajectory-only / dual | text changed vs dense"
                     + (" | d NLL(GT CoC)" if label == "OOD" else ""))
        for b in B4 + ["all"]:
            s = sub if b == "all" else sub[sub.b == b]
            row = (f"  {b:11s} " + " / ".join(f"{s[f'deg_{a}'].mean():6.2%}" for a in ARMS) + " | "
                   + " / ".join(f"{s[f'chg_{a}'].mean():5.1%}" for a in ("coc", "traj", "dual")))
            if label == "OOD":
                row += " | " + " / ".join(f"{s[f'dn_{a}'].mean():+.3f}" for a in ("coc", "traj", "dual"))
            lines.append(row)
        lines.append("")
        out[label] = res

    # ------------------------------------------------------------------ 3 confound
    lines.append("== is the action damage carried by the clips whose own CoC collapsed? (all clips)")
    out["confound"] = {}
    for a in ("coc", "traj", "dual"):
        h, g = df[df[f"deg_{a}"] == 0], df[df[f"deg_{a}"] == 1]
        share = float(g[f"d_{a}"].sum() / df[f"d_{a}"].sum())
        rel = {b: float(h[h.b == b][f"d_{a}"].mean() / h[h.b == b].ade_base.mean()) for b in B4}
        out["confound"][a] = {"n_degenerate": len(g), "d_healthy": float(h[f"d_{a}"].mean()),
                              "d_degenerate": float(g[f"d_{a}"].mean()) if len(g) else None,
                              "share_of_damage_from_degenerate": share, "rel_damage_healthy_only": rel}
        lines.append(f"  {a:5s} degenerate {len(g):4d} clips ({len(g) / len(df):.1%}): d minADE {g[f'd_{a}'].mean():+.3f} vs healthy "
                     f"{h[f'd_{a}'].mean():+.3f}; they carry {share:.0%} of the arm's damage. Healthy-only relative damage: "
                     + "  ".join(f"{b} {v:+.0%}" for b, v in rel.items()))
    lines.append("")

    # ------------------------------------------------------------------ 4 which clips make each score
    apath = REPO / "outputs" / args.anatomy
    if (apath / "anatomy_perclip_q.npz").exists():
        cfg = json.loads((apath / "config.json").read_text())
        met = json.loads((apath / "metrics.json").read_text())["per_clip"]
        man = dict(sc.calib_samples(REPO, cfg["calib_manifest"]))
        with np.load(apath / "anatomy_perclip_q.npz") as pq:
            fm, ce = np.abs(pq["fm_full"].sum(1)), np.abs(pq["ce"].sum(1))  # (N, 36, 32)
        rows = []
        for i, cid in enumerate(cfg["clip_ids"]):
            z = np.load(sc.path_for(cfg["cache"], cid, man[cid]), allow_pickle=True)
            rows.append({"b": el.bucket(np.asarray(z["ego_future_xyz"]).reshape(-1, 3)[:, :2]),
                         "fm": fm[i].sum(), "ce": ce[i].sum(), "fm_loss": met[i]["fm_loss"]})
        c = pd.DataFrame(rows)
        lines.append(f"== which calibration clips make each score ({cfg['calib_manifest']}, Q heads, per-clip |gate gradient|)")
        lines.append(f"  {'manoeuvre':11s} {'clips':>6s} | {'I_traj mass':>11s} {'I_CoC mass':>11s} | mean FM loss")
        out["calib_mass"] = {}
        for b in B4:
            s = c[c.b == b]
            out["calib_mass"][b] = {"clip_share": len(s) / len(c), "traj_mass": float(s.fm.sum() / c.fm.sum()),
                                    "coc_mass": float(s.ce.sum() / c.ce.sum()), "fm_loss": float(s.fm_loss.mean())}
            r = out["calib_mass"][b]
            lines.append(f"  {b:11s} {r['clip_share']:6.1%} | {r['traj_mass']:11.1%} {r['coc_mass']:11.1%} | {r['fm_loss']:.4f}")
        top_t = float(c.fm.sort_values(ascending=False).head(10).sum() / c.fm.sum())
        top_c = float(c.ce.sort_values(ascending=False).head(10).sum() / c.ce.sum())
        out["calib_mass"]["top10_share"] = {"traj": top_t, "coc": top_c}
        lines.append(f"  the 10 largest clips carry {top_t:.0%} of I_traj mass and {top_c:.0%} of I_CoC mass")

    out_dir = REPO / "outputs" / args.out
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax_, label in zip(axs, ("ID", "OOD")):
        for i, a in enumerate(("coc", "traj", "dual")):
            rel = [out[label][b][a]["rel"] for b in B4]
            ax_.bar(np.arange(len(B4)) + 0.27 * (i - 1), rel, width=0.25, color=COL[a],
                    label={"coc": "CoC-only", "traj": "trajectory-only", "dual": "dual"}[a])
        ax_.set_xticks(np.arange(len(B4)), B4)
        ax_.axhline(0, color=MUTED, lw=0.6)
        ax_.set_title(f"{label}: minADE increase / dense minADE of that manoeuvre")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_dir / "plots" / "relative_damage_by_manoeuvre.png", dpi=150)
    plt.close(fig)

    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps(out, indent=2))
    (out_dir / "config.json").write_text(json.dumps({
        "arms": ARMS, "sets": {k: list(v) for k, v in SETS.items()}, "anatomy": args.anatomy,
        "n_boot": args.n_boot, "metric": "minADE_rollout (K=8), paired per clip against baseline_ada"}, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
