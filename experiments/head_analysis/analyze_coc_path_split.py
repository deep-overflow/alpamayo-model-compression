"""Gates of plans/2026-09-25_coc-position-disagreement.md, test 1: does the cross-position part of
I_CoC at generated-CoC positions agree with I_traj there, while the own-token part does not?

Reads the shards of run_coc_path_split.py and the calib_100 anatomy (I_traj at CoC positions =
the CoC row of anatomy_perclip_q.npz fm_full; I_CoC = its ce row, which the G0 check reproduces).

  G0    the CoC row of the full CE backward reproduces the anatomy's ce row (per-layer Spearman
        of the 100-clip means >= 0.99)
  T1-a  ceiling-corrected within-layer agreement of I_traj@CoC with the CROSS-position part of
        I_CoC@CoC >= 0.70 in both bands (like vision / prompt / ego-history positions)
  T1-b  agreement of I_traj@CoC with the OWN-TOKEN part <= 0.45 in both bands
  also  the own-token share of I_CoC@CoC (abs and additive), and the two parts' mutual agreement

Usage:
  python experiments/head_analysis/analyze_coc_path_split.py --shards coc_pathsplit_v1_s0 ... --out coc_pathsplit_v1
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
VIS, HIST, PT, COC, SINK = range(5)
LAST = 35
BANDS = {"0-21": slice(0, 22), "22-34": slice(22, LAST)}


def rho_layers(a, b):
    return np.array([spearmanr(a[l], b[l])[0] if np.ptp(a[l]) > 0 and np.ptp(b[l]) > 0 else np.nan for l in range(a.shape[0])])


def split_half(t, c, n_split, rng):
    n = len(t)
    st, sc_, cr = [], [], []
    for _ in range(n_split):
        p = rng.permutation(n)
        a, b = p[:n // 2], p[n // 2:]
        ta, tb, ca, cb = t[a].mean(0), t[b].mean(0), c[a].mean(0), c[b].mean(0)
        st.append(rho_layers(ta, tb))
        sc_.append(rho_layers(ca, cb))
        cr.append(0.5 * (rho_layers(ta, cb) + rho_layers(tb, ca)))
    return np.nanmean(st, 0), np.nanmean(sc_, 0), np.nanmean(cr, 0)


def corrected(st, sc_, cr):
    ok = (st > 0.2) & (sc_ > 0.2)
    return np.where(ok, cr / np.sqrt(np.where(ok, st * sc_, 1.0)), np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--anatomy", default="gradanat_v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-split", type=int, default=50)
    args = ap.parse_args()
    out = REPO / "outputs" / args.out
    (out / "plots").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    clips, q_full, q_dir, m_full, m_dir = [], [], [], [], []
    for s in args.shards:
        m = json.loads((REPO / "outputs" / s / "metrics.json").read_text())
        z = np.load(REPO / "outputs" / s / "pathsplit_perclip.npz")
        clips += m["clip_ids"]
        q_full.append(z["q_full"])
        q_dir.append(z["q_direct"])
        if "mlp_coc_full" in z.files:
            m_full.append(z["mlp_coc_full"])
            m_dir.append(z["mlp_coc_direct"])
    q_full, q_dir = np.concatenate(q_full).astype(np.float64), np.concatenate(q_dir).astype(np.float64)  # (N, 5, L, H)
    pq = np.load(REPO / "outputs" / args.anatomy / "anatomy_perclip_q.npz")
    met = json.loads((REPO / "outputs" / args.anatomy / "metrics.json").read_text())
    anat_ids = [r["clip_id"] for r in met["per_clip"]]
    order = [anat_ids.index(c) for c in clips]
    fm_coc = pq["fm_full"][order][:, COC].astype(np.float64)  # (N, L, H)
    ce_coc_anat = pq["ce"][order][:, COC].astype(np.float64)
    lines, gates = [f"CoC path split -- {len(clips)} clips, shards {', '.join(args.shards)}", ""], {}

    axes_data = {"q": (fm_coc, q_full[:, COC], q_dir[:, COC], args.n_split)}
    if m_full:
        mf, md = np.concatenate(m_full).astype(np.float64), np.concatenate(m_dir).astype(np.float64)  # (N, L, I)
        fm_mlp = np.load(REPO / "outputs" / args.anatomy / "anatomy_perclip_mlp.npz")["fm_type"][order][:, COC].astype(np.float64)
        axes_data["mlp"] = (fm_mlp, mf, md, 10)
    # G0
    g0 = rho_layers(np.abs(q_full[:, COC]).mean(0), np.abs(ce_coc_anat).mean(0))[:LAST]
    gates["G0"] = {"min_rho": float(np.nanmin(g0)), "pass": bool(np.nanmin(g0) >= 0.99)}
    lines.append(f"G0 full CE backward vs anatomy ce (CoC row, Q): per-layer Spearman min {np.nanmin(g0):.3f} mean {np.nanmean(g0):.3f} -> {'PASS' if gates['G0']['pass'] else 'FAIL'}")
    for axis, (T, Cf, Cd, n_split) in axes_data.items():
        Cc = Cf - Cd  # cross-position part, signed per clip
        aT, aF, aD, aC = np.abs(T), np.abs(Cf), np.abs(Cd), np.abs(Cc)
        # shares
        abs_share = aD.sum((1, 2)) / aF.sum((1, 2))
        add_share = (Cd * np.sign(Cf)).sum((1, 2)) / aF.sum((1, 2))
        lines.append(f"\n[{axis}] own-token share of I_CoC@CoC: |.| share mean {abs_share.mean():.3f} (clip sd {abs_share.std():.3f}), additive share {add_share.mean():+.3f}; "
                     f"by band (100-clip means, |.|): " + ", ".join(f"{b} {aD.mean(0)[sl].sum() / aF.mean(0)[sl].sum():.3f}" for b, sl in BANDS.items()))
        # raw agreement on the 100-clip means
        raw = {name: rho_layers(aT.mean(0), X.mean(0))[:LAST] for name, X in (("full", aF), ("own-token", aD), ("cross", aC))}
        raw["own vs cross"] = rho_layers(aD.mean(0), aC.mean(0))[:LAST]
        # corrected
        corr = {}
        for name, X in (("full", aF), ("own-token", aD), ("cross", aC)):
            st, sc_, cr = split_half(aT[:, :LAST], X[:, :LAST], n_split, rng)
            corr[name] = (corrected(st, sc_, cr), st, sc_)
        st, sc_, cr = split_half(aD[:, :LAST], aC[:, :LAST], n_split, rng)
        corr["own vs cross"] = (corrected(st, sc_, cr), st, sc_)
        lines.append(f"[{axis}] within-layer agreement of I_traj@CoC with each part of I_CoC@CoC (raw on 100-clip means / split-half corrected), band means:")
        for name in ("full", "own-token", "cross", "own vs cross"):
            lines.append(f"    {name:12s} " + " | ".join(f"{b}: raw {np.nanmean(raw[name][sl]):+.2f} corr {np.nanmean(corr[name][0][sl]):+.2f} (self {np.nanmean(corr[name][1][sl]):.2f}/{np.nanmean(corr[name][2][sl]):.2f})" for b, sl in BANDS.items()))
        ga = {b: float(np.nanmean(corr["cross"][0][sl])) for b, sl in BANDS.items()}
        gb = {b: float(np.nanmean(corr["own-token"][0][sl])) for b, sl in BANDS.items()}
        gates[axis] = {"T1a_cross_corrected": ga, "T1a_pass": all(v >= 0.70 for v in ga.values()),
                       "T1b_own_corrected": gb, "T1b_pass": all(v <= 0.45 for v in gb.values()),
                       "raw": {k: {b: float(np.nanmean(v[sl])) for b, sl in BANDS.items()} for k, v in raw.items()},
                       "own_token_abs_share": float(abs_share.mean()), "own_token_additive_share": float(add_share.mean())}
        lines.append(f"    gates: T1-a cross >= 0.70 both bands -> {'PASS' if gates[axis]['T1a_pass'] else 'FAIL'} ({ga['0-21']:.2f} / {ga['22-34']:.2f}); "
                     f"T1-b own-token <= 0.45 -> {'PASS' if gates[axis]['T1b_pass'] else 'FAIL'} ({gb['0-21']:.2f} / {gb['22-34']:.2f})")
        if axis == "q":
            fig, ax = plt.subplots(figsize=(4.5, 3.2))
            L = np.arange(LAST)
            for name, col in (("full", "black"), ("cross", "#2a78d6"), ("own-token", "#e87ba4")):
                ax.plot(L, corr[name][0], color=col, lw=1.4, label=f"I_traj@CoC vs I_CoC@CoC ({name})")
            ax.axvspan(21.5, 35.5, color="#777777", alpha=0.13, lw=0)
            ax.axhline(0, color="black", lw=0.6)
            ax.set_xlabel("VLM layer"); ax.set_ylabel("corrected rank agreement"); ax.set_ylim(-0.3, 1.1); ax.legend(fontsize=6)
            fig.tight_layout(); fig.savefig(out / "plots" / "coc_path_split_q.png", dpi=150); plt.close(fig)
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    (out / "metrics.json").write_text(json.dumps({"gates": gates, "n_clips": len(clips)}, indent=1, default=float))
    (out / "config.json").write_text(json.dumps({"shards": args.shards, "anatomy": args.anatomy, "plan": "plans/2026-09-25_coc-position-disagreement.md"}, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
