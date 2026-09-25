"""Gates A0-A3 of plans/2026-09-25_coc-position-content.md: token weighting and same-token content
inside the generated CoC, from run_coc_position_anatomy.py shards.

  A0  the sum over CoC positions reproduces the anatomy's CoC row (per-layer Spearman >= 0.99)
  A1  share of each loss's residual-gradient norm (and of the Q heads' |contribution|) on the
      head clause / rest / end token, per band
  A2  agreement of I_traj and I_CoC restricted to the same sub-span (raw on the 100-clip means,
      split-half corrected alongside): >= 0.70 corrected on a sub-span -> token mixture inside the
      CoC; < 0.50 on every sub-span -> different content at the same tokens
  A3  single-position agreement: Spearman across heads at one CoC position of one clip, averaged

Usage:
  python experiments/head_analysis/analyze_coc_position_anatomy.py --shards coc_posanat_v1_s0 ... --out coc_posanat_v1
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
COC, LAST = 3, 35
BANDS = {"0-21": slice(0, 22), "22-34": slice(22, LAST)}
SPANS = ("head clause", "rest", "<|traj_future_start|>")


def rho_layers(a, b):
    return np.array([spearmanr(a[l], b[l])[0] if np.ptp(a[l]) > 0 and np.ptp(b[l]) > 0 else np.nan for l in range(a.shape[0])])


def corrected(t, c, n_split, rng):
    n = len(t)
    st, sc_, cr = [], [], []
    for _ in range(n_split):
        p = rng.permutation(n)
        a, b = p[:n // 2], p[n // 2:]
        st.append(rho_layers(t[a].mean(0), t[b].mean(0)))
        sc_.append(rho_layers(c[a].mean(0), c[b].mean(0)))
        cr.append(0.5 * (rho_layers(t[a].mean(0), c[b].mean(0)) + rho_layers(t[b].mean(0), c[a].mean(0))))
    st, sc_, cr = (np.nanmean(v, 0) for v in (st, sc_, cr))
    ok = (st > 0.2) & (sc_ > 0.2)
    return np.where(ok, cr / np.sqrt(np.where(ok, st * sc_, 1.0)), np.nan), st, sc_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--anatomy", default="gradanat_v1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = REPO / "outputs" / args.out
    (out / "plots").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows, per = [], {}
    for s in args.shards:
        m = json.loads((REPO / "outputs" / s / "metrics.json").read_text())
        z = np.load(REPO / "outputs" / s / "posanat_perclip.npz")
        rows += m["per_clip"]
        for k in z.files:
            per.setdefault(k, []).append(z[k])
    per = {k: np.concatenate(v) for k, v in per.items()}
    clips = [r["clip_id"] for r in rows]
    N = len(clips)
    n_coc = np.array([r["n_coc"] for r in rows])
    sub = np.full((N, per["q_ce_pos"].shape[1]), -1)
    for i, r in enumerate(rows):
        sub[i, : r["n_coc"]] = r["sub"]
    qce, qfm = np.nan_to_num(per["q_ce_pos"].astype(np.float64)), np.nan_to_num(per["q_fm_pos"].astype(np.float64))  # (N, PAD, L, H)
    lines, gates = [f"CoC per-position anatomy -- {N} clips, shards {', '.join(args.shards)}", ""], {}

    # A0
    pq = np.load(REPO / "outputs" / args.anatomy / "anatomy_perclip_q.npz")
    ids = [r["clip_id"] for r in json.loads((REPO / "outputs" / args.anatomy / "metrics.json").read_text())["per_clip"]]
    order = [ids.index(c) for c in clips]
    for name, X, key in (("CE", qce, "ce"), ("FM", qfm, "fm_full")):
        ref = np.abs(pq[key][order][:, COC].astype(np.float64)).mean(0)
        r = rho_layers(np.abs(X.sum(1)).mean(0), ref)[:LAST]
        gates[f"A0_{name}"] = {"min_rho": float(np.nanmin(r)), "pass": bool(np.nanmin(r) >= 0.99)}
        lines.append(f"A0 {name}: sum over CoC positions vs anatomy CoC row, per-layer Spearman min {np.nanmin(r):.3f} -> {'PASS' if gates[f'A0_{name}']['pass'] else 'FAIL'}")

    # A1 token weighting
    lines.append("\nA1 token weighting inside the CoC (share of the clip's CoC total, mean over clips; bands 0-21 / 22-34)")
    for name, res, X in (("CE", per["res_ce_norm"], qce), ("FM", per["res_fm_norm"], qfm)):
        res = np.nan_to_num(res.astype(np.float64))  # (N, L, PAD)
        for what, arr in (("residual-gradient norm", res), ("|Q-head contribution|", np.abs(X).sum(3).transpose(0, 2, 1))):  # both (N, L, PAD)
            shares = {}
            for si, sname in enumerate(SPANS):
                num = np.array([[arr[i, l][sub[i] == si].sum() for l in range(LAST)] for i in range(N)])
                den = np.array([[arr[i, l][sub[i] >= 0].sum() for l in range(LAST)] for i in range(N)])
                sh = num / np.maximum(den, 1e-30)
                shares[sname] = {b: float(sh[:, sl].mean()) for b, sl in BANDS.items()}
            tok_share = {sname: float(np.mean([(sub[i] == si).sum() / n_coc[i] for i in range(N)])) for si, sname in enumerate(SPANS)}
            gates[f"A1_{name}_{what}"] = shares
            lines.append(f"  {name} {what:24s}: " + " | ".join(f"{sname} {shares[sname]['0-21']:.2f} / {shares[sname]['22-34']:.2f} (tokens {tok_share[sname]:.2f})" for sname in SPANS))
    # per-position profile from the end (last 5 positions) for the residual gradient
    for name, res in (("CE", per["res_ce_norm"]), ("FM", per["res_fm_norm"])):
        res = np.nan_to_num(res.astype(np.float64))
        prof = np.zeros(6)
        for i in range(N):
            tot = res[i, 7:22, : n_coc[i]].sum()
            for j in range(6):
                if j < n_coc[i]:
                    prof[j] += res[i, 7:22, n_coc[i] - 1 - j].sum() / max(tot, 1e-30) / N
        lines.append(f"  {name} residual-gradient share by position from the END (layers 7-21): last {prof[0]:.2f}, -1 {prof[1]:.2f}, -2 {prof[2]:.2f}, -3 {prof[3]:.2f}, -4 {prof[4]:.2f}, -5 {prof[5]:.2f}")

    # finer split: the last two CoC positions are the special tokens <|cot_end|> and <|traj_future_start|>;
    # <|cot_end|> is separated from the rest of the sentence because the FM gradient concentrates there
    fine = sub.copy()  # 0 head clause, 1 rest of sentence, 2 <|traj_future_start|>, 3 <|cot_end|>
    for i in range(N):
        if n_coc[i] >= 2 and sub[i, n_coc[i] - 2] == 1:
            fine[i, n_coc[i] - 2] = 3
    FINE = ((0, "head clause"), (1, "rest of sentence"), (3, "<|cot_end|>"), (2, "<|traj_future_start|>"))
    lines.append("\nA1b the same shares with <|cot_end|> separated from the rest of the sentence (|Q-head contribution|, bands 0-21 / 22-34)")
    for name, X in (("CE", qce), ("FM", qfm)):
        arr = np.abs(X).sum(3).transpose(0, 2, 1)  # (N, L, PAD)
        parts = []
        for fi, fname in FINE:
            num = np.array([[arr[i, l][fine[i] == fi].sum() for l in range(LAST)] for i in range(N)])
            den = np.array([[arr[i, l][fine[i] >= 0].sum() for l in range(LAST)] for i in range(N)])
            sh = num / np.maximum(den, 1e-30)
            tok = float(np.mean([(fine[i] == fi).sum() / n_coc[i] for i in range(N)]))
            gates[f"A1b_{name}_{fname}"] = {b: float(sh[:, sl].mean()) for b, sl in BANDS.items()} | {"tokens": tok}
            parts.append(f"{fname} {sh[:, BANDS['0-21']].mean():.2f} / {sh[:, BANDS['22-34']].mean():.2f} (tokens {tok:.2f})")
        lines.append(f"  {name}: " + " | ".join(parts))

    # per-clip robustness of the boundary-token concentration (the last two positions)
    for name, X in (("CE", qce), ("FM", qfm)):
        arr = np.abs(X).sum(3).transpose(0, 2, 1)
        for b, sl in BANDS.items():
            tot = np.array([arr[i, sl, : n_coc[i]].sum() for i in range(N)])
            bnd = np.array([arr[i, sl, n_coc[i] - 2 : n_coc[i]].sum() for i in range(N)]) / np.maximum(tot, 1e-30)
            per_b = np.array([arr[i, sl, n_coc[i] - 2 : n_coc[i]].mean() for i in range(N)])
            per_w = np.array([arr[i, sl, : n_coc[i] - 2].mean() for i in range(N)])
            ratio = per_b / np.maximum(per_w, 1e-30)
            gates[f"A1c_{name}_{b}"] = {"boundary_share_mean": float(bnd.mean()), "boundary_share_median": float(np.median(bnd)),
                                        "clips_above_half": float((bnd > 0.5).mean()), "per_token_ratio_median": float(np.median(ratio))}
            lines.append(f"  {name} {b}: share of |Q contributions| on <|cot_end|> + <|traj_future_start|> per clip: mean {bnd.mean():.3f}, median {np.median(bnd):.3f}, "
                         f"IQR {np.percentile(bnd, 25):.3f}-{np.percentile(bnd, 75):.3f}, clips > 0.5: {(bnd > 0.5).mean():.2f}; per token vs a word: median {np.median(ratio):.1f}x")
    res_fm = np.nan_to_num(per["res_fm_norm"].astype(np.float64))
    rb = np.array([res_fm[i, 7:22, n_coc[i] - 2 : n_coc[i]].mean() for i in range(N)])
    rw = np.array([res_fm[i, 7:22, : n_coc[i] - 2].mean() for i in range(N)])
    lines.append(f"  FM residual-gradient norm per token, boundary vs word (layers 7-21): median {np.median(rb / rw):.1f}x (IQR {np.percentile(rb / rw, 25):.1f}-{np.percentile(rb / rw, 75):.1f})")

    # A2 same sub-span agreement (Q)
    lines.append("\nA2 agreement of I_traj and I_CoC restricted to the same sub-span (Q heads; raw on 100-clip means / split-half corrected, self-reliabilities)")
    curves = {}
    sels = [("all CoC", fine >= 0)] + [(fname, fine == fi) for fi, fname in FINE] + \
        [("words only (head + rest)", (fine == 0) | (fine == 1)), ("drop <|traj_future_start|> only", fine != 2)]
    for label, m in sels:
        msk = m[:, :, None, None]
        T = np.abs((qfm * msk).sum(1)); C = np.abs((qce * msk).sum(1))
        raw = rho_layers(T.mean(0), C.mean(0))[:LAST]
        corr, st, sc_ = corrected(T[:, :LAST], C[:, :LAST], 50, rng)
        if label in ("all CoC", "head clause", "rest of sentence", "<|cot_end|>"):
            curves[label] = raw
        gates[f"A2_{label}"] = {b: {"raw": float(np.nanmean(raw[sl])), "corrected": float(np.nanmean(corr[sl]))} for b, sl in BANDS.items()}
        lines.append(f"  {label:26s} " + " | ".join(f"{b}: raw {np.nanmean(raw[sl]):+.2f} corr {np.nanmean(corr[sl]):+.2f} (self {np.nanmean(st[sl]):.2f}/{np.nanmean(sc_[sl]):.2f})" for b, sl in BANDS.items()))
    best = max(gates[f"A2_{s}"]["22-34"]["corrected"] for _, s in FINE if np.isfinite(gates[f"A2_{s}"]["22-34"]["corrected"]))
    lines.append(f"  verdict (22-34): best same-sub-span corrected agreement {best:.2f} -> " + ("token mixture inside the CoC (>= 0.70)" if best >= 0.70 else "different content at the same tokens (< 0.50)" if best < 0.50 else "intermediate"))
    # MLP sub-spans
    if "mlp_ce_sub" in per:
        mce, mfm = per["mlp_ce_sub"].astype(np.float64), per["mlp_fm_sub"].astype(np.float64)  # (N, 3, L, I)
        lines.append("  MLP channels, same sub-span (raw / corrected, 10 splits):")
        for si, sname in enumerate(SPANS):
            T, C = np.abs(mfm[:, si]), np.abs(mce[:, si])
            raw = rho_layers(T.mean(0), C.mean(0))[:LAST]
            corr, st, sc_ = corrected(T[:, :LAST], C[:, :LAST], 10, rng)
            lines.append(f"    {sname:12s} " + " | ".join(f"{b}: raw {np.nanmean(raw[sl]):+.2f} corr {np.nanmean(corr[sl]):+.2f}" for b, sl in BANDS.items()))

    # A3 single-position agreement
    lines.append("\nA3 single-position agreement across heads (one clip, one layer, one CoC position; mean), against the same clip's agreement after summing its CoC positions")
    single, pooled = {b: [] for b in BANDS}, {b: [] for b in BANDS}
    for i in range(N):
        for b, sl in BANDS.items():
            for l in range(sl.start, sl.stop):
                for p in range(n_coc[i]):
                    a, c = np.abs(qfm[i, p, l]), np.abs(qce[i, p, l])
                    if np.ptp(a) > 0 and np.ptp(c) > 0:
                        single[b].append(spearmanr(a, c)[0])
                pooled[b].append(spearmanr(np.abs(qfm[i, :, l].sum(0)), np.abs(qce[i, :, l].sum(0)))[0])
    for b in BANDS:
        gates[f"A3_{b}"] = {"single_position": float(np.nanmean(single[b])), "per_clip_pooled": float(np.nanmean(pooled[b]))}
        lines.append(f"  {b}: single position {np.nanmean(single[b]):+.2f} | same clip, positions summed {np.nanmean(pooled[b]):+.2f}")
    # per clip, positions summed within one span only
    for fi, fname in FINE[:3]:
        vals = {b: [] for b in BANDS}
        for i in range(N):
            m = fine[i] == fi
            if not m.any():
                continue
            for b, sl in BANDS.items():
                for l in range(sl.start, sl.stop):
                    a, c = np.abs(qfm[i, m, l].sum(0)), np.abs(qce[i, m, l].sum(0))
                    if np.ptp(a) > 0 and np.ptp(c) > 0:
                        vals[b].append(spearmanr(a, c)[0])
        gates[f"A3_pooled_{fname}"] = {b: float(np.nanmean(vals[b])) for b in BANDS}
        lines.append(f"  same clip, positions summed within {fname:12s}: " + " | ".join(f"{b} {np.nanmean(vals[b]):+.2f}" for b in BANDS))
    # by sub-span
    for si, sname in enumerate(SPANS):
        vals = []
        for i in range(N):
            for l in range(22, LAST):
                for p in np.where(sub[i] == si)[0]:
                    a, c = np.abs(qfm[i, p, l]), np.abs(qce[i, p, l])
                    if np.ptp(a) > 0 and np.ptp(c) > 0:
                        vals.append(spearmanr(a, c)[0])
        lines.append(f"  22-34 single position within {sname}: {np.nanmean(vals):+.2f} (n={len(vals)})")

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for label, col in zip(curves, ("black", "#2a78d6", "#e87ba4", "#008300")):
        ax.plot(range(LAST), curves[label], color=col, lw=1.3, label=label)
    ax.axvspan(21.5, 35.5, color="#777777", alpha=0.13, lw=0); ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("VLM layer"); ax.set_ylabel("raw rank agreement, I_traj vs I_CoC"); ax.set_ylim(-0.3, 1.05); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "plots" / "same_subspan_agreement.png", dpi=150); plt.close(fig)
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    (out / "metrics.json").write_text(json.dumps({"gates": gates, "n_clips": N}, indent=1, default=float))
    (out / "config.json").write_text(json.dumps({"shards": args.shards, "anatomy": args.anatomy, "plan": "plans/2026-09-25_coc-position-content.md"}, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
