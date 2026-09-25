"""Which text tokens carry I_FM (and I_CE)? From run_text_position_anatomy.py shards: every non-vision
position (sink, 48 ego-history tokens, prompt text, generated CoC) has its own signed Q-head
contribution under each loss; vision is one summed type.

Roles of the prompt positions (by decoded token; the prompt is identical across clips, so roles
are printed once for audit):
  chat markers    <|im_start|> / <|im_end|> / system / user / assistant / newlines
  system text     the system sentence
  vision markers  <|vision_start|> / <|vision_end|>
  camera text     text between the first <|vision_start|> and the last <|vision_end|> (camera names etc.)
  hist markers    <|traj_history_start|> / <|traj_history_end|>
  hist            the 48 ego-history tokens
  instruction     text after <|traj_history_end|> up to <|im_end|>
  cot_start       <|cot_start|> (the assistant-turn start, right before the generated CoC)
  coc words / cot_end / traj_start   the generated CoC and its two boundary tokens
  sink            position 0

Reports per band (0-21 / 22-34): share of |Q-head contribution| by role (mean over clips; the
vision share alongside), token counts and enrichment, FM - CE with clip-bootstrap CIs; the
prompt positions with the largest FM share; the ego-history index profile; residual-gradient
shares by role; and the same-role rank agreement of I_FM and I_CE (raw / split-half corrected).

Usage:
  python experiments/head_analysis/analyze_text_position_anatomy.py --shards textpos_v1_s0 ... --out textpos_v1
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
LAST = 35
BANDS = {"0-21": slice(0, 22), "22-34": slice(22, LAST)}
ROLES = ("sink", "chat markers", "system text", "vision markers", "camera text", "hist markers", "hist", "instruction", "cot_start",
         "coc words", "cot_end", "traj_start", "other text")
CHAT = {"<|im_start|>", "<|im_end|>", "system", "user", "assistant", "\n", "\n\n"}
rng = np.random.default_rng(0)


def roles_for(kind, tokens):
    n = len(tokens)
    role = ["other text"] * n
    first_vs = next((i for i, t in enumerate(tokens) if t == "<|vision_start|>"), n)
    last_ve = max((i for i, t in enumerate(tokens) if t == "<|vision_end|>"), default=-1)
    hist_end = next((i for i, t in enumerate(tokens) if t == "<|traj_history_end|>"), n)
    im_end_after = next((i for i, t in enumerate(tokens) if t == "<|im_end|>" and i > hist_end), n)
    n_coc = sum(k == 3 for k in kind)
    coc_idx = [i for i, k in enumerate(kind) if k == 3]
    for i, (k, t) in enumerate(zip(kind, tokens)):
        if k == 0:
            role[i] = "sink"
        elif k == 1:
            role[i] = "hist"
        elif k == 3:
            role[i] = "coc words"
        elif t in ("<|vision_start|>", "<|vision_end|>"):
            role[i] = "vision markers"
        elif t in ("<|traj_history_start|>", "<|traj_history_end|>"):
            role[i] = "hist markers"
        elif t == "<|cot_start|>":
            role[i] = "cot_start"
        elif t.strip() in CHAT or t in CHAT:
            role[i] = "chat markers"
        elif i < first_vs:
            role[i] = "system text"
        elif i < last_ve:
            role[i] = "camera text"
        elif hist_end < i < im_end_after:
            role[i] = "instruction"
    if n_coc >= 2:
        role[coc_idx[-2]] = "cot_end"
        role[coc_idx[-1]] = "traj_start"
    return role


def rho_layers(a, b):
    return np.array([spearmanr(a[l], b[l])[0] if np.ptp(a[l]) > 0 and np.ptp(b[l]) > 0 else np.nan for l in range(a.shape[0])])


def corrected(t, c, n_split):
    n = len(t)
    st, sc_, cr = [], [], []
    for _ in range(n_split):
        p = rng.permutation(n)
        a, b = p[: n // 2], p[n // 2 :]
        st.append(rho_layers(t[a].mean(0), t[b].mean(0)))
        sc_.append(rho_layers(c[a].mean(0), c[b].mean(0)))
        cr.append(0.5 * (rho_layers(t[a].mean(0), c[b].mean(0)) + rho_layers(t[b].mean(0), c[a].mean(0))))
    st, sc_, cr = (np.nanmean(v, 0) for v in (st, sc_, cr))
    ok = (st > 0.2) & (sc_ > 0.2)
    return np.where(ok, cr / np.sqrt(np.where(ok, st * sc_, 1.0)), np.nan)


def ci_of(v, n=3000):
    v = np.asarray(v, float)
    boot = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)])
    return float(v.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = REPO / "outputs" / args.out
    (out / "plots").mkdir(parents=True, exist_ok=True)
    rows, per = [], {}
    for s in args.shards:
        m = json.loads((REPO / "outputs" / s / "metrics.json").read_text())
        z = np.load(REPO / "outputs" / s / "textpos_perclip.npz")
        rows += m["per_clip"]
        for k in z.files:
            per.setdefault(k, []).append(z[k])
    per = {k: np.concatenate(v) for k, v in per.items()}
    N = len(rows)
    q = {"CE": np.abs(np.nan_to_num(per["q_ce_pos"].astype(np.float64))), "FM": np.abs(np.nan_to_num(per["q_fm_pos"].astype(np.float64)))}  # (N, PAD, L, H)
    qv = {"CE": np.abs(np.array([r["q_vision_ce"] for r in rows], float)), "FM": np.abs(np.array([r["q_vision_fm"] for r in rows], float))}  # (N, L, H)
    res = {"CE": np.nan_to_num(per["res_ce_norm"].astype(np.float64)), "FM": np.nan_to_num(per["res_fm_norm"].astype(np.float64))}  # (N, L, PAD)
    roles = [roles_for(r["kind"], r["tokens"]) for r in rows]
    n_pos = np.array([r["n_pos"] for r in rows])
    n_prompt = np.array([sum(k != 3 for k in r["kind"]) for r in rows])
    aligned = all(r["tokens"][: n_prompt[0]] == rows[0]["tokens"][: n_prompt[0]] for r in rows) and (n_prompt == n_prompt[0]).all()
    lines = [f"Text-position anatomy -- {N} clips, shards {', '.join(args.shards)}; prompt positions aligned across clips: {aligned}", ""]
    lines.append("prompt roles (clip 0):")
    for role in ROLES:
        toks = [rows[0]["tokens"][i] for i in range(n_prompt[0]) if roles[0][i] == role]
        if toks:
            lines.append(f"  {role:15s} n={len(toks):3d}: " + " ".join(repr(t) for t in toks[:40]) + (" ..." if len(toks) > 40 else ""))
    lines.append("")
    metrics = {"n_clips": N, "aligned": bool(aligned), "bands": {}}

    for b, sl in BANDS.items():
        mb = {}
        lines.append(f"===== band {b} =====")
        # per-position |Q| summed over the band's heads and layers -> (N, PAD); vision total per clip
        tok = {L: q[L][:, :, sl, :].sum((2, 3)) for L in q}
        vis = {L: qv[L][:, sl, :].sum((1, 2)) for L in q}
        # 1. role shares (over ALL positions including vision, and over non-vision only)
        for normname, with_vis in (("all positions (vision as one block)", True), ("non-vision positions only", False)):
            share = {L: {r: np.zeros(N) for r in ROLES + ("vision",)} for L in q}
            tshare = {r: np.zeros(N) for r in ROLES}
            for i in range(N):
                rl = np.array(roles[i])
                for L in q:
                    v = tok[L][i, : n_pos[i]]
                    tot = v.sum() + (vis[L][i] if with_vis else 0.0)
                    for r in ROLES:
                        share[L][r][i] = v[rl == r].sum() / max(tot, 1e-30)
                    share[L]["vision"][i] = vis[L][i] / max(tot, 1e-30) if with_vis else 0.0
                for r in ROLES:
                    tshare[r][i] = (rl == r).mean()
            lines.append(f"[{b}] |Q-head contribution| share by role, normalisation: {normname} (token share among non-vision tokens; enrichment = share / token share)")
            lines.append(f"  {'role':15s} {'n tok':>5s} {'tok sh':>6s} | {'CE share':>8s} {'enr':>5s} | {'FM share':>8s} {'enr':>5s} | FM-CE share [CI]")
            res_ = {}
            for r in ROLES + (("vision",) if with_vis else ()):
                if r != "vision" and tshare[r].mean() == 0:
                    continue
                ts = tshare[r].mean() if r != "vision" else np.nan
                d = share["FM"][r] - share["CE"][r]
                m, lo, hi = ci_of(d)
                res_[r] = {"token_share": float(ts) if r != "vision" else None, "CE_share": float(share["CE"][r].mean()), "FM_share": float(share["FM"][r].mean()),
                           "d_share": m, "d_share_ci": [lo, hi]}
                ntok = round(ts * n_pos.mean()) if r != "vision" else 2880
                e_ce = f"{share['CE'][r].mean() / ts:5.2f}" if r != "vision" else "    -"
                e_fm = f"{share['FM'][r].mean() / ts:5.2f}" if r != "vision" else "    -"
                lines.append(f"  {r:15s} {ntok:5d} {ts if r != 'vision' else float('nan'):6.3f} | {share['CE'][r].mean():8.3f} {e_ce} | {share['FM'][r].mean():8.3f} {e_fm} | {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            mb["roles_" + ("all" if with_vis else "nonvis")] = res_
        # 2. top prompt positions by FM share (non-vision normalisation), aligned across clips
        if aligned:
            P = n_prompt[0]
            sh = {L: np.stack([np.pad(tok[L][i, : n_pos[i]] / max(tok[L][i, : n_pos[i]].sum(), 1e-30), (0, tok[L].shape[1] - n_pos[i])) for i in range(N)]) for L in q}  # (N, PAD)
            mean_sh = {L: sh[L][:, :P].mean(0) for L in q}
            order = np.argsort(-mean_sh["FM"])
            lines.append(f"[{b}] prompt positions with the largest FM share (of the clip's non-vision total; mean over clips), with the CE share")
            lines.append(f"  {'pos':>4s} {'role':15s} {'token':24s} | {'FM':>7s} {'CE':>7s} {'FM/CE':>6s}")
            top = []
            for p in order[:25]:
                t = rows[0]["tokens"][p]
                top.append({"pos": int(p), "role": roles[0][p], "token": t, "FM": float(mean_sh["FM"][p]), "CE": float(mean_sh["CE"][p])})
                lines.append(f"  {p:4d} {roles[0][p]:15s} {t!r:24s} | {mean_sh['FM'][p]:7.4f} {mean_sh['CE'][p]:7.4f} {mean_sh['FM'][p] / max(mean_sh['CE'][p], 1e-30):6.2f}")
            mb["top_prompt_positions_FM"] = top
            mb["prompt_position_share"] = {L: mean_sh[L].tolist() for L in q}
            # ego-history index profile
            hist_pos = [p for p in range(P) if roles[0][p] == "hist"]
            prof = {L: mean_sh[L][hist_pos] for L in q}
            lines.append(f"[{b}] ego-history tokens: share by history index (0 = first, {len(hist_pos) - 1} = last; of the non-vision total)")
            for L in q:
                lines.append(f"  {L}: " + " ".join(f"{v:.4f}" for v in prof[L]) + f" | total {prof[L].sum():.3f}, last 8 / first 8 = {prof[L][-8:].sum() / max(prof[L][:8].sum(), 1e-30):.2f}, Spearman(index, share) {spearmanr(np.arange(len(hist_pos)), prof[L])[0]:+.2f}")
            mb["hist_profile"] = {L: prof[L].tolist() for L in q}
        # 3. residual-gradient norm shares by role (token weighting independent of units)
        lines.append(f"[{b}] residual-gradient norm share by role (non-vision positions; vision not measured)")
        rs = {L: {r: [] for r in ROLES} for L in q}
        for i in range(N):
            rl = np.array(roles[i])
            for L in q:
                v = res[L][i, sl, : n_pos[i]].sum(0)
                tot = max(v.sum(), 1e-30)
                for r in ROLES:
                    rs[L][r].append(v[rl == r].sum() / tot)
        lines.append("  " + " ".join(f"{r[:12]:>12s}" for r in ROLES if np.mean(rs['CE'][r]) > 0 or np.mean(rs['FM'][r]) > 0))
        for L in q:
            lines.append(f"  {L}: " + " ".join(f"{np.mean(rs[L][r]):12.3f}" for r in ROLES if np.mean(rs['CE'][r]) > 0 or np.mean(rs['FM'][r]) > 0))
        mb["residual_share"] = {L: {r: float(np.mean(rs[L][r])) for r in ROLES} for L in q}
        # 4. same-role rank agreement of I_FM and I_CE (Q heads; raw / corrected)
        lines.append(f"[{b}] same-role rank agreement of I_FM and I_CE across Q heads (raw on the clip means / split-half corrected; single-position mean where the role is one token)")
        agr = {}
        for r in ROLES:
            T_, C_ = [], []
            for i in range(N):
                rl = np.array(roles[i])
                idx = np.where(rl == r)[0]
                if len(idx) == 0:
                    continue
                T_.append(q["FM"][i, idx][:, :LAST].sum(0))
                C_.append(q["CE"][i, idx][:, :LAST].sum(0))
            if len(T_) < 10:
                continue
            T_, C_ = np.stack(T_), np.stack(C_)
            raw = rho_layers(T_.mean(0), C_.mean(0))
            corr = corrected(T_, C_, 30)
            single = np.nanmean([spearmanr(T_[i, l], C_[i, l])[0] for i in range(len(T_)) for l in range(sl.start, sl.stop) if np.ptp(T_[i, l]) > 0 and np.ptp(C_[i, l]) > 0])
            agr[r] = {"raw": float(np.nanmean(raw[sl])), "corrected": float(np.nanmean(corr[sl])), "single_clip": float(single)}
            lines.append(f"  {r:15s} raw {np.nanmean(raw[sl]):+.2f} corr {np.nanmean(corr[sl]):+.2f} | per clip {single:+.2f}")
        mb["agreement"] = agr
        lines.append("")
        metrics["bands"][b] = mb

    # plots
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, b in zip(axes, BANDS):
        m = metrics["bands"][b]["roles_nonvis"]
        rs_ = [r for r in ROLES if r in m]
        x = np.arange(len(rs_))
        ax.bar(x - 0.2, [m[r]["token_share"] for r in rs_], 0.2, color="#bbbbbb", label="token share")
        ax.bar(x, [m[r]["CE_share"] for r in rs_], 0.2, color="#ECAE3C", label="I_CoC")
        ax.bar(x + 0.2, [m[r]["FM_share"] for r in rs_], 0.2, color="#618BC8", label="I_traj")
        ax.set_xticks(x); ax.set_xticklabels(rs_, rotation=45, ha="right", fontsize=7); ax.set_title(f"layers {b}, non-vision positions", fontsize=9)
    axes[0].set_ylabel("share of the clip's non-vision total"); axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "plots" / "role_shares.png", dpi=150); plt.close(fig)
    if aligned:
        fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
        P = n_prompt[0]
        cols = {r: c for r, c in zip(ROLES, plt.cm.tab20(np.linspace(0, 1, len(ROLES))))}
        for ax, b in zip(axes, BANDS):
            sh = metrics["bands"][b]["prompt_position_share"]
            for L, ls in (("FM", "-"), ("CE", "--")):
                ax.plot(range(P), sh[L][:P], ls, color="black", lw=0.8, label=f"I_{'traj' if L == 'FM' else 'CoC'}")
            for p in range(P):
                ax.axvspan(p - 0.5, p + 0.5, color=cols[roles[0][p]], alpha=0.25, lw=0)
            ax.set_ylabel(f"share, layers {b}", fontsize=8)
        axes[0].legend(fontsize=7)
        axes[1].set_xlabel("non-vision prompt position (colour = role)")
        fig.tight_layout(); fig.savefig(out / "plots" / "prompt_position_share.png", dpi=150); plt.close(fig)

    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    (out / "config.json").write_text(json.dumps({"shards": args.shards, "roles": ROLES}, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
