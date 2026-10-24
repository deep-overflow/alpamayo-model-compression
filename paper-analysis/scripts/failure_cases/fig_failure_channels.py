"""fig7_failure_channels: (a) OOD damage per arm split into action channel (GT CoC teacher-forced)
and language channel (own rollout minus teacher-forced), all clips and healthy-CoC clips;
(b) action-channel damage by manoeuvre, relative to the dense minADE of the manoeuvre;
(c) tail transfer: median d minADE (OOD) and median d score (alpasim 150) of every arm on the
trajectory arm's and the CoC arm's worst decile."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
COL = {"coc": "#e87ba4", "traj": "#2a78d6", "dual": "#008300"}; LAB = {"coc": "CoC-only", "traj": "trajectory-only", "dual": "dual"}
ood = pd.read_parquet("openloop_ood.parquet"); cl = pd.read_parquet("closedloop_150.parquet")
for a in ("coc", "traj", "dual"):
    ood[f"lang_{a}"] = ood[f"d_{a}"] - ood[f"dtf_{a}"]
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
# (a) channels
ax = axes[0]; x = np.arange(3); w = 0.38
for i, a in enumerate(("coc", "traj", "dual")):
    for j, (sel, hatch, lab) in enumerate(((np.ones(len(ood), bool), "", "all clips"), (~ood[f"deg_{a}"].values, "//", "healthy CoC only"))):
        act, lang = ood.loc[sel, f"dtf_{a}"].mean(), ood.loc[sel, f"lang_{a}"].mean()
        xx = x[i] + (j - 0.5) * w
        ax.bar(xx, act, w, color=COL[a], alpha=0.95, hatch=hatch, edgecolor="white", label="action channel (GT CoC teacher-forced)" if i == 0 and j == 0 else None)
        ax.bar(xx, lang, w, bottom=act, color=COL[a], alpha=0.45, hatch=hatch, edgecolor="white", label="language channel (own CoC on top)" if i == 0 and j == 0 else None)
        ax.text(xx, act + max(lang, 0) + 0.01, f"{act + lang:+.2f}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels([LAB[a] for a in ("coc", "traj", "dual")]); ax.set_ylabel("d minADE vs dense (m)")
ax.set_title("(a) damage by channel, OOD (solid: all clips; hatched: healthy CoC)", fontsize=8.5); ax.legend(fontsize=7, loc="upper right")
# (b) action channel by manoeuvre, relative
ax = axes[1]; man = ["cruise", "accel", "decel_stop", "turn"]; x = np.arange(4)
for i, a in enumerate(("coc", "traj", "dual")):
    rel = [ood.loc[ood.bucket == m, f"dtf_{a}"].mean() / ood.loc[ood.bucket == m, "adetf_base"].mean() for m in man]
    ax.bar(x + (i - 1) * 0.27, rel, 0.27, color=COL[a], label=LAB[a])
ax.set_xticks(x); ax.set_xticklabels(man); ax.set_ylabel("action damage / dense minADE"); ax.axhline(0, color="black", lw=0.6)
ax.set_title("(b) action channel by manoeuvre (GT CoC teacher-forced)", fontsize=8.5); ax.legend(fontsize=7)
# (c) tail transfer
ax = axes[2]; k = len(ood) // 10; kc = 15
rows = []
for a in ("traj", "coc"):
    t = ood[f"d_{a}"].rank(ascending=False) <= k; tc = cl[f"d_{a}"].rank() <= kc
    rows.append((f"{LAB[a]} worst decile\n(open loop, OOD)", [ood.loc[t, f"d_{b}"].median() for b in ("coc", "traj", "dual")]))
    rows.append((f"{LAB[a]} worst 15 scenes\n(alpasim 150)", [-cl.loc[tc, f"d_{b}"].median() for b in ("coc", "traj", "dual")]))
y = np.arange(len(rows))
for i, b in enumerate(("coc", "traj", "dual")):
    ax.barh(y + (i - 1) * 0.27, [r[1][i] for r in rows], 0.27, color=COL[b], label=LAB[b])
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=7); ax.invert_yaxis(); ax.axvline(0, color="black", lw=0.6)
ax.set_xlabel("median damage on that tail (m; alpasim: -d score)"); ax.set_title("(c) each arm on the other arm's worst samples", fontsize=8.5); ax.legend(fontsize=7, loc="lower right")
fig.tight_layout(pad=0.6); fig.savefig("fig7_failure_channels.png", dpi=200); print("figure written")
