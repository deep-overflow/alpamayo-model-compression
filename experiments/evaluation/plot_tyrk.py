"""tyrK 리포트의 그림 두 장.

score_ci.png   세 arm 의 씬 평균 점수와 95% 부트스트랩 CI. 두 tyr arm 이 겹치고 baseline 만
               떨어져 있는 것이 이 실험의 1차 결과다.
coc_scatter.png 씬별 CoC 퇴화율을 두 arm 축에 뿌린다. 주행은 같은데 CoC 만 갈리는 것이
               대각선 아래로 치우친 구름으로 보인다. 점수 산점도를 나란히 두어 "주행은
               대각선 위에 붙어 있다" 를 같은 그림에서 대조한다.

matplotlib 에 이 서버용 한글 TTF 가 없어 그림 안 글자는 전부 ASCII 로 쓴다.

Usage:
  .venv/bin/python experiments/evaluation/plot_tyrk.py \
      --pairs outputs/tyrK_pairs --out outputs/tyrK_pairs/plots
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})

ARMS = [("baseline", "baseline\n(uncompressed)", MUTED),
        ("tyr_c100", "tyr @ calib_100\n(KL + MSE)", "#008300"),
        ("tyrK", "tyrK\n(KL only)", "#D97757")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=REPO / "outputs/tyrK_pairs")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    src = args.pairs if args.pairs.is_absolute() else REPO / args.pairs
    out = args.out or (src / "plots")
    out.mkdir(parents=True, exist_ok=True)
    M = json.loads((src / "metrics.json").read_text())
    A = M["arms"]

    # --- 1) 점수 + CI
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for i, (k, label, col) in enumerate(ARMS):
        e = A[k]
        ax.bar(i, e["score"], color=col, width=0.6, alpha=0.9)
        ax.errorbar(i, e["score"],
                    yerr=[[e["score"] - e["ci_lo"]], [e["ci_hi"] - e["score"]]],
                    fmt="none", ecolor=INK, capsize=5, lw=1.3)
        ax.text(i, e["ci_hi"] + 0.012, f"{e['score']:.3f}", ha="center", fontsize=10)
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([lab for _, lab, _ in ARMS], fontsize=9)
    ax.set_ylabel("scene score")
    ax.set_ylim(0, max(A[k]["ci_hi"] for k, _, _ in ARMS) + 0.06)
    ax.set_title("Search fitness: dropping the trajectory term costs nothing\n"
                 "(150 scenes x 2 rollouts, 95% bootstrap CI)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "score_ci.png", dpi=150)
    plt.close(fig)

    # --- 2) 씬별 산점도: 주행은 대각선, CoC 는 아래로
    cocf = src / "coc_paired.json"
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
    s = M["per_scene_score"]
    a, b = np.array(s["tyr_c100"]), np.array(s["tyrK"])
    axes[0].scatter(a, b, s=18, color="#008300", alpha=0.45, edgecolors="none")
    axes[0].plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    axes[0].set_xlabel("tyr @ calib_100 — scene score")
    axes[0].set_ylabel("tyrK — scene score")
    axes[0].set_title(f"Driving: no shift (mean {b.mean() - a.mean():+.4f})", fontsize=10)
    axes[0].set_xlim(-0.03, 1.03)
    axes[0].set_ylim(-0.03, 1.03)

    if cocf.exists():
        C = json.loads(cocf.read_text())["per_scene"]
        ca = np.array(C["tyr_c100"]["degenerate_frac"]) * 100
        cb = np.array(C["tyrK"]["degenerate_frac"]) * 100
        hi = max(ca.max(), cb.max()) * 1.05
        axes[1].scatter(ca, cb, s=18, color="#D97757", alpha=0.45, edgecolors="none")
        axes[1].plot([0, hi], [0, hi], color=MUTED, lw=1, ls="--")
        axes[1].set_xlabel("tyr @ calib_100 — CoC degeneracy (%)")
        axes[1].set_ylabel("tyrK — CoC degeneracy (%)")
        axes[1].set_title(f"CoC: shifted below the line ({cb.mean() - ca.mean():+.2f} pp)",
                          fontsize=10)
        axes[1].set_xlim(-hi * 0.03, hi)
        axes[1].set_ylim(-hi * 0.03, hi)
    else:
        axes[1].text(0.5, 0.5, "coc_paired.json missing", ha="center", va="center",
                     transform=axes[1].transAxes, color=MUTED)
        axes[1].set_axis_off()
    fig.suptitle("Same 150 scenes, paired: the two arms differ in CoC, not in driving",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "coc_scatter.png", dpi=150)
    plt.close(fig)
    print(f"-> {out}/score_ci.png, {out}/coc_scatter.png")


if __name__ == "__main__":
    main()
