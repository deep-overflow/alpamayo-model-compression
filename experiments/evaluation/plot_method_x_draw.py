"""방법 x 캘리브레이션 추출 격자의 대표 그림 두 장.

`analyze_calibsize.py` 의 페어 그림은 21쌍이라 세로로 너무 길고, 이 실험의 요점 -- "추출을
바꿨을 때 방법마다 얼마나 흔들리는가" -- 이 한눈에 안 들어온다. 그래서 두 장을 따로 그린다:

  draw_spread.png   방법별로 세 추출의 점수를 나란히. 같은 방법의 세 점이 얼마나 벌어지는지가
                    이 실험의 결과다 (tyr 범위 0.009 vs dfh4 0.076).
  score_vs_coc.png  주행 점수와 CoC 퇴화율의 산점도. 두 방법이 서로 반대 축에서 흔들린다는
                    관찰 -- tyr 은 CoC 가, dfh4 는 주행이 -- 을 한 장에 담는다.

Usage:
  .venv/bin/python experiments/evaluation/plot_method_x_draw.py \
      --pairs outputs/method_x_draw_pairs --out outputs/method_x_draw_pairs/plots
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

# (표시 이름, arm 키 3개, 색)
METHODS = [
    ("tyr (output reconstruction)", ["tyr_c100", "tyr_rd_a", "tyr_rd_b"], "#008300"),
    ("dual+fisher+h4 (2nd-order Fisher)", ["dfh4_c100", "dfh4_rd_a", "dfh4_rd_b"], "#D97757"),
]
DRAWS = ["calib_100", "rd100_a", "rd100_b"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=REPO / "outputs/method_x_draw_pairs")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    src = args.pairs if args.pairs.is_absolute() else REPO / args.pairs
    out = args.out or (src / "plots")
    out.mkdir(parents=True, exist_ok=True)
    M = json.loads((src / "metrics.json").read_text())["arms"]
    base = M["baseline"]["score"]

    # --- 1) 추출 분산
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for mi, (label, keys, col) in enumerate(METHODS):
        xs = np.arange(3) + mi * 3.6
        for x, k in zip(xs, keys):
            e = M[k]
            ax.bar(x, e["score"], color=col, width=0.66, alpha=0.9)
            ax.errorbar(x, e["score"],
                        yerr=[[e["score"] - e["ci_lo"]], [e["ci_hi"] - e["score"]]],
                        fmt="none", ecolor=INK, capsize=4, lw=1.2)
            ax.text(x, e["ci_hi"] + 0.012, f"{e['score']:.3f}", ha="center", fontsize=9)
        v = np.array([M[k]["score"] for k in keys])
        ax.text(xs.mean(), 0.03,
                f"{label}\nrange {v.max() - v.min():.3f}   sd {v.std(ddof=1):.3f}",
                ha="center", fontsize=9, color=INK)
    ax.axhline(base, color=MUTED, ls="--", lw=1.2)
    # 오른쪽 끝은 rd100_b 막대와 겹친다 -- 두 그룹 사이 빈 자리에 둔다
    ax.text(2.8, base + 0.008, f"baseline {base:.3f}", ha="center", fontsize=9, color=MUTED)
    ax.set_xticks(list(np.arange(3)) + list(np.arange(3) + 3.6))
    ax.set_xticklabels(DRAWS * 2, fontsize=8)
    ax.set_ylabel("scene score")
    ax.set_ylim(0, max(M[k]["ci_hi"] for _, ks, _ in METHODS for k in ks) + 0.07)
    ax.set_title("Two methods over the same three calibration draws\n"
                 "(150 scenes x 2 rollouts, 95% bootstrap CI)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "draw_spread.png", dpi=150)
    plt.close(fig)

    # --- 2) 주행 vs CoC
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for label, keys, col in METHODS:
        xs = [M[k]["coc_degenerate"] * 100 for k in keys]
        ys = [M[k]["score"] for k in keys]
        ax.plot(xs, ys, "o-", color=col, ms=8, lw=1.2, label=label.split(" (")[0])
        for x, y, d in zip(xs, ys, DRAWS):
            ax.annotate(d, (x, y), textcoords="offset points", xytext=(7, -3),
                        fontsize=8, color=MUTED)
    b = M["baseline"]
    ax.plot(b["coc_degenerate"] * 100, b["score"], "s", color=MUTED, ms=8)
    ax.annotate("baseline", (b["coc_degenerate"] * 100, b["score"]),
                textcoords="offset points", xytext=(7, -3), fontsize=8, color=MUTED)
    ax.set_xlabel("CoC degeneracy (%)")
    ax.set_ylabel("scene score")
    ax.set_title("Driving score and CoC move on different axes", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "score_vs_coc.png", dpi=150)
    plt.close(fig)
    print(f"-> {out}/draw_spread.png, {out}/score_vs_coc.png")


if __name__ == "__main__":
    main()
