"""hard100 5-arm 리포트의 그림 두 장.

score_5arm.png  다섯 arm 의 씬 평균 점수와 95% 부트스트랩 CI. baseline 만 떨어져 있고 넷은
                CI 가 서로 크게 겹친다 -- 이 실험의 1차 결과다.
forest.png      전 쌍의 페어드 차이를 CI 와 함께 한 줄씩. 위 네 줄(vs baseline)은 0을 벗어나고
                아래 여섯 줄(방법 간)은 전부 0을 가로지른다. "압축 > 무압축은 유지되는데 방법
                간 서열은 없다" 를 한 장으로 보여주는 것이 목적이다.

matplotlib 에 이 서버용 한글 TTF 가 없어 그림 안 글자는 전부 ASCII 로 쓴다.

Usage:
  .venv/bin/python experiments/evaluation/plot_h100_tyrk.py \
      --pairs outputs/h100_tyrK_pairs
"""

import argparse
import itertools
import json
from pathlib import Path

import matplotlib

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

ARMS = [("baseline", "baseline", MUTED),
        ("dual", "dual", "#6A9BCC"),
        ("tyr_c100", "tyr @ calib_100", "#008300"),
        ("tyrK", "tyrK", "#D97757"),
        ("lp_r50", "lp_r50", "#B0A99A")]
COMP = [a for a, _, _ in ARMS if a != "baseline"]
LBL = {a: lab for a, lab, _ in ARMS}


def pk(P, a, b):
    if f"{b} - {a}" in P:
        v = P[f"{b} - {a}"]
        return v["delta"], v["ci_lo"], v["ci_hi"]
    v = P[f"{a} - {b}"]
    return -v["delta"], -v["ci_hi"], -v["ci_lo"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=REPO / "outputs/h100_tyrK_pairs")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    src = args.pairs if args.pairs.is_absolute() else REPO / args.pairs
    out = args.out or (src / "plots")
    out.mkdir(parents=True, exist_ok=True)
    M = json.loads((src / "metrics.json").read_text())
    A, P = M["arms"], M["pairs"]

    # --- 1) 점수 + CI
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for i, (k, label, col) in enumerate(ARMS):
        e = A[k]
        ax.bar(i, e["score"], color=col, width=0.62, alpha=0.9)
        ax.errorbar(i, e["score"],
                    yerr=[[e["score"] - e["ci_lo"]], [e["ci_hi"] - e["score"]]],
                    fmt="none", ecolor=INK, capsize=5, lw=1.3)
        ax.text(i, e["ci_hi"] + 0.014, f"{e['score']:.3f}", ha="center", fontsize=10)
    ax.axhline(A["baseline"]["score"], color=MUTED, ls="--", lw=1)
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([lab for _, lab, _ in ARMS], fontsize=9)
    ax.set_ylabel("scene score")
    ax.set_ylim(0, max(A[k]["ci_hi"] for k, _, _ in ARMS) + 0.08)
    ax.set_title(f"hard100: every compressed arm sits above the uncompressed baseline\n"
                 f"({M['n_scenes']} hard scenes x 2 rollouts, 95% bootstrap CI)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "score_5arm.png", dpi=150)
    plt.close(fig)

    # --- 2) forest
    rows = []
    for k in COMP:
        d, lo, hi = pk(P, "baseline", k)
        rows.append((f"{LBL[k]}  -  baseline", d, lo, hi, "#29261B"))
    for a, b in itertools.combinations(COMP, 2):
        d, lo, hi = pk(P, a, b)
        rows.append((f"{LBL[b]}  -  {LBL[a]}", d, lo, hi, MUTED))
    rows = rows[::-1]

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    for i, (label, d, lo, hi, col) in enumerate(rows):
        crosses = lo <= 0 <= hi
        ax.plot([lo, hi], [i, i], color=col, lw=2, alpha=0.5 if crosses else 0.95)
        ax.plot([d], [i], "o", color=col, ms=7, alpha=0.5 if crosses else 0.95)
    ax.axvline(0, color=INK, lw=1)
    ax.axhline(len(rows) - len(COMP) - 0.5, color=MUTED, lw=0.8, ls=":")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlabel("paired scene-score difference")
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.text(0.985, (len(rows) - len(COMP) / 2 - 0.5) / len(rows), "vs baseline",
            transform=ax.transAxes, ha="right", fontsize=9, color=MUTED)
    ax.text(0.985, 0.06, "method vs method", transform=ax.transAxes, ha="right",
            fontsize=9, color=MUTED)
    ax.set_title("Compression beats no compression; the methods do not separate\n"
                 "(faded = 95% CI crosses zero)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "forest.png", dpi=150)
    plt.close(fig)
    print(f"-> {out}/score_5arm.png, {out}/forest.png")


if __name__ == "__main__":
    main()
