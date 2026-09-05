"""hard100 폐루프 분석 — 있는 arm 만큼 처리한다 (baseline / dual / tyr_r).

150씬 매트릭스가 자기가 뽑혀 나온 스위트보다 쉽다는 사실(0.742 vs 0.660, p=0.039) 때문에 만든
새 스플릿이다. 그래서 1단계는 결과 해석이 아니라 셋업 검증이다 — 사전등록 게이트 S1 은
baseline 이 0.42-0.58 밖이면 씬이 아니라 셋업을 의심하라고 정해 두었고, 사전 예측 0.499 는
sangoh 님 913씬 런에서 이 100씬만 뽑은 실측이다.

arm 이 둘 이상이면 모든 쌍에 대해 페어드 비교를 낸다. G1/G2 는 baseline 대비 dual 에,
dual vs tyr_r 은 "같은 예산(2,657,452,032 = 24.0%)에서 고르기(Taylor) vs 다시 쓰기(재구성)"
대조에 해당한다. 150씬에서는 dual 0.828 / tyr 0.786, 어려움 계층(편향제거)에서 +0.175 / +0.118.

Usage:
  python analyze_hard100.py --out outputs/hard100_eval
"""

import argparse
import collections
import csv
import itertools
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy import stats

REPO = Path("/home/cvlab21/project/chan/alpamayo-model-compression")
SG = Path("/mnt/nvme1n1/ad_vla/results/sangoh/alpasim_runs/eval2601_a1_5")
FEATS = REPO / "outputs/scene_difficulty/scene_feats_public2601.json"
S1_LO, S1_HI, S1_PRIOR = 0.42, 0.58, 0.499
ARMS = [("baseline", "h100_merged_baseline"),
        ("dual", "h100_merged_slim_dual_u40_v2"),
        ("tyr_r", "h100_merged_slim_tyr_u40_r"),
        ("lp_r50", "h100_merged_lp_r50")]
# 150씬 매트릭스에서의 전체 평균 점수 (계층별 값이 아니라 arm 단위 평균).
# lp_r50 은 soowon 님 런(cl150_merged_lp_r50)의 값이고, 예산이 25.0% 로 우리 24.0% 보다 크다.
REF_150 = {"baseline": 0.750, "dual": 0.828, "tyr_r": 0.786, "lp_r50": 0.810}
# arm 별 고정 색 (arm 이 늘어도 플롯에서 서로 안 겹치도록)
ARM_COLOR = {"baseline": "#6B6555", "dual": "#D97757", "tyr_r": "#008300", "lp_r50": "#2a78d6"}

BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
C1, C2, ACC = "#2a78d6", "#008300", "#D97757"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load(p):
    f = Path(p) / "aggregate" / "results-summary.json"
    if not f.exists():
        return None
    per = collections.defaultdict(list)
    for r in json.loads(f.read_text())["rollouts"]:
        per[r["clipgt_id"]].append(r)
    return per


def sc(per, keys):
    return np.array([np.mean([r["score"] for r in per[s]]) for s in keys])


def rate(per, keys, k):
    rr = [r for s in keys for r in per[s]]
    return sum(1 for r in rr if r["metrics"].get(k, 0) > 0), len(rr)


def paired(a, b, name_a, name_b):
    """b - a 의 페어드 통계."""
    d = b - a
    return {"pair": f"{name_b} - {name_a}", "delta": float(d.mean()),
            "se": float(d.std(ddof=1) / np.sqrt(len(d))), "median": float(np.median(d)),
            "wilcoxon_p": float(stats.wilcoxon(d).pvalue) if np.any(d) else 1.0,
            "better": int((d > 0).sum()), "worse": int((d < 0).sum()),
            "tie": int((d == 0).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path,
                    default=Path("/home/cvlab21/project/chan/alpasim-runs"))
    ap.add_argument("--out", type=Path, default=REPO / "outputs/hard100_eval")
    args = ap.parse_args()
    R = args.runs_root
    out = args.out if args.out.is_absolute() else REPO / args.out
    (out / "plots").mkdir(parents=True, exist_ok=True)

    runs = {n: load(R / d) for n, d in ARMS}
    runs = {n: v for n, v in runs.items() if v is not None}
    if "baseline" not in runs:
        raise SystemExit("h100_merged_baseline 이 아직 없습니다")
    keys = sorted(set.intersection(*(set(v) for v in runs.values())))
    S = {n: sc(v, keys) for n, v in runs.items()}
    b = S["baseline"]
    print(f"arm {list(runs)} | 공통 씬 {len(keys)}")

    suite = {r["scene_id"] for r in csv.DictReader((R / "hard100/hard100_suite.csv").open())}
    M = {"n_scenes": len(keys), "suite_match": len(set(keys) & suite), "arms": {}}
    for n, v in runs.items():
        e = {"mean": float(S[n].mean()), "median": float(np.median(S[n])),
             "sd": float(S[n].std(ddof=1)), "perfect": int((S[n] == 1).sum()),
             "zero": int((S[n] == 0).sum()), "below_0.7": int((S[n] < 0.7).sum()),
             "ref_150": REF_150.get(n)}
        for k in ("offroad", "collision_at_fault", "collision_any", "img_is_black"):
            c, N = rate(v, keys, k)
            e[k] = {"n": c, "N": N, "pct": round(100 * c / N, 2)}
        M["arms"][n] = e

    M["S1"] = {"lo": S1_LO, "hi": S1_HI, "prior": S1_PRIOR, "observed": float(b.mean()),
               "pass": bool(S1_LO <= b.mean() <= S1_HI)}

    # 모든 쌍 페어드 비교
    M["pairs"] = {}
    for x, y in itertools.combinations(runs, 2):
        p = paired(S[x], S[y], x, y)
        nx, N = rate(runs[x], keys, "collision_at_fault")
        ny, _ = rate(runs[y], keys, "collision_at_fault")
        odds, pf = stats.fisher_exact([[ny, N - ny], [nx, N - nx]])
        p["at_fault"] = {f"{x}": nx, f"{y}": ny, "N": N,
                         "odds_ratio": float(odds), "fisher_p": float(pf),
                         "underpowered": bool(nx + ny < 20)}
        M["pairs"][f"{x}_vs_{y}"] = p

    if "dual" in runs:
        g1 = M["pairs"]["baseline_vs_dual"]
        M["G1"] = {**g1, "pass": bool(g1["delta"] >= 0)}
        af = g1["at_fault"]
        M["G2"] = {**af, "pass": bool(af["odds_ratio"] <= 1.0)}

    # sangoh 913씬 런 대조 (셋업 재현성)
    sg = load(SG)
    if sg:
        common = [s for s in keys if s in sg]
        if common:
            a = sc(sg, common)
            c = sc(runs["baseline"], common)
            M["vs_sangoh"] = {"n": len(common), "sangoh": float(a.mean()),
                              "ours": float(c.mean()), "delta": float(c.mean() - a.mean()),
                              "wilcoxon_p": float(stats.wilcoxon(c - a).pvalue)
                              if np.any(c - a) else 1.0,
                              "pearson_r": float(np.corrcoef(a, c)[0, 1])}

    # 150씬 매트릭스 대비
    m150 = load(R / "m2601_merged_baseline")
    if m150:
        k150 = sorted(m150)
        o = sc(m150, k150)
        M["vs_150"] = {"score_150": float(o.mean()), "score_hard100": float(b.mean()),
                       "delta": float(b.mean() - o.mean())}
        for k in ("offroad", "collision_at_fault"):
            n0, N0 = rate(m150, k150, k)
            n1, N1 = rate(runs["baseline"], keys, k)
            M["vs_150"][k] = {"pct_150": round(100 * n0 / N0, 2),
                              "pct_hard100": round(100 * n1 / N1, 2)}

    (out / "metrics.json").write_text(json.dumps(M, indent=2))
    (out / "config.json").write_text(json.dumps(
        {"runs_root": str(R), "suite": "public_2601_hard100", "n_scenes": len(keys),
         "n_rollouts": 2, "arms": list(runs),
         "gates": {"S1": f"{S1_LO}-{S1_HI} (prior {S1_PRIOR})",
                   "G1": "dual - baseline >= 0", "G2": "at-fault OR <= 1.0"}}, indent=2))

    # --- plot 1: 점수 분포 (hard100 vs 150씬)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bins = np.linspace(0, 1, 21)
    if m150:
        ax.hist(o, bins=bins, alpha=.5, color=C1, label=f"150-scene baseline ({o.mean():.3f})")
    ax.hist(b, bins=bins, alpha=.75, color=ACC, label=f"hard100 baseline ({b.mean():.3f})")
    ax.axvline(S1_PRIOR, color=MUTED, ls="--", lw=1, label=f"pre-registered prior {S1_PRIOR}")
    ax.set_xlabel("unpruned baseline scene score"); ax.set_ylabel("scenes")
    ax.set_title("hard100 really is harder")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(out / "plots/score_dist.png", dpi=150); plt.close(fig)

    # --- plot 2: arm 별 점수 + 과실 충돌
    if len(runs) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        names = list(runs)
        cols = [ARM_COLOR.get(n, C1) for n in names]
        axes[0].bar(names, [S[n].mean() for n in names], .55, color=cols)
        for i, n in enumerate(names):
            axes[0].text(i, S[n].mean() + .012, f"{S[n].mean():.3f}", ha="center", fontsize=10)
            if REF_150.get(n):
                axes[0].plot([i - .27, i + .27], [REF_150[n]] * 2, color=C1, lw=1.4, ls="--")
        axes[0].set_ylabel("scene score"); axes[0].set_ylim(0, 1.0)
        axes[0].set_title("hard100 score (dashed = 150-scene result)")
        # 사건이 6-8 건 뿐이라 막대 높이만 보면 큰 차이처럼 읽힌다.
        # Wilson 95% 구간과 사건 수를 함께 그려 "구분되지 않는다"를 그림이 말하게 한다.
        pcts, los, his, cnts = [], [], [], []
        for n in names:
            e = M["arms"][n]["collision_at_fault"]
            k, N = e["n"], e["N"]
            lo, hi = stats.binomtest(k, N).proportion_ci(0.95)
            pcts.append(100 * k / N); los.append(100 * lo); his.append(100 * hi)
            cnts.append(f"{k}/{N}")
        axes[1].bar(names, pcts, .55, color=cols)
        axes[1].errorbar(np.arange(len(names)), pcts,
                         yerr=[np.array(pcts) - los, np.array(his) - np.array(pcts)],
                         fmt="none", ecolor=INK, capsize=6, lw=1.4)
        for i, (v, c, h) in enumerate(zip(pcts, cnts, his)):
            axes[1].text(i, h + .25, f"{v:.1f}%  ({c})", ha="center", fontsize=9.5)
        axes[1].set_ylim(0, max(his) * 1.35)
        axes[1].set_ylabel("at-fault collision rate (%)")
        axes[1].set_title("safety axis - Wilson 95% CI (overlapping)", fontsize=10)
        fig.tight_layout(); fig.savefig(out / "plots/arms.png", dpi=150); plt.close(fig)

    # --- plot 3: 페어드 델타 (baseline 대비 + dual vs tyr)
    pairs = [(x, y) for x, y in itertools.combinations(runs, 2)]
    if pairs:
        fig, axes = plt.subplots(1, len(pairs), figsize=(4.6 * len(pairs), 3.8), squeeze=False)
        for ax, (x, y) in zip(axes[0], pairs):
            d = np.sort(S[y] - S[x])
            ax.bar(np.arange(len(d)), d, color=[C2 if v >= 0 else ACC for v in d], width=1.0)
            ax.axhline(0, color=MUTED, lw=.8)
            ax.set_title(f"{y} − {x}   평균 {d.mean():+.3f}\n"
                         f"better {int((d > 0).sum())} / worse {int((d < 0).sum())}", fontsize=10)
            ax.set_xlabel("scenes (sorted by delta)")
        axes[0][0].set_ylabel("score delta")
        fig.tight_layout(); fig.savefig(out / "plots/paired_delta.png", dpi=150); plt.close(fig)

    # --- summary
    s1 = "통과" if M["S1"]["pass"] else "벗어남"
    s1_line = f"S1 {s1} (기준 {S1_LO}-{S1_HI}, 예측 {S1_PRIOR}, 실측 {b.mean():.3f})"
    lines = [f"hard100 폐루프 — {len(keys)}씬 x 2 rollout, arm {len(runs)}개",
             f"스위트 일치 {M['suite_match']}/100", s1_line, ""]
    for n in runs:
        e = M["arms"][n]
        ref = f" | 150씬 {e['ref_150']:.3f}" if e["ref_150"] else ""
        lines.append(f"{n:9s} {e['mean']:.3f}  0.7미만 {e['below_0.7']:3d}  "
                     f"이탈 {e['offroad']['pct']:5.1f}%  과실충돌 "
                     f"{e['collision_at_fault']['pct']:4.1f}%{ref}")
    lines.append("")
    for k, p in M["pairs"].items():
        af = p["at_fault"]
        lines.append(f"{p['pair']:24s} Δ{p['delta']:+.3f} ± {p['se']:.3f}  "
                     f"p={p['wilcoxon_p']:.4f}  개선 {p['better']}/악화 {p['worse']}  "
                     f"충돌 OR {af['odds_ratio']:.2f} (p={af['fisher_p']:.3f})")
    if "G1" in M:
        lines += ["", f"G1 {'통과' if M['G1']['pass'] else '실패'}  "
                      f"G2 {'통과' if M['G2']['pass'] else '실패'}"
                      + ("  [G2 검정력 부족]" if M["G2"]["underpowered"] else "")]
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
