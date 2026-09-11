"""캘리브레이션 크기·추출이 폐루프 성능에 미치는 영향 — 150씬 매트릭스 4-arm 분석.

묻는 것: 출하된 `dual_u40_v2` 는 100클립(`calib_100`)으로 고른 것이다. 같은 기준을 20배 많은
2,000클립으로 추정하면 더 좋아지는가?

`analyze_alpasim.py` 는 각 arm 을 baseline 하고만 비교하는데, 여기서 필요한 것은 arm 끼리의
직접 비교(특히 st2000_a vs st2000_b)라서 모든 쌍을 낸다. 예산·배분·규칙은 네 arm 에서 완전히
고정돼 있다 -- 세 압축 arm 모두 정확히 2,657,452,032 파라미터를 지우고 층당 Q 19/32,
MLP 7390/12288 을 남긴다. 바뀌는 것은 within-layer 점수를 어느 클립에서 쟀는가뿐이다.

두 개의 2,000클립 추출은 서로 겹치지 않는다: `calib_st4000` 매니페스트가 `calib_100` 을
명시적으로 제외하고 뽑혔고(`config_calib_st4000.json` 의 `excluded`), 앞 2,000과 뒤 2,000도
당연히 분리돼 있다. 그래서 "크기"와 "추출"이 함께 바뀌며, 그 둘을 가르는 것이 st2000_b 의
존재 이유다:

  st2000_b ≈ st2000_a  ->  -0.112 는 수렴한 선택의 성질이고 calib_100 이 운 좋은 추출
  st2000_b ≈ calib100  ->  n=2,000 에도 추출 분산이 크게 남아 있고 st2000_a 가 불운했던 것

CoC 퇴화율은 `analyze_alpasim.py` 가 alpasim venv 에서 ASL 을 읽어 이미 계산해 둔 값을
가져온다(이 스크립트는 리포의 .venv 로 돈다). 점수와 게이트는 병합 런의
`aggregate/results-summary.json` 에서 직접 읽는다.

Usage:
  .venv/bin/python experiments/evaluation/analyze_calibsize.py \
      --coc-from outputs/st2000b_eval/metrics.json --out outputs/calibsize_eval
"""

import argparse
import collections
import itertools
import json
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy import stats

REPO = Path(__file__).resolve().parents[2]

# arm 이름 -> (병합 런 디렉터리, slim 체크포인트 디렉터리 또는 "-", 캘리브레이션 설명)
# 기본값은 dual 캘리브레이션 크기 연구(2026-09-06). --arm 으로 얼마든지 갈아끼울 수 있게 한 것은
# tyr 계열에 같은 질문을 다시 물으면서다 -- 같은 표·같은 검정을 쓰는데 arm 목록만 다른 스크립트를
# 하나 더 만들 이유가 없다. `-` 는 체크포인트 없음(baseline)을 뜻한다.
DEFAULT_ARMS = [
    ("baseline", "m2601_merged_baseline", "-", "무압축"),
    ("calib100", "m2601_merged_slim_dual_u40_v2", "slim_dual_u40_v2", "calib_100 (100클립)"),
    ("st2000_a", "m2601_merged_slim_dual_st2000", "slim_dual_st2000",
     "calib_st4000 앞 2,000클립"),
    ("st2000_b", "m2601_merged_slim_dual_st2000b", "slim_dual_st2000b",
     "calib_st4000 뒤 2,000클립"),
]
GATES = ["collision_at_fault", "offroad", "wrong_lane"]

BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
ARM_COLOR = {"baseline": "#6B6555", "calib100": "#D97757",
             "st2000_a": "#2a78d6", "st2000_b": "#008300"}
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load_run(run_dir):
    """scene -> 그 씬의 rollout 레코드 목록."""
    f = Path(run_dir) / "aggregate" / "results-summary.json"
    if not f.exists():
        return None
    per = collections.defaultdict(list)
    for r in json.loads(f.read_text())["rollouts"]:
        per[r["clipgt_id"]].append(r)
    return per


def scene_scores(per, scenes):
    """씬 점수 = 그 씬 rollout 들의 평균. 순서는 scenes 로 고정한다."""
    return np.array([float(np.mean([r["score"] for r in per[s]])) for s in scenes])


def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    b = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(n)])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def paired(a, b):
    """b - a 의 페어드 통계. 동률이 많아 Wilcoxon 은 0 을 제외하고 돌린다."""
    d = b - a
    nz = d[d != 0]
    lo, hi = boot_ci(d)
    return {"delta": float(d.mean()), "ci_lo": lo, "ci_hi": hi,
            "median": float(np.median(d)),
            "wilcoxon_p": float(stats.wilcoxon(nz).pvalue) if len(nz) else 1.0,
            "better": int((d > 0).sum()), "worse": int((d < 0).sum()),
            "tie": int((d == 0).sum())}


def gate_count(per, scenes, key):
    rr = [r for s in scenes for r in per[s]]
    return sum(1 for r in rr if r["metrics"].get(key, 0) > 0), len(rr)


def kept_masks(slim_dir):
    """slim_meta.json 의 남긴 인덱스를 0/1 마스크로."""
    m = json.loads((REPO / "outputs" / slim_dir / "slim_meta.json").read_text())
    nl = len(m["vlm"])
    q = np.zeros((nl, 32), dtype=bool)
    mlp = np.zeros((nl, 12288), dtype=bool)
    for i, layer in enumerate(m["vlm"]):
        q[i, layer["q"]] = True
        mlp[i, layer["mlp"]] = True
    return q, mlp


def plot_scores(stats_, out):
    names = list(stats_)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, n in enumerate(names):
        s = stats_[n]
        ax.bar(i, s["score"], color=ARM_COLOR.get(n, MUTED), width=0.62)
        ax.errorbar(i, s["score"], yerr=[[s["score"] - s["ci_lo"]], [s["ci_hi"] - s["score"]]],
                    fmt="none", ecolor=INK, capsize=4, lw=1.2)
        ax.text(i, s["ci_hi"] + 0.012, f"{s['score']:.3f}", ha="center", fontsize=9)
    ax.axhline(stats_["baseline"]["score"], color=MUTED, ls=":", lw=1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("scene score")
    ax.set_ylim(0, max(s["ci_hi"] for s in stats_.values()) + 0.06)
    ax.set_title("closed-loop scene score, 150 scenes x 2 rollouts (95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(out / "scene_score.png", dpi=150)
    plt.close(fig)


def plot_pairs(pairs, out):
    keys = list(pairs)
    fig, ax = plt.subplots(figsize=(7.0, 0.52 * len(keys) + 1.6))
    for i, k in enumerate(keys):
        p = pairs[k]
        y = len(keys) - 1 - i
        col = "#008300" if p["ci_lo"] > 0 else ("#c0392b" if p["ci_hi"] < 0 else MUTED)
        ax.plot([p["ci_lo"], p["ci_hi"]], [y, y], color=col, lw=2.4)
        ax.plot(p["delta"], y, "o", color=col, ms=6)
        ax.text(p["ci_hi"] + 0.006, y, f"p={p['wilcoxon_p']:.2g}", va="center", fontsize=8,
                color=MUTED)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([k for k in reversed(keys)], fontsize=9)
    # p 라벨을 CI 오른쪽 끝에 찍으므로 그만큼 오른쪽 여백을 비워 둔다 (안 그러면 잘린다)
    span = max(p["ci_hi"] for p in pairs.values()) - min(p["ci_lo"] for p in pairs.values())
    ax.set_xlim(min(p["ci_lo"] for p in pairs.values()) - 0.04 * span,
                max(p["ci_hi"] for p in pairs.values()) + 0.34 * span)
    ax.set_xlabel("paired scene-score difference (95% bootstrap CI)")
    ax.set_title("all pairwise comparisons")
    fig.tight_layout()
    fig.savefig(out / "pairwise_deltas.png", dpi=150)
    plt.close(fig)


def plot_gates(stats_, out):
    names = list(stats_)
    fig, axes = plt.subplots(1, len(GATES), figsize=(10.5, 3.2))
    for ax, g in zip(axes, GATES):
        for i, n in enumerate(names):
            c, tot = stats_[n]["gates"][g]
            lo, hi = stats.binomtest(c, tot).proportion_ci(0.95, method="wilson")
            ax.bar(i, c / tot * 100, color=ARM_COLOR.get(n, MUTED), width=0.62)
            ax.errorbar(i, c / tot * 100, yerr=[[c / tot * 100 - lo * 100], [hi * 100 - c / tot * 100]],
                        fmt="none", ecolor=INK, capsize=3, lw=1)
            ax.text(i, hi * 100 + 0.6, f"{c}", ha="center", fontsize=8, color=MUTED)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_title(g, fontsize=10)
        ax.set_ylabel("% of rollouts" if g == GATES[0] else "")
    fig.suptitle("gate rates with Wilson 95% CI (event counts above bars)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "incident_rates.png", dpi=150)
    plt.close(fig)


def plot_overlap(ov, names, out):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for ax, axis in zip(axes, ("q", "mlp")):
        n = len(names)
        M = np.full((n, n), np.nan)
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                M[i, j] = 100.0 if a == b else ov[f"{a}|{b}"][axis] * 100
        im = ax.imshow(M, cmap="YlOrRd", vmin=80, vmax=100)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=9,
                        color=INK if M[i, j] < 96 else "white")
        ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=8)
        ax.set_title("Q head" if axis == "q" else "MLP channel", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("kept-unit overlap between calibrations (%)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "selection_overlap.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path,
                    default=Path("/home/cvlab21/project/chan/alpasim-runs"))
    ap.add_argument("--coc-from", type=Path, default=None,
                    help="analyze_alpasim.py 가 쓴 metrics.json (CoC 퇴화율을 여기서 읽는다)")
    ap.add_argument("--out", type=Path, default=REPO / "outputs/calibsize_eval")
    ap.add_argument("--arm", nargs=4, action="append", metavar=("NAME", "RUNDIR", "SLIMDIR", "LABEL"),
                    help="arm 하나 (반복 가능). SLIMDIR 이 '-' 면 체크포인트 없음(baseline). "
                         "생략하면 dual 캘리브레이션 크기 연구의 4 arm")
    args = ap.parse_args()

    out = args.out if args.out.is_absolute() else REPO / args.out
    (out / "plots").mkdir(parents=True, exist_ok=True)

    arms = [tuple(a) for a in args.arm] if args.arm else DEFAULT_ARMS
    runs, meta = {}, {}
    for name, run_dir, slim, desc in arms:
        slim = None if slim == "-" else slim
        per = load_run(args.runs_root / run_dir)
        if per is None:
            print(f"{name}: {run_dir} 에 aggregate 가 없습니다 -- 건너뜁니다", flush=True)
            continue
        runs[name] = per
        meta[name] = {"run": run_dir, "slim": slim, "calib": desc}

    # 모든 arm 이 본 씬으로 고정한다. 하나라도 빠진 씬이 있으면 페어드 비교가 성립하지 않는다.
    scenes = sorted(set.intersection(*(set(p) for p in runs.values())))
    for name, per in runs.items():
        missing = len(set(per)) - len(scenes)
        if missing:
            print(f"{name}: 공통 씬 밖 {missing}개 제외", flush=True)
    print(f"공통 씬 {len(scenes)}개, arm {len(runs)}개", flush=True)

    S = {n: scene_scores(p, scenes) for n, p in runs.items()}
    coc = {}
    if args.coc_from and Path(args.coc_from).exists():
        m = json.loads(Path(args.coc_from).read_text())
        # analyze_alpasim 의 키는 config 이름(slim_dual_st2000 ...)이라 arm 이름으로 되돌린다.
        # 병합 런의 접두사는 매트릭스마다 다르므로(m2601_merged_ / h100_merged_ ...) 첫
        # "..._merged_" 까지를 통째로 벗긴다 -- 하나만 하드코딩하면 다른 매트릭스에서 조용히 빈다.
        by_run = {re.sub(r"^.*?_merged_", "", v["run"]): k for k, v in meta.items()}
        for cfg, v in m.get("coc", {}).items():
            if cfg in by_run:
                coc[by_run[cfg]] = v

    stats_ = {}
    for n, s in S.items():
        lo, hi = boot_ci(s)
        g = {k: gate_count(runs[n], scenes, k) for k in GATES}
        stats_[n] = {"score": float(s.mean()), "ci_lo": lo, "ci_hi": hi,
                     "median": float(np.median(s)),
                     "pass_frac": float(np.mean([r["passed"] for sc_ in scenes
                                                 for r in runs[n][sc_]])),
                     "n_rollouts": sum(len(runs[n][sc_]) for sc_ in scenes),
                     "gates": g,
                     "coc_degenerate": coc.get(n, {}).get("mean_degenerate_frac"),
                     **meta[n]}

    names = list(S)
    pairs = {f"{b} - {a}": paired(S[a], S[b]) for a, b in itertools.combinations(names, 2)}
    # 게이트도 arm 쌍마다 Fisher
    gate_pairs = {}
    for a, b in itertools.combinations(names, 2):
        for g in GATES:
            ca, na = stats_[a]["gates"][g]
            cb, nb = stats_[b]["gates"][g]
            gate_pairs[f"{b} - {a}|{g}"] = float(
                stats.fisher_exact([[cb, nb - cb], [ca, na - ca]]).pvalue)

    # 선택 집합 겹침
    masks = {n: kept_masks(stats_[n]["slim"]) for n in names if stats_[n]["slim"]}
    ov = {}
    for a, b in itertools.permutations(masks, 2):
        qa, ma = masks[a]; qb, mb = masks[b]
        ov[f"{a}|{b}"] = {"q": float((qa & qb).sum() / qa.sum()),
                          "mlp": float((ma & mb).sum() / ma.sum())}

    plot_scores(stats_, out / "plots")
    plot_pairs(pairs, out / "plots")
    plot_gates(stats_, out / "plots")
    if len(masks) > 1:
        plot_overlap(ov, list(masks), out / "plots")

    lines = [f"arm 간 폐루프 비교 — {len(scenes)}씬, arm {len(names)}개", ""]
    lines.append(f"{'arm':11s} {'캘리브레이션':22s} {'score':>6s} {'95% CI':>17s} "
                 f"{'중앙값':>7s} {'CoC퇴화':>8s}")
    lines.append("-" * 80)
    for n in names:
        s = stats_[n]
        c = f"{s['coc_degenerate']*100:.2f}%" if s["coc_degenerate"] is not None else "-"
        lines.append(f"{n:11s} {s['calib']:22s} {s['score']:6.3f} "
                     f"[{s['ci_lo']:.3f}, {s['ci_hi']:.3f}] {s['median']:7.3f} {c:>8s}")
    lines += ["", "페어드 비교 (씬 단위, 95% bootstrap CI)"]
    for k, p in pairs.items():
        star = "*" if (p["ci_lo"] > 0) or (p["ci_hi"] < 0) else " "
        lines.append(f"  {k:24s} {p['delta']:+.4f} [{p['ci_lo']:+.4f}, {p['ci_hi']:+.4f}]{star} "
                     f"p={p['wilcoxon_p']:.2g}  W/L/T {p['better']}/{p['worse']}/{p['tie']}")
    lines += ["", "게이트 (rollout 단위, Wilson 95% CI)"]
    for g in GATES:
        lines.append(f"  -- {g}")
        for n in names:
            c, t = stats_[n]["gates"][g]
            lo, hi = stats.binomtest(c, t).proportion_ci(0.95, method="wilson")
            lines.append(f"     {n:11s} {c/t*100:5.1f}% ({c}/{t})  [{lo*100:.1f}, {hi*100:.1f}]")
    if ov:
        lines += ["", "선택 집합 겹침 (켜진 유닛 기준, %)"]
        for a, b in itertools.combinations(masks, 2):
            v = ov[f"{a}|{b}"]
            lines.append(f"  {a:11s} vs {b:11s}  Q {v['q']*100:5.1f}%   MLP {v['mlp']*100:5.1f}%")

    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    (out / "metrics.json").write_text(json.dumps({
        "n_scenes": len(scenes), "arms": stats_, "pairs": pairs,
        "gate_fisher_p": gate_pairs, "selection_overlap": ov,
        "per_scene_score": {n: S[n].tolist() for n in names}, "scenes": scenes,
    }, indent=2, ensure_ascii=False))
    print("\n".join(lines))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
