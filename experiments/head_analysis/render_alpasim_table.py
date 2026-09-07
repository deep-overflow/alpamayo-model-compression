"""Render collect_alpasim_table's metrics.json as the consolidated results table.

Writes `table.csv` (one row per arm x suite, every metric), `summary.txt` (the two
suites side by side) and two plots.  Kept separate from the collector because the
collector is the expensive half (it parses every rollout's ASL) and the layout of a
results table is the part that gets iterated on.

The four headline metrics are not independent of one another: alpasim's own
`score_criteria` is

    score = 0 if (collision_at_fault or offroad) else min(progress_clipped_rel / 0.8, 1)

so score, at-fault collision, offroad and clipped progress are one quantity and its three
inputs.  Everything in the SECONDARY block is outside the score and is the only place a
difference the score cannot see can show up.

    python experiments/head_analysis/render_alpasim_table.py --out outputs/alpasim_table
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG = "#FAF9F5"
INK = "#29261B"
MUTED = "#6B6555"
C1, C2 = "#2a78d6", "#e87ba4"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})

# family -> arms, in the order the report reads them.  Grouping is by what the arm is
# testing, not by score, so a family that lost is still next to the one it lost to.
FAMILIES = [
    ("무압축", ["baseline"]),
    ("기준 단독 (VLM 24.0%)", ["dual_u40_v2", "jtraj_u40_v2", "traj_u40_v2",
                                "coc_u40_v2", "j_u40_v2"]),
    ("재구성 (24.0%)", ["tyr_u40_r", "dualr_u40", "dualr_wl_u40"]),
    ("expert 포함", ["expert_znorm_r25", "dualexp_u40_em93p75", "dualrwl_em93p75_u40"]),
    ("캘리브레이션 변형", ["dual_st2000", "dual_st2000b"]),
    ("폭/회복", ["dual_u40_w4", "dual_u40_w8", "recover_dual_u55"]),
    ("외부 기준선", ["lp_r50", "lp_r50_dual", "wanda_u40_v2", "G_default_r40"]),
]

HEAD = [("score", "score", "{:.3f}"), ("d_score", "Δ", "{:+.3f}"),
        ("d_p", "p", "{:.3f}"), ("progress_clipped_rel", "progress", "{:.3f}"),
        ("collision_at_fault", "과실충돌%", "{:.1f}"), ("offroad", "이탈%", "{:.1f}")]
SECOND = [("collision_any", "전체충돌%", "{:.1f}"), ("collision_rear", "후미추돌%", "{:.1f}"),
          ("wrong_lane", "차선이탈%", "{:.1f}"),
          ("coc_degenerate_frac", "CoC퇴화%", "{:.1f}"),
          ("dist_to_gt_trajectory", "d2GT(m)", "{:.2f}"),
          ("min_distance_to_obstacle_m", "최소거리(m)", "{:.2f}"),
          ("plan_deviation", "plan편차", "{:.3f}"),
          ("perfect_pct", "만점%", "{:.1f}"), ("zero_pct", "영점%", "{:.1f}"),
          ("repeat_abs_diff", "반복잡음", "{:.3f}")]


def ordered(arms):
    """FAMILIES order, then anything the registry has not been taught about."""
    out, seen = [], set()
    for fam, names in FAMILIES:
        got = [n for n in names if n in arms]
        if got:
            out.append((fam, got))
            seen |= set(got)
    rest = sorted(set(arms) - seen)
    if rest:
        out.append(("미분류", rest))
    return out


def fmt(v, spec):
    return spec.format(v) if isinstance(v, (int, float)) and not np.isnan(v) else "--"


def block(f, suite, data, cols, title):
    arms = data["arms"]
    f.write(f"\n{title}  ({suite}, {data['n_scenes']}씬 x "
            f"{arms['baseline']['n_rollouts'] // data['n_scenes']} rollout)\n")
    hdr = "arm".ljust(24) + "제거%".rjust(7) + "".join(c[1].rjust(11) for c in cols)
    f.write(hdr + "\n" + "-" * len(hdr.encode("utf-8")) * 0 + "-" * 118 + "\n")
    for fam, names in ordered(arms):
        f.write(f"[{fam}]\n")
        for n in names:
            a = arms[n]
            pct = a["params"]["pct"]
            f.write(n[:23].ljust(24) + (f"{pct:6.1f} " if pct is not None else "    -- ")
                    + "".join(fmt(a.get(k), s).rjust(11) for k, _, s in cols) + "\n")


def plots(m, out):
    s150 = m["suites"]["s150"]["arms"]
    hard = m["suites"]["hard100"]["arms"]
    both = sorted(set(s150) & set(hard))

    fig, ax = plt.subplots(1, 2, figsize=(12, 5.6))
    x = [s150[a]["score"] for a in both]
    y = [hard[a]["score"] for a in both]
    ax[0].scatter(x, y, s=70, c=[MUTED if a == "baseline" else C1 for a in both], zorder=3)
    for a, xi, yi in zip(both, x, y):
        ax[0].annotate(a.replace("_u40_v2", "").replace("_u40", ""), (xi, yi),
                       textcoords="offset points", xytext=(6, -3), fontsize=9, color=INK)
    lo, hi = min(x + y) - 0.05, max(x + y) + 0.05
    ax[0].plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, ls="--", zorder=1)
    ax[0].set_xlabel("150-scene score")
    ax[0].set_ylabel("hard100 score")
    ax[0].set_title("arms measured on both suites", fontsize=11)

    # the score cannot see reasoning health.  A scatter of the two puts 19 of 21 arms
    # inside 0-7% and one at 46%, so the labels collide -- sorted bars read cleanly and
    # still carry the point, with each arm's score printed on its bar.
    names = sorted((a for a in s150
                    if not np.isnan(s150[a].get("coc_degenerate_frac", np.nan))),
                   key=lambda a: s150[a]["coc_degenerate_frac"])
    y = np.arange(len(names))
    base = s150["baseline"]["score"]
    ax[1].barh(y, [s150[a]["coc_degenerate_frac"] for a in names],
               color=[MUTED if a == "baseline"
                      else (C1 if s150[a]["score"] >= base else C2) for a in names],
               height=0.72, zorder=3)
    ax[1].set_yticks(y)
    ax[1].set_yticklabels([a.replace("_u40_v2", "").replace("_u40", "") for a in names],
                          fontsize=8)
    for i, a in enumerate(names):
        ax[1].text(s150[a]["coc_degenerate_frac"] + 0.9, i, f"{s150[a]['score']:.3f}",
                   va="center", fontsize=7.5, color=MUTED)
    ax[1].set_xlim(0, 52)
    ax[1].set_xlabel("CoC degeneracy (%)   ·   number on bar = 150-scene score")
    ax[1].set_title("the score cannot see reasoning health", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "plots" / "suites_and_coc.png", dpi=150)
    plt.close(fig)

    from scipy.stats import spearmanr

    rho, p = spearmanr([s150[a]["coc_degenerate_frac"] for a in names],
                       [s150[a]["score"] for a in names])
    (out / "plots" / "coc_score_rho.json").write_text(
        json.dumps({"spearman_rho": float(rho), "p": float(p), "n": len(names)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    m = json.loads((args.out / "metrics.json").read_text())
    (args.out / "plots").mkdir(parents=True, exist_ok=True)

    keys = [k for k, _, _ in HEAD + SECOND]
    with (args.out / "table.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["suite", "arm", "removed_params", "removed_pct", "n_scenes",
                    "n_rollouts", "score_ci_lo", "score_ci_hi", "score_median",
                    "d_lo", "d_hi", "wins", "losses",
                    "collision_at_fault_or", "collision_at_fault_p",
                    "offroad_or", "offroad_p", *keys])
        for suite, d in m["suites"].items():
            for arm, a in sorted(d["arms"].items()):
                w.writerow([suite, arm, a["params"]["removed"], a["params"]["pct"],
                            d["n_scenes"], a["n_rollouts"],
                            *[a.get(k) for k in ("score_ci_lo", "score_ci_hi",
                                                 "score_median", "d_lo", "d_hi",
                                                 "wins", "losses",
                                                 "collision_at_fault_or",
                                                 "collision_at_fault_p",
                                                 "offroad_or", "offroad_p")],
                            *[a.get(k) for k in keys]])
        # the pooled rows carry suite="both"; score lands in the `score` column so a
        # reader filtering on suite gets a directly comparable column across all three
        for arm, a in sorted(m["both"]["arms"].items()):
            row = dict(a, score=a["score_pooled"])
            w.writerow(["both", arm, a["params"]["removed"], a["params"]["pct"],
                        a["n_scenes"], a["n_rollouts"],
                        *[row.get(k) for k in ("score_ci_lo", "score_ci_hi",
                                               "score_median", "d_lo", "d_hi",
                                               "wins", "losses",
                                               "collision_at_fault_or",
                                               "collision_at_fault_p",
                                               "offroad_or", "offroad_p")],
                        *[row.get(k) for k in keys]])

    with (args.out / "summary.txt").open("w") as f:
        f.write("alpasim 폐루프 통합 표 — 두 스위트는 서로소 "
                f"(공유 씬 {m['suite_overlap']}개)\n")
        f.write("score = 0 if (과실충돌 or 이탈) else min(progress_clipped_rel/0.8, 1)\n")
        for suite in ("s150", "hard100"):
            d = m["suites"][suite]
            block(f, suite, d, HEAD, "주요 지표")
            block(f, suite, d, SECOND, "점수 밖 지표")
        # accounting check, in counts rather than rates: a rollout scores 0 exactly when
        # it failed the gate OR never ran at all.  The scene-side `error` rollouts carry
        # no metrics, so they are absent from the gate's own denominator but present in
        # the zero-score count -- comparing rates alone reads that as a discrepancy.
        f.write("\n[검산] 게이트 실패 + 채점불가 = 0점 rollout (개수)\n")
        for suite, d in m["suites"].items():
            n_bad = 0
            for a, v in sorted(d["arms"].items()):
                gate = v.get("offroad_or_collision_at_fault_n")
                if gate is None:
                    continue
                per = v["n_rollouts"] // d["n_scenes"]
                dead = len(v["own_unscorable_scenes"]) * per
                zero = round(v["zero_pct"] / 100 * v["n_rollouts"])
                ok = gate + dead == zero
                n_bad += not ok
                if not ok or dead:
                    f.write(f"  {suite:8s} {a:24s} 게이트 {gate:3d} + 채점불가 {dead:2d} "
                            f"= {gate + dead:3d}  vs 0점 {zero:3d}  "
                            f"{'OK' if ok else '불일치'}\n")
            f.write(f"  {suite}: 불일치 {n_bad}개 / {len(d['arms'])} arm\n")

        b = m["both"]
        f.write(f"\n두 스위트 모두 측정된 arm ({' + '.join(b['suites'])}, "
                f"n={b['n_scenes']}씬)\n")
        hdr = ("arm".ljust(16) + "".join(h.rjust(11) for h in
               ("150씬(순위)", "h100(순위)", "pooled", "macro", "Δ150", "Δh100",
                "Δpooled", "CI저", "CI고", "p", "승", "패")))
        f.write(hdr + "\n" + "-" * 150 + "\n")
        for a, v in sorted(b["arms"].items(), key=lambda x: -x[1]["score_pooled"]):
            r = v["rank_by_suite"]
            f.write(f"{a[:15]:16s}"
                    f"{v['by_suite']['s150']:.3f}({r['s150']})".rjust(11)
                    + f"{v['by_suite']['hard100']:.3f}({r['hard100']})".rjust(11)
                    + f"{v['score_pooled']:11.3f}{v['score_macro']:11.3f}"
                    + ("".join(x.rjust(11) for x in ("--",) * 8) if "d_score" not in v
                       else f"{v['d_by_suite']['s150']:+11.3f}"
                            f"{v['d_by_suite']['hard100']:+11.3f}"
                            f"{v['d_score']:+11.3f}{v['d_lo']:+11.3f}{v['d_hi']:+11.3f}"
                            f"{v['d_p']:11.4f}{v['wins']:11d}{v['losses']:11d}") + "\n")
        f.write("\n  pooled = 250씬 각각 1회씩(150씬이 60% 가중), macro = 두 스위트 평균의 "
                "단순평균(50/50)\n")
        f.write("\n  쌍 비교 (pooled, 씬 단위 페어드)\n")
        for k, v in b["pairs"].items():
            f.write(f"    {k:30s} Δ{v['delta']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]  "
                    f"p={v['p']:.4f}  {v['wins']}승 {v['losses']}패\n")

        # the score's own inputs first, then everything outside it -- same split the
        # report's grouped header makes, so the two artifacts read the same way
        inputs = [("progress_clipped_rel", "progress", "{:.3f}"),
                  ("collision_at_fault", "과실충돌%", "{:.1f}"),
                  ("offroad", "이탈%", "{:.1f}")]
        f.write("\n  지표 전체 (pooled, rollout 기준)   |점수와 그 입력| 점수 밖 ...\n")
        hdr = ("arm".ljust(16) + "score".rjust(11)
               + "".join(c[1].rjust(11) for c in inputs + SECOND))
        f.write("  " + hdr + "\n  " + "-" * 160 + "\n")
        for a, v in sorted(b["arms"].items(), key=lambda x: -x[1]["score_pooled"]):
            f.write("  " + a[:15].ljust(16) + f"{v['score_pooled']:11.3f}"
                    + "".join(fmt(v.get(k), s).rjust(11)
                              for k, _, s in inputs + SECOND) + "\n")

        f.write("\n[맵 결함 씬 제외] 어느 arm에서든 채점 불가였던 씬을 전 arm에서 제거\n")
        for suite, d in m["suites"].items():
            dead = d["unscorable_scenes"]
            f.write(f"  {suite}: {len(dead)}씬 제외 -> n={d['n_scenes_clean']}"
                    f"  {[s[7:15] for s in dead]}\n")
            for a, v in sorted(d["arms"].items(), key=lambda x: -x[1]["score"]):
                if v.get("d_score") is None:
                    f.write(f"    {a:24s} score {v['score']:.3f} -> "
                            f"{v['score_clean']:.3f}\n")
                else:
                    f.write(f"    {a:24s} score {v['score']:.3f} -> "
                            f"{v['score_clean']:.3f}   Δ {v['d_score']:+.3f} -> "
                            f"{v['d_score_clean']:+.3f}  (p {v['d_p']:.3f} -> "
                            f"{v['d_p_clean']:.3f})\n")
    plots(m, args.out)
    print((args.out / "summary.txt").read_text())


if __name__ == "__main__":
    main()
