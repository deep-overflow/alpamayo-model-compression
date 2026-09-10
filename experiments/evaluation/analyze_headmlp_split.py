"""head↔MLP 예산 재배분: 개루프와 폐루프가 반대로 나온다.

`dual_u40_qcut4_v2`(= dual+h4)는 출하 `dual_u40_v2`와 기준·캘리브레이션·expert·KV가 모두
같고 제거 파라미터도 정확히 같은 2,657,452,032(24.0%)다. 바뀌는 것은 그 24%를 어디서
가져오는지 하나뿐 — dual은 층당 Q head 13개 + MLP 4898채널, qcut4는 head 4개 + 5666채널.
`make_slim`의 `_qcut` 경로가 채널 수를 예산에서 유도하고 나누어떨어지지 않으면 assert로
막으므로, 예산 동일성은 구조적으로 보장된다.

두 모드를 한 화면에 올리는 것이 이 스크립트의 전부다:

  개루프  세 고정 세트, rollout-only, minADE@6 / minFDE@6 (평균이 프로토콜의 헤드라인,
          중앙값을 옆에 둔다 -- 델타가 heavy-tail이라 둘이 갈릴 수 있고 실제로 갈린다)
  폐루프  alpasim 150씬 x 2 rollout, per-rollout -> per-scene 평균 -> paired delta.
          씬 절반이 baseline에서 정확히 1.0이라 Wilcoxon은 동률을 버린다: 부트스트랩
          평균 CI가 primary, Wilcoxon은 secondary (CLAUDE.md의 폐루프 관례).

게이트는 Wilson CI로 내고 **씬 단위로 대응 검정까지 한다**. 300 rollout에서 20건 vs 28건은
Wilson 구간이 크게 겹치고, 눈으로 보면 offroad가 6.7%->9.3%로 "올랐다"고 읽고 싶어지지만
대응 검정은 그것을 지지하지 않는다(Wilcoxon p=0.087, McNemar p=0.55, 불일치 씬 4 vs 7).
rollout은 짝지을 수 없다 -- id가 런마다 새로 뽑는 UUID이고 설계가 의도적으로 seed-free다 --
그러나 **씬은 짝지어지고** arm마다 씬당 rollout이 정확히 2개라, 씬별 적중 수가 올바른 단위다.
결론: 종합 점수는 유의하게 움직이는데 게이트는 어느 것도 분리하지 못한다. 기전은 미해결이고,
그것을 풀 도구는 `analyze_longitudinal.py`의 연속 대리지표다.

Usage:
  .venv/bin/python experiments/evaluation/analyze_headmlp_split.py \
      --out /home/cvlab21/project/chan/alpamayo-model-compression/outputs/headmlp_split
"""

import argparse
import collections
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

from coc_action_consistency import K, at6, rows

OUT = Path("/mnt/nvme1n1/ad_vla/outputs/chan")
RUNS = Path("/home/cvlab21/project/chan/alpasim-runs")
BOOT = 20000
GATES = ("collision_at_fault", "offroad")

BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
ORANGE, BLUE, GREEN, RED = "#D97757", "#2a78d6", "#008300", "#b3261e"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})

OPEN_SETS = [("val500", "baseline_ada_ps_indist", "dual_u40_v2_ps_indist",
              "dualqc4_u40_v2_indist", "dualexp_em93p75_indist"),
             ("test500", "baseline_ada_ps_test", "dual_u40_v2_ps_test",
              "dualqc4_u40_v2_test", "dualexp_em93p75_test"),
             ("OOD-val", "baseline_ada_ps_oodval", "dual_u40_v2_ps_ood",
              "dualqc4_u40_v2_oodval", "dualexp_em93p75_oodval")]
ARMS = ("baseline", "dual", "dual+h4", "dual+expMLP93.75")
CL = {"baseline": "m2601_merged_baseline",
      "dual": "m2601_merged_slim_dual_u40_v2",
      "dual+h4": "m2601_merged_slim_dual_u40_qcut4_v2",
      "dual+expMLP93.75": "m2601_merged_slim_dualexp_u40_em93p75"}


def bootci(x, stat=np.mean, seed=0):
    g = np.random.default_rng(seed)
    b = stat(np.asarray(x)[g.integers(0, len(x), (BOOT, len(x)))], axis=1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


# ---------------------------------------------------------------- open loop
def open_loop(res):
    res["open"] = {}
    for name, b_d, d_d, h_d, e_d in OPEN_SETS:
        tab = dict(zip(ARMS, (rows(b_d), rows(d_d), rows(h_d), rows(e_d))))
        tab = {k: v for k, v in tab.items() if v}
        ids = sorted(set.intersection(*[set(v) for v in tab.values()]))
        s = {"n": len(ids), "abs": {}, "vs_dual": {}, "vs_baseline": {}}
        for a, A in tab.items():
            ade = np.array([at6(A[c]) for c in ids])
            fde = np.array([at6(A[c], "fde_rollout_k") for c in ids])
            s["abs"][a] = {
                "ade_mean": float(ade.mean()), "ade_median": float(np.median(ade)),
                "fde_mean": float(fde.mean()), "fde_median": float(np.median(fde)),
                "coc_degen": float(np.mean([A[c]["coc_degenerate"] for c in ids])),
            }
        for ref, key in (("dual", "vs_dual"), ("baseline", "vs_baseline")):
            for a in tab:
                if a == ref:
                    continue
                cell = {}
                for mk, lbl in (("ade_rollout_k", "ade"), ("fde_rollout_k", "fde")):
                    d = np.array([at6(tab[a][c], mk) - at6(tab[ref][c], mk) for c in ids])
                    lo, hi = bootci(d, np.mean)
                    mlo, mhi = bootci(d, np.median)
                    cell[lbl] = {"mean": float(d.mean()), "mean_ci": [lo, hi],
                                 "median": float(np.median(d)), "median_ci": [mlo, mhi],
                                 "p": float(stats.wilcoxon(d).pvalue)}
                s[key][a] = cell
        res["open"][name] = s


# --------------------------------------------------------------- closed loop
def scene_scores(run):
    f = RUNS / run / "aggregate" / "results-summary.json"
    per = collections.defaultdict(list)
    for r in json.loads(f.read_text())["rollouts"]:
        per[r["clipgt_id"]].append(float(r["score"]))
    return {k: float(np.mean(v)) for k, v in per.items()}


def scene_gate_counts(run, gate):
    """scene -> how many of that scene's rollouts tripped `gate` (0, 1 or 2)."""
    d = json.loads((RUNS / run / "aggregate" / "results-summary.json").read_text())["rollouts"]
    c = collections.defaultdict(int)
    for x in d:
        sm = x.get("score_metrics") or {}
        if sm.get(gate) is not None:
            c[x["clipgt_id"]] += float(sm[gate]) > 0
    return c


def gate_counts(run):
    d = json.loads((RUNS / run / "aggregate" / "results-summary.json").read_text())["rollouts"]
    out = {}
    for g in GATES:
        v = [x["score_metrics"].get(g) for x in d if isinstance(x.get("score_metrics"), dict)]
        v = [float(x) for x in v if x is not None]
        out[g] = (int(sum(y > 0 for y in v)), len(v))
    return out


def closed_loop(res):
    sc = {a: scene_scores(r) for a, r in CL.items()}
    scenes = sorted(set.intersection(*[set(v) for v in sc.values()]))
    res["closed"] = {"n_scenes": len(scenes), "abs": {}, "vs_dual": {}, "vs_baseline": {},
                     "gates": {}}
    for a in ARMS:
        v = np.array([sc[a][s] for s in scenes])
        res["closed"]["abs"][a] = {"score": float(v.mean())}
        g = gate_counts(CL[a])
        res["closed"]["gates"][a] = {
            k: {"k": kk, "n": nn, "rate": kk / nn, "ci": wilson(kk, nn)}
            for k, (kk, nn) in g.items()}
    for ref, key in (("dual", "vs_dual"), ("baseline", "vs_baseline")):
        for a in ARMS:
            if a == ref:
                continue
            d = np.array([sc[a][s] - sc[ref][s] for s in scenes])
            lo, hi = bootci(d, np.mean)
            res["closed"][key][a] = {
                "mean": float(d.mean()), "ci": [lo, hi],
                "p": float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else float("nan"),
                "win": int(np.sum(d > 0)), "loss": int(np.sum(d < 0)),
                "tie": int(np.sum(d == 0))}
    res["_scene_delta"] = {a: [sc[a][s] - sc["dual"][s] for s in scenes]
                           for a in ARMS if a != "dual"}

    # Gate rates alone invite an unpaired eyeball comparison, and at 20 vs 28 hits in 300
    # rollouts the Wilson intervals overlap heavily. Rollout ids are per-run UUIDs and the
    # design is deliberately seed-free, so rollouts cannot be paired -- but SCENES can, and
    # each arm has exactly 2 rollouts per scene. Test the per-scene hit count.
    res["closed"]["gate_paired_vs_dual"] = {}
    for a in ARMS:
        if a == "dual":
            continue
        cell = {}
        for g in GATES:
            ca, cb = scene_gate_counts(CL["dual"], g), scene_gate_counts(CL[a], g)
            ss = sorted(set(ca) & set(cb))
            x = np.array([ca[s] for s in ss])
            y = np.array([cb[s] for s in ss])
            d = y - x
            b01 = int(np.sum((x > 0) & (y == 0)))
            b10 = int(np.sum((x == 0) & (y > 0)))
            cell[g] = {
                "dual_hits": int(x.sum()), "arm_hits": int(y.sum()),
                "scenes_changed": int(np.sum(d != 0)),
                "wilcoxon_p": float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else float("nan"),
                "discordant": [b01, b10],
                "mcnemar_p": (float(stats.binomtest(b10, b01 + b10, 0.5).pvalue)
                              if b01 + b10 else float("nan")),
            }
        res["closed"]["gate_paired_vs_dual"][a] = cell


# -------------------------------------------------------------------- plots
def make_plots(res, plots):
    # 1. the dissociation, one panel per mode
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6))
    sets = list(res["open"])
    w, arms = 0.35, ["dual+h4", "dual+expMLP93.75"]
    for j, a in enumerate(arms):
        m = [res["open"][s]["vs_dual"][a]["ade"]["mean"] for s in sets]
        err = np.array([[m[i] - res["open"][s]["vs_dual"][a]["ade"]["mean_ci"][0],
                         res["open"][s]["vs_dual"][a]["ade"]["mean_ci"][1] - m[i]]
                        for i, s in enumerate(sets)]).T
        ax[0].bar(np.arange(len(sets)) + (j - 0.5) * w, m, w, yerr=err,
                  color=[ORANGE, BLUE][j], alpha=0.85, label=a,
                  error_kw={"ecolor": INK, "capsize": 3, "lw": 1})
    ax[0].axhline(0, color=INK, lw=0.9)
    ax[0].set_xticks(range(len(sets)))
    ax[0].set_xticklabels(sets)
    ax[0].set_ylabel(f"mean minADE@{K} delta vs dual (m)")
    ax[0].set_title("open loop — lower is better: no harm detected (CIs span 0)",
                    fontsize=9.5)
    ax[0].legend(frameon=False, fontsize=8)

    names = [a for a in ARMS if a != "dual"]
    m = [res["closed"]["vs_dual"][a]["mean"] for a in names]
    err = np.array([[m[i] - res["closed"]["vs_dual"][a]["ci"][0],
                     res["closed"]["vs_dual"][a]["ci"][1] - m[i]]
                    for i, a in enumerate(names)]).T
    cols = [RED if res["closed"]["vs_dual"][a]["ci"][1] < 0 else MUTED for a in names]
    ax[1].barh(range(len(names)), m, xerr=err, color=cols, alpha=0.85,
               error_kw={"ecolor": INK, "capsize": 3, "lw": 1})
    ax[1].axvline(0, color=INK, lw=0.9)
    ax[1].set_yticks(range(len(names)))
    ax[1].set_yticklabels(names, fontsize=8)
    ax[1].invert_yaxis()
    ax[1].set_xlabel("closed-loop score delta vs dual, 150 scenes")
    ax[1].set_title("closed loop — higher is better: dual+h4 clearly worse", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(plots / "dissociation.png", dpi=150)
    plt.close(fig)

    # 2. gates with Wilson CIs
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.2))
    for i, g in enumerate(GATES):
        v = [100 * res["closed"]["gates"][a][g]["rate"] for a in ARMS]
        err = np.array([[v[j] - 100 * res["closed"]["gates"][a][g]["ci"][0],
                         100 * res["closed"]["gates"][a][g]["ci"][1] - v[j]]
                        for j, a in enumerate(ARMS)]).T
        ax[i].bar(range(len(ARMS)), v, 0.6, yerr=err,
                  color=[MUTED, GREEN, RED, ORANGE], alpha=0.85,
                  error_kw={"ecolor": INK, "capsize": 3, "lw": 1})
        ax[i].set_xticks(range(len(ARMS)))
        ax[i].set_xticklabels([a.replace("dual+expMLP93.75", "d+expMLP") for a in ARMS],
                              fontsize=7.5, rotation=12)
        ax[i].set_ylabel("% of 300 rollouts")
        n = res["closed"]["gates"]["dual"][g]["n"]
        ax[i].set_title(f"{g}  (Wilson 95%, n={n})", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(plots / "gates.png", dpi=150)
    plt.close(fig)

    # 3. per-scene paired delta vs dual -- where the -0.091 lives
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    for a, col in (("dual+h4", RED), ("dual+expMLP93.75", ORANGE), ("baseline", MUTED)):
        d = np.sort(res["_scene_delta"][a])
        ax.plot(d, np.arange(1, len(d) + 1) / len(d) * 100, color=col, lw=1.7, label=a)
    ax.axvline(0, color=INK, lw=0.9)
    ax.set_xlabel("per-scene score delta vs dual")
    ax.set_ylabel("scenes (%)")
    w = res["closed"]["vs_dual"]["dual+h4"]
    ax.set_title(f"closed loop: dual+h4 loses on {w['loss']} scenes, wins {w['win']}, "
                 f"ties {w['tie']}", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "scene_ecdf.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------- longitudinal surrogates
# analyze_longitudinal.py (alpasim venv) re-reads the same rollouts as continuous
# measures. Its output is read here rather than recomputed: the gates are rare events
# with no power, and the surrogates are what decide whether the loss is longitudinal.
LONG_DIR = Path("/home/cvlab21/project/chan/alpamayo-model-compression/outputs/"
                "headmlp_longitudinal_vsdual")
# (key, label, sign that means SAFER). matplotlib has no Korean face on this box, so plot
# labels stay English while the prose around them is Korean -- same as the rest of the repo.
LONG_KEYS = [("brake_frac_when_close", "braking rate (near obstacle)", 1),
             ("brake_accel_when_close", "braking accel (near obstacle)", -1),
             ("mean_speed_when_close", "speed when close", -1),
             ("frac_thw_below_1s", "time with THW < 1 s", -1),
             ("lead_thw_p05", "lead-vehicle THW p05", 1),
             ("lead_ttc_p05", "lead-vehicle TTC p05", 1)]


def longitudinal_block(res, plots):
    f = LONG_DIR / "metrics.json"
    if not f.exists():
        return
    m = json.loads(f.read_text())
    arm = "slim_dual_u40_qcut4_v2"
    d = m["paired_vs_baseline"][arm]
    res["longitudinal"] = {"reference": m.get("reference"), "n_scenes": m["n_scenes"],
                           "arm": arm,
                           **{k: d[k] for k, _, _ in LONG_KEYS}}

    # Each measure has its own unit, so plot the delta normalised by the reference arm's
    # own level -- the point is direction and significance, not magnitude across rows.
    ref_abs = m["configs"][m["reference"]]
    fig, ax = plt.subplots(figsize=(7.8, 3.4))
    ys, labs, cols = [], [], []
    for k, lab, better in LONG_KEYS:
        base = abs(ref_abs[k]["mean"]) or 1.0
        ys.append(100 * d[k]["delta"] / base)
        labs.append(lab)
        sig = d[k]["wilcoxon_p"] is not None and d[k]["wilcoxon_p"] < 0.05
        good = (d[k]["delta"] * better) > 0
        cols.append((GREEN if good else RED) if sig else MUTED)
    y = np.arange(len(ys))
    ax.barh(y, ys, color=cols, alpha=0.85)
    ax.axvline(0, color=INK, lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("dual+h4 - dual, % of dual's own level")
    ax.set_title("longitudinal surrogates: dual+h4 is SAFER than dual\n"
                 "green = significantly safer, grey = not significant", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(plots / "longitudinal.png", dpi=150)
    plt.close(fig)


def write_summary(res, path):
    L = [f"head<->MLP reallocation: open loop vs closed loop  (minADE@{K})", ""]
    for s, v in res["open"].items():
        L.append(f"== open loop {s} (n={v['n']})")
        L.append(f"   {'arm':18s} {'ADE mean(med)':>18s} {'FDE mean(med)':>18s} {'degen':>7s}")
        for a, m in v["abs"].items():
            L.append(f"   {a:18s} {m['ade_mean']:9.4f} ({m['ade_median']:.4f}) "
                     f"{m['fde_mean']:9.4f} ({m['fde_median']:.4f}) "
                     f"{100 * m['coc_degen']:6.1f}%")
        for a, c in v["vs_dual"].items():
            d = c["ade"]
            L.append(f"     {a + ' - dual':28s} ADE mean {d['mean']:+.4f} "
                     f"[{d['mean_ci'][0]:+.4f},{d['mean_ci'][1]:+.4f}]  "
                     f"median {d['median']:+.4f} "
                     f"[{d['median_ci'][0]:+.4f},{d['median_ci'][1]:+.4f}]  p={d['p']:.3g}")
        L.append("")
    c = res["closed"]
    L.append(f"== closed loop, {c['n_scenes']} scenes x 2 rollouts")
    for a, m in c["abs"].items():
        L.append(f"   {a:18s} score {m['score']:.3f}")
    L.append("")
    for a, m in c["vs_dual"].items():
        L.append(f"   {a + ' - dual':28s} {m['mean']:+.3f} [{m['ci'][0]:+.3f},{m['ci'][1]:+.3f}]"
                 f"{'*' if m['ci'][1] < 0 or m['ci'][0] > 0 else ' '} p={m['p']:.3g}  "
                 f"W/L/T {m['win']}/{m['loss']}/{m['tie']}")
    L.append("")
    for g in GATES:
        L.append(f"   {g}  (rate [Wilson 95%])")
        for a in ARMS:
            x = c["gates"][a][g]
            L.append(f"     {a:18s} {x['k']:3d}/{x['n']} = {100 * x['rate']:5.1f}% "
                     f"[{100 * x['ci'][0]:.1f},{100 * x['ci'][1]:.1f}]")
    if "longitudinal" in res:
        lg = res["longitudinal"]
        L.append(f"\n   longitudinal surrogates, paired vs {lg['reference']} "
                 f"({lg['n_scenes']} scenes) -- from analyze_longitudinal.py")
        for k, lab, _ in LONG_KEYS:
            e = lg[k]
            star = "*" if (e["wilcoxon_p"] is not None and e["wilcoxon_p"] < 0.05) else " "
            L.append(f"     {k:24s} {e['delta']:+9.4f} [{e['ci_lo']:+.4f},{e['ci_hi']:+.4f}] "
                     f"p={e['wilcoxon_p']:.4f}{star} n={e['n']}")
        L.append("     -> the loss is NOT longitudinal: dual+h4 brakes more and travels"
                 " slower near obstacles.")

    L.append("\n   gates, PAIRED on the 150 scenes (rollouts are not pairable: per-run UUIDs,"
             " seed-free design)")
    for a, cell in c["gate_paired_vs_dual"].items():
        for g, m in cell.items():
            L.append(f"     {a + ' vs dual':28s} {g:18s} hits {m['dual_hits']:3d} -> "
                     f"{m['arm_hits']:3d}  scenes changed {m['scenes_changed']:3d}  "
                     f"Wilcoxon p={m['wilcoxon_p']:.3g}  "
                     f"McNemar(any) {m['discordant'][0]}/{m['discordant'][1]} "
                     f"p={m['mcnemar_p']:.3g}")
    path.write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    res = {}
    open_loop(res)
    closed_loop(res)
    make_plots(res, plots)
    longitudinal_block(res, plots)
    (out / "config.json").write_text(json.dumps({
        "purpose": "head<->MLP 예산 재배분의 개루프-폐루프 해리",
        "open_sets": [s[0] for s in OPEN_SETS], "closed_runs": CL,
        "k": K, "bootstrap": BOOT, "gates": list(GATES),
    }, ensure_ascii=False, indent=2))
    res.pop("_scene_delta", None)
    (out / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    write_summary(res, out / "summary.txt")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
