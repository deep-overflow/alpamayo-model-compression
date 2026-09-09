"""dual 기준에 신호를 더하는 두 경로 — 안전 가중(`dualsafe`)과 CoC-행동 일관성.

두 실험은 같은 질문의 두 형태다: `max(rank I_traj, rank I_CoC)` 에 무엇을 더 넣으면
나아지는가? 하나는 클립 축(위험한 클립에 가중), 하나는 목적 축(세 번째 항).

A. `dualsafe` — `collision_lib.score_path` 의 `min_center_dist` (GT 경로의 연속 여유거리)
   로 클립 가중 `w = exp(-d/tau)`, tau=5 m. 위험한 클립에서 잰 중요도를 더 믿는다는 뜻.
   예산·배분·expert·KV 는 `dual_u40_v2` 와 완전히 동일하고 within-layer 점수의 클립
   가중만 다르다.

B. CoC-행동 일관성 — Alpamayo-R1 (arXiv 2511.00088, S5.3) 의 `r_consistency` 를 저장된
   rollout 에서 재현. 릴리스 체크포인트에 meta-action 토큰이 없으므로(base_model 의
   SPECIAL_TOKENS_KEYS 에 키 자체가 없고 그 자리는 `_padding_*`) 토큰 NLL 경로는 막혀
   있지만, CoC 문장의 머리 어절이 곧 meta-action 이라 텍스트로 파싱된다. 자세한 규칙과
   한계는 `coc_action_consistency.py` docstring 참조 -- 이 스크립트는 그 로직을 import
   해서 보고서용 수치와 그림만 만든다.

사전등록한 판정선이 있는 실험이 아니라, `dualsafe` 는 기각 기록이고 B 는 "세 번째 항을
지을 가치가 있는가"의 사전 조사다. 그래서 게이트 대신 두 개의 반증 가능한 물음을 둔다:

  Q1  dualsafe 가 dual 과 구분되지 않으면(CI 가 0 포함) 가중은 무해무익이다.
  Q2  CoC 항이 없는 arm 만 일관성을 잃는다면, `I_CoC` 가 이미 그 신호를 사고 있는
      것이므로 세 번째 항의 여지는 그 잔차뿐이다.

Usage:
  .venv/bin/python experiments/evaluation/analyze_criterion_aug.py \
      --out /home/cvlab21/project/chan/alpamayo-model-compression/outputs/criterion_aug
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

from coc_action_consistency import HEAD2BUCKET, at6, head, rows

OUT = Path("/mnt/nvme1n1/ad_vla/outputs/chan")
K, BOOT = 6, 10000
SAFE_TAU = 5.0
U40_RATIO = 0.3985632694  # run_grid.allocations() matched to slim_integrated_mag

BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
ORANGE, BLUE, GREEN, RED = "#D97757", "#2a78d6", "#008300", "#b3261e"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})

# (디렉터리, 표시 이름) -- 첫 행이 기준. 일관성/결합성 표에 함께 쓴다.
CONS_ARMS = [
    ("baseline_ada_ps_indist", "baseline"),
    ("tyr_u40_r_indist", "tyr_u40_r"),
    ("cachedual_u40_v2_indist", "cachedual"),
    ("dual_u40_v2_ps_indist", "dual"),
    ("dualfix_u40_v2_indist", "dualfix"),
    ("jtraj_u40_v2_ps_indist", "j_traj"),
    ("maxstep11_u40_v2_indist", "maxstep11"),
    ("coc_u40_v2_indist", "coc"),
    ("znorm11_u40_v2_indist", "znorm11"),
    ("traj_u40_v2_indist", "traj"),
    ("cacheonly_u40_v2_indist", "cacheonly"),
]


def bootmed(x, seed=0):
    g = np.random.default_rng(seed)
    idx = g.integers(0, len(x), (BOOT, len(x)))
    b = np.median(np.asarray(x)[idx], axis=1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


# --------------------------------------------------------------------------
# A. dualsafe
# --------------------------------------------------------------------------
def clearance_weights():
    """캘리브레이션 클립의 여유거리와 그것이 만드는 가중 -- 유효 표본수까지."""
    clr = json.loads((OUT / "calib_clearance" / "clearance.json").read_text())
    meta = json.loads((OUT / "importance_v2" / "metrics.json").read_text())["per_clip"]
    raw = [clr.get(r["clip_id"]) for r in meta]
    d = np.array([np.nan if v is None else float(v) for v in raw])
    n_lab = int(np.sum(~np.isnan(d)))
    d = np.where(np.isnan(d), np.nanmedian(d), d)  # 미라벨은 중앙값 -- 버리지 않는다
    w = np.exp(-d / SAFE_TAU)
    w = w / w.sum()
    # 유효 표본수(Kish): 가중이 얼마나 소수 클립에 몰렸는가
    ess = float(1.0 / np.sum(w**2))
    return d, w, n_lab, ess


def overlap(a_meta, b_meta):
    out = {}
    for axis in ("q", "mlp"):
        ov = [len(set(a_meta["vlm"][i][axis]) & set(b_meta["vlm"][i][axis]))
              / len(a_meta["vlm"][i][axis]) for i in range(len(a_meta["vlm"]))]
        out[axis] = (float(np.mean(ov)), float(np.min(ov)))
    return out


def dualsafe_block(res, plots):
    d, w, n_lab, ess = clearance_weights()
    res["clearance"] = {
        "n_labelled": n_lab, "tau_m": SAFE_TAU, "ess": ess,
        "min": float(d.min()), "p10": float(np.percentile(d, 10)),
        "median": float(np.median(d)), "p90": float(np.percentile(d, 90)),
        "max": float(d.max()),
        "weight_top10_share": float(np.sort(w)[-10:].sum()),
    }

    safe, dual, base = (rows("dualsafe_u40_v2_indist"), rows("dual_u40_v2_ps_indist"),
                        rows("baseline_ada_ps_indist"))
    ids = sorted(set(safe) & set(dual) & set(base))
    arms = {"baseline": base, "dual": dual, "dualsafe": safe}
    res["val500"] = {"n": len(ids), "arms": {}}
    for nm, A in arms.items():
        v = np.array([at6(A[c]) for c in ids])
        f = np.array([at6(A[c], "fde_rollout_k") for c in ids])
        res["val500"]["arms"][nm] = {
            "minADE6_mean": float(v.mean()), "minADE6_median": float(np.median(v)),
            "minFDE6_mean": float(f.mean()), "minFDE6_median": float(np.median(f)),
            "coc_degen": float(np.mean([A[c]["coc_degenerate"] for c in ids])),
        }
    res["val500"]["pairs"] = {}
    for x, y, nm in (("dual", "baseline", "dual - baseline"),
                     ("dualsafe", "baseline", "dualsafe - baseline"),
                     ("dualsafe", "dual", "dualsafe - dual")):
        dd = np.array([at6(arms[x][c]) - at6(arms[y][c]) for c in ids])
        lo, hi = bootmed(dd)
        res["val500"]["pairs"][nm] = {
            "median": float(np.median(dd)), "mean": float(dd.mean()),
            "ci": [lo, hi], "p": float(stats.wilcoxon(dd).pvalue),
            "worse_frac": float(np.mean(dd > 0)),
        }
    dd = np.array([at6(safe[c]) - at6(dual[c]) for c in ids])
    res["val500"]["safe_vs_dual_identical_coc"] = float(
        np.mean([safe[c]["gen_coc"] == dual[c]["gen_coc"] for c in ids]))
    res["val500"]["safe_vs_dual_unchanged_ade"] = float(np.mean(dd == 0))

    # 체크포인트는 cvlab20 에서 지었지만 slim_meta.json 은 여기로 가져와 두었다 --
    # config 의 model 경로는 그쪽 상대경로라 이름만 떼어 outputs 아래에서 찾는다
    sm = json.loads((OUT / "dualsafe_u40_v2_indist" / "config.json").read_text())
    res["kept_overlap"] = None
    meta_p = OUT / Path(sm.get("model", "")).name / "slim_meta.json"
    if meta_p.exists():
        res["kept_overlap"] = overlap(
            json.loads(meta_p.read_text()),
            json.loads((OUT / "slim_dual_u40_v2" / "slim_meta.json").read_text()))

    # --- plot 1: 여유거리와 가중 ---
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.3))
    ax[0].hist(np.clip(d, 0, 40), bins=40, color=MUTED, alpha=0.75)
    ax[0].axvline(np.median(d), color=ORANGE, lw=1.6,
                  label=f"median {np.median(d):.2f} m")
    ax[0].set_xlabel("GT-path clearance (m, clipped at 40)")
    ax[0].set_ylabel("calibration clips")
    ax[0].set_title("min_center_dist over calib_100", fontsize=10)
    ax[0].legend(frameon=False, fontsize=8)

    o = np.argsort(-w)
    ax[1].plot(np.arange(1, len(w) + 1), np.cumsum(w[o]) * 100, color=BLUE, lw=1.8)
    ax[1].plot([0, len(w)], [0, 100], color=MUTED, ls=":", lw=1,
               label="uniform (dual)")
    ax[1].axhline(50, color=MUTED, ls="--", lw=0.8)
    ax[1].set_xlabel("clips, heaviest first")
    ax[1].set_ylabel("cumulative weight (%)")
    ax[1].set_title(f"exp(-d/{SAFE_TAU:.0f}) weighting: ESS {ess:.1f} of {len(w)}",
                    fontsize=10)
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "clearance.png", dpi=150)
    plt.close(fig)

    # --- plot 2: val500 페어드 ---
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.3))
    for nm, col in (("dual", BLUE), ("dualsafe", ORANGE)):
        dd = np.sort([at6(arms[nm][c]) - at6(base[c]) for c in ids])
        ax[0].plot(dd, np.arange(1, len(dd) + 1) / len(dd) * 100, color=col, lw=1.7,
                   label=f"{nm} - baseline")
    ax[0].axvline(0, color=MUTED, lw=0.8)
    ax[0].set_xlim(-0.6, 1.6)
    ax[0].set_xlabel("paired minADE@6 delta (m)")
    ax[0].set_ylabel("clips (%)")
    ax[0].set_title("val500 paired deltas vs baseline", fontsize=10)
    ax[0].legend(frameon=False, fontsize=8)

    names = ["dual - baseline", "dualsafe - baseline", "dualsafe - dual"]
    med = [res["val500"]["pairs"][n]["median"] for n in names]
    err = np.array([[m - res["val500"]["pairs"][n]["ci"][0],
                     res["val500"]["pairs"][n]["ci"][1] - m]
                    for n, m in zip(names, med)]).T
    cols = [BLUE, ORANGE, RED]
    ax[1].barh(range(3), med, xerr=err, color=cols, alpha=0.85,
               error_kw={"ecolor": INK, "capsize": 3, "lw": 1})
    ax[1].axvline(0, color=MUTED, lw=0.8)
    ax[1].set_yticks(range(3))
    ax[1].set_yticklabels(names, fontsize=8)
    ax[1].invert_yaxis()
    ax[1].set_xlabel("median paired delta (m), 95% bootstrap CI")
    ax[1].set_title("weighting costs more than the pruning", fontsize=10)
    fig.tight_layout()
    fig.savefig(plots / "dualsafe.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# B. CoC-행동 일관성
# --------------------------------------------------------------------------
def consistency_block(res, plots):
    tab = {n: rows(d) for d, n in CONS_ARMS}
    tab = {n: v for n, v in tab.items() if v}
    names = [n for _, n in CONS_ARMS if n in tab]
    ids = sorted(set.intersection(*[set(tab[n]) for n in names]))
    base = names[0]

    hit, cov = {}, {}
    for n in names:
        b = [HEAD2BUCKET.get(head(tab[n][c]["gen_coc"])) for c in ids]
        cov[n] = np.array([x is not None for x in b])
        hit[n] = np.array([x == tab[n][c]["bucket"] for x, c in zip(b, ids)], float)

    res["consistency"] = {"n_clips": len(ids), "arms": {}, "vs_baseline": {}}
    for n in names:
        res["consistency"]["arms"][n] = {
            "coverage": float(cov[n].mean()), "agree_all": float(hit[n].mean()),
            "agree_mapped": float(hit[n][cov[n]].mean()),
            "coc_degen": float(np.mean([tab[n][c]["coc_degenerate"] for c in ids])),
            "minADE6_median": float(np.median([at6(tab[n][c]) for c in ids])),
        }
    for n in names[1:]:
        x, y = hit[base], hit[n]
        b01 = int(np.sum((x == 1) & (y == 0)))
        b10 = int(np.sum((x == 0) & (y == 1)))
        p = stats.binomtest(b10, b01 + b10, 0.5).pvalue if b01 + b10 else float("nan")
        res["consistency"]["vs_baseline"][n] = {
            "delta_pp": float(100 * (y.mean() - x.mean())), "p": float(p),
            "p_bonf": float(min(1.0, p * (len(names) - 1))),
            "discordant": [b01, b10],
            "identical_coc": float(np.mean(
                [tab[base][c]["gen_coc"] == tab[n][c]["gen_coc"] for c in ids])),
        }
    # traj 를 기준으로 한 직접 대응비교 -- CoC 항의 기여
    if "traj" in tab:
        res["consistency"]["vs_traj"] = {}
        x = hit["traj"]
        for n in ("dual", "coc", "j_traj", base):
            if n not in tab:
                continue
            y = hit[n]
            b01 = int(np.sum((x == 1) & (y == 0)))
            b10 = int(np.sum((x == 0) & (y == 1)))
            p = stats.binomtest(b10, b01 + b10, 0.5).pvalue if b01 + b10 else float("nan")
            res["consistency"]["vs_traj"][n] = {
                "delta_pp": float(100 * (y.mean() - x.mean())), "p": float(p)}

    # 결합성: CoC 가 바뀐 클립에서 궤적이 더 상하는가
    res["coupling"] = {}
    for n in names[1:]:
        chg = np.array([tab[base][c]["gen_coc"] != tab[n][c]["gen_coc"] for c in ids])
        dd = np.array([at6(tab[n][c]) - at6(tab[base][c]) for c in ids])
        bb = np.array([at6(tab[base][c]) for c in ids])
        p = (stats.mannwhitneyu(dd[chg], dd[~chg]).pvalue
             if chg.any() and (~chg).any() else float("nan"))
        res["coupling"][n] = {
            "n_changed": int(chg.sum()), "med_changed": float(np.median(dd[chg])),
            "med_same": float(np.median(dd[~chg])), "p": float(p),
            "base_med_changed": float(np.median(bb[chg])),
            "base_med_same": float(np.median(bb[~chg])),
        }

    # --- plot 3: 일관성 ---
    ordr = sorted(names[1:], key=lambda n: res["consistency"]["vs_baseline"][n]["delta_pp"])
    dl = [res["consistency"]["vs_baseline"][n]["delta_pp"] for n in ordr]
    sig = [res["consistency"]["vs_baseline"][n]["p_bonf"] < 0.05 for n in ordr]
    fig, ax = plt.subplots(1, 2, figsize=(10.2, 4.2))
    ax[0].barh(range(len(ordr)), dl,
               color=[RED if s else MUTED for s in sig], alpha=0.85)
    ax[0].axvline(0, color=INK, lw=0.9)
    ax[0].set_yticks(range(len(ordr)))
    ax[0].set_yticklabels(ordr, fontsize=8)
    ax[0].set_xlabel("consistency delta vs baseline (pp)")
    ax[0].set_title(f"red = survives Bonferroni({len(names) - 1})", fontsize=9.5)

    # 라벨이 겹치는 밀집 구간(dual/dualfix/j_traj)은 오프셋을 번갈아 준다
    off = {"dual": (5, 5), "j_traj": (-33, -3), "dualfix": (12, -3),
           "maxstep11": (2, 6), "znorm11": (5, -12)}
    for n in names:
        a = res["consistency"]["arms"][n]
        c = ORANGE if n == base else (RED if n in ("traj", "cacheonly") else BLUE)
        ax[1].scatter(a["coc_degen"] * 100, a["agree_all"] * 100, color=c, s=34, zorder=3)
        ax[1].annotate(n, (a["coc_degen"] * 100, a["agree_all"] * 100),
                       textcoords="offset points", xytext=off.get(n, (5, 3)),
                       fontsize=7.5, color=MUTED)
    ax[1].set_xscale("symlog", linthresh=1)
    ax[1].set_xlabel("CoC degeneracy (%, symlog)")
    ax[1].set_ylabel("consistency (%)")
    ax[1].set_title("consistency is not degeneracy", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(plots / "consistency.png", dpi=150)
    plt.close(fig)

    # --- plot 4: 결합성 ---
    ordr = [n for n in names[1:] if n in res["coupling"]]
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    y = np.arange(len(ordr))
    ax.barh(y - 0.2, [res["coupling"][n]["med_changed"] for n in ordr], height=0.38,
            color=ORANGE, alpha=0.9, label="CoC changed vs baseline")
    ax.barh(y + 0.2, [res["coupling"][n]["med_same"] for n in ordr], height=0.38,
            color=BLUE, alpha=0.9, label="CoC identical")
    for i, n in enumerate(ordr):
        m = res["coupling"][n]
        if m["p"] < 0.05:
            # znorm11 은 유의하지만 부호가 반대다 -- 그림만 보고 오독하지 않게 표시한다
            rev = " (reversed)" if m["med_changed"] < m["med_same"] else ""
            ax.text(max(m["med_changed"], m["med_same"]) + 0.01, i,
                    f"p={m['p']:.3f}{rev}", va="center", fontsize=7.5,
                    color=RED if not rev else MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels(ordr, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("median paired minADE@6 delta vs baseline (m)")
    ax.set_title("trajectory damage does not follow CoC drift, "
                 "except where the CoC collapses", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "coupling.png", dpi=150)
    plt.close(fig)


def write_summary(res, path):
    L = []
    a = res["clearance"]
    L.append("A. dualsafe -- clearance-weighted dual importance")
    L.append(f"   clearance over calib_100 ({a['n_labelled']}/100 labelled): "
             f"min {a['min']:.2f}  p10 {a['p10']:.2f}  med {a['median']:.2f}  "
             f"p90 {a['p90']:.2f}  max {a['max']:.2f} m")
    L.append(f"   w = exp(-d/{a['tau_m']:.0f}): effective sample size {a['ess']:.1f} "
             f"of 100, top-10 clips hold {100 * a['weight_top10_share']:.1f}% of weight")
    if res.get("kept_overlap"):
        o = res["kept_overlap"]
        L.append(f"   kept-set overlap vs dual: Q {100 * o['q'][0]:.1f}% "
                 f"(min {100 * o['q'][1]:.1f}%)  MLP {100 * o['mlp'][0]:.1f}% "
                 f"(min {100 * o['mlp'][1]:.1f}%)")
    v = res["val500"]
    L.append(f"\n   val500, paired on {v['n']} clips")
    L.append(f"   {'arm':10s} {'minADE@6 mean(med)':>22s} {'minFDE@6 mean(med)':>22s} "
             f"{'CoC degen':>10s}")
    for nm, m in v["arms"].items():
        L.append(f"   {nm:10s} {m['minADE6_mean']:12.4f} ({m['minADE6_median']:.4f}) "
                 f"{m['minFDE6_mean']:12.4f} ({m['minFDE6_median']:.4f}) "
                 f"{100 * m['coc_degen']:9.1f}%")
    L.append(f"\n   {'pair':22s} {'median':>9s} {'mean':>9s} {'95% CI':>21s} {'p':>10s}")
    for nm, m in v["pairs"].items():
        star = "*" if m["ci"][0] > 0 or m["ci"][1] < 0 else " "
        L.append(f"   {nm:22s} {m['median']:+9.4f} {m['mean']:+9.4f}  "
                 f"[{m['ci'][0]:+.4f},{m['ci'][1]:+.4f}]{star} {m['p']:10.3g}")
    L.append(f"   dualsafe vs dual: minADE unchanged on "
             f"{100 * v['safe_vs_dual_unchanged_ade']:.1f}% of clips, identical CoC on "
             f"{100 * v['safe_vs_dual_identical_coc']:.1f}%")

    c = res["consistency"]
    L.append(f"\nB. CoC-action consistency (val500, n={c['n_clips']}, "
             f"unparseable = 0 as in the paper)")
    L.append(f"   {'arm':12s} {'coverage':>9s} {'agree':>7s} {'d vs base':>10s} "
             f"{'p':>10s} {'p Bonf':>9s} {'degen':>7s} {'minADE@6':>9s}")
    for n, m in c["arms"].items():
        d = c["vs_baseline"].get(n)
        dd = f"{d['delta_pp']:+9.1f}pp" if d else f"{'--':>11s}"
        pp = f"{d['p']:10.3g} {d['p_bonf']:9.3g}" if d else f"{'':>20s}"
        L.append(f"   {n:12s} {100 * m['coverage']:8.1f}% {100 * m['agree_all']:6.1f}% "
                 f"{dd} {pp} {100 * m['coc_degen']:6.1f}% {m['minADE6_median']:9.4f}")
    if "vs_traj" in c:
        L.append("\n   paired vs traj (the arm with no reasoning term)")
        for n, m in c["vs_traj"].items():
            L.append(f"     {n:12s} {m['delta_pp']:+6.1f}pp  p={m['p']:.3g}")

    if "union" in res:
        L.append("\nC. does a third max term displace, and does correlation govern it?")
        L.append(f"   {'axis':5s} {'keep':>12s} {'rho traj-coc':>13s} {'rho traj-J':>11s} "
                 f"{'rho 10 steps':>13s} {'displaced by J':>15s} {'by 10 steps':>12s}")
        for ax, m in res["union"].items():
            L.append(f"   {ax:5s} {m['kept']:5d}/{m['units']:<6d} "
                     f"{m['rho_traj_coc']:+13.3f} {m['rho_traj_j']:+11.3f} "
                     f"{m['rho_steps']:+13.3f} {100 * m['displaced_by_j']:14.1f}% "
                     f"{100 * m['displaced_by_steps']:11.1f}%")
    if "overlap_vs_cost" in res:
        L.append("\n   built arms, all measured against shipped dual on the same val500 clips"
                 f" (minADE@{K})")
        L.append(f"   {'arm':13s} {'Q overlap':>10s} {'MLP':>7s} {'median':>9s} {'mean':>9s} "
                 f"{'95% CI (med)':>21s} {'p':>10s}")
        for arm, m in res["overlap_vs_cost"].items():
            L.append(f"   {arm:13s} {100 * m['overlap_q']:9.1f}% {100 * m['overlap_mlp']:6.1f}% "
                     f"{m['median']:+9.4f} {m['mean']:+9.4f} "
                     f"[{m['ci'][0]:+.4f},{m['ci'][1]:+.4f}] {m['p']:10.3g}")
        r = res.get("overlap_vs_cost_rho")
        if r:
            L.append(f"   Spearman(Q overlap, median cost) over {r['n_arms']} arms = "
                     f"{r['rho']:+.3f} (p={r['p']:.3g}) -- illustrative at n=4, not inferential.")
        L.append("   the pair that needs no correlation: dual_st2000 92.1% / +0.1105 (p=6e-22)")
        L.append("   against maxstep11 92.5% / -0.0117 (n.s.) -- same displacement, opposite")
        L.append("   verdicts. Read the mean column too: the two significant arms sit at 2.4-2.8x")
        L.append("   their medians, so the damage is a tail of clips, not a shift of all of them.")

    L.append("\n   coupling: is the damage larger where the CoC changed?")
    L.append(f"   {'arm':12s} {'n_chg':>6s} {'d|changed':>10s} {'d|same':>9s} "
             f"{'p':>9s} {'base med chg/same':>19s}")
    for n, m in res["coupling"].items():
        L.append(f"   {n:12s} {m['n_changed']:6d} {m['med_changed']:+10.4f} "
                 f"{m['med_same']:+9.4f} {m['p']:9.3g} "
                 f"{m['base_med_changed']:8.3f} /{m['base_med_same']:8.3f}")
    path.write_text("\n".join(L) + "\n")
    print("\n".join(L))


# --------------------------------------------------------------------------
# C. Does a third max term displace, and does the added term's correlation govern it?
#
# Selection only -- no build, no GPU, so this settles the *selection* half of the
# displacement conjecture without settling whether displacement costs anything. The two
# added terms bracket the correlation range already on disk: the J-lens is the decorrelated
# one (traj-J rho ~0.2-0.3) and the ten FM steps are the correlated ones (~0.8-0.9), which
# is exactly the maxstep11 arm whose measured cost is -0.0033 vs dualfix.
# --------------------------------------------------------------------------
def union_block(res):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "head_analysis"))
    import mask_lib as ml
    from run_cocsafe import rank_norm

    imp = dict(np.load(OUT / "importance_v2" / "importance.npz"))
    jl = dict(np.load(OUT / "jlens_v2" / "jlens.npz"))
    step = dict(np.load(OUT / "importance_stepvlm_v1" / "step_importance_vlm.npz"))

    def kept(s):
        m = ml.select_mask(s, U40_RATIO, list(range(s.shape[0])))
        return [set(np.flatnonzero(m[i]).tolist()) for i in range(s.shape[0])]

    def displaced(a, b):
        """Fraction of a's kept units that b drops, averaged over layers."""
        return float(np.mean([1 - len(x & y) / len(x) for x, y in zip(a, b)]))

    def meanrho(mats):
        r = [stats.spearmanr(mats[a][i], mats[b][i]).statistic
             for i in range(mats[0].shape[0])
             for a in range(len(mats)) for b in range(a + 1, len(mats))]
        return float(np.nanmean(r))

    # Overlap-vs-cost on ONE basis: every arm is measured against shipped `dual`, kept sets
    # from slim_meta.json and cost paired over the same val500 clips. Built arms only, so
    # the third max term is absent here -- this table is about whether the *magnitude* of a
    # kept-set move orders the arms at all, which the report's J row cannot speak to.
    base_meta = json.loads((OUT / "slim_dual_u40_v2" / "slim_meta.json").read_text())
    dual = rows("dual_u40_v2_ps_indist")
    res["overlap_vs_cost"] = {}
    for arm, slim, run in (("dualfix", "slim_dualfix_u40_v2", "dualfix_u40_v2_indist"),
                           ("maxstep11", "slim_maxstep11_u40_v2", "maxstep11_u40_v2_indist"),
                           ("dual_st2000", "slim_dual_st2000", "dual_u40_st2000_indist"),
                           ("dualsafe", "slim_dualsafe_u40_v2", "dualsafe_u40_v2_indist")):
        p = OUT / slim / "slim_meta.json"
        v = rows(run)
        if not p.exists() or not v:
            continue
        m = json.loads(p.read_text())
        ids = sorted(set(v) & set(dual))
        dd = np.array([at6(v[c]) - at6(dual[c]) for c in ids])
        lo, hi = bootmed(dd)
        # mean beside the median, per the frozen protocol: minADE deltas are heavy-tailed,
        # and the ratio is the point -- dual_st2000 is 2.8x its median, i.e. the damage sits
        # in a tail of clips rather than spread across them, which the median alone hides.
        res["overlap_vs_cost"][arm] = {
            "n": len(ids), "median": float(np.median(dd)), "mean": float(dd.mean()),
            "ci": [lo, hi], "p": float(stats.wilcoxon(dd).pvalue),
            **{f"overlap_{ax}": float(np.mean(
                [len(set(m["vlm"][i][ax]) & set(base_meta["vlm"][i][ax]))
                 / len(m["vlm"][i][ax]) for i in range(len(m["vlm"]))]))
               for ax in ("q", "mlp")},
        }
    ov = [x["overlap_q"] for x in res["overlap_vs_cost"].values()]
    cost = [x["median"] for x in res["overlap_vs_cost"].values()]
    if len(ov) > 2:
        r = stats.spearmanr(ov, cost)
        res["overlap_vs_cost_rho"] = {"rho": float(r.statistic), "p": float(r.pvalue),
                                      "n_arms": len(ov)}

    res["union"] = {}
    for axis, key, jkey, skey in (("q", "vlm_q", "q_j", "q_abs_step"),
                                  ("mlp", "vlm_mlp", "mlp_j", "mlp_abs_step")):
        t, c = rank_norm(imp[f"traj_{key}"]), rank_norm(imp[f"coc_{key}"])
        j = rank_norm(jl[jkey])
        steps = [rank_norm(a.astype(np.float64)) for a in step[skey]]
        two = kept(np.maximum(t, c))
        res["union"][axis] = {
            "units": int(t.shape[1]), "kept": len(two[0]),
            "rho_traj_coc": meanrho([t, c]), "rho_traj_j": meanrho([t, j]),
            "rho_steps": meanrho(steps),
            "displaced_by_j": displaced(two, kept(np.maximum(np.maximum(t, c), j))),
            "displaced_by_steps": displaced(two, kept(np.maximum.reduce([c, *steps]))),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    res = {}
    dualsafe_block(res, plots)
    consistency_block(res, plots)
    union_block(res)

    (out / "config.json").write_text(json.dumps({
        "purpose": "dual 기준 확장 두 경로: 안전 가중(dualsafe)과 CoC-행동 일관성",
        "safe_tau_m": SAFE_TAU, "k": K, "bootstrap": BOOT,
        "consistency_arms": [d for d, _ in CONS_ARMS],
    }, ensure_ascii=False, indent=2))
    (out / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    write_summary(res, out / "summary.txt")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
