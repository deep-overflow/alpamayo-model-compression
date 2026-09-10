"""alpasim 전체 결과 표를 metrics.json 에서 만든다.

arm 이름만으로는 무엇을 시험한 것인지 알 수 없어서, 여기서만 유지하는 설명 사전
(`DESC`)이 있다. 숫자는 전부 collect_alpasim_all.py 의 산출물에서 오고, 이 파일이 더하는
것은 (a) 사람이 읽을 이름과 한 줄 설명, (b) 계열 분류, (c) 정렬뿐이다.

Usage:
  .venv/bin/python experiments/evaluation/fill_alpasim_all_report.py \
      experiments/evaluation/alpasim_all_report_template.html /tmp/filled.html \
      outputs/alpasim_all
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TPL, OUT = Path(sys.argv[1]), Path(sys.argv[2])
SRC = Path(sys.argv[3]) if len(sys.argv) > 3 else REPO / "outputs/alpasim_all"
SRC = SRC if SRC.is_absolute() else REPO / SRC
B = json.loads((SRC / "metrics.json").read_text())

# config -> (표시 이름, 계열, 한 줄 설명)
FAM = {"base": "기준", "sel": "선택 기준", "recon": "출력 재구성", "ext": "외부·타 연구원",
       "comp": "합성·회복", "cal": "캘리브레이션"}
DESC = {
    "baseline": ("baseline", "base", "무압축 nvidia/Alpamayo-1.5-10B"),
    # 선택 기준 (Taylor / J-lens), VLM only, 균등 24.0%
    "slim_dual_u40_v2": ("dual", "sel", "dual Taylor max(rank I_traj, rank I_CoC) — 출하본"),
    "slim_traj_u40_v2": ("traj", "sel", "궤적 Taylor 단독"),
    "slim_coc_u40_v2": ("coc", "sel", "CoC NLL Taylor 단독"),
    "slim_j_u40_v2": ("j", "sel", "J-lens 단독"),
    "slim_jtraj_u40_v2": ("j_traj", "sel", "max(rank I_traj, rank J) — rollout 없는 기준"),
    "slim_wanda_u40_v2": ("wanda", "sel", "Wanda (활성 규모 x 가중치), 외부 기준"),
    "slim_dual_u40_w4": ("dual_w4", "sel", "dual, 층당 Q 4개 제한 배분"),
    "slim_dual_u40_w8": ("dual_w8", "sel", "dual, 층당 Q 8개 제한 배분"),
    # 출력 재구성 (Tyr / OSSCAR) — 가중치를 다시 쓴다
    "slim_tyr_u40_r": ("tyr", "recon", "Tyr 탐색, 적합도 KL_coc + MSE_vf"),
    "slim_tyrK": ("tyrK", "recon", "Tyr 탐색, 적합도 KL_coc 단독"),
    "slim_dualr_u40": ("dualr", "recon", "dual 선택 + 재구성 refit"),
    "slim_dualr_wl_u40": ("dualr_wl", "recon", "dualr, Hessian 에 LingoQA train 포함"),
    "slim_dualrwl_em93p75_u40": ("dualr_wl+e", "recon", "dualr_wl + expert MLP 93.75%"),
    # 캘리브레이션 추출/크기
    "slim_dual_st2000": ("dual@st2000_a", "cal", "dual 기준을 2,000클립 추출 A 로 추정"),
    "slim_dual_st2000b": ("dual@st2000_b", "cal", "dual 기준을 2,000클립 추출 B 로 (A 와 서로소)"),
    "slim_tyr_rd_a": ("tyr@rd_a", "cal", "Tyr, 캘리브레이션 rd100_a"),
    "slim_tyr_rd_b": ("tyr@rd_b", "cal", "Tyr, 캘리브레이션 rd100_b"),
    "lp_dfh4_calib100": ("dfh4@calib100", "cal", "LLM-Pruner param_mix 2차 + 층당 Q4, calib_100"),
    "lp_dfh4_rd_a": ("dfh4@rd_a", "cal", "같은 방법, 캘리브레이션 rd100_a"),
    "lp_dfh4_rd_b": ("dfh4@rd_b", "cal", "같은 방법, 캘리브레이션 rd100_b"),
    # 합성 / 회복
    "slim_dualexp_u40_em93p75": ("dual+expert", "comp", "dual(VLM) + expert MLP 93.75% 추가 제거"),
    "slim_expert_znorm_r25": ("expert_znorm", "comp", "expert 전용 25% (znorm 스텝 집계)"),
    "slim_recover_dual_u55": ("dual_u55+LoRA", "comp", "55% 제거 후 KI-LoRA 회복"),
    # 외부
    "lp_r50": ("lp_r50", "ext", "LLM-Pruner (soowon), 25.0% 제거"),
    "G_default_r40": ("G_default_r40", "ext", "sangoh, expert pruning"),
}
ORDER = ["base", "sel", "recon", "cal", "comp", "ext"]


def pfmt(p):
    if p is None:
        return "&mdash;"
    if p < 1e-4:
        return "&lt;1e-4"
    return f"{p:.4f}" if p < 0.001 else f"{p:.3f}"


def sig(lo, hi):
    return " <strong>*</strong>" if (lo > 0 or hi < 0) else ""


def arm_table(suite, sort_by_score=True):
    arms = list(suite["arms"])
    if sort_by_score:
        arms.sort(key=lambda a: -a["score"])
    rows = []
    for a in arms:
        cfg = a["config"]
        name, fam, desc = DESC.get(cfg, (cfg, "sel", ""))
        base = cfg == "baseline"
        cls = " class='base'" if base else (" class='ext'" if fam == "ext" else "")
        if base:
            dcell = "<td class='r sub' colspan='3'>&mdash; 기준 &mdash;</td>"
        else:
            dcell = (f"<td class='r'><strong>{a['delta']:+.4f}</strong></td>"
                     f"<td class='r sub'>[{a['delta_lo']:+.4f}, {a['delta_hi']:+.4f}]"
                     f"{sig(a['delta_lo'], a['delta_hi'])}</td>"
                     f"<td class='r'>{pfmt(a['wilcoxon_p'])}</td>")
        rm = f"{a['removed'] / 1e9:.2f}B" if a.get("removed") else "&mdash;"
        coc = f"{a['coc'] * 100:.2f}%" if a.get("coc") is not None else "&mdash;"
        gates = a["gates"]

        def g(k, gates=gates):     # 루프 변수를 기본값으로 묶는다 (늦은 바인딩 방지)
            n_k, n = gates[k]
            return f"{n_k / n * 100:.1f}%" if n else "&mdash;"

        rows.append(
            f"<tr{cls}><td><strong>{name}</strong><br>"
            f"<span class='sub'>{desc}</span></td>"
            f"<td class='r sub'>{FAM[fam]}</td>"
            f"<td class='r'><strong>{a['score']:.3f}</strong><br>"
            f"<span class='sub'>[{a['ci_lo']:.3f}, {a['ci_hi']:.3f}]</span></td>"
            f"{dcell}"
            f"<td class='r'>{g('collision_at_fault')}</td>"
            f"<td class='r'>{g('offroad')}</td>"
            f"<td class='r'>{coc}</td>"
            f"<td class='r sub'>{rm}</td></tr>")
    tbl = ("<table><thead><tr><th>arm</th><th class='r'>계열</th>"
           "<th class='r'>scene score</th><th class='r'>vs baseline</th>"
           "<th class='r'>95% CI</th><th class='r'>p</th>"
           "<th class='r'>과실충돌</th><th class='r'>offroad</th>"
           "<th class='r'>CoC 퇴화</th><th class='r'>제거</th>"
           "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")
    # 게이트 값이 없는 rollout 이 있으면 분모(전체 rollout)에 그대로 남는다 -- 비율이 아주
    # 약간 보수적이 되므로 몇 개인지 밝힌다.
    miss = {a["config"]: max(a["gate_missing"].values()) for a in arms if a.get("gate_missing")}
    if miss:
        worst = max(miss.values())
        tbl += (f"<p class='sub'>게이트 지표가 없는 rollout이 arm당 최대 {worst}개 있다"
                f"({len(miss)}/{len(arms)} arm). 분모는 전체 rollout으로 두었으므로 위 비율은"
                f" 그만큼 보수적이다.</p>")
    return tbl


S150 = next(s for s in B["suites"] if s["n_scenes"] == 150)
SH100 = next(s for s in B["suites"] if s["n_scenes"] == 100)
TABLE_150 = arm_table(S150)
TABLE_H100 = arm_table(SH100)

# --- 두 suite 를 모두 돈 arm
by100 = {a["config"]: a for a in SH100["arms"]}
rows = []
for a in sorted(S150["arms"], key=lambda x: -x["score"]):
    b = by100.get(a["config"])
    if not b:
        continue
    name = DESC.get(a["config"], (a["config"],))[0]
    d150 = f"{a['delta']:+.4f}" if "delta" in a else "&mdash;"
    d100 = f"{b['delta']:+.4f}" if "delta" in b else "&mdash;"
    rows.append(f"<tr><td>{name}</td>"
                f"<td class='r'>{a['score']:.3f}</td><td class='r'>{d150}</td>"
                f"<td class='r'>{b['score']:.3f}</td><td class='r'>{d100}</td></tr>")
TABLE_BOTH = ("<table><thead><tr><th>arm</th>"
              "<th class='r'>150씬 score</th><th class='r'>vs base</th>"
              "<th class='r'>hard100 score</th><th class='r'>vs base</th>"
              "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")

# --- 계열별 요약
rows = []
for fam in ORDER:
    if fam == "base":
        continue
    xs = [a for a in S150["arms"] if DESC.get(a["config"], ("", "sel"))[1] == fam and "delta" in a]
    if not xs:
        continue
    wins = sum(1 for a in xs if a["delta"] > 0)
    sigw = sum(1 for a in xs if a["delta_lo"] > 0)
    best = max(xs, key=lambda a: a["delta"])
    worst = min(xs, key=lambda a: a["delta"])
    rows.append(f"<tr><td>{FAM[fam]}</td><td class='r'>{len(xs)}</td>"
                f"<td class='r'>{wins}</td><td class='r'>{sigw}</td>"
                f"<td class='r'>{DESC[best['config']][0]} {best['delta']:+.4f}</td>"
                f"<td class='r'>{DESC[worst['config']][0]} {worst['delta']:+.4f}</td></tr>")
TABLE_FAM = ("<table><thead><tr><th>계열</th><th class='r'>arm</th>"
             "<th class='r'>baseline 상회</th><th class='r'>유의하게 상회</th>"
             "<th class='r'>최고</th><th class='r'>최저</th>"
             "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")

n_sig_up = sum(1 for a in S150["arms"] if a.get("delta_lo", 0) > 0)
n_sig_dn = sum(1 for a in S150["arms"] if a.get("delta_hi") is not None and a["delta_hi"] < 0)
best150 = max((a for a in S150["arms"] if "delta" in a), key=lambda a: a["delta"])
worst150 = min((a for a in S150["arms"] if "delta" in a), key=lambda a: a["delta"])

KPI = {
    "KPI_N150": str(S150["n_scenes"]), "KPI_NH100": str(SH100["n_scenes"]),
    "KPI_ARMS150": str(len(S150["arms"])), "KPI_ARMSH100": str(len(SH100["arms"])),
    "KPI_ROLLOUTS": f"{sum(a['n_rollouts'] for s in B['suites'] for a in s['arms']):,}",
    "KPI_SIG_UP": str(n_sig_up), "KPI_SIG_DN": str(n_sig_dn),
    "KPI_BASE150": f"{next(a for a in S150['arms'] if a['config'] == 'baseline')['score']:.3f}",
    "KPI_BASEH100": f"{next(a for a in SH100['arms'] if a['config'] == 'baseline')['score']:.3f}",
    "KPI_BEST": f"{DESC[best150['config']][0]} {best150['delta']:+.4f}",
    "KPI_WORST": f"{DESC[worst150['config']][0]} {worst150['delta']:+.4f}",
    "KPI_SPREAD": f"{best150['delta'] - worst150['delta']:.3f}",
}

html = TPL.read_text()
for name, frag in (("TABLE_150", TABLE_150), ("TABLE_H100", TABLE_H100),
                   ("TABLE_BOTH", TABLE_BOTH), ("TABLE_FAM", TABLE_FAM)):
    marker = f"<!--{name}-->"
    if marker not in html:
        print(f"경고: 템플릿에 {marker} 가 없습니다", file=sys.stderr)
    html = html.replace(marker, frag)
for k, v in KPI.items():
    html = html.replace(f"<!--{k}-->", v)

missing = [a["config"] for s in B["suites"] for a in s["arms"] if a["config"] not in DESC]
if missing:
    print(f"경고: 설명이 없는 arm {sorted(set(missing))}", file=sys.stderr)
left = [m for m in ("TABLE_", "KPI_") if f"<!--{m}" in html]
if left:
    sys.exit(f"채우지 못한 자리표시자가 남았습니다 ({left})")
OUT.write_text(html)
print(f"-> {OUT}  ({len(html):,} bytes)")
