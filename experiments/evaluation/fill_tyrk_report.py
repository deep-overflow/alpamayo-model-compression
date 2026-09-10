"""템플릿의 <!--TABLE_*--> / <!--KPI_*--> 를 metrics.json 에서 만든 조각으로 채운다.

숫자를 손으로 옮기지 않는 것이 이 리포의 규칙이다 (hard100 초판에서 한 번 어긋난 적이 있다).
CoC 페어드 통계는 `coc_paired_tyrk.py` 가 남긴 씬별 배열에서 여기서 다시 계산한다 -- 저장된
것은 배열뿐이므로 검정 결과도 이 스크립트가 책임진다.

Usage:
  .venv/bin/python experiments/evaluation/fill_tyrk_report.py \
      experiments/evaluation/tyrk_report_template.html /tmp/filled.html
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
TPL, OUT = Path(sys.argv[1]), Path(sys.argv[2])
SRC = Path(sys.argv[3]) if len(sys.argv) > 3 else REPO / "outputs/tyrK_pairs"
M = json.loads((SRC / "metrics.json").read_text())
A, P, G = M["arms"], M["pairs"], M["gate_fisher_p"]

ARMS = [("baseline", "baseline <span class='sub'>무압축</span>"),
        ("tyr_c100", "tyr @ calib_100 <span class='sub'>KL<sub>coc</sub> + MSE<sub>vf</sub></span>"),
        ("tyrK", "tyrK <span class='sub'>KL<sub>coc</sub> 단독</span>")]
GK = [("collision_at_fault", "at-fault 충돌"), ("offroad", "offroad"), ("wrong_lane", "wrong lane")]


def sig(lo, hi):
    return " <strong>*</strong>" if (lo > 0 or hi < 0) else ""


def wilson(k, n, z=1.96):
    """이항 비율의 Wilson 95% 구간 -- 0에 가까운 게이트에서 정규근사는 음수 하한을 준다."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, c - h), min(1.0, c + h)


# --- 1) arm 요약
rows = []
for k, label in ARMS:
    e = A[k]
    rows.append(
        f"<tr><td>{label}</td>"
        f"<td class='r'><strong>{e['score']:.3f}</strong></td>"
        f"<td class='r sub'>[{e['ci_lo']:.3f}, {e['ci_hi']:.3f}]</td>"
        f"<td class='r'>{e['median']:.3f}</td>"
        f"<td class='r'>{e['pass_frac'] * 100:.1f}%</td>"
        f"<td class='r'>{e['coc_degenerate'] * 100:.2f}%</td></tr>")
TABLE_ARMS = ("<table><thead><tr><th>arm</th><th class='r'>scene score</th>"
              "<th class='r'>95% CI</th><th class='r'>중앙값</th><th class='r'>pass%</th>"
              "<th class='r'>CoC 퇴화</th></tr></thead><tbody>"
              + "".join(rows) + "</tbody></table>")

# --- 2) 페어드 비교
LBL = {"baseline": "baseline", "tyr_c100": "tyr @ calib_100", "tyrK": "tyrK"}
rows = []
for key in ("tyr_c100 - baseline", "tyrK - baseline", "tyrK - tyr_c100"):
    v = P[key]
    b, a = key.split(" - ")
    cls = " class='hl'" if key == "tyrK - tyr_c100" else ""
    rows.append(
        f"<tr{cls}><td>{LBL[b]} &minus; {LBL[a]}</td>"
        f"<td class='r'><strong>{v['delta']:+.4f}</strong></td>"
        f"<td class='r sub'>[{v['ci_lo']:+.4f}, {v['ci_hi']:+.4f}]{sig(v['ci_lo'], v['ci_hi'])}</td>"
        f"<td class='r'>{v['wilcoxon_p']:.4f}</td>"
        f"<td class='r'>{v['better']} / {v['worse']} / {v['tie']}</td></tr>")
TABLE_PAIRS = ("<table><thead><tr><th>비교</th><th class='r'>차이</th><th class='r'>95% CI</th>"
               "<th class='r'>Wilcoxon p</th><th class='r'>승/패/무</th></tr></thead><tbody>"
               + "".join(rows) + "</tbody></table>")

# --- 3) 게이트
rows = []
for k, label in ARMS:
    cells = ""
    for g, _ in GK:
        n_k, n = A[k]["gates"][g]
        lo, hi = wilson(n_k, n)
        cells += (f"<td class='r'>{n_k / n * 100:.1f}% <span class='sub'>({n_k}/{n})</span>"
                  f"<br><span class='sub'>[{lo * 100:.1f}, {hi * 100:.1f}]</span></td>")
    rows.append(f"<tr><td>{label}</td>{cells}</tr>")
fisher = "".join(
    f"<td class='r'>{G[f'tyrK - tyr_c100|{g}']:.3f}</td>" for g, _ in GK)
rows.append(f"<tr class='hl'><td>tyrK vs tyr@calib100 <span class='sub'>Fisher p</span></td>{fisher}</tr>")
TABLE_GATES = ("<table><thead><tr><th>arm</th>"
               + "".join(f"<th class='r'>{lab}</th>" for _, lab in GK)
               + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")

# --- 4) CoC 페어드 (씬별 배열에서 직접 계산)
cocf = SRC / "coc_paired.json"
if cocf.exists():
    C = json.loads(cocf.read_text())
    ps = C["per_scene"]
    rows = []
    coc_kpi = {}
    for kind, klabel in (("degenerate_frac", "퇴화 전체"), ("empty_frac", "빈 출력"),
                         ("soup_frac", "soup")):
        a = np.array(ps["tyr_c100"][kind])
        b = np.array(ps["tyrK"][kind])
        base = np.array(ps["baseline"][kind])
        d = b - a
        rng = np.random.default_rng(0)
        boot = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(10000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        nz = d[d != 0]
        p = float(stats.wilcoxon(nz).pvalue) if len(nz) else float("nan")
        rows.append(
            f"<tr><td>{klabel}</td>"
            f"<td class='r'>{base.mean() * 100:.2f}%</td>"
            f"<td class='r'>{a.mean() * 100:.2f}%</td>"
            f"<td class='r'>{b.mean() * 100:.2f}%</td>"
            f"<td class='r'><strong>{d.mean() * 100:+.2f} pp</strong></td>"
            f"<td class='r sub'>[{lo * 100:+.2f}, {hi * 100:+.2f}]{sig(lo, hi)}</td>"
            f"<td class='r'>{p:.4f}</td>"
            f"<td class='r'>{(d < 0).sum()} / {(d > 0).sum()} / {(d == 0).sum()}</td></tr>")
        if kind == "degenerate_frac":
            coc_kpi = {"KPI_COC_DELTA": f"{d.mean() * 100:+.2f}", "KPI_COC_P": f"{p:.4f}",
                       "KPI_COC_LO": f"{lo * 100:+.2f}", "KPI_COC_HI": f"{hi * 100:+.2f}",
                       "KPI_COC_WLT": f"{(d < 0).sum()}/{(d > 0).sum()}/{(d == 0).sum()}",
                       "KPI_COC_N": str(C["n_scenes"])}
    TABLE_COC = ("<table><thead><tr><th>지표</th><th class='r'>baseline</th>"
                 "<th class='r'>tyr@calib100</th><th class='r'>tyrK</th>"
                 "<th class='r'>차이</th><th class='r'>95% CI</th>"
                 "<th class='r'>Wilcoxon p</th><th class='r'>개선/악화/동일</th>"
                 "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")
else:
    TABLE_COC = "<p class='sub'>coc_paired.json 이 없습니다 — coc_paired_tyrk.py 를 먼저 돌리세요.</p>"
    coc_kpi = dict.fromkeys(
        ("KPI_COC_DELTA", "KPI_COC_P", "KPI_COC_LO", "KPI_COC_HI", "KPI_COC_WLT", "KPI_COC_N"), "—")

# --- 5) 선택 중첩. 남긴 유닛 수는 slim_meta.json 에서 직접 센다 -- tyr 은 층마다 폭이 달라
#        "층당 19" 같은 상수를 쓸 수 없다.
Q_PER_LAYER, MLP_PER_LAYER = 32, 12288


def kept_counts(slim_dir):
    """slim_meta.json['vlm'] 는 층당 {'q': [...], 'mlp': [...]} 의 남긴 인덱스 목록이다.
    tyr 은 층마다 폭이 달라 '층당 19' 같은 상수를 쓸 수 없으므로 직접 센다."""
    m = json.loads((REPO / "outputs" / slim_dir / "slim_meta.json").read_text())
    layers = m["vlm"]
    q = sum(len(x["q"]) for x in layers)
    mlp = sum(len(x["mlp"]) for x in layers)
    return q, len(layers) * Q_PER_LAYER, mlp, len(layers) * MLP_PER_LAYER


ov = M["selection_overlap"]["tyr_c100|tyrK"]
kq, tq, km, tm = kept_counts(A["tyrK"]["slim"])
TABLE_OVERLAP = (
    "<table><thead><tr><th>축</th><th class='r'>중첩률</th><th class='r'>남긴 유닛</th>"
    "<th class='r'>서로 다른 유닛</th></tr></thead><tbody>"
    f"<tr><td>Q head</td><td class='r'><strong>{ov['q'] * 100:.1f}%</strong></td>"
    f"<td class='r'>{kq:,} / {tq:,}</td>"
    f"<td class='r'>{round((1 - ov['q']) * kq):,}</td></tr>"
    f"<tr><td>MLP 채널</td><td class='r'><strong>{ov['mlp'] * 100:.1f}%</strong></td>"
    f"<td class='r'>{km:,} / {tm:,}</td>"
    f"<td class='r'>{round((1 - ov['mlp']) * km):,}</td></tr>"
    "</tbody></table>")

p = P["tyrK - tyr_c100"]
KPI = {
    "KPI_N": str(M["n_scenes"]),
    "KPI_DELTA": f"{p['delta']:+.4f}",
    "KPI_DELTA_LO": f"{p['ci_lo']:+.4f}",
    "KPI_DELTA_HI": f"{p['ci_hi']:+.4f}",
    "KPI_P": f"{p['wilcoxon_p']:.2f}",
    "KPI_WLT": f"{p['better']}/{p['worse']}/{p['tie']}",
    "KPI_OV_Q": f"{ov['q'] * 100:.1f}",
    "KPI_OV_MLP": f"{ov['mlp'] * 100:.1f}",
    "KPI_OV_DIFF": str(round((1 - ov["q"]) * kq)),
    "KPI_KEPT_Q": f"{kq:,}",
    "KPI_TOT_Q": f"{tq:,}",
    "KPI_BASE": f"{A['baseline']['score']:.3f}",
    "KPI_TYR": f"{A['tyr_c100']['score']:.3f}",
    "KPI_TYRK": f"{A['tyrK']['score']:.3f}",
    "KPI_TYR_VS_BASE": f"{P['tyr_c100 - baseline']['delta']:+.4f}",
    "KPI_TYR_VS_BASE_P": f"{P['tyr_c100 - baseline']['wilcoxon_p']:.4f}",
    "KPI_TYRK_VS_BASE": f"{P['tyrK - baseline']['delta']:+.4f}",
    "KPI_TYRK_VS_BASE_P": f"{P['tyrK - baseline']['wilcoxon_p']:.4f}",
    **coc_kpi,
}

html = TPL.read_text()
for name, frag in (("TABLE_ARMS", TABLE_ARMS), ("TABLE_PAIRS", TABLE_PAIRS),
                   ("TABLE_GATES", TABLE_GATES), ("TABLE_COC", TABLE_COC),
                   ("TABLE_OVERLAP", TABLE_OVERLAP)):
    marker = f"<!--{name}-->"
    if marker not in html:
        print(f"경고: 템플릿에 {marker} 가 없습니다", file=sys.stderr)
    html = html.replace(marker, frag)
for k, v in KPI.items():
    html = html.replace(f"<!--{k}-->", v)

left = [m for m in ("TABLE_", "KPI_") if f"<!--{m}" in html]
if left:
    sys.exit(f"채우지 못한 자리표시자가 남았습니다 ({left})")
OUT.write_text(html)
print(f"-> {OUT}  ({len(html):,} bytes)")
