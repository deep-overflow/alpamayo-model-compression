"""템플릿의 <!--TABLE_*--> / <!--KPI_*--> 를 metrics.json 에서 만든 조각으로 채운다.

숫자를 손으로 옮기지 않는 것이 이 리포의 규칙이다 (hard100 초판에서 한 번 어긋난 적이 있다).

Usage:
  .venv/bin/python experiments/evaluation/fill_method_x_draw_report.py \
      experiments/evaluation/method_x_draw_report_template.html /tmp/filled.html
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TPL, OUT = Path(sys.argv[1]), Path(sys.argv[2])
SRC = Path(sys.argv[3]) if len(sys.argv) > 3 else REPO / "outputs/method_x_draw_pairs/metrics.json"
M = json.loads(SRC.read_text())
A, P = M["arms"], M["pairs"]

DRAWS = [("c100", "<code>calib_100</code>"), ("rd_a", "<code>rd100_a</code>"),
         ("rd_b", "<code>rd100_b</code>")]
METHODS = [("tyr", "tyr <span class='sub'>출력 재구성</span>"),
           ("dfh4", "dual+fisher+h4 <span class='sub'>2차 Fisher</span>")]


def pk(a, b):
    """pairs 의 키는 'b - a' 형태. 없으면 부호를 뒤집어 찾는다."""
    if f"{b} - {a}" in P:
        v = P[f"{b} - {a}"]
        return v["delta"], v["ci_lo"], v["ci_hi"], v["wilcoxon_p"], v["better"], v["worse"], v["tie"]
    v = P[f"{a} - {b}"]
    return (-v["delta"], -v["ci_hi"], -v["ci_lo"], v["wilcoxon_p"],
            v["worse"], v["better"], v["tie"])


def sig(lo, hi):
    return " <strong>*</strong>" if (lo > 0 or hi < 0) else ""


# --- 격자
rows = []
for m, mlabel in METHODS:
    cells = "".join(
        f"<td class='r'><strong>{A[f'{m}_{d}']['score']:.3f}</strong>"
        f"<br><span class='sub'>[{A[f'{m}_{d}']['ci_lo']:.3f}, {A[f'{m}_{d}']['ci_hi']:.3f}]</span></td>"
        for d, _ in DRAWS)
    v = np.array([A[f"{m}_{d}"]["score"] for d, _ in DRAWS])
    rows.append(f"<tr><td>{mlabel}</td>{cells}"
                f"<td class='r'><strong>{v.max() - v.min():.3f}</strong></td>"
                f"<td class='r'>{v.std(ddof=1):.3f}</td></tr>")
b = A["baseline"]
rows.append(f"<tr><td>baseline <span class='sub'>무압축</span></td>"
            f"<td class='r' colspan='3'>{b['score']:.3f} "
            f"<span class='sub'>[{b['ci_lo']:.3f}, {b['ci_hi']:.3f}]</span></td>"
            f"<td class='r sub'>&mdash;</td><td class='r sub'>&mdash;</td></tr>")
TABLE_GRID = ("<table><thead><tr><th>방법</th>"
              + "".join(f"<th class='r'>{lab}</th>" for _, lab in DRAWS)
              + "<th class='r'>범위</th><th class='r'>sd</th></tr></thead><tbody>"
              + "".join(rows) + "</tbody></table>")

# --- 방법별 추출 간 비교
rows = []
for m, mlabel in METHODS:
    for i, (d1, l1) in enumerate(DRAWS):
        for d2, l2 in DRAWS[i + 1:]:
            delta, lo, hi, p, w, ls, t = pk(f"{m}_{d1}", f"{m}_{d2}")
            rows.append(
                f"<tr><td>{mlabel.split(' <')[0]}</td><td>{l2} &minus; {l1}</td>"
                f"<td class='r'><strong>{delta:+.4f}</strong></td>"
                f"<td class='r sub'>[{lo:+.4f}, {hi:+.4f}]{sig(lo, hi)}</td>"
                f"<td class='r'>{p:.4f}</td><td class='r'>{w} / {ls} / {t}</td></tr>")
TABLE_WITHIN = ("<table><thead><tr><th>방법</th><th>비교</th><th class='r'>차이</th>"
                "<th class='r'>95% CI</th><th class='r'>Wilcoxon p</th>"
                "<th class='r'>승/패/무</th></tr></thead><tbody>"
                + "".join(rows) + "</tbody></table>")

# --- baseline 대비
rows = []
for m, mlabel in METHODS:
    for d, dlabel in DRAWS:
        delta, lo, hi, p, w, ls, t = pk("baseline", f"{m}_{d}")
        rows.append(f"<tr><td>{mlabel.split(' <')[0]}</td><td>{dlabel}</td>"
                    f"<td class='r'><strong>{delta:+.4f}</strong></td>"
                    f"<td class='r sub'>[{lo:+.4f}, {hi:+.4f}]{sig(lo, hi)}</td>"
                    f"<td class='r'>{p:.4f}</td><td class='r'>{w} / {ls} / {t}</td></tr>")
TABLE_VS_BASE = ("<table><thead><tr><th>방법</th><th>추출</th><th class='r'>vs baseline</th>"
                 "<th class='r'>95% CI</th><th class='r'>Wilcoxon p</th>"
                 "<th class='r'>승/패/무</th></tr></thead><tbody>"
                 + "".join(rows) + "</tbody></table>")

# --- 같은 추출에서 방법 비교
rows = []
for d, dlabel in DRAWS:
    delta, lo, hi, p, w, ls, t = pk(f"dfh4_{d}", f"tyr_{d}")
    rows.append(f"<tr><td>{dlabel}</td>"
                f"<td class='r'><strong>{delta:+.4f}</strong></td>"
                f"<td class='r sub'>[{lo:+.4f}, {hi:+.4f}]{sig(lo, hi)}</td>"
                f"<td class='r'>{p:.4f}</td><td class='r'>{w} / {ls} / {t}</td></tr>")
TABLE_CROSS = ("<table><thead><tr><th>추출</th><th class='r'>tyr &minus; dual+fisher+h4</th>"
               "<th class='r'>95% CI</th><th class='r'>Wilcoxon p</th>"
               "<th class='r'>승/패/무</th></tr></thead><tbody>"
               + "".join(rows) + "</tbody></table>")

# --- 주행 vs CoC
rows = []
for m, mlabel in METHODS:
    for d, dlabel in DRAWS:
        e = A[f"{m}_{d}"]
        rows.append(f"<tr><td>{mlabel.split(' <')[0]}</td><td>{dlabel}</td>"
                    f"<td class='r'>{e['score']:.3f}</td>"
                    f"<td class='r'><strong>{e['coc_degenerate'] * 100:.2f}%</strong></td></tr>")
e = A["baseline"]
rows.append(f"<tr><td>baseline</td><td>&mdash;</td><td class='r'>{e['score']:.3f}</td>"
            f"<td class='r'>{e['coc_degenerate'] * 100:.2f}%</td></tr>")
TABLE_COC = ("<table><thead><tr><th>방법</th><th>추출</th><th class='r'>scene score</th>"
             "<th class='r'>CoC 퇴화율</th></tr></thead><tbody>"
             + "".join(rows) + "</tbody></table>")

# --- 게이트
GK = [("collision_at_fault", "at-fault 충돌"), ("offroad", "offroad"), ("wrong_lane", "wrong lane")]
rows = []
for key in ["baseline"] + [f"{m}_{d}" for m, _ in METHODS for d, _ in DRAWS]:
    name = "baseline" if key == "baseline" else \
        f"{'tyr' if key.startswith('tyr') else 'dual+fisher+h4'} @ {key.split('_', 1)[1]}"
    cells = "".join(f"<td class='r'>{A[key]['gates'][g][0] / A[key]['gates'][g][1] * 100:.1f}%"
                    f" <span class='sub'>({A[key]['gates'][g][0]})</span></td>" for g, _ in GK)
    rows.append(f"<tr><td>{name}</td>{cells}</tr>")
TABLE_GATES = ("<table><thead><tr><th>arm</th>"
               + "".join(f"<th class='r'>{lab}</th>" for _, lab in GK)
               + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")

tv = np.array([A[f"tyr_{d}"]["score"] for d, _ in DRAWS])
dv = np.array([A[f"dfh4_{d}"]["score"] for d, _ in DRAWS])
KPI = {
    "KPI_N": str(M["n_scenes"]),
    "KPI_TYR_RANGE": f"{tv.max() - tv.min():.3f}",
    "KPI_DFH4_RANGE": f"{dv.max() - dv.min():.3f}",
    "KPI_TYR_SD": f"{tv.std(ddof=1):.3f}",
    "KPI_DFH4_SD": f"{dv.std(ddof=1):.3f}",
    "KPI_SD_RATIO": f"{dv.std(ddof=1) / tv.std(ddof=1):.0f}",
    "KPI_BASE": f"{A['baseline']['score']:.3f}",
    "KPI_CROSS_C100": f"{pk('dfh4_c100', 'tyr_c100')[0]:+.3f}",
    "KPI_CROSS_C100_P": f"{pk('dfh4_c100', 'tyr_c100')[3]:.4f}",
    "KPI_CROSS_RDB": f"{pk('dfh4_rd_b', 'tyr_rd_b')[0]:+.3f}",
    "KPI_CROSS_RDB_P": f"{pk('dfh4_rd_b', 'tyr_rd_b')[3]:.2f}",
}

html = TPL.read_text()
for name, frag in (("TABLE_GRID", TABLE_GRID), ("TABLE_WITHIN", TABLE_WITHIN),
                   ("TABLE_VS_BASE", TABLE_VS_BASE), ("TABLE_CROSS", TABLE_CROSS),
                   ("TABLE_COC", TABLE_COC), ("TABLE_GATES", TABLE_GATES)):
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
