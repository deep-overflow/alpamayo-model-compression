"""hard100 5-arm 리포트의 <!--TABLE_*--> / <!--KPI_*--> 를 metrics.json 에서 채운다.

숫자를 손으로 옮기지 않는 것이 이 리포의 규칙이다 (hard100 초판에서 한 번 어긋난 적이 있다).
검정력(MDE)도 여기서 씬별 배열로 직접 계산한다 -- "유의하지 않다" 를 "차이가 없다" 로 읽지
않으려면 이 설계가 무엇을 분해할 수 있는지 같이 적어야 한다.

Usage:
  .venv/bin/python experiments/evaluation/fill_h100_tyrk_report.py \
      experiments/evaluation/h100_tyrk_report_template.html /tmp/filled.html \
      outputs/h100_tyrK_pairs
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
TPL, OUT = Path(sys.argv[1]), Path(sys.argv[2])
SRC = Path(sys.argv[3]) if len(sys.argv) > 3 else REPO / "outputs/h100_tyrK_pairs"
SRC = SRC if SRC.is_absolute() else REPO / SRC
M = json.loads((SRC / "metrics.json").read_text())
A, P, G = M["arms"], M["pairs"], M["gate_fisher_p"]

ARMS = [
    ("baseline", "baseline", "무압축"),
    ("dual", "dual", "Taylor 선택, 재구성 없음"),
    ("tyr_c100", "tyr @ calib_100", "재구성, 적합도 KL+MSE"),
    ("tyrK", "tyrK", "재구성, 적합도 KL 단독"),
    ("lp_r50", "lp_r50", "LLM-Pruner (외부, 25.0% 제거)"),
]
COMP = [a for a, _, _ in ARMS if a != "baseline"]
GK = [("collision_at_fault", "at-fault 충돌"), ("offroad", "offroad"), ("wrong_lane", "wrong lane")]


def sig(lo, hi):
    return " <strong>*</strong>" if (lo > 0 or hi < 0) else ""


def pfmt(p):
    """p=0.0000 은 "정확히 0" 으로 읽히므로 아주 작은 값은 상한으로 적는다."""
    return "&lt;1e-4" if p < 1e-4 else f"{p:.4f}"


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, c - h), min(1.0, c + h)


def pk(a, b):
    """pairs 의 키는 'b - a'. 없으면 부호를 뒤집어 찾는다."""
    if f"{b} - {a}" in P:
        v = P[f"{b} - {a}"]
        return v["delta"], v["ci_lo"], v["ci_hi"], v["wilcoxon_p"], v["better"], v["worse"], v["tie"]
    v = P[f"{a} - {b}"]
    return (-v["delta"], -v["ci_hi"], -v["ci_lo"], v["wilcoxon_p"],
            v["worse"], v["better"], v["tie"])


S = {k: np.array(v) for k, v in M["per_scene_score"].items()}

# --- 1) arm 요약
rows = []
for k, label, desc in ARMS:
    e = A[k]
    coc = e.get("coc_degenerate")
    rows.append(
        f"<tr><td><strong>{label}</strong><br><span class='sub'>{desc}</span></td>"
        f"<td class='r'><strong>{e['score']:.3f}</strong></td>"
        f"<td class='r sub'>[{e['ci_lo']:.3f}, {e['ci_hi']:.3f}]</td>"
        f"<td class='r'>{e['median']:.3f}</td>"
        f"<td class='r'>{e['pass_frac'] * 100:.1f}%</td>"
        f"<td class='r'>{coc * 100:.2f}%</td></tr>" if coc is not None else
        f"<tr><td><strong>{label}</strong><br><span class='sub'>{desc}</span></td>"
        f"<td class='r'><strong>{e['score']:.3f}</strong></td>"
        f"<td class='r sub'>[{e['ci_lo']:.3f}, {e['ci_hi']:.3f}]</td>"
        f"<td class='r'>{e['median']:.3f}</td>"
        f"<td class='r'>{e['pass_frac'] * 100:.1f}%</td>"
        f"<td class='r sub'>&mdash;</td></tr>")
TABLE_ARMS = ("<table><thead><tr><th>arm</th><th class='r'>scene score</th>"
              "<th class='r'>95% CI</th><th class='r'>중앙값</th><th class='r'>pass%</th>"
              "<th class='r'>CoC 퇴화</th></tr></thead><tbody>"
              + "".join(rows) + "</tbody></table>")

# --- 2) baseline 대비 (G1)
rows = []
for k in COMP:
    label = next(lb for a, lb, _ in ARMS if a == k)
    d, lo, hi, p, w, ls, t = pk("baseline", k)
    rows.append(f"<tr><td>{label}</td>"
                f"<td class='r'><strong>{d:+.4f}</strong></td>"
                f"<td class='r sub'>[{lo:+.4f}, {hi:+.4f}]{sig(lo, hi)}</td>"
                f"<td class='r'>{p:.4f}</td>"
                f"<td class='r'>{w} / {ls} / {t}</td></tr>")
TABLE_VS_BASE = ("<table><thead><tr><th>arm</th><th class='r'>vs baseline</th>"
                 "<th class='r'>95% CI</th><th class='r'>Wilcoxon p</th>"
                 "<th class='r'>승/패/무</th></tr></thead><tbody>"
                 + "".join(rows) + "</tbody></table>")

# --- 3) 방법 간 전 쌍 (G2)
rows = []
for a, b in itertools.combinations(COMP, 2):
    la = next(lb for x, lb, _ in ARMS if x == a)
    lb_ = next(lb for x, lb, _ in ARMS if x == b)
    d, lo, hi, p, w, ls, t = pk(a, b)
    cls = " class='hl'" if {a, b} == {"tyr_c100", "tyrK"} else ""
    ov = M["selection_overlap"].get(f"{a}|{b}") or M["selection_overlap"].get(f"{b}|{a}")
    ovs = f"{ov['q'] * 100:.1f}%" if ov else "&mdash;"
    rows.append(f"<tr{cls}><td>{lb_} &minus; {la}</td>"
                f"<td class='r'><strong>{d:+.4f}</strong></td>"
                f"<td class='r sub'>[{lo:+.4f}, {hi:+.4f}]{sig(lo, hi)}</td>"
                f"<td class='r'>{p:.2f}</td>"
                f"<td class='r'>{w} / {ls} / {t}</td>"
                f"<td class='r'>{ovs}</td></tr>")
TABLE_METHODS = ("<table><thead><tr><th>비교</th><th class='r'>차이</th><th class='r'>95% CI</th>"
                 "<th class='r'>Wilcoxon p</th><th class='r'>승/패/무</th>"
                 "<th class='r'>Q 선택 중첩</th></tr></thead><tbody>"
                 + "".join(rows) + "</tbody></table>")

# --- 4) 게이트
rows = []
for k, label, _ in ARMS:
    cells = ""
    for g, _ in GK:
        n_k, n = A[k]["gates"][g]
        lo, hi = wilson(n_k, n)
        cells += (f"<td class='r'>{n_k / n * 100:.1f}% <span class='sub'>({n_k}/{n})</span>"
                  f"<br><span class='sub'>[{lo * 100:.1f}, {hi * 100:.1f}]</span></td>")
    rows.append(f"<tr><td>{label}</td>{cells}</tr>")
TABLE_GATES = ("<table><thead><tr><th>arm</th>"
               + "".join(f"<th class='r'>{lab}</th>" for _, lab in GK)
               + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")

# --- 5) CoC 분해 (씬별 배열이 있으면 페어드까지)
cocf = SRC / "coc_paired.json"
KPI_COC = {}
if cocf.exists():
    C = json.loads(cocf.read_text())["per_scene"]
    # CoC 텍스트가 한 줄도 없는 씬이 있다 (전 arm 공통 2개). nanmean 으로 빼고, 페어드는
    # 양쪽 다 값이 있는 씬만 쓴다 -- 평범한 mean 은 배열 전체를 NaN 으로 만든다.
    rows = []
    for k, label, _ in ARMS:
        if k not in C:
            continue
        d = np.array(C[k]["degenerate_frac"]) * 100
        e = np.array(C[k]["empty_frac"]) * 100
        s = np.array(C[k]["soup_frac"]) * 100
        rows.append(f"<tr><td>{label}</td><td class='r'>{np.nanmean(d):.2f}%</td>"
                    f"<td class='r'>{np.nanmean(e):.2f}%</td>"
                    f"<td class='r'>{np.nanmean(s):.2f}%</td></tr>")
    a = np.array(C["tyr_c100"]["degenerate_frac"])
    b = np.array(C["tyrK"]["degenerate_frac"])
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    dd = b - a
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(dd, len(dd), replace=True).mean() for _ in range(10000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    nz = dd[dd != 0]
    pv = float(stats.wilcoxon(nz).pvalue) if len(nz) else float("nan")
    rows.append(f"<tr class='hl'><td>tyrK &minus; tyr@calib100 <span class='sub'>페어드</span></td>"
                f"<td class='r'><strong>{dd.mean() * 100:+.2f} pp</strong><br>"
                f"<span class='sub'>[{lo * 100:+.2f}, {hi * 100:+.2f}]{sig(lo, hi)} "
                f"p={pfmt(pv)}</span></td>"
                f"<td class='r sub' colspan='2'>개선/악화/동일 "
                f"{(dd < 0).sum()}/{(dd > 0).sum()}/{(dd == 0).sum()}</td></tr>")
    TABLE_COC = ("<table><thead><tr><th>arm</th><th class='r'>퇴화 전체</th>"
                 "<th class='r'>빈 출력</th><th class='r'>soup</th>"
                 "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")
    KPI_COC = {"KPI_COC_DELTA": f"{dd.mean() * 100:+.2f}", "KPI_COC_P": pfmt(pv)}
else:
    rows = []
    for k, label, _ in ARMS:
        c = A[k].get("coc_degenerate")
        rows.append(f"<tr><td>{label}</td><td class='r'>"
                    + (f"{c * 100:.2f}%" if c is not None else "&mdash;") + "</td></tr>")
    TABLE_COC = ("<table><thead><tr><th>arm</th><th class='r'>CoC 퇴화</th></tr></thead><tbody>"
                 + "".join(rows) + "</tbody></table>")
    KPI_COC = {"KPI_COC_DELTA": "&mdash;", "KPI_COC_P": "&mdash;"}

# --- 6) 검정력. 이 설계가 무엇을 분해할 수 있는지 쌍마다 적는다.
#     MDE(80% 검정력, 양측 5%) = 2.80 x SE, SE = sd(페어드 차이)/sqrt(n)
rows = []
for a, b in (("tyr_c100", "tyrK"), ("dual", "tyrK"), ("dual", "tyr_c100")):
    d = S[b] - S[a]
    se = d.std(ddof=1) / np.sqrt(len(d))
    la = next(lb for x, lb, _ in ARMS if x == a)
    lb_ = next(lb for x, lb, _ in ARMS if x == b)
    rows.append(f"<tr><td>{lb_} &minus; {la}</td>"
                f"<td class='r'>{d.mean():+.4f}</td>"
                f"<td class='r'>{d.std(ddof=1):.3f}</td>"
                f"<td class='r'>{se:.4f}</td>"
                f"<td class='r'><strong>{2.80 * se:.3f}</strong></td></tr>")
TABLE_POWER = ("<table><thead><tr><th>비교</th><th class='r'>관측 차이</th>"
               "<th class='r'>씬 단위 sd</th><th class='r'>SE</th>"
               "<th class='r'>MDE <span class='sub'>(80%)</span></th>"
               "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")

d_tt = S["tyrK"] - S["tyr_c100"]
se_tt = d_tt.std(ddof=1) / np.sqrt(len(d_tt))
ov_tt = M["selection_overlap"].get("tyr_c100|tyrK") or M["selection_overlap"].get("tyrK|tyr_c100")
ov_dt = M["selection_overlap"].get("dual|tyr_c100") or M["selection_overlap"].get("tyr_c100|dual")

KPI = {
    "KPI_N": str(M["n_scenes"]),
    "KPI_BASE": f"{A['baseline']['score']:.3f}",
    "KPI_TYRK": f"{A['tyrK']['score']:.3f}",
    "KPI_TYRK_VS_BASE": f"{pk('baseline', 'tyrK')[0]:+.4f}",
    "KPI_TYRK_VS_BASE_P": f"{pk('baseline', 'tyrK')[3]:.4f}",
    "KPI_LP_VS_BASE": f"{pk('baseline', 'lp_r50')[0]:+.4f}",
    "KPI_LP_VS_BASE_P": f"{pk('baseline', 'lp_r50')[3]:.3f}",
    "KPI_TT": f"{pk('tyr_c100', 'tyrK')[0]:+.4f}",
    "KPI_TT_P": f"{pk('tyr_c100', 'tyrK')[3]:.2f}",
    "KPI_MDE": f"{2.80 * se_tt:.3f}",
    "KPI_PMIN": f"{min(pk(a, b)[3] for a, b in itertools.combinations(COMP, 2)):.2f}",
    "KPI_OV_TT": f"{ov_tt['q'] * 100:.1f}" if ov_tt else "—",
    "KPI_OV_DT": f"{ov_dt['q'] * 100:.1f}" if ov_dt else "—",
    "KPI_COL_TYRK": f"{A['tyrK']['gates']['collision_at_fault'][0]}",
    "KPI_COL_BASE": f"{A['baseline']['gates']['collision_at_fault'][0]}",
    **KPI_COC,
}

html = TPL.read_text()
for name, frag in (("TABLE_ARMS", TABLE_ARMS), ("TABLE_VS_BASE", TABLE_VS_BASE),
                   ("TABLE_METHODS", TABLE_METHODS), ("TABLE_GATES", TABLE_GATES),
                   ("TABLE_COC", TABLE_COC), ("TABLE_POWER", TABLE_POWER)):
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
