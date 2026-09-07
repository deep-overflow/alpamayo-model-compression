"""템플릿의 <!--TABLE_*--> / <!--KPI_*--> 자리를 metrics.json 에서 만든 조각으로 채운다.

arm 이 늘어도(추출을 더 뽑는다면) 같은 명령으로 다시 채우면 되므로 숫자를 손으로 옮기지 않는다.
hard100 리포트 초판에서 손으로 옮긴 값이 한 번 어긋난 적이 있어 이 방식이 규칙이 됐다.

Usage:
  .venv/bin/python experiments/evaluation/fill_calibsize_report.py \
      experiments/evaluation/calibsize_report_template.html /tmp/filled.html \
      [outputs/calibsize_eval/metrics.json]
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TPL, OUT = Path(sys.argv[1]), Path(sys.argv[2])
SRC = Path(sys.argv[3]) if len(sys.argv) > 3 else REPO / "outputs/calibsize_eval/metrics.json"
M = json.loads(SRC.read_text())

LABEL = {"baseline": "baseline <span class='sub'>무압축</span>",
         "calib100": "<code>dual @ calib100</code> <span class='sub'>출하본</span>",
         "st2000_a": "<code>dual @ st2000_a</code>",
         "st2000_b": "<code>dual @ st2000_b</code>"}
CALIB = {"baseline": "&mdash;",
         "calib100": "<code>calib_100</code> &middot; 100 클립",
         "st2000_a": "<code>calib_st4000</code> 앞 2,000",
         "st2000_b": "<code>calib_st4000</code> 뒤 2,000"}
order = [a for a in ("baseline", "calib100", "st2000_a", "st2000_b") if a in M["arms"]]
A = M["arms"]


def pct(x):
    return "&mdash;" if x is None else f"{x * 100:.2f}%"


def sig(p):
    return " <strong>*</strong>" if p["ci_lo"] > 0 or p["ci_hi"] < 0 else ""


rows = []
for a in order:
    e = A[a]
    rows.append(
        f"<tr><td>{LABEL[a]}</td><td>{CALIB[a]}</td>"
        f"<td class='r'><strong>{e['score']:.3f}</strong></td>"
        f"<td class='r sub'>[{e['ci_lo']:.3f}, {e['ci_hi']:.3f}]</td>"
        f"<td class='r'>{e['median']:.3f}</td>"
        f"<td class='r'>{pct(e['coc_degenerate'])}</td></tr>")
TABLE_ARMS = "<table><thead><tr><th>arm</th><th>캘리브레이션</th><th class='r'>scene score</th>" \
             "<th class='r'>95% CI</th><th class='r'>중앙값</th><th class='r'>CoC 퇴화율</th>" \
             "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"

prows = []
for k, p in M["pairs"].items():
    b, a = [s.strip() for s in k.split(" - ")]
    prows.append(
        f"<tr><td><code>{b}</code> &minus; <code>{a}</code></td>"
        f"<td class='r'><strong>{p['delta']:+.4f}</strong></td>"
        f"<td class='r sub'>[{p['ci_lo']:+.4f}, {p['ci_hi']:+.4f}]{sig(p)}</td>"
        f"<td class='r'>{p['wilcoxon_p']:.2g}</td>"
        f"<td class='r'>{p['better']} / {p['worse']} / {p['tie']}</td></tr>")
TABLE_PAIRS = "<table><thead><tr><th>비교</th><th class='r'>평균 차이</th>" \
              "<th class='r'>95% CI</th><th class='r'>Wilcoxon p</th>" \
              "<th class='r'>승 / 패 / 무</th></tr></thead><tbody>" \
              + "".join(prows) + "</tbody></table>"

GATES = ["collision_at_fault", "offroad", "wrong_lane"]
GNAME = {"collision_at_fault": "at-fault 충돌", "offroad": "offroad", "wrong_lane": "wrong lane"}
grows = []
for a in order:
    cells = []
    for g in GATES:
        c, n = A[a]["gates"][g]
        cells.append(f"<td class='r'>{c / n * 100:.1f}% <span class='sub'>({c}/{n})</span></td>")
    grows.append(f"<tr><td>{LABEL[a]}</td>" + "".join(cells) + "</tr>")
TABLE_GATES = "<table><thead><tr><th>arm</th>" \
              + "".join(f"<th class='r'>{GNAME[g]}</th>" for g in GATES) \
              + "</tr></thead><tbody>" + "".join(grows) + "</tbody></table>"

ov = M["selection_overlap"]
slim = [a for a in order if A[a].get("slim")]
orows = []
for i, a in enumerate(slim):
    for b in slim[i + 1:]:
        v = ov[f"{a}|{b}"]
        orows.append(f"<tr><td><code>{a}</code> vs <code>{b}</code></td>"
                     f"<td class='r'>{v['q'] * 100:.1f}%</td>"
                     f"<td class='r'>{v['mlp'] * 100:.1f}%</td></tr>")
TABLE_OVERLAP = "<table><thead><tr><th>비교</th><th class='r'>Q head</th>" \
                "<th class='r'>MLP 채널</th></tr></thead><tbody>" \
                + "".join(orows) + "</tbody></table>"

# 페어 키는 "b - a" 형태다. 없는 쌍을 조용히 0 으로 만들지 않도록 KeyError 를 그대로 낸다.
def d(b, a, field="delta"):
    return M["pairs"][f"{b} - {a}"][field]


KPI = {
    "KPI_N": str(M["n_scenes"]),
    "KPI_CALIB100": f"{A['calib100']['score']:.3f}",
    "KPI_BASE": f"{A['baseline']['score']:.3f}",
    "KPI_DUAL_GAIN": f"{d('calib100', 'baseline'):+.3f}",
}
if "st2000_a" in A:
    KPI["KPI_ST2000A"] = f"{A['st2000_a']['score']:.3f}"
    KPI["KPI_A_VS_C"] = f"{d('st2000_a', 'calib100'):+.3f}"
if "st2000_b" in A:
    KPI["KPI_ST2000B"] = f"{A['st2000_b']['score']:.3f}"
    KPI["KPI_B_VS_C"] = f"{d('st2000_b', 'calib100'):+.3f}"
    KPI["KPI_B_VS_A"] = f"{d('st2000_b', 'st2000_a'):+.3f}"
    KPI["KPI_B_VS_A_P"] = f"{d('st2000_b', 'st2000_a', 'wilcoxon_p'):.2g}"

# 개루프 표는 다른 실험의 산출물에서 읽는다 (calib_size_2x2, 2026-09-06). 여기서도 숫자를
# 손으로 옮기지 않는다 -- 그 파일이 없으면 자리표시자를 비우지 말고 알린다.
OL = REPO / "outputs/calib_size_2x2/metrics.json"
TABLE_OPENLOOP = ""
if OL.exists():
    o = json.loads(OL.read_text())
    A_, H1 = o["absolute"], o["H1_size"]
    SET = {"test": "test 500", "indist": "val 500", "oodval": "OOD"}
    rows = []
    for st, label in SET.items():
        c = A_[f"dual|calib_100|{st}"]
        t = A_[f"dual|st2000|{st}"]
        h = H1[f"dual|{st}"]
        star = " <strong>*</strong>" if h["sig"] else ""
        rows.append(
            f"<tr><td>{label} <span class='sub'>n={h['n']}</span></td>"
            f"<td class='r'>{c['minADE6']:.4f}</td><td class='r'>{t['minADE6']:.4f}</td>"
            f"<td class='r'><strong>{h['median']:+.4f}</strong></td>"
            f"<td class='r sub'>[{h['lo']:+.4f}, {h['hi']:+.4f}]{star}</td>"
            f"<td class='r'>{h['p']:.2g}</td></tr>")
    TABLE_OPENLOOP = (
        "<table><thead><tr><th>평가 세트</th><th class='r'>dual @ calib100</th>"
        "<th class='r'>dual @ st2000</th><th class='r'>차이(중앙값)</th>"
        "<th class='r'>95% CI</th><th class='r'>p</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>")
    dc = o["draw_context"]
    sm = dc["_summary"]
    KPI.update({
        "KPI_N100_MEAN": f"{sm['n100_mean']:.3f}",
        "KPI_N100_SD": f"{sm['n100_sd']:.3f}",
        "KPI_ST2000_OL": f"{dc['st2000']['minADE6']:.3f}",
        "KPI_ST2000_RANK": str(sm["st2000_rank_among_n100"]),
        "KPI_ST2000_VS_MEAN": f"{sm['st2000_minus_n100_mean']:+.3f}",
        "KPI_CALIB_OL": f"{dc['calib_100']['minADE6']:.3f}",
    })
else:
    print(f"경고: {OL} 가 없어 개루프 표를 채우지 못합니다", file=sys.stderr)

html = TPL.read_text()
for name, frag in (("TABLE_ARMS", TABLE_ARMS), ("TABLE_PAIRS", TABLE_PAIRS),
                   ("TABLE_GATES", TABLE_GATES), ("TABLE_OVERLAP", TABLE_OVERLAP),
                   ("TABLE_OPENLOOP", TABLE_OPENLOOP)):
    marker = f"<!--{name}-->"
    if marker not in html:
        print(f"경고: 템플릿에 {marker} 가 없습니다", file=sys.stderr)
    html = html.replace(marker, frag)
for k, v in KPI.items():
    html = html.replace(f"<!--{k}-->", v)

left = [m for m in ("TABLE_", "KPI_") if f"<!--{m}" in html]
if left:
    print(f"경고: 채우지 못한 자리표시자가 남았습니다 ({left})", file=sys.stderr)
OUT.write_text(html)
print(f"-> {OUT}  ({len(html):,} bytes)")
