"""템플릿의 <!--TABLE_*--> 자리에 metrics.json 에서 만든 표 조각을 넣는다.

arm 이 늘어도(예: lp_r50 합류) 같은 명령으로 다시 채우면 되므로, 숫자를 손으로 옮기지 않는다.
"""
import json
import sys
from pathlib import Path

TPL = Path(sys.argv[1])
OUT = Path(sys.argv[2])
M = json.loads(Path("/home/cvlab21/project/chan/alpamayo-model-compression/"
                    "outputs/hard100_eval/metrics.json").read_text())

LABEL = {"baseline": "baseline <span class='sub'>비압축</span>",
         "dual": "<code>dual_u40_v2</code>", "tyr_r": "<code>tyr_u40_r</code>",
         "lp_r50": "<code>lp_r50</code> <span class='sub'>LLM-Pruner</span>"}
BUDGET = {"baseline": "—", "dual": "24.0%", "tyr_r": "24.0%", "lp_r50": "25.0%"}
KIND = {"baseline": "—", "dual": "Taylor 선택", "tyr_r": "출력 재구성",
        "lp_r50": "외부 baseline"}
order = [a for a in ("baseline", "dual", "tyr_r", "lp_r50") if a in M["arms"]]

arms = []
for a in order:
    e = M["arms"][a]
    ref = f"{e['ref_150']:.3f}" if e.get("ref_150") else "—"
    arms.append(
        f"<tr><td>{LABEL[a]}</td><td class='r'>{BUDGET[a]}</td><td>{KIND[a]}</td>"
        f"<td class='r'><strong>{e['mean']:.3f}</strong></td>"
        f"<td class='r'>{e['median']:.3f}</td><td class='r'>{e['below_0.7']}</td>"
        f"<td class='r'>{e['perfect']}</td>"
        f"<td class='r'>{100 * e['offroad']['n'] / e['offroad']['N']:.1f}%</td>"
        f"<td class='r'>{e['collision_at_fault']['n']}/{e['collision_at_fault']['N']}</td>"
        f"<td class='r sub'>{ref}</td></tr>")

pairs = []
for k, p in M["pairs"].items():
    x, y = k.split("_vs_")
    af = p["at_fault"]
    sig = "" if p["wilcoxon_p"] >= 0.05 else " <strong>*</strong>"
    pairs.append(
        f"<tr><td><code>{y}</code> − <code>{x}</code></td>"
        f"<td class='r'>{p['delta']:+.3f} ± {p['se']:.3f}</td>"
        f"<td class='r'>{p['wilcoxon_p']:.4f}{sig}</td>"
        f"<td class='r'>{p['better']} / {p['worse']} / {p['tie']}</td>"
        f"<td class='r'>{af[x]} → {af[y]}</td><td class='r'>{af['odds_ratio']:.2f}</td>"
        f"<td class='r'>{af['fisher_p']:.3f}</td></tr>")

v = M["vs_150"]
off, af = v["offroad"], v["collision_at_fault"]
row_score = (f"<tr><td>비압축 baseline 점수</td><td class='r'>{v['score_150']:.3f}</td>"
             f"<td class='r'><strong>{v['score_hard100']:.3f}</strong></td>"
             f"<td class='r'>{v['delta']:+.3f}</td></tr>")
row_off = (f"<tr><td>도로 이탈률</td><td class='r'>{off['pct_150']:.1f}%</td>"
           f"<td class='r'><strong>{off['pct_hard100']:.1f}%</strong></td>"
           f"<td class='r'>{off['pct_hard100'] / off['pct_150']:.1f}배</td></tr>")
row_af = (f"<tr><td>과실 충돌률</td><td class='r'>{af['pct_150']:.1f}%</td>"
          f"<td class='r'>{af['pct_hard100']:.1f}%</td>"
          f"<td class='r'>{af['pct_hard100'] / af['pct_150']:.2f}배</td></tr>")
vs150 = [row_score, row_off, row_af]

html = TPL.read_text()
for marker, rows in [("<!--TABLE_ARMS-->", arms), ("<!--TABLE_PAIRS-->", pairs),
                     ("<!--TABLE_VS150-->", vs150)]:
    if marker not in html:
        sys.exit(f"템플릿에 {marker} 가 없습니다")
    html = html.replace(marker, "\n".join(rows))
OUT.write_text(html)
print(f"arm {len(order)}개 ({', '.join(order)}), 쌍 {len(pairs)}개 -> {OUT}")
