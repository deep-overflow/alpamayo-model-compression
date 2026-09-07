"""Emit the consolidated alpasim table as a self-contained HTML report.

The other reports in `reports/` are a hand-written template plus `build_report.py`, which
only substitutes `PLOT::` placeholders.  This one is generated instead: it is 25 arm-rows
across 16 columns, and hand-maintaining that markup would guarantee a stale cell the first
time an arm is added.  The prose lives here as module constants so it is still reviewed
and versioned like a template; only the tables and the figure are computed.

    python experiments/head_analysis/alpasim_table_html.py \
        --out outputs/alpasim_table --html reports/evaluation/<date>_alpasim-table.html
"""

import argparse
import base64
import html
import json
import math
from pathlib import Path

from render_alpasim_table import ordered

# column -> (metrics.json key, header, format, "higher is better"? None = neutral)
HEAD = [("params_pct", "제거%", "{:.1f}", None),
        ("score", "score", "{:.3f}", True),
        ("d_score", "Δ vs base", "{:+.3f}", True),
        ("d_p", "p", "{:.3f}", None),
        ("progress_clipped_rel", "progress", "{:.3f}", True),
        ("collision_at_fault", "과실충돌%", "{:.1f}", False),
        ("offroad", "이탈%", "{:.1f}", False)]
SECOND = [("params_pct", "제거%", "{:.1f}", None),
          ("collision_any", "전체충돌%", "{:.1f}", False),
          ("collision_rear", "후미추돌%", "{:.1f}", False),
          ("wrong_lane", "차선이탈%", "{:.1f}", False),
          ("coc_degenerate_frac", "CoC퇴화%", "{:.1f}", False),
          ("dist_to_gt_trajectory", "d2GT m", "{:.2f}", None),
          ("min_distance_to_obstacle_m", "최소장애물 m", "{:.2f}", True),
          ("plan_deviation", "plan편차", "{:.3f}", False),
          ("perfect_pct", "만점%", "{:.1f}", True),
          ("zero_pct", "영점%", "{:.1f}", False),
          ("repeat_abs_diff", "반복잡음", "{:.3f}", None)]

CSS = """
  :root {
    --bg: #FAF9F5; --bg-card: #FFFFFF; --bg-code: #F0EEE6;
    --text: #29261B; --text-muted: #6B6555;
    --accent: #D97757; --accent-dark: #C15F3C;
    --border: #E8E6DC; --table-stripe: #F5F4EF;
    --good: #2C6E2C; --bad: #C15F3C;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: var(--bg); color: var(--text);
    font-family: "Söhne", "Pretendard", "Inter", -apple-system, BlinkMacSystemFont,
                 "Segoe UI", "Malgun Gothic", sans-serif;
    font-size: 16px; line-height: 1.7; padding: 3rem 1.5rem 5rem;
  }
  .container { max-width: 1120px; margin: 0 auto; }
  header { margin-bottom: 2.5rem; border-bottom: 2px solid var(--accent); padding-bottom: 1.5rem; }
  .eyebrow { color: var(--accent-dark); font-size: 0.78rem; font-weight: 600;
             letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.75rem; }
  h1 { font-family: "Tiempos Headline", Georgia, "Times New Roman", serif;
       font-size: 2.0rem; font-weight: 500; line-height: 1.3; margin-bottom: 0.75rem;
       text-wrap: balance; }
  .meta { color: var(--text-muted); font-size: 0.9rem; }
  .meta code { background: none; padding: 0; color: var(--text-muted); }
  h2 { font-family: "Tiempos Headline", Georgia, "Times New Roman", serif;
       font-size: 1.4rem; font-weight: 500; margin: 2.75rem 0 1rem; padding-top: 0.5rem; }
  h2 .num { color: var(--accent); margin-right: 0.4rem; }
  h3 { font-size: 1.02rem; font-weight: 600; margin: 1.75rem 0 0.5rem; }
  p { margin-bottom: 1rem; }
  ul, ol { margin: 0 0 1rem 1.4rem; }
  li { margin-bottom: 0.45rem; }
  code { font-family: "Berkeley Mono", "SF Mono", "Fira Code", Menlo, Consolas, monospace;
         font-size: 0.86em; background: var(--bg-code); padding: 0.12em 0.35em;
         border-radius: 4px; }
  pre { background: var(--bg-code); border-radius: 8px; padding: 1rem 1.25rem;
        overflow-x: auto; margin: 1rem 0 1.5rem; font-size: 0.85rem; line-height: 1.6; }
  pre code { background: none; padding: 0; }
  .scroll { overflow-x: auto; margin: 1.25rem 0 1.75rem;
            border: 1px solid var(--border); border-radius: 8px; background: var(--bg-card); }
  table { width: 100%; border-collapse: collapse; font-size: 0.86rem; }
  th { background: var(--bg-code); text-align: right; font-weight: 600;
       padding: 0.5rem 0.7rem; border-bottom: 2px solid var(--border); white-space: nowrap; }
  th:first-child { text-align: left; }
  td { padding: 0.42rem 0.7rem; border-bottom: 1px solid var(--border);
       text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
  td:first-child { text-align: left; font-family: "Berkeley Mono", "SF Mono", Menlo,
                   Consolas, monospace; font-size: 0.9em; }
  tbody tr:last-child td { border-bottom: none; }
  tr.fam td { background: var(--bg-code); font-weight: 600; color: var(--text-muted);
              font-size: 0.8rem; letter-spacing: 0.04em; text-align: left;
              font-family: inherit; }
  tr.base td { background: #F3F1E8; }
  td.good { color: var(--good); }
  td.bad { color: var(--bad); }
  td.dim { color: var(--text-muted); }
  .callout { background: #FBF0EB; border-left: 4px solid var(--accent);
             border-radius: 0 8px 8px 0; padding: 1rem 1.25rem; margin: 1.5rem 0; }
  .callout strong:first-child { color: var(--accent-dark); }
  .warn { background: #FFF8E6; border-left: 4px solid #eda100;
          border-radius: 0 8px 8px 0; padding: 1rem 1.25rem; margin: 1.5rem 0;
          font-size: 0.95rem; }
  .note { background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px;
          padding: 1rem 1.25rem; margin: 1.5rem 0; color: var(--text-muted);
          font-size: 0.92rem; }
  figure { margin: 1.5rem 0; background: var(--bg-card); border: 1px solid var(--border);
           border-radius: 8px; padding: 1rem; overflow-x: auto; }
  figure img { max-width: 100%; height: auto; display: block; margin: 0 auto; }
  figcaption { color: var(--text-muted); font-size: 0.88rem; margin-top: 0.75rem;
               line-height: 1.55; }
  .kpi-row { display: flex; flex-wrap: wrap; gap: 0.75rem; margin: 1.25rem 0 1.5rem; }
  .kpi { flex: 1 1 150px; background: var(--bg-card); border: 1px solid var(--border);
         border-radius: 8px; padding: 0.8rem 1rem; }
  .kpi .v { font-size: 1.3rem; font-weight: 650; font-variant-numeric: tabular-nums; }
  .kpi .l { color: var(--text-muted); font-size: 0.8rem; margin-top: 0.1rem; }
  @media (max-width: 640px) { body { padding: 1.5rem 1rem 3rem; } h1 { font-size: 1.5rem; } }
"""


def cell(a, key, spec, better, base):
    v = a["params"]["pct"] if key == "params_pct" else a.get(key)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return '<td class="dim">&mdash;</td>'
    cls = ""
    if better is not None and base is not None and key != "params_pct":
        b = base.get(key)
        if isinstance(b, (int, float)) and abs(v - b) > 1e-9:
            cls = " class=\"good\"" if (v > b) == better else " class=\"bad\""
    if key == "d_p" and v < 0.05:
        cls = ' class="good"'
    return f"<td{cls}>{spec.format(v)}</td>"


def table(arms, cols, base_label="baseline"):
    base = arms.get(base_label)
    out = ['<div class="scroll"><table><thead><tr><th>arm</th>']
    out += [f"<th>{html.escape(h)}</th>" for _, h, _, _ in cols]
    out.append("</tr></thead><tbody>")
    for fam, names in ordered(arms):
        out.append(f'<tr class="fam"><td colspan="{len(cols) + 1}">{html.escape(fam)}</td></tr>')
        for n in names:
            a = arms[n]
            row = "".join(cell(a, k, s, b, None if n == base_label else base)
                          for k, _, s, b in cols)
            cls = ' class="base"' if n == base_label else ""
            out.append(f"<tr{cls}><td>{html.escape(n)}</td>{row}</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def pooled_arms(m):
    """`both` with `score_pooled` renamed to `score`, so the pooled set goes through the
    same table() as the two suites and the three read as one shape."""
    return {n: dict(a, score=a["score_pooled"]) for n, a in m["both"]["arms"].items()}


def both_score_table(m):
    """Levels and ranks side by side -- the ranks are the point: they disagree."""
    b = m["both"]
    head = ('<tr><th rowspan="2">arm</th>'
            '<th colspan="2" style="text-align:center">스위트별 (순위)</th>'
            '<th colspan="2" style="text-align:center">합산</th></tr>'
            "<tr><th>150씬</th><th>hard100</th>"
            "<th>pooled  n=250</th><th>macro  50/50</th></tr>")
    out = [f'<div class="scroll"><table><thead>{head}</thead><tbody>']
    for n, a in sorted(b["arms"].items(), key=lambda x: -x[1]["score_pooled"]):
        base = n == "baseline"
        r = a["rank_by_suite"]
        out.append(
            f'<tr{" class=\"base\"" if base else ""}><td>{html.escape(n)}</td>'
            f'<td>{a["by_suite"]["s150"]:.3f} '
            f'<span class="dim">({r["s150"]}위)</span></td>'
            f'<td>{a["by_suite"]["hard100"]:.3f} '
            f'<span class="dim">({r["hard100"]}위)</span></td>'
            f'<td><strong>{a["score_pooled"]:.3f}</strong> '
            f'<span class="dim">[{a["score_ci_lo"]:.3f}, {a["score_ci_hi"]:.3f}]</span></td>'
            f'<td>{a["score_macro"]:.3f}</td></tr>')
    out.append("</tbody></table></div>")
    return "\n".join(out)


def both_delta_table(m):
    """baseline 대비 Δ: per-suite next to pooled, so the reader sees the pooled number is
    not a new claim -- it is the same effect measured with more scenes."""
    b = m["both"]
    head = ('<tr><th rowspan="2">arm</th>'
            '<th colspan="2" style="text-align:center">스위트별 Δ (p)</th>'
            '<th colspan="4" style="text-align:center">합산 Δ</th></tr>'
            "<tr><th>150씬</th><th>hard100</th>"
            "<th>pooled</th><th>95% CI</th><th>p</th><th>승/패</th></tr>")
    out = [f'<div class="scroll"><table><thead>{head}</thead><tbody>']
    for n, a in sorted(b["arms"].items(), key=lambda x: -x[1]["score_pooled"]):
        if n == "baseline":
            continue
        per = "".join(
            f'<td class="{"good" if a["d_by_suite"][s] > 0 else "bad"}">'
            f'{a["d_by_suite"][s]:+.3f} '
            f'<span class="dim">({a["p_by_suite"][s]:.3f})</span></td>'
            for s in ("s150", "hard100"))
        out.append(
            f"<tr><td>{html.escape(n)}</td>{per}"
            f'<td class="{"good" if a["d_score"] > 0 else "bad"}">'
            f'<strong>{a["d_score"]:+.3f}</strong></td>'
            f'<td>[{a["d_lo"]:+.3f}, {a["d_hi"]:+.3f}]</td>'
            f'<td class="{"good" if a["d_p"] < 0.05 else ""}">{a["d_p"]:.4f}</td>'
            f'<td>{a["wins"]}/{a["losses"]}</td></tr>')
    out.append("</tbody></table></div>")
    return "\n".join(out)


def both_pairs_table(m):
    out = [('<div class="scroll"><table><thead><tr><th>쌍 (pooled, n=250)</th>'
            "<th>Δ</th><th>95% CI</th><th>p</th><th>승/패</th><th>판정</th>"
            "</tr></thead><tbody>")]
    for k, v in m["both"]["pairs"].items():
        sig = v["lo"] > 0 or v["hi"] < 0
        y, x = k.split("-", 1)
        out.append(
            f"<tr><td>{html.escape(y)} &minus; {html.escape(x)}</td>"
            f'<td class="{"good" if sig and v["delta"] > 0 else "bad" if sig else ""}">'
            f'{v["delta"]:+.3f}</td>'
            f'<td>[{v["lo"]:+.3f}, {v["hi"]:+.3f}]</td><td>{v["p"]:.4f}</td>'
            f'<td>{v["wins"]}/{v["losses"]}</td>'
            f'<td style="text-align:center">{"구분됨" if sig else "구분 안 됨"}</td></tr>')
    out.append("</tbody></table></div>")
    return "\n".join(out)


# the score's own three inputs, then everything outside it.  Splitting them in the header
# is the whole argument of section 1 made visible in one table.
INPUTS = [("progress_clipped_rel", "progress", "{:.3f}", True),
          ("collision_at_fault", "과실충돌%", "{:.1f}", False),
          ("offroad", "이탈%", "{:.1f}", False)]


def both_metric_table(m):
    arms = m["both"]["arms"]
    base = arms["baseline"]
    outside = [c for c in SECOND if c[0] != "params_pct"]
    head = (f'<tr><th rowspan="2">arm</th>'
            f'<th colspan="{len(INPUTS) + 1}" style="text-align:center">'
            f"점수와 그 입력 &mdash; 새 정보 없음</th>"
            f'<th colspan="{len(outside)}" style="text-align:center">'
            f"점수 밖 &mdash; 여기서만 보이는 것</th></tr><tr><th>score</th>"
            + "".join(f"<th>{html.escape(h)}</th>" for _, h, _, _ in INPUTS + outside)
            + "</tr>")
    out = [f'<div class="scroll"><table><thead>{head}</thead><tbody>']
    for n, a in sorted(arms.items(), key=lambda x: -x[1]["score_pooled"]):
        b = None if n == "baseline" else base
        row = "".join(cell(a, k, s, bt, b) for k, _, s, bt in INPUTS + outside)
        sc = cell(dict(a, score=a["score_pooled"]), "score", "{:.3f}", True,
                  None if b is None else dict(base, score=base["score_pooled"]))
        out.append(f'<tr{" class=\"base\"" if n == "baseline" else ""}>'
                   f"<td>{html.escape(n)}</td>{sc}{row}</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def b64(p):
    return base64.b64encode(Path(p).read_bytes()).decode()


def build(m, out_dir, date):
    s150, hard = m["suites"]["s150"], m["suites"]["hard100"]
    a150, ah = s150["arms"], hard["arms"]
    b = m["both"]
    dead = hard["unscorable_scenes"]
    dead_all = sorted(set.intersection(*(set(v["own_unscorable_scenes"])
                                         for v in ah.values())))
    n_arms = len({*a150, *ah})
    fig = b64(out_dir / "plots" / "suites_and_coc.png")
    r = json.loads((out_dir / "plots" / "coc_score_rho.json").read_text())
    rho, rho_p, rho_n = r["spearman_rho"], r["p"], r["n"]

    p = []
    p.append(f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>alpasim 폐루프 통합 표</title><style>{CSS}</style></head><body><div class="container">
<header>
  <div class="eyebrow">Model Compression &middot; 폐루프 평가</div>
  <h1>alpasim 폐루프 통합 표 &mdash; 150씬과 hard100</h1>
  <div class="meta">
    {date} &middot; arm {n_arms}개 &middot; 150씬 {len(a150)} arm, hard100 {len(ah)} arm
    &middot; 두 스위트 공유 씬 <strong>{m['suite_overlap']}개</strong>
    &middot; 각 씬 2 rollout &middot; 드라이버 전부 Ada
    &middot; 수집 <code>collect_alpasim_table.py</code>
    &middot; 산출물 <code>outputs/alpasim_table/</code>
    (<code>table.csv</code> / <code>metrics.json</code> / <code>summary.txt</code>)
  </div>
</header>

<div class="kpi-row">
  <div class="kpi"><div class="v">{a150['baseline']['score']:.3f}</div>
    <div class="l">150씬 baseline</div></div>
  <div class="kpi"><div class="v">{ah['baseline']['score']:.3f}</div>
    <div class="l">hard100 baseline</div></div>
  <div class="kpi"><div class="v">+{a150['dual_u40_v2']['d_score']:.3f}</div>
    <div class="l">150씬 최고 Δ (dual)</div></div>
  <div class="kpi"><div class="v">+{ah['dual_u40_v2']['d_score']:.3f}</div>
    <div class="l">hard100 최고 Δ (dual)</div></div>
  <div class="kpi"><div class="v">{a150['baseline']['repeat_abs_diff']:.3f}</div>
    <div class="l">150씬 반복 잡음 (baseline)</div></div>
</div>

<h2><span class="num">1.</span>먼저: 네 지표는 서로 독립이 아니다</h2>
<p>alpasim이 씬 점수를 만드는 규칙은 실행 결과에 그대로 기록되어 있다
(<code>aggregate/results-summary.json</code>의 <code>score_criteria</code>).</p>
<pre><code>score = 0                                    if (collision_at_fault or offroad)
      = min(clamp(progress_clipped_rel,0,1) / 0.8, 1.0)   otherwise
      = 1.0                                  if gt_dist_traveled_m &lt; 5 m</code></pre>
<div class="callout">
  <p><strong>즉 score / 과실충돌 / 이탈 / progress는 <em>하나의 양과 그 세 입력</em>이다.</strong>
  네 개를 같이 두는 건 &ldquo;왜 그 점수가 나왔나&rdquo;를 분해해 주므로 좋은 선택이지만,
  점수가 모르는 것은 여전히 아무것도 알려주지 않는다. 새 정보는 <strong>점수 밖</strong>에서만 나온다
  &mdash; 아래 표 2가 그것이다.</p>
</div>
<p>검산도 여기서 나온다. 게이트 실패율(<code>offroad_or_collision_at_fault</code>)은
0점 rollout 비율과 정확히 같아야 하고, 150씬에서는 21 arm 전부 일치한다.</p>

<h2><span class="num">2.</span>주요 지표 &mdash; score / progress / 과실충돌 / 이탈</h2>
<p>세 집합을 같은 모양의 표로 싣는다. 열은 어디서나 동일하므로 한 arm을 세 표에서 세로로
읽으면 된다. 색은 baseline 대비 &mdash; 좋아지면 초록, 나빠지면 주황.
<code>p</code>는 씬 단위 페어드 Wilcoxon(양측)이고 0.05 미만이면 초록,
Δ는 씬별 평균 점수의 페어드 차이, <code>제거%</code>는 전체 11.08B 대비다.</p>

<h3>2.1 &nbsp;150씬 &mdash; <code>public_2601</code> 앞 150개, {len(a150)}개 arm</h3>
{table(a150, HEAD)}

<h3>2.2 &nbsp;hard100 &mdash; 난이도 80&ndash;100 백분위, 150씬과 교집합 0, {len(ah)}개 arm</h3>
{table(ah, HEAD)}

<h3>2.3 &nbsp;합산 (pooled, n={b['n_scenes']}씬) &mdash; 두 집합에 다 있는 {len(b['arms'])}개 arm</h3>
{table(pooled_arms(m), HEAD)}
<p class="note">합산 표의 <code>score</code>는 <strong>pooled</strong>, 즉 250씬 각각을 한 번씩
센 평균이다(150씬이 60% 가중). 두 스위트 평균의 단순평균인 <strong>macro</strong>와
스위트별 순위, 방법 간 쌍 비교는 4절에 있다.</p>

<h2><span class="num">3.</span>점수가 보지 못하는 지표</h2>
<p>2절의 네 열은 서로 독립이 아니다(1절). 여기가 새 정보다.
전체충돌 = 과실 여부 무관 모든 충돌, 후미추돌 = 뒤에서 받힌 경우.
CoC퇴화 = 빈 출력 또는 soup(비ASCII&gt;5% / 고유어비&lt;0.5 / 300자 초과), 임계값은
<code>analyze_alpasim.coc_stats</code>와 동일. d2GT = GT 궤적까지 거리.
반복잡음 = 같은 씬 두 rollout 점수차의 절댓값 평균.</p>

<h3>3.1 &nbsp;150씬</h3>
{table(a150, SECOND)}

<h3>3.2 &nbsp;hard100</h3>
{table(ah, SECOND)}

<h3>3.3 &nbsp;합산 (pooled, rollout {b['arms']['baseline']['n_rollouts']}개)</h3>
{table(pooled_arms(m), SECOND)}
<p class="note">CoC 퇴화율만은 rollout 수가 아니라 <strong>실제로 텍스트를 낸 rollout 수</strong>로
가중평균했다 &mdash; hard100에서 arm마다 4개가 CoC를 하나도 남기지 않았기 때문에
전체 rollout으로 나누면 그만큼 희석된다.</p>

<figure>
  <img src="data:image/png;base64,{fig}" alt="두 스위트 점수 대응과 arm별 CoC 퇴화율">
  <figcaption>왼쪽: 두 스위트에서 모두 측정된 4개 arm. 점선은 y=x &mdash; 넷 다 아래에 있으므로
  hard100이 실제로 더 어렵고, arm 사이 간격은 대체로 보존된다.
  오른쪽: 150씬 21개 arm의 CoC 퇴화율(막대)과 점수(막대 옆 숫자). 파랑은 baseline 이상,
  분홍은 baseline 미만. <strong>두 축의 순서가 서로 무관하다</strong>는 것이 요점이다 &mdash;
  Spearman &rho;={rho:+.2f} (p={rho_p:.2f}, n={rho_n}).
  <code>lp_r50_dual</code>은 CoC의 {a150['lp_r50_dual']['coc_degenerate_frac']:.1f}%가 퇴화했는데
  점수는 {a150['lp_r50_dual']['score']:.3f}로 상위권이고,
  <code>j_u40_v2</code>는 퇴화가 {a150['j_u40_v2']['coc_degenerate_frac']:.1f}%로 낮은데
  점수는 전체 최하위({a150['j_u40_v2']['score']:.3f})다.
  <code>recover_dual_u55</code>의 0.0%는 LoRA 회복이 실제로 추론을 되돌렸다는 유일한 증거다.
  </figcaption>
</figure>

<h2><span class="num">4.</span>합산을 어떻게 읽나</h2>
<p>네 arm(<code>baseline</code>, <code>dual_u40_v2</code>, <code>tyr_u40_r</code>,
<code>lp_r50</code>)만 두 스위트에 다 있다. 두 집합은 서로소이므로 합치는 건 그냥 씬 250개의
평균이지만, 가중을 주는 방식이 둘이고 <strong>서로 다른 질문에 답한다</strong>.</p>
<ul>
  <li><strong>pooled</strong> (n=250) &mdash; 씬 하나가 한 번씩. 150씬이 60% 가중을 갖는다.
  가장 검정력이 높은 추정이고, 아래 Δ &middot; CI &middot; Wilcoxon은 전부 이걸로 계산했다.</li>
  <li><strong>macro</strong> &mdash; 두 스위트 평균의 단순평균. 쉬움과 어려움이 50/50.
  어느 쪽도 <code>public_2601</code> 전체의 표본이 아니므로(하나는 쉬운 앞부분, 하나는
  난이도 80&ndash;100% 구간) &ldquo;난이도 구간별로 한 숫자씩&rdquo;이라는 읽기다.</li>
</ul>
<p class="note">둘 다 913씬 전체 스위트의 점수 추정치가 <em>아니다</em>. 숫자를 인용할 때
어느 쪽인지 밝혀야 한다.</p>

<h3>4.1 점수와 서열</h3>
{both_score_table(m)}
<p class="note">순위는 두 스위트 사이에서 <strong>바뀐다</strong> &mdash; 150씬에서는
2위 <code>lp_r50</code> / 3위 <code>tyr_u40_r</code>인데 hard100에서는 뒤집힌다.
pooled 옆 대괄호는 부트스트랩 95% CI(씬 단위 재표본, 10,000회)이고,
네 arm의 CI가 서로 크게 겹친다는 점을 4.3이 검정으로 확인한다.</p>

<h3>4.2 baseline 대비 Δ</h3>
{both_delta_table(m)}
<div class="callout">
  <p><strong>세 압축 arm 모두 무압축을 이긴다.</strong> n=250에서 Δ가 각각
  {b['arms']['dual_u40_v2']['d_score']:+.3f} / {b['arms']['lp_r50']['d_score']:+.3f} /
  {b['arms']['tyr_u40_r']['d_score']:+.3f}이고 세 CI 모두 0을 포함하지 않는다.
  스위트별 Δ와 방향·크기가 같으므로 pooled는 새 주장이 아니라 같은 효과를 씬을 더 써서
  잰 것이다.</p>
</div>

<h3>4.3 방법끼리는 갈리는가</h3>
{both_pairs_table(m)}
<div class="callout">
  <p><strong>어느 쌍도 갈리지 않는다.</strong> hard100만의 검정력 문제가 아니었다 &mdash;
  이 데이터로 낼 수 있는 최대 표본인 250씬에서도 세 쌍 전부 CI가 0을 가로지르고, 승/패도
  반반에 가깝다. 150씬의 서열(dual 0.828 &gt; lp_r50 0.810 &gt; tyr 0.786)은 애초에
  해상도 밖이었다는 뜻이다. 가장 가까운 쌍조차
  <code>tyr_u40_r</code>&minus;<code>dual_u40_v2</code>의
  [{b['pairs']['tyr_u40_r-dual_u40_v2']['lo']:+.3f},
  {b['pairs']['tyr_u40_r-dual_u40_v2']['hi']:+.3f}]로 0을 넘는다.</p>
</div>

<h3>4.4 점수가 못 보는 축에서는 갈린다</h3>
<p>종합 점수로 세 방법이 동률이라고 해서 셋이 같은 모델이라는 뜻은 아니다.
표 3.3에서 <code>lp_r50</code>은 종합 점수가 <code>tyr_u40_r</code>보다 높은데
전체 충돌이 {b['arms']['lp_r50']['collision_any']:.1f}% 대
{b['arms']['tyr_u40_r']['collision_any']:.1f}%, 후미추돌이
{b['arms']['lp_r50']['collision_rear']:.1f}% 대
{b['arms']['tyr_u40_r']['collision_rear']:.1f}%로 오히려 나쁘다 &mdash;
점수는 <em>과실</em> 충돌만 세므로 과실 열만 보면 안 보이는 차이다.
그리고 세 압축 arm 전부 baseline보다 반복 잡음이
작다({b['arms']['baseline']['repeat_abs_diff']:.3f} 대
{min(v['repeat_abs_diff'] for k, v in b['arms'].items() if k != 'baseline'):.3f}&ndash;
{max(v['repeat_abs_diff'] for k, v in b['arms'].items() if k != 'baseline'):.3f}),
즉 압축이 폐루프 거동을 더 재현 가능하게 만든다.</p>
<h3>4.5 지표 전체 한 장에 (pooled)</h3>
{both_metric_table(m)}
<p class="note">2.3과 3.3을 한 표로 붙인 것. 왼쪽 네 칸이 1절의 항등식(score와 그 세 입력),
오른쪽이 새 정보다.</p>

<h2><span class="num">5.</span>읽을 때 조심할 것</h2>
<div class="warn">
  <p><strong>1. 절대값은 낙관적이다.</strong> 150씬은 <code>public_2601</code>의
  scene_id 정렬 앞 150개인데, 실측상 나머지 763씬보다 쉽다(0.742 대 0.660, Mann&ndash;Whitney
  p=0.039; 과실충돌 2.0% 대 5.5%). 페어드 Δ는 거의 영향이 없지만
  <em>절대</em> 충돌률·이탈률을 인용할 때는 이 사실을 같이 말해야 한다.</p>
  <p><strong>2. hard100에는 채점 불가 씬이 {len(dead)}개 있다.</strong>
  {', '.join('<code>' + s[7:15] + '</code>' for s in dead)} 가
  <code>route folds back on itself with angle ~180 deg</code>로 실패해 metric을 하나도 남기지 않고
  0점 처리된다. 두 rollout에서 waypoint 번호와 각도까지 같으므로 모델의 확률적 rollout이 아니라
  <strong>맵 쪽 결함</strong>이다. 그런데 완전히 페어드는 아니다 &mdash;
  {', '.join('<code>' + s[7:15] + '</code>' for s in dead_all)}는 네 arm 전부가 걸리지만
  나머지 2개는 <code>dual_u40_v2</code>만 걸린다. 결함 구간에 도달하는지가 얼마나 멀리
  주행했는지에 달려 있기 때문이다. 4개를 전 arm에서 빼면(n={hard['n_scenes_clean']})
  baseline {ah['baseline']['score']:.3f} &rarr; {ah['baseline']['score_clean']:.3f},
  dual의 Δ는 {ah['dual_u40_v2']['d_score']:+.3f} &rarr;
  {ah['dual_u40_v2']['d_score_clean']:+.3f}로 <em>커진다</em>. 즉 발표된 +0.085는 보수적인 값이고
  결론은 바뀌지 않는다. 150씬에는 이런 씬이 없다.</p>
  <p><strong>3. Δ를 반복 잡음과 비교하라.</strong> 같은 씬을 두 번 돌린 점수차가 평균
  {a150['baseline']['repeat_abs_diff']:.3f}(150씬 baseline)&ndash;{ah['baseline']['repeat_abs_diff']:.3f}
  (hard100)이다. hard100에서 dual&ndash;tyr 차이는 0.010으로 잡음의 1/7이며, 실제로 검정에서
  구분되지 않는다.</p>
  <p><strong>4. 충돌·이탈률의 Fisher p는 선별용이다.</strong> 한 씬의 두 rollout은 독립이 아니므로
  <code>table.csv</code>의 <code>*_p</code>는 눈에 띄는 칸을 고르는 데만 쓰고,
  그것만으로 유의성을 주장하면 안 된다.</p>
  <p><strong>5. <code>progress_rel</code>은 쓰지 말 것.</strong> 표에 쓴 건
  <code>progress_clipped_rel</code>이다. 날것의 <code>progress_rel</code>은 최악 arm
  <code>j_u40_v2</code>에서 0.893으로 <em>가장 높다</em> &mdash; 살아남은 rollout만 평균하기 때문이다.</p>
</div>

<h2><span class="num">6.</span>추가로 넣을 만한 지표</h2>
<p>아래는 실제로 이 데이터에서 arm을 갈라놓는지 확인한 뒤 고른 것이다.</p>
""")

    rec = [
        ("전체충돌 % + 후미추돌 %", "필수",
         (f"점수는 <em>과실</em> 충돌만 본다. 150씬 baseline은 과실 "
          f"{a150['baseline']['collision_at_fault']:.1f}%지만 전체 충돌은 "
          f"{a150['baseline']['collision_any']:.1f}%이고 그중 "
          f"{a150['baseline']['collision_rear']:.1f}%p가 후미추돌이다 &mdash; 충돌의 3분의 2가 "
          f"점수에도 과실 열에도 안 보인다. 그리고 갈라놓는다: "
          f"<code>dual</code> {a150['dual_u40_v2']['collision_any']:.1f}% 대 "
          f"<code>lp_r50</code> {a150['lp_r50']['collision_any']:.1f}%."
         )),
        ("CoC 퇴화 %", "필수",
         (f"이 연구가 자르는 건 추론 타워인데 점수는 그걸 못 본다. "
          f"baseline {a150['baseline']['coc_degenerate_frac']:.1f}% &rarr; "
          f"<code>tyr_u40_r</code> {a150['tyr_u40_r']['coc_degenerate_frac']:.1f}% &rarr; "
          f"<code>wanda_u40_v2</code> {a150['wanda_u40_v2']['coc_degenerate_frac']:.1f}%인데 "
          f"tyr의 점수(0.786)는 dual(0.828)과 크게 다르지 않다. "
          f"<code>recover_dual_u55</code>는 0.0%로 회복이 실제로 작동했음을 이 열에서만 확인할 수 있다."
         )),
        ("제거 파라미터 %", "필수",
         ("점수표만으로는 해석이 불가능하다. 이 표의 arm은 4.8%부터 39.4%까지 걸쳐 있고, "
          "24.0%(2,657,452,032개 제거)에 13개 arm이 몰려 있어 그 구간만이 서로 직접 비교된다."
         )),
        ("차선이탈 %", "권장",
         (f"이탈(offroad)보다 4배 흔해 검정력이 훨씬 높다 &mdash; 150씬에서 이탈은 "
          f"{min(v['offroad'] for v in a150.values()):.1f}&ndash;"
          f"{max(v['offroad'] for v in a150.values()):.1f}%인데 차선이탈은 "
          f"{min(v['wrong_lane'] for v in a150.values()):.1f}&ndash;"
          f"{max(v['wrong_lane'] for v in a150.values()):.1f}%다. "
          f"그리고 이탈이 못 가르는 arm을 가른다."
         )),
        ("반복 잡음", "권장",
         ("alpasim에는 시드 통제가 없고 폐루프는 첫 행동이 갈리는 순간 발산하므로, 같은 씬 두 "
          "rollout의 점수차가 모든 Δ가 올라앉은 바닥이다. 이 열이 없으면 hard100의 서열 소멸을 "
          "&ldquo;차이가 없다&rdquo;로 읽어야 할지 &ldquo;못 재고 있다&rdquo;로 읽어야 할지 알 수 없다."
         )),
        ("만점 % / 영점 %", "권장",
         (f"점수 분포가 이봉형이다 &mdash; 150씬 baseline rollout의 "
          f"{a150['baseline']['perfect_pct']:.1f}%가 정확히 1.0이고 "
          f"{a150['baseline']['zero_pct']:.1f}%가 0이다. 평균만 보면 어느 쪽 끝이 움직였는지 알 수 없다. "
          f"덤으로 영점%는 게이트 실패율과 같아야 하므로 표의 자체 검산이 된다."
         )),
        ("d2GT (GT 궤적까지 거리)", "선택",
         (f"점수에 안 들어가는 &lsquo;사람 궤적과 얼마나 닮았나&rsquo; 축. 흥미로운 해리가 있다: "
          f"<code>dual</code>은 baseline보다 점수가 높은데"
          f"({a150['dual_u40_v2']['score']:.3f} 대 {a150['baseline']['score']:.3f}) "
          f"GT에서는 더 멀다({a150['dual_u40_v2']['dist_to_gt_trajectory']:.2f} 대 "
          f"{a150['baseline']['dist_to_gt_trajectory']:.2f} m). 점수는 모방이 아니라 진행+안전을 보상한다."
         )),
        ("최소 장애물 거리", "선택",
         (f"과실충돌은 150씬에서 2&ndash;17% 사건이라 n=150에서 검정력이 거의 없다 &mdash; "
          f"<code>analyze_longitudinal.py</code>가 존재하는 이유다. 연속 대리 지표로 쓰면 "
          f"<code>dual</code> {a150['dual_u40_v2']['min_distance_to_obstacle_m']:.2f} m 대 baseline "
          f"{a150['baseline']['min_distance_to_obstacle_m']:.2f} m처럼 방향이 보인다."
         )),
        ("plan 편차", "선택",
         (f"플래너 안정성. 점수와 역방향으로 잘 따라간다 &mdash; 최악 "
          f"<code>j_u40_v2</code> {a150['j_u40_v2']['plan_deviation']:.3f}, 최선 "
          f"<code>dualr_wl_u40</code> {a150['dualr_wl_u40']['plan_deviation']:.3f}."
         )),
        ("<s>safety_monitor_triggered</s>", "제외",
         ("25개 arm 전부 정확히 0.0이다. 이 설정에서는 아무것도 재지 않는다."
         )),
    ]
    p.append('<div class="scroll"><table><thead><tr><th>지표</th><th>등급</th>'
             '<th style="text-align:left">근거</th></tr></thead><tbody>')
    for name, grade, why in rec:
        p.append(f'<tr><td style="font-family:inherit">{name}</td>'
                 f'<td style="text-align:center">{grade}</td>'
                 f'<td style="text-align:left;white-space:normal;font-variant-numeric:normal">'
                 f'{why}</td></tr>')
    p.append("</tbody></table></div>")

    p.append(f"""
<h2><span class="num">7.</span>재현</h2>
<pre><code>cd /home/cvlab21/project/chan/alpasim &amp;&amp; uv run python \\
    $REPO/experiments/head_analysis/collect_alpasim_table.py \\
    --out $REPO/outputs/alpasim_table --workers 12
$REPO/.venv/bin/python $REPO/experiments/head_analysis/render_alpasim_table.py \\
    --out $REPO/outputs/alpasim_table
$REPO/.venv/bin/python $REPO/experiments/head_analysis/alpasim_table_html.py \\
    --out $REPO/outputs/alpasim_table --html reports/evaluation/{date}_alpasim-table.html</code></pre>
<p class="note">수집기는 디렉터리 모양으로 실행을 찾는다 &mdash;
<code>alpasim-runs/m2601_merged_*</code>,
<code>alpasim-runs/h100_merged_*</code>,
<code>fmp_G_default_r40_merged</code>, 그리고 soowon의
<code>cl150_merged_*</code>. 새 실행을 병합해 두면 레지스트리를 고칠 필요 없이 표에 나타난다.
CoC는 rollout마다 ASL 로그를 파싱해서 다시 계산한다(캐시 재사용 아님):
검증으로 baseline 0.55% / dual 2.72% / tyr 5.92%가
<code>outputs/alpasim_tyr_2601</code>의 기존 값과 정확히 일치한다.</p>
</div></body></html>""")
    return "\n".join(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--html", type=Path, required=True)
    ap.add_argument("--date", default="2026-09-07")
    args = ap.parse_args()
    m = json.loads((args.out / "metrics.json").read_text())
    args.html.parent.mkdir(parents=True, exist_ok=True)
    args.html.write_text(build(m, args.out, args.date))
    print(f"wrote {args.html}  ({args.html.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
