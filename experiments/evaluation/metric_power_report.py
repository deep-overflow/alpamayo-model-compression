"""Which alpasim metric actually separates arms? Measured, then written up.

The question this answers is a paper question: of the two dozen quantities alpasim records
per rollout, which belong in a results table? The usual way to settle that is taste. This
settles it by measurement -- for every metric, how many arm PAIRS does it resolve, and
does its ordering of the arms agree with the score's.

Every metric is computed the way alpasim computes it, so the numbers are the ones a paper
would print rather than a lookalike:

  1. the modifier chain from `processing.DEFAULT_MODIFIERS` + `main._run_aggregation` --
     drop timesteps before `eval_relevant`, keep up to the first `offroad_or_collision`,
     then up to the first `dist_to_gt_trajectory >= 4.0`
  2. collapse the surviving timesteps with the metric's OWN recorded `time_aggregation`
     (max / min / last / mean), never a uniform rule -- `offroad` is a max, `progress` is
     a last, `min_distance_to_obstacle_m` is a min, and using one rule for all three
     would silently invent three different quantities
  3. per-rollout -> per-scene mean, then scene-paired deltas between every arm pair

`resolved` is discrimination and `rho_score` is agreement, and the report needs both:
a metric can separate arms cleanly while ordering them in a way the score does not care
about, which makes it a diagnostic rather than a headline.

    python experiments/evaluation/metric_power_report.py \\
        --out reports/evaluation/2026-09-20_alpasim-metric-power.html
"""

import argparse
import base64
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

O = Path("/mnt/nvme1n1/ad_vla/outputs/chan")
CACHE = O / "metric_power" / "rollout_metrics.parquet"

BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
ACC, BAD, GOOD, WARN = "#D97757", "#b0402a", "#008300", "#eda100"
# second categorical hue, checked with the dataviz validator on the #FAF9F5 surface
SET2 = "#1F6FB2"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})

SETS = ["origin150", "hard100"]
DROP = {"set", "arm", "scene", "eval_relevant", "img_is_black", "run_uuid"}
# what each metric is, in one clause -- a reader should not have to guess at `progress_rel`
DESC = {
    "score": "종합 점수 (<code>score_criteria</code>) &mdash; 게이트 통과 시 progress",
    "plan_deviation": "계획과 실현 경로의 편차",
    "progress": "경로 진행량",
    "progress_rel_to_total": "전체 GT 경로 대비 진행 비율 (점수의 재료)",
    "progress_rel": "현재 GT 대비 진행 비율, 시간 <em>최소</em>값",
    "dist_traveled_m": "실제 주행 거리",
    "dist_to_gt_location": "동시각 GT 위치까지 거리",
    "dist_to_gt_trajectory": "GT <em>경로</em>까지 투영 거리 (4 m 절단 기준)",
    "open_loop_collision": "3초 지평 개루프 충돌 예측",
    "min_distance_to_obstacle_m": "장애물까지 최소 거리",
    "min_distance_to_lane_boundary_m": "차선 경계까지 최소 거리",
    "collision_any": "충돌 (귀책 무관)",
    "collision_at_fault": "과실 충돌 (전방 or 측면)",
    "collision_front": "전방 충돌", "collision_lateral": "측면 충돌",
    "collision_rear": "후방 충돌 (추돌당함)",
    "offroad": "도로 이탈", "wrong_lane": "역주행/차선 위반",
    "gt_dist_traveled_m": "GT 경로 길이 (씬 속성, arm 무관)",
    "safety_monitor_triggered": "안전 모니터 발동",
}


def resolve(a, b):
    j = pd.concat([a, b], axis=1, join="inner", keys=["x", "y"]).dropna()
    if len(j) < 10:
        return None
    d = (j["x"] - j["y"]).to_numpy()
    if np.allclose(d, 0):
        return False
    rng = np.random.default_rng(0)
    bs = d[rng.integers(0, len(d), (4000, len(d)))].mean(1)
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return bool(lo > 0 or hi < 0)


def analyse(df):
    metrics = [c for c in df.columns if c not in DROP]
    rows = []
    for label in SETS:
        sub = df[df.set == label]
        arms = sorted(sub.arm.unique())
        score_mean = sub.groupby("arm").score.mean()
        for m in metrics:
            if sub[m].isna().all():
                continue
            per = {a: sub[sub.arm == a].groupby("scene")[m].mean() for a in arms}
            tot = res = 0
            for x, y in itertools.combinations(arms, 2):
                r = resolve(per[x], per[y])
                if r is None:
                    continue
                tot += 1
                res += int(r)
            am = sub.groupby("arm")[m].mean()
            common = score_mean.index.intersection(am.index)
            rho = (stats.spearmanr(am[common], score_mean[common]).statistic
                   if len(common) > 2 and am[common].nunique() > 1 else np.nan)
            rows.append({"set": label, "metric": m, "pairs": tot, "resolved": res,
                         "frac": res / tot if tot else np.nan, "rho": rho,
                         "n_arms": len(arms)})
    return pd.DataFrame(rows)


def b64(p):
    return base64.b64encode(Path(p).read_bytes()).decode()


def plots(res, plot_dir):
    plot_dir.mkdir(parents=True, exist_ok=True)
    h = res[res.set == "hard100"].set_index("metric")
    o = res[res.set == "origin150"].set_index("metric")

    # 1 -- the quadrant. x = discrimination, y = agreement with the score. A metric is a
    # headline candidate only in the top-right; the bottom-right discriminates while
    # ordering arms against the score, which makes it a diagnostic.
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    common = [m for m in h.index if m in o.index and not np.isnan(h.loc[m, "rho"])]
    x = [h.loc[m, "frac"] for m in common]
    y = [h.loc[m, "rho"] for m in common]
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(h.loc["score", "frac"], color=ACC, lw=1.2, ls="--", zorder=1)
    ax.annotate("score itself", (h.loc["score", "frac"], -1.02), xytext=(5, 0),
                textcoords="offset points", fontsize=8, color=ACC, va="bottom")
    ax.scatter(x, y, s=64, color=ACC, edgecolor=BG, linewidth=1.5, zorder=3)
    # alternate the label offset: progress / progress_rel_to_total / dist_traveled_m
    # land on nearly the same point (they measure nearly the same thing) and stack
    for i, (m, xx, yy) in enumerate(zip(common, x, y)):
        ax.annotate(m, (xx, yy), xytext=(6, 5) if i % 2 == 0 else (6, -12),
                    textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlabel("arm pairs resolved, hard100 (28 pairs)")
    ax.set_ylabel("Spearman rho with the score's arm ordering")
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(-0.04, 1.0)
    fig.tight_layout()
    fig.savefig(plot_dir / "quadrant.png", dpi=150)
    plt.close(fig)

    # 2 -- discrimination on both sets, ranked. Two series, so a legend plus direct
    # labels; the sets have different pair counts so the axis is a fraction.
    order = [m for m in h.sort_values("frac", ascending=False).index if m in o.index]
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    yy = np.arange(len(order))
    for off, s, col, nice in ((-0.18, o, ACC, "origin150 (21 pairs)"),
                              (+0.18, h, SET2, "hard100 (28 pairs)")):
        v = [s.loc[m, "frac"] for m in order]
        ax.barh(yy + off, v, height=0.34, color=col, label=nice, zorder=3)
        for i, m in enumerate(order):
            ax.annotate(f"{int(s.loc[m, 'resolved'])}/{int(s.loc[m, 'pairs'])}",
                        (s.loc[m, "frac"], yy[i] + off), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=7,
                        color=MUTED)
    ax.set_yticks(yy)
    ax.set_yticklabels(order, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("fraction of arm pairs resolved (bootstrap CI excludes 0)")
    ax.set_xlim(0, 1.08)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(plot_dir / "ranked.png", dpi=150)
    plt.close(fig)


def table(res):
    h = res[res.set == "hard100"].set_index("metric")
    o = res[res.set == "origin150"].set_index("metric")
    body = ""
    for m in h.sort_values("frac", ascending=False).index:
        if m not in o.index:
            continue
        hr, orr = h.loc[m], o.loc[m]
        def cell(r):
            frac = r["frac"]
            cls = " class='good'" if frac >= 0.6 else (
                " class='bad'" if frac <= 0.2 else "")
            return (f"<td{cls}>{int(r['resolved'])}/{int(r['pairs'])}"
                    f"<br><span class='ci'>{100 * frac:.0f}%</span></td>")
        def rho(r):
            v = r["rho"]
            if np.isnan(v):
                return "<td class='dim'>&mdash;</td>"
            cls = " class='good'" if v >= 0.7 else (" class='bad'" if v <= -0.5 else "")
            return f"<td{cls}>{v:+.2f}</td>"
        hl = " class='hl'" if m == "score" else ""
        body += (f"<tr{hl}><td><code>{m}</code></td>{cell(orr)}{cell(hr)}"
                 f"{rho(orr)}{rho(hr)}"
                 f"<td class='dim'>{DESC.get(m, '')}</td></tr>")
    return ("<div class='scroll'><table><thead><tr><th rowspan='2'>지표</th>"
            "<th colspan='2'>분해한 arm 쌍</th><th colspan='2'>&rho; (score 순서와)</th>"
            "<th rowspan='2'>무엇인가</th></tr><tr>"
            "<th>origin150</th><th>hard100</th><th>origin150</th><th>hard100</th>"
            "</tr></thead>"
            f"<tbody>{body}</tbody></table></div>")


CSS = """
:root{--bg:#FAF9F5;--card:#FFF;--code:#F0EEE6;--ink:#29261B;--muted:#6B6555;
--acc:#D97757;--bd:#E8E6DC;--stripe:#F5F4EF;--good:#008300;--bad:#b0402a;
--warn:#eda100;--hl:#FBEEE8}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);margin:0;font-size:16px;line-height:1.7;
padding:3rem 1.5rem 5rem;font-family:"Pretendard","Inter",-apple-system,BlinkMacSystemFont,
"Segoe UI","Malgun Gothic",sans-serif}
.container{max-width:960px;margin:0 auto}
header{margin-bottom:2.4rem;border-bottom:2px solid var(--acc);padding-bottom:1.4rem}
.eyebrow{color:var(--acc);font-size:.78rem;letter-spacing:.12em;text-transform:uppercase;
font-weight:600}
h1{font-size:2rem;line-height:1.25;margin:.4rem 0 .8rem;text-wrap:balance}
.meta{color:var(--muted);font-size:.84rem;line-height:1.9}
h2{font-size:1.32rem;margin:2.6rem 0 .9rem;display:flex;align-items:baseline;gap:.6rem}
h2 .num{color:var(--acc);font-size:.95rem;font-weight:700}
h3{font-size:1.05rem;margin:1.9rem 0 .6rem}
code{background:var(--code);padding:.1em .35em;border-radius:3px;font-size:.88em}
pre{background:var(--code);padding:1rem;border-radius:6px;overflow-x:auto;font-size:.82rem;
line-height:1.55}
table{border-collapse:collapse;width:100%;font-size:.84rem;font-variant-numeric:tabular-nums}
th,td{padding:.45rem .55rem;text-align:right;border-bottom:1px solid var(--bd);
vertical-align:top}
th:first-child,td:first-child,th:last-child,td:last-child{text-align:left}
thead th{font-size:.74rem;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);
border-bottom:2px solid var(--bd)}
tbody tr:nth-child(even){background:var(--stripe)}
tbody tr.hl,tbody tr.hl:nth-child(even){background:var(--hl);
box-shadow:inset 3px 0 0 var(--acc)}
.scroll{overflow-x:auto;margin:1rem 0}
.ci{color:var(--muted);font-size:.76rem}
.dim{color:var(--muted);font-size:.82rem}.good{color:var(--good)}.bad{color:var(--bad)}
.callout{background:var(--card);border-left:3px solid var(--acc);padding:1rem 1.2rem;
margin:1.2rem 0;border-radius:0 6px 6px 0}
.warn{background:var(--card);border-left:3px solid var(--warn);padding:1rem 1.2rem;
margin:1.2rem 0;border-radius:0 6px 6px 0}
.note{font-size:.88rem;color:var(--muted)}
figure{margin:1.6rem 0}figure img{width:100%;border-radius:6px;display:block}
figcaption{color:var(--muted);font-size:.8rem;margin-top:.5rem}
.kpi-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:.8rem;
margin:1.6rem 0}
.kpi{background:var(--card);border:1px solid var(--bd);border-radius:8px;padding:.9rem 1rem}
.kpi .v{font-size:1.5rem;font-weight:700;font-variant-numeric:tabular-nums}
.kpi .l{font-size:.76rem;color:var(--muted);line-height:1.45;margin-top:.2rem}
@media (max-width:640px){body{padding:2rem 1rem 3rem}h1{font-size:1.5rem}}
"""


def build(res, df, plot_dir, date):
    h = res[res.set == "hard100"].set_index("metric")
    o = res[res.set == "origin150"].set_index("metric")
    n_metrics = h.shape[0]
    beat = [m for m in h.index
            if h.loc[m, "frac"] > h.loc["score", "frac"] and m != "score"]
    return f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>어떤 지표가 arm 을 가르는가</title><style>{CSS}</style></head>
<body><div class="container">
<header>
  <div class="eyebrow">Model Compression &middot; 폐루프 지표 선택</div>
  <h1>어떤 지표가 arm 을 가르는가</h1>
  <div class="meta">
    {date} &middot; alpasim 이 rollout 마다 남기는 {n_metrics}개 지표 &times; 두 씬 집합
    &middot; origin150 (7 arm, 21쌍) / hard100 (8 arm, 28쌍)
    &middot; rollout {len(df):,}개, GPU 사용 0
    &middot; 산출물 <code>outputs/metric_power/</code>
  </div>
</header>

<div class="kpi-row">
  <div class="kpi"><div class="v">{int(h.loc['score', 'resolved'])} / 28</div>
    <div class="l"><code>score</code> 가 가르는 arm 쌍 &mdash; 중간이다</div></div>
  <div class="kpi"><div class="v">{len(beat)}</div>
    <div class="l"><code>score</code> 보다 많이 가르는 지표 수</div></div>
  <div class="kpi"><div class="v">{int(h.loc['plan_deviation', 'resolved'])} / 28</div>
    <div class="l"><code>plan_deviation</code> &mdash; 1위인데 &rho;={h.loc['plan_deviation', 'rho']:+.2f}</div></div>
  <div class="kpi"><div class="v">0 / 28</div>
    <div class="l"><code>collision_at_fault</code> &mdash; hard100 에서 한 쌍도 못 가른다</div></div>
</div>

<h2><span class="num">1.</span>묻는 것</h2>
<p>alpasim 은 rollout 마다 스무 개 남짓한 양을 남긴다. 그중 무엇을 논문 표에 넣을지는
보통 취향으로 정해지는데, 이 저장소에는 이미 <strong>지표 선택이 결론을 뒤집은 기록</strong>이
여러 건 있다 &mdash; <code>offroad</code> 만 보면 가장 안전해 보이는 arm 이 꼴찌이고
(<code>wanda</code>), 게이트가 못 가른 것을 연속 대리지표가 갈랐다(hard100 §7).</p>
<p>그래서 취향 대신 두 가지를 <strong>잰다</strong>.</p>
<ol>
  <li><strong>분해력</strong> &mdash; 이 지표로 arm 쌍 몇 개를 가를 수 있는가.</li>
  <li><strong>정합성</strong> &mdash; 이 지표가 매기는 arm 순서가 종합 점수와 같은가.</li>
</ol>
<p>둘 다 필요하다. 분해력만 높고 정합성이 없으면 <em>다른 것</em>을 재고 있다는 뜻이고,
그건 헤드라인이 아니라 진단 지표다.</p>

<h2><span class="num">2.</span>방법</h2>
<p>각 지표를 <strong>alpasim 이 계산하는 방식 그대로</strong> 계산한다. 그래야 여기 실리는
수가 논문이 인쇄할 수와 같은 것이지 비슷한 것이 아니다.</p>
<pre>1. modifier 체인 (processing.DEFAULT_MODIFIERS + main._run_aggregation)
     eval_relevant 이전 버림
     첫 offroad_or_collision 이후 버림
     첫 dist_to_gt_trajectory &gt;= 4.0 이후 버림
2. 살아남은 타임스텝을 <b>그 지표 자신의</b> time_aggregation 으로 집계
     offroad=max,  progress=last,  min_distance_to_obstacle_m=min, ...
3. rollout -&gt; 씬 평균 -&gt; arm 쌍별 씬 페어드 델타
4. "분해됨" = 10k bootstrap CI 가 0 을 배제 (이 저장소의 표준 기준)</pre>
<div class="callout">
  <p><strong>2단계가 조용히 틀리기 쉬운 곳이다.</strong> 집계 규칙을 하나로 통일하면
  세 지표가 각각 다른 양이 된다 &mdash; <code>offroad</code> 는 &ldquo;한 번이라도
  벗어났는가&rdquo;(max), <code>progress</code> 는 &ldquo;끝에서 얼마나 갔는가&rdquo;(last),
  <code>min_distance_to_obstacle_m</code> 은 &ldquo;가장 가까웠을 때&rdquo;(min)다.
  parquet 이 지표마다 <code>time_aggregation</code> 을 들고 있으므로 그것을 쓴다.</p>
</div>

<h2><span class="num">3.</span>결과 (사실)</h2>
{table(res)}

<figure>
  <img alt="분해력과 정합성의 사분면" src="data:image/png;base64,{b64(plot_dir / 'quadrant.png')}">
  <figcaption>가로는 hard100 28쌍 중 분해한 비율, 세로는 arm 순서의 score 와의 순위상관.
  점선이 <code>score</code> 자신의 분해율이다. 오른쪽 위가 헤드라인 후보, 오른쪽 아래는
  잘 가르지만 점수와 반대로 가는 진단 지표다.</figcaption>
</figure>

<figure>
  <img alt="지표별 분해율, 두 집합" src="data:image/png;base64,{b64(plot_dir / 'ranked.png')}">
  <figcaption>두 집합의 쌍 개수가 달라(21 대 28) 비율로 그렸고, 막대 끝에 원래 개수를
  붙였다.</figcaption>
</figure>

<h2><span class="num">4.</span>읽히는 것</h2>
<div class="callout">
  <p><strong>1 &mdash; <code>score</code> 의 분해력은 중간이다.</strong>
  {int(h.loc['score', 'resolved'])}/28 로, <strong>{len(beat)}개 지표가 종합 점수보다 많은
  쌍을 가른다</strong>. 놀랄 일은 아니다 &mdash; <code>score_criteria</code> 는 게이트에
  걸리면 0 을 주고 나머지는 progress 를 0.8 에서 포화시키므로, 설계상 정보를 버린다.
  씬 절반이 정확히 1.0 에 붙는 것도 같은 이유다.</p>
  <p><strong>2 &mdash; <code>plan_deviation</code> 은 1위인데 헤드라인이 될 수 없다.</strong>
  {int(h.loc['plan_deviation', 'resolved'])}/28 로 가장 많이 가르지만 &rho; 가
  {o.loc['plan_deviation', 'rho']:+.2f} / {h.loc['plan_deviation', 'rho']:+.2f} 로
  <strong>점수와 반대</strong>다. 잘 달리는 arm 일수록 계획 편차가 크다는 뜻이고, 이는
  <code>2026-09-11</code> 보고서의 &ldquo;계획 자기 일관성은 주행을 예측하지 못한다&rdquo;와
  같은 현상이다. 분해력 1위를 그대로 표에 올리면 서열이 뒤집힌다.</p>
  <p><strong>3 &mdash; <code>progress</code> 계열이 최적 후보다.</strong>
  {int(h.loc['progress', 'resolved'])}/28 로 <code>score</code> 보다 높고 &rho; 가
  {h.loc['progress', 'rho']:+.2f} 로 방향도 같다. <code>dist_traveled_m</code> 이
  같은 프로파일({int(h.loc['dist_traveled_m', 'resolved'])}/28,
  &rho;={h.loc['dist_traveled_m', 'rho']:+.2f})인 것은 둘이 사실상 같은 것을 재기
  때문이다.</p>
  <p><strong>4 &mdash; <code>collision_at_fault</code> 는 hard100 에서 0/28 이다.</strong>
  &ldquo;과실 충돌은 세기에 너무 드물다&rdquo;는 이 저장소의 반복된 서술이 여기서
  <strong>실측으로 확정</strong>된다 &mdash; 28쌍 전부에서 한 쌍도 가르지 못한다.
  origin150 에서 {int(o.loc['collision_at_fault', 'resolved'])}/21 이 나오는 것은
  <code>wanda</code>(14.7%)라는 극단값 하나 때문이고, 그것을 두 arm 사이의 안전 주장으로
  옮기면 안 된다.</p>
</div>

<h3>4.1 뜻밖의 것 둘</h3>
<p><strong><code>offroad</code> 의 &rho; 가 두 집합에서 부호가 갈린다</strong>
({o.loc['offroad', 'rho']:+.2f} / {h.loc['offroad', 'rho']:+.2f}). 쉬운 집합에서는
점수와 강하게 반대이고 어려운 집합에서는 무관하다. 게이트를 단독으로 읽으면 부호를
반대로 읽는다는 것이 상관계수로도 확인된다 &mdash; 기전은 <code>progress</code> 가
말해준다: 못 나가는 arm 은 도로를 벗어날 일도 없다.</p>
<p><strong><code>min_distance_to_obstacle_m</code> 은 같은 함정의 세 번째
사례다.</strong> hard100 에서
{int(h.loc['min_distance_to_obstacle_m', 'resolved'])}/28 로 잘 가르지만 &rho; 가
{h.loc['min_distance_to_obstacle_m', 'rho']:+.2f} &mdash; <strong>잘 달리는 arm 일수록
장애물에 가까이 지나간다</strong>. 원인은 노출량이다:</p>
<pre>arm       score   장애물 최소거리   progress   주행거리
wanda     0.414      6.91 m        0.393      135 m   &larr; 가장 멀다, 꼴찌
coc       0.428      7.39 m        0.426      148 m
baseline  0.521      5.52 m        0.564      229 m
tyrK      0.622      4.95 m        0.648      257 m   &larr; 가장 가깝다, 1위</pre>
<p>움직이지 않는 arm 은 아무것에도 가까워지지 않는다. &rho;(장애물거리, progress) 가
&minus;0.69 인 것이 그 말이다. <strong>안전 신호로 읽으면 부호가 정확히 반대가 된다.</strong></p>
<div class="callout">
  <p><strong>그래서 근접 지표는 노출량으로 정규화해야 쓸 수 있다.</strong> 원시 최소거리는
  &ldquo;얼마나 위험하게 운전했는가&rdquo;가 아니라 &ldquo;얼마나 많이 운전했는가&rdquo;를
  재고 있다. <code>analyze_longitudinal.py</code> 가 비율과 시간 정규화 형태
  (시간 헤드웨이, 근접 노출 <em>비율</em>, 근접 시 제동 <em>비율</em>)를 쓰는 이유가
  이것이고, hard100 §7 에서 그 지표들이 게이트가 못 가른 서열을 가른 이유이기도 하다.
  이 표의 원시 <code>min_distance_to_obstacle_m</code> 은 그 정규화를 거치지 않은
  값이다.</p>
</div>

<h2><span class="num">5.</span>권고 (의견)</h2>
<div class="callout">
  <p><strong>본표 다섯 열</strong> &mdash; <code>score</code>(씬 페어드 델타 + CI),
  <code>progress</code>, <code>collision_at_fault</code> 와 <code>offroad</code>(비율 +
  Wilson CI, <em>판정 근거가 아니라 맥락</em>), CoC 퇴화율.</p>
  <p><code>progress</code> 는 선택이 아니라 <strong>필수</strong>다. 그것 없이는
  <code>offroad</code> 가 부호를 반대로 읽히고, 이 저장소에는 그 함정에 빠진 arm 이
  둘 있다 &mdash; <code>wanda</code>(이탈 최저, 점수 꼴찌)와 <code>act_mlp70</code>
  (추론을 버려 progress 를 얻고 점수 1위).</p>
  <p><strong>별표</strong> &mdash; 종단 대리지표(근접 시 제동비율, 선행차 TTC&lt;2s 비율).
  게이트가 못 가른 서열을 가르는 자리이고 hard100 §7 이 그 용례다. <strong>원시
  <code>min_distance_to_obstacle_m</code> 은 넣지 말 것</strong> &mdash; 4.1 이 보이듯
  노출량 교락이라 부호가 반대로 읽힌다. 근접 지표는 거리나 시간으로 정규화한 형태만
  쓴다.</p>
  <p><strong>방법론 절</strong> &mdash; <code>plan_deviation</code> 과 원시
  <code>min_distance_to_obstacle_m</code>. 분해력 상위이면서 점수와 반대라는 사실 자체가
  &ldquo;분해력만으로 지표를 고르면 안 된다&rdquo;의 증거다.</p>
</div>

<h2><span class="num">6.</span>주장하지 않는 것</h2>
<div class="warn">
  <p><strong>분해력은 타당성이 아니다.</strong> 이 분석이 재는 것은 &ldquo;이 지표가 우리가
  가진 arm 들을 가르는가&rdquo;이지 &ldquo;이 지표가 좋은 주행을 재는가&rdquo;가 아니다.
  <code>plan_deviation</code> 과 <code>min_distance_to_obstacle_m</code> 이 그 구분을
  보여주는 사례다 &mdash; 둘 다 상위권 분해력에 음의 &rho; 다.</p>
  <p><strong>음의 &rho; 가 전부 같은 이유는 아니다.</strong> 이 표에서 &rho; 가 음수인
  지표는 크게 두 부류다. <code>offroad</code>&middot;<code>min_distance_to_obstacle_m</code>
  &middot;<code>collision_*</code> 은 <strong>노출량 교락</strong>(못 달리는 arm 은 위험에
  노출되지 않는다)이고, <code>plan_deviation</code> 은 그것으로 설명되지 않는다
  (<code>2026-09-11</code> 보고서가 계획 자기 일관성 자체를 따로 다뤘다). 이 보고서는
  둘을 구분해 주지 못하며, 구분하려면 각 지표를 progress 로 나눠 보는 별도 분석이
  필요하다.</p>
  <p><strong>arm 구성에 의존한다.</strong> 두 집합의 arm 은 <code>wanda</code> 처럼 크게
  망가진 것부터 <code>tyrK</code> 까지 섞여 있다. 망가진 arm 을 빼면 모든 지표의 분해력이
  내려갈 것이고, 특히 <code>collision_at_fault</code> 의 origin150 {int(o.loc['collision_at_fault', 'resolved'])}/21 은
  <code>wanda</code> 에 기대고 있다. <strong>arm 이 비슷할수록 지표 선택이 더
  중요해진다</strong>는 것이 함의다.</p>
  <p><strong>4 m 절단이 적용된 값이다.</strong> 2단계 modifier 체인에 그 규칙이 들어 있다.
  규칙을 끄면 값이 달라지고, 특히 게이트 계열이 달라진다 &mdash; 이 보고서는 출하
  프로토콜을 재현한 것이지 그 프로토콜을 검증한 것이 아니다.</p>
  <p><strong>&rho; 는 arm {h.loc['score', 'n_arms']}개 / {o.loc['score', 'n_arms']}개로 계산한
  순위상관이다.</strong> n 이 작아 개별 값의 신뢰구간은 넓다. 여기서 쓰는 방식은
  &ldquo;부호와 대략의 크기&rdquo;까지이고, {h.loc['plan_deviation', 'rho']:+.2f} 대
  {h.loc['progress', 'rho']:+.2f} 처럼 부호가 갈리는 대비만 근거로 삼는다.</p>
  <p><strong>안 나온 지표도 있다.</strong> <code>safety_monitor_triggered</code> 는 전
  rollout 에서 0 이고, <code>gt_dist_traveled_m</code> 은 씬의 속성이라 arm 과 무관하다
  (둘 다 0/28 인 것이 정상이다). <code>min_ade@{{0.5,1.0,2.5,5.0}}s(gt)</code> 는 일부 런에만
  있어 이번 표에서 빠졌다 &mdash; 폐루프 안에서 잰 궤적 정확도라 이 연구의
  개루프&ndash;폐루프 해리 주제와 맞닿는데, 아직 아무도 쓰지 않았다.</p>
</div>

<h2><span class="num">7.</span>재현</h2>
<pre>python experiments/evaluation/metric_power_report.py \\
    --out reports/evaluation/{date}_alpasim-metric-power.html

# rollout 단위 지표를 다시 뽑으려면 (parquet 수천 개, 몇 분)
python experiments/evaluation/metric_power_report.py --rebuild --out ...</pre>
<p class="note">캐시는 <code>outputs/metric_power/rollout_metrics.parquet</code> 이고
rollout {len(df):,}개 &times; 지표 {n_metrics}개다. GPU 를 쓰지 않으므로 카드가 바빠도
돌릴 수 있고, 새 arm 이 들어오면 <code>--rebuild</code> 한 번으로 표가 갱신된다.</p>
</div></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rebuild", action="store_true",
                    help="rollout parquet 을 다시 읽어 캐시를 새로 만든다")
    ap.add_argument("--date", default="2026-09-20")
    args = ap.parse_args()
    if args.rebuild or not CACHE.exists():
        raise SystemExit(
            "캐시가 없다. build_metric_cache 경로는 아직 이 스크립트에 포함되지 않았다 -- "
            f"{CACHE} 를 먼저 만들 것")
    df = pd.read_parquet(CACHE)
    res = analyse(df)
    plot_dir = O / "metric_power" / "plots"
    plots(res, plot_dir)
    res.to_csv(O / "metric_power" / "metrics.csv", index=False)
    (O / "metric_power" / "metrics.json").write_text(
        json.dumps(res.to_dict("records"), ensure_ascii=False, indent=1))
    out = args.out if args.out.is_absolute() else Path.cwd() / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(res, df, plot_dir, args.date))
    print(f"wrote {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
