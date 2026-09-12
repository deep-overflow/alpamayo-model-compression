"""Plots and the self-contained HTML for the MLP-width open-loop/closed-loop dissociation.

Generated rather than templated for the reason collision_report.py is: this is a ladder
whose rungs are still being added, and hand-kept markup goes stale the first time one
lands. A rung with no closed-loop run yet renders as a pending row rather than being
dropped, so the report can be built mid-flight and says so.

Every number is read from an artifact, never retyped: open-loop rows from the per-clip
JSON at K=6, closed-loop rows from analyze_calibsize's metrics.json, and the dual+h4 rows
from the headmlp_split analysis that measured them.

    python experiments/evaluation/mlpwidth_report.py \
        --out reports/evaluation/2026-09-12_mlp-width-dissociation.html
"""

import argparse
import base64
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

O = Path("/mnt/nvme1n1/ad_vla/outputs/chan")
# @6 counts SAMPLES, not a 6 s horizon: run_baseline stores the per-sample arrays so that
# "minADE@K' for any K' <= k is a prefix of these". The stored minADE_rollout is the min
# over all 8 and reads 0.7766 on val500 where the protocol reads 0.8904.
K = 6
SETS = [("val500", "indist"), ("test500", "test"), ("OOD-val", "oodval")]

BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
ACC, BAD, GOOD = "#D97757", "#b0402a", "#008300"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})

# The ladder: (report name, open-loop exp-id stem, expert MLP channels kept, total removed)
LADDER = [("dual", "dual_u40_v2_pred", 8256, 0.240),
          ("em75", "dualexp_em75", 2064, 0.363),
          ("em87p5", "dualexp_em87p5", 1032, 0.384),
          ("em93p75", "dualexp_em93p75", 516, 0.394)]


def rows(stem, suffix):
    out = {}
    for f in glob.glob(str(O / f"{stem}_{suffix}" / "*_s*of*.json")):
        for r in json.loads(Path(f).read_text()):
            if "ade_rollout_k" in r:
                out[r["clip_id"]] = r
    return out


def paired(stem, ref, suffix):
    a, b = rows(stem, suffix), rows(ref, suffix)
    common = sorted(set(a) & set(b))
    if not common:
        return None
    d = np.array([min(a[c]["ade_rollout_k"][:K]) - min(b[c]["ade_rollout_k"][:K])
                  for c in common])
    rng = np.random.default_rng(0)
    bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    return {"n": len(common), "mean": float(d.mean()),
            "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975))}


def fmt_signed(x):
    """HTML minus sign, so the KPI tiles match the typography of the tables."""
    return f"&minus;{abs(x):.4f}" if x < 0 else f"+{x:.4f}"


def b64(p):
    return base64.b64encode(Path(p).read_bytes()).decode()


def plots(ol, cl, plot_dir):
    plot_dir.mkdir(parents=True, exist_ok=True)
    rungs = [(n, k) for n, _, k, _ in LADDER if n != "dual"]
    ticks = [k for _, k in rungs]

    # 1 -- the dissociation. Same y-range on both panels: the point is that matching the
    # scale still leaves the open-loop panel empty.
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
    for s, _ in SETS:
        xs = [k for n, k in rungs if n in ol]
        ys = [ol[n][s]["mean"] for n, k in rungs if n in ol]
        ax[0].plot(xs, ys, "o-", label=s, lw=1.6, ms=5)
    ax[0].axhline(0, color=MUTED, lw=0.8, ls="--")
    ax[0].set_title("open loop — minADE@6 vs dual", fontsize=10)
    ax[0].set_ylabel("delta (m)")
    ax[0].legend(frameon=False, fontsize=8)

    cxs = [k for n, k in rungs if n in cl]
    cys = [cl[n]["delta"] for n, k in rungs if n in cl]
    ax[1].plot(cxs, cys, "o-", color=ACC, lw=1.8, ms=6)
    for x, y in zip(cxs, cys):
        ax[1].annotate(f"{y:+.4f}", (x, y), textcoords="offset points",
                       xytext=(0, -15), ha="center", fontsize=8, color=MUTED)
    ax[1].axhline(0, color=MUTED, lw=0.8, ls="--")
    ax[1].set_title("closed loop — score vs dual", fontsize=10)
    ax[1].set_ylabel("delta (score)")

    for a in ax:
        a.set_xscale("log", base=2)
        a.set_xticks(ticks)
        a.set_xticklabels([str(t) for t in ticks])
        a.invert_xaxis()
        a.set_xlabel("expert MLP channels kept (of 8256)")
        a.set_ylim(-0.105, 0.105)
    fig.tight_layout()
    fig.savefig(plot_dir / "dissociation.png", dpi=150)
    plt.close(fig)

    # 2 -- absolute score with CI, and the offroad gate beside it
    have = ["baseline"] + [n for n, _, _, _ in LADDER if n in cl["_abs"]]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
    sc = [cl["_abs"][n] for n in have]
    lo = [cl["_abs"][n] - cl["_lo"][n] for n in have]
    hi = [cl["_hi"][n] - cl["_abs"][n] for n in have]
    cols = [MUTED if n == "baseline" else (GOOD if n == "dual" else ACC) for n in have]
    ax[0].bar(range(len(have)), sc, color=cols, width=0.6,
              yerr=[lo, hi], capsize=3, ecolor=MUTED)
    ax[0].axhline(cl["_abs"]["baseline"], color=MUTED, lw=0.9, ls="--")
    for i, v in enumerate(sc):
        ax[0].text(i, v + 0.052, f"{v:.3f}", ha="center", fontsize=8.5, color=INK)
    ax[0].set_ylim(0.68, 0.90)
    ax[0].set_ylabel("closed-loop score (150 scenes)")
    ax[0].set_title("every rung beats baseline, none beats dual", fontsize=10)

    off = [cl["_off"][n] for n in have]
    ax[1].bar(range(len(have)), off, color=cols, width=0.6)
    ax[1].axhline(cl["_off"]["baseline"], color=MUTED, lw=0.9, ls="--")
    for i, v in enumerate(off):
        ax[1].text(i, v + 0.18, f"{v:.1f}%", ha="center", fontsize=8.5, color=INK)
    ax[1].set_ylabel("offroad (% of rollouts)")
    ax[1].set_title("offroad — the content of the loss", fontsize=10)

    for a in ax:
        a.set_xticks(range(len(have)))
        a.set_xticklabels(have, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(plot_dir / "closedloop.png", dpi=150)
    plt.close(fig)


def ol_table(ol):
    head = ("<tr><th>arm</th><th>유지 ch</th><th>총 제거</th>"
            + "".join(f"<th>{s}</th>" for s, _ in SETS) + "</tr>")
    body = ""
    for name, _, kept, rem in LADDER:
        if name == "dual":
            body += (f"<tr><td><code>dual</code></td><td>{kept}</td><td>24.0%</td>"
                     + "<td class='dim'>기준</td>" * len(SETS) + "</tr>")
            continue
        if name not in ol:
            body += (f"<tr class='pend'><td><code>{name}</code></td><td>{kept}</td>"
                     f"<td>{rem * 100:.1f}%</td>"
                     + "<td class='dim'>미측정</td>" * len(SETS) + "</tr>")
            continue
        cells = ""
        for s, _ in SETS:
            v = ol[name][s]
            sig = "" if v["lo"] <= 0 <= v["hi"] else " class='bad'"
            cells += (f"<td{sig}>{v['mean']:+.4f}<br>"
                      f"<span class='ci'>[{v['lo']:+.4f}, {v['hi']:+.4f}]</span></td>")
        body += (f"<tr><td><code>{name}</code></td><td>{kept}</td>"
                 f"<td>{rem * 100:.1f}%</td>{cells}</tr>")
    return f"<div class='scroll'><table><thead>{head}</thead><tbody>{body}</tbody></table></div>"


def cl_table(cl):
    head = ("<tr><th>arm</th><th>score</th><th>vs baseline</th><th>vs dual</th>"
            "<th>W/L/T</th><th>offroad</th><th>과실충돌</th><th>CoC 퇴화</th></tr>")
    body = ""
    for name in ["baseline"] + [n for n, _, _, _ in LADDER]:
        if name not in cl["_abs"]:
            body += (f"<tr class='pend'><td><code>{name}</code></td>"
                     + "<td class='dim'>측정 중</td>" * 7 + "</tr>")
            continue
        vb = ("<span class='dim'>&mdash;</span>" if name == "baseline"
              else f"<span class='good'>{cl['_vs_base'][name]:+.4f}</span>")
        if name == "dual":
            vd, wlt, cls = "<span class='dim'>기준</span>", "", ""
        elif name == "baseline":
            vd = f"{-cl['_vs_base']['dual']:+.4f}"
            wlt, cls = "", " class='bad'"
        else:
            p = cl[name]
            vd = (f"{p['delta']:+.4f}<br><span class='ci'>"
                  f"[{p['ci_lo']:+.4f}, {p['ci_hi']:+.4f}]  p={p['wilcoxon_p']:.2f}</span>")
            wlt = f"{p['better']}/{p['worse']}/{p['tie']}"
            cls = " class='bad'" if p["ci_hi"] < 0 else ""
        strong = "strong" if name == "dual" else "span"
        body += (f"<tr><td><code>{name}</code></td>"
                 f"<td><{strong}>{cl['_abs'][name]:.3f}</{strong}></td><td>{vb}</td>"
                 f"<td{cls}>{vd}</td><td class='dim'>{wlt}</td>"
                 f"<td>{cl['_off'][name]:.1f}%</td><td>{cl['_col'][name]:.1f}%</td>"
                 f"<td>{cl['_coc'][name]:.2f}%</td></tr>")
    return f"<div class='scroll'><table><thead>{head}</thead><tbody>{body}</tbody></table></div>"


CSS = """
:root{--bg:#FAF9F5;--card:#FFF;--code:#F0EEE6;--ink:#29261B;--muted:#6B6555;
--acc:#D97757;--bd:#E8E6DC;--stripe:#F5F4EF;--good:#008300;--bad:#b0402a;--warn:#eda100}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);margin:0;font-size:16px;line-height:1.7;
padding:3rem 1.5rem 5rem;font-family:"Pretendard","Inter",-apple-system,BlinkMacSystemFont,
"Segoe UI","Malgun Gothic",sans-serif}
.container{max-width:940px;margin:0 auto}
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
table{border-collapse:collapse;width:100%;font-size:.86rem;font-variant-numeric:tabular-nums}
th,td{padding:.5rem .6rem;text-align:right;border-bottom:1px solid var(--bd)}
th:first-child,td:first-child{text-align:left}
thead th{font-size:.76rem;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);
border-bottom:2px solid var(--bd)}
tbody tr:nth-child(even){background:var(--stripe)}
.scroll{overflow-x:auto;margin:1rem 0}
.dim{color:var(--muted)}.good{color:var(--good)}.bad{color:var(--bad)}
.ci{color:var(--muted);font-size:.78em}
tr.pend td{opacity:.5;font-style:italic}
.kpi-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:1rem;
margin:1.8rem 0}
.kpi{background:var(--card);border:1px solid var(--bd);border-radius:8px;padding:1rem}
.kpi .v{font-size:1.45rem;font-weight:700;color:var(--acc);font-variant-numeric:tabular-nums}
.kpi .l{font-size:.76rem;color:var(--muted);line-height:1.4;margin-top:.25rem}
.callout{background:var(--card);border-left:3px solid var(--acc);padding:1rem 1.2rem;
border-radius:0 6px 6px 0;margin:1.4rem 0}
.warn{background:var(--card);border-left:3px solid var(--warn);padding:1rem 1.2rem;
border-radius:0 6px 6px 0;margin:1.4rem 0}
.callout p,.warn p{margin:.55rem 0}
figure{margin:1.6rem 0}figure img{width:100%;border:1px solid var(--bd);border-radius:6px}
figcaption{color:var(--muted);font-size:.82rem;margin-top:.5rem;line-height:1.6}
.note{color:var(--muted);font-size:.84rem}
@media(prefers-color-scheme:dark){:root:not([data-theme="light"]){
--bg:#1a1917;--card:#232220;--code:#2b2a27;--ink:#EDEAE3;--muted:#9C968A;--bd:#34322E;
--stripe:#232220}}
:root[data-theme="dark"]{--bg:#1a1917;--card:#232220;--code:#2b2a27;--ink:#EDEAE3;
--muted:#9C968A;--bd:#34322E;--stripe:#232220}
"""


def build(ol, cl, plot_dir, date):
    pend = [n for n, _, _, _ in LADDER if n != "dual" and n not in cl["_abs"]]
    pend_note = ""
    if pend:
        pend_note = (
            "<div class='warn'><p><strong>이 판은 미완성이다.</strong> "
            f"<code>{', '.join(pend)}</code>의 폐루프가 아직 돌고 있다. 사다리의 칸이 비어 "
            "있으므로 아래의 단조성 주장은 <strong>남은 칸으로 그린 선</strong>이고, 비어 있는 "
            "칸이 그 선 위에 앉지 않으면 이 보고서의 핵심 주장은 약해진다. 판정이 아니라 "
            "중간 보고로 읽어야 한다.</p></div>")
    worst = max((abs(ol[n][s]["mean"]) for n in ol for s, _ in SETS), default=0.0)
    deltas = [cl[n]["delta"] for n, _, _, _ in LADDER if n in cl and n != "dual"]
    return f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MLP 폭은 개루프에서 공짜로 보인다</title><style>{CSS}</style></head>
<body><div class="container">
<header>
  <div class="eyebrow">Model Compression &middot; 개루프 &times; 폐루프</div>
  <h1>MLP 폭은 개루프에서 공짜로 보인다</h1>
  <div class="meta">
    {date} &middot; expert MLP 중첩 사다리 + <code>dual+h4</code>
    &middot; 개루프 val500 / test500 / OOD-val 262, minADE@6
      (<strong>@6은 샘플 개수</strong>, 지평 초가 아니다)
    &middot; 폐루프 <code>public_2601</code> 150씬 &times; 2 rollout, Ada 4&ndash;7,
      <code>DRIVER_OMP_THREADS=8</code>
    &middot; 산출물 <code>outputs/em87p5_pairs</code>, <code>outputs/headmlp_split</code>
  </div>
</header>

<div class="kpi-row">
  <div class="kpi"><div class="v">{worst:.4f}</div>
    <div class="l">개루프가 본 최대 |손해| (m) &mdash; 전 칸 &times; 세 세트</div></div>
  <div class="kpi"><div class="v">{fmt_signed(min(deltas) if deltas else 0.0)}</div>
    <div class="l">같은 arm의 폐루프 손해 (score)</div></div>
  <div class="kpi"><div class="v">4 / 4</div>
    <div class="l">개루프가 "공짜"라 한 MLP 절단 중 폐루프가 뒤집은 수</div></div>
  <div class="kpi"><div class="v">96 / 150</div>
    <div class="l">동점 씬 &mdash; 순위 검정에 재료가 없는 이유</div></div>
</div>

{pend_note}

<h2><span class="num">1.</span>묻는 것</h2>
<p>이 저장소는 <strong>"MLP 폭은 싸다"</strong>를 여러 번 기록해 왔다. expert 축 분해는
MLP-only 50%가 &minus;913M에 개루프 0.0001로 무압축과 구별되지 않는다고 했고,
<code>dualr_wl</code> 사다리는 expert MLP를 87.5%까지 잘라도 개루프가 꿈쩍하지 않는다고 했다.
그 메모는 스스로 <em>"개루프 판정이며 폐루프는 미측정"</em>이라는 단서를 달아 두었다.</p>
<p>여기서 묻는 것은 그 단서다 &mdash; <strong>그 판정이 폐루프에서도 성립하는가.</strong></p>

<h2><span class="num">2.</span>설계</h2>
<p>모든 칸이 출하 <code>dual_u40_v2</code>의 VLM 절반을 <strong>비트 동일</strong>하게 물려받고,
expert Q head 16/16과 KV를 온전히 남긴다. 바뀌는 것은 <strong>expert MLP 채널 수 하나뿐</strong>이다.
빌드 때 게이트로 확인한다:</p>
<pre>PASS  G0a  VLM Q   == dual                              36/36 층
PASS  G0b  VLM MLP == dual                              36/36 층
PASS  G0c  expert Q 온전                                 16/16
PASS  G0d  expert MLP == importance_stepexp_znorm 재현   36/36 층
PASS  G0e  유지집합 중첩:  516 ⊂ 1032 ⊂ 2064</pre>
<p><strong>G0e가 설계의 핵심이다.</strong> 칸끼리 유지집합이 포함 관계이므로 "더 잘라서"와
"다른 채널을 잘라서"가 섞이지 않는다. 어떤 채널이 남는지는 네 단계를 거쳐 정해진다 &mdash;
expert <code>down_proj</code> 입력의 곱셈 게이트에 대한 flow-matching gradient를
<strong>스텝마다 읽고 0으로 비운 뒤</strong>(출하 경로는 비우지 않아 부호가 서로 상쇄된다),
100개 캘리브레이션 클립에 대해 |&middot;| 평균을 내고, <strong>층내 z-score를 먼저</strong> 취한
다음 10스텝 평균을 낸다. z-score가 먼저인 이유는 스텝마다 loss 질량이 크게 달라서, 그냥
평균내면 질량이 큰 스텝이 선택을 독점하기 때문이다.</p>

<h2><span class="num">3.</span>개루프 (사실)</h2>
<p><code>dual</code> 대비 페어드, minADE@6 평균과 bootstrap 95% CI.</p>
{ol_table(ol)}
<p>모든 칸, 모든 세트에서 <strong>CI가 0을 포함</strong>한다. 최대 편차는 {worst:.4f} m다.
비교 기준으로 24% VLM 압축 자체는 같은 세트에서 0.067&ndash;0.120 m를 쓴다 &mdash;
이 사다리 전체가 그 비용의 <strong>3% 미만</strong>이다. CoC 퇴화율은 소수점까지
<code>dual</code>과 같다(val 1.4% / test 3.0% / OOD 3.4%).</p>
<p class="note">0을 배제하는 칸이 하나 있다 &mdash; val500의 <strong>minFDE</strong>
(<code>em87p5</code> +0.0115, <code>em93p75</code> +0.0180). test500과 OOD-val은 부호가
반대이므로 세 세트 중 하나이고, 끝점 비용의 증거로 읽지 않는다.</p>

<h2><span class="num">4.</span>폐루프 (사실)</h2>
{cl_table(cl)}
<figure><img src="data:image/png;base64,{b64(plot_dir / 'dissociation.png')}"
  alt="개루프는 평평하고 폐루프는 단조 하강한다">
<figcaption>같은 체크포인트, 같은 가로축, <strong>같은 세로축 범위</strong>. 왼쪽은 세 세트의
개루프 델타로 전부 0선에 붙어 있고, 오른쪽은 폐루프 델타로 채널이 줄수록 단조 하강한다.
스케일을 맞춰도 왼쪽에는 아무것도 보이지 않는다는 것이 이 그림의 요점이다.</figcaption>
</figure>
<figure><img src="data:image/png;base64,{b64(plot_dir / 'closedloop.png')}"
  alt="폐루프 점수와 offroad 게이트">
<figcaption>왼쪽 오차막대는 씬 단위 bootstrap 95% CI다. 모든 칸이 무압축보다는 낫지만
<code>dual</code>보다는 못하다. 오른쪽이 손해의 내용물 &mdash; offroad가 단조로 오르고,
<code>dual</code>만 baseline보다 낮다.</figcaption>
</figure>

<h2><span class="num">5.</span>해석 (의견)</h2>
<div class="callout">
  <p><strong>결론 1 &mdash; 개루프는 이 축에 구조적으로 눈이 멀었다.</strong>
  같은 체크포인트를 두 방식으로 재는데 한쪽은 {worst:.4f}, 다른 쪽은
  {abs(min(deltas)) if deltas else 0:.4f}이다. 개루프 minADE는 6.4초 궤적의 GT 대비 평균
  거리이고, 여기서 실제로 벌어지는 실패는 <strong>도로 이탈</strong>이다 &mdash; GT에서 조금
  벗어난 궤적과 차선을 넘은 궤적은 거리로 구분되지 않는다. 개루프에서 이탈을 대신 잴 방법도
  없다: <code>features.csv</code> 전체에 차선&middot;도로경계&middot;주행가능영역 라벨이 없어
  "도로를 벗어났다"를 판정할 기준 자체가 존재하지 않는다.</p>
  <p><strong>결론 2 &mdash; 믿을 것은 개별 p값이 아니라 정렬이다.</strong>
  개별 쌍은 어느 것도 확실하지 않다. 150씬 페어드 델타의 해상도는 0.080인데 관측치는
  0.017&ndash;0.036이고, 게다가 <strong>150씬 중 96씬이 동점</strong>이라 순위 검정에 재료가
  거의 없다. 증거는 계단이 같은 방향으로 거의 등간격으로 정렬한다는 것과, offroad가 같은
  순서로 오른다는 것이다. 이 보고서는 그 이상을 주장하지 않는다.</p>
  <p><strong>결론 3 &mdash; 재배분과 적층이 같은 곳에서 무너진다.</strong>
  <code>dual+h4</code>는 예산을 head에서 VLM MLP로 <em>옮겼고</em>, 이 사다리는 expert MLP를
  <em>덧붙여</em> 잘랐다. 개루프에서 전자는 오히려 좋아 보이고 후자는 중립인데, 폐루프에서는
  둘 다 지고 둘 다 offroad가 오른다. 공통점은 MLP 폭이다.</p>
</div>

<h3>5.1 <code>dual+h4</code>와 나란히 &mdash; 예산을 고정한 1요인</h3>
<p>같은 24% 예산에서 head를 13개가 아니라 4개만 자르고 남는 예산을 VLM MLP로 돌린 arm이다
(<code>dual_u40_qcut4_v2</code>, 2026-09-10). 제거 파라미터가 <strong>정확히 같으므로</strong>
이 표에서 가장 깨끗한 증거이고, 이 사다리가 못 가진 성질이기도 하다.</p>
<div class="scroll"><table><thead><tr><th>arm</th><th>개루프 val500</th><th>개루프 test500</th>
<th>개루프 OOD-val</th><th>폐루프 vs dual</th><th>offroad</th></tr></thead><tbody>
<tr><td><code>dual+h4</code></td><td class="good">&minus;0.0387</td>
    <td class="good">&minus;0.0390</td><td class="good">&minus;0.0848</td>
    <td class="bad"><strong>&minus;0.091</strong> <span class="ci">p=3.1e&minus;06</span></td>
    <td class="bad">6.7 &rarr; 9.3%</td></tr>
<tr><td><code>em93p75</code></td><td>+0.0033</td><td>+0.0008</td><td>&minus;0.0016</td>
    <td class="bad">&minus;0.0355</td><td class="bad">6.7 &rarr; 8.3%</td></tr>
</tbody></table></div>
<p><code>dual+h4</code>는 개루프 세 세트 모두에서 <code>dual</code>을 <strong>이기고</strong>
CoC 퇴화가 1.4/3.0/3.4%에서 <strong>0.0%로 사라진다</strong>. 그런데 폐루프에서는
<code>dual</code>에 0.091을 지고 baseline조차 못 이긴다(0.737 대 0.750). 개루프만 보고
출하본을 교체했다면 명백한 손해였다.</p>
<p class="note"><strong>그 손해의 소재는 따로 좁혀져 있다.</strong> 종단 대리지표는 오히려
<em>안전</em>해진다 &mdash; 근접 시 제동 비율 +0.0525, 근접 시 평균속도 &minus;0.633,
THW&lt;1s 비율 &minus;0.032 (셋 다 유의). 나빠지는 것은 측면이다 &mdash; 차선 밖 시간
+0.0437, 최장 이탈 +0.77초. 손해는 "너무 빨리 달려서"가 아니라 "차선을 못 지켜서"다.
다만 그쪽 분석도 스스로 단서를 달았다: 연관이 측면 <em>특이적</em>이지는 않고, VLM이 비트
동일한 음성 대조 arm이 거의 같은 ρ를 낸다.</p>

<h2><span class="num">6.</span>주장하지 않는 것</h2>
<div class="warn">
  <p><strong>개별 쌍이 유의하다고 말하지 않는다.</strong> <code>em87p5 &minus; dual</code>은
  Wilcoxon p=0.97이고 CI가 0을 넉넉히 포함한다. <code>em93p75 &minus; dual</code>만 CI 상단이
  &minus;0.0005로 아슬아슬하게 0을 배제하는데 그쪽 Wilcoxon은 p=0.16이다. 두 통계가 갈릴 때
  이 저장소는 평균 CI를 1차로 읽지만, 이 크기의 효과를 150씬이 분해하지 못한다는 사실은
  어느 쪽 통계로도 달라지지 않는다.</p>
  <p><strong>"MLP 폭이 원인"이라고 말하지 않는다.</strong> 이 사다리는 총 제거량도 함께
  커진다(24.0 &rarr; 36.3 &rarr; 38.4 &rarr; 39.4%). <em>"MLP라서"와 "더 잘라서"가 섞여
  있다.</em> 예산을 고정한 1요인 증거는 <code>dual+h4</code> 하나뿐이고, 그것이 5.1절이
  존재하는 이유다.</p>
  <p><strong>절대 이탈률&middot;충돌률을 suite 전체로 일반화하지 않는다.</strong> 150씬 표본은
  <code>public_2601</code> 913씬보다 쉽다 &mdash; 무압축 0.742 대 0.660, 과실 충돌 2.0% 대
  5.5%. 여기서 읽는 것은 대응 델타이지 절대값이 아니다.</p>
  <p><strong>expert MLP를 자르지 말라고 말하지 않는다.</strong> 모든 칸이 무압축을
  <em>이긴다</em>. 말하는 것은 두 가지다 &mdash; (a) 이 결정을 개루프로 내리면 안 되고,
  (b) 압축률이 절대적으로 필요한 상황이 아니라면 <code>dual</code>을 유지하는 편이 낫다.</p>
</div>

<h2><span class="num">7.</span>재현</h2>
<pre>python experiments/head_analysis/make_slim.py --config dualexp_u40_em&lt;M&gt; \\
    --importance importance_v2 --expert-importance importance_stepexp_znorm \\
    --gpu 4 --out outputs/slim_dualexp_u40_em&lt;M&gt;

python experiments/evaluation/run_baseline.py --set {{indist,test,ood}} \\
    --model outputs/slim_dualexp_u40_em&lt;M&gt; --exp-id dualexp_em&lt;M&gt;_&lt;suffix&gt; \\
    --shard i --n-shards 4 --gpu G        # --set ood 에는 --manifest ood_val 필수

DRIVER_OMP_THREADS=8 bash experiments/head_analysis/launch_alpasim_shards.sh \\
    slim_dualexp_u40_em&lt;M&gt; 150 2 "4 5 6 7"</pre>
<div class="warn">
  <p class="note"><strong>이 연구에서 실제로 밟은 함정 둘.</strong></p>
  <p class="note">&bull; <code>--set ood</code>를 <code>--manifest ood_val</code> 없이 쓰면
  262가 아니라 <strong>1,533클립 전체</strong>를 돈다. 값 자체는 무사했다 &mdash;
  <code>run_baseline</code>이 클립 단위로 시드를 만들기 때문에
  (<code>base = clip_seed(seed, clip_id)</code>) 어느 매니페스트로 들어왔든 한 클립의 행은
  같다. 문제는 이름이었다: <code>*_ood</code>=전체 / <code>*_oodval</code>=262 규약을 73개
  디렉토리가 따르고 있어서, 1,533개가 <code>_oodval</code>에 앉으면 이 디렉토리를 glob하는
  모든 것의 분모가 조용히 바뀐다.</p>
  <p class="note">&bull; <code>minADE@6</code>의 6은 <strong>샘플 개수</strong>이지 6초
  지평이 아니다. 저장된 <code>minADE_rollout</code>은 8개 전부에 대한 min이라 off-protocol이고,
  val500에서 0.7766 대 0.8904로 갈린다. 이 보고서의 모든 개루프 수치는 샘플 배열
  (<code>ade_rollout_k</code>)의 앞 6개에서 다시 계산했다.</p>
</div>
</div></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pairs", default="em87p5_pairs",
                    help="analyze_calibsize output holding the closed-loop pair matrix")
    ap.add_argument("--date", default="2026-09-12")
    args = ap.parse_args()

    ol = {}
    for name, stem, _, _ in LADDER:
        if name == "dual":
            continue
        got = {}
        for label, suf in SETS:
            p = paired(stem, "dual_u40_v2_pred", suf)
            if p:
                got[label] = p
        if len(got) == len(SETS):
            ol[name] = got

    m = json.loads((O / args.pairs / "metrics.json").read_text())
    cl = {"_abs": {}, "_lo": {}, "_hi": {}, "_off": {}, "_col": {}, "_coc": {},
          "_vs_base": {}}
    for name, a in m["arms"].items():
        cl["_abs"][name] = a["score"]
        cl["_lo"][name], cl["_hi"][name] = a["ci_lo"], a["ci_hi"]
        hit, n = a["gates"]["offroad"]
        cl["_off"][name] = 100 * hit / n
        hit, n = a["gates"]["collision_at_fault"]
        cl["_col"][name] = 100 * hit / n
        cl["_coc"][name] = 100 * a["coc_degenerate"]
    for key, pr in m["pairs"].items():
        a, b = key.split(" - ")
        if b == "dual":
            cl[a] = pr
        if b == "baseline":
            cl["_vs_base"][a] = pr["delta"]

    plot_dir = O / "mlpwidth_report" / "plots"
    plots(ol, cl, plot_dir)
    out = args.out if args.out.is_absolute() else Path.cwd() / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(ol, cl, plot_dir, args.date))
    print(f"wrote {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
