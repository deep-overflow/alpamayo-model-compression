"""Plots and the self-contained HTML for the open-loop collision proxy.

Generated rather than templated: it is four arms x three sets x several statistics plus a
six-row pair table per set, and hand-maintained markup would go stale the first time an
arm is added. The prose lives here as module constants so it is still reviewed like a
template.

    python experiments/evaluation/collision_report.py \
        --out reports/evaluation/2026-09-09_openloop-collision.html
"""

import argparse
import base64
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

O = Path("/mnt/nvme1n1/ad_vla/outputs/chan")
ARMS = ["baseline_pred", "dual_u40_v2_pred", "tyr_u40_r_pred", "tyrK_pred"]
SHORT = {"baseline_pred": "baseline", "dual_u40_v2_pred": "dual_u40_v2",
         "tyr_u40_r_pred": "tyr_u40_r", "tyrK_pred": "tyrK"}
SETS = ["test500", "val500", "OOD-val"]
SUF = {"test500": "test", "val500": "indist", "OOD-val": "oodval"}

BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
C = ["#6B6555", "#2a78d6", "#008300", "#e87ba4"]
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})


def load():
    coll = {a: json.loads((O / f"{a}_collision" / "metrics.json").read_text())["sets"]
            for a in ARMS}
    gates = json.loads((O / "collision_gates" / "metrics.json").read_text())
    geom = json.loads((O / "collision_geom_check" / "metrics.json").read_text())
    return coll, gates, geom


def minade(arm, suffix):
    rows = {}
    for f in sorted((O / f"{arm}_{suffix}").glob("*_s*of*.json")):
        for r in json.loads(f.read_text()):
            rows[r["clip_id"]] = r["minADE_rollout"]
    return rows


def plots(coll, gates, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1 -- collision rate by arm and set, GT floor drawn as the noise level
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for k, s in enumerate(SETS):
        vals = [coll[a][s]["any_pct"] for a in ARMS]
        ax[k].bar(range(4), vals, color=C, width=0.66, zorder=3)
        gt = coll["baseline_pred"][s]["gt_pct"]
        ax[k].axhline(gt, color="#c0392b", lw=1.2, ls="--", zorder=4)
        ax[k].text(3.45, gt + 1.0, f"GT {gt:.1f}%", color="#c0392b", fontsize=8, ha="right")
        for i, v in enumerate(vals):
            ax[k].text(i, v + 0.7, f"{v:.1f}", ha="center", fontsize=8.5, color=INK)
        ax[k].set_xticks(range(4))
        ax[k].set_xticklabels([SHORT[a] for a in ARMS], rotation=30, ha="right", fontsize=8)
        ax[k].set_title(f"{s}  (n={coll['baseline_pred'][s]['n']})", fontsize=10)
        if k == 0:
            ax[k].set_ylabel("clips where any of 8 samples collides (%)")
    fig.tight_layout()
    fig.savefig(out_dir / "rates.png", dpi=150)
    plt.close(fig)

    # 2 -- the G2 reading: minADE of colliding vs clean clips, per arm
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    for k, s in enumerate(SETS):
        for i, a in enumerate(ARMS):
            per = {r["clip_id"]: r for r in coll[a][s]["rows"]}
            m = minade(a, SUF[s])
            hit = np.array([m[c] for c in per if per[c]["collide_any"] and c in m])
            cln = np.array([m[c] for c in per if not per[c]["collide_any"] and c in m])
            bp = ax[k].boxplot([cln, hit], positions=[i - 0.17, i + 0.17], widths=0.28,
                               showfliers=False, patch_artist=True,
                               medianprops={"color": INK})
            bp["boxes"][0].set_facecolor("#dfe6ee")
            bp["boxes"][1].set_facecolor(C[i])
            for b in bp["boxes"]:
                b.set_edgecolor(MUTED)
        ax[k].set_xticks(range(4))
        ax[k].set_xticklabels([SHORT[a] for a in ARMS], rotation=30, ha="right", fontsize=8)
        ax[k].set_title(f"{s}: minADE, clean (grey) vs colliding", fontsize=10)
        if k == 0:
            ax[k].set_ylabel("minADE@6 (m)")
    fig.tight_layout()
    fig.savefig(out_dir / "g2.png", dpi=150)
    plt.close(fig)

    # 3 -- what gets hit, in-distribution vs OOD
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    order = ["automobile", "person", "heavy_truck", "rider", "bus", "trailer", "stroller"]
    for j, s in enumerate(["test500", "OOD-val"]):
        tot = {}
        for a in ARMS:
            for k2, v in coll[a][s]["classes"].items():
                tot[k2] = tot.get(k2, 0) + v
        n = sum(tot.values())
        vals = [100 * tot.get(k2, 0) / n for k2 in order]
        ax.barh([i + (j - 0.5) * 0.36 for i in range(len(order))], vals,
                height=0.34, color=C[1] if j == 0 else C[3],
                label=f"{s} (n={n} hits)", zorder=3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("share of collisions (%)   ·   summed over the four arms")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "classes.png", dpi=150)
    plt.close(fig)


def b64(p):
    return base64.b64encode(Path(p).read_bytes()).decode()


CSS = """
  :root { --bg:#FAF9F5; --card:#FFF; --code:#F0EEE6; --text:#29261B; --muted:#6B6555;
          --accent:#D97757; --accent2:#C15F3C; --bd:#E8E6DC; --good:#2C6E2C; --bad:#C15F3C; }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:var(--bg); color:var(--text); font-size:16px; line-height:1.7;
         padding:3rem 1.5rem 5rem;
         font-family:"Söhne","Pretendard","Inter",-apple-system,"Malgun Gothic",sans-serif; }
  .container { max-width:1040px; margin:0 auto; }
  header { margin-bottom:2.5rem; border-bottom:2px solid var(--accent); padding-bottom:1.5rem; }
  .eyebrow { color:var(--accent2); font-size:.78rem; font-weight:600; letter-spacing:.12em;
             text-transform:uppercase; margin-bottom:.75rem; }
  h1 { font-family:"Tiempos Headline",Georgia,serif; font-size:2rem; font-weight:500;
       line-height:1.3; margin-bottom:.75rem; text-wrap:balance; }
  .meta { color:var(--muted); font-size:.9rem; }
  .meta code { background:none; padding:0; color:var(--muted); }
  h2 { font-family:"Tiempos Headline",Georgia,serif; font-size:1.4rem; font-weight:500;
       margin:2.75rem 0 1rem; }
  h2 .num { color:var(--accent); margin-right:.4rem; }
  h3 { font-size:1.02rem; font-weight:600; margin:1.75rem 0 .5rem; }
  p { margin-bottom:1rem; } ul,ol { margin:0 0 1rem 1.4rem; } li { margin-bottom:.45rem; }
  code { font-family:"Berkeley Mono","SF Mono",Menlo,monospace; font-size:.86em;
         background:var(--code); padding:.12em .35em; border-radius:4px; }
  pre { background:var(--code); border-radius:8px; padding:1rem 1.25rem; overflow-x:auto;
        margin:1rem 0 1.5rem; font-size:.85rem; line-height:1.6; }
  pre code { background:none; padding:0; }
  .scroll { overflow-x:auto; margin:1.25rem 0 1.75rem; border:1px solid var(--bd);
            border-radius:8px; background:var(--card); }
  table { width:100%; border-collapse:collapse; font-size:.87rem; }
  th { background:var(--code); text-align:right; font-weight:600; padding:.5rem .7rem;
       border-bottom:2px solid var(--bd); white-space:nowrap; }
  th:first-child { text-align:left; }
  td { padding:.42rem .7rem; border-bottom:1px solid var(--bd); text-align:right;
       font-variant-numeric:tabular-nums; white-space:nowrap; }
  td:first-child { text-align:left; font-family:"Berkeley Mono",Menlo,monospace;
                   font-size:.9em; }
  tbody tr:last-child td { border-bottom:none; }
  tr.base td { background:#F3F1E8; }
  td.good { color:var(--good); } td.bad { color:var(--bad); } td.dim { color:var(--muted); }
  .callout { background:#FBF0EB; border-left:4px solid var(--accent);
             border-radius:0 8px 8px 0; padding:1rem 1.25rem; margin:1.5rem 0; }
  .callout strong:first-child { color:var(--accent2); }
  .warn { background:#FFF8E6; border-left:4px solid #eda100; border-radius:0 8px 8px 0;
          padding:1rem 1.25rem; margin:1.5rem 0; font-size:.95rem; }
  .note { background:var(--card); border:1px solid var(--bd); border-radius:8px;
          padding:1rem 1.25rem; margin:1.5rem 0; color:var(--muted); font-size:.92rem; }
  figure { margin:1.5rem 0; background:var(--card); border:1px solid var(--bd);
           border-radius:8px; padding:1rem; overflow-x:auto; }
  figure img { max-width:100%; height:auto; display:block; margin:0 auto; }
  figcaption { color:var(--muted); font-size:.88rem; margin-top:.75rem; line-height:1.55; }
  .kpi-row { display:flex; flex-wrap:wrap; gap:.75rem; margin:1.25rem 0 1.5rem; }
  .kpi { flex:1 1 150px; background:var(--card); border:1px solid var(--bd);
         border-radius:8px; padding:.8rem 1rem; }
  .kpi .v { font-size:1.3rem; font-weight:650; font-variant-numeric:tabular-nums; }
  .kpi .l { color:var(--muted); font-size:.8rem; margin-top:.1rem; }
  .pass { color:var(--good); font-weight:650; } .fail { color:var(--bad); font-weight:650; }
"""


def rate_table(coll):
    out = ['<div class="scroll"><table><thead><tr><th>arm</th>']
    for s in SETS:
        out.append(f'<th colspan="3" style="text-align:center">{s}</th>')
    out.append("</tr><tr><th></th>")
    for _ in SETS:
        out.append("<th>any</th><th>frac</th><th>best</th>")
    out.append("</tr></thead><tbody>")
    for a in ["GT"] + ARMS:
        if a == "GT":
            cells = "".join(f'<td class="dim">{coll[ARMS[0]][s]["gt_pct"]:.1f}</td>'
                            f'<td class="dim">&mdash;</td><td class="dim">&mdash;</td>'
                            for s in SETS)
            out.append(f'<tr><td>GT 궤적 (바닥)</td>{cells}</tr>')
            continue
        base = coll["baseline_pred"]
        cells = ""
        for s in SETS:
            for k in ("any_pct", "frac_pct", "best_pct"):
                v, b = coll[a][s][k], base[s][k]
                cls = "" if a == "baseline_pred" else (" class=\"bad\"" if v > b else " class=\"good\"")
                cells += f"<td{cls}>{v:.1f}</td>"
        cls = ' class="base"' if a == "baseline_pred" else ""
        out.append(f"<tr{cls}><td>{SHORT[a]}</td>{cells}</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def pair_table(gates):
    out = [('<div class="scroll"><table><thead><tr><th>쌍</th><th>세트</th>'
            "<th>충돌 %</th><th>McNemar p</th><th>minADE Δ</th><th>Wilcoxon p</th>"
            "<th>판정</th></tr></thead><tbody>")]
    keys = list(gates["sets"]["test500"]["pairs"])
    for k in keys:
        for s in SETS:
            v = gates["sets"][s]["pairs"][k]
            cs, ms = v["collision_separates"], v["minade_separates"]
            tag = ("충돌만" if cs and not ms else "둘 다" if cs
                   else "minADE만" if ms else "둘 다 못 가름")
            cls = "good" if cs and not ms else ("dim" if not (cs or ms) else "")
            nm = k.replace("_pred", "")
            out.append(
                f"<tr><td>{nm}</td><td>{s}</td>"
                f'<td>{v["collide_pct"][0]:.1f}&rarr;{v["collide_pct"][1]:.1f}</td>'
                f'<td>{v["collide_p"]:.1e}</td><td>{v["minADE_delta"]:+.4f}</td>'
                f'<td>{v["minADE_p"]:.3f}</td>'
                f'<td class="{cls}" style="text-align:center">{tag}</td></tr>')
    out.append("</tbody></table></div>")
    return "\n".join(out)


def g2_table(gates):
    out = ['<div class="scroll"><table><thead><tr><th>arm</th>']
    for s in SETS:
        out.append(f'<th colspan="3" style="text-align:center">{s}</th>')
    out.append("</tr><tr><th></th>")
    for _ in SETS:
        out.append("<th>ρ</th><th>충돌 클립</th><th>무충돌</th>")
    out.append("</tr></thead><tbody>")
    for a in ARMS:
        cells = ""
        for s in SETS:
            v = gates["sets"][s]["per_clip"][a]
            cells += (f'<td>{v["rho"]:+.3f}</td><td>{v["minADE_hit"]:.3f}</td>'
                      f'<td class="dim">{v["minADE_clean"]:.3f}</td>')
        out.append(f"<tr><td>{SHORT[a]}</td>{cells}</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def build(coll, gates, geom, plot_dir, date):
    t = gates["sets"]["test500"]
    b, d = coll["baseline_pred"], coll["dual_u40_v2_pred"]
    rhos = [gates["sets"][s]["per_clip"][a]["rho"] for s in SETS for a in ARMS]
    dpp = d["test500"]["any_pct"] - b["test500"]["any_pct"]
    return f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>개루프 충돌 대리지표</title><style>{CSS}</style></head><body><div class="container">
<header>
  <div class="eyebrow">Model Compression &middot; 개루프 평가</div>
  <h1>개루프에 안전 축을 하나 붙이면 무엇이 보이나</h1>
  <div class="meta">
    {date} &middot; arm 4개 &times; 세트 3개 (test500 489 / val500 481 / OOD-val 240)
    &middot; 라벨 <code>obstacle.offline</code> 오토라벨 3D 트랙
    &middot; 코드 <code>collision_lib.py</code> / <code>score_collisions.py</code> /
    <code>judge_collision_gates.py</code>
    &middot; 계획 <code>plans/2026-09-08_openloop-collision-proxy.md</code>
    &middot; 실행 cvlab20 Ada 0&ndash;3, 채점 cvlab21 CPU
  </div>
</header>

<div class="kpi-row">
  <div class="kpi"><div class="v">{geom['rate_pct']:.1f}%</div>
    <div class="l">GT 궤적 충돌 (기하 검증)</div></div>
  <div class="kpi"><div class="v">{b['test500']['any_pct']:.1f}%</div>
    <div class="l">무압축 baseline (test500)</div></div>
  <div class="kpi"><div class="v">+{dpp:.1f}pp</div>
    <div class="l">24% 압축의 대가</div></div>
  <div class="kpi"><div class="v">{min(rhos):+.2f}&ndash;{max(rhos):+.2f}</div>
    <div class="l">클립 수준 충돌&harr;minADE ρ</div></div>
</div>

<h2><span class="num">1.</span>목적</h2>
<p>개루프는 minADE와 minFDE만 본다. 둘 다 GT 궤적과의 <em>거리</em>이고,
<strong>무엇에 부딪히는지는 보지 않는다</strong>. 폐루프에서는 이 구분이 실제로 arm을 갈랐다 &mdash;
합산 250씬에서 <code>lp_r50</code>은 종합 점수가 <code>tyr_u40_r</code>보다 높은데 전체 충돌은
16.1% 대 9.7%로 나빴다.</p>
<p>데이터셋에 오토라벨 3D 장애물 트랙이 있으므로 개루프에서도 접촉을 대리지표로 잴 수 있다.
<strong>이탈(offroad)은 불가능하다</strong> &mdash; <code>features.csv</code> 전체에 차선·도로경계·
주행가능영역 라벨이 없어 "도로를 벗어났다"를 판정할 기준이 존재하지 않는다.</p>

<h2><span class="num">2.</span>방법 &mdash; 좌표계가 전부다</h2>
<p>장애물은 <strong>각 행 시각의 rig 프레임</strong>에 있고(<code>reference_frame_timestamp_us
== timestamp_us</code>), 예측 궤적은 <strong>t0의 rig 프레임</strong>에 있다. 둘 사이의 변환이
곧 GT 미래 pose다:</p>
<pre><code>c_t0 = R_i · c_rig(t_i) + p_i     R_i = ego_future_rot[i], p_i = ego_future_xyz[i]</code></pre>
<p>판정은 2D bird's-eye OBB 분리축 테스트. 자차 박스는 <code>vehicle_dimensions</code>에서
클립별로 읽고(<code>rear_axle_to_bbox_center</code>만큼 앞에 놓임), heading은 모델이 pose를
내지 않으므로 경로 접선에서 유도하되 0.5 m/s 미만에서는 직전 값을 유지한다.</p>

<div class="callout">
  <p><strong>GT가 기하를 검증한다.</strong> 자차는 실제로 그 경로를 주행했으므로 충돌하면 안 된다.
  실측 <strong>{geom['collide']}/{geom['n']} = {geom['rate_pct']:.1f}%</strong>로 게이트(2% 미만)를
  통과했다. 그리고 네 arm 전부에서 <strong>GT만 충돌하고 예측은 안 한 클립이 세 세트 통틀어
  1개</strong>다 &mdash; 불일치가 완전히 한쪽이라는 것이 기하가 옳다는 가장 강한 증거다.</p>
</div>

<div class="warn">
  <p><strong>게이트가 바로 일했다.</strong> 첫 실행에서 모든 충돌이 step 0, 즉 자차 위치에 몰렸다.
  <code>track 4</code>가 중심 (1.00, 0.01), 크기 4.62&times;1.97인데 자차가 4.69&times;2.00이고
  박스 중심이 (1.31, 0) &mdash; <strong>오토라벨러가 자차를 장애물로 라벨링</strong>해 둔 것이다.
  t0에서 자차 footprint와 겹치는 트랙을 버리도록 고쳤다.</p>
</div>

<h2><span class="num">3.</span>결과 (사실)</h2>
<h3>3.1 충돌률</h3>
<p><code>any</code> = 8개 샘플 중 하나라도 충돌한 클립 비율, <code>frac</code> = 충돌한 샘플의
평균 비율, <code>best</code> = minADE가 가장 낮은 샘플(실제로 고를 궤적)의 충돌 여부.
색은 baseline 대비.</p>
{rate_table(coll)}
<figure><img src="data:image/png;base64,{b64(plot_dir / 'rates.png')}"
  alt="arm과 세트별 충돌률, GT 바닥선 표시">
<figcaption>세 압축 arm이 세 세트 모두에서 무압축보다 높고, <strong>서로는 거의 같다</strong>
(test500 18.8&ndash;19.6%). 빨간 점선이 GT 궤적의 충돌률로, 이 지표의 잡음 바닥이다.</figcaption>
</figure>

<h3>3.2 G2 &mdash; 충돌은 minADE의 재표현인가</h3>
{g2_table(gates)}
<p class="note">사전등록한 G2는 <strong>arm 수준</strong> Spearman |ρ|&lt;0.7이다. 실측은
{t['arm_level']['spearman_rho']:+.2f} / {gates['sets']['val500']['arm_level']['spearman_rho']:+.2f} /
{gates['sets']['OOD-val']['arm_level']['spearman_rho']:+.2f}이지만 <strong>n=4라 검정력이 없다</strong>
(|ρ|=0.8이 p=0.33). 그 통계만으로 PASS를 주장하는 건 정직하지 않으므로, 위 표의 클립 수준
읽기가 실질적 근거다.</p>
<figure><img src="data:image/png;base64,{b64(plot_dir / 'g2.png')}"
  alt="충돌 클립과 무충돌 클립의 minADE 분포">
<figcaption>충돌한 클립의 minADE가 높긴 하지만 분포가 크게 겹친다. 클립 수준 ρ는
{min(rhos):+.2f}&ndash;{max(rhos):+.2f}로, 충돌의 대부분은 그 클립의 minADE로 설명되지 않는다.</figcaption>
</figure>

<h3>3.3 G3 &mdash; arm을 가르는가</h3>
{pair_table(gates)}
<figure><img src="data:image/png;base64,{b64(plot_dir / 'classes.png')}"
  alt="충돌 대상 클래스 구성, in-distribution 대 OOD">
<figcaption>부딪히는 대상이 세트마다 다르다. in-distribution은 automobile이 압도적인데
OOD에서는 person과 rider의 비중이 크게 는다.</figcaption>
</figure>

<h2><span class="num">4.</span>해석 (의견)</h2>
<div class="callout">
  <p><strong>결론 1 &mdash; 압축은 충돌을 늘린다. minADE가 말하지 않던 것이다.</strong>
  세 arm 모두, 세 세트 모두 +5&ndash;7pp. minADE는 "GT에서 얼마나 멀어지는가"만 말하는데,
  그 이탈이 <em>실제로 무언가와 겹치는 방향</em>임이 확인됐다. 서로 다른 기준과 재구성을 쓰는
  세 방법이 거의 같은 폭으로 늘어난다는 것은, 이 손해가 특정 기준의 결함이 아니라
  <strong>24% 압축 자체의 대가</strong>임을 시사한다.</p>
  <p><strong>결론 2 &mdash; 그러나 방법 선택에는 쓸 수 없다.</strong> minADE가 못 가르는 아홉 쌍을
  충돌도 하나도 가르지 못했다(G3 FAIL). 폐루프 250씬에서 세 쌍이 모두 구분되지 않았던 것과
  같은 그림이고, 개루프 충돌은 그 벽을 넘지 못했다.</p>
</div>
<p><strong>왜 G2는 통과하는데 G3는 실패하는가.</strong> 둘은 모순이 아니다. G2는 "충돌이 minADE와
다른 것을 잰다"이고 실제로 그렇다(ρ≈0.1). G3는 "그 다른 것이 방법 사이에서 다르다"인데,
방법들은 <em>그 축에서도</em> 서로 같다. 새 축이긴 한데, 그 축 위에서도 세 방법이 구분되지 않는다.</p>
<p><strong>가장 눈에 띄는 것은 <code>best</code>다.</strong> test500에서 <code>any</code>는
{b['test500']['any_pct']:.1f}&rarr;{d['test500']['any_pct']:.1f}%로 {dpp:.1f}pp 오르는데
<code>best</code>는 {b['test500']['best_pct']:.1f}&rarr;{d['test500']['best_pct']:.1f}%로
{d['test500']['best_pct'] - b['test500']['best_pct']:.1f}pp만 오른다. 압축은 <em>나쁜 샘플</em>을
더 만들지, 고르는 궤적을 그만큼 나쁘게 만들지는 않는다. 샘플 다양성이 늘어난 것인지 최빈 모드가
나빠진 것인지는 이 지표만으로 갈리지 않는다.</p>
<p><strong>OOD가 두 배다.</strong> baseline 기준 {b['test500']['any_pct']:.1f}% &rarr;
{b['OOD-val']['any_pct']:.1f}%. minADE로도 OOD가 나쁘지만 충돌은 훨씬 크게 벌어진다.
거리 지표가 압축하는 실패를 충돌이 드러내는 지점이고, 취약 도로사용자 비중이 함께 느는 것도 여기다.</p>

<h2><span class="num">5.</span>주장하지 않는 것</h2>
<div class="warn">
  <p><strong>폐루프 충돌이 아니다.</strong> 다른 차들은 로그를 재생할 뿐 우리 예측에 반응하지 않으므로,
  실제로는 피했을 상황도 충돌로 센다. 절대 충돌률은 <strong>상한</strong>이고, 같은 라벨을 공유하는
  arm 간 차이가 이 지표의 용도다.</p>
  <p><strong>오토라벨이다.</strong> <code>scene:obstacles:autolabels:v2</code>, 사람 검수 없음.
  arm 비교에는 무해하지만 절대값은 라벨 품질에 종속된다. GT 충돌률 {geom['rate_pct']:.1f}%가
  그 잡음 바닥의 실측치다.</p>
  <p><strong>G2를 사전등록한 형태로 통과했다고 말하지 않는다.</strong> arm 수준 n=4 통계는 세트마다
  ρ가 +0.20에서 +0.80까지 흔들리고 셋 다 p&gt;0.2다. 근거는 클립 수준 쪽이며, 계획서의 게이트
  정의가 이 arm 수에 맞지 않았다.</p>
  <p><strong>heading은 유도값이다.</strong> 경로 접선이므로 저속·정지 구간에서 불안정하다.</p>
  <p><strong>남은 GT 충돌 1건은 진짜가 아닐 가능성이 크다.</strong> 같은 클립 3.5초, 중심거리
  0.21 m &mdash; t0 필터가 못 잡는 두 번째 자차 자기라벨로 보인다.</p>
</div>

<h2><span class="num">6.</span>재현</h2>
<pre><code># 1. 라벨 (게이티드, cvlab21의 full_right 토큰) -- 1,208 클립 706 MB
python experiments/evaluation/fetch_obstacles.py --sets test_500 indist_500 ood_val

# 2. 기하 게이트 (GPU 불필요)
python experiments/evaluation/verify_collision_geom.py --set calib_100

# 3. arm 실행 (cvlab20; --save-pred 없이는 궤적이 저장되지 않는다)
ARM=&lt;arm&gt; CARDS="0 1 2 3" bash cvlab20-server/eval_arm_pred.sh

# 4. 채점과 판정 (CPU)
python experiments/evaluation/score_collisions.py --arm &lt;arm&gt;_pred
python experiments/evaluation/judge_collision_gates.py --arms baseline_pred \\
    dual_u40_v2_pred tyr_u40_r_pred tyrK_pred</code></pre>
<p class="note">기존 실행은 쓸 수 없다 &mdash; <code>run_baseline.py</code>가 8개 샘플 경로를
계산하고 ADE/FDE 스칼라만 남긴 뒤 버리기 때문에, 네 arm을 <code>--save-pred</code>로 다시 돌렸다
(cvlab20 4카드, arm당 약 70분). 궤적은 클립당 7 KB다.</p>
</div></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--date", default="2026-09-09")
    args = ap.parse_args()
    coll, gates, geom = load()
    plot_dir = O / "collision_gates" / "plots"
    plots(coll, gates, plot_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build(coll, gates, geom, plot_dir, args.date))
    print(f"wrote {args.out}  ({args.out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
