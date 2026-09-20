"""Every alpasim metric, eight arms, two scene sets, as one HTML reference table.

This is a lookup table, not an argument: no metric is filtered out, including the ones
that are constant by construction, so that "what did arm X do on metric Y" has an answer
here instead of requiring a fresh parquet pass.

Values are read from alpasim's OWN aggregation -- `rollout["metrics"]` inside each run's
`aggregate/results-summary.json` -- rather than recomputed from the per-timestep parquets.
That choice is load-bearing and was made after measuring the alternative: recomputing drops
every ABORTED rollout (route sanity-check failures carry a forced score of 0.0 and write no
parquet), which moved the score column +0.010 to +0.025 away from the published number.
alpasim's own output also carries four metrics the parquet path lacks -- in-sim minADE at
four horizons, `duration_frac_20s`, and `offroad_or_collision_at_fault`.

The reduction on top of that is per-rollout -> per-scene mean, then the difference against
the baseline of the SAME set (the parenthesised figure).

Three lookups are load-bearing and each has cost an error in this repo before:

  - `lp_r50`'s 150-scene run is NOT under our runs_root nor the `m2601_merged_` prefix.
  - CoC degeneracy must be keyed on (config, n_scenes), never config alone -- the same
    config ran on both sets with different rates.
  - gate metrics live under `rollout["metrics"]`, not at the top level of the summary.

    python experiments/evaluation/alpasim_full_metrics_report.py \\
        --out reports/evaluation/2026-09-20_alpasim-all-metrics.html
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = Path("/home/cvlab21/project/chan/alpasim-runs")
SOOWON = Path("/mnt/nvme1n1/ad_vla/outputs/soowon/alpasim-analysis/runs_root")
O = Path("/mnt/nvme1n1/ad_vla/outputs/chan")
CACHE = O / "alpasim_all_metrics" / "rollout_metrics.parquet"

ARMS = [
    ("baseline", "baseline", "&mdash;", "무압축"),
    ("dual", "slim_dual_u40_v2", "24.0%", "<code>max(rank I_traj, rank I_CoC)</code>"),
    ("dual+em93.75", "slim_dualexp_u40_em93p75", "39.4%",
     "dual VLM + expert MLP 93.75%"),
    ("coc", "slim_coc_u40_v2", "24.0%", "<code>I_CoC</code> 단독"),
    ("traj", "slim_traj_u40_v2", "24.0%", "<code>I_traj</code> 단독"),
    ("tyr", "slim_tyr_u40_r", "24.0%", "Týr / OSSCAR 출력 재구성"),
    ("llm-pruner", "lp_r50", "25.0%", "LLM-Pruner <code>param_first</code> (외부)"),
    ("wanda", "slim_wanda_u40_v2", "24.0%", "<code>|W|&middot;‖X‖</code>, 기울기 없음"),
]
SETS = [("origin150", "m2601_merged_", 150), ("hard100", "h100_merged_", 100)]

# (key, label, family, unit, better-direction). unit: 'rate' 0/1 -> %, 'pct' already a
# fraction, 'm' metres, 'raw' plain. None direction = no meaningful direction.
SPEC = [
    ("score", "score (종합)", "점수", "raw", "up"),
    ("passed", "pass 비율", "점수", "rate", "up"),
    ("progress_clipped_rel", "progress_clipped_rel", "진행", "raw", "up"),
    ("progress", "progress", "진행", "raw", "up"),
    ("progress_rel", "progress_rel (시간 min)", "진행", "raw", "up"),
    ("progress_rel_to_total", "progress_rel_to_total", "진행", "raw", "up"),
    ("dist_traveled_m", "dist_traveled_m", "진행", "m", "up"),
    ("gt_dist_traveled_m", "gt_dist_traveled_m (씬 속성)", "진행", "m", None),
    ("collision_at_fault", "collision_at_fault (파생)", "충돌", "rate", "down"),
    ("collision_any", "collision_any", "충돌", "rate", "down"),
    ("collision_front", "collision_front", "충돌", "rate", "down"),
    ("collision_lateral", "collision_lateral", "충돌", "rate", "down"),
    ("collision_rear", "collision_rear (추돌당함)", "충돌", "rate", None),
    ("open_loop_collision", "open_loop_collision (3s 예측)", "충돌", "raw", "down"),
    ("offroad_or_collision_at_fault", "offroad_or_collision_at_fault (파생)", "충돌",
     "rate", "down"),
    ("offroad", "offroad", "도로·차선", "rate", "down"),
    ("offroad_or_collision", "offroad_or_collision (파생)", "도로·차선", "rate", "down"),
    ("wrong_lane", "wrong_lane", "도로·차선", "rate", "down"),
    ("min_distance_to_lane_boundary_m", "min_distance_to_lane_boundary_m",
     "도로·차선", "m", None),
    ("dist_to_gt_location", "dist_to_gt_location", "GT 근접", "m", "down"),
    ("dist_to_gt_trajectory", "dist_to_gt_trajectory (4 m 절단 기준)", "GT 근접",
     "m", "down"),
    ("min_ade@0.5s(gt)", "min_ade@0.5s (미산출)", "시뮬 내 궤적", "m", "down"),
    ("min_ade@1.0s(gt)", "min_ade@1.0s (미산출)", "시뮬 내 궤적", "m", "down"),
    ("min_ade@2.5s(gt)", "min_ade@2.5s (미산출)", "시뮬 내 궤적", "m", "down"),
    ("min_ade@5.0s(gt)", "min_ade@5.0s (미산출)", "시뮬 내 궤적", "m", "down"),
    ("min_distance_to_obstacle_m", "min_distance_to_obstacle_m", "기타", "m", None),
    ("duration_frac_20s", "duration_frac_20s (채점 구간 길이)", "기타", "raw", None),
    ("plan_deviation", "plan_deviation", "기타", "raw", None),
    ("safety_monitor_triggered", "safety_monitor_triggered", "기타", "rate", "down"),
    ("eval_relevant", "eval_relevant (전 구간 1)", "기타", "raw", None),
    ("img_is_black", "img_is_black (ASL_SKIP_IMAGES=1)", "기타", "raw", None),
    ("coc_degenerate", "CoC 퇴화율", "CoC", "pct", "down"),
    ("coc_empty", "└ 빈 출력", "CoC", "pct", "down"),
    ("coc_soup", "└ soup", "CoC", "pct", "down"),
    ("coc_len", "CoC 평균 길이 (자)", "CoC", "raw", None),
    ("coc_uniq", "CoC unique ratio", "CoC", "raw", "up"),
]
FAMILIES = ["점수", "진행", "충돌", "도로·차선", "GT 근접", "시뮬 내 궤적", "기타",
            "CoC"]


def run_dir(prefix, cfg, n):
    d = RUNS / f"{prefix}{cfg}"
    if d.exists():
        return d
    if cfg == "lp_r50" and n == 150:
        return SOOWON / "lp_r50"          # see the module docstring
    return None


def build_cache():
    """One row per scored rollout, from alpasim's own aggregation.

    `rollout["metrics"]` is what alpasim wrote after running its modifier chain and each
    metric's own time aggregation, so these are the protocol's numbers rather than a
    reimplementation of them. Aborted rollouts appear here too -- they carry a score and a
    partial metrics dict, and dropping them is what skewed the earlier parquet-based pass.
    """
    rows = []
    for label, prefix, n in SETS:
        for key, cfg, _, _ in ARMS:
            d = run_dir(prefix, cfg, n)
            if d is None:
                print(f"  {label:10s} {key:13s} 런 없음")
                continue
            j = json.loads((d / "aggregate/results-summary.json").read_text())
            n_ab = 0
            for r in j["rollouts"]:
                if r.get("score") is None:
                    continue
                m = {k: v for k, v in (r.get("metrics") or {}).items()
                     if isinstance(v, (int, float))}
                m.update({"set": label, "arm": key, "scene": r["clipgt_id"],
                          "score": r["score"], "passed": float(bool(r.get("passed")))})
                rows.append(m)
                if str(r.get("failure_reason")) not in (
                        "collision_at_fault", "offroad", "None", "none"):
                    n_ab += r.get("failure_reason") is not None
            print(f"  {label:10s} {key:13s} {sum(1 for x in rows if x['set'] == label and x['arm'] == key)}"
                  f" rollout" + (f"  (중단 {n_ab})" if n_ab else ""), flush=True)
    df = pd.DataFrame(rows)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE)
    return df


def coc_lookup():
    """CoC stats keyed on (config, n_scenes) -- never on config alone.

    The breakdown is spread over many analyze_alpasim outputs rather than collected in
    one place, so every metrics.json under outputs/ is scanned and the richest entry per
    key wins. Keying on config alone would silently return the other scene set's rate:
    slim_tyr_u40_r is 5.92% over 150 scenes and 6.33% over hard100.
    """
    out = {}

    def put(cfg, n, rec):
        rec = {k: v for k, v in rec.items() if v is not None}
        if not rec:
            return
        prev = out.get((cfg, n), {})
        if len(rec) >= len(prev):
            out[(cfg, n)] = {**prev, **rec}

    for p in sorted(O.glob("*/metrics.json")):
        try:
            j = json.loads(p.read_text())
        except (ValueError, OSError):
            continue
        if not isinstance(j, dict):
            continue
        # the collector's shape: suites -> arms -> coc (degenerate fraction only)
        for suite in j.get("suites") or []:
            if not isinstance(suite, dict) or "n_scenes" not in suite:
                continue
            for a in suite.get("arms") or []:
                if isinstance(a, dict) and a.get("coc") is not None and "config" in a:
                    put(a["config"], suite["n_scenes"], {"coc_degenerate": a["coc"]})
        # analyze_alpasim's shape: coc -> config -> mean_*_frac
        n = j.get("n_scenes")
        if not isinstance(n, int):
            continue
        coc = j.get("coc")
        if not isinstance(coc, dict):
            continue
        for cfg, v in coc.items():
            if isinstance(v, dict):
                put(cfg, n, {"coc_degenerate": v.get("mean_degenerate_frac"),
                             "coc_empty": v.get("mean_empty_frac"),
                             "coc_soup": v.get("mean_soup_frac"),
                             "coc_len": v.get("mean_len"),
                             "coc_uniq": v.get("mean_unique_ratio")})

    p = O / "lp150_coc.json"
    if p.exists():
        j = json.loads(p.read_text())
        put("lp_r50", 150, {"coc_degenerate": j["degenerate_frac"],
                            "coc_empty": j["empty_frac"],
                            "coc_soup": j["soup_frac"],
                            "coc_len": j["mean_len"],
                            "coc_uniq": j["mean_unique_ratio"]})
    return out


def fmt(v, unit):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "&mdash;"
    if unit == "rate":
        return f"{100 * v:.1f}%"
    if unit == "pct":
        return f"{100 * v:.2f}%"
    if unit == "m":
        return f"{v:.2f}"
    return f"{v:.3f}" if abs(v) < 100 else f"{v:.1f}"


def fmt_d(v, unit):
    if v is None or (isinstance(v, float) and np.isnan(v)) or v == 0:
        return ""
    if unit in ("rate", "pct"):
        return f"{100 * v:+.1f}pp"
    return f"{v:+.3f}" if abs(v) < 100 else f"{v:+.1f}"


def table(df, coc, label, n):
    vals = arm_values(df, coc, label, n)
    arms = list(vals)
    deltas = {}
    for key in arms:
        deltas[key] = {m: (None if key == "baseline" or vals[key].get(m) is None
                           or vals["baseline"].get(m) is None
                           else vals[key][m] - vals["baseline"][m])
                       for m, *_ in SPEC}

    head = "".join(f"<th>{k}</th>" for k in arms)
    body = ""
    for fam in FAMILIES:
        rows = [s for s in SPEC if s[2] == fam]
        if not rows:
            continue
        body += (f"<tr class='fam'><td colspan='{len(arms) + 1}'>{fam}</td></tr>")
        for m, nice, _, unit, direction in rows:
            tds = ""
            for k in arms:
                v, d = vals[k].get(m), deltas[k].get(m)
                cls = ""
                if d is not None and d != 0 and direction:
                    good = (d > 0) if direction == "up" else (d < 0)
                    cls = " class='good'" if good else " class='bad'"
                ds = fmt_d(d, unit)
                tds += (f"<td>{fmt(v, unit)}"
                        + (f"<br><span{cls}>({ds})</span>" if ds else "") + "</td>")
            body += f"<tr><td><code>{nice}</code></td>{tds}</tr>"
    return (f"<div class='scroll'><table><thead><tr><th>metric</th>{head}</tr></thead>"
            f"<tbody>{body}</tbody></table></div>")


def arm_values(df, coc, label, n):
    """{metric: value} for every arm on one scene set."""
    sub = df[df.set == label]
    out = {}
    for key, cfg, _, _ in ARMS:
        if key not in set(sub.arm):
            continue
        a = sub[sub.arm == key]
        per = {}
        for m, *_ in SPEC:
            if m.startswith("coc_"):
                per[m] = (coc.get((cfg, n)) or {}).get(m)
            elif m in a.columns:
                v = a.groupby("scene")[m].mean()
                per[m] = float(v.mean()) if len(v) else None
            else:
                per[m] = None
        out[key] = per
    return out


def combined_table(df, coc):
    """Every metric x 8 arms x both sets in one grid.

    Each arm gets two adjacent columns, origin150 then hard100, so moving along a row
    compares arms and moving within a pair compares scene sets. Cells carry the absolute
    value with the vs-baseline difference beside it; the per-set tables below repeat the
    same numbers with more room if a cell needs reading closely.
    """
    per = {lab: arm_values(df, coc, lab, n) for lab, _, n in SETS}
    arms = [k for k, *_ in ARMS if all(k in per[lab] for lab, _, _ in SETS)]

    head1 = "".join(f"<th colspan='2' class='grp'>{k}</th>" for k in arms)
    head2 = "".join("<th class='sub'>o150</th><th class='sub h'>h100</th>"
                    for _ in arms)
    body = ""
    for fam in FAMILIES:
        rows = [x for x in SPEC if x[2] == fam]
        if not rows:
            continue
        body += f"<tr class='fam'><td colspan='{2 * len(arms) + 1}'>{fam}</td></tr>"
        for m, nice, _, unit, direction in rows:
            tds = ""
            for k in arms:
                for li, (lab, _, _) in enumerate(SETS):
                    v = per[lab][k].get(m)
                    b = per[lab]["baseline"].get(m)
                    d = None if (k == "baseline" or v is None or b is None) else v - b
                    cls = ""
                    if d is not None and d != 0 and direction:
                        good = (d > 0) if direction == "up" else (d < 0)
                        cls = " good" if good else " bad"
                    ds = fmt_d(d, unit)
                    tds += (f"<td class='cell{' h' if li else ''}'>{fmt(v, unit)}"
                            + (f"<span class='d{cls}'>{ds}</span>" if ds else "")
                            + "</td>")
            body += f"<tr><td class='mname'><code>{nice}</code></td>{tds}</tr>"
    return ("<div class='scroll'><table class='combined'><thead>"
            f"<tr><th rowspan='2'>metric</th>{head1}</tr><tr>{head2}</tr></thead>"
            f"<tbody>{body}</tbody></table></div>")


CSS = """
:root{--bg:#FAF9F5;--card:#FFF;--code:#F0EEE6;--ink:#29261B;--muted:#6B6555;
--acc:#D97757;--bd:#E8E6DC;--stripe:#F5F4EF;--good:#008300;--bad:#b0402a;
--warn:#eda100;--fam:#F0EEE6}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);margin:0;font-size:16px;line-height:1.7;
padding:3rem 1.5rem 5rem;font-family:"Pretendard","Inter",-apple-system,BlinkMacSystemFont,
"Segoe UI","Malgun Gothic",sans-serif}
.container{max-width:1180px;margin:0 auto}
header{margin-bottom:2rem;border-bottom:2px solid var(--acc);padding-bottom:1.2rem}
.eyebrow{color:var(--acc);font-size:.78rem;letter-spacing:.12em;text-transform:uppercase;
font-weight:600}
h1{font-size:1.9rem;line-height:1.25;margin:.4rem 0 .8rem;text-wrap:balance}
.meta{color:var(--muted);font-size:.84rem;line-height:1.9}
h2{font-size:1.28rem;margin:2.4rem 0 .8rem;display:flex;align-items:baseline;gap:.6rem}
h2 .num{color:var(--acc);font-size:.95rem;font-weight:700}
code{background:var(--code);padding:.08em .3em;border-radius:3px;font-size:.86em}
pre{background:var(--code);padding:1rem;border-radius:6px;overflow-x:auto;font-size:.8rem;
line-height:1.55}
table{border-collapse:collapse;width:100%;font-size:.8rem;
font-variant-numeric:tabular-nums}
th,td{padding:.4rem .5rem;text-align:right;border-bottom:1px solid var(--bd);
white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{font-size:.74rem;text-transform:none;color:var(--muted);
border-bottom:2px solid var(--bd);position:sticky;top:0;background:var(--bg)}
tbody tr:nth-child(even){background:var(--stripe)}
tr.fam td{background:var(--fam);font-weight:700;font-size:.78rem;letter-spacing:.04em;
color:var(--ink);border-bottom:1px solid var(--bd)}
tbody tr.fam:nth-child(even) td{background:var(--fam)}
.scroll{overflow-x:auto;margin:1rem 0}
.good{color:var(--good);font-size:.74rem}.bad{color:var(--bad);font-size:.74rem}
table.combined{font-size:.74rem}
table.combined th.grp{text-align:center;border-bottom:1px solid var(--bd);
border-left:2px solid var(--bd);font-weight:700;color:var(--ink)}
table.combined th.sub{font-size:.68rem;font-weight:400;padding:.15rem .4rem}
table.combined th.sub.h,table.combined td.cell.h{background:#F3F1EA}
table.combined th.sub:not(.h),table.combined td.cell:not(.h){
border-left:2px solid var(--bd)}
table.combined td.cell{padding:.3rem .4rem}
table.combined td.mname{position:sticky;left:0;background:var(--bg);z-index:1}
table.combined tbody tr:nth-child(even) td.mname{background:var(--stripe)}
table.combined tr.fam td{position:sticky;left:0}
span.d{display:block;font-size:.66rem;color:var(--muted);line-height:1.2}
span.d.good{color:var(--good)}span.d.bad{color:var(--bad)}
.dim{color:var(--muted)}
.callout{background:var(--card);border-left:3px solid var(--acc);padding:1rem 1.2rem;
margin:1.2rem 0;border-radius:0 6px 6px 0}
.warn{background:var(--card);border-left:3px solid var(--warn);padding:1rem 1.2rem;
margin:1.2rem 0;border-radius:0 6px 6px 0}
.note{font-size:.88rem;color:var(--muted)}
@media (max-width:640px){body{padding:2rem 1rem 3rem}h1{font-size:1.4rem}}
"""


def build(df, coc, date):
    arm_rows = "".join(
        f"<tr><td><code>{k}</code></td><td>{b}</td><td class='dim'>{d}</td></tr>"
        for k, _, b, d in ARMS)
    return f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>폐루프 전체 지표표</title><style>{CSS}</style></head>
<body><div class="container">
<header>
  <div class="eyebrow">Model Compression &middot; 폐루프 레퍼런스</div>
  <h1>폐루프 전체 지표표</h1>
  <div class="meta">
    {date} &middot; arm {len(ARMS)}개 &times; 씬 집합 2개 &times; alpasim 전 지표
    &middot; origin150 (150씬 &times; 2 rollout) / hard100 (100씬 &times; 2 rollout)
    &middot; rollout {len(df):,}개 &middot; Ada 4&ndash;7,
      <code>DRIVER_OMP_THREADS=8</code>
    &middot; 캐시 <code>outputs/alpasim_all_metrics/</code>
  </div>
</header>

<h2><span class="num">1.</span>arm</h2>
<div class="scroll"><table><thead><tr><th>arm</th><th>제거</th><th>기준</th>
</tr></thead><tbody>{arm_rows}</tbody></table></div>
<p class="note"><code>dual+em93.75</code> 만 예산이 다르다(39.4%) &mdash; VLM 절반은
<code>dual</code> 과 비트 동일하고 expert MLP 를 93.75% 추가로 자른다.
<code>llm-pruner</code> 는 외부 구현이라 25.0%로 1%p 크다. 나머지 다섯은 24.0%
동일이다.</p>

<h2><span class="num">2.</span>각 칸의 뜻</h2>
<p>절대값 위, 괄호 안이 <strong>같은 세트의 무압축 대비 차</strong>다. 비율 지표는
퍼센트포인트, 나머지는 원단위. 색은 그 지표의 방향에 맞췄다(진행은 클수록, 게이트는
작을수록 초록). 방향이 정의되지 않는 지표는 색을 넣지 않았다 &mdash; 예를 들어
<code>collision_rear</code>(추돌<em>당함</em>)와 <code>gt_dist_traveled_m</code>(씬 속성)
은 좋고 나쁨이 없다.</p>
<pre>값의 출처: 각 런의 aggregate/results-summary.json 안 rollout["metrics"]
          = alpasim 이 <b>직접 집계한 결과</b> (재계산 아님)
집계:     rollout -&gt; 씬 평균 -&gt; 같은 세트 무압축과의 차</pre>
<div class="callout">
  <p><strong>parquet 에서 재계산하지 않은 이유.</strong> 처음에는 per-timestep parquet 에서
  modifier 체인을 재현해 계산했는데, 그러면 <strong>중단된 rollout 이 통째로 빠진다</strong>
  &mdash; 경로 정합성 검사 실패는 강제 0.0 점을 받고 parquet 을 남기지 않는다. 그 결과
  score 열이 출하값보다 <strong>+0.010&ndash;+0.025</strong> 높게 나왔다(hard100 의
  <code>dual</code> 은 0.620 대 출하 0.596). alpasim 자신의 집계를 읽으면 그 문제가 없고,
  parquet 경로에 없던 지표 넷(시뮬 내 minADE 4개 지평,
  <code>duration_frac_20s</code>, <code>offroad_or_collision_at_fault</code>)도 함께
  얻는다.</p>
</div>

<h2><span class="num">3.</span>8 arm 통합 (두 세트 나란히)</h2>
<p>arm 마다 두 칸이다 &mdash; 왼쪽이 <strong>origin150</strong>, 오른쪽(음영)이
<strong>hard100</strong>. 가로로 읽으면 arm 끼리, 한 쌍 안에서 읽으면 씬 집합끼리
비교된다. 각 칸은 절대값과 그 아래 무압축 대비 차다. 좁은 칸이라 아래 4·5절이 같은
수치를 더 넓게 반복한다.</p>
{combined_table(df, coc)}
<p class="note">두 세트의 절대값을 <strong>가로질러</strong> 비교하지는 말 것 &mdash;
난이도가 다르다(무압축 score 0.750 대 0.510). 한 쌍 안에서 읽을 값은 괄호 안의
<em>무압축 대비 차</em>이고, 그것이 집합 간 비교가 성립하는 유일한 형태다.</p>

<h2><span class="num">4.</span>origin150 (150씬 &times; 2 rollout)</h2>
{table(df, coc, "origin150", 150)}

<h2><span class="num">5.</span>hard100 (100씬 &times; 2 rollout)</h2>
{table(df, coc, "hard100", 100)}

<h2><span class="num">6.</span>이 표를 읽을 때</h2>
<div class="warn">
  <p><strong>게이트를 <code>progress</code> 없이 읽으면 부호가 뒤집힌다.</strong>
  <code>wanda</code> 는 hard100 에서 <code>offroad</code> 가 전 arm 최저인데 점수는
  꼴찌다 &mdash; <code>progress</code> 가 최저라서다. 못 움직이는 arm 은 도로를 벗어날
  일도, 장애물에 가까워질 일도 없다. 같은 이유로
  <code>min_distance_to_obstacle_m</code> 은 잘 달리는 arm 일수록 <em>작다</em>.</p>
  <p><strong>충돌 계열은 개별 arm 쌍을 가르지 못한다.</strong> hard100 28쌍 중
  <code>collision_at_fault</code> 가 분해한 쌍은 <strong>0개</strong>다(별도 측정).
  비율은 맥락으로 읽고 두 arm 사이의 안전 주장 근거로 쓰지 말 것.</p>
  <p><strong>절대값은 두 세트 모두 낙관적이다.</strong> origin150 은
  <code>public_2601</code> 913씬보다 쉽고(무압축 0.742 대 0.660, 과실 충돌 2.0% 대
  5.5%), hard100 은 반대쪽 꼬리다. 두 세트를 가로질러 비교하지 말고 각 세트 안의
  대응 델타를 읽을 것.</p>
  <p><strong>모든 값에 4 m 절단이 적용돼 있다.</strong> 자차가 GT 경로에서 4 m 벗어난
  뒤는 채점되지 않는다 &mdash; 무압축 rollout 의 57.7%가 이에 걸리고 시뮬 시간의
  16.2%가 버려진다. 이 표는 출하 프로토콜을 재현한 것이지 그 프로토콜을 검증한 것이
  아니다.</p>
  <p><strong>상수인 지표도 빼지 않고 실었다.</strong> <code>eval_relevant</code> 는
  절단 후 전 구간 1, <code>img_is_black</code> 은
  <code>ALPASIM_ASL_SKIP_IMAGES=1</code> 이라 이미지 채점기가 값을 만들지 않아 0,
  <code>safety_monitor_triggered</code> 는 전 rollout 0 이다. 빈 칸이 아니라 0 이라는
  것을 확인할 수 있어야 레퍼런스로 쓸 수 있다.</p>
  <p><strong><code>min_ade@*</code> 네 행이 비어 있는 것은 결측이 아니라 미산출이다.</strong>
  alpasim 의 지표 스키마에는 네 지평이 선언돼 있지만 우리 런에서는
  <strong>값이 한 번도 채워지지 않았다</strong> &mdash; 두 세트 전 arm 에서 0/300, 0/200
  rollout 이다. 그래서 이것은 &ldquo;쓸지 말지 고를 수 있는 지표&rdquo;가 아니라
  <em>해당 채점기를 켜고 전 런을 다시 돌려야 생기는 값</em>이다. 폐루프 안에서 잰 궤적
  정확도라 이 연구의 개루프&ndash;폐루프 해리 주제와 맞닿는 만큼, 필요하면 그 비용을
  먼저 인정하고 계획해야 한다.</p>
</div>

<h2><span class="num">7.</span>재현</h2>
<pre>python experiments/evaluation/alpasim_full_metrics_report.py \\
    --out reports/evaluation/{date}_alpasim-all-metrics.html

# rollout parquet 을 다시 읽어 캐시를 새로 만들려면 (수천 개, 몇 분)
python experiments/evaluation/alpasim_full_metrics_report.py --rebuild --out ...</pre>
<p class="note">새 arm 이 생기면 <code>ARMS</code> 에 한 줄 추가하고
<code>--rebuild</code> 하면 두 표가 함께 갱신된다. GPU 는 쓰지 않는다.</p>
</div></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--date", default="2026-09-20")
    args = ap.parse_args()
    if args.rebuild or not CACHE.exists():
        print("rollout parquet 읽는 중")
        df = build_cache()
    else:
        df = pd.read_parquet(CACHE)
        print(f"캐시 사용 ({len(df)} rollout)")
    out = args.out if args.out.is_absolute() else Path.cwd() / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(df, coc_lookup(), args.date))
    print(f"wrote {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
