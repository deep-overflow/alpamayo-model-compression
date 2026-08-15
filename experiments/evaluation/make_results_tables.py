"""Tables-only reference page for every arm the evaluation harness has measured.

The narrative reports argue one claim each and quote the few numbers that claim needs.
This is the opposite: no argument, no plots -- every open-loop and closed-loop number
for every arm, in tables, so a result can be looked up without re-reading a report or
re-running an analysis.

Everything is recomputed from the per-clip rows (`outputs/<tag>_<set>/<ckpt>_s*of*.json`)
and the merged alpasim runs, not copied from a previous report, so the page cannot drift
from the data. Arms with no rows yet are skipped rather than shown blank.

Usage:
  python experiments/evaluation/make_results_tables.py --out reports/evaluation/results.html
  python experiments/evaluation/make_results_tables.py --arms baseline dual jtraj --out ...
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))
sys.path.insert(0, str(Path(__file__).parent))

import eval_lib as el
from analyze_arms import ARM_ORDER, ARMS
from analyze_baseline import BUCKETS, load_rows

SETS = [("indist", "val 500", "공식 val"), ("test", "test 500", "공식 test"),
        ("ood", "OOD 1,533", "ood_reasoning")]
# open-loop metric -> (column label, decimals, lower-is-better)
METRICS = [("minADE_rollout", "minADE@8", 4), ("minFDE_rollout", "minFDE@8", 4),
           ("nll_self", "NLL(self)", 4)]
OOD_ONLY = [("minADE_tf", "minADE@8 (GT CoC)", 4), ("nll_gtcoc", "NLL(GT CoC)", 4)]
# closed-loop run dirs are named by checkpoint, open-loop rows by arm tag
CLOSED = {"baseline": "baseline", "traj": "slim_traj_u40_v2", "coc": "slim_coc_u40_v2",
          "j": "slim_j_u40_v2", "dual": "slim_dual_u40_v2", "jtraj": "slim_jtraj_u40_v2"}
LONG = {"baseline": "무손상 기준선", "traj": "궤적 Taylor 단독",
        "coc": "CoC NLL Taylor 단독 (라벨 필요)", "j": "J-lens 단독 (라벨 없음)",
        "dual": "max(rank I_traj, rank I_CoC) — 라벨 필요",
        "jtraj": "max(rank I_traj, rank J) — 라벨 없음"}

CSS = """
:root { --bg:#FAF9F5; --card:#FFF; --code:#F0EEE6; --ink:#29261B; --muted:#6B6555;
  --accent:#D97757; --border:#E8E6DC; --stripe:#F5F4EF; --good:#008300; --bad:#b0402a; }
* { box-sizing:border-box; }
body { background:var(--bg); color:var(--ink); margin:0; padding:3rem 1.5rem 5rem;
  font-family:"Söhne","Pretendard","Inter",-apple-system,"Segoe UI","Malgun Gothic",sans-serif;
  font-size:16px; line-height:1.65; }
.container { max-width:1080px; margin:0 auto; }
header { border-bottom:2px solid var(--accent); padding-bottom:1.25rem; margin-bottom:2.5rem; }
.eyebrow { color:var(--accent); font-size:.78rem; font-weight:600; letter-spacing:.08em;
  text-transform:uppercase; margin-bottom:.5rem; }
h1 { font-family:"Tiempos Headline",Georgia,serif; font-size:1.9rem; font-weight:500; margin:0 0 .6rem; }
h2 { font-family:"Tiempos Headline",Georgia,serif; font-size:1.3rem; font-weight:500;
  margin:2.5rem 0 .5rem; }
h2 .num { color:var(--accent); margin-right:.4rem; }
h3 { font-size:1rem; font-weight:600; margin:1.6rem 0 .4rem; }
.meta,.note { color:var(--muted); font-size:.88rem; }
.note { margin:.4rem 0 1rem; }
code { font-family:"Berkeley Mono","SF Mono",Menlo,monospace; font-size:.85em;
  background:var(--code); padding:.12em .35em; border-radius:4px; }
.scroll { overflow-x:auto; }
table { width:100%; border-collapse:collapse; margin:.75rem 0 1.5rem; font-size:.87rem;
  background:var(--card); border:1px solid var(--border); border-radius:8px; }
th { background:var(--code); text-align:left; font-weight:600; padding:.5rem .7rem;
  border-bottom:2px solid var(--border); white-space:nowrap; }
td { padding:.45rem .7rem; border-bottom:1px solid var(--border); white-space:nowrap; }
tbody tr:nth-child(even) { background:var(--stripe); }
tbody tr:last-child td { border-bottom:none; }
.r { text-align:right; font-variant-numeric:tabular-nums; }
.base td { font-weight:600; }
.up { color:var(--bad); } .down { color:var(--good); }
footer { margin-top:3.5rem; padding-top:1.2rem; border-top:1px solid var(--border);
  color:var(--muted); font-size:.85rem; }
"""


def stats(vals):
    a = np.asarray(vals, dtype=float)
    mean, lo, hi = el.paired_bootstrap_ci(a)
    return {"n": len(a), "mean": mean, "ci": [lo, hi], "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90))}


def paired(a_rows, b_rows, metric):
    a = {r["clip_id"]: r for r in a_rows}
    b = {r["clip_id"]: r for r in b_rows}
    ids = [i for i in sorted(set(a) & set(b)) if metric in a[i] and metric in b[i]]
    if not ids:
        return None
    d = np.array([b[i][metric] - a[i][metric] for i in ids])
    mean, lo, hi = el.paired_bootstrap_ci(d)
    return {"n": len(ids), "mean": mean, "ci": [lo, hi], "median": float(np.median(d)),
            "p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0}


def pct(v, base):
    if base in (None, 0) or v is None:
        return ""
    r = (v - base) / base * 100
    return f' <span class="{"up" if r > 0 else "down"}">({r:+.1f}%)</span>'


def sig(p):
    """* when the 95% CI excludes zero -- the only significance mark on the page."""
    return "" if p is None or p["ci"][0] <= 0 <= p["ci"][1] else " <strong>*</strong>"


def table(head, rows, sub=None):
    h = "".join(f"<th class='r'>{c}</th>" if i else f"<th>{c}</th>"
                for i, c in enumerate(head))
    s = f"<tr>{''.join(f'<th class=r>{c}</th>' for c in sub)}</tr>" if sub else ""
    return (f"<div class='scroll'><table><thead><tr>{h}</tr>{s}</thead><tbody>"
            + "\n".join(rows) + "</tbody></table></div>")


def open_loop_tables(rows_by_set, arms):
    out = []
    for metric, mlbl, dg in METRICS:
        head = [mlbl + " — 평균 (baseline 대비)"] + [lbl for _, lbl, _ in SETS]
        body = []
        for arm in arms:
            cells = []
            for s, _, _ in SETS:
                r = rows_by_set.get(s, {}).get(arm)
                base = rows_by_set.get(s, {}).get("baseline")
                if not r:
                    cells.append("<td class='r'>&mdash;</td>")
                    continue
                v = stats([x[metric] for x in r])["mean"]
                extra = "" if arm == "baseline" or not base else pct(
                    v, stats([x[metric] for x in base])["mean"])
                cells.append(f"<td class='r'>{v:.{dg}f}{extra}</td>")
            cls = " class='base'" if arm == "baseline" else ""
            body.append(f"<tr{cls}><td><code>{arm}</code></td>{''.join(cells)}</tr>")
        out.append(f"<h3>{mlbl} 평균</h3>" + table(head, body))
    return "\n".join(out)


def detail_table(rows_by_set, arms):
    body = []
    for s, slbl, _ in SETS:
        for arm in arms:
            r = rows_by_set.get(s, {}).get(arm)
            if not r:
                continue
            cells = [f"<td>{slbl}</td>", f"<td><code>{arm}</code></td>",
                     f"<td class='r'>{len(r)}</td>"]
            for metric, _, dg in METRICS:
                st = stats([x[metric] for x in r])
                cells.append(f"<td class='r'>{st['median']:.{dg}f}</td>")
                cells.append(f"<td class='r'>{st['mean']:.{dg}f}</td>")
                cells.append(f"<td class='r'>[{st['ci'][0]:.{dg}f}, {st['ci'][1]:.{dg}f}]</td>")
            deg = float(np.mean([x["coc_degenerate"] for x in r]))
            cells.append(f"<td class='r'>{deg * 100:.1f}%</td>")
            cells.append(f"<td class='r'>{int(np.median([x['coc_len'] for x in r]))}</td>")
            cls = " class='base'" if arm == "baseline" else ""
            body.append(f"<tr{cls}>{''.join(cells)}</tr>")
    head = ["set", "arm", "n"]
    sub = ["", "", ""]
    for _, mlbl, _ in METRICS:
        head += [mlbl, "", ""]
        sub += ["중앙값", "평균", "95% CI"]
    head += ["CoC 퇴화", "CoC 길이(자)"]   # coc_degenerate() measures the decoded string
    sub += ["", ""]
    h = "".join(f"<th class='r' colspan='1'>{c}</th>" for c in head)
    s = "".join(f"<th class='r'>{c}</th>" for c in sub)
    return (f"<div class='scroll'><table><thead><tr>{h}</tr><tr>{s}</tr></thead><tbody>"
            + "\n".join(body) + "</tbody></table></div>")


def ood_table(rows_by_set, arms):
    r_by = rows_by_set.get("ood", {})
    if not r_by:
        return ""
    body = []
    for arm in arms:
        r = r_by.get(arm)
        if not r:
            continue
        cells = [f"<td><code>{arm}</code></td>"]
        for metric, _, dg in [("minADE_rollout", "", 4)] + [(m, ll, d) for m, ll, d in OOD_ONLY]:
            st = stats([x[metric] for x in r])
            base = r_by.get("baseline")
            extra = "" if arm == "baseline" or not base else pct(
                st["mean"], stats([x[metric] for x in base])["mean"])
            cells.append(f"<td class='r'>{st['mean']:.{dg}f}{extra}</td>")
        gap = stats([x["minADE_rollout"] - x["minADE_tf"] for x in r])
        cells.append(f"<td class='r'>{gap['mean']:+.4f}</td>")
        cls = " class='base'" if arm == "baseline" else ""
        body.append(f"<tr{cls}>{''.join(cells)}</tr>")
    head = ["arm", "minADE@8 (자체 CoC)", "minADE@8 (GT CoC)", "NLL(GT CoC)",
            "자체 − GT"]
    return table(head, body)


def contrast_table(rows_by_set, arms, metric):
    pairs = [(a, b) for i, a in enumerate(arms) for b in arms[i + 1:]]
    body = []
    for a, b in pairs:
        cells = [f"<td><code>{b}</code> − <code>{a}</code></td>"]
        for s, _, _ in SETS:
            r = rows_by_set.get(s, {})
            p = paired(r[a], r[b], metric) if a in r and b in r else None
            if not p:
                cells.append("<td class='r'>&mdash;</td><td class='r'>&mdash;</td>")
                continue
            cells.append(f"<td class='r'>{p['median']:+.4f}</td>")
            cells.append(f"<td class='r'>{p['mean']:+.4f} "
                         f"[{p['ci'][0]:+.4f}, {p['ci'][1]:+.4f}]{sig(p)}</td>")
        body.append(f"<tr>{''.join(cells)}</tr>")
    head = ["페어드 대조"]
    sub = [""]
    for _, lbl, _ in SETS:
        head += [lbl, ""]
        sub += ["중앙값", "평균 [95% CI]"]
    h = "".join(f"<th class='r'>{c}</th>" for c in head)
    s = "".join(f"<th class='r'>{c}</th>" for c in sub)
    return (f"<div class='scroll'><table><thead><tr>{h}</tr><tr>{s}</tr></thead><tbody>"
            + "\n".join(body) + "</tbody></table></div>")


def bucket_table(rows_by_set, arms):
    body = []
    for s, slbl, _ in SETS:
        for arm in arms:
            r = rows_by_set.get(s, {}).get(arm)
            if not r:
                continue
            cells = [f"<td>{slbl}</td>", f"<td><code>{arm}</code></td>"]
            for b in BUCKETS:
                v = [x["minADE_rollout"] for x in r if x["bucket"] == b]
                cells.append(f"<td class='r'>{np.median(v):.4f} <span class='meta'>"
                             f"(n={len(v)})</span></td>" if v else "<td class='r'>&mdash;</td>")
            cls = " class='base'" if arm == "baseline" else ""
            body.append(f"<tr{cls}>{''.join(cells)}</tr>")
    return table(["set", "arm"] + BUCKETS, body)


def closed_loop_table(metrics_paths, arms):
    """Per-scene closed-loop aggregates, read from whichever analyze_alpasim runs exist.

    Several runs are merged because the arms were scored in separate batches; the
    baseline is identical across them (same merged run directory), so a later file
    simply overwrites an identical entry.
    """
    merged, paired_d, coc, n_scenes = {}, {}, {}, set()
    for p in metrics_paths:
        if not p.exists():
            continue
        m = json.loads(p.read_text())
        merged.update(m.get("configs", {}))
        paired_d.update(m.get("paired_vs_baseline", {}))
        coc.update(m.get("coc", {}))
        n_scenes.add(m.get("n_scenes"))
    if not merged:
        return ""
    body = []
    for arm in arms:
        cfg = CLOSED.get(arm)
        c = merged.get(cfg)
        if not c:
            continue
        d = paired_d.get(cfg)
        dcell = "&mdash;"
        if d:
            lo, hi = d["d_score_ci_lo"], d["d_score_ci_hi"]
            star = "" if lo <= 0 <= hi else " <strong>*</strong>"
            dcell = (f"{d['d_score_mean']:+.4f} [{lo:+.4f}, {hi:+.4f}]{star}"
                     f" <span class='meta'>p={d['wilcoxon_p']:.1e}, "
                     f"W/L/T {d['wins']}/{d['losses']}/{d['ties']}</span>")
        deg = coc.get(cfg, {}).get("mean_degenerate_frac")
        degcell = "&mdash;" if deg is None else f"{deg * 100:.1f}%"
        body.append(
            f"<tr{' class=base' if arm == 'baseline' else ''}><td><code>{arm}</code></td>"
            f"<td class='r'>{c['score']:.4f} "
            f"<span class='meta'>[{c['score_ci_lo']:.4f}, {c['score_ci_hi']:.4f}]</span></td>"
            f"<td class='r'>{c['passed'] * 100:.1f}%</td>"
            f"<td class='r'>{c['collision_at_fault']:.3f}</td>"
            f"<td class='r'>{c['offroad']:.3f}</td>"
            f"<td class='r'>{c['progress_clipped_rel']:.3f}</td>"
            f"<td class='r'>{c['dist_to_gt_trajectory']:.2f}</td>"
            f"<td class='r'>{degcell}</td>"
            f"<td class='r'>{dcell}</td></tr>")
    n = " / ".join(str(x) for x in sorted(n_scenes) if x)
    # name the arms that have open-loop rows but no closed-loop run yet: a table that
    # silently lists 4 of 6 arms reads as "these are all of them"
    pending = [a for a in arms if CLOSED.get(a) not in merged]
    note = f"{n}씬 × 2 롤아웃, 롤아웃 → 씬 평균 → baseline과 페어드 차이."
    if pending:
        note += (" 아직 폐루프 미완료: "
                 + ", ".join(f"<code>{a}</code>" for a in pending) + ".")
    return (f"<p class='note'>{note}</p>"
            + table(["arm", "scene score [95% CI]", "pass%", "at-fault 충돌", "offroad",
                     "progress", "d2gt", "CoC 퇴화", "Δ score vs baseline"], body))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=ARM_ORDER)
    ap.add_argument("--closed-metrics", nargs="+", type=Path,
                    default=[REPO / "outputs" / "alpasim_arms_2601" / "metrics.json",
                             REPO / "outputs" / "alpasim_singles_2601" / "metrics.json"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--date", default="2026-08-12")
    args = ap.parse_args()

    rows_by_set = {}
    for s, _, _ in SETS:
        rows_by_set[s] = {}
        for arm in args.arms:
            prefix, tag = ARMS[arm]
            r = load_rows(REPO / "outputs" / f"{prefix}{s}", tag)
            if r:
                rows_by_set[s][arm] = r
    have = [a for a in args.arms if any(a in rows_by_set[s] for s, _, _ in SETS)]
    if not have:
        raise SystemExit("no rows found for any requested arm")

    legend = table(
        ["arm", "기준", "체크포인트"],
        [f"<tr><td><code>{a}</code></td><td>{LONG.get(a, '')}</td>"
         f"<td class='r'><code>{ARMS[a][1]}</code></td></tr>" for a in have])

    closed = closed_loop_table(args.closed_metrics, have)
    body = f"""<header>
  <div class="eyebrow">Alpamayo 1.5 · 모델 압축 연구</div>
  <h1>평가 결과 표 모음</h1>
  <div class="meta">{args.date} · <code>nvidia/Alpamayo-1.5-10B</code> ·
  개루프 공식 val 500 / test 500 / OOD 1,533 · minADE@8 · RTX 5880 Ada ·
  폐루프 alpasim <code>public_2601</code> 150씬 × 2 롤아웃</div>
</header>
<p class="note">서술 없이 숫자만 모은 참조 페이지. 모든 값은 클립 단위 원본 행에서 다시 계산했다.
<strong>*</strong> 는 페어드 95% 부트스트랩 CI가 0을 포함하지 않는 경우.
압축률은 모든 pruned arm이 동일하다 — 11.079B → 8.421B (−2.657B, 24.0%).</p>

<h2><span class="num">1</span>arm 정의</h2>{legend}

<h2><span class="num">2</span>개루프 요약</h2>
<p class="note">괄호는 baseline 대비 상대 변화. 빨강이 악화, 초록이 개선.</p>
{open_loop_tables(rows_by_set, have)}

<h2><span class="num">3</span>개루프 상세</h2>{detail_table(rows_by_set, have)}

<h2><span class="num">4</span>페어드 대조 (minADE@8)</h2>
<p class="note">양수 = 뒤쪽 arm이 더 나쁨. 같은 클립·같은 시드로 짝지은 차이.</p>
{contrast_table(rows_by_set, have, "minADE_rollout")}

<h2><span class="num">5</span>페어드 대조 (NLL, 자체 CoC)</h2>
{contrast_table(rows_by_set, have, "nll_self")}

<h2><span class="num">6</span>OOD 채널 분해</h2>
<p class="note">OOD만 전 arm이 동일한 curated CoC를 받는다 — 궤적 헤드 손상과 추론 채널
손상을 분리할 수 있는 유일한 셋.</p>
{ood_table(rows_by_set, have)}

<h2><span class="num">7</span>시나리오 버킷별 minADE@8 중앙값</h2>
<p class="note">버킷은 GT 경로 기하에서 유도 (우선순위 decel_stop &gt; turn &gt; accel &gt; cruise).</p>
{bucket_table(rows_by_set, have)}

<h2><span class="num">8</span>폐루프 (alpasim)</h2>
{closed or "<p class='note'>아직 폐루프 결과 없음.</p>"}

<footer>생성: <code>experiments/evaluation/make_results_tables.py</code> ·
원본 행: <code>outputs/&lt;arm&gt;_&lt;set&gt;/</code></footer>"""

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        f"<!DOCTYPE html>\n<html lang='ko'>\n<head>\n<meta charset='utf-8'>\n"
        f"<meta name='viewport' content='width=device-width, initial-scale=1'>\n"
        f"<title>평가 결과 표 모음 — Alpamayo 1.5 압축</title>\n<style>{CSS}</style>\n"
        f"</head>\n<body>\n<div class='container'>\n{body}\n</div>\n</body>\n</html>\n")
    print(f"arms: {', '.join(have)}")
    print(f"-> {args.out}  {args.out.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
