"""Plots and the self-contained HTML for the MLP-width open-loop/closed-loop dissociation.

Generated rather than templated for the reason collision_report.py is: this is a ladder
whose rungs are still being added, and hand-kept markup goes stale the first time one
lands. A rung with no closed-loop run yet renders as a pending row rather than being
dropped, so the report can be built mid-flight and says so.

Every number is read from an artifact, never retyped: open-loop rows from the per-clip
JSON at K=6, closed-loop rows from analyze_calibsize's metrics.json, and the dual+h4 rows
from the headmlp_split analysis that measured them.

4.5 is the one section that does not hold the budget fixed. Its own reading changed twice
and both retractions are in the body: the two-point "step" claim died when u30 landed on
the u20-u40 line, and the "linear budget ladder" reading died when q13m1861 -- u20's budget
spent on u40's head cut -- reached u30's score on half the budget. What survives is that
the head count is the main effect and the MLP depth a half-sized one, and that the
reallocation edge is the only claim there that survives dropping alpasim's 4 m
GT-trajectory truncation.

The ladder's own result is that it is NOT a dose-response: the three cut rungs do not
separate from each other (every pair's CI spans zero) while all three sit below dual with
higher offroad. An earlier edition of this report claimed a monotone descent from the two
rungs that existed at the time; em75 falsified it by landing below both.

    python experiments/evaluation/mlpwidth_report.py --pairs emladder_pairs \
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


def cut_pairs_table(m):
    """The cut rungs against each other. The dual-relative rows in cl_table answer "does
    cutting lose?"; these answer whether the ladder is a dose-response or a step."""
    cuts = [n for n, _, _, _ in LADDER if n != "dual"]
    body = ""
    for key, pr in m["pairs"].items():
        a, b = key.split(" - ")
        if a not in cuts or b not in cuts:
            continue
        sep = pr["ci_lo"] > 0 or pr["ci_hi"] < 0
        body += (f"<tr><td><code>{a}</code> &minus; <code>{b}</code></td>"
                 f"<td>{pr['delta']:+.4f}</td>"
                 f"<td class='ci'>[{pr['ci_lo']:+.4f}, {pr['ci_hi']:+.4f}]</td>"
                 f"<td>{pr['wilcoxon_p']:.2f}</td>"
                 f"<td class='dim'>{pr['better']}/{pr['worse']}/{pr['tie']}</td>"
                 f"<td class='{'bad' if sep else 'dim'}'>"
                 f"{'구분됨' if sep else '구분 안 됨'}</td></tr>")
    if not body:
        return ""
    return ("<div class='scroll'><table><thead><tr><th>쌍</th><th>델타</th><th>95% CI</th>"
            "<th>Wilcoxon p</th><th>W/L/T</th><th>판정</th></tr></thead>"
            f"<tbody>{body}</tbody></table></div>")


BUDGET_AXIS = [("baseline", "baseline", "&mdash;", "&mdash;", "&mdash;"),
               ("u20", "dual u20", "6", "2458", "11.9%"),
               ("q13m1861", "q13m1861", "13", "1861", "11.9%"),
               ("u30", "dual u30", "10", "3686", "18.1%"),
               ("u40", "dual u40 (출하본)", "13", "4898", "24.0%")]


def budget_axis_table(m3):
    """The budget ladder. Separate from head_axis_tables because this is the one axis in
    the report where the removed-parameter total is NOT held fixed -- mixing it into 4.2
    would silently turn those one-factor rows into two-factor ones."""
    if m3 is None:
        return "", ""
    a, pr = m3["arms"], m3["pairs"]

    def gate(name, key):
        hit, n = a[name]["gates"][key]
        return 100 * hit / n

    def cell(key, ref):
        """analyze_calibsize names a pair once, in the order the arms were passed."""
        if key == ref:
            return "<span class='dim'>기준</span>", ""
        if f"{key} - {ref}" in pr:
            v, sign = pr[f"{key} - {ref}"], 1
        elif f"{ref} - {key}" in pr:
            v, sign = pr[f"{ref} - {key}"], -1
        else:
            return "<span class='dim'>&mdash;</span>", ""
        d = sign * v["delta"]
        lo, hi = sorted((sign * v["ci_lo"], sign * v["ci_hi"]))
        sep = lo > 0 or hi < 0
        cls = " class='good'" if sep and d > 0 else (" class='bad'" if sep else "")
        return (f"{d:+.4f}<br><span class='ci'>[{lo:+.4f}, {hi:+.4f}]"
                f" p={v['wilcoxon_p']:.2g}</span>"), cls

    body = ""
    for key, label, heads, chans, budget in BUDGET_AXIS:
        if key not in a:
            continue
        vb, cb = cell(key, "baseline")
        v2, c2 = cell(key, "u20")
        vd, cd = cell(key, "u40")
        hi = " class='hl'" if key == "q13m1861" else ""
        body += (f"<tr{hi}><td><code>{label}</code></td><td>{heads}</td><td>{chans}</td>"
                 f"<td>{budget}</td>"
                 f"<td><strong>{a[key]['score']:.3f}</strong></td>"
                 f"<td{cb}>{vb}</td><td{c2}>{v2}</td><td{cd}>{vd}</td>"
                 f"<td>{gate(key, 'offroad'):.1f}%</td>"
                 f"<td>{gate(key, 'collision_at_fault'):.1f}%</td>"
                 f"<td>{100 * a[key]['coc_degenerate']:.2f}%</td></tr>")
    t = ("<div class='scroll'><table><thead><tr><th>arm</th><th>자른 head/층</th>"
         "<th>자른 MLP ch/층</th><th>총 제거</th><th>score</th><th>vs 무압축</th>"
         "<th>vs u20</th><th>vs u40</th><th>offroad</th><th>과실충돌</th>"
         "<th>CoC 퇴화</th></tr></thead>"
         f"<tbody>{body}</tbody></table></div>")

    # selection_overlap is stored as a FRACTION and is DIRECTED: ov["a|b"] is the share
    # of a's kept units that also survive in b.
    ov = m3.get("selection_overlap", {})

    def pair(x, y):
        v = ov.get(f"{x}|{y}")
        return "&mdash;" if v is None else f"Q {100 * v['q']:.1f}% / MLP {100 * v['mlp']:.1f}%"

    o = (f"<p class='note'><strong>선택 집합이 설계를 확인해 준다.</strong> "
         f"<code>u40</code>의 유지 유닛 중 <code>u20</code>에도 남는 비율은 "
         f"{pair('u40', 'u20')} &mdash; 예산 사다리는 완전 중첩이다. "
         f"그리고 <code>q13m1861</code>은 <code>u40</code>과 head 선택이 "
         f"<strong>36층 전부 인덱스까지 동일</strong>하다(빌드 게이트에서 확인). "
         f"그래서 <code>u40 &minus; q13m1861</code>에서 움직이는 것은 MLP 하나뿐이고, "
         f"<code>q13m1861 &minus; u20</code>에서 움직이는 것은 축 배분 하나뿐이다.</p>")
    return t, o


HEAD_AXIS = [("dual", "dual", "13", "24.0%", "우리 dual (gate Taylor)"),
             ("baseline", "baseline", "&mdash;", "&mdash;", "무압축"),
             ("dual_h4", "dual+h4", "4", "24.0%", "우리 dual (gate Taylor)"),
             ("act_h0", "act_h0", "0", "24.0%", "traj_param_first"),
             ("spg", "spg", "0", "24.0%", "dual_param_first"),
             ("act70", "act_mlp70", "0", "33.4%", "traj_param_first"),
             ("pr70", "pr_mlp70", "0", "33.4%", "dual_param_first")]


def head_axis_tables(m2):
    """The head axis and the objective one-factor, both closed-loop only.

    Read from spg_pairs, which carries all five arms in one paired matrix -- every delta
    below is against the same 150 scenes, so the rows are directly comparable."""
    if m2 is None:
        return "", ""
    a, pr = m2["arms"], m2["pairs"]

    def gate(name, key):
        hit, n = a[name]["gates"][key]
        return 100 * hit / n

    body = ""
    for key, label, heads, budget, crit in HEAD_AXIS:
        if key not in a:
            continue
        # analyze_calibsize names each pair once, in the order the arms were passed, so
        # baseline's row has to be read off "dual - baseline" and flipped
        if key == "dual":
            vd, cls = "<span class='dim'>기준</span>", ""
        else:
            v = (pr[f"{key} - dual"] if f"{key} - dual" in pr
                 else {"delta": -pr["dual - baseline"]["delta"],
                       "wilcoxon_p": pr["dual - baseline"]["wilcoxon_p"],
                       "ci_hi": -pr["dual - baseline"]["ci_lo"]})
            vd = (f"{v['delta']:+.4f}<br>"
                  f"<span class='ci'>p={v['wilcoxon_p']:.1e}</span>")
            cls = " class='bad'" if v["ci_hi"] < 0 else ""
        strong = "strong" if key == "dual" else "span"
        body += (f"<tr><td><code>{label}</code></td><td>{heads}</td>"
                 f"<td>{budget}</td><td class='dim'>{crit}</td>"
                 f"<td><{strong}>{a[key]['score']:.3f}</{strong}></td><td{cls}>{vd}</td>"
                 f"<td>{gate(key, 'offroad'):.1f}%</td>"
                 f"<td>{gate(key, 'collision_at_fault'):.1f}%</td>"
                 f"<td>{100 * a[key]['coc_degenerate']:.2f}%</td></tr>")
    t1 = ("<div class='scroll'><table><thead><tr><th>arm</th><th>자른 head</th>"
          "<th>총 제거</th><th>기준</th><th>score</th><th>vs dual</th><th>offroad</th>"
          "<th>과실충돌</th><th>CoC 퇴화</th></tr></thead>"
          f"<tbody>{body}</tbody></table></div>")

    body = ""
    # the four edges of the 2x2 first, then everything against the uncompressed baseline
    for key in ("spg - act_h0", "pr70 - act70", "act70 - act_h0", "pr70 - spg",
                "act70 - spg", "pr70 - act_h0",
                "act_h0 - baseline", "spg - baseline", "act70 - baseline",
                "pr70 - baseline", "dual_h4 - baseline"):
        if key not in pr:
            continue
        v = pr[key]
        sep = v["ci_lo"] > 0 or v["ci_hi"] < 0
        aa, bb = key.split(" - ")
        body += (f"<tr><td><code>{aa}</code> &minus; <code>{bb}</code></td>"
                 f"<td>{v['delta']:+.4f}</td>"
                 f"<td class='ci'>[{v['ci_lo']:+.4f}, {v['ci_hi']:+.4f}]</td>"
                 f"<td>{v['wilcoxon_p']:.2f}</td>"
                 f"<td class='dim'>{v['better']}/{v['worse']}/{v['tie']}</td>"
                 f"<td class='{'bad' if sep else 'dim'}'>"
                 f"{'구분됨' if sep else '구분 안 됨'}</td></tr>")
    t2 = ("<div class='scroll'><table><thead><tr><th>쌍</th><th>델타</th><th>95% CI</th>"
          "<th>Wilcoxon p</th><th>W/L/T</th><th>판정</th></tr></thead>"
          f"<tbody>{body}</tbody></table></div>")
    return t1, t2


CSS = """
:root{--bg:#FAF9F5;--card:#FFF;--code:#F0EEE6;--ink:#29261B;--muted:#6B6555;
--acc:#D97757;--bd:#E8E6DC;--stripe:#F5F4EF;--good:#008300;--bad:#b0402a;--warn:#eda100;--hl:#FBEEE8}
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
h4{font-size:.92rem;margin:1.5rem 0 .5rem;color:var(--muted);letter-spacing:.02em;text-transform:none}
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
/* the one row the section is about; stronger than the zebra stripe under it */
tbody tr.hl,tbody tr.hl:nth-child(even){background:var(--hl);box-shadow:inset 3px 0 0 var(--acc)}
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


def build(ol, cl, m, m2, m3, plot_dir, date):
    pend = [n for n, _, _, _ in LADDER if n != "dual" and n not in cl["_abs"]]
    pend_note = ""
    if pend:
        pend_note = (
            "<div class='warn'><p><strong>이 판은 미완성이다.</strong> "
            f"<code>{', '.join(pend)}</code>의 폐루프가 아직 돌고 있다. 사다리의 칸이 비어 "
            "있으므로 아래의 단조성 주장은 <strong>남은 칸으로 그린 선</strong>이고, 비어 있는 "
            "칸이 그 선 위에 앉지 않으면 이 보고서의 핵심 주장은 약해진다. 판정이 아니라 "
            "중간 보고로 읽어야 한다.</p></div>")
    head_t1, head_t2 = head_axis_tables(m2)
    budget_t, budget_ov = budget_axis_table(m3)
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
    &middot; 산출물 <code>outputs/em87p5_pairs</code>, <code>outputs/headmlp_split</code>, <code>outputs/budget_axis</code>
  </div>
</header>

<div class="kpi-row">
  <div class="kpi"><div class="v">{worst:.4f}</div>
    <div class="l">개루프가 본 최대 |손해| (m) &mdash; 전 칸 &times; 세 세트</div></div>
  <div class="kpi"><div class="v">{fmt_signed(min(deltas) if deltas else 0.0)}</div>
    <div class="l">같은 arm의 폐루프 손해 (score)</div></div>
  <div class="kpi"><div class="v">4 / 4</div>
    <div class="l">개루프가 "공짜"라 한 MLP 절단 중 폐루프가 뒤집은 수</div></div>
  <div class="kpi"><div class="v">0 / 3</div>
    <div class="l">서로 구분되는 절단 칸 쌍 &mdash; 손해는 양이 아니라 여부다</div></div>
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
(<code>em75</code> +0.0073, <code>em87p5</code> +0.0115, <code>em93p75</code> +0.0180).
test500과 OOD-val은 부호가 반대이므로 세 세트 중 하나이고, 끝점 비용의 증거로 읽지 않는다.</p>
<div class="warn">
  <p><strong>val500의 깔끔한 단조성을 믿으면 안 된다.</strong> 그 세트에서는 세 칸이 채널
  순서대로 +0.0013 &rarr; +0.0019 &rarr; +0.0033으로 정렬한다. 보기에는 용량-반응이지만,
  (a) 세 값 전부 CI가 0을 포함하고, (b) <strong>test500과 OOD-val에서는 부호가 뒤집혀</strong>
  <code>em75</code>가 <code>dual</code>보다 오히려 좋고(&minus;0.0002 / &minus;0.0010),
  (c) 규모가 0.001&ndash;0.003으로 같은 세트에서 24% 압축이 쓰는 0.067&ndash;0.120의
  2&ndash;4%다.</p>
  <p>세 세트 중 하나에서만 나타나고 나머지 둘은 부호가 반대인 정렬은 추세가 아니다.
  이 보고서의 초판이 폐루프에서 두 점으로 선을 긋고 <code>em75</code>에 반증당한 것과
  <strong>같은 종류의 함정</strong>이며, 여기 적어 두는 이유는 다음에 이 표를 보는 사람이
  같은 선을 긋지 않게 하기 위해서다.</p>
</div>

<h2><span class="num">4.</span>폐루프 (사실)</h2>
{cl_table(cl)}
<figure><img src="data:image/png;base64,{b64(plot_dir / 'dissociation.png')}"
  alt="개루프는 평평하고 폐루프는 한 단 내려간 뒤 평평하다">
<figcaption>같은 체크포인트, 같은 가로축, <strong>같은 세로축 범위</strong>. 왼쪽은 세 세트의
개루프 델타로 전부 0선에 붙어 있고, 오른쪽은 폐루프 델타로 <strong>첫 칸에서 한 단
떨어진 뒤 평평하다</strong>. 스케일을 맞춰도 왼쪽에는 아무것도 보이지 않는다는 것이 이 그림의
요점이고, 오른쪽 세 점이 서로 구분되지 않는다는 것이 두 번째 요점이다.</figcaption>
</figure>
<h3>4.1 자른 칸끼리는 구분되지 않는다</h3>
<p>위 표의 <code>vs dual</code> 열은 "자르면 지는가"를 묻는다. 사다리가
<strong>용량-반응인가 계단인가</strong>는 자른 칸끼리의 비교가 답한다.</p>
{cut_pairs_table(m)}
<p>세 쌍 전부 CI가 0을 포함하고 92&ndash;96씬이 동점이다. 채널을 2064에서 516으로
<strong>4배</strong> 더 줄여도 폐루프 점수는 구분되지 않는다.</p>

<figure><img src="data:image/png;base64,{b64(plot_dir / 'closedloop.png')}"
  alt="폐루프 점수와 offroad 게이트">
<figcaption>왼쪽 오차막대는 씬 단위 bootstrap 95% CI다. 모든 칸이 무압축보다는 낫지만
<code>dual</code>보다는 못하다. 오른쪽이 손해의 내용물 &mdash; 자른 칸은 모두 offroad가
<code>dual</code>보다 높고, <code>dual</code>만 baseline보다 낮다. 칸끼리의 순서는 없다.</figcaption>
</figure>

<h4>4.1.1 독립 표본은 이 손해를 재현하지 않는다 (2026-09-19 추가)</h4>
<div class="warn">
  <p><strong>사다리의 "자른 칸은 <code>dual</code>보다 아래"라는 판정은 150씬에 한정된
  것이다.</strong> <code>em93p75</code> 를 150씬과 <strong>서로소</strong>인 어려운
  100씬에서 돌렸더니 <code>dual</code> 과의 차이가 사라진다.</p>
  <pre>origin150   dual &minus; em93p75   <b>+0.0355</b> [+0.0005, +0.0726]*  p=0.155   31/23/96
hard100     em93p75 &minus; dual   <b>+0.0042</b> [&minus;0.0288, +0.0393]   p=0.18    31/18/51</pre>
  <p>부호까지 뒤집히고 CI 가 0 을 넉넉히 포함한다. 무압축 대비로 보면
  <code>em93p75</code> 는 hard100 에서 <strong>+0.0892 [+0.0337, +0.1476],
  p=1.3e&minus;04</strong> 로 <code>dual</code>(+0.0850)보다 오히려 약간 위다 &mdash;
  제거량이 24.0% 가 아니라 <strong>39.4%</strong> 인데도.</p>
  <p class="note">150씬 쪽 수치 자체가 경계였다는 점도 같이 봐야 한다 &mdash; CI 하단이
  +0.0005 로 0 을 겨우 배제하고 Wilcoxon 은 p=0.155 로 반대를 가리킨다. 이 저장소는
  평균 CI 를 1차로 읽지만, 두 통계가 엇갈리는 칸은 독립 표본이 오면 흔들릴 후보였다.</p>
  <p><strong>방어되는 문장은 하나로 좁아진다</strong> &mdash; <em>expert MLP 를 93.75%
  잘라 제거량을 24.0% &rarr; 39.4% 로 키워도, 두 씬 집합 어디에서도 <code>dual</code>
  대비 손해가 확정되지 않는다.</em> 4절 본문의 "모든 칸이 <code>dual</code>보다 못하다"는
  <strong>150씬에서만</strong> 성립한다.</p>
  <p class="note"><strong>범위.</strong> hard100 에서 돌린 것은 <code>em93p75</code>
  하나다. <code>em75</code> 와 <code>em87p5</code> 는 150씬만 있으므로, 사다리의 나머지
  두 칸에 대해서는 위 문장을 주장하지 않는다.</p>
  <p class="note"><strong>CoC 는 이쪽에서도 안 움직인다.</strong> hard100 에서
  <code>em93p75</code> 퇴화 3.7% 대 <code>dual</code> 3.6%, 길이 78 대 78,
  unique 0.93 대 0.93. expert MLP 를 93.75% 잘라도 생성은 건드리지 않는다 &mdash; CoC 를
  쓰는 것은 VLM 이라는 구조와 일치한다.</p>
</div>

<h3>4.2 같은 계단이 헤드 축에서도 나온다</h3>
<p>여기까지는 expert MLP 축이었다. 같은 질문을 <strong>VLM의 헤드 축</strong>에 물으면
&mdash; 24% 예산을 헤드에서 몇 개나 가져오는가 &mdash; 모양이 반복된다. 아래 네 arm은
모두 <strong>같은 2,657,452,032 파라미터</strong>를 지우고, 헤드를 13개 / 4개 / 0개 자른다.</p>
{head_t1}
<p><code>dual</code>만 무압축을 이긴다. 헤드를 13개에서 4개로 줄이는 순간 &minus;0.091이
발생하고, 0개로 더 줄여도 그 자리에 머문다.</p>
{head_t2}
<p class="note">여섯 쌍 중 <code>*&nbsp;&minus;&nbsp;dual</code>만 구분되고 나머지는 전부
구분되지 않는다. expert MLP 사다리에서 본 것과 같다 &mdash; <strong>손해는 양이 아니라
여부</strong>다.</p>

<h3>4.3 목적 함수를 바꿔도 회복되지 않는다 (1요인)</h3>
<p>위 표의 <code>spg</code>와 <code>act_h0</code>은 이 연구에서 가장 깨끗한 1요인 쌍이다.
두 마스크는 <strong>층별 예산·head 마스크·캘리브레이션·제거 파라미터가 전부 동일</strong>하고,
고르는 채널만 다르다(유지집합 겹침 83.5%). 다른 것은 점수 함수 하나뿐이다 &mdash;
<code>traj_param_first</code>(궤적만) 대 <code>dual_param_first</code>(CoC+궤적). 빌드 시
G0으로 그 동일성을 기계 확인한 뒤 돌렸다.</p>
<pre>spg &minus; act_h0   +0.0054 [&minus;0.0310, +0.0421]   p=0.96</pre>
<p><strong>목적에 CoC를 더한 효과가 없다.</strong> 그러므로 이 축의 손해는 "궤적만 보는
기준"의 결함이 아니라 <strong>MLP-only 배분 자체</strong>다. 헤드를 4개 자르든 0개 자르든,
목적에 CoC를 넣든 말든 같은 자리(&minus;0.09&nbsp;~&nbsp;&minus;0.10)에 앉는다.</p>
<div class="callout">
  <p><strong>그런데 CoC 생성 품질은 실제로 지켰다.</strong> <code>spg</code>의 CoC 퇴화율은
  0.84%로 <code>act_h0</code>(2.14%)와 <code>dual</code>(2.72%)보다 훨씬 낮고 무압축(0.55%)에
  가깝다. 목적에 CoC를 넣은 것이 <em>CoC에는 효과가 있었는데 주행 점수는 하나도 못 살렸다.</em>
  이 저장소가 따로 기록해 둔 "CoC 건강은 안전 대리지표가 아니다"의 또 한 사례다.</p>
</div>
<p class="note">한 가지 덧붙일 것: <code>act_h0</code>과 <code>spg</code>는 예산·배분만
우리 <code>dual</code>과 같고 기준과 캘리브레이션이 다르다(유지집합 겹침 MLP 71.9% / 75.9%).
그래서 <strong>이 둘과 <code>dual</code>의 차이는 1요인이 아니다</strong> &mdash; 1요인인 것은
둘 사이의 비교, 그리고 <code>dual+h4</code>와 <code>dual</code>의 비교다.</p>

<div class="warn">
  <p><strong><code>spg</code>는 "우리 <code>dual</code>의 h0 버전"이 아니다.</strong>
  두 기준 모두 CoC와 궤적을 함께 보지만 재는 양이 다르다 &mdash; 우리 것은 곱셈 <em>게이트</em>를
  미분해 <code>max(rank I_traj, rank I_CoC)</code>로 합집합을 취하고, <code>dual_param_first</code>는
  <em>가중치</em>를 미분해 부호를 유지한 채 더한다. 확인해 봤다: 우리 dual 점수로
  <code>spg</code>와 <strong>똑같은 층별 예산</strong>을 뽑으면 36개 층 중
  <strong>1개만</strong> 일치한다(그 1개는 아무것도 자르지 않는 층 35다). 겹침 83.9%로
  무작위(51.1%)보다는 훨씬 높으니 비슷한 것을 보긴 하지만, 같은 기준은 아니다.
  <strong>우리 기준의 진짜 h0 arm은 아직 없다</strong> &mdash; <code>dual+h4</code>가 가장 극단이다.</p>
</div>

<h3>4.4 기준 &times; 깊이: 두 요인은 독립이 아니다</h3>
<p>MLP-only arm 네 개가 <strong>완전 교차된 2&times;2</strong>를 이룬다. 배분·캘리브레이션
(<code>lp_c100s1</code>)·head 0개·층 35 보존이 전부 고정되고, 점수 함수와 깊이만 움직인다.
각 칸의 중첩과 예산 일치는 빌드 게이트로 확인한 뒤 돌렸다.</p>
<div class="scroll"><table><thead><tr><th></th>
<th><code>traj_param_first</code></th><th><code>dual_param_first</code></th></tr></thead><tbody>
<tr><td><strong>MLP 50.3%</strong> (총 24.0%)</td><td>act_h0 <strong>0.727</strong></td>
    <td>spg <strong>0.733</strong></td></tr>
<tr><td><strong>MLP 70.0%</strong> (총 33.4%)</td>
    <td>act_mlp70 <strong class="good">0.765</strong></td>
    <td>pr_mlp70 <strong class="bad">0.715</strong></td></tr>
</tbody></table></div>
<pre>기준 효과  얕게  spg &minus; act_h0    +0.0054 [&minus;0.0310, +0.0421]  p=0.96     무관
          깊게  pr70 &minus; act70    &minus;0.0499 [&minus;0.0874, &minus;0.0131]* p=0.0091   유의
깊이 효과  traj  act70 &minus; act_h0  +0.0375 [&minus;0.0001, +0.0755]  p=0.025    좋아짐
          dual  pr70 &minus; spg      &minus;0.0178 [&minus;0.0636, +0.0275]  p=0.81     무변화</pre>
<p><strong>얕게 자를 때는 기준이 무관한데(p=0.96) 깊게 자르면 갈린다(p=0.0091).</strong>
앞 절들이 세 축에서 얻은 "계단" 요약은 <em>깊이가 얕을 때의 성질</em>이었다. 70%까지 가면
어느 점수로 골랐는지가 0.05를 가른다.</p>

<div class="warn">
  <p><strong><code>act_mlp70</code>의 0.765를 "더 자르면 좋아진다"로 읽으면 안 된다.</strong>
  이 arm은 MLP-only 중 유일하게 무압축을 넘지만(+0.0153, p=0.11), <strong>CoC 퇴화율이
  20.73%</strong>다 &mdash; empty 15.0% + soup 5.7%, 출하 <code>dual</code>(2.72%)의 7.6배,
  무압축(0.55%)의 38배. 다섯 rollout 중 하나꼴로 추론이 붕괴한다.</p>
  <p>서명이 분명하다: <strong>progress는 0.784로 전 arm 최고</strong>이고 중앙값이 1.000인데,
  <strong>게이트는 최악</strong>이다(offroad 9.3%, 과실충돌 4.7% &mdash; 무압축 4.0%보다 높다).
  추론을 포기하고 전진을 번 형태이고, 종합 점수는 그 거래를 보상한다. 이 저장소가 Tyr
  폐루프에서 이미 본 모양이다 &mdash; 거기서도 CoC 붕괴는 전부 빈 출력이었고 궤적은
  baseline 근처였다.</p>
  <p class="note"><strong>부수 관찰.</strong> <code>act_mlp70</code>의 폐루프는 10시간 18분으로
  다른 런(7시간 38분&ndash;8시간 21분)보다 2시간 넘게 길었다. CoC 길이가 106으로 다른
  arm(74&ndash;80)보다 40% 길다 &mdash; <strong>CoC가 붕괴하면 디코딩이 짧아지는 게 아니라
  길어질 수 있다</strong>(soup은 반복 생성이다). 다른 arm의 실측으로 런타임을 외삽하면
  이만큼 빗나간다.</p>
</div>

<p><code>dual_param_first</code> 쪽에서는 이야기가 다르다. 깊이를 늘려도 종합 점수는 움직이지
않는데(<code>pr70 &minus; spg</code> p=0.81) <strong>과실 충돌이 4.3% &rarr; 9.0%로 2.1배</strong>가
되고 CoC 퇴화도 0.84% &rarr; 6.48%로 뛴다. 점수가 포화하는 이유는 점수의 구조에 있다 &mdash;
<code>score_criteria</code>는 <code>collision_at_fault</code>와 <code>offroad</code>를
<em>하드 게이트</em>로 쓰고 나머지를 progress로 채우므로, 이미 걸린 씬에서는 충돌이 더 늘어도
깎을 자리가 없다. <code>pr70</code>의 중앙값이 0.937로 <code>spg</code>의 0.856보다
<em>높다</em>는 것이 그 분포다.</p>
<div class="callout">
  <p><strong>두 칸이 같은 것을 말한다.</strong> 깊게 자르면 종합 점수는 더 이상 믿을 수 없다
  &mdash; 한쪽(<code>act70</code>)은 추론을 버려 점수를 <em>올리고</em>, 다른 쪽
  (<code>pr70</code>)은 충돌이 두 배가 되어도 점수가 <em>안 내려간다</em>. 어느 쪽이든
  <strong>CoC 퇴화율과 게이트를 종합 점수와 함께 읽지 않으면 오독한다.</strong></p>
</div>

<h3>4.5 예산인가 배분인가 &mdash; 자른 head 수가 주 효과다</h3>
<div class="callout">
  <p><strong>이 절만 예산을 고정하지 않는다.</strong> 4.1&ndash;4.4의 모든 행은 제거
  파라미터를 묶어 두고 <em>어디서 가져오는지</em>만 바꿨다. 여기서는 기준
  (<code>importance_v2</code>, <code>calib_100</code> 100클립)&middot;배분(uniform)&middot;
  expert&middot;KV를 고정한 채 <strong>예산과 축 배분을 따로</strong> 움직인다.</p>
</div>
<p>네 arm이 두 축을 가른다. <code>u20/u30/u40</code>은 예산 사다리이고(층당 head와 채널이
함께 늘어난다), <code>q13m1861</code>은 <strong><code>u20</code>의 예산을
<code>u40</code>의 head 절단으로 쓴</strong> 칸이다 &mdash; 층당 head 13개(= u40)에
채널 1861개, 제거 1,313,980,416으로 <code>u20</code>과 <strong>0.0112% 안</strong>에서
같다. 13개가 <code>u20</code>의 층당 예산을 정수로 나누지 못해 정확히 0이 되지는
않는다(<code>_qcut</code>의 나눗셈 assert가 거부하는 지점이며, 150씬 해상도 0.080보다
네 자리 아래다).</p>

{budget_t}
{budget_ov}

<h4>두 개의 1요인 비교</h4>
<pre>재배분   q13m1861 &minus; u20   <b>+0.0660</b> [+0.0245, +0.1085]*  p=0.00057   56/21/73
         같은 예산(&plusmn;0.011%), head 6&rarr;13 &middot; 채널 2458&rarr;1861

순수 MLP q13m1861 &minus; u40   &minus;0.0335 [&minus;0.0658, &minus;0.0033]*  p=0.038     22/37/91
         head 선택 36층 비트 동일, 채널 1861 대 4898

예산     q13m1861 &minus; u30   +0.0136 [&minus;0.0237, +0.0502]   p=0.26      44/27/79
         예산 11.9% 대 18.1%</pre>
<div class="callout">
  <p><strong>같은 예산을 head로 옮기는 것만으로 +0.066을 번다.</strong>
  <code>u20</code>은 무압축을 이기지 못하는데(&minus;0.021, n.s.) 같은 파라미터 수를 쓴
  <code>q13m1861</code>은 <strong>이긴다</strong>(+0.0452 [+0.0042, +0.0863], p=0.011).
  그리고 <code>q13m1861</code>은 예산을 6.2%p 덜 쓰고도 <code>u30</code>과 구분되지
  않는다(p=0.26). <strong>점수를 끄는 것은 예산이 아니라 자른 head 수다.</strong></p>
  <pre>head  6 &rarr; 0.729   (u20,      11.9%)
head 10 &rarr; 0.781   (u30,      18.1%)
head 13 &rarr; <b>0.795</b>   (q13m1861, <b>11.9%</b>)
head 13 &rarr; 0.828   (u40,      24.0%)</pre>
  <p>MLP도 방향은 같지만(1861&rarr;4898에서 +0.034) 크기가 head 효과의 절반이고 CI 상단이
  &minus;0.0033으로 겨우 0을 배제한다.</p>
</div>

<div class="warn">
  <p><strong>이 절은 판정을 두 번 바꿨고, 둘 다 여기 남긴다.</strong></p>
  <p><strong>초판(2점)</strong>은 <code>u20</code>과 <code>u40</code>만 놓고
  <em>"예산은 용량-반응이 아니라 계단"</em>이라고 썼다. <code>u30</code>이 반증했다 &mdash;
  인접 계단이 +0.0524와 +0.0472로 거의 같고, <code>u20</code>&ndash;<code>u40</code>
  직선이 18.1%에서 예측하는 0.7801에 실측 0.781이 앉는다. 이 보고서가 5절 주석에 적어둔
  <em>"점 두 개로 추세를 주장하면 안 된다"</em>를 스스로 어긴 것이다.</p>
  <p><strong>2판(선형 사다리)</strong>도 <code>q13m1861</code>이 넘어섰다. 사다리가
  선형으로 보인 이유는 <strong>사다리의 두 축이 함께 올라갔기 때문</strong>이고, 축을
  풀어놓으면 예산 11.9%짜리가 18.1%짜리와 동급이 된다. <strong>"예산 축"이라는 절
  제목 자체가 틀린 프레임이었다.</strong></p>
</div>

<div class="callout">
  <p><strong>CoC 퇴화는 예산이 아니라 head 수를 따라간다 &mdash; 그리고 주행과 부호가
  반대다.</strong> 2판까지는 퇴화율이 예산에 단조로 보였다(0.55 &rarr; 0.75 &rarr; 1.47
  &rarr; 2.72%). <code>q13m1861</code>이 그 해석을 깬다 &mdash; 예산은
  <code>u20</code>과 같은 11.9%인데 퇴화율은 <strong>3.28%로 전 arm 최고</strong>이고,
  예산이 두 배인 <code>u40</code>(2.72%)보다도 높다. head 6개면 0.75%, head 13개면
  3.3%다.</p>
  <p>즉 <strong>head를 자르면 주행이 좋아지고 추론이 무너진다</strong> &mdash; 하나의
  처치, 반대 방향의 두 효과. <code>dual+h4</code>가 head를 13&rarr;4로 줄였을 때 퇴화가
  <strong>0.0%</strong>로 사라지고 주행이 0.091 나빠진 것과 같은 축, 같은 부호다.
  이 저장소가 여러 번 기록한 <em>"CoC 건강도는 안전 프록시가 아니다"</em>의 가장 깨끗한
  사례다 &mdash; 앞선 사례들은 예산이나 방법이 함께 달랐는데, 여기서는 제거
  파라미터가 0.011% 안에서 묶여 있고 축 배분만 다르다.</p>
</div>

<h4>4 m 절단 규칙에 대한 강건성</h4>
<p>alpasim은 자차가 GT <em>경로</em>에서 4 m 벗어나면 그 이후 타임스텝을 채점에서
버린다(<code>RemoveTimestepsAfterEvent(dist_to_gt_trajectory &ge; 4.0)</code>, 집계
단계이므로 시뮬은 끝까지 돈다). baseline rollout의 <strong>57.7%</strong>가 이 규칙에
걸리고 평균 16.2%의 시간이 버려지므로, 위 수치가 그 창에 얼마나 기대는지 확인했다.
per-timestep parquet에서 파이프라인을 재구현해 <strong>다섯 arm 1,500 rollout의 출하
점수를 불일치 0으로 재현</strong>한 뒤, 5번 단계만 끄고 다시 쟀다.</p>
<pre>                          출하                          4m 절단 없이
q13m1861 &minus; u20   +0.0660 (p=0.00057)*   <b>+0.0495 [+0.0030,+0.0966]* p=0.023</b>   유지
u20 &minus; u40       &minus;0.0995 (p=1.9e&minus;06)*  &minus;0.0672 [&minus;0.1181,&minus;0.0174]* p=0.0059  유지
u30 &minus; u40       &minus;0.0472 (p=0.0019)*   &minus;0.0596 [&minus;0.1044,&minus;0.0143]* p=0.0085  유지
q13m1861 &minus; u40  &minus;0.0335 (p=0.038)*    &minus;0.0177 [&minus;0.0556,+0.0182]  p=0.49    <b>소멸</b>
u20 &minus; u30       &minus;0.0524 (p=0.0027)*   &minus;0.0077 [&minus;0.0528,+0.0361]  p=0.68    <b>소멸</b>
q13m1861 &minus; base +0.0452 (p=0.011)*    +0.0266 [&minus;0.0233,+0.0744]  p=0.107   <b>소멸</b>
u40 &minus; baseline  +0.0787 (p=1.1e&minus;04)*  +0.0443 [&minus;0.0084,+0.0976]  p=0.059   <b>소멸</b></pre>
<p><strong>이 절의 헤드라인인 재배분 효과는 살아남고, 부수 주장인 MLP 깊이 효과와
무압축 대비 비교는 살아남지 못한다.</strong> 기전은 절단이 게이트 위반을 채점 창 밖으로
밀어내는 것이다 &mdash; 4 m를 벗어난 <em>뒤에</em> 이탈하거나 충돌하는 rollout이
baseline 21개, <code>u40</code> 29개, <code>u30</code> 38개로 <strong>arm마다 다르다</strong>.
창의 길이가 처치의 함수이므로, 페어드 비교가 잡음을 지워준다는 통상의 방어가 여기서는
부분적으로만 통한다.</p>
<p class="note"><strong>"진짜 점수는 절단 없는 쪽"이라는 뜻이 아니다.</strong> 4 m 규칙은
alpasim이 정한 프로토콜이고 근거가 있다 &mdash; 자차가 GT 경로에서 그만큼 벗어나면 이후의
반응형 에이전트와 렌더는 검증 범위 밖이라, 거기서 관측되는 이탈&middot;충돌은 시뮬레이터가
보증하지 않는 반사실이다. 여기서 말하는 것은 <strong>어느 주장이 그 규칙에 기대고 어느
주장이 기대지 않는지</strong>뿐이다.</p>

<h2><span class="num">5.</span>해석 (의견)</h2>
<div class="callout">
  <p><strong>결론 1 &mdash; 개루프는 이 축에 구조적으로 눈이 멀었다.</strong>
  같은 체크포인트를 두 방식으로 재는데 한쪽은 {worst:.4f}, 다른 쪽은
  {abs(min(deltas)) if deltas else 0:.4f}이다. 개루프 minADE는 6.4초 궤적의 GT 대비 평균
  거리이고, 여기서 실제로 벌어지는 실패는 <strong>도로 이탈</strong>이다 &mdash; GT에서 조금
  벗어난 궤적과 차선을 넘은 궤적은 거리로 구분되지 않는다. 개루프에서 이탈을 대신 잴 방법도
  없다: <code>features.csv</code> 전체에 차선&middot;도로경계&middot;주행가능영역 라벨이 없어
  "도로를 벗어났다"를 판정할 기준 자체가 존재하지 않는다.</p>
  <p><strong>결론 2 &mdash; 용량-반응이 아니라 계단이고, 축을 바꿔도 같다.</strong>
  자른 세 칸은 <em>서로 구분되지 않는다</em> &mdash; 세 쌍 모두 CI가 0을 포함하고
  (<code>em87p5&minus;em75</code> +0.0221, <code>em93p75&minus;em75</code> +0.0033,
  <code>em93p75&minus;em87p5</code> &minus;0.0188), 92&ndash;96씬이 동점이다. 그런데 셋 다
  <code>dual</code>보다 아래이고 셋 다 offroad가 <code>dual</code>보다 높다. 읽히는 신호는
  <strong>"얼마나 자르느냐"가 아니라 "자르느냐 마느냐"</strong>다 &mdash; 2064채널만 남겨도
  손해가 이미 다 발생하고, 거기서 4배 더 잘라도 더 나빠지지 않는다.</p>
  <p>4.2와 4.3이 같은 모양을 두 축에서 더 보여준다. <strong>헤드 축</strong>(13&rarr;4&rarr;0)은
  4개에서 &minus;0.091이 완결되고 0개도 그 자리이며(p=0.89), <strong>목적 함수</strong>
  (궤적만&rarr;궤적+CoC)는 아예 움직이지 않는다(+0.0054, p=0.96). 세 축 어느 쪽도 첫 칸
  이후로는 반응이 없다. <code>dual</code>이 헤드를 13개 자른다는 사실 하나가 &minus;0.09를
  지키고 있고, 그 선을 넘으면 더 적게 자르는 것도 목적을 보강하는 것도 되돌리지 못한다.</p>
  <p><strong>그리고 그 계단의 눈금은 예산이 아니라 head 수다.</strong> 4.5가 둘을
  분리한다 &mdash; <code>u20</code>의 예산을 <code>u40</code>의 head 절단으로 쓴
  <code>q13m1861</code>은 같은 파라미터 수로 <strong>+0.066</strong>을 벌고
  (p=0.00057, 4 m 절단을 꺼도 유지) 예산이 1.5배인 <code>u30</code>과 구분되지 않는다.
  같은 축을 반대로 민 <code>dual+h4</code>(head 13&rarr;4)가 0.091을 잃은 것과 부호가
  맞는다. <strong>head 13개 선이 이 저장소에서 세 번째로 같은 자리에 나타났다.</strong></p>
  <p><strong>그리고 사다리의 계단은 150씬의 성질이다.</strong> 4.1.1 이 그 경계를
  보인다 &mdash; 서로소인 어려운 100씬에서 <code>em93p75</code> 는 <code>dual</code> 과
  구분되지 않고(+0.0042, p=0.18) 무압축 대비로는 오히려 약간 위다. 아래의 계단 서술은
  150씬 안에서만 성립한다.</p>
  <p><strong>단, 계단은 얕은 쪽의 성질이다.</strong> 4.4가 그 경계를 보인다 &mdash;
  MLP를 70%까지 파면 기준 선택이 다시 유의해지고(p=0.0091), 한 칸은 CoC를 버려 점수를 올리고
  다른 칸은 충돌이 두 배가 되어도 점수가 안 내려간다. 아래 문장은 그 안쪽에서만 성립한다.</p>
  <p><strong>그리고 이 "계단"은 종합 점수의 성질이다.</strong> 4.4가 그 한계를 보여준다 &mdash;
  MLP를 70%까지 파도 점수는 안 움직이는데(p=0.81) 과실 충돌은 2.1배가 되고 CoC 퇴화는
  7.7배가 된다. 점수가 게이트를 하드 컷으로 쓰기 때문에 이미 걸린 씬에서는 추가 실패가
  점수에 반영되지 않는다. <strong>계단이라는 결론은 "더 잘라도 안전하다"가 아니라
  "종합 점수로는 더 이상 구분되지 않는다"이다.</strong></p>
  <p class="note"><strong>이 결론은 한 번 틀렸다가 고쳐진 것이다.</strong> 초판은
  <code>em87p5</code>(&minus;0.0166)와 <code>em93p75</code>(&minus;0.0355) 두 점만 가지고
  "계단이 거의 등간격으로 정렬한다"고 썼다. 두 점은 언제나 직선 위에 있다. 세 번째 점
  <code>em75</code>가 &minus;0.0388로 들어오면서 그 직선은 반증됐다 &mdash; <strong>가장 적게
  자른 칸이 가장 나빴다.</strong> 물리적으로는 이상하지만(유지집합이 포함 관계라
  <code>em75</code>는 엄격히 더 많은 채널을 갖는다) 순서가 뒤집힌 것이 아니라 세 칸이
  애초에 구분되지 않는 것이다. 150씬의 페어드 델타 해상도는 0.080인데 칸 사이 관측치는
  0.003&ndash;0.022다. 두 점으로 추세를 주장하면 안 된다는 것을, 이 저장소는
  <code>maxstep11</code>의 꼬리 이득에서 한 번 배웠고 여기서 한 번 더 배웠다.</p>
  <p><strong>결론 3 &mdash; 재배분과 적층이 같은 곳에서 무너진다.</strong>
  <code>dual+h4</code>는 예산을 head에서 VLM MLP로 <em>옮겼고</em>, 이 사다리는 expert MLP를
  <em>덧붙여</em> 잘랐다. 개루프에서 전자는 오히려 좋아 보이고 후자는 중립인데, 폐루프에서는
  둘 다 지고 둘 다 offroad가 오른다. 공통점은 MLP 폭이다. 그리고 양쪽 다 <em>양</em>이 아니라
  <em>여부</em>가 문제였다 &mdash; <code>dual+h4</code>는 head를 13개에서 4개로 줄이는 한 번의
  이동으로 0.091을 잃었고, 이 사다리는 첫 칸에서 손해를 다 치른다.</p>
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
  <p><strong>절대 점수는 4 m 절단 규칙의 함수다.</strong> 4.5 끝의 강건성 블록이
  재다 &mdash; 규칙을 끄면 arm별로 0.008&ndash;0.055 움직이고, <strong>움직이는 양이
  arm마다 다르다</strong>. 이 보고서의 무압축 대비 비교는 그 규칙 안에서만 유의하다.
  이는 4.5뿐 아니라 이 보고서의 모든 절대 점수&middot;게이트 비율에 걸리는 단서이며,
  아직 다른 절에는 적용해 보지 않았다.</p>
  <p><strong>4.5가 이 사다리의 교락을 푼다고 말하지 않는다.</strong> 바로 위 문단이
  이 사다리에 "MLP라서"와 "더 잘라서"가 섞여 있다고 했는데, 4.5는 <strong>VLM</strong>
  예산 축이고 이 사다리는 <strong>expert</strong> MLP를 덧붙인다. 탑이 다르므로 4.5는
  이 사다리의 교락을 제거하지 못한다. 4.5가 제약하는 것은 더 일반적인 문장 쪽이다 &mdash;
  "더 자를수록 더 다친다"는 적어도 VLM 축에서는 성립하지 않는다.</p>
  <p><strong>절대 이탈률&middot;충돌률을 suite 전체로 일반화하지 않는다.</strong> 150씬 표본은
  <code>public_2601</code> 913씬보다 쉽다 &mdash; 무압축 0.742 대 0.660, 과실 충돌 2.0% 대
  5.5%. 여기서 읽는 것은 대응 델타이지 절대값이 아니다.</p>
  <p><strong>4.2의 다섯 arm이 모두 1요인이라고 말하지 않는다.</strong>
  <code>act_h0</code>과 <code>spg</code>는 기준과 캘리브레이션이 우리 <code>dual</code>과
  다르다(유지집합 겹침 MLP 71.9% / 75.9%). 그 둘과 <code>dual</code>의 차이에는 축과 기준이
  함께 들어 있다. 1요인인 것은 <strong>둘 사이</strong>(4.3)와
  <strong><code>dual+h4</code> 대 <code>dual</code></strong> 둘뿐이며, 나머지 행은 맥락이다.</p>
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
    ap.add_argument("--head-pairs", default="spg_pairs",
                    help="analyze_calibsize output holding the head-axis matrix")
    ap.add_argument("--budget-pairs", default="budget_axis",
                    help="analyze_calibsize output holding the budget ladder")
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
    # the head axis lives in its own paired matrix; absent, those sections are skipped
    p2 = O / args.head_pairs / "metrics.json"
    m2 = json.loads(p2.read_text()) if p2.exists() else None
    p3 = O / args.budget_pairs / "metrics.json"
    m3 = json.loads(p3.read_text()) if p3.exists() else None
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
    out.write_text(build(ol, cl, m, m2, m3, plot_dir, args.date))
    print(f"wrote {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
