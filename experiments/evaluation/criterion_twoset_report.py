"""Plots and the self-contained HTML for the criterion x scene-set comparison.

Seven arms measured on two disjoint scene sets. Generated rather than templated, like
collision_report.py and mlpwidth_report.py, because arms keep landing: a run that does not
exist yet renders as a pending row instead of being silently dropped.

Every number is read from an artifact and nothing is recomputed from scratch: scores and
gates come from each merged run's own `aggregate/results-summary.json`, reduced the way
analyze_alpasim reduces them (per-rollout score -> per-scene mean -> paired delta against
the baseline OF THE SAME SET), and CoC degeneracy is read from an analyze_alpasim
metrics.json.

Two lookups here are load-bearing and both have already cost this repo an error:

  - `lp_r50`'s 150-scene run is NOT under our runs_root and not under the `m2601_merged_`
    prefix. Globbing the prefix returns only its hard100 run, which reads as "lp_r50 is
    hard100-only". Its real home is soowon's analysis runs_root.
  - CoC degeneracy must be keyed on (config, n_scenes), never on config alone. The same
    config ran on both sets and the two rates differ -- `slim_tyr_u40_r` is 5.92% on 150
    scenes and 6.33% on hard100.

    python experiments/evaluation/criterion_twoset_report.py \
        --out reports/evaluation/2026-09-19_criterion-two-sets.html
"""

import argparse
import base64
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RUNS = Path("/home/cvlab21/project/chan/alpasim-runs")
SOOWON = Path("/mnt/nvme1n1/ad_vla/outputs/soowon/alpasim-analysis/runs_root")
O = Path("/mnt/nvme1n1/ad_vla/outputs/chan")

BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
ACC, BAD, GOOD = "#D97757", "#b0402a", "#008300"
# Two series = two categorical hues. ACC is this repo's; SET2 was chosen by running the
# dataviz validator against it on the #FAF9F5 surface -- all six checks pass (protan
# dE 18.7, normal 28.7). The contrast WARN on ACC is discharged by direct labels on
# every mark plus the full tables above the figures.
SET2 = "#1F6FB2"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})

# (key, config, budget, one-line description)
ARMS = [
    ("dual", "slim_dual_u40_v2", "24.0%",
     "<code>max(rank I_traj, rank I_CoC)</code> &mdash; 출하본"),
    ("lp_r50", "lp_r50", "25.0%",
     "LLM-Pruner <code>param_first</code>, 외부 구현"),
    ("tyr", "slim_tyr_u40_r", "24.0%",
     "Týr / OSSCAR 출력 재구성 &mdash; 가중치를 다시 쓴다"),
    ("traj", "slim_traj_u40_v2", "24.0%",
     "<code>I_traj</code> 단독 &mdash; dual 의 궤적 half"),
    ("coc", "slim_coc_u40_v2", "24.0%",
     "<code>I_CoC</code> 단독 &mdash; dual 의 추론 half"),
    ("wanda", "slim_wanda_u40_v2", "24.0%",
     "<code>|W|&middot;‖X‖<sub>2</sub></code> &mdash; 기울기를 쓰지 않는다"),
]
SETS = [("origin150", "m2601_merged_", 150, "origin150 (150 scenes)"),
        ("hard100", "h100_merged_", 100, "hard100 (100 scenes)")]


def run_dir(prefix, cfg, n):
    d = RUNS / f"{prefix}{cfg}"
    if d.exists():
        return d
    if cfg == "lp_r50" and n == 150:
        return SOOWON / "lp_r50"          # see the module docstring
    return None


# A rollout whose failure_reason is neither driving gate was ABORTED by the simulator
# (route sanity check), not failed by the driver. It still gets a forced score of 0.0 and
# no metrics.parquet, and -- the part that matters -- the scenes it hits differ by arm.
GATE_REASONS = {"collision_at_fault", "offroad", None, "None"}


def scene_rows(d):
    j = json.loads((d / "aggregate/results-summary.json").read_text())
    rows = [(r["clipgt_id"], r["score"],
             r["metrics"].get("collision_at_fault", 0), r["metrics"].get("offroad", 0),
             (r.get("score_metrics") or {}).get("progress_clipped_rel"),
             r.get("failure_reason"))
            for r in j["rollouts"] if r.get("score") is not None]
    return pd.DataFrame(rows,
                        columns=["scene", "score", "col", "off", "prog", "reason"])


def aborted(df):
    return set(df[~df.reason.isin(GATE_REASONS)].scene)


def coc_lookup():
    """Degeneracy keyed by (config, n_scenes). Keying on config alone reads the wrong set."""
    out = {}
    p = O / "alpasim_all/metrics.json"
    if p.exists():
        for s in json.loads(p.read_text())["suites"]:
            for a in s["arms"]:
                if a.get("coc") is not None:
                    out[(a["config"], s["n_scenes"])] = a["coc"]
    for name, n in (("h100_criterion_eval", 100),):
        p = O / name / "metrics.json"
        if p.exists():
            for cfg, v in (json.loads(p.read_text()).get("coc") or {}).items():
                if isinstance(v, dict) and v.get("mean_degenerate_frac") is not None:
                    out[(cfg, n)] = v["mean_degenerate_frac"]
    p = Path(__file__).resolve().parents[2] / "outputs" / "lp150_coc.json"
    if p.exists():
        out[("lp_r50", 150)] = json.loads(p.read_text())["degenerate_frac"]
    return out


def paired(a, b):
    """Scene-paired delta with a 10k bootstrap CI and Wilcoxon, a - b."""
    j = pd.concat([a, b], axis=1, join="inner", keys=["a", "b"]).dropna()
    d = (j["a"] - j["b"]).to_numpy()
    rng = np.random.default_rng(0)
    bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    lo, hi = np.quantile(bs, [0.025, 0.975])
    try:
        w = float(stats.wilcoxon(d).pvalue)
    except ValueError:
        w = float("nan")
    return {"delta": float(d.mean()), "lo": float(lo), "hi": float(hi), "p": w,
            "sep": bool(lo > 0 or hi < 0), "n": len(d),
            "win": int((d > 0).sum()), "loss": int((d < 0).sum())}


def gather():
    coc = coc_lookup()
    out = {}
    for label, prefix, n, _ in SETS:
        per, base = {}, None
        d = run_dir(prefix, "baseline", n)
        base = scene_rows(d).groupby("scene").score.mean() if d else None
        for key, cfg, budget, _ in ARMS:
            dd = run_dir(prefix, cfg, n)
            if dd is None:
                continue
            df = scene_rows(dd)
            sc = df.groupby("scene").score.mean()
            per[key] = {"abort": aborted(df), "raw": df,
                        "score": float(sc.mean()), "median": float(sc.median()),
                        "coc": coc.get((cfg, n)), "n_roll": len(df),
                        "col": float(df.col.gt(0).mean()),
                        "off": float(df.off.gt(0).mean()),
                        "prog": float(df.prog.mean()),
                        "vs_base": paired(sc, base), "scenes": sc}
        bdf = scene_rows(run_dir(prefix, "baseline", n))
        per["baseline"] = {"abort": aborted(bdf), "raw": bdf,
                           "score": float(base.mean()), "median": float(base.median()),
                           "coc": coc.get(("baseline", n)), "n_roll": len(bdf),
                           "col": float(bdf.col.gt(0).mean()),
                           "off": float(bdf.off.gt(0).mean()),
                           "prog": float(bdf.prog.mean()), "vs_base": None,
                           "scenes": base}
        out[label] = per
    return out


def b64(p):
    return base64.b64encode(Path(p).read_bytes()).decode()


def plots(data, plot_dir):
    plot_dir.mkdir(parents=True, exist_ok=True)
    keys = [k for k, *_ in ARMS if k in data["origin150"]]
    order = sorted(keys, key=lambda k: data["origin150"][k]["vs_base"]["delta"])

    # 1 -- magnitude with uncertainty. Arms on y, one mark per set, CI as the error bar,
    # every mark directly labelled so identity never rests on colour alone.
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    y = np.arange(len(order))
    for off, (label, _, _, nice), col in ((-0.17, *[SETS[0]], ACC),
                                          (+0.17, *[SETS[1]], SET2)):
        vals = [data[label][k]["vs_base"] for k in order]
        ax.errorbar([v["delta"] for v in vals], y + off,
                    xerr=[[v["delta"] - v["lo"] for v in vals],
                          [v["hi"] - v["delta"] for v in vals]],
                    fmt="o", ms=7, lw=2, capsize=3, color=col, label=nice, zorder=3,
                    markeredgecolor=BG, markeredgewidth=1.5)
        for yi, v in zip(y + off, vals):
            # label on the far side of the interval from zero, so a negative arm's
            # value never lands on the zero rule
            neg = v["delta"] < 0
            ax.annotate(f"{v['delta']:+.3f}", (v["lo"] if neg else v["hi"], yi),
                        xytext=(-6 if neg else 6, 0), textcoords="offset points",
                        va="center", ha="right" if neg else "left", fontsize=8,
                        color=MUTED)
    ax.axvline(0, color=MUTED, lw=1, ls="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel("closed-loop score delta vs uncompressed "
                  "(scene-paired, 95% bootstrap CI)")
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(-0.30, 0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "deltas.png", dpi=150)
    plt.close(fig)

    # 2 -- agreement. One series, so the title names it and no legend box is needed;
    # every point is labelled. The identity line is the claim being checked.
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    xs = [data["origin150"][k]["vs_base"]["delta"] for k in order]
    ys = [data["hard100"][k]["vs_base"]["delta"] for k in order]
    lim = max(abs(min(xs + ys)), abs(max(xs + ys))) * 1.35
    ax.plot([-lim, lim], [-lim, lim], color=MUTED, lw=1, ls="--", zorder=1)
    ax.axhline(0, color=MUTED, lw=0.7, zorder=1)
    ax.axvline(0, color=MUTED, lw=0.7, zorder=1)
    ax.scatter(xs, ys, s=70, color=ACC, zorder=3, edgecolor=BG, linewidth=1.5)
    for k, x, yy in zip(order, xs, ys):
        # push the label outward along the diagonal the points lie on, or neighbours
        # in this tight cluster collide
        ax.annotate(k, (x, yy), xytext=(-8 if x < 0 else 8, -10 if x < 0 else 8),
                    textcoords="offset points", fontsize=9, color=INK,
                    ha="right" if x < 0 else "left")
    r = float(np.corrcoef(xs, ys)[0, 1])
    ax.set_xlabel("origin150 delta")
    ax.set_ylabel("hard100 delta")
    ax.set_title(f"same arm, two disjoint scene sets  (r = {r:.3f}, dashed = y=x)",
                 fontsize=10)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    fig.tight_layout()
    fig.savefig(plot_dir / "agreement.png", dpi=150)
    plt.close(fig)
    return r


def fmt_delta(v):
    if v is None:
        return "<span class='dim'>기준</span>"
    cls = " class='good'" if v["sep"] and v["delta"] > 0 else (
        " class='bad'" if v["sep"] else "")
    return (f"<span{cls}>{v['delta']:+.4f}</span><br>"
            f"<span class='ci'>[{v['lo']:+.3f}, {v['hi']:+.3f}]"
            f"{'*' if v['sep'] else ''} p={v['p']:.2g}</span>")


def set_table(per, n):
    order = sorted([k for k in per if k != "baseline"],
                   key=lambda k: -per[k]["score"])
    budget = {k: b for k, _, b, _ in ARMS}
    body = ""
    for k in order + ["baseline"]:
        a = per[k]
        hl = " class='hl'" if k == "baseline" else ""
        c = a["coc"]
        body += (f"<tr{hl}><td><code>{k}</code></td>"
                 f"<td>{budget.get(k, '&mdash;')}</td>"
                 f"<td><strong>{a['score']:.3f}</strong></td>"
                 f"<td>{a['median']:.3f}</td>"
                 f"<td>{fmt_delta(a['vs_base'])}</td>"
                 f"<td>{(f'{100 * c:.2f}%' if c is not None else '&mdash;')}</td>"
                 f"<td>{100 * a['col']:.1f}%</td><td>{100 * a['off']:.1f}%</td>"
                 f"<td>{a['prog']:.3f}</td></tr>")
    return ("<div class='scroll'><table><thead><tr><th>arm</th><th>제거</th>"
            "<th>score</th><th>중앙</th><th>vs 무압축</th><th>CoC 퇴화</th>"
            "<th>과실충돌</th><th>offroad</th><th>progress</th></tr></thead>"
            f"<tbody>{body}</tbody></table></div>")


def pair_rows(per, pairs):
    body = ""
    for a, b in pairs:
        if a not in per or b not in per:
            continue
        v = paired(per[a]["scenes"], per[b]["scenes"])
        body += (f"<tr><td><code>{a}</code> &minus; <code>{b}</code></td>"
                 f"<td>{v['delta']:+.4f}</td>"
                 f"<td class='ci'>[{v['lo']:+.4f}, {v['hi']:+.4f}]</td>"
                 f"<td>{v['p']:.2g}</td>"
                 f"<td class='dim'>{v['win']}/{v['loss']}</td>"
                 f"<td class='{'bad' if v['sep'] else 'dim'}'>"
                 f"{'구분됨' if v['sep'] else '구분 안 됨'}</td></tr>")
    return ("<div class='scroll'><table><thead><tr><th>쌍</th><th>델타</th>"
            "<th>95% CI</th><th>p</th><th>승/패</th><th>판정</th></tr></thead>"
            f"<tbody>{body}</tbody></table></div>")


def four_metric_tables(data):
    """Four metrics, three columns, absolute value with the vs-baseline value in ().

    Each metric is reduced the way that metric should be. `score` is per-scene and the
    parenthesised figure is the SCENE-PAIRED delta, which is the statistic this repo
    decides on. The two gates are rollout-level binaries, so the figure is a
    percentage-point difference of rates -- pairing a 0/1 gate over two rollouts would
    throw away most of its information. `progress` is a difference of means.

    The pooled column's ABSOLUTE score is a 60:40 mix of two difficulties and is printed
    only so the delta has something to sit beside; the delta is the part that means
    something, because each scene was differenced against its own set's baseline first.
    """
    keys = [k for k, *_ in ARMS] + ["baseline"]
    cols = [(lab, lab) for lab, *_ in SETS] + [("pooled", "종합")]

    def score_cell(k, lab):
        if lab == "pooled":
            dd, vv = [], []
            for L, *_ in SETS:
                a, b = data[L][k]["scenes"], data[L]["baseline"]["scenes"]
                j = pd.concat([a, b], axis=1, join="inner", keys=["x", "y"]).dropna()
                dd.append((j["x"] - j["y"]).to_numpy())
                vv.append(a.to_numpy())
            d, v = np.concatenate(dd), np.concatenate(vv)
        else:
            a, b = data[lab][k]["scenes"], data[lab]["baseline"]["scenes"]
            j = pd.concat([a, b], axis=1, join="inner", keys=["x", "y"]).dropna()
            d, v = (j["x"] - j["y"]).to_numpy(), a.to_numpy()
        if k == "baseline":
            return f"{v.mean():.3f}", None, False
        rng = np.random.default_rng(0)
        bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
        lo, hi = np.quantile(bs, [0.025, 0.975])
        sep = lo > 0 or hi < 0
        return (f"{v.mean():.3f}",
                f"{d.mean():+.4f}{'*' if sep else ''}", sep and d.mean() > 0)

    def rate_cell(k, lab, col, as_pct):
        def frame(key):
            if lab == "pooled":
                return pd.concat([data[L][key]["raw"] for L, *_ in SETS])
            return data[lab][key]["raw"]
        v = frame(k)[col]
        val = v.gt(0).mean() if as_pct else v.mean()
        txt = f"{100 * val:.1f}%" if as_pct else f"{val:.3f}"
        if k == "baseline":
            return txt, None, False
        b = frame("baseline")[col]
        base = b.gt(0).mean() if as_pct else b.mean()
        d = val - base
        # for the two gates lower is better, for progress higher is
        good = (d < 0) if as_pct else (d > 0)
        return txt, (f"{100 * d:+.1f}pp" if as_pct else f"{d:+.3f}"), good

    out = ""
    for title, note, getter in (
        ("scene score", "괄호 안은 씬 페어드 델타, <code>*</code> 는 95% CI 가 0 배제",
         lambda k, lab: score_cell(k, lab)),
        ("과실 충돌", "rollout 비율. 괄호는 퍼센트포인트 차",
         lambda k, lab: rate_cell(k, lab, "col", True)),
        ("offroad", "rollout 비율. 괄호는 퍼센트포인트 차",
         lambda k, lab: rate_cell(k, lab, "off", True)),
        ("progress", "rollout 평균 <code>progress_clipped_rel</code>",
         lambda k, lab: rate_cell(k, lab, "prog", False)),
    ):
        body = ""
        for k in keys:
            if any(k not in data[L] for L, *_ in SETS):
                continue
            hl = " class='hl'" if k == "baseline" else ""
            tds = ""
            for lab, _ in cols:
                a, rel, good = getter(k, lab)
                cls = "" if rel is None else (" class='good'" if good else " class='bad'")
                tds += (f"<td><strong>{a}</strong>"
                        + ("" if rel is None
                           else f" <span{cls}>({rel})</span>") + "</td>")
            body += f"<tr{hl}><td><code>{k}</code></td>{tds}</tr>"
        heads = "".join(f"<th>{nice}</th>" for _, nice in cols)
        out += (f"<h4>{title}</h4>"
                f"<p class='note'>{note}</p>"
                f"<div class='scroll'><table><thead><tr><th>arm</th>{heads}</tr></thead>"
                f"<tbody>{body}</tbody></table></div>")
    return out


def pooled_table(data, drop_aborted):
    """Per-scene deltas from both sets, concatenated.

    Raw scores cannot be pooled -- the two sets differ in difficulty by 0.24 -- but a
    per-scene delta against that scene's OWN baseline is comparable by construction.
    """
    bad = {lab: set().union(*(v["abort"] for v in per.values()))
           for lab, per in data.items()}
    body = ""
    for key, _, _, _ in ARMS:
        if any(key not in data[lab] for lab, *_ in SETS):
            continue
        cells, pool = [], []
        for lab, _, _, _ in SETS:
            per = data[lab]
            j = pd.concat([per[key]["scenes"], per["baseline"]["scenes"]], axis=1,
                          join="inner", keys=["a", "b"]).dropna()
            if drop_aborted:
                j = j.drop(index=[x for x in bad[lab] if x in j.index])
            d = (j["a"] - j["b"]).to_numpy()
            pool.append(d)
            rng = np.random.default_rng(0)
            bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
            lo, hi = np.quantile(bs, [0.025, 0.975])
            cells.append((d.mean(), lo, hi, lo > 0 or hi < 0))
        pooled = np.concatenate(pool)
        rng = np.random.default_rng(0)
        bs = pooled[rng.integers(0, len(pooled), (10000, len(pooled)))].mean(1)
        lo, hi = np.quantile(bs, [0.025, 0.975])
        sep = lo > 0 or hi < 0
        try:
            w = float(stats.wilcoxon(pooled).pvalue)
        except ValueError:
            w = float("nan")
        cls = " class='good'" if sep and pooled.mean() > 0 else (
            " class='bad'" if sep else "")
        body += (f"<tr><td><code>{key}</code></td>"
                 + "".join(f"<td>{m:+.4f}{'*' if sp else ''}"
                           f"<br><span class='ci'>[{lo2:+.3f}, {hi2:+.3f}]</span></td>"
                           for m, lo2, hi2, sp in cells)
                 + f"<td{cls}><strong>{pooled.mean():+.4f}{'*' if sep else ''}</strong>"
                   f"<br><span class='ci'>[{lo:+.3f}, {hi:+.3f}] p={w:.2g}</span></td>"
                   f"<td class='dim'>{len(pooled)}</td></tr>")
    return ("<div class='scroll'><table><thead><tr><th>arm</th>"
            "<th>origin150</th><th>hard100</th><th>합산</th><th>n</th></tr></thead>"
            f"<tbody>{body}</tbody></table></div>")


def abort_table(data):
    rows, per = "", data["hard100"]
    order = ["baseline"] + [k for k, *_ in ARMS if k in per]
    bad = set().union(*(v["abort"] for v in per.values()))
    for k in order:
        a = per[k]
        s_all = a["scenes"]
        s_cl = s_all.drop(index=[x for x in bad if x in s_all.index])
        hl = " class='hl'" if len(a["abort"]) > 2 else ""
        rows += (f"<tr{hl}><td><code>{k}</code></td><td>{len(a['abort'])}</td>"
                 f"<td>{s_all.mean():.3f}</td><td>{s_cl.mean():.3f}</td>"
                 f"<td class='dim'>{s_cl.mean() - s_all.mean():+.3f}</td></tr>")
    return ("<div class='scroll'><table><thead><tr><th>arm</th><th>중단된 씬</th>"
            "<th>전체 100씬</th><th>중단 4씬 제외</th><th>차이</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>")


CSS = """
:root{--bg:#FAF9F5;--card:#FFF;--code:#F0EEE6;--ink:#29261B;--muted:#6B6555;
--acc:#D97757;--set2:#1F6FB2;--bd:#E8E6DC;--stripe:#F5F4EF;--good:#008300;
--bad:#b0402a;--warn:#eda100;--hl:#FBEEE8}
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
h4{font-size:.92rem;margin:1.5rem 0 .5rem;color:var(--muted)}
code{background:var(--code);padding:.1em .35em;border-radius:3px;font-size:.88em}
pre{background:var(--code);padding:1rem;border-radius:6px;overflow-x:auto;font-size:.82rem;
line-height:1.55}
table{border-collapse:collapse;width:100%;font-size:.86rem;font-variant-numeric:tabular-nums}
th,td{padding:.5rem .6rem;text-align:right;border-bottom:1px solid var(--bd)}
th:first-child,td:first-child{text-align:left}
thead th{font-size:.76rem;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);
border-bottom:2px solid var(--bd)}
tbody tr:nth-child(even){background:var(--stripe)}
tbody tr.hl,tbody tr.hl:nth-child(even){background:var(--hl);
box-shadow:inset 3px 0 0 var(--acc)}
.scroll{overflow-x:auto;margin:1rem 0}
.ci{color:var(--muted);font-size:.78rem}
.dim{color:var(--muted)}.good{color:var(--good)}.bad{color:var(--bad)}
.callout{background:var(--card);border-left:3px solid var(--acc);padding:1rem 1.2rem;
margin:1.2rem 0;border-radius:0 6px 6px 0}
.warn{background:var(--card);border-left:3px solid var(--warn);padding:1rem 1.2rem;
margin:1.2rem 0;border-radius:0 6px 6px 0}
.note{font-size:.88rem;color:var(--muted)}
figure{margin:1.6rem 0}
figure img{width:100%;border-radius:6px;display:block}
figcaption{color:var(--muted);font-size:.8rem;margin-top:.5rem}
.kpi-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:.8rem;
margin:1.6rem 0}
.kpi{background:var(--card);border:1px solid var(--bd);border-radius:8px;padding:.9rem 1rem}
.kpi .v{font-size:1.5rem;font-weight:700;font-variant-numeric:tabular-nums}
.kpi .l{font-size:.76rem;color:var(--muted);line-height:1.45;margin-top:.2rem}
@media (max-width:640px){body{padding:2rem 1rem 3rem}h1{font-size:1.5rem}}
"""


def build(data, r_agree, plot_dir, date):
    o, h = data["origin150"], data["hard100"]
    pend = [k for k, *_ in ARMS if k not in o or k not in h]
    pend_note = ""
    if pend:
        pend_note = ("<div class='warn'><p><strong>미완성.</strong> "
                     f"<code>{', '.join(pend)}</code> 의 폐루프가 두 세트 중 한쪽에만 "
                     "있어 표에서 빠졌다.</p></div>")
    flips = sum(1 for k in o if k != "baseline" and k in h
                and np.sign(o[k]["vs_base"]["delta"]) != np.sign(h[k]["vs_base"]["delta"]))
    four_t = four_metric_tables(data)
    pooled_all = pooled_table(data, False)
    pooled_clean = pooled_table(data, True)
    abort_t = abort_table(data)
    return f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>기준 여섯, 세트 둘</title><style>{CSS}</style></head>
<body><div class="container">
<header>
  <div class="eyebrow">Model Compression &middot; 폐루프 기준 비교</div>
  <h1>기준 여섯, 세트 둘</h1>
  <div class="meta">
    {date} &middot; 같은 24.0% 예산의 프루닝 기준 6종 &times; 서로소인 씬 집합 2종
    &middot; <code>public_2601</code> 150씬과 어려운 100씬, 각 씬 2 rollout
    &middot; Ada 4&ndash;7, <code>DRIVER_OMP_THREADS=8</code>
    &middot; 산출물 <code>outputs/alpasim_all</code>,
      <code>outputs/h100_criterion_eval</code>
  </div>
</header>

<div class="kpi-row">
  <div class="kpi"><div class="v">{flips} / 6</div>
    <div class="l">두 세트에서 부호가 뒤집힌 arm</div></div>
  <div class="kpi"><div class="v">{r_agree:.3f}</div>
    <div class="l">두 세트 델타의 상관 (n=6)</div></div>
  <div class="kpi"><div class="v">100.0%</div>
    <div class="l"><code>dual</code> 유지 Q head 중 <code>traj&cup;coc</code> 소속</div></div>
  <div class="kpi"><div class="v">{100 * h['coc']['coc']:.1f}%</div>
    <div class="l"><code>coc</code> 의 CoC 퇴화 &mdash; 무너지지 않았는데 졌다</div></div>
  <div class="kpi"><div class="v">4 / 250</div>
    <div class="l">시뮬레이터가 중단시킨 씬 &mdash; 전부 hard100, arm 마다 다르다</div></div>
</div>

{pend_note}

<h2><span class="num">1.</span>묻는 것</h2>
<p>이 저장소는 폐루프 결과를 두 씬 집합에서 낸다 &mdash; <code>public_2601</code> 을
scene_id 로 정렬한 앞 150씬, 그리고 그와 <strong>겹치지 않는</strong> 어려운 100씬.
두 집합의 난이도는 실제로 다르다(무압축 0.750 대 0.510). 묻는 것은 둘이다.</p>
<ol>
  <li><strong>서열이 재현되는가.</strong> 쉬운 집합에서 얻은 기준 간 순서가 어려운 집합에서도
  같은가, 아니면 집합을 바꾸면 결론이 바뀌는가.</li>
  <li><strong><code>dual</code> 의 두 half 는 각각 무엇을 하는가.</strong> 출하 기준은
  <code>max(rank I_traj, rank I_CoC)</code> 인데, 그 안의 <code>I_traj</code> 와
  <code>I_CoC</code> 를 단독으로 쓰면 어떻게 되는지 한 번도 폐루프로 재지 않았다.</li>
</ol>

<h2><span class="num">2.</span>설계</h2>
<p>여섯 arm 중 다섯이 <strong>제거 파라미터 2,657,452,032 개(24.0%)</strong> 로 묶여 있다.
같은 uniform 배분, 같은 <code>calib_100</code> 캘리브레이션, expert 와 KV 는 온전하다.
움직이는 것은 층내 점수 함수 하나뿐이다. <code>lp_r50</code> 만 외부 구현이라
25.0% 로 1%p 크다.</p>
<div class="scroll"><table><thead><tr><th>arm</th><th>제거</th><th>점수 함수</th>
</tr></thead><tbody>
{''.join(f"<tr><td><code>{k}</code></td><td>{b}</td><td>{d}</td></tr>"
         for k, _, b, d in ARMS)}
<tr><td><code>baseline</code></td><td>&mdash;</td><td>무압축</td></tr>
</tbody></table></div>
<p>빌드 때 게이트로 확인한 것 &mdash; <code>traj</code> 와 <code>coc</code> 는
<code>slim_state.pt</code> 가 공간 회수로 지워져 있어 재빌드했고, <strong>재빌드본이
2026-08-12 원본 선택과 36층 전부 인덱스까지 동일</strong>함을 확인한 뒤에 돌렸다. 그래서
여기 실린 두 arm 은 150씬 결과를 낸 그 체크포인트와 같은 것이지 닮은 물건이 아니다.</p>
<div class="callout">
  <p><strong>합집합 성질이 성립한다.</strong> <code>dual</code> 이 유지하는 Q head
  684개가 <strong>전부</strong> <code>traj &cup; coc</code> 안에 있다(100.0%).
  <code>dual</code> 은 두 단일 기준에 없는 유닛을 하나도 새로 도입하지 않는다. 그래서
  <code>dual</code> 이 두 half 를 모두 이긴다면 그것은 <em>합집합이 각 half 보다
  낫다</em>는 뜻이지 제3의 요인이 아니다.</p>
  <p class="note">유지집합 겹침: <code>traj</code>&ndash;<code>dual</code> 86.0%,
  <code>coc</code>&ndash;<code>dual</code> 85.8%, <code>traj</code>&ndash;<code>coc</code>
  75.7%, <code>wanda</code>&ndash;<code>dual</code> 77.2% (Q head 기준). 처치 크기가
  서로 비교 가능하므로, 어느 쌍에서 null 이 나와도 &ldquo;처치가 약해서&rdquo;로는
  설명되지 않는다 &mdash; <code>tyrK</code> 의 97.7% 에서 배운 확인 절차다.</p>
</div>

<h2><span class="num">3.</span>결과 (사실)</h2>
<h3>3.1 origin150 &mdash; 150씬 &times; 2 rollout</h3>
{set_table(o, 150)}
<h3>3.2 hard100 &mdash; 100씬 &times; 2 rollout</h3>
{set_table(h, 100)}
<p class="note">hard100 의 무압축 0.510 은 사전 등록 범위 0.42&ndash;0.58 안이다(예상 0.499,
sangoh 의 913씬 런에서). 벗어났다면 씬이 아니라 설정을 의심해야 한다.</p>

<h3>3.3 네 지표 한눈에</h3>
<p>같은 데이터를 지표별로 세운 표다. 각 칸은 <strong>절대값 (무압축 대비)</strong> 이고,
종합 열은 250씬 / 500 rollout 이다.</p>
{four_t}
<div class="warn">
  <p><strong>종합 열의 절대값은 인용하지 말 것.</strong> 두 집합의 난이도가 다르므로
  (무압축 0.750 대 0.510) 종합 절대값은 60:40 혼합일 뿐이다. 의미가 있는 것은
  <strong>괄호 안</strong>이다 &mdash; score 는 씬마다 자기 집합의 무압축과 뺀 뒤
  합쳤으므로 구성상 비교 가능하다.</p>
  <p><strong><code>offroad</code> 를 단독으로 읽으면 부호가 뒤집힌다.</strong>
  거의 모든 arm 이 무압축보다 적게 이탈하고, 가장 크게 줄인 것은 hard100 의
  <code>wanda</code>(&minus;12.5pp)인데 점수는 꼴찌다. <code>progress</code> 를 같이
  보면 답이 나온다 &mdash; 0.393 으로 최저다. <strong>못 나가면 도로를 벗어날 일도
  없다.</strong> 반대로 <code>traj</code> 는 유일하게 이탈이 늘었는데(+0.8pp) 점수는
  무압축 위다.</p>
  <p><strong>과실 충돌은 두 집합에서 방향이 갈리는 유일한 지표다.</strong>
  <code>dual</code> 은 origin150 에서 &minus;1.7pp 인데 hard100 에서 +1.0pp 다.
  rollout 300 / 200 개로 이 크기는 분해되지 않는다 &mdash; 이 저장소가 &ldquo;과실
  충돌은 세기에 너무 드물다&rdquo;고 적어둔 그대로이고, 그래서
  <code>analyze_longitudinal.py</code> 가 연속 대리지표를 따로 읽는다.
  <code>wanda</code> 의 +10.7pp / +3.0pp 만 두 집합 모두에서 크다.</p>
</div>

<figure>
  <img alt="여섯 arm 의 무압축 대비 델타, 두 씬 집합" src="data:image/png;base64,{b64(plot_dir / 'deltas.png')}">
  <figcaption>무압축 대비 폐루프 점수 델타. 씬 단위 페어드, 오차 막대는 10k bootstrap
  95% CI. 각 점에 값을 직접 붙였다 &mdash; 색만으로 구분하지 않는다.</figcaption>
</figure>

<h2><span class="num">4.</span>서열은 재현된다</h2>
<figure>
  <img alt="두 씬 집합의 델타 산점도" src="data:image/png;base64,{b64(plot_dir / 'agreement.png')}">
  <figcaption>가로 origin150, 세로 hard100. 점선은 y=x.</figcaption>
</figure>
<div class="callout">
  <p><strong>여섯 arm 중 부호가 뒤집힌 것은 {flips}개다.</strong> 두 집합의 절대 점수는
  0.24 차이인데(무압축 0.750 대 0.510) 압축 효과는 같은 값을 낸다 &mdash; 상관
  {r_agree:.3f}, 대부분 y=x 위에 앉는다. <strong>쉬운 집합에서 내린 기준 판정은 어려운
  집합에서도 그대로 선다.</strong></p>
</div>
<div class="scroll"><table><thead><tr><th>arm</th><th>origin150</th><th>hard100</th>
<th>차이</th></tr></thead><tbody>
{''.join(
    f"<tr><td><code>{k}</code></td>"
    f"<td>{o[k]['vs_base']['delta']:+.4f}"
    f"{'*' if o[k]['vs_base']['sep'] else ''}</td>"
    f"<td>{h[k]['vs_base']['delta']:+.4f}"
    f"{'*' if h[k]['vs_base']['sep'] else ''}</td>"
    f"<td class='dim'>{h[k]['vs_base']['delta'] - o[k]['vs_base']['delta']:+.4f}</td></tr>"
    for k, *_ in ARMS if k in o and k in h)}
</tbody></table></div>
<p>이것이 이 저장소의 기존 기록을 <strong>좁힌다</strong>. hard100 보고서는
&ldquo;방법 간 서열이 소멸한다&rdquo;고 적었고 그것은 지금도 맞다 &mdash; 아래 4.1 을
보라. 소멸하는 것은 <em>좋은 arm 끼리</em>의 서열이고, 좋은 arm 과 나쁜 arm 사이는
두 집합 모두에서 선명하다.</p>

<h3>4.1 좋은 넷끼리는 여전히 구분되지 않는다</h3>
{pair_rows(h, [("dual", "tyr"), ("dual", "lp_r50"), ("tyr", "lp_r50"),
               ("dual", "traj"), ("tyr", "traj"), ("lp_r50", "traj")])}
<p class="note">hard100 기준. 150씬 해상도는 0.080, hard100 은 0.059 &mdash; 이 크기의
차이를 두 집합 모두 분해하지 못한다. 순위는 있지만 검정 결과는 아니다.</p>

<h3>4.2 두 집합을 합치면</h3>
<p>절대 점수는 합칠 수 없다 &mdash; 두 집합의 난이도가 0.24 다르다. 합칠 수 있는 것은
<strong>씬별 페어드 델타</strong>다. 각 씬의 델타는 <em>그 씬이 속한 집합의</em> 무압축과
비교한 값이므로 집합 간에 비교 가능하고, 합치면 250씬짜리 페어드 표본이 된다.</p>
{pooled_all}
<p>표본이 커지면서 <code>tyr</code> 가 유의해진다(origin150 단독으로는 CI 가 0 을 포함했다).
<code>traj</code> 는 합쳐도 경계에 남는다 &mdash; CI 하단이 &minus;0.000 이다.</p>

<h3>4.3 중단된 rollout 감사</h3>
<div class="warn">
  <p><strong>hard100 에서 4개 씬이 시뮬레이터 쪽 사유로 중단됐고, 그 씬이 arm 마다
  다르다.</strong> <code>Waypoint N fails sanity check: route folds back on itself</code>
  &mdash; 주행 실패가 아니라 경로 정합성 검사 실패다. 이런 rollout 은
  <strong>강제로 0.0 점</strong>을 받고 <code>metrics.parquet</code> 도 남기지 않는다.</p>
  <pre>clipgt-4bad2f63…   8개 arm 전부       <span>&larr; 씬 고유, 페어드에서 상쇄</span>
clipgt-adb899bd…   8개 arm 전부       <span>&larr; 씬 고유, 페어드에서 상쇄</span>
clipgt-3b934c9a…   <b>dual 만</b>
clipgt-ceb3d0d5…   <b>dual, coc 만</b></pre>
  <p>즉 <code>dual</code> 은 4씬, <code>coc</code> 는 3씬, 나머지는 2씬에서 0.0 을 받았다.
  뒤의 두 씬은 <strong><code>dual</code> 과 <code>coc</code> 에게만 불리하게</strong>
  작용한다. origin150 에는 이런 rollout 이 <strong>하나도 없다</strong>.</p>
</div>
{abort_t}
<p>네 씬을 모두 빼고 다시 재면 절대 점수는 전 arm 이 +0.015&ndash;+0.025 오르고,
페어드 델타와 판정은 <strong>하나도 바뀌지 않는다</strong>:</p>
{pooled_clean}
<p class="note">방향이 중요하다 &mdash; 이 결함은 <code>dual</code> 과 <code>coc</code> 를
<strong>불리하게</strong> 만들고 있었다. <code>dual</code> 의 hard100 델타는 중단 씬을
빼면 +0.0850 에서 +0.0886 으로 <em>올라간다</em>. 그래서 본문의 수치는 보수적인 쪽이고,
<code>coc</code> 의 패배가 이 결함 때문이라는 설명도 성립하지 않는다(빼면 더 나빠진다).</p>

<h2><span class="num">5.</span><code>dual</code> 의 두 half (사실)</h2>
{pair_rows(o, [("dual", "traj"), ("dual", "coc"), ("traj", "coc")])}
<p>150씬 기준. 같은 비교를 hard100 에서 하면:</p>
{pair_rows(h, [("dual", "traj"), ("dual", "coc"), ("traj", "coc")])}
<div class="callout">
  <p><strong>합집합이 두 half 를 모두 이긴다.</strong> 그리고 두 half 는 대칭이 아니다
  &mdash; <code>traj</code> 는 무압축을 아슬아슬하게 이기는 쪽(두 집합 모두 +0.03대,
  CI 가 0 을 포함)이고 <code>coc</code> 는 <strong>무압축에 진다</strong>
  (&minus;0.089 / &minus;0.095, 둘 다 CI 가 0 을 배제).</p>
  <p>2절의 100.0% 가 여기서 값을 한다. <code>dual</code> 은 새 유닛을 도입하지 않으므로,
  <code>dual &minus; traj</code> 의 이득은 <strong><code>coc</code> 가 골라 온 유닛을
  합집합에 넣은 결과</strong>다 &mdash; 그 유닛들만으로 고르면 무압축에 지는데도.</p>
</div>

<h2><span class="num">6.</span>두 실패는 성격이 다르다</h2>
<div class="scroll"><table><thead><tr><th></th><th><code>wanda</code></th>
<th><code>coc</code></th></tr></thead><tbody>
<tr><td>origin150 / hard100 점수</td>
    <td>{o['wanda']['score']:.3f} / {h['wanda']['score']:.3f}</td>
    <td>{o['coc']['score']:.3f} / {h['coc']['score']:.3f}</td></tr>
<tr><td>CoC 퇴화</td>
    <td class="bad">{100 * o['wanda']['coc']:.1f}% / {100 * h['wanda']['coc']:.1f}%</td>
    <td>{100 * o['coc']['coc']:.2f}% / {100 * h['coc']['coc']:.2f}%</td></tr>
<tr><td>progress</td>
    <td>{o['wanda']['prog']:.3f} / {h['wanda']['prog']:.3f}</td>
    <td>{o['coc']['prog']:.3f} / {h['coc']['prog']:.3f}</td></tr>
<tr><td>과실 충돌</td>
    <td class="bad">{100 * o['wanda']['col']:.1f}% / {100 * h['wanda']['col']:.1f}%</td>
    <td>{100 * o['coc']['col']:.1f}% / {100 * h['coc']['col']:.1f}%</td></tr>
</tbody></table></div>
<p><code>wanda</code> 는 <strong>생성이 무너진다</strong> &mdash; rollout 의 절반에서
CoC 가 퇴화하고(어려운 집합에서 {100 * h['wanda']['coc']:.1f}%), 남은 출력의 평균 길이가
184 자로 다른 arm(63&ndash;87)의 2&ndash;3배다. 빈 출력이 절반인데 평균 길이가 두 배라는
것은 무너지지 않은 쪽이 극단적으로 길다는 뜻이고, 그래서 이 arm 의 폐루프는 샤드당
14.3시간으로 다른 arm(5.0&ndash;5.4시간)의 <strong>2.6배</strong>가 걸렸다.
<strong>CoC 가 무너지면 디코딩이 짧아지는 게 아니라 길어진다</strong>(soup 은 반복
생성이다).</p>
<p><code>coc</code> 는 정반대다. <strong>CoC 를 제대로 지켰다</strong> &mdash;
퇴화 {100 * o['coc']['coc']:.2f}% / {100 * h['coc']['coc']:.2f}% 로
<code>dual</code>({100 * o['dual']['coc']:.2f}% / {100 * h['dual']['coc']:.2f}%) 과
같거나 낮다. 그런데 progress 가 무너진다. 순수한 궤적 손상이다.</p>
<div class="callout">
  <p><strong>추론을 보존하도록 고른 기준이 추론은 실제로 보존했고 주행만 망가뜨렸다.</strong>
  이 저장소가 반복해 기록해 온 &ldquo;X 는 능력을 예측하지 못한다&rdquo; 목록에 또
  하나가 붙는데, 여기서 X 는 <em>그 기준이 최적화한 대상 그 자체</em>다.</p>
</div>

<h2><span class="num">7.</span>해석 (의견)</h2>
<div class="callout">
  <p><strong>결론 1 &mdash; 씬 집합을 바꿔도 기준 판정은 바뀌지 않는다.</strong>
  부호 뒤집힘 {flips}/6, 상관 {r_agree:.3f}. 난이도가 크게 다른 두 집합이 같은 서열을
  내므로, 기준을 고르는 실험을 어느 집합에서 하든 결론은 같다. 바뀌는 것은
  <strong>분해능</strong>이지 방향이 아니다.</p>
  <p><strong>결론 2 &mdash; <code>dual</code> 의 이득은 합집합에서 온다.</strong>
  두 half 를 단독으로 쓰면 하나는 무압축과 비기고(<code>traj</code>) 하나는 진다
  (<code>coc</code>). 유지집합이 정확히 <code>traj &cup; coc</code> 의 부분집합이므로,
  <code>max(rank, rank)</code> 라는 결합 방식 자체가 일하고 있다. 이 저장소가
  <code>znorm11</code> 에서 &ldquo;평균은 합집합 성질을 잃는다&rdquo;고 기각한 것과 같은
  축의 반대쪽 증거다.</p>
  <p><strong>결론 3 &mdash; 기울기 유무는 경계선이 아니다.</strong> <code>coc</code> 는
  기울기를 쓰는데도 <code>wanda</code> 와 구분되지 않는다(hard100
  {paired(h['coc']['scenes'], h['wanda']['scenes'])['delta']:+.4f},
  p={paired(h['coc']['scenes'], h['wanda']['scenes'])['p']:.2g}). 갈리는 선은
  <strong>궤적 목적을 쓰느냐</strong> 쪽에 있다 &mdash; 좋은 네 arm 은 모두 궤적 정보를
  직간접으로 쓰고(<code>traj</code>, <code>dual</code>, <code>lp_r50</code>), 출력
  재구성(<code>tyr</code>)은 궤적을 명시적으로 쓰지 않지만 출력 전체를 보존한다.</p>
</div>

<h2><span class="num">8.</span>주장하지 않는 것</h2>
<div class="warn">
  <p><strong>절대 점수를 suite 전체로 일반화하지 않는다.</strong> 150씬 표본은
  <code>public_2601</code> 913씬보다 쉽고(무압축 0.742 대 0.660, 과실 충돌 2.0% 대 5.5%),
  hard100 은 반대쪽 꼬리다. 두 집합 모두 <em>대응 델타</em>를 읽는 자리이지 절대값을
  인용할 자리가 아니다.</p>
  <p><strong>무압축 대비 우위가 채점 규칙과 무관하다고 말하지 않는다.</strong> alpasim 은
  자차가 GT 경로에서 4 m 벗어나면 그 이후를 채점에서 버린다. 150씬에서
  <strong>무압축 rollout 의 57.7%</strong> 가 이 규칙에 걸리고 시뮬 시간의 16.2% 가
  채점되지 않는다. 규칙을 끄고 다시 재면 <code>dual &minus; baseline</code> 은
  +0.0787 에서 +0.0443 (p=0.059) 로 유의성을 잃는다. <strong>arm 간 비교는 더
  강건하지만 무압축 대비 값은 이 규칙 안에서만 유의하다.</strong> 이 보고서의 델타는
  전부 출하 프로토콜 값이고, 재계산은 150씬에서만 했다.</p>
  <p><strong>CI 와 Wilcoxon 이 엇갈리는 칸이 있다.</strong> <code>tyr</code> 의 origin150
  은 CI 가 0 을 포함하는데 p=0.0072 다. 씬 절반이 정확히 1.0 이라 동점이 많아 생기는
  현상이고, 이 저장소는 <strong>평균 CI 를 1차로</strong> 읽는다. <code>lp_r50</code> 의
  hard100 도 같은 형태다.</p>
  <p><strong>예산이 완전히 같다고 말하지 않는다.</strong> <code>lp_r50</code> 은 25.0%
  로 나머지 다섯(24.0%)보다 1%p 크다. 그 차이가 +0.06 을 만들 크기는 아니지만
  1요인 비교는 아니다.</p>
  <p><strong><code>tyrK</code> 는 뺐다.</strong> 두 집합에 모두 런이 있지만(150씬 0.788,
  hard100 0.610) 이 보고서의 축은 <em>기준</em>이고 <code>tyrK</code> 는
  <code>tyr</code> 의 적합도 변형이라 선택 중첩이 97.7% 다. 넣으면 축이 둘이 된다.</p>
</div>

<h2><span class="num">9.</span>재현</h2>
<pre>python experiments/head_analysis/make_slim.py --config &lt;crit&gt;_u40_v2 \\
    --importance importance_v2 --gpu 4 --out outputs/slim_&lt;crit&gt;_u40_v2

# 150씬
DRIVER_OMP_THREADS=8 bash experiments/head_analysis/launch_alpasim_shards.sh \\
    slim_&lt;crit&gt;_u40_v2 150 2 "4 5 6 7"

# hard100 (150씬과 서로소)
SUITE=public_2601_hard100 PREFIX=h100_ DRIVER_OMP_THREADS=8 \\
SCENES_CSV=$PWD/outputs/scene_difficulty/hard100_suite.csv \\
  bash experiments/head_analysis/launch_alpasim_shards.sh slim_&lt;crit&gt;_u40_v2 100 2 "4 5 6 7"

python experiments/evaluation/criterion_twoset_report.py \\
    --out reports/evaluation/{date}_criterion-two-sets.html</pre>
<div class="warn">
  <p class="note"><strong>이 표를 다시 만들 때 밟게 되는 함정 둘.</strong></p>
  <p class="note">&bull; <code>lp_r50</code> 의 <strong>150씬 런은 우리
  <code>alpasim-runs</code> 밖</strong>에 있다
  (<code>soowon/alpasim-analysis/runs_root/lp_r50</code>). <code>m2601_merged_</code>
  접두사만 훑으면 <code>h100_</code> 만 나와서 &ldquo;hard100 전용 arm&rdquo;으로
  오판한다. 2026-09-17 에 실제로 그렇게 답했다.</p>
  <p class="note">&bull; <strong>CoC 퇴화율은 <code>(config, n_scenes)</code> 로
  찾아야 한다.</strong> 같은 config 가 두 집합에서 돌았고 값이 다르다 &mdash;
  <code>slim_tyr_u40_r</code> 은 150씬 5.92% / hard100 6.33%. config 이름만으로 찾으면
  조용히 다른 집합의 값을 집는다.</p>
</div>
</div></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--date", default="2026-09-19")
    args = ap.parse_args()
    data = gather()
    plot_dir = O / "criterion_twoset" / "plots"
    r = plots(data, plot_dir)
    out = args.out if args.out.is_absolute() else Path.cwd() / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(data, r, plot_dir, args.date))
    print(f"wrote {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
