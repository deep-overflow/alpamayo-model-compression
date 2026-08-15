"""Plots for the LingoQA reasoning-probe report.

Usage: python experiments/lingoqa/plot_lingo.py --exp-id lingo_report
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))

BG = "#FAF9F5"
INK = "#29261B"
MUTED = "#6B6555"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})

ARMS = [("lingo_judge_baseline", "baseline", C1),
        ("lingo_judge_slim_dual_u40_v2", "dual_u40_v2", C2),
        ("lingo_judge_slim_jtraj_u40_v2", "j_traj_u40_v2", C3),
        ("lingo_judge_blind", "blind floor", MUTED)]

FIXED = "What is the current action and its justification?"
CATS = ["고정(action+justif)", "존재/예-아니오", "행동/판단", "기타 서술", "개수 세기", "색상"]
# plots must stay ASCII: no Korean glyphs in any matplotlib font on this host,
# and installing one would mean touching the host environment
EN = {"고정(action+justif)": "action+justif\n(fixed Q)", "존재/예-아니오": "existence\n/ yes-no",
      "행동/판단": "action\n/ judgement", "기타 서술": "other\ndescriptive",
      "개수 세기": "counting", "색상": "colour"}


def cat(q):
    ql = q.lower()
    if q.startswith(FIXED):
        return "고정(action+justif)"
    if re.search(r"how many|number of", ql):
        return "개수 세기"
    if re.search(r"what colou?r", ql):
        return "색상"
    if re.search(r"should you|do you need|next|plan|safe|decision", ql):
        return "행동/판단"
    if re.search(r"^(are|is|do|does|can|did) ", ql) or "any" in ql:
        return "존재/예-아니오"
    return "기타 서술"


def load(exp_id):
    """Per-question verdicts. VQA runs keep generations in rows.json and the judge's
    verdicts in scored.json; the probe's rows.json already carries `correct`."""
    d = REPO / "outputs" / exp_id
    scored = d / "scored.json"
    rows = json.loads((scored if scored.exists() else d / "rows.json").read_text())
    return {r["question_id"] + "|" + r["segment_id"]: r for r in rows}


def cluster_boot(delta_by_seg, n_boot=10000, seed=0):
    segs = list(delta_by_seg)
    per = [np.asarray(delta_by_seg[s], float) for s in segs]
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(segs), size=(n_boot, len(segs)))
    boots = np.array([np.concatenate([per[i] for i in row]).mean() for row in idx])
    return np.concatenate(per).mean(), boots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="lingo_report")
    args = ap.parse_args()
    out = REPO / "outputs" / args.exp_id / "plots"
    out.mkdir(parents=True, exist_ok=True)

    runs = {name: load(eid) for eid, name, _ in ARMS}
    keys = sorted(runs["baseline"])

    # --- 1. overall accuracy with the blind floor marked -------------------------
    fig, ax = plt.subplots(figsize=(7, 3.6))
    names = [n for _, n, _ in ARMS]
    accs = [100 * np.mean([runs[n][k]["correct"] for k in keys]) for n in names]
    cols = [c for _, _, c in ARMS]
    bars = ax.bar(names, accs, color=cols, width=0.6)
    floor = accs[-1]
    ax.axhline(floor, color=MUTED, ls="--", lw=1)
    ax.text(3.42, floor + 1.2, "blind floor", color=MUTED, fontsize=8.5, ha="right")
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 1.0, f"{a:.1f}%", ha="center", fontsize=9.5)
    ax.set_ylabel("Lingo-Judge accuracy (%)")
    ax.set_ylim(0, 75)
    ax.set_title("CoC-as-evidence accuracy (500 questions / 100 segments)")
    fig.tight_layout()
    fig.savefig(out / "accuracy.png", dpi=160)
    plt.close(fig)

    # --- 2. paired delta vs baseline, cluster-bootstrap CI -----------------------
    fig, ax = plt.subplots(figsize=(7, 3.0))
    labels, obs, los, his, cs = [], [], [], [], []
    for name, col in [("dual_u40_v2", C2), ("j_traj_u40_v2", C3), ("blind floor", MUTED)]:
        by_seg = defaultdict(list)
        for k in keys:
            by_seg[runs[name][k]["segment_id"]].append(
                float(runs[name][k]["correct"]) - float(runs["baseline"][k]["correct"]))
        o, boots = cluster_boot(by_seg)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        labels.append(name); obs.append(100 * o); los.append(100 * lo); his.append(100 * hi)
        cs.append(col)
    y = np.arange(len(labels))
    ax.axvline(0, color=MUTED, lw=1)
    for i in range(len(labels)):
        ax.plot([los[i], his[i]], [y[i], y[i]], color=cs[i], lw=2.5, solid_capstyle="round")
        ax.plot(obs[i], y[i], "o", color=cs[i], ms=7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("accuracy difference vs baseline (pp)")
    ax.set_title("Paired delta, 95% cluster-bootstrap CI (segments resampled)")
    fig.tight_layout()
    fig.savefig(out / "delta_ci.png", dpi=160)
    plt.close(fig)

    # --- 3. per-question-type accuracy ------------------------------------------
    n_by = Counter(cat(runs["baseline"][k]["question"]) for k in keys)
    order = [c for c in CATS if n_by[c]]
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    w = 0.2
    x = np.arange(len(order))
    for j, (name, col) in enumerate([("baseline", C1), ("dual_u40_v2", C2),
                                     ("j_traj_u40_v2", C3), ("blind floor", MUTED)]):
        vals = []
        for c in order:
            sel = [k for k in keys if cat(runs["baseline"][k]["question"]) == c]
            vals.append(100 * np.mean([runs[name][k]["correct"] for k in sel]))
        ax.bar(x + (j - 1.5) * w, vals, width=w, color=col, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{EN[c]}\n(n={n_by[c]})" for c in order], fontsize=8)
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="upper right")
    ax.set_title("Accuracy by question type")
    fig.tight_layout()
    fig.savefig(out / "by_type.png", dpi=160)
    plt.close(fig)

    # --- 4. j_traj - dual, per question type ------------------------------------
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    diffs = []
    for c in order:
        sel = [k for k in keys if cat(runs["baseline"][k]["question"]) == c]
        d = np.mean([runs["j_traj_u40_v2"][k]["correct"] for k in sel]) - \
            np.mean([runs["dual_u40_v2"][k]["correct"] for k in sel])
        diffs.append(100 * d)
    cols = [C2 if d >= 0 else C3 for d in diffs]
    ax.bar(range(len(order)), diffs, color=cols, width=0.6)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{EN[c]}\n(n={n_by[c]})" for c in order], fontsize=8)
    ax.set_ylabel("j_traj − dual (pp)")
    ax.set_title("Criterion swap by question type: own-action narration vs scene perception")
    fig.tight_layout()
    fig.savefig(out / "criterion_by_type.png", dpi=160)
    plt.close(fig)

    print(f"4 plots -> {out}")
    vqa_plots(out)
    arm_plots(out)




# --- VQA track ------------------------------------------------------------------
VQA = [("lingo_vqa_baseline", "baseline", C1),
       ("lingo_vqa_slim_dual_u40_v2", "dual_u40_v2", C2),
       ("lingo_vqa_slim_jtraj_u40_v2", "j_traj_u40_v2", C3)]
VQA_C = [("lingo_vqa_baseline_concise", "baseline", C1),
         ("lingo_vqa_slim_dual_u40_v2_concise", "dual_u40_v2", C2),
         ("lingo_vqa_slim_jtraj_u40_v2_concise", "j_traj_u40_v2", C3)]
# Table 5 of the paper. Frame count is a disclosed per-row property there.
PAPER = [("Human (5f)", 96.6), ("Human (1f)", 81.8),
         ("LingoQA FT (5f)", 60.8), ("GPT-4V 0-shot (5f)", 59.6),
         ("LLaVA 0-shot (1f)", 49.4), ("FUYU 0-shot (1f)", 45.4)]


def vqa_plots(out):
    runs = {n: load(e) for e, n, _ in VQA}
    runs_c = {n: load(e) for e, n, _ in VQA_C}
    keys = sorted(runs["baseline"])

    # 5. our zero-shot numbers placed against the paper's Table 5
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    ours = [(n, 100 * np.mean([runs[n][k]["correct"] for k in keys])) for _, n, _ in VQA]
    labels = [p[0] for p in PAPER] + [f"Alpamayo 1.5\n{n} (4f)" for n, _ in ours]
    vals = [p[1] for p in PAPER] + [v for _, v in ours]
    cols = [MUTED] * len(PAPER) + [c for _, _, c in VQA]
    order = np.argsort(vals)[::-1]
    ax.bar(range(len(vals)), [vals[i] for i in order],
           color=[cols[i] for i in order], width=0.66)
    for r, i in enumerate(order):
        ax.text(r, vals[i] + 1.2, f"{vals[i]:.1f}", ha="center", fontsize=8.5)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels([labels[i] for i in order], fontsize=7.6, rotation=30, ha="right")
    ax.set_ylabel("Lingo-Judge accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Zero-shot LingoQA: ours (colour) vs paper Table 5 (grey)")
    fig.tight_layout()
    fig.savefig(out / "vqa_vs_paper.png", dpi=160)
    plt.close(fig)

    # 6. accuracy under both response-style conditions
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(3)
    a_u = [100 * np.mean([runs[n][k]["correct"] for k in keys]) for _, n, _ in VQA]
    a_c = [100 * np.mean([runs_c[n][k]["correct"] for k in keys]) for _, n, _ in VQA_C]
    ax.bar(x - 0.19, a_u, width=0.36, color=C1, label="unprompted")
    ax.bar(x + 0.19, a_c, width=0.36, color=C4, label="concise")
    for i in range(3):
        ax.text(x[i] - 0.19, a_u[i] + 0.7, f"{a_u[i]:.1f}", ha="center", fontsize=8.5)
        ax.text(x[i] + 0.19, a_c[i] + 0.7, f"{a_c[i]:.1f}", ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n, _ in VQA], fontsize=9)
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(55, 80)
    ax.legend(frameon=False, fontsize=8.5, ncol=2)
    ax.set_title("Response-style sensitivity: the arm ordering is not robust")
    fig.tight_layout()
    fig.savefig(out / "vqa_style.png", dpi=160)
    plt.close(fig)

    # 7. answer length vs the human reference, with truncation marked
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    for j, (eid, name, col) in enumerate(VQA):
        g = json.loads((REPO / "outputs" / eid / "rows.json").read_text())
        w = [len(r["answer"].split()) for r in g]
        tr = 100 * sum(r["truncated"] for r in g) / len(g)
        ax.hist(w, bins=np.arange(0, 130, 5), histtype="step", lw=2, color=col,
                label=f"{name}  (trunc {tr:.1f}%)")
    ax.axvline(9, color=INK, ls="--", lw=1.2)
    ax.text(10.5, ax.get_ylim()[1] * 0.9, "human GT median = 9", fontsize=8.5, color=INK)
    ax.set_xlabel("answer length (words)")
    ax.set_ylabel("questions")
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title("Answer length: all arms far above the human reference")
    fig.tight_layout()
    fig.savefig(out / "vqa_length.png", dpi=160)
    plt.close(fig)
    print("3 VQA plots ->", out)

# --- single-criterion arms + relation to the existing metrics ---------------------
# All six share the 24.0% budget (8,421,074,162 params). nll_gtcoc / minADE are the
# proper eval numbers (1,533 OOD clips, 500 in-dist), not the one-clip smoke values in
# each checkpoint's summary.txt.
ALL_ARMS = [("baseline", 73.2, None, None), ("dual_u40_v2", 68.8, 3.0375, 0.7766),
            ("j_traj_u40_v2", 67.8, 3.0075, 0.8148), ("traj_u40_v2", 37.0, 3.1483, 0.8411),
            ("j_u40_v2", 32.2, 2.9861, 1.7827), ("coc_u40_v2", 30.2, 3.1967, 1.4442)]


def arm_plots(out):
    # 8. every arm at matched budget
    fig, ax = plt.subplots(figsize=(7.8, 3.6))
    names = [a[0] for a in ALL_ARMS]
    vals = [a[1] for a in ALL_ARMS]
    cols = [C1, C2, C3, C4, MUTED, "#8e5ea8"]
    ax.bar(names, vals, color=cols, width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.1, f"{v:.1f}", ha="center", fontsize=9.5)
    ax.axhline(vals[0], color=MUTED, ls="--", lw=1)
    ax.set_ylabel("Lingo-Judge accuracy (%)")
    ax.set_ylim(0, 82)
    ax.set_xticklabels(names, fontsize=8.5, rotation=18, ha="right")
    ax.set_title("Matched budget (-24.0%): combined criteria hold, single criteria collapse")
    fig.tight_layout()
    fig.savefig(out / "arms_all.png", dpi=160)
    plt.close(fig)

    # 9. what the existing metrics see of that 38.6pp spread
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5))
    pts = [a for a in ALL_ARMS if a[2] is not None]
    for ax, idx, lab, note in [(axes[0], 2, "nll_gtcoc (OOD, 1533 clips)", "Spearman -0.30 (p=0.62)"),
                               (axes[1], 3, "in-dist minADE (500 clips)", "Spearman -0.90 (p=0.04)")]:
        for (n, acc, nll, ade), c in zip(pts, [C2, C3, C4, MUTED, "#8e5ea8"]):
            v = nll if idx == 2 else ade
            ax.scatter(v, acc, s=70, color=c, zorder=3)
            ax.annotate(n, (v, acc), fontsize=7.5, xytext=(4, 4),
                        textcoords="offset points", color=MUTED)
        ax.set_xlabel(lab)
        ax.set_ylabel("Lingo-Judge accuracy (%)")
        ax.set_title(note, fontsize=9.5)
        ax.set_ylim(20, 80)
    fig.suptitle("LingoQA vs the metrics already in use", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "metric_relation.png", dpi=160)
    plt.close(fig)
    print("2 arm plots ->", out)


if __name__ == "__main__":
    main()
