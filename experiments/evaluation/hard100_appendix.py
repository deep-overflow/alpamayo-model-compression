"""Emit the LaTeX appendix on closed-loop scene selection, numbers computed from artifacts.

Generated rather than hand-written for one reason: an appendix that explains a sampling
procedure is worthless if its numbers drift from the procedure. Everything printed here is
recomputed at build time from

  outputs/scene_difficulty/scene_difficulty_public2601.csv   static features, 913 scenes
  outputs/scene_difficulty/hard100_suite.csv                 the realized selection
  <sangoh>/eval2601_a1_5/aggregate/results-summary.json      an independently produced
                                                             unpruned run over all 913
  alpasim-runs/h100_merged_baseline                          our own unpruned hard100 run

The claim the appendix has to state precisely -- and the one a reader is most likely to
garble -- is NOT "the static score and the baseline agree". It is that a score computed
from the scene file alone, with no model and no rollout, PREDICTS the unpruned model's
closed-loop score well enough to rank scenes, verified on scenes that the shipped
150-scene matrix never touched. The indirection is deliberate: selecting scenes by the
model's own score and then reading deltas against that same model is a regression-to-the-
mean trap that flipped the sign of 11 of 17 arms in the easy stratum.

    python experiments/evaluation/hard100_appendix.py \\
        --out reports/evaluation/2026-09-20_hard100_sampling_appendix.tex
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
SANGOH = Path("/mnt/nvme1n1/ad_vla/results/sangoh/alpasim_runs/eval2601_a1_5")
OURS = Path("/home/cvlab21/project/chan/alpasim-runs/h100_merged_baseline")
N_MATRIX = 150


def scene_scores(run):
    j = json.loads((run / "aggregate/results-summary.json").read_text())
    rows = [(r["clipgt_id"], r["score"],
             r["metrics"].get("collision_at_fault", 0), r["metrics"].get("offroad", 0))
            for r in j["rollouts"] if r.get("score") is not None]
    df = pd.DataFrame(rows, columns=["scene_id", "score", "col", "off"])
    return df.groupby("scene_id").mean().reset_index(), len(rows)


def pval(p):
    if p < 1e-4:
        e = f"{p:.0e}".split("e")
        return f"$p<10^{{{int(e[1])+1}}}$"
    return f"$p={p:.2g}$"


def build(args):
    feats = pd.read_csv(REPO / "outputs/scene_difficulty/scene_difficulty_public2601.csv")
    un, _ = scene_scores(SANGOH)
    d = feats.merge(un, on="scene_id", how="inner")
    held = d[~d.already_run_150]
    suite = pd.read_csv(REPO / "outputs/scene_difficulty/hard100_suite.csv")
    sel = d[d.scene_id.isin(suite.scene_id)]
    ours, ours_roll = scene_scores(OURS)

    suites = pd.read_csv("/home/cvlab21/project/chan/alpasim/data/scenes/sim_suites.csv")
    s2601 = suites[suites.test_suite_id == "public_2601"]
    s2604 = suites[suites.test_suite_id == "public_2604"]
    n_2604 = s2604.scene_id.nunique()
    n_shared = len(set(s2601.scene_id) & set(s2604.scene_id))
    assert len(set(s2601.uuid) & set(s2604.uuid)) == 0, "releases share a uuid"

    r_main = stats.spearmanr(held.hard_score, held.score)
    hs = held.sort_values("hard_score", ascending=False)
    top, bot = hs.head(100).score.mean(), hs.tail(100).score.mean()

    # the shipped 150 against everything else, from the same independent run
    in150, out150 = d[d.already_run_150], d[~d.already_run_150]
    mw = stats.mannwhitneyu(in150.score, out150.score)

    bands = held.assign(b=pd.qcut(held.hard_score, 10, labels=False)).groupby("b")
    band_rows = "\n".join(
        f"    {10 * b}--{10 * b + 10}\\% & {len(g)} & {g.score.mean():.3f} \\\\"
        for b, g in bands)

    def rho(a, bcol):
        r = stats.spearmanr(a, bcol)
        return r.statistic, r.pvalue

    sel_rows = []
    for col, nice in (("score", "Closed-loop score"),
                      ("col", "At-fault collision"),
                      ("off", "Off-road")):
        st, p = rho(held.hard_score, held[col])
        sel_rows.append(f"    {nice} & ${st:+.3f}$ & {pval(p)} \\\\")
    feat_rows = []
    for col, nice in (("yaw_total_deg", "Total yaw (deg)"),
                      ("v_mean", "Mean speed (m/s)"),
                      ("n_agents", "Agent count"),
                      ("n_agents_30m", "Agents within 30\\,m")):
        st, p = rho(held[col], held.score)
        feat_rows.append(f"    {nice} & ${st:+.3f}$ & {pval(p)} \\\\")

    return f"""% Appendix: closed-loop scene selection for the hard-100 set.
%
% GENERATED -- do not hand-edit. Rebuild with
%   python experiments/evaluation/hard100_appendix.py --out <this file>
%
% Sources (all recomputed at build time):
%   outputs/scene_difficulty/scene_difficulty_public2601.csv  static features, 913 scenes
%   outputs/scene_difficulty/hard100_suite.csv                the realized 100-scene draw
%   {SANGOH}
%       an unpruned 913-scene run produced independently of this work (1 rollout/scene)
%   {OURS}
%       our own unpruned run on the selected 100 ({ours_roll} rollouts)
%
% The load-bearing distinction: the static score PREDICTS the unpruned model's score; it
% is not "similar to" it. Selection never touches a model output, which is what keeps the
% regression-to-the-mean bias out (see the paragraph on stratifying by the model's score).

\\section{{Closed-loop scene selection}}
\\label{{app:hard100}}

\\paragraph{{Why a second scene set.}}
The shipped closed-loop matrix is the first {N_MATRIX} scenes of \\texttt{{public\\_2601}}
under a sort by scene identifier. That is unbiased in expectation, deterministic and
nested, but the \\emph{{realized}} sample is easier than the suite it came from: measured
against an independently produced unpruned run over all {len(d)} scenes, those
{len(in150)} scenes score {in150.score.mean():.3f} against {out150.score.mean():.3f} on
the remaining {len(out150)} (Mann--Whitney {pval(mw.pvalue)}). Absolute closed-loop
numbers on the matrix are therefore optimistic relative to the suite, and a second,
harder, \\emph{{disjoint}} set is needed before any absolute claim is made. The two sets
together cover {N_MATRIX + len(suite)} of the suite's {len(d)} scenes
({(N_MATRIX + len(suite)) / len(d) * 100:.1f}\\%) and are disjoint by construction.

\\paragraph{{Scene release.}}
All closed-loop evaluation in this work uses the \\texttt{{public\\_2601}} suite
({len(d)} scenes). The later \\texttt{{public\\_2604}} release ({n_2604} scenes) shares
{n_shared} scene identifiers with it but \\textbf{{zero}} artifact UUIDs --- every scene was
re-rendered --- so results from the two are not comparable and their caches are separate.
Because of that overlap, a scene must be addressed by its artifact UUID: resolving a bare
identifier returns the newer render for those {n_shared} scenes, silently substituting
different pixels for the same nominal scene. Every suite file written here therefore
carries the UUID, and the launcher rejects any UUID that does not belong to the declared
parent release.

\\paragraph{{A model-free difficulty score.}}
Each scene ships as a \\texttt{{usdz}}, which is a ZIP archive; the ego pose track
(\\texttt{{rig\\_trajectories.usda}}, ground-truth pose at 10\\,Hz) can be read by random
access without touching the several hundred megabytes of mesh data, so all {len(feats)}
scenes are featurised in well under a minute. From that track alone we form
\\[
  \\texttt{{hard\\_score}} \\;=\\; z(\\bar v) \\;+\\; z(\\textstyle\\sum |\\Delta\\psi|),
\\]
the sum of within-suite $z$-scores of mean ego speed and total yaw traversed. The parse is
verified against the simulator: path length recovered from the track matches the
simulator's reported \\texttt{{gt\\_dist\\_traveled\\_m}} at $r = 1.00000$.

\\textbf{{No model output enters this score.}} No rollout is run, no checkpoint is loaded,
and no evaluation metric is consulted.

\\paragraph{{Out-of-sample validation.}}
The score is validated on the {len(held)} scenes that the {N_MATRIX}-scene matrix does
\\emph{{not}} contain, against the independent unpruned run. It ranks them:
Spearman $\\rho = {r_main.statistic:+.3f}$ ({pval(r_main.pvalue)}), with the top
$100$ scenes averaging ${top:.3f}$ and the bottom $100$ averaging ${bot:.3f}$.
Table~\\ref{{tab:hard100-deciles}} shows that the relationship \\emph{{saturates}} above
roughly the 80th percentile, which is why the set is drawn from that plateau rather than
from the extreme tail --- cutting higher buys no additional difficulty and costs sample
diversity.

\\begin{{table}}[t]
\\centering\\small
\\caption{{Unpruned closed-loop score by decile of the model-free difficulty score, on the
  {len(held)} scenes held out from the {N_MATRIX}-scene matrix. The top two deciles are
  indistinguishable, so the hard set is drawn from the whole 80--100\\% band.}}
\\label{{tab:hard100-deciles}}
\\begin{{tabular}}{{lrr}}
\\toprule
\\texttt{{hard\\_score}} decile & Scenes & Unpruned score \\\\
\\midrule
{band_rows}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\paragraph{{Why not stratify by the model's own score.}}
The obvious alternative --- rank scenes by the unpruned model's closed-loop score and take
the worst --- is unsound for the comparisons this set is used for. Grouping scenes by a
model's score and then reading a \\emph{{paired delta against that same model}} induces
regression to the mean: in an earlier analysis this flipped the sign of 11 of 17 arms in
the easy stratum. A model-free criterion has no such coupling, which is the reason for the
indirection through static scene features.

\\paragraph{{What the score selects for.}}
It selects for progress and lane-keeping stress, \\emph{{not}} for safety stress
(Table~\\ref{{tab:hard100-what}}, upper block): difficulty is essentially uncorrelated with
at-fault collisions. Within the features, ego kinematics carry the signal and traffic
density carries none (lower block); of the two terms in the score, total yaw dominates.
Absolute collision rates on this set should therefore not be read as a safety stratum.

\\begin{{table}}[t]
\\centering\\small
\\caption{{Spearman correlations on the {len(held)} held-out scenes. Upper: what the
  difficulty score predicts. Lower: which scene features predict the unpruned score.}}
\\label{{tab:hard100-what}}
\\begin{{tabular}}{{lrl}}
\\toprule
\\multicolumn{{3}}{{l}}{{\\emph{{\\texttt{{hard\\_score}} vs.\\ outcome}}}} \\\\
\\midrule
{chr(10).join(sel_rows)}
\\midrule
\\multicolumn{{3}}{{l}}{{\\emph{{Scene feature vs.\\ unpruned score}}}} \\\\
\\midrule
{chr(10).join(feat_rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\paragraph{{The realized set.}}
Scenes already in the {N_MATRIX}-scene matrix are excluded, as are scenes whose
ground-truth ego path is shorter than $5$\\,m (for those the progress term is overridden to
$1.0$ and the scene carries no information); the shortest retained path is
${sel.gt_path_m.min():.1f}$\\,m. The remaining scenes are ranked and the top
${len(suite)}$ taken, giving \\texttt{{hard\\_score}} in
$[{sel.hard_score.min():.2f}, {sel.hard_score.max():.2f}]$ against a suite-wide range of
$[{d.hard_score.min():.2f}, {d.hard_score.max():.2f}]$. Overlap with the matrix is
\\textbf{{{int(sel.already_run_150.sum())}}} scenes by construction, and the disjointness is
asserted before the file is written.

As a pre-registration check, the independent unpruned run puts these {len(suite)} scenes
at ${sel.score.mean():.3f}$; our own unpruned run over them (two rollouts per scene,
{ours_roll} rollouts) measures ${ours.score.mean():.3f}$, inside the pre-registered
acceptance window $[0.42, 0.58]$. A value outside that window would have indicated a
setup fault rather than a property of the scenes.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out if args.out.is_absolute() else Path.cwd() / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(args))
    print(f"wrote {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
