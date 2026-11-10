"""Closed-loop gap decomposition (experiments/evaluation/analyze_closedloop_gap.py, 2026-10-07) on all 100 Hard100 scenes.

The 10-07 run dropped every Hard100 scene with an aborted rollout in any arm (96 scenes left). Here every scene stays:
- paper (default): an aborted rollout keeps AlpaSim's forced score 0 and its whole loss goes to a fifth term
  `aborted`, so slow + frozen + cut4m + zeroed + aborted = 1 - score and the arm means equal the paper's Table 2.
- measured: the two mid-run-abort scenes (3b934c9a, ceb3d0d5) use the first two launched attempts, decomposed from
  the parquets h100_reeval_aborted.py re-scored; the two init-fail scenes (no ASL) stay `aborted`.
Recomputed: per-arm terms [1], how the scored window ended [2], loss by ending rule [2b], scene-paired differences vs
dense [3], the pooled split, MoOP - Tyr / LLM-Pruner (alt-route, 4 m rule, score; added 2026-11-09), and gates G1-G3. The behaviour sections of the 10-07 run (pace, stops, gate events per km)
need per-timestep data that aborted rollouts lack and stay as in outputs/closedloop_gap_v1 (96 scenes).
Input: outputs/closedloop_gap_v1/rollouts.parquet (per-rollout anatomy, aborted rows included).
Usage (repo root): .venv/bin/python cvpr27/code/closedloop_gap_100.py [--hard100 paper|measured] [--draws 100000]
"""
import argparse
import glob
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f'{REPO}/experiments/evaluation')
import analyze_closedloop_gap as G

V1 = f'{REPO}/outputs/closedloop_gap_v1'
REEVAL = f'{REPO}/outputs/cvpr27/closedloop_ci_2026-11-07/aborted_reeval'
OUT = f'{REPO}/outputs/cvpr27/closedloop_gap_100'
NUREC = '/mnt/nvme1n1/ad_vla/cache/hub/datasets--nvidia--PhysicalAI-Autonomous-Vehicles-NuRec/snapshots'
MIDRUN = ('3b934c9a', 'ceb3d0d5')
TERMS = G.TERMS + ['aborted']            # slow, frozen, cut4m, zeroed (pre-registered) + aborted
WHY = G.WHY + ['aborted']                # gate, rear, cut, full (post hoc) + aborted
ENDS = ['full', 'offroad', 'atfault', 'rear', 'cut4m', 'aborted']
SETS = [('origin150', 150), ('hard100', 100)]
VS_MOOP = ['tyr', 'lp']                                  # MoOP - arm block (2026-11-09)
MOOP_X = ['alt', 'cut_cost', 'cut4m', 'L_cut', 'score']


def fill_aborted(df):
    """Aborted rows: forced score kept, the whole loss in the fifth term; every other term 0."""
    ab = df['end'] == 'aborted'
    df.loc[ab, 'score'] = df.loc[ab, 'shipped'].astype(float)
    for c in G.TERMS + [f'L_{w}' for w in G.WHY]:
        df.loc[ab, c] = 0.0
    df['aborted'] = np.where(ab, 1.0 - df['score'], 0.0)
    df['L_aborted'] = df['aborted']
    df.loc[ab, 'cut'] = ''
    return df


def measured_rows(df):
    """Replace the mid-run-abort scenes of Hard100 by the first two launched attempts, decomposed from ASL re-scores."""
    uid = pd.read_csv(G.SCENES).set_index('scene_id')['uuid']
    tmp = tempfile.mkdtemp()
    rows = []
    for key, cfg, _ in G.ARMS:
        for s in MIDRUN:
            agg = json.loads(Path(f'{REEVAL}/reeval_aggregates/{s}_{cfg}.json').read_text())['rollouts']
            for r in sorted(agg, key=lambda r: uuid.UUID(r['rollout_id']).time)[:2]:
                sc = r['clipgt_id']
                u = uid[sc]
                link = f'{tmp}/{u}.usdz'
                if not os.path.exists(link):
                    src = G.USDZ / f'{u}.usdz'
                    if not src.exists():
                        src = glob.glob(f'{NUREC}/*/sample_set/*/{sc[7:]}/*.usdz')[0]
                    os.symlink(src, link)
                usdz_dir, G.USDZ = G.USDZ, Path(tmp)
                gt = G.gt_track(u)
                G.USDZ = usdz_dir
                a = G.anatomy(f"{REEVAL}/parquets/{s}_{cfg}/{r['rollout_id']}/metrics.parquet", gt)
                assert a is not None and abs(a[0]['score'] - r['score']) < 1e-6, (cfg, r['rollout_id'])
                rows.append({'set': 'hard100', 'arm': key, 'scene': sc, 'rollout': r['rollout_id'], 'shipped': r['score'],
                             'failure_reason': str(r.get('failure_reason')), **a[0]})
    keep = ~((df['set'] == 'hard100') & df['scene'].str[7:15].isin(MIDRUN))
    return pd.concat([df[keep], pd.DataFrame(rows)], ignore_index=True)


def ci(x, idx):
    """x (S,) scene values (NaN-free), idx (B, S) -> [mean, lo, hi, p]."""
    b = x[idx].mean(1)                                                     # (B,)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return [float(x.mean()), float(lo), float(hi), float(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean())))]


def fmt(c, d=4):
    return f"{c[0]:+.{d}f} [{c[1]:+.{d}f},{c[2]:+.{d}f}]{'*' if c[1] > 0 or c[2] < 0 else ' '}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hard100', choices=['paper', 'measured'], default='paper')
    ap.add_argument('--draws', type=int, default=100_000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    df = pd.read_parquet(f'{V1}/rollouts.parquet')
    if args.hard100 == 'measured':
        df = measured_rows(df)
    df = fill_aborted(df)
    df['alt'] = (df['cut'] == 'alt_route').astype(float)
    df['cut_cost'] = df['cut'].isin(['alt_route', 'excursion', 'hidden_failure', 'stalled']).astype(float)
    v1 = json.loads(Path(f'{V1}/metrics.json').read_text())
    rng = np.random.default_rng(args.seed)
    arms = [k for k, _, _ in G.ARMS]
    lines, res, SC, IDX = [f'# closed-loop gap decomposition, Hard100 = 100 scenes ({args.hard100})'], {}, {}, {}

    def say(s=''):
        lines.append(s)

    for label, n in SETS:
        d = df[df['set'] == label]
        sc = d.groupby(['scene', 'arm'])[['score', 'alt', 'cut_cost'] + TERMS + [f'L_{w}' for w in WHY]].mean().unstack('arm')
        assert len(sc) == n and not sc.isna().any().any(), (label, len(sc))
        assert (d.groupby('arm').size() == 2 * n).all(), label
        SC[label] = sc
        idx = IDX[label] = rng.integers(0, n, (args.draws, n))                 # (B, S)
        R = res[label] = {'n_scenes': n, 'arms': {}, 'paired': {}}
        say(f'\n================ {label}: {n} scenes x 2 rollouts')
        say('\n[1] where each arm\'s score is lost (rollout means; the five terms sum to 1 - score)')
        say(f"  {'arm':15s} paper  score   slow  frozen  cut4m zeroed aborted")
        for k in arms:
            a = d[d['arm'] == k]
            A = R['arms'][k] = {c: float(a[c].mean()) for c in ['score'] + TERMS + [f'L_{w}' for w in WHY]}
            A['paper'] = float(a['shipped'].astype(float).mean())
            if args.hard100 == 'paper' or label == 'origin150':
                assert abs(A['paper'] - A['score']) < 1e-9, (label, k)
            assert abs(sum(A[t] for t in TERMS) - (1 - A['score'])) < 1e-9, (label, k)
            e, c = a['end'].value_counts(normalize=True), a['cut'].value_counts(normalize=True)
            A['end'] = {x: float(e.get(x, 0.0)) for x in ENDS}
            A['cut'] = {x: float(c.get(x, 0.0)) for x in G.CUTS}
            say(f"  {G.LABEL[k]:15s} {A['paper']:.3f}  {A['score']:.3f}  " + '  '.join(f'{A[t]:.3f}' for t in TERMS))
        say('\n[2] how the scored window ended (% of rollouts)')
        say(f"  {'arm':15s} full 20 s  off-road  at-fault  rear-ended  aborted | 4 m rule: route end  >=80% done"
            f'  alt-route  excursion  hidden-fail  stalled')
        for k in arms:
            A = R['arms'][k]
            say(f"  {G.LABEL[k]:15s}   {100 * A['end']['full']:5.1f}     {100 * A['end']['offroad']:5.1f}"
                f"     {100 * A['end']['atfault']:5.1f}       {100 * A['end']['rear']:5.1f}    {100 * A['end']['aborted']:5.1f} |"
                + ''.join(f"     {100 * A['cut'][x]:6.1f}" for x in G.CUTS))
        say('\n[2b] the same loss, each rollout given to the one rule that ended its window')
        say(f"  {'arm':15s} 1-score = gate (zeroed)  + rear-ended  + 4 m rule  + ran 20 s short  + aborted")
        for k in arms:
            A = R['arms'][k]
            say(f"  {G.LABEL[k]:15s}  {1 - A['score']:.3f}  =     {A['L_gate']:.3f}           {A['L_rear']:.3f}"
                f"        {A['L_cut']:.3f}           {A['L_full']:.3f}        {A['L_aborted']:.3f}")
        say('\n[3] score difference vs Baseline (scene-paired, 95% scene-cluster bootstrap); every term is'
            ' -(loss_arm - loss_ref), so the terms of a row add up to the score difference')
        for k in arms[1:]:
            row = {}
            for x in ['score', 'alt', 'cut_cost'] + TERMS + [f'L_{w}' for w in WHY]:
                sign = -1.0 if (x in TERMS or x.startswith('L_')) else 1.0
                row[x] = ci((sign * (sc[x][k] - sc[x]['baseline'])).to_numpy(), idx)
            R['paired'][k] = row
            say(f"  {G.LABEL[k]:15s} score {fmt(row['score'])}")
            say('  ' + ' ' * 15 + ' terms:   ' + '  '.join(f'{t} {fmt(row[t])}' for t in TERMS))
            say('  ' + ' ' * 15 + ' by rule: ' + '  '.join(f'{w} {fmt(row["L_" + w])}' for w in WHY))
            say('  ' + ' ' * 15 + f" alt-route rate {fmt(row['alt'])}   4 m-rule costly cuts {fmt(row['cut_cost'])}")

    # pooled, set-stratified
    P = res['pooled'] = {'n_scenes': sum(n for _, n in SETS), 'paired': {}}
    say(f"\n================ both sets pooled: {P['n_scenes']} scenes (set-stratified), difference vs Baseline")
    say(f"  {'arm':15s} score                          = " + '  + '.join(f'{w:21s}' for w in WHY))
    for k in arms[1:]:
        row = {}
        for x in ['score', 'alt'] + TERMS + [f'L_{w}' for w in WHY]:
            sign = -1.0 if (x in TERMS or x.startswith('L_')) else 1.0
            dd = {s: (sign * (SC[s][x][k] - SC[s][x]['baseline'])).to_numpy() for s, _ in SETS}   # (S,) each
            b = sum(dd[s][IDX[s]].sum(1) for s, _ in SETS) / P['n_scenes']                         # (B,)
            lo, hi = np.percentile(b, [2.5, 97.5])
            row[x] = [float(np.concatenate(list(dd.values())).mean()), float(lo), float(hi),
                      float(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean())))]
        P['paired'][k] = row
        say(f"  {G.LABEL[k]:15s} {fmt(row['score'])}  " + '  '.join(fmt(row['L_' + w]) for w in WHY))

    # MoOP vs the other two external baselines (added 2026-11-09). Reuses the scene resamples of [3], so every number
    # above is unchanged. Rates are MoOP - arm; score terms are score contributions -(loss_MoOP - loss_arm).
    res['moop_minus'] = {}
    say('\n================ MoOP - Tyr / LLM-Pruner (scene-paired, same resamples as [3]); rates MoOP - arm,'
        ' terms as score contribution -(loss_MoOP - loss_arm)')
    for k in VS_MOOP:
        res['moop_minus'][k] = {}
        for label in [s for s, _ in SETS] + ['pooled']:
            row = {}
            for x in MOOP_X:
                sign = -1.0 if (x in TERMS or x.startswith('L_')) else 1.0
                if label == 'pooled':
                    dd = {s: (sign * (SC[s][x]['dual'] - SC[s][x][k])).to_numpy() for s, _ in SETS}   # (S,) each
                    b = sum(dd[s][IDX[s]].sum(1) for s, _ in SETS) / P['n_scenes']                     # (B,)
                    lo, hi = np.percentile(b, [2.5, 97.5])
                    row[x] = [float(np.concatenate(list(dd.values())).mean()), float(lo), float(hi),
                              float(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean())))]
                else:
                    row[x] = ci((sign * (SC[label][x]['dual'] - SC[label][x][k])).to_numpy(), IDX[label])
            res['moop_minus'][k][label] = row
            say(f"  MoOP - {G.LABEL[k]:10s} {label:9s} alt-route {fmt(row['alt'])}  4 m-rule costly cuts "
                f"{fmt(row['cut_cost'])}  cut4m {fmt(row['cut4m'])}  L_cut {fmt(row['L_cut'])}  score {fmt(row['score'])}")

    # gates, as pre-registered in plans/2026-10-07_closedloop-gap-decomposition.md (G3 via the slow term only)
    o, h = res['origin150']['paired']['dual'], res['hard100']['paired']['dual']
    say('\n================ pre-registered gates (MoOP vs Baseline); `aborted` is a post hoc row, not a gate')
    g1 = {}
    for tm in TERMS:
        sig = o[tm][1] > 0 or o[tm][2] < 0
        half = abs(o[tm][0]) >= 0.5 * abs(o['score'][0])
        same = np.sign(o[tm][0]) == np.sign(h[tm][0])
        g1[tm] = bool(sig and half and same)
        say(f"  G1 {tm:7s}: Origin150 {fmt(o[tm])} = {100 * o[tm][0] / o['score'][0]:.0f}% of the gap;"
            f" Hard100 {fmt(h[tm])} = {100 * h[tm][0] / h['score'][0]:.0f}%  ->  {'EXPLAINS' if g1[tm] else 'no'}")
    g2 = bool(o['alt'][2] < 0 and h['alt'][2] < 0)
    g3 = bool(o['slow'][1] > 0 and h['slow'][1] > 0)
    say(f"  G2 H-route: alt-route rate MoOP - dense  Origin150 {fmt(o['alt'])}  Hard100 {fmt(h['alt'])}"
        f"  ->  {'SUPPORTED' if g2 else 'not supported'}")
    say(f"  G3 H-stop : slow term {fmt(o['slow'])} / {fmt(h['slow'])}  ->  {'SUPPORTED' if g3 else 'not supported'}")
    res['gates'] = {'G1': g1, 'G2': g2, 'G3': g3}

    # side by side with the 10-07 run (Hard100 96 scenes)
    say('\n================ Hard100: 96 scenes (closedloop_gap_v1) vs 100 scenes (this run)')
    hv = v1['hard100']
    say(f"  {'arm':15s} alt-route %  96 -> 100 | cut4m term  96 -> 100 | score  96 -> 100")
    for k in arms:
        a96, a100 = hv['arms'][k], res['hard100']['arms'][k]
        say(f"  {G.LABEL[k]:15s}   {100 * a96['cut']['alt_route']:5.1f} -> {100 * a100['cut']['alt_route']:5.1f}"
            f"   |   {a96['cut4m']:.3f} -> {a100['cut4m']:.3f}   |   {a96['score']:.3f} -> {a100['score']:.3f}")
    p96 = hv['paired']['dual']
    say(f"  G2 alt-route MoOP - dense, Hard100: 96 scenes {p96['alt'][0]:+.4f} [{p96['alt'][1]:+.4f},{p96['alt'][2]:+.4f}]"
        f"  ->  100 scenes {fmt(h['alt'])}")
    say(f"  G1 cut4m MoOP - dense, Hard100: 96 scenes {p96['cut4m'][0]:+.4f} [{p96['cut4m'][1]:+.4f},{p96['cut4m'][2]:+.4f}]"
        f"  ->  100 scenes {fmt(h['cut4m'])}")

    os.makedirs(OUT, exist_ok=True)
    tag = '' if args.hard100 == 'paper' else '_measured'
    Path(f'{OUT}/summary{tag}.txt').write_text('\n'.join(lines) + '\n')
    Path(f'{OUT}/metrics{tag}.json').write_text(json.dumps({'hard100': args.hard100, 'draws': args.draws,
                                                            'seed': args.seed, **res}, indent=1))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
