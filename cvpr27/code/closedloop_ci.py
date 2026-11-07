"""95% CIs for the 9-arm Origin150 / Hard100 closed-loop results (scene-cluster bootstrap).

Per arm: score, progress (progress_clipped_rel), offroad and at-fault collision rates with a scene-cluster
percentile CI (+ rollout-level Wilson for the rates) and the paired difference vs baseline. Then the
scene-score difference of dual / dualexp against every other arm, per set and pooled (set-stratified).
Hard100 uses the corrected rollout set: the two init-fail scenes are dropped for every arm and the two
mid-run-abort scenes use the first two launched attempts re-scored by h100_reeval_aborted.py.
Origin150 lp_r50 comes from the 2026-09-20 cache (its run dir was deleted; the cache matches the
originals rollout-for-rollout on the other seven arms).
Usage (repo root): .venv/bin/python cvpr27/code/closedloop_ci.py [--seed 0] [--draws 100000]
"""
import argparse
import json
import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = '/mnt/nvme1n1/ad_vla/results/chan/alpasim-runs'
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = f'{REPO}/outputs/cvpr27/closedloop_ci_2026-11-07'
REEVAL = f'{OUT}/aborted_reeval/reeval_aggregates'
LP150_CACHE = f'{REPO}/outputs/alpasim_all_metrics/rollout_metrics.parquet'
ARMS = [('baseline', 'baseline'), ('dual', 'slim_dual_u40_v2'), ('dualexp', 'slim_dualexp_u40_em93p75'),
        ('coc', 'slim_coc_u40_v2'), ('traj', 'slim_traj_u40_v2'), ('tyr', 'slim_tyr_u40_r'), ('tyrK', 'slim_tyrK'),
        ('lp_r50', 'lp_r50'), ('wanda', 'slim_wanda_u40_v2')]
METRICS = ['score', 'progress', 'offroad', 'at_fault']
SETS = ['origin150', 'hard100']
INIT_FAIL = {'4bad2f63', 'adb899bd'}
MIDRUN = {'3b934c9a', 'ceb3d0d5'}


def rec(r):
    m = r['metrics']
    return {'scene': r['clipgt_id'], 'score': r['score'], 'progress': m.get('progress_clipped_rel'),
            'offroad': None if m.get('offroad') is None else float(m['offroad'] > 0),
            'at_fault': None if m.get('collision_at_fault') is None else float(m['collision_at_fault'] > 0)}


def load(set_, cfg):
    if set_ == 'origin150' and cfg == 'lp_r50':
        c = pd.read_parquet(LP150_CACHE)
        c = c[(c.set == 'origin150') & (c.arm == 'llm-pruner')]
        return [{'scene': r.scene, 'score': r.score, 'progress': r.progress_clipped_rel,
                 'offroad': float(r.offroad > 0), 'at_fault': float(r.collision_at_fault > 0)} for r in c.itertuples()]
    if set_ == 'origin150':
        return [rec(r) for r in json.loads(Path(f'{RUNS}/m2601_merged_{cfg}/aggregate/results-summary.json').read_text())['rollouts']]
    out = [rec(r) for r in json.loads(Path(f'{RUNS}/h100_merged_{cfg}/aggregate/results-summary.json').read_text())['rollouts']
           if r['clipgt_id'][7:15] not in INIT_FAIL | MIDRUN]
    for s in sorted(MIDRUN):
        rs = json.loads(Path(f'{REEVAL}/{s}_{cfg}.json').read_text())['rollouts']
        out += [rec(r) for r in sorted(rs, key=lambda r: uuid.UUID(r['rollout_id']).time)[:2]]
    return out


def wilson(k, n, z=1.959964):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(c - h), float(c + h)


def boot(x, idx):
    """x (S,) per-scene values, idx (B, S) resampled scene indices -> mean, CI, two-sided bootstrap p."""
    b = x[idx].mean(1)  # (B,)
    lo, hi = np.percentile(b, [2.5, 97.5])
    p = min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()))
    return float(x.mean()), [float(lo), float(hi)], float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--draws', type=int, default=100_000)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    per, data_all, result, lines = {}, {}, {}, []
    for set_ in SETS:
        data = {k: load(set_, c) for k, c in ARMS}
        scenes = sorted({x['scene'] for x in data['baseline']})
        si = {s: i for i, s in enumerate(scenes)}
        for k, v in data.items():
            assert sorted({x['scene'] for x in v}) == scenes, (set_, k)
            assert all(x[m] is not None for x in v for m in METRICS), (set_, k, 'unmeasured rollout')
            for m in METRICS:
                acc, cnt = np.zeros(len(scenes)), np.zeros(len(scenes))
                for x in v:
                    acc[si[x['scene']]] += x[m]
                    cnt[si[x['scene']]] += 1
                per[set_, k, m] = acc / cnt  # (S,)
        data_all[set_] = data
        idx = rng.integers(0, len(scenes), (args.draws, len(scenes)))  # (B, S)
        res = {'n_scenes': len(scenes), 'n_rollouts': {k: len(v) for k, v in data.items()}, 'arms': {}}
        lines.append(f"### {set_}  ({len(scenes)} scenes, rollouts/arm {sorted(set(res['n_rollouts'].values()))})")
        for m in METRICS:
            pct = m in ('offroad', 'at_fault')
            lines.append(f'  [{m}]' + (' (%)' if pct else ''))
            f = (lambda v: f'{100 * v:.1f}') if pct else (lambda v: f'{v:.3f}')
            for k, _ in ARMS:
                mean, ci, _ = boot(per[set_, k, m], idx)
                row = {'mean': mean, 'ci': ci}
                s = f'    {k:9s} {f(mean):>6s} [{f(ci[0])}, {f(ci[1])}]'
                if pct:
                    kk = int(sum(r[m] for r in data[k]))
                    row.update(count=kk, wilson=wilson(kk, len(data[k])))
                    s += f"  n={kk:3d}  wilson[{f(row['wilson'][0])}, {f(row['wilson'][1])}]"
                if k != 'baseline':
                    dm, dci, p = boot(per[set_, k, m] - per[set_, 'baseline', m], idx)
                    row['delta'] = {'mean': dm, 'ci': dci, 'p_boot': p, 'sig': dci[0] > 0 or dci[1] < 0}
                    s += f"   Δ {'+' if dm >= 0 else ''}{f(dm)} [{f(dci[0])}, {f(dci[1])}]{'*' if row['delta']['sig'] else ' '} p={p:.3g}"
                res['arms'].setdefault(k, {})[m] = row
                lines.append(s)
        result[set_] = res

    # paired difference ref - k per set and pooled (set-stratified); one resample shared by all comparisons
    n = {s: len(per[s, 'baseline', 'score']) for s in SETS}
    idx = {s: rng.integers(0, n[s], (args.draws, n[s])) for s in SETS}

    def paired(ref, k, m):
        d = {s: per[s, ref, m] - per[s, k, m] for s in SETS}  # (S,) each
        row = {}
        for s in SETS:
            dm, dci, p = boot(d[s], idx[s])
            row[s] = {'mean': dm, 'ci': dci, 'p_boot': p}
        bp = sum(d[s][idx[s]].sum(1) for s in SETS) / sum(n.values())  # (B,)
        lo, hi = np.percentile(bp, [2.5, 97.5])
        row['pooled'] = {'mean': float(np.concatenate([d[s] for s in SETS]).mean()), 'ci': [float(lo), float(hi)],
                         'p_boot': float(min(1.0, 2 * min((bp <= 0).mean(), (bp >= 0).mean())))}
        return row

    def fmt(row, pct=False):
        sc = 100 if pct else 1
        nd = 1 if pct else 3
        out = []
        for s in SETS + ['pooled']:
            r = row[s]
            sig = '*' if r['ci'][0] > 0 or r['ci'][1] < 0 else ' '
            out.append(f"{s} {sc * r['mean']:+.{nd}f} [{sc * r['ci'][0]:+.{nd}f}, {sc * r['ci'][1]:+.{nd}f}]{sig} "
                       f"p={r['p_boot']:.3g}")
        return '   '.join(out)

    pair = {}
    lines += ['', '### scene score: reference arm minus other arm (pooled = origin150 + hard100, set-stratified)']
    for ref in ('dual', 'dualexp'):
        lines.append(f'  {ref} minus ...')
        for k, _ in ARMS:
            if k != ref:
                pair[f'{ref}-{k}'] = paired(ref, k, 'score')
                lines.append(f'    {k:9s} ' + fmt(pair[f'{ref}-{k}']))

    # MoOP (dual) against the paper's Tyr, LLM-Pruner and FM-only rows on every metric (rates in pp)
    focus = {}
    lines += ['', '### all metrics: dual (MoOP) minus tyr / lp_r50 / traj (same resample as above)']
    for k in ('tyr', 'lp_r50', 'traj'):
        lines.append(f'  dual minus {k}')
        for m in METRICS:
            focus[f'dual-{k}', m] = paired('dual', k, m)
            lines.append(f'    {m:9s} ' + fmt(focus[f'dual-{k}', m], pct=m in ('offroad', 'at_fault')))

    os.makedirs(OUT, exist_ok=True)
    Path(f'{OUT}/metrics_seed{args.seed}.json').write_text(json.dumps(
        {'seed': args.seed, 'draws': args.draws, 'per_arm': result, 'pairwise_score': pair,
         'dual_vs_baselines': {f'{a}|{m}': v for (a, m), v in focus.items()}}, indent=1))
    Path(f'{OUT}/summary_seed{args.seed}.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
