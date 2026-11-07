"""hard100 offroad / scene score with the mid-run-aborted rollouts measured from their ASL.

Inputs: each arm's original h100_merged_<cfg> results-summary.json and the re-scored aggregates from
h100_reeval_aborted.py. All 100 scenes x 2 rollouts stay in every denominator. P0 = as shipped (the
paper's Table 2: an aborted rollout scores 0 and counts no offroad; /valid = the 2026-09-20 report's
denominator), P1 = the two mid-run-abort scenes replaced by their first two launched attempts (uuid1 time
order), P2 = by the attempts AlpaSim recorded after retries. The two init-fail scenes have no ASL and stay
as shipped in P1/P2. Asserts that recorded completed rollouts re-score to the shipped score and offroad exactly.
Usage (repo root): .venv/bin/python cvpr27/code/h100_include_aborted.py
"""
import json
import os
import uuid
from pathlib import Path

RUNS = '/mnt/nvme1n1/ad_vla/results/chan/alpasim-runs'
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = f'{REPO}/outputs/cvpr27/closedloop_ci_2026-11-07/aborted_reeval'
CFGS = ['baseline', 'slim_dual_u40_v2', 'slim_dualexp_u40_em93p75', 'slim_coc_u40_v2', 'slim_traj_u40_v2',
        'slim_tyr_u40_r', 'slim_tyrK', 'lp_r50', 'slim_wanda_u40_v2']
REDO = ['3b934c9a', 'ceb3d0d5']          # abort mid-run (route extension near the route end), ASL kept


def off(m):
    v = m.get('offroad')
    return None if v is None else float(v > 0)


lines = ['== per attempt, scenes with mid-run aborts ==']
rows = {}
for a in CFGS:
    orig = json.loads(Path(f'{RUNS}/h100_merged_{a}/aggregate/results-summary.json').read_text())['rollouts']
    re_ = {}
    for s in REDO:
        for r in json.loads(Path(f'{OUT}/reeval_aggregates/{s}_{a}.json').read_text())['rollouts']:
            re_[r['rollout_id']] = r
    for r in orig:  # G0
        q = re_.get(r['rollout_id'])
        if q and r['failure_reason'] in (None, 'offroad', 'collision_at_fault'):
            assert abs(q['score'] - r['score']) < 1e-9 and off(q['metrics']) == off(r['metrics']), (a, r['rollout_id'])
    first, final = [], []
    for s in REDO:
        ids = sorted({rid for rid, q in re_.items() if q['clipgt_id'][7:15] == s}, key=lambda u: uuid.UUID(u).time)
        recorded = {r['rollout_id'] for r in orig if r['clipgt_id'][7:15] == s}
        for i, rid in enumerate(ids):
            q = re_[rid]
            done = os.path.exists(f"{RUNS}/h100_merged_{a}/rollouts/{q['clipgt_id']}/{rid}/_complete")
            tag = ('first' if i < 2 else 'retry') + (' *recorded*' if rid in recorded else '')
            lines.append(f"{a:26s} {s} {rid[:8]} {'complete' if done else 'ABORTED '} {tag:20s} "
                         f"offroad={off(q['metrics'])} score={q['score']:.3f} "
                         f"progress={q['metrics'].get('progress_clipped_rel'):.3f}")
            if i < 2:
                first.append(q)
            if rid in recorded:
                final.append(q)
    keep = [r for r in orig if r['clipgt_id'][7:15] not in REDO]
    rows[a] = {'orig': orig, 'keep': keep, 'first': first, 'final': final}

lines += ['', '== hard100 offroad / score under each policy ==',
          (f"{'arm':26s} {'P0 /200':>8s} {'P0 /valid':>9s} {'P1 first2':>10s} {'P2 recorded':>11s}   "
           f"{'score P0':>8s} {'score P1':>8s} {'score P2':>8s}")]
metrics = {}
for a, d in rows.items():
    o = d['orig']
    n_off = sum(1 for r in o if off(r['metrics']) == 1)
    n_val = sum(1 for r in o if off(r['metrics']) is not None)
    pol = {}
    for p in ('first', 'final'):
        S = d['keep'] + d[p]
        assert len(S) == 200, (a, p, len(S))
        pol[p] = (sum(off(r['metrics']) or 0 for r in S) / 200, sum(r['score'] for r in S) / 200)
    metrics[a] = {'offroad_P0_200': n_off / 200, 'offroad_P0_valid': n_off / n_val, 'n_valid': n_val,
                  'offroad_P1': pol['first'][0], 'offroad_P2': pol['final'][0],
                  'score_P0_200': sum(r['score'] for r in o) / 200, 'score_P1': pol['first'][1], 'score_P2': pol['final'][1]}
    m = metrics[a]
    lines.append(f"{a:26s} {m['offroad_P0_200']:8.3f} {m['offroad_P0_valid']:9.3f} {m['offroad_P1']:10.3f} "
                 f"{m['offroad_P2']:11.3f}   {m['score_P0_200']:8.3f} {m['score_P1']:8.3f} {m['score_P2']:8.3f}")

Path(f'{OUT}/include_aborted.txt').write_text('\n'.join(lines) + '\n')
Path(f'{OUT}/include_aborted.json').write_text(json.dumps(metrics, indent=1))
print('\n'.join(lines))
