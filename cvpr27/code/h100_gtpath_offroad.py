"""Per hard100 scene: how many arm-rollouts go offroad (AlpaSim aggregated flag) and how far the ego is
from the recorded GT path at the first offroad timestep (raw per-timestep parquet). Rollouts without
metrics (aborted) are skipped. Heuristic for map-flaw candidates, not a map audit.
Usage (repo root): .venv/bin/python cvpr27/code/h100_gtpath_offroad.py
"""
import collections
import glob
import json
import os
from pathlib import Path

import pandas as pd

RUNS = '/mnt/nvme1n1/ad_vla/results/chan/alpasim-runs'
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = f'{REPO}/outputs/cvpr27/closedloop_ci_2026-11-07/aborted_reeval'

sc = collections.defaultdict(lambda: {'n': 0, 'off': 0, 'd': [], 't': []})
for d in sorted(glob.glob(RUNS + '/h100_merged_*')):
    for r in json.loads(Path(d + '/aggregate/results-summary.json').read_text())['rollouts']:
        m = r['metrics']
        if m.get('offroad') is None:
            continue
        s = sc[r['clipgt_id'][7:15]]
        s['n'] += 1
        if m['offroad'] > 0:
            s['off'] += 1
            x = pd.read_parquet(f"{d}/rollouts/{r['clipgt_id']}/{r['rollout_id']}/metrics.parquet",
                                columns=['name', 'timestamps_us', 'values'])
            t0 = x.timestamps_us.min()
            o = x[(x.name == 'offroad') & (x['values'] > 0)].timestamps_us.min()
            g = x[(x.name == 'dist_to_gt_trajectory') & (x.timestamps_us <= o)].sort_values('timestamps_us')
            s['d'].append(g['values'].iloc[-1])
            s['t'].append((o - t0) / 1e6)

tot = sum(s['off'] for s in sc.values())
n = sum(s['n'] for s in sc.values())
rows = []
for k, s in sc.items():
    if s['off']:
        rows.append((k, s['off'], s['n'], sorted(s['d'])[len(s['d']) // 2], sorted(s['t'])[len(s['t']) // 2]))
rows.sort(key=lambda r: -r[1])
near = [r for r in rows if r[3] < 1.0]
lines = [f'offroad rollouts {tot}/{n} over {len(sc)} scenes; scenes with any offroad: {len(rows)}',
         (f'scenes whose median first offroad is <1 m from the GT path: {len(near)}, '
          f'holding {sum(r[1] for r in near)}/{tot} offroad rollouts'),
         f'scenes offroad in every rollout: {sum(1 for r in rows if r[1] == r[2])}',
         'scene     off/n   med_dGT_m  med_t_s']
lines += [f'{r[0]}  {r[1]:3d}/{r[2]:<3d}  {r[3]:8.2f}  {r[4]:7.2f}' for r in rows]
Path(f'{OUT}/gtpath_offroad.txt').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines[:3]))
