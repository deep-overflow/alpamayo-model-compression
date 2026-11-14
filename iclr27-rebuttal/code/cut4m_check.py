"""What the cut4m term (paper Table 6, "score lost to the 4 m rule") measures, per rollout, for MoOP (u40).

Reads the per-rollout anatomy of the 10-07 gap run (outputs/closedloop_gap_v1/rollouts.parquet) with the paper's
aborted-rollout convention (cvpr27/code/closedloop_gap_100.py: fill_aborted) and prints, per set:
score + cut4m and the alternatives (score + L_cut, score_no4m), how cut4m splits over the 4 m cut categories, and
the per-rollout bounds (cut4m on gated rollouts, min / max, score + cut4m <= 1).
Usage (repo root): .venv/bin/python iclr27-rebuttal/code/cut4m_check.py
"""
import os
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f'{REPO}/cvpr27/code')
sys.path.insert(0, f'{REPO}/experiments/evaluation')
import closedloop_gap_100 as C

ARM = 'dual'   # MoOP = slim_dual_u40_v2


def main():
    df = C.fill_aborted(pd.read_parquet(f'{C.V1}/rollouts.parquet'))
    for s, n_sc in C.SETS:
        d = df[(df['set'] == s) & (df['arm'] == ARM)]
        n = len(d)
        assert n == 2 * n_sc and d['scene'].nunique() == n_sc, (s, n)
        sc = d.groupby('scene')[['score', 'cut4m']].mean().mean()                    # scene means, then mean
        print(f'== {s}: MoOP, {n} rollouts / {n_sc} scenes, aborted {(d["end"] == "aborted").sum()}')
        print(f"  rollout mean  score {d['score'].mean():.4f}  cut4m {d['cut4m'].mean():.4f}  L_cut {d['L_cut'].mean():.4f}")
        print(f"  scene mean    score {sc['score']:.4f}  cut4m {sc['cut4m']:.4f}")
        print(f"  score + cut4m {d['score'].mean() + d['cut4m'].mean():.4f}   score + L_cut "
              f"{d['score'].mean() + d['L_cut'].mean():.4f}   score_no4m {d['score_no4m'].mean():.4f}")
        cut = d[d['end'] == 'cut4m']
        print(f"  stopped by the 4 m rule: {len(cut)}; cut4m != 0 on {(d['cut4m'].abs() > 1e-12).sum()}")
        g = d[d['cut'] != ''].groupby('cut').agg(n=('cut4m', 'size'), cut4m=('cut4m', 'sum'), L_cut=('L_cut', 'sum'))
        g[['cut4m', 'L_cut']] /= n                                                    # contribution to the set mean
        g['cut4m_share'] = g['cut4m'] / d['cut4m'].mean()
        print(g.round(4).to_string())
        gated = d[d['why'] == 'gate']
        neg = d[d['cut4m'] < -1e-12]
        print(f"  zeroed by a gate inside the scored window: {len(gated)}, |cut4m| sum {gated['cut4m'].abs().sum():.4f}")
        print(f"  cut4m min {d['cut4m'].min():.4f} (negative on {len(neg)}, {neg['cut4m'].sum() / n:+.4f} of the mean)"
              f"  max {d['cut4m'].max():.4f};  max(score + cut4m) {(d['score'] + d['cut4m']).max():.4f}")


if __name__ == '__main__':
    main()
