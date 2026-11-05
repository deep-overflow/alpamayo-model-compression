"""CoC degeneration per rollout for the 76-scene x 4-rollout runs (cvpr27/plans/2026-11-05_rollout4-76scenes.md).

Same measure as the s600 analysis (analyze_dual_s600.py: analyze_alpasim.coc_stats(...)["degenerate_frac"], the share
of driver steps whose CoC is empty or degenerate), read from each rollout's rollout.asl. Needs alpasim's venv:
  /mnt/nvme1n1/ad_vla/venvs/alpasim-chan/bin/python cvpr27/code/coc_rollout4_76.py [run ...]
Writes outputs/cvpr27/rollout4_76/coc_<run>.csv (scene, rollout, coc_degenerate); analyze_rollout4_76.py reads them.
"""

import asyncio
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))
import analyze_alpasim as AA

RUNS = Path("/home/cvlab21/project/chan/alpasim-runs")
OUT = REPO / "outputs" / "cvpr27" / "rollout4_76"
DEFAULT = ["r4l0_merged_slim_dual_u40_l0keep_v2", "r4ty_merged_slim_tyr_off", "r4lp_merged_slim_lp_m24",
           "r4dn_merged_baseline"]


def coc(asl):
    return AA.coc_stats(asyncio.run(AA._read_coc(asl)))["degenerate_frac"] if asl.exists() else float("nan")


def main():
    for run in sys.argv[1:] or DEFAULT:
        asls = sorted((RUNS / run / "rollouts").glob("*/*/rollout.asl"))
        if not asls:
            print(f"{run}: no rollouts yet")
            continue
        with Pool(12) as pool:
            vals = pool.map(coc, asls, chunksize=4)
        d = pd.DataFrame({"scene": [a.parent.parent.name for a in asls], "rollout": [a.parent.name for a in asls],
                          "coc_degenerate": vals})
        d.to_csv(OUT / f"coc_{run}.csv", index=False)
        print(f"{run}: {len(d)} rollouts, CoC degenerate {d.coc_degenerate.mean():.4f}")


if __name__ == "__main__":
    main()
