"""LingoQA train QA for the mtw24_kd2 run (minitron/plans/2026-10-26_kisti-training-plan.md).

Driving : VQA = 3 : 1 by sample count, so the VQA set holds round(n_drive_train / 3) QA drawn at
random (seed 0) from LingoQA's Scenery train split, after holding out 32 whole segments (seed 0);
256 QA of those segments form the held-out loss set. Context is the GT answer (teacher-forced), so
only question, answer and the segment's frames are needed.

Usage:
  .venv/bin/python minitron/code/make_vqa_set.py
Writes outputs/minitron/kd_sets/vqa_lingo_{train,heldout}.parquet (+ vqa_lingo_config.json).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "minitron" / "kd_sets"
LINGO = Path("/mnt/nvme1n1/ad_vla/data/lingoqa/train/train.parquet")
N_HELD_SEG, N_HELD_QA, SEED = 32, 256, 0


def main():
    kd5 = pd.read_parquet(OUT / "kd_r5.parquet")
    held_clips = set(json.loads((OUT / "heldout_clips_r0.json").read_text()))
    n_drive = int((~kd5.clip_id.isin(held_clips)).sum())
    n_vqa = round(n_drive / 3)

    qa = pd.read_parquet(LINGO)[["question_id", "segment_id", "question", "answer"]]
    rng = np.random.default_rng(SEED)
    segs = np.array(sorted(qa.segment_id.unique()))
    held_segs = set(rng.choice(segs, N_HELD_SEG, replace=False))
    held = qa[qa.segment_id.isin(held_segs)]
    held = held.iloc[rng.permutation(len(held))[:N_HELD_QA]].reset_index(drop=True)
    pool = qa[~qa.segment_id.isin(held_segs)]
    train = pool.iloc[rng.permutation(len(pool))[:n_vqa]].reset_index(drop=True)
    train.to_parquet(OUT / "vqa_lingo_train.parquet", index=False)
    held.to_parquet(OUT / "vqa_lingo_heldout.parquet", index=False)
    cfg = {"source": str(LINGO), "seed": SEED, "n_drive_train": n_drive, "ratio_drive_to_vqa": 3,
           "n_train": len(train), "n_train_segments": int(train.segment_id.nunique()),
           "n_heldout_segments": N_HELD_SEG, "n_heldout": len(held),
           "context": "GT answer, teacher-forced; KL on answer tokens + <|answer_end|>"}
    (OUT / "vqa_lingo_config.json").write_text(json.dumps(cfg, indent=2))
    print(json.dumps(cfg))


if __name__ == "__main__":
    main()
