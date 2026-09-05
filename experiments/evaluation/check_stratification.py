"""Was every calibration set drawn with the same stratification calib_100 got?

calib_100 came from make_eval_sets.py's greedy distribution matcher over six weighted
attributes (country 4.0, platform_class 2.0, time_of_day 2.0, season 1.5, month 1.0,
radar_config 1.0). Whether the later sets got the same treatment decides whether the
comparisons built on them are one-factor.

Two things are checked per set:
  1. does the manifest even carry the six attribute columns? (a set drawn without the
     matcher has no reason to)
  2. weighted L1 against the parent split, recomputed here from the clip index -- the
     matcher's own objective, so a matched draw scores ~0.03 and a random one ~0.2
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/cvlab21/project/chan/alpamayo-model-compression")
sys.path.insert(0, str(REPO / "experiments" / "evaluation"))
from make_eval_sets import ATTR, derive, weighted_l1

AV = Path("/mnt/nvme1n1/ad_vla/data/physicalai_av")
SETS = Path("/mnt/nvme1n1/ad_vla/outputs/chan/eval_sets")

ci = pd.read_parquet(AV / "clip_index.parquet")
dc = pd.read_parquet(AV / "metadata" / "data_collection.parquet")
splits = {s: derive(ci[ci.split == s], dc) for s in ("train", "val", "test")}

# manifest -> which official split it was drawn from (the target distribution)
PARENT = {
    "calib_100": "train", "calib_nt500": "train", "calib_st4000": "train",
    "calib_tr500": "train", "calib_train500": "train",
    "calib_val100": "val", "calib_ood_100": None,   # OOD pool, no single parent split
}

print(f"{'manifest':16s} {'clips':>6s} {'6속성 열':>9s} {'weighted L1':>12s} "
      f"{'무작위 기준':>12s}  판정")
for name, parent in PARENT.items():
    p = SETS / f"{name}.parquet"
    if not p.exists():
        continue
    d = pd.read_parquet(p)
    col = "clip_id" if "clip_id" in d.columns else d.columns[0]
    ids = [str(x) for x in d[col]]
    has_attr = all(a in d.columns for a in ATTR)

    if parent is None:
        print(f"{name:16s} {len(ids):6d} {has_attr!s:>9s} {'—':>12s} {'—':>12s}  "
              f"OOD 풀 — 부모 split 없음")
        continue

    full = splits[parent]
    sel = full[full.index.isin(ids)]
    if len(sel) == 0:
        print(f"{name:16s} {len(ids):6d} {has_attr!s:>9s} "
              f"{'인덱스 불일치':>12s}")
        continue

    # the matcher's objective, recomputed: weighted L1 of the draw against the full split
    counts, targets = {}, {}
    for a in ATTR:
        fs, ps = full[a].map(str), sel[a].map(str)
        cats = sorted(set(fs) | set(ps))
        counts[a] = np.array([(ps == c).sum() for c in cats], dtype=float)
        t = fs.value_counts(normalize=True)
        targets[a] = np.array([t.get(c, 0.0) for c in cats])
    wl1 = weighted_l1(counts, len(sel), targets)

    rng = np.random.default_rng(0)
    rnd = []
    for _ in range(20):
        idx = rng.choice(len(full), len(sel), replace=False)
        rs = full.iloc[idx]
        c = {}
        for a in ATTR:
            fs, ps = full[a].map(str), rs[a].map(str)
            cats = sorted(set(fs) | set(ps))
            c[a] = np.array([(ps == c2).sum() for c2 in cats], dtype=float)
        rnd.append(weighted_l1(c, len(rs), targets))
    rmean = float(np.mean(rnd))
    verdict = "매칭됨" if wl1 < rmean / 3 else ("무작위 수준" if wl1 > rmean * 0.7
                                              else "부분적")
    print(f"{name:16s} {len(sel):6d} {has_attr!s:>9s} {wl1:12.4f} {rmean:12.4f}  "
          f"{verdict}")
