"""Scenario-bucket composition of the calibration sets that produced the draw spread.

The six natural draws span 0.950 to 1.774 on test500 while the five recovery-pool draws
span only 0.101, and the standing guess is that a natural bucket mix is cruise-dominated,
so the informative clips -- turns and stops -- are few and swing between draws. That guess
has never been checked on the sets themselves: `recovery_sets/bucket_pool_train.parquet`
covers 9,874 clips, excludes calib_100 by construction, and matches only 7-14 of each
natural block, so it cannot answer it.

Buckets come from the GT ego path alone (`make_train_set.bucket5`), read out of the
egomotion label zips -- no video decode. 621 chunks are on disk; anything else streams.

Usage:
  python experiments/evaluation/bucket_calib_sets.py [--workers 16]
"""

import argparse
import configparser
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Clips outside the 621 local egomotion chunks stream from the gated repo, so the token
# has to be here. It lives in $HOME while the blobs live on /mnt/nvme1n1; setting only one
# of the two breaks the other (see CLAUDE.md). `full_right` is the section with access --
# never take the first section, the file also holds other lab members' tokens.
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("HF_HUB_CACHE", "/mnt/nvme1n1/ad_vla/cache/hub")
if "HF_TOKEN" not in os.environ:
    _cp = configparser.ConfigParser()
    _cp.read(os.path.expanduser("~/.cache/huggingface/stored_tokens"))
    if "full_right" in _cp:
        os.environ["HF_TOKEN"] = _cp["full_right"]["hf_token"]

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "recovery"))
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))

from make_train_set import bucket_clip

O = REPO / "outputs"
ES = O / "eval_sets"
CALIB_T0 = 5_100_000          # the fixed calibration window, in microseconds
ORDER = ["cruise", "decel_stop", "accel", "turn_left", "turn_right"]

SETS = ["calib_100"] + [f"calib_nt100_{b}" for b in "abcde"] \
       + [f"calib_tr100_{b}" for b in "abcde"]
# test500 minADE@6 of `dual` built on each, for reading composition against outcome
OUTCOME = {"calib_100": 0.9498, "calib_nt100_a": 1.0219, "calib_nt100_b": 1.0602,
           "calib_nt100_c": 1.3479, "calib_nt100_d": 1.7742, "calib_nt100_e": 1.7178,
           "calib_tr100_a": 1.2142, "calib_tr100_b": 1.1463, "calib_tr100_c": 1.1008,
           "calib_tr100_d": 1.1688, "calib_tr100_e": 1.2514}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="outputs/calib_buckets")
    args = ap.parse_args()

    # one (clip, t0) task per manifest row; a clip shared by two sets is read once
    tasks, membership = {}, []
    for name in SETS:
        p = ES / f"{name}.parquet"
        if not p.exists():
            print(f"{name}: 매니페스트 없음, 건너뜀")
            continue
        man = pd.read_parquet(p)
        t0s = (man["t0_us"].astype("int64") if "t0_us" in man.columns
               else pd.Series(CALIB_T0, index=man.index, dtype="int64"))
        for cid, t0 in zip(man["clip_id"].astype(str), t0s):
            tasks.setdefault(cid, set()).add(int(t0))
            membership.append((name, cid, int(t0)))
    print(f"{len(membership)} 행, 고유 클립 {len(tasks)}개", flush=True)

    rows = []
    todo = [(c, sorted(t)) for c, t in tasks.items()]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(bucket_clip, t) for t in todo]
        for i, f in enumerate(as_completed(futs), 1):
            rows.extend(f.result())
            if i % 200 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    got = pd.DataFrame(rows)
    err = got["error"].notna().sum() if "error" in got.columns else 0
    print(f"버킷 계산 완료: {len(got)} 행, 실패 {err}")
    got = got[got.get("bucket").notna()] if "bucket" in got.columns else got
    key = {(c, t): b for c, t, b in zip(got.clip_id, got.t0_us, got.bucket)}

    mem = pd.DataFrame(membership, columns=["set", "clip_id", "t0_us"])
    mem["bucket"] = [key.get((c, t)) for c, t in zip(mem.clip_id, mem.t0_us)]

    out_rows = {}
    for name in SETS:
        g = mem[mem["set"] == name]
        b = g.bucket.dropna()
        if len(b) == 0:
            continue
        frac = b.value_counts(normalize=True)
        out_rows[name] = {k: float(frac.get(k, 0.0)) for k in ORDER}
        out_rows[name]["n"] = len(b)
        out_rows[name]["커버"] = len(b) / len(g)
        out_rows[name]["minADE6"] = OUTCOME.get(name, np.nan)

    df = pd.DataFrame(out_rows).T
    pd.set_option("display.width", 220)
    print("\n=== 캘리브레이션 세트의 실제 버킷 구성 ===")
    print(df.to_string(float_format=lambda x: f"{x:.3f}"))

    info = 1 - df["cruise"].astype(float)
    print("\n=== cruise가 아닌(정보량 있는) 비중 ===")
    fams = {"nt (자연)": [k for k in df.index if "nt100" in k or k == "calib_100"],
            "tr (회복 풀)": [k for k in df.index if "tr100" in k]}
    summary = {}
    for fam, keys in fams.items():
        v = info.loc[[k for k in keys if k in info.index]]
        a = df.loc[v.index, "minADE6"].astype(float)
        summary[fam] = {"info_mean": float(v.mean()), "info_sd": float(v.std()),
                        "ade_sd": float(a.std())}
        print(f"  {fam:12s} 평균 {v.mean():.3f}  세트간 SD {v.std():.4f}"
              f"   (결과 SD {a.std():.4f})")

    nt_keys = [k for k in fams["nt (자연)"] if k in info.index]
    if len(nt_keys) >= 5:
        from scipy.stats import spearmanr
        r = spearmanr(info.loc[nt_keys], df.loc[nt_keys, "minADE6"].astype(float))
        print(f"\n자연 6추출에서 info 비중 vs 결과: rho {r.statistic:+.3f} p={r.pvalue:.3f}")
        summary["nt_info_vs_ade_rho"] = float(r.statistic)

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "composition.csv")
    mem.to_parquet(out / "clip_buckets.parquet")
    (out / "metrics.json").write_text(
        json.dumps({"composition": df.to_dict(), "summary": summary},
                   indent=2, ensure_ascii=False, default=float))
    print("\n->", out)


if __name__ == "__main__":
    main()
