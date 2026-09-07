"""Did the rebuilt supernet reproduce the one the shipped checkpoints came from?

The cvlab21 copy of `tyr_supernet_u40` lost its 648 level files; only metadata survived.
tyrK needs that supernet, so it was rebuilt -- but "same flags" is not "same weights", and
if the rebuild differs then `tyr_u40_r`'s published numbers are not a fair reference for
tyrK and the reference has to be rebuilt too.

The check does not need a GPU or a 16 GB write. make_slim derives a tyr arm's kept sets
from the non-zero columns of the supernet weight at the level `final_config.json` assigns
(`col = w.abs().sum(0)`; mlp keeps `col > 0`, attention keeps heads whose head_dim columns
sum above zero). So deriving those sets from the REBUILT supernet and comparing them to
the shipped `slim_tyr_u40_r/slim_meta.json` settles it on CPU in a couple of minutes.

Exit code 0 = reproduced, 1 = differs (caller must not silently reuse the old reference).

    python experiments/head_analysis/verify_supernet_u40.py \
        --supernet outputs/tyr_supernet_u40 \
        --levels outputs/tyr_search_u40/final_config.json \
        --reference outputs/slim_tyr_u40_r/slim_meta.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

NH, HD = 32, 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--supernet", type=Path, required=True)
    ap.add_argument("--levels", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    levels = json.loads(args.levels.read_text())
    ref = json.loads(args.reference.read_text())["vlm"]

    ok, mism = 0, []
    for name, lv in sorted(levels.items()):
        i = int(name.split(".")[1])
        f = args.supernet / name / f"{lv}.pth"
        if not f.exists():
            mism.append((name, lv, "파일 없음", ""))
            continue
        col = torch.load(f, map_location="cpu").abs().sum(0).float()
        if "mlp" in name:
            want = np.where((col > 0).numpy())[0]
            got = np.asarray(ref[i]["mlp"])
        else:
            want = np.where((col.reshape(NH, HD).sum(1) > 0).numpy())[0]
            got = np.asarray(ref[i]["q"])
        if len(want) == len(got) and np.array_equal(np.sort(want), np.sort(got)):
            ok += 1
        else:
            inter = len(set(want.tolist()) & set(got.tolist()))
            mism.append((name, lv, f"{len(want)} vs {len(got)}", f"겹침 {inter}"))

    n = len(levels)
    print(f"모듈 {n}개 중 유지집합 일치 {ok}, 불일치 {len(mism)}")
    for m in mism[:10]:
        print(f"  MISMATCH {m[0]} level={m[1]} {m[2]} {m[3]}")
    verdict = "REPRODUCED" if ok == n else "DIFFERS"
    print(f"판정: {verdict}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"verdict": verdict, "modules": n, "match": ok,
             "mismatch": [list(m) for m in mism]}, indent=2, ensure_ascii=False))
    sys.exit(0 if ok == n else 1)


if __name__ == "__main__":
    main()
