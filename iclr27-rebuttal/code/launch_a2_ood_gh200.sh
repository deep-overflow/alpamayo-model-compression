#!/bin/bash
# Gate G0 on Hangang, then queue and submit the Alpamayo 2 u40 OOD run (iclr27-rebuttal/plans/2026-11-10_a2-u40-ood-gh200.md).
# From cvlab21 after push_a2_ood_gh200.sh; needs the kisti6 login socket (ssh -fN kisti6).
#
#   bash iclr27-rebuttal/code/launch_a2_ood_gh200.sh          # G0, queue, 1 holder
#   G0_ONLY=1 bash ...    checks only;   HOLDERS=2 bash ...    more holders (5thEx: 5 submitted per user, shared with lance)
#   HOLDER_TIME=10:00:00 (default; ~24 tasks, about 4.5 h on one 4-GPU holder if GH200 runs ~10 s/clip)
#   SETS="oodval" bash ...                 check and queue only these sets (default "oodval oodtrain")
#   SETS="oodtrain" HOLDERS=0 bash ...     add OOD-train to a running holder's queue once its cache is pushed
#                                          (idle workers poll the queue for IDLE_MAX = 2 h)
#
# Queue lines `<arm> <set> <shard> <n>`: OOD-Val262 in 2 shards per arm first (dense first), then OOD-train1271 in 4.
set -euo pipefail
stamp() { echo "$(date -d '+9 hours' '+%F %T') KST $*"; }
ssh -O check kisti6 > /dev/null 2>&1 || { stamp "kisti6 login socket down -- run: ssh -fN kisti6"; exit 2; }
REMOTE=$(cat <<'EOS'
set -euo pipefail
W=/appsdata/e1997a06/a2_ood
Q=$W/queue
PY=/scratch/e1997a06/project/alpamayo-model-compression/.venv/bin/python
echo "== G0 /appsdata group quota"
lfs quota -g sa0713 /appsdata | tail -1 | awk '{free = $8 - $6; printf "  files %d / %d, free %d\n", $6, $8, free
  if (free < 20000) { print "  FAIL: under 20000 free files"; exit 1 }}'
echo "== G0 inputs"
# glogin: one thread per library and no GPU -- on 11-10 the import hung with OpenBLAS failing pthread_create at the
# login node's RLIMIT_NPROC (8192) and cudaGetDeviceCount reporting out of memory
SETS="$SETS" A2_DISTILL=$W/code/a2d OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  "$PY" - <<'PY'
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
W = Path("/appsdata/e1997a06/a2_ood")
bad = 0
def chk(ok, msg):
    global bad
    bad += not ok
    print(f"  {'ok ' if ok else 'BAD'} {msg}")
for f in ("code/amc/iclr27-rebuttal/code/run_a2_openloop.py", "code/amc/iclr27-rebuttal/code/a2_ood_gh200.sbatch",
          "code/amc/experiments/head_analysis/eval_lib.py", "code/a2d/experiments/oodval/run_oodval.py",
          "code/a2d/src/alpamayo2_super/models/alpamayo2_super.py"):
    chk((W / f).exists(), f)
snap = W / "hub/models--nvidia--Alpamayo2-Super/snapshots/00554695e729a6ff0b6281fd2c81b18d06e33dbe"
shards = sorted(snap.glob("model-*.safetensors"))
chk(len(shards) == 15 and all(s.resolve().exists() for s in shards), f"A2 weights: {len(shards)} / 15 shards resolve")
for arm in ("moop", "fm_only", "ce_only"):
    z = np.load(W / "masks" / f"masks_{arm}.npz")
    q, m = z["q_mask"], z["mlp_mask"]
    chk(q.shape == (64, 64) and m.shape == (64, 25600) and int((q == 0).sum()) == 64 * 26 and int((m == 0).sum()) == 64 * 10203,
        f"masks_{arm}: Q cut {int((q == 0).sum())}, MLP cut {int((m == 0).sum())} (u40: 1664 / 652992)")
want = os.environ["SETS"].split()
for stem, cache, n in [x for x in (("ood_val", "ood_a2", 262), ("ood_train", "ood_train_a2", 1271))
                       if x[0].replace("_", "") in want]:
    man = pd.read_parquet(W / "code/amc/outputs/eval_sets" / f"{stem}.parquet")
    pre = W / "data/physicalai_av/pre_processed" / cache / "samples"
    miss = [c for c, t in zip(man.clip_id, man.t0_us) if not (pre / f"{c}__t0_{t}.npz").exists()]
    chk(len(man) == n and not miss, f"{stem}: {len(man)} clips, {len(miss)} cache files missing ({cache})")
sys.path.insert(0, str(W / "code/a2d/src"))
import alpamayo2_super.models.alpamayo2_super as m2  # noqa: F401
chk(True, f"alpamayo2_super imports from {Path(m2.__file__).parent}")
sys.exit(1 if bad else 0)
PY
echo "G0 PASS"
[ -n "${G0_ONLY:-}" ] && exit 0
mkdir -p "$Q/claimed"
touch "$Q/tasks.txt"
add() { grep -qxF "$*" "$Q/tasks.txt" || { echo "$*" >> "$Q/tasks.txt"; echo "  + $*"; }; }
if [[ " $SETS " == *" oodval "* ]]; then
  for a in dense moop fm_only ce_only; do for s in 0 1; do add "$a" oodval "$s" 2; done; done
fi
if [[ " $SETS " == *" oodtrain "* ]]; then
  for a in dense moop fm_only ce_only; do for s in 0 1 2 3; do add "$a" oodtrain "$s" 4; done; done
fi
echo "queue: $(grep -cvE '^(#|$)' "$Q/tasks.txt") lines"
cd "$W"
for i in $(seq 1 "$HOLDERS"); do
  sbatch --time="$HOLDER_TIME" "$W/code/amc/iclr27-rebuttal/code/a2_ood_gh200.sbatch"
done
sleep 5
squeue -u e1997a06 -o '%.10i %.16j %.9T %.10M %.20S %R'
EOS
)
ssh -o BatchMode=yes kisti6 "G0_ONLY=${G0_ONLY:-} HOLDERS=${HOLDERS:-1} HOLDER_TIME=${HOLDER_TIME:-10:00:00} \
SETS='${SETS:-oodval oodtrain}' bash -s" \
  <<< "$REMOTE" 2>&1 | grep -vE 'Deprecation|Redirecting'
