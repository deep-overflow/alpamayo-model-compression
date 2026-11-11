#!/bin/bash
# Gate G0 on Hangang, then queue and submit the Alpamayo 2 u40 OOD run (iclr27-rebuttal/plans/2026-11-10_a2-u40-ood-gh200.md).
# From cvlab21 after push_a2_ood_gh200.sh; needs the kisti6 login socket (ssh -fN kisti6).
#
#   bash iclr27-rebuttal/code/launch_a2_ood_gh200.sh          # G0, queue, 1 holder
#   G0_ONLY=1 bash ...    checks only;   HOLDERS=2 bash ...    more holders (5thEx: 5 submitted per user, shared with lance)
#   HOLDER_TIME=10:00:00 (default; ~24 tasks, about 4.5 h on one 4-GPU holder if GH200 runs ~10 s/clip)
#   SETS="oodval" bash ...                 check and queue only these sets (default "oodval oodtrain"; also test500)
#   ARMS="dense moop_u50 ..." bash ...     arms to queue (default the u40 run's "dense moop fm_only ce_only"); masks are
#                                          masks/masks_<arm>.npz, their cut counts checked by suffix (u40 / _u50 / _u50_q8)
#   QUEUE=/appsdata/e1997a06/a2_ood/queue_u50q8 GATE_MASKS=moop_u50_q8 REPRO_CHECK=1   a separate run (2026-11-11,
#                                          plans/2026-11-11_a2-u50-q8-gh200.md): own queue and gate, gate masks, dense check
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
Q=${QUEUE:-$W/queue}
PY=/scratch/e1997a06/project/alpamayo-model-compression/.venv/bin/python
echo "== G0 /appsdata group quota"
lfs quota -g sa0713 /appsdata | tail -1 | awk '{free = $8 - $6; printf "  files %d / %d, free %d\n", $6, $8, free
  if (free < 20000) { print "  FAIL: under 20000 free files"; exit 1 }}'
echo "== G0 inputs"
# glogin: one thread per library and no GPU -- on 11-10 the import hung with OpenBLAS failing pthread_create at the
# login node's RLIMIT_NPROC (8192) and cudaGetDeviceCount reporting out of memory
SETS="$SETS" ARMS="$ARMS" A2_DISTILL=$W/code/a2d OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
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
CUTS = {"_u50_q8": (8, 14848), "_u50": (32, 12800), "": (26, 10203)}  # Q heads / MLP channels cut per layer
for arm in [a for a in os.environ["ARMS"].split() if not a.startswith("dense")]:
    sfx = next(k for k in CUTS if arm.endswith(k))
    nq, nm = CUTS[sfx]
    z = np.load(W / "masks" / f"masks_{arm}.npz")
    q, m = z["q_mask"], z["mlp_mask"]
    chk(q.shape == (64, 64) and m.shape == (64, 25600) and ((q == 0).sum(1) == nq).all() and ((m == 0).sum(1) == nm).all(),
        f"masks_{arm}: Q cut {int((q == 0).sum())}, MLP cut {int((m == 0).sum())} (want {64 * nq} / {64 * nm})")
want = os.environ["SETS"].split()
for stem, cache, n in [x for x in (("test_500", "test500_a2", 500), ("ood_val", "ood_a2", 262),
                                   ("ood_train", "ood_train_a2", 1271))
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
declare -A NSH=([test500]=2 [oodval]=2 [oodtrain]=4)
for set in test500 oodval oodtrain; do
  [[ " $SETS " == *" $set "* ]] || continue
  for a in $ARMS; do for s in $(seq 0 $((NSH[$set] - 1))); do add "$a" "$set" "$s" "${NSH[$set]}"; done; done
done
echo "queue: $(grep -cvE '^(#|$)' "$Q/tasks.txt") lines"
cd "$W"
for i in $(seq 1 "$HOLDERS"); do
  sbatch --time="$HOLDER_TIME" --export="ALL,QUEUE=$Q,GATE_MASKS=$GATE_MASKS,REPRO_CHECK=$REPRO_CHECK" \
    "$W/code/amc/iclr27-rebuttal/code/a2_ood_gh200.sbatch"
done
sleep 5
squeue -u e1997a06 -o '%.10i %.16j %.9T %.10M %.20S %R'
EOS
)
ssh -o BatchMode=yes kisti6 "G0_ONLY=${G0_ONLY:-} HOLDERS=${HOLDERS:-1} HOLDER_TIME=${HOLDER_TIME:-10:00:00} \
SETS='${SETS:-oodval oodtrain}' ARMS='${ARMS:-dense moop fm_only ce_only}' QUEUE='${QUEUE:-/appsdata/e1997a06/a2_ood/queue}' \
GATE_MASKS='${GATE_MASKS:-moop}' REPRO_CHECK='${REPRO_CHECK:-0}' bash -s" \
  <<< "$REMOTE" 2>&1 | grep -vE 'Deprecation|Redirecting'
