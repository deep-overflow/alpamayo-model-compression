#!/bin/bash
# Gate G0 on Hangang, then queue and submit the GH200 OOD-train run of the ICLR27 arms
# (iclr27-rebuttal/plans/2026-11-09_oodtrain-iclr27-arms-gh200.md, steps 0 and 2). From cvlab21, after
# push_oodtrain_gh200.sh; needs the kisti6 login socket (ssh -fN kisti6).
#
#   bash iclr27-rebuttal/code/launch_oodtrain_gh200.sh             # G0, queue, 4 holders
#   G0_ONLY=1 bash ...    checks only;   HOLDERS=2 bash ...    fewer holders (5thEx allows 5 running per user)
#
# Queue lines (cvpr27/code/eval_a100.sbatch kinds): `ol <arm> oodtrain <s> 4` for the 8 arms, dense first so
# gate G1 (dense shard 0 vs A100) can be read early, then `nll <arm> oodtrain <s> 2`. Rows land in
# /scratch/e1997a06/outputs/iclr27-rebuttal/gh200/ (EXP_ROOT=iclr27-rebuttal, OUT_SITE=gh200).
set -euo pipefail
stamp() { echo "$(date -d '+9 hours' '+%F %T') KST $*"; }
ssh -O check kisti6 > /dev/null 2>&1 || { stamp "kisti6 login socket down -- run: ssh -fN kisti6"; exit 2; }
F='Deprecation|Redirecting'
REMOTE=$(cat <<'EOS'
set -euo pipefail
R=/scratch/e1997a06/project/alpamayo-model-compression
Q=$R/outputs/iclr27-rebuttal/gh200/queue_oodtrain
cd "$R"
. ./env.sh
echo "== G0 quota (group sa0713, /scratch)"
lfs quota -g sa0713 /scratch | tail -1 | awk '{files = $6; flim = $8; free = flim - files
  printf "  files %d / %d, free %d\n", files, flim, free; if (free < 20000) { print "  FAIL: fewer than 20,000 files free"; exit 1 }}'
echo "== G0 recipes and inputs"
"$ALPAMAYO_REPO/.venv/bin/python" - <<'PY'
import json, sys
from pathlib import Path
import pandas as pd
out = Path("outputs")
want = {"slim_dual_u40_v2": 2657452032, "slim_dualexp_u40_em93p75": 4369416192, "slim_traj_u40_v2": 2657452032,
        "slim_coc_u40_v2": 2657452032, "slim_wanda_u40_v2": 2657452032, "slim_tyr_u40_r": 2657452032,
        "cvpr27/slim_lp_r50": 2768240640}
bad = 0
for r, n in want.items():
    m = out / r / "slim_meta.json"
    got = json.loads(m.read_text())["params"]["removed"] if m.exists() else None
    ok = got == n
    bad += not ok
    print(f"  {'ok ' if ok else 'BAD'} {r}: removed {got} (want {n})")
st = out / "slim_tyr_u40_r" / "slim_state.pt"
print(f"  {'ok ' if st.exists() else 'BAD'} slim_tyr_u40_r/slim_state.pt {st.stat().st_size if st.exists() else 'missing'}")
bad += not st.exists()
man = pd.read_parquet(out / "eval_sets" / "ood_train.parquet")
pre = Path("/scratch/e1997a06/datasets/physicalai_av/pre_processed/ood/samples")
miss = [c for c, t in zip(man.clip_id, man.t0_us) if not (pre / f"{c}__t0_{t}.npz").exists()]
print(f"  {'ok ' if len(man) == 1271 and not miss else 'BAD'} ood_train: {len(man)} clips, {len(miss)} npz missing,"
      f" gt_coc column {'present' if 'gt_coc' in man.columns else 'MISSING'}")
bad += len(man) != 1271 or bool(miss) or "gt_coc" not in man.columns
sys.exit(1 if bad else 0)
PY
echo "G0 PASS"
[ -n "${G0_ONLY:-}" ] && exit 0
mkdir -p "$Q/claimed" "$Q/running"
touch "$Q/tasks.txt"
add() { grep -qxF "$*" "$Q/tasks.txt" || { echo "$*" >> "$Q/tasks.txt"; echo "  + $*"; }; }
ARMS="moop moop_em93p75 fm_only ce_only wanda lp_r50 tyr_u40_r"
for s in 0 1 2 3; do add ol dense oodtrain "$s" 4; done
for a in $ARMS; do for s in 0 1 2 3; do add ol "$a" oodtrain "$s" 4; done; done
for a in dense $ARMS; do for s in 0 1; do add nll "$a" oodtrain "$s" 2; done; done
echo "queue: $(grep -cvE '^(#|$)' "$Q/tasks.txt") lines"
for i in $(seq 1 "$HOLDERS"); do
  sbatch -p 5thEx --exclusive --gres=gpu:4 --time=03:00:00 --job-name=oodtrain_gh200 \
    --export=ALL,OUT_SITE=gh200,EXP_ROOT=iclr27-rebuttal,A100_QUEUE=$Q,IDLE_MAX=900 cvpr27/code/eval_a100.sbatch
done
sleep 5
squeue -u e1997a06 -o '%.10i %.16j %.9T %.10M %.20S %R'
EOS
)
ssh -o BatchMode=yes kisti6 "G0_ONLY=${G0_ONLY:-} HOLDERS=${HOLDERS:-4} bash -s" <<< "$REMOTE" 2>&1 | grep -vE "$F"
