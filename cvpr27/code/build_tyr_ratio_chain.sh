#!/bin/bash
# Tyr official procedure at MoOP u35 / u45 budgets, built locally after tyr_u50 (user, 2026-11-03: NEURON builds
# cancelled, "Tyr u35, u45는 로컬에서 빌드해줘"). cvpr27/plans/2026-11-03_baselines-ratio-sweep.md.
#
# For each finished build (u50 by the running build, then u35, u45 here): the recipe goes to NEURON (slim_meta.json
# last, so the A100 holders only claim a complete recipe), then supernet_r1..r3 are deleted -- regenerable
# intermediates, ~110 GB each, and the mount cannot hold three full builds (373 GB free at start). supernet_r4 and
# the search configs stay, so make_slim can be rerun.
set -uo pipefail
cd "$(dirname "$0")/../.."
L=outputs/cvpr27/ratio_sweep/logs
F='Deprecation|Redirecting'
stamp() { echo "$(date -d '+9 hours' '+%F %T') KST $*"; }
declare -A SQ=([35]=0.328125 [45]=0.421875) SM=([35]=0.3499755859375 [45]=0.4499918619791667)
declare -A BUDGET=([35]=2317860864 [45]=2974777344 [50]=3321888768)

finish() {  # <N>: push the recipe and drop rounds 1-3 once the build wrote its checkpoint
  local n=$1 slim=outputs/cvpr27/slim_tyr_u$1 o=outputs/cvpr27/tyr_u$1 removed
  [ -f "$slim/summary.txt" ] || { stamp "tyr_u$n: no summary.txt -- build failed? not pushing"; return 1; }
  # a zero-output placeholder head (make_slim attn_placeholder_layers) is kept physically: 1,048,576 params each
  removed=$(.venv/bin/python -c "import json,sys; m=json.load(open(sys.argv[1])); \
print(m['params']['removed'] + 1048576 * len(m.get('attn_placeholder_layers', [])))" "$slim/slim_meta.json")
  [ "$removed" = "${BUDGET[$n]}" ] || { stamp "tyr_u$n: removed (+placeholders) $removed != ${BUDGET[$n]}, not pushing"; return 1; }
  if rsync -a --exclude slim_meta.json "$slim/" "neuron-dm:/scratch/e1997a06/outputs/cvpr27/slim_tyr_u$n/" 2>&1 | grep -vE "$F"; [ "${PIPESTATUS[0]}" -eq 0 ] \
     && rsync -a "$slim/slim_meta.json" "neuron-dm:/scratch/e1997a06/outputs/cvpr27/slim_tyr_u$n/slim_meta.json"; then
    stamp "tyr_u$n: pushed (removed $removed)"
  else
    stamp "tyr_u$n: push FAILED"; return 1
  fi
  rm -rf "$o/supernet_r1" "$o/supernet_r2" "$o/supernet_r3"
  stamp "tyr_u$n: deleted supernet_r1-r3; $(df -h /mnt/nvme1n1 | awk 'NR==2 {print $4 " free"}')"
}

# 1) wait for the running u50 build
until grep -q "all done" "$L/tyr_u50_local.log" 2>/dev/null; do
  sleep 120
done
stamp "tyr_u50 build finished"
finish 50 || exit 1

# 2) u35 then u45 on the same Blackwell cards
for n in 35 45; do
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1,2,3 | awk '$1 >= 100' | wc -l)" -eq 0 ]; do
    stamp "waiting for Blackwell 0-3 to be empty"; sleep 300
  done
  stamp "tyr_u$n build start (cards 0,1,2,3)"
  CARDS=0,1,2,3 TYR_OUT=cvpr27/tyr_u$n TYR_SLIM=outputs/cvpr27/slim_tyr_u$n TYR_TEACHER=cvpr27/tyr_off/teacher \
    TYR_SUPERNET_ARGS="--sparsity-mha ${SQ[$n]} --sparsity-mlp ${SM[$n]}" TYR_BUDGET=${BUDGET[$n]} \
    bash cvpr27/code/tyr_off/build_tyr_off_local.sh > "$L/tyr_u${n}_local.log" 2>&1
  stamp "tyr_u$n build exit $?: $(tail -n 1 "$L/tyr_u${n}_local.log")"
  finish "$n" || exit 1
done
stamp "chain done"
