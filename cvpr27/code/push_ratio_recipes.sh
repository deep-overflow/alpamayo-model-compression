#!/bin/bash
# Push the locally built ratio-sweep recipes to NEURON as they finish (cvpr27/plans/2026-11-03_baselines-ratio-sweep.md).
# A recipe goes only when its summary.txt exists (the builder's last write) and slim_meta.json removes exactly the MoOP
# u{N} budget. slim_meta.json is sent LAST: the A100 holders claim an arm's lines once that file is on /scratch.
set -uo pipefail
cd "$(dirname "$0")/../.."
DST=neuron-dm:/scratch/e1997a06/outputs/cvpr27
declare -A BUDGET=([35]=2317860864 [45]=2974777344 [50]=3321888768)
DIRS=()
for n in 35 45 50; do
  DIRS+=("slim_wandasp_u$n" "slim_lp_u$n" "slim_trajlastcoc_u${n}_l0keep_v2" "slim_coc_u${n}_l0keep_v2")
done
DIRS+=(slim_tyr_u50)
stamp() { echo "$(date -d '+9 hours' '+%F %T') KST $*"; }
declare -A DONE=()
for _ in $(seq 1 600); do
  for d in "${DIRS[@]}"; do
    [ -n "${DONE[$d]:-}" ] && continue
    src=outputs/cvpr27/$d
    [ -f "$src/summary.txt" ] && [ -f "$src/slim_meta.json" ] || continue
    n=$(grep -oE '_u(35|45|50)' <<< "$d" | head -1 | tr -dc 0-9)
    removed=$(.venv/bin/python -c "import json,sys; print(json.load(open(sys.argv[1]))['params']['removed'])" "$src/slim_meta.json")
    if [ "$removed" != "${BUDGET[$n]}" ]; then
      stamp "HOLD $d: removed $removed != budget ${BUDGET[$n]}"
      DONE[$d]=held
      continue
    fi
    if rsync -a --exclude slim_meta.json "$src/" "$DST/$d/" && rsync -a "$src/slim_meta.json" "$DST/$d/slim_meta.json"; then
      stamp "pushed $d (removed $removed)"
      DONE[$d]=pushed
    else
      stamp "rsync failed for $d, retry next round"
    fi
  done
  [ "${#DONE[@]}" -eq "${#DIRS[@]}" ] && { stamp "all ${#DIRS[@]} recipes handled"; exit 0; }
  sleep 120
done
stamp "timeout; handled ${#DONE[@]}/${#DIRS[@]}"
