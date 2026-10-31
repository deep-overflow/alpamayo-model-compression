#!/bin/bash
# Blackwell half of both LLM-Pruner arms on s600, started no earlier than 2026-11-01 03:00 KST (user: GPUs 0-3 are in
# use until then; an empty reading before that is a pause in their job, not a free card). Replaces the pair script's
# Blackwell waiters (killed) and finish_lp_r50_s600.sh (stopped) -- cvpr27/plans/2026-10-31_lp-closedloop-s600.md §7.
#   card 0  p24b_  lp_m24 baseline-Blackwell shard 0 (150)      card 2  r50d_  lp_r50 baseline-Blackwell shard 0 (150)
#   card 1  p24c_  lp_m24 baseline-Blackwell shard 1 (150)      card 3  r50e_  lp_r50 baseline-Blackwell shard 1 (150)
# From 03:00 each card starts once it reads empty, at least 90 s after the previous start (two wizards starting in
# the same second picked the same host port on 10-31). When an arm's shards are all in (its Ada launcher too), merge
# 600 and run the paired analyses; then lp_r50 vs lp_m24.
#
#   bash cvpr27/code/finish_lp_pair_s600.sh > outputs/cvpr27/lp_s600_finish.log 2>&1
set -u
REPO=/home/cvlab21/project/chan/alpamayo-model-compression
RUNS=/home/cvlab21/project/chan/alpasim-runs
PY=/mnt/nvme1n1/ad_vla/venvs/alpasim-chan/bin/python
S=$REPO/outputs/cvpr27/lp_s600_scenes
O=$REPO/outputs/cvpr27
L0=l600_merged_slim_dual_u40_l0keep_v2
START=$(date -d '2026-11-01 03:00 KST' +%s)
stamp() { echo "$(date -u '+%F %T') UTC $*"; }
empty() { [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1")" -le 10 ]; }
run() {   # <prefix> <driver folder> <scene csv> <gpu> <out dir>
  SUITE=public_2601_${1%_} PARENT_SUITE=public_2601 PREFIX=$1 DRIVER_OMP_THREADS=8 SCENES_CSV=$3 \
    bash experiments/head_analysis/launch_alpasim_shards.sh "$2" 150 1 "$4" > "$5/${1%_}.log" 2>&1
  for c in $(docker ps -a --format '{{.Names}}' | grep "^$1"); do docker rm -f "$c" > /dev/null; done
  for n in $(docker network ls --format '{{.Name}}' | grep "^$1"); do
    [ "$(docker network inspect -f '{{len .Containers}}' "$n")" = 0 ] && docker network rm "$n" > /dev/null
  done
  stamp "$1 finished"
}
cd "$REPO"

now=$(date +%s)
[ "$now" -lt "$START" ] && { stamp "sleeping until 2026-11-01 03:00 KST"; sleep $((START - now)); }
stamp "start window open"

# card -> prefix, driver folder, scene csv, out dir
declare -A PRE=([0]=p24b_ [1]=p24c_ [2]=r50d_ [3]=r50e_)
declare -A DRV=([0]=slim_lp_m24 [1]=slim_lp_m24 [2]=lp_r50 [3]=lp_r50)
declare -A CSV=([0]=$S/main_bw300_sh0.csv [1]=$S/main_bw300_sh1.csv [2]=$S/main_bw300_sh0.csv [3]=$S/main_bw300_sh1.csv)
declare -A DIR=([0]=$O/alpasim_lp_m24_s600 [1]=$O/alpasim_lp_m24_s600 [2]=$O/alpasim_lp_r50_s600 [3]=$O/alpasim_lp_r50_s600)
todo="0 1 2 3"
while [ -n "$todo" ]; do
  for g in $todo; do
    if empty "$g"; then
      stamp "${PRE[$g]}: ${DRV[$g]} on GPU $g"
      run "${PRE[$g]}" "${DRV[$g]}" "${CSV[$g]}" "$g" "${DIR[$g]}" &
      todo=$(echo $todo | tr ' ' '\n' | grep -vx "$g" | tr '\n' ' ' | sed 's/ $//')
      sleep 90
      continue 2
    fi
  done
  sleep 60
done
stamp "all four Blackwell shards launched"
wait

until ! pgrep -f "launch_alpasim_shards.sh (slim_lp_m24|lp_r50) (296|148)" > /dev/null; do sleep 120; done
sleep 300   # let the pair script's (failing) merges finish before replacing their output

assemble() {   # <tag> <driver folder> <label> <shard dirs...>
  local tag=$1 cfg=$2 label=$3 out=$O/alpasim_$3_s600; shift 3
  rm -rf "${RUNS:?}/${tag}_merged_$cfg"
  "$REPO/.venv/bin/python" experiments/head_analysis/merge_alpasim_shards.py --runs-root "$RUNS" --shards "$@" \
    --out ${tag}_merged_$cfg --expect-scenes 600 > "$out/merge.log" 2>&1 || { stamp "$label: merge failed"; return 1; }
  local A="--arm ${tag}_merged_$cfg --arm-label $label --no-pre-fix"
  (cd experiments/evaluation
   $PY analyze_dual_s600.py $A --base d600_merged_slim_dual_u40_v2 --base-label MoOP --out "$out/vs_moop"
   $PY analyze_dual_s600.py $A --base $L0 --base-label MoOP_l0keep --out "$out/vs_moop_l0keep"
   $PY analyze_dual_s600.py $A --base $L0 --base-label MoOP_l0keep --out "$out/vs_moop_l0keep_same_card" \
     --exclude "$REPO/outputs/alpasim_l0keep_s600/card_mismatch.csv"
   $PY analyze_dual_s600.py $A --out "$out/vs_dense")
  touch "$out/analysis.DONE"
  stamp "$label: merged 600 and analysed"
}
assemble p24 slim_lp_m24 lp_m24 p24smoke_merged_slim_lp_m24 p24a_slim_lp_m24_sh0 p24a_slim_lp_m24_sh1 \
  p24b_slim_lp_m24_sh0 p24c_slim_lp_m24_sh0
assemble r50 lp_r50 lp_r50 r50smoke_merged_lp_r50 r50a_lp_r50_sh0 r50c_lp_r50_sh0 r50d_lp_r50_sh0 r50e_lp_r50_sh0
if [ -e "$O/alpasim_lp_m24_s600/analysis.DONE" ] && [ -e "$O/alpasim_lp_r50_s600/analysis.DONE" ]; then
  (cd experiments/evaluation
   $PY analyze_dual_s600.py --arm r50_merged_lp_r50 --arm-label lp_r50 --base p24_merged_slim_lp_m24 \
     --base-label lp_m24 --no-pre-fix --out "$O/alpasim_lp_r50_s600/vs_lp_m24")
fi
stamp "pair finished"
