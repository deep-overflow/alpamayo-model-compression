#!/bin/bash
# plans/2026-09-19_fastpath-single-card-rounds.md: the graphed fast path of the expert-MLP
# ladder re-measured on ONE Ada card, 5 arms x ROUNDS rounds. The 2026-09-16 run put every arm
# on whichever card was free, and the four arms that share the dual VLM came back with prefill
# medians 4-5% apart -- a per-run offset five times the em87p5/em93p75 denoise gap. Pinning one
# card removes the card term; repeating the 5-arm sequence measures what is left.
#
#   bash experiments/head_analysis/launch_fastpath_rounds.sh [cards]     # default "4 5 6 7"
#
# The first COMPLETELY empty card (< 100 MiB) of [cards] is claimed for the whole sequence and
# remembered in $LOGDIR/fastpath_rounds_card, so a relaunch resumes on the same card (finished
# runs are skipped). Round r runs the arms rotated by r (cyclic Latin square: every arm takes
# every position once, so thermal/time drift is not confounded with the arm). Each run waits for
# the card to be empty again, and a 20 s sampler logs the compute processes on it; a run that
# ever shared the card is set aside as <exp>_cotenant<k> and repeated.
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export ALPAMAYO_REPO=$REPO
export CUDA_DEVICE_ORDER=PCI_BUS_ID
cd "$REPO" || exit 1
CARDS=${1-"4 5 6 7"}
ROUNDS=${ROUNDS-5}
NCLIPS=${NCLIPS-24}
TAG=${TAG-ada1c}
LOGDIR=${LOGDIR-logs}
CACHE=${CACHE-outputs/clip_cache_fastpipe}
REDO=${REDO-3}
mkdir -p "$LOGDIR"
CLAIMS=$LOGDIR/profile_claims # same claim dir as launch_profile_arms.sh, so the two never collide
mkdir -p "$CLAIMS"
CARDFILE=$LOGDIR/fastpath_rounds_card

# arm  slim-ckpt ("-" = unpruned HF model)
NAMES=(base dual em75 em87p5 em93p75)
CKPTS=(- outputs/slim_dual_u40_v2 outputs/slim_dualexp_u40_em75
  outputs/slim_dualexp_u40_em87p5 outputs/slim_dualexp_u40_em93p75)
NA=${#NAMES[@]}

card_used() { nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits; }

claim_card() {
  local g used
  while :; do
    g=$(
      exec 9>"$CLAIMS/.lock"
      flock -x 9
      for g in $1; do
        if [ -f "$CLAIMS/$g" ] && kill -0 "$(cat "$CLAIMS/$g")" 2>/dev/null; then continue; fi
        used=$(card_used "$g")
        [ "${used:-99999}" -lt 100 ] && { echo $$ >"$CLAIMS/$g"; echo "$g"; exit 0; }
      done
      exit 1
    )
    [ -n "$g" ] && { echo "$g"; return 0; }
    sleep 60
  done
}

wait_card_empty() {
  local used
  while :; do
    used=$(card_used "$1")
    [ "${used:-99999}" -lt 100 ] && return 0
    sleep 60
  done
}

sampler() { # $1 gpu, $2 file: one line per 20 s, the compute processes on the card
  local pids
  while :; do
    pids=$(nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader 2>/dev/null |
      tr -d ' ' | paste -sd, -)
    echo "$(date '+%H:%M:%S') n=$(echo -n "$pids" | tr ',' '\n' | grep -c .) pids=$pids" >>"$2"
    sleep 20
  done
}

echo "$(date '+%H:%M:%S') prefetch $((NCLIPS + 1)) clips -> $CACHE"
CUDA_VISIBLE_DEVICES= bash experiments/head_analysis/run_retry_host.sh 5 \
  experiments/head_analysis/bench_fastpipeline.py --prefetch-only --num-clips "$NCLIPS" \
  --clip-cache "$CACHE" >>"$LOGDIR/fastpath_rounds_prefetch.log" 2>&1 || exit 1

if [ -f "$CARDFILE" ]; then
  GPU=$(cat "$CARDFILE")
  echo $$ >"$CLAIMS/$GPU"
  echo "$(date '+%H:%M:%S') resuming on card $GPU"
else
  echo "$(date '+%H:%M:%S') waiting for an empty card among: $CARDS"
  GPU=$(claim_card "$CARDS")
  echo "$GPU" >"$CARDFILE"
  echo "$(date '+%H:%M:%S') claimed card $GPU for all $((ROUNDS * NA)) runs"
fi
SAMPLER_PID=
cleanup() {
  [ -n "$SAMPLER_PID" ] && kill "$SAMPLER_PID" 2>/dev/null
  rm -f "$CLAIMS/$GPU"
}
trap cleanup EXIT

for r in $(seq 0 $((ROUNDS - 1))); do
  for k in $(seq 0 $((NA - 1))); do
    i=$(((r + k) % NA))
    arm=${NAMES[$i]}
    ckpt=${CKPTS[$i]}
    exp=fastpipe_${arm}_${TAG}_r$r
    [ -f "outputs/$exp/summary.txt" ] && { echo "skip $exp (exists)"; continue; }
    slim=()
    [ "$ckpt" != "-" ] && slim=(--slim-ckpt "$ckpt")
    for attempt in $(seq 1 "$REDO"); do
      wait_card_empty "$GPU"
      echo "$(date '+%H:%M:%S') round $r pos $k gpu$GPU -> $exp (load $(cut -d' ' -f1-3 /proc/loadavg))"
      : >"$LOGDIR/$exp.cotenant"
      sampler "$GPU" "$LOGDIR/$exp.cotenant" &
      SAMPLER_PID=$!
      PYTORCH_CUDA_ALLOC_CONF= bash experiments/head_analysis/run_retry_host.sh "${RETRIES-20}" \
        experiments/head_analysis/bench_fastpipeline.py --gpu "$GPU" --exp-id "$exp" \
        --num-clips "$NCLIPS" --reserve-gb 8 --clip-cache "$CACHE" "${slim[@]}" \
        >>"$LOGDIR/$exp.log" 2>&1
      rc=$?
      kill "$SAMPLER_PID" 2>/dev/null
      SAMPLER_PID=
      shared=$(grep -cE ' n=([2-9]|[1-9][0-9]+) ' "$LOGDIR/$exp.cotenant")
      echo "$(date '+%H:%M:%S') $exp exit=$rc shared_samples=$shared (load $(cut -d' ' -f1-3 /proc/loadavg))"
      if [ "$shared" -gt 0 ] && [ -d "outputs/$exp" ]; then
        mv "outputs/$exp" "outputs/${exp}_cotenant$attempt"
        mv "$LOGDIR/$exp.cotenant" "$LOGDIR/${exp}_cotenant$attempt.cotenant"
        echo "  card was shared during the run -> set aside, repeating ($attempt/$REDO)"
        continue
      fi
      cp "$LOGDIR/$exp.cotenant" "outputs/$exp/cotenant.log" 2>/dev/null
      break
    done
  done
  echo "$(date '+%H:%M:%S') round $r done"
done
echo "$(date '+%H:%M:%S') all rounds done"
