#!/bin/bash
# plans/2026-09-15_dualexp-latency-memory.md: latency + memory of the expert-MLP ladder against
# the unpruned model, release path (profile_stages, 2 shards x 12 clips) and graphed fast path
# (bench_fastpipeline, 12 clips), on the Ada cards only.
#
#   bash experiments/head_analysis/launch_profile_arms.sh [cards]     # default "4 5 6 7"
#
# 15 jobs (5 arms x {profile s0, profile s13, fastpipe}) are pulled from a flock queue by K
# workers (K=4); every job waits for a COMPLETELY empty card (< 100 MiB), claims it under a
# lock and releases it afterwards, so nothing lands on a card another member is using and
# two workers cannot take the same card in the same polling minute. The profile shard 0 of
# each arm also counts FLOPs (hardware-independent, once per arm). The fast-path bench needs
# PYTORCH_CUDA_ALLOC_CONF empty for graph capture; run_retry_host.sh keeps an empty value.
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export ALPAMAYO_REPO=$REPO
export CUDA_DEVICE_ORDER=PCI_BUS_ID
cd "$REPO" || exit 1
CARDS=${1-"4 5 6 7"}
K=${K-4}
LOGDIR=${LOGDIR-logs}
mkdir -p "$LOGDIR"
CLAIMS=$LOGDIR/profile_claims
mkdir -p "$CLAIMS"
TAG=${TAG-ada2}

# arm  slim-ckpt ("-" = unpruned HF model)
ARMS="base -
dual outputs/slim_dual_u40_v2
em75 outputs/slim_dualexp_u40_em75
em87p5 outputs/slim_dualexp_u40_em87p5
em93p75 outputs/slim_dualexp_u40_em93p75"

empty_card() {
  for g in $1; do
    if [ -f "$CLAIMS/$g" ] && kill -0 "$(cat "$CLAIMS/$g")" 2>/dev/null; then continue; fi
    used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "${used:-99999}" -lt 100 ] && { echo "$g"; return 0; }
  done
  return 1
}

wait_empty() {
  local g
  while :; do
    g=$(
      exec 9>"$CLAIMS/.lock"
      flock -x 9
      g=$(empty_card "$1") || exit 1
      echo $$ >"$CLAIMS/$g"
      echo "$g"
    )
    [ -n "$g" ] && { echo "$g"; return 0; }
    sleep 60
  done
}

release_card() { rm -f "$CLAIMS/$1"; }

Q=$LOGDIR/profile_queue.txt
CUR=$LOGDIR/profile_cursor
: >"$Q"
echo "$ARMS" | while read -r arm ckpt; do
  echo "profile $arm $ckpt 0" >>"$Q"
  echo "profile $arm $ckpt 13" >>"$Q"
  echo "fastpipe $arm $ckpt 0" >>"$Q"
done
echo 0 >"$CUR"
echo "$(date '+%H:%M:%S') queued $(wc -l <"$Q") jobs, $K workers over cards $CARDS"

worker() {
  local w=$1 idx kind arm ckpt off gpu exp done_marker slim=()
  while :; do
    idx=$(flock "$CUR" bash -c 'i=$(cat '"$CUR"'); n=$(wc -l < '"$Q"');
          [ "$i" -lt "$n" ] && echo $((i + 1)) > '"$CUR"'; echo $i')
    [ "$idx" -ge "$(wc -l <"$Q")" ] && break
    read -r kind arm ckpt off < <(sed -n "$((idx + 1))p" "$Q")
    slim=()
    [ "$ckpt" != "-" ] && slim=(--slim-ckpt "$ckpt")
    if [ "$kind" = profile ]; then
      exp=profile_${arm}_${TAG}_s$off
      done_marker=outputs/$exp/summary.txt
    else
      exp=fastpipe_${arm}_${TAG}
      done_marker=outputs/$exp/summary.txt
    fi
    if [ -f "$done_marker" ]; then echo "skip $exp (exists)"; continue; fi
    gpu=$(wait_empty "$CARDS")
    echo "$(date '+%H:%M:%S') worker$w gpu$gpu -> $exp"
    if [ "$kind" = profile ]; then
      flops=(--no-flops)
      [ "$off" = 0 ] && flops=()
      bash experiments/head_analysis/run_retry_host.sh "${RETRIES-480}" \
        experiments/head_analysis/profile_stages.py --gpu "$gpu" --exp-id "$exp" \
        --clip-offset "$off" --num-clips 12 --warmup 1 "${flops[@]}" "${slim[@]}" \
        >>"$LOGDIR/$exp.log" 2>&1
    else
      PYTORCH_CUDA_ALLOC_CONF= bash experiments/head_analysis/run_retry_host.sh "${RETRIES-480}" \
        experiments/head_analysis/bench_fastpipeline.py --gpu "$gpu" --exp-id "$exp" \
        --num-clips 12 --reserve-gb 8 "${slim[@]}" >>"$LOGDIR/$exp.log" 2>&1
    fi
    echo "$(date '+%H:%M:%S') worker$w $exp exit=$?"
    release_card "$gpu"
  done
  echo "$(date '+%H:%M:%S') worker$w done"
}

for w in $(seq 1 "$K"); do worker "$w" & sleep 5; done
wait
echo "$(date '+%H:%M:%S') all jobs done"
