#!/bin/bash
# Stage 2 of plans/2026-09-11_action-stratified-calib.md: nine dual_u40_v2 arms that differ
# only in the calibration draw rule (rd / se / su x seeds a b c).
#
#   importance [cards]  Taylor importance on each set's 100 clips (~9 min each, 40.5 GB).
#                       Sets already measured are skipped.
#   slim [cards]        make_slim --no-state for every set whose importance exists.
#   eval                queue test500 for every built arm (2 shards each) through
#                       launch_arms.sh; then start `launch_arms.sh worker <gpu>` per free card.
#
# `cards` defaults to the Ada cards "4 5 6 7". Before every run the script waits for one of
# them to be EMPTY (< 100 MiB used, polled every 60 s) and pins the run to that card: this
# box is shared, and "has room" is not "free" -- a card another member is ramping up on
# would otherwise be taken by reserve_gpu().
#
# Every arm keeps the shipped recipe: dual_u40_v2, jlens_v2, expert and KV untouched. Only
# --importance changes. Runs from this checkout (ALPAMAYO_REPO), so a worktree evaluates
# with its own code.
#
# On cvlab20 set DIRECT_ENV=<path to cvlab20-server/env.sh> and LOGDIR=/mnt/dataset1/chan/logs:
# run_retry_host.sh must not be used there (it forces HF_HOME onto the 97%-full home
# partition and reads a shared stored_tokens), so the script is sourced and .venv/bin/python
# is called directly with the same 60 s retry loop. Results are bitwise identical across the
# two boxes (Ada, verified 2026-09-08), so arms can be merged with cvlab21's baseline.
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export ALPAMAYO_REPO=$REPO
export CUDA_DEVICE_ORDER=PCI_BUS_ID
cd "$REPO" || exit 1
LOGDIR=${LOGDIR-logs}
mkdir -p "$LOGDIR"
[ -n "${DIRECT_ENV-}" ] && . "$DIRECT_ENV"

run_py() {
  # <log> <script.py> [args...]: retry every 60 s while the card is contended
  local log=$1
  shift
  if [ -n "${DIRECT_ENV-}" ]; then
    local i
    for i in $(seq 1 "${RETRIES-480}"); do
      echo "$(date '+%H:%M:%S') attempt $i" >>"$log"
      "$REPO/.venv/bin/python" "$@" >>"$log" 2>&1 && return 0
      sleep 60
    done
    return 1
  fi
  bash experiments/head_analysis/run_retry_host.sh "${RETRIES-480}" "$@" >>"$log" 2>&1
}

empty_card() {
  # first card in $1 with < 100 MiB in use; nvidia-smi -i and --gpu agree under PCI order
  for g in $1; do
    used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "${used:-99999}" -lt 100 ] && { echo "$g"; return 0; }
  done
  return 1
}

wait_empty() {
  local g
  until g=$(empty_card "$1"); do sleep 60; done
  echo "$g"
}

# tag  manifest  cache
SETS="rd_a calib_rd100_a calib_rd_a
rd_b calib_rd100_b calib_rd_b
rd_c calib_rd100_c calib_strat
se_a calib_se100_a calib_strat
se_b calib_se100_b calib_strat
se_c calib_se100_c calib_strat
su_a calib_su100_a calib_strat
su_b calib_su100_b calib_strat
su_c calib_su100_c calib_strat"

case ${1-} in
importance)
  cards=${2-"4 5 6 7"}
  echo "$SETS" | while read -r tag man cache; do
    imp=importance_${tag/_/100_}
    if [ -f "outputs/$imp/importance.npz" ]; then
      echo "skip $imp (exists)"; continue
    fi
    [ -f "outputs/eval_sets/$man.parquet" ] || { echo "no manifest $man"; continue; }
    gpu=$(wait_empty "$cards")
    echo "$(date '+%H:%M:%S') importance $tag on gpu $gpu"
    bash experiments/head_analysis/run_retry_host.sh "${RETRIES-480}" \
      experiments/head_analysis/run_importance.py \
      --calib-manifest "$man" --cache "$cache" --num-clips 100 \
      --exp-id "$imp" --gpu "$gpu" >>"logs/$imp.log" 2>&1
    echo "$(date '+%H:%M:%S') $imp exit=$?"
  done
  ;;

slim)
  cards=${2-"4 5 6 7"}
  echo "$SETS" | while read -r tag man cache; do
    imp=importance_${tag/_/100_}
    out=outputs/slim_dual_$tag
    if [ -f "$out/slim_meta.json" ]; then
      echo "skip $out (exists)"; continue
    fi
    [ -f "outputs/$imp/importance.npz" ] || { echo "no importance for $tag"; continue; }
    gpu=$(wait_empty "$cards")
    echo "$(date '+%H:%M:%S') slim $tag on gpu $gpu"
    bash experiments/head_analysis/run_retry_host.sh "${RETRIES-480}" \
      experiments/head_analysis/make_slim.py --config dual_u40_v2 \
      --importance "$imp" --jlens jlens_v2 --out "$out" --no-state --gpu "$gpu" \
      >>"logs/slim_dual_$tag.log" 2>&1
    echo "$(date '+%H:%M:%S') slim_dual_$tag exit=$?"
  done
  ;;

eval)
  specs=()
  tags=()
  while read -r tag man cache; do
    if [ -f "outputs/slim_dual_$tag/slim_meta.json" ]; then
      specs+=("dual_$tag=outputs/slim_dual_$tag")
      tags+=("$tag")
    fi
  done <<<"$SETS"
  [ ${#specs[@]} -gt 0 ] || { echo "no built arms"; exit 1; }
  if [ -z "${DIRECT_ENV-}" ]; then
    SETS=test bash experiments/evaluation/launch_arms.sh "${MODE-init}" 2 "${specs[@]}"
    echo "now: bash experiments/evaluation/launch_arms.sh worker <gpu> &   (one per free Ada card)"
    exit 0
  fi
  # cvlab20: the same flock queue as launch_arms.sh, but calling python directly. One worker
  # per card in $2; a worker takes a job only once its card is completely empty.
  cards=${2-"4 5 6 7"}
  Q=$LOGDIR/strat_eval_queue.txt
  CUR=$LOGDIR/strat_eval_cursor
  : >"$Q"
  for tag in "${tags[@]}"; do for sh in 0 1; do echo "$tag $sh" >>"$Q"; done; done
  echo 0 >"$CUR"
  echo "queued $(wc -l <"$Q") test500 shards for: ${tags[*]}"
  worker() {
    local gpu=$1 idx tag sh
    while :; do
      idx=$(flock "$CUR" bash -c 'i=$(cat '"$CUR"'); n=$(wc -l < '"$Q"');
            [ "$i" -lt "$n" ] && echo $((i + 1)) > '"$CUR"'; echo $i')
      [ "$idx" -ge "$(wc -l <"$Q")" ] && break
      read -r tag sh < <(sed -n "$((idx + 1))p" "$Q")
      wait_empty "$gpu" >/dev/null
      echo "$(date '+%H:%M:%S') gpu$gpu -> dual_$tag test shard $sh/2"
      run_py "$LOGDIR/eval_dual_${tag}_test_s$sh.log" experiments/evaluation/run_baseline.py \
        --set test --model "outputs/slim_dual_$tag" --exp-id "dual_${tag}_test" \
        --shard "$sh" --n-shards 2 --gpu "$gpu" --reserve-gb 26
    done
    echo "$(date '+%H:%M:%S') gpu$gpu done"
  }
  for g in $cards; do worker "$g" & sleep 5; done
  wait
  echo "$(date '+%H:%M:%S') eval done"
  ;;

*)
  echo "usage: $0 {importance [cards]|slim [cards]|eval}" >&2
  exit 1
  ;;
esac
