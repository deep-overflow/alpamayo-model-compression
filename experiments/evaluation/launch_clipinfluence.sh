#!/bin/bash
# plans/2026-09-14_calibration-clip-influence.md: M calibration-subset arms of
# dual_u40_v2, each evaluated on the same 150 val500 clips, so a per-clip marginal
# contribution can be estimated from (subset -> performance) pairs.
#
#   recipes [card]      one model load builds every subset's slim_meta.json (~5 min for
#                       32). Gate V0 must have passed first -- see make_subset_recipes.py.
#   fit [cards]         K workers pull arms from a flock-guarded queue; each waits for,
#                       claims and releases a completely EMPTY card.
#   validate [cards]    Stage B: the drop-20 arms on the full test500. Run only after
#                       Stage A has been read -- the arms are defined by its tau.
#
# `cards` defaults to "4 5 6 7". GPUs 0-3 are Blackwell and are NOT to be used: this
# repo's numbers are Ada, and the two architectures disagree on discrete metrics
# (CLAUDE.md, "Determinism"). Everything here is one architecture by construction.
#
# The fit set is `--limit 150`, the first 150 clips of val500. That is a prefix of a
# greedy distribution-matched draw, so it is itself matched -- and it is the same 150
# clips for every arm, with clip-derived seeds, which is what makes the arm-to-arm
# comparison paired. The baseline for that pairing needs no run of its own:
# `baseline_ada_ps_indist` already holds all 500 val clips in manifest order.
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export ALPAMAYO_REPO=$REPO
export CUDA_DEVICE_ORDER=PCI_BUS_ID
cd "$REPO" || exit 1
LOGDIR=${LOGDIR-logs}
mkdir -p "$LOGDIR"

PREFIX=${PREFIX-slim_subinf}
FIT_LIMIT=${FIT_LIMIT-150}
ARMS=${ARMS-32}

run_py() {
  local log=$1
  shift
  bash experiments/head_analysis/run_retry_host.sh "${RETRIES-480}" "$@" >>"$log" 2>&1
}

CLAIMS=$LOGDIR/clipinf_claims
mkdir -p "$CLAIMS"

empty_card() {
  for g in $1; do
    if [ -f "$CLAIMS/$g" ] && kill -0 "$(cat "$CLAIMS/$g")" 2>/dev/null; then continue; fi
    used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "${used:-99999}" -lt 100 ] && { echo "$g"; return 0; }
  done
  return 1
}

wait_empty() {
  # selection and claim happen under one lock, so two workers polling in the same minute
  # cannot both take the same card before either has allocated on it
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

n_records() {
  # <exp-id> <shard-file-glob>: clips this shard has written so far. run_baseline
  # CHECKPOINTS its json while running -- a 150-clip arm has a readable 40-clip file
  # eight minutes in -- so file existence is NOT completion, and a relaunch keyed on it
  # would skip a half-finished arm forever. This is the same trap `run_importance`'s
  # "n_clips == 100" marker exists for.
  # shellcheck disable=SC2086
  grep -ho '"clip_id"' outputs/$1/$2 2>/dev/null | wc -l
}

job_pool() {
  # <queue-file> <cards>: each line is
  #   "<arm-dir> <exp-id> <set> <expect> <shard-glob> <extra args...>"
  local Q=$1 cards=$2
  local CUR=$Q.cursor
  echo 0 >"$CUR"
  echo "queued $(wc -l <"$Q") jobs over cards $cards (${K-4} workers)"
  worker() {
    local w=$1 idx line dir exp which expect glob gpu rest have
    while :; do
      idx=$(flock "$CUR" bash -c 'i=$(cat '"$CUR"'); n=$(wc -l < '"$Q"');
            [ "$i" -lt "$n" ] && echo $((i + 1)) > '"$CUR"'; echo $i')
      [ "$idx" -ge "$(wc -l <"$Q")" ] && break
      line=$(sed -n "$((idx + 1))p" "$Q")
      read -r dir exp which expect glob rest <<<"$line"
      have=$(n_records "$exp" "$glob")
      if [ "$have" -ge "$expect" ]; then
        echo "$(date '+%H:%M:%S') worker$w skip $exp ($have/$expect done)"
        continue
      fi
      [ "$have" -gt 0 ] && echo "$(date '+%H:%M:%S') worker$w redo $exp ($have/$expect)"
      gpu=$(wait_empty "$cards")
      echo "$(date '+%H:%M:%S') worker$w gpu$gpu -> $exp"
      # shellcheck disable=SC2086
      run_py "$LOGDIR/$exp.log" experiments/evaluation/run_baseline.py \
        --set "$which" --model "outputs/$dir" --exp-id "$exp" \
        --gpu "$gpu" --reserve-gb 26 $rest
      echo "$(date '+%H:%M:%S') worker$w $exp exit=$?"
      release_card "$gpu"
    done
    echo "$(date '+%H:%M:%S') worker$w done"
  }
  for w in $(seq 1 "${K-4}"); do worker "$w" & sleep 5; done
  wait
}

case ${1-} in
recipes)
  card=${2-4}
  gpu=$(wait_empty "$card")
  run_py "$LOGDIR/clipinf_recipes.log" experiments/evaluation/make_subset_recipes.py \
    --build --arms 0 "$ARMS" --gpu "$gpu"
  release_card "$gpu"
  tail -3 "$LOGDIR/clipinf_recipes.log"
  ;;

fit)
  Q=$LOGDIR/clipinf_fit_queue.txt
  : >"$Q"
  for i in $(seq 0 $((ARMS - 1))); do
    a=$(printf '%02d' "$i")
    [ -f "outputs/${PREFIX}_$a/slim_meta.json" ] || { echo "no recipe for arm $a"; exit 1; }
    echo "${PREFIX}_$a subinf_${a}_fit$FIT_LIMIT indist $FIT_LIMIT *_s0of1.json" \
      "--limit $FIT_LIMIT" >>"$Q"
  done
  job_pool "$Q" "${2-"4 5 6 7"}"
  ;;

validate)
  # Stage B arms are built by make_subset_recipes.py --drop, named slim_dropinf_<tag>
  Q=$LOGDIR/clipinf_validate_queue.txt
  : >"$Q"
  for d in outputs/slim_dropinf_*/; do
    [ -f "$d/slim_meta.json" ] || continue
    tag=$(basename "$d")
    tag=${tag#slim_dropinf_}
    for sh in 0 1; do
      echo "slim_dropinf_$tag dropinf_${tag}_test test 250 *_s${sh}of2.json" \
        "--shard $sh --n-shards 2" >>"$Q"
    done
  done
  [ -s "$Q" ] || { echo "no slim_dropinf_* arms built"; exit 1; }
  job_pool "$Q" "${2-"4 5 6 7"}"
  ;;

*)
  echo "usage: $0 {recipes [card]|fit [cards]|validate [cards]}" >&2
  exit 1
  ;;
esac
