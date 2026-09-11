#!/bin/bash
# Per-step VLM trajectory importance over the 4,000 st4000 clips -- the half `max11`
# needs and the only material missing for maxstep11 @ st4000.
#
#   bash cvlab20-server/run_stepvlm_st4000.sh "0 2"     # card ids, default "0 2"
#
# ~5.9 s/clip on top of the ordinary pass (measured, see the script's docstring), so
# 2,000 clips is ~8.5 GPU-h: about 4.25 h on two cards.
#
# Why 2,000 and not the full 4,000: today's stability curve puts Q-head disagreement
# between two disjoint 2,000-clip draws at 2.92%, which is the churn G0b showed costs
# nothing measurable (+0.0012, p=0.68). Going to 4,000 only reaches 2.22% and doubles the
# time; MLP stays above the floor either way (4.32% vs 3.25%, converging near n=6,000).
#
# The first 2,000 of calib_st4000 are used as-is. make_eval_sets' greedy is prefix-nested,
# and it holds here by measurement: weighted L1 0.0015 for the prefix against 0.0281 for a
# random 2,000 from the same pool, and 0.0306 for calib_100.
#
# --no-perclip is deliberate. That file holds (10, 36, 12288) fp32 PER CLIP -- 16.9 MB
# each, so 4,000 clips would be 70 GB on disk and the same again in RAM at every
# ten-clip checkpoint. The shipped accumulated file is 42.6 MB whatever the clip count.
# The cost is that no subset of this run can be reconstructed later; unlike
# run_importance, a smaller n means measuring again.
set -uo pipefail
GPUS=${1:-"0 2"}
N=2000
CHAN=/mnt/dataset1/chan
REPO=/home/cvlab20/project/chan/alpamayo-model-compression
. /home/cvlab20/project/chan/cvlab20-server/env.sh
cd "$REPO" || exit 1
PY=$REPO/.venv/bin/python
mkdir -p "$CHAN/logs"

K=${K:-2}
NEED_MIB=46000                      # --reserve-gb 44 plus headroom
WAIT_MAX=${WAIT_MAX:-720}           # 720 x 60 s = 12 h before giving up

# The box is shared and cards come and go, so this WAITS for K genuinely idle cards
# rather than refusing outright -- and re-reads them every minute, because a card that
# is free now may be taken by the time the run starts. Whichever K are free get used;
# $GPUS is only a preference for the order they are considered in.
free_cards() {
  local out=""
  for g in $GPUS 0 1 2 3 4 5 6 7; do
    case " $out " in *" $g "*) continue ;; esac
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$g")
    [ $((total - used)) -ge $NEED_MIB ] && out="$out $g"
    [ "$(echo $out | wc -w)" -ge "$K" ] && break
  done
  echo $out
}

echo "$(date -u '+%F %T') waiting for $K cards with ${NEED_MIB} MiB free"
for try in $(seq 1 $WAIT_MAX); do
  cards=$(free_cards)
  [ "$(echo $cards | wc -w)" -ge "$K" ] && break
  [ $((try % 15)) -eq 1 ] && echo "  $(date -u '+%H:%M') free: [${cards:- none}] -- waiting"
  sleep 60
done
read -ra CARDS <<<"$cards"
[ "${#CARDS[@]}" -lt "$K" ] && { echo "GAVE UP: never saw $K free cards"; exit 1; }
echo "$(date -u '+%F %T') using cuda: ${CARDS[*]}"
K=${#CARDS[@]}

for i in "${!CARDS[@]}"; do
  g=${CARDS[$i]}
  nohup "$PY" experiments/head_analysis/run_step_importance_vlm.py \
    --calib-manifest calib_st4000 --cache calib_st --num-clips $N \
    --exp-id "importance_stepvlm_st2000_s$i" --shard "$i" --n-shards "$K" \
    --no-perclip --gpu "$g" \
    >>"$CHAN/logs/stepvlm_st4000_s$i.log" 2>&1 &
  echo "  shard $i -> cuda:$g (pid $!)"
  sleep 5
done
echo "$(date -u '+%F %T') $K shards launched"
wait
echo "$(date -u '+%F %T') shards finished"

shards=""
for i in $(seq 0 $((K - 1))); do shards="$shards importance_stepvlm_st2000_s$i"; done
"$PY" experiments/head_analysis/merge_step_importance.py --shards $shards \
  --out importance_stepvlm_st2000 ||
  { echo "MERGE FAILED -- the shards are intact"; exit 1; }
echo "$(date -u '+%F %T') step importance complete"
