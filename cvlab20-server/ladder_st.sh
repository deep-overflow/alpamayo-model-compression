#!/bin/bash
# Nested calibration ladder inside one draw (plans/2026-09-07_calib-nested-ladder.md).
#
# calib_st4000's first 500 / 1000 / 2000 are prefixes of one greedy order, so the draw is
# held fixed and only n moves. st2000 already exists; this adds the two lower rungs and
# evaluates them on test500.
#
# Derivation is GPU-free: mean(per-clip) == importance.npz holds exactly, so a prefix's
# importance is the mean of its rows. make_block_importance accumulates in fp64 for it.
#
# The box is shared and cards 0-3 and 6 were held by another member when this was written,
# so every step takes only cards with nothing resident at all.
set -uo pipefail
REM=/home/cvlab20/project/chan
CHAN=/mnt/dataset1/chan
cd $REM/alpamayo-model-compression
. $REM/cvlab20-server/env.sh
mkdir -p "$CHAN/logs"

idle_cards() {   # <how many> -> prints that many completely-idle card ids
  local want=$1 got=""
  for g in 4 5 7 2 3 0 1 6; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
    [ "$used" -le 16 ] && got="$got $g"
    [ "$(echo $got | wc -w)" -ge "$want" ] && break
  done
  echo $got
}

echo "=== 1/3  중요도 파생 (GPU 없음)"
if [ ! -f "$CHAN/outputs/importance_st4000_c1000/importance.npz" ]; then
  .venv/bin/python experiments/evaluation/make_block_importance.py \
    --importance importance_st4000 --manifest calib_st4000 --cumulative 500 1000 \
    2>&1 | tee "$CHAN/logs/ladder_derive.log" | tail -5
else
  echo "  이미 있음, 건너뜀"
fi

echo "=== 2/3  빌드"
BGPU=$(idle_cards 1)
[ -z "$BGPU" ] && { echo "REFUSING: 완전히 비어 있는 카드 없음"; exit 1; }
echo "  building on cuda:$BGPU"
for k in 500 1000; do
  out=outputs/slim_dual_u40_st${k}
  if [ -f "$out/slim_meta.json" ]; then echo "  st$k: 이미 있음"; continue; fi
  .venv/bin/python experiments/head_analysis/make_slim.py \
    --config dual_u40_v2 --importance "importance_st4000_c${k}" \
    --out "$out" --no-state --gpu "$BGPU" \
    >>"$CHAN/logs/ladder_build_st${k}.log" 2>&1 \
    || { echo "  BUILD FAILED: st$k"; exit 1; }
  echo "  st$k 빌드 완료"
done

echo "=== 3/3  test500 평가"
cards=$(idle_cards 2)
[ "$(echo $cards | wc -w)" -lt 2 ] && { echo "REFUSING: 비어 있는 카드 2장 미만 ($cards)"; exit 1; }
set -- $cards
echo "  eval on cuda:$cards"
run() {
  local k=$1 gpu=$2
  .venv/bin/python experiments/evaluation/run_baseline.py \
    --set test --model "outputs/slim_dual_u40_st${k}" --exp-id "dual_u40_st${k}_test" \
    --gpu "$gpu" --reserve-gb 26 >"$CHAN/logs/ladder_eval_st${k}.log" 2>&1
  echo "  $(date -u '+%H:%M') st$k done on gpu$gpu"
}
run 500 "$1" &
sleep 5
run 1000 "$2" &
wait
echo "$(date -u '+%F %T') ladder done"
