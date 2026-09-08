#!/bin/bash
# H1 end to end on cvlab20: wait out the supernet rebuild, check it reproduces, run the
# KL-only search, build the arm, evaluate it on the three sets.
# plans/2026-09-08_tyr-objective-2x2.md
#
# The reproducibility check is a GATE, not a formality. If the rebuilt supernet does not
# give tyr_u40_r's kept sets back, then tyr_u40_r's published numbers were produced by a
# different supernet and are not a fair reference for tyrK -- so the chain also rebuilds
# and evaluates the reference from the new supernet. It never silently compares across
# two supernets.
#
# Each stage refuses rather than degrades: a stage that fails stops the chain instead of
# handing a broken artefact to the next one.
set -uo pipefail
REM=/home/cvlab20/project/chan
CHAN=/mnt/dataset1/chan
CARDS=${CARDS:-"0 1 2 3"}
SEARCH_GPU=${SEARCH_GPU:-1,2,3}
BUILD_GPU=${BUILD_GPU:-0}
cd $REM/alpamayo-model-compression || exit 1
. $REM/cvlab20-server/env.sh
LOG=$CHAN/logs/chain_tyrk.out
say() { echo "$(date -u '+%F %T') CHAIN | $*"; }

# ---- 0. wait for the supernet build that is already running --------------------------
while pgrep -f "[b]uild_supernet_u40.sh" >/dev/null; do sleep 120; done
n=$(find "$CHAN/outputs/tyr_supernet_u40" -name '*.pth' 2>/dev/null | wc -l)
say "supernet done: $n level files (expect 648)"
[ "$n" -eq 648 ] || { say "ABORT: supernet incomplete"; exit 1; }

# ---- 1. gate: does it reproduce the shipped tyr_u40_r? -------------------------------
say "verifying reproducibility (CPU, no GPU needed)"
.venv/bin/python experiments/head_analysis/verify_supernet_u40.py \
  --supernet outputs/tyr_supernet_u40 \
  --levels outputs/tyr_search_u40/final_config.json \
  --reference outputs/slim_tyr_u40_r_ref/slim_meta.json \
  --out outputs/tyrK_verify/verdict.json \
  >>"$CHAN/logs/verify_supernet_u40.log" 2>&1
REPRO=$?
if [ "$REPRO" -eq 0 ]; then
  say "GATE PASS: rebuild reproduces tyr_u40_r -- its published numbers stand as reference"
else
  say "GATE FAIL: rebuild differs -- the reference will be rebuilt from this supernet too"
fi

# ---- 2. the KL-only search -----------------------------------------------------------
say "tyrK search on cuda:$SEARCH_GPU (~4 h)"
.venv/bin/python experiments/head_analysis/run_tyr_search.py \
  --supernet tyr_supernet_u40 --teacher-id tyr_teacher_u40 \
  --exp-id tyrK_search_u40 \
  --num-clips 100 --calib-manifest calib_100 --cache calib \
  --topk 1024 --noise-draws 2 --max-gen 256 \
  --generations 20 --offspring 32 --seed 42 \
  --workers 3 --reserve-gb 40.0 --gpu "$SEARCH_GPU" \
  --fitness kl \
  >>"$CHAN/logs/tyrK_search.log" 2>&1
rc=$?
say "search exit=$rc"
[ "$rc" -eq 0 ] || { say "ABORT: search failed"; exit 1; }
[ -s outputs/tyrK_search_u40/final_config.json ] || { say "ABORT: no final_config.json"; exit 1; }
say "search: $(head -1 outputs/tyrK_search_u40/summary.txt)"

# ---- 3. build ------------------------------------------------------------------------
build() {  # $1 = arm name, $2 = search dir
  say "building slim_$1 on gpu$BUILD_GPU"
  .venv/bin/python experiments/head_analysis/make_slim.py \
    --config tyr_u40_r --out "outputs/slim_$1" \
    --importance importance_v1 \
    --tyr-supernet tyr_supernet_u40 \
    --tyr-config "$2/final_config.json" \
    --gpu "$BUILD_GPU" >>"$CHAN/logs/build_$1.log" 2>&1
  local r=$?
  say "build $1 exit=$r -- $(cat "outputs/slim_$1/summary.txt" 2>/dev/null | head -2 | tr '\n' ' ')"
  return $r
}
build tyrK tyrK_search_u40 || { say "ABORT: tyrK build failed"; exit 1; }
ARMS="tyrK"
if [ "$REPRO" -ne 0 ]; then
  build tyrRef tyr_search_u40 && ARMS="tyrK tyrRef" || say "WARN: reference rebuild failed"
fi

# ---- 4. evaluate ---------------------------------------------------------------------
for arm in $ARMS; do
  say "evaluating $arm on cuda:$CARDS"
  ARM="$arm" CARDS="$CARDS" NSH=4 bash $REM/cvlab20-server/eval_arm_sharded.sh \
    >>"$CHAN/logs/eval_$arm.out" 2>&1
  say "eval $arm exit=$?"
  for s in test indist oodval; do
    say "  $arm $s: $(cat "$CHAN/outputs/${arm}_${s}"/summary_s0of4.txt 2>/dev/null | head -1)"
  done
done
say "H1 chain done (arms: $ARMS)"
