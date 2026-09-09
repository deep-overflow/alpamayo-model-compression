#!/bin/bash
# Re-run three arms with --save-pred so the open-loop collision proxy can score them.
#
# No finished run can be used: run_baseline computes the k sampled paths and keeps only
# their ADE/FDE. Phase 2 of plans/2026-09-08_openloop-collision-proxy.md, after G1 passed
# on the baseline (12.9 / 15.4 / 30.4% against a GT floor of 0.2 / 1.2 / 1.7%).
#
# Checkpoint state differs per arm and it is not an accident:
#   dual_u40_v2  meta only, and that is enough -- selection-only configs are rebuilt
#                bit-identically from the base weights by load_slim
#   tyr_u40_r    absent, and meta alone would NOT do: OSSCAR rewrites o_proj/down_proj, so
#                a recipe rebuild would silently evaluate selection-only. Rebuilt here from
#                the local supernet, whose reproduction of this arm's kept sets was already
#                verified 72/72 by verify_supernet_u40.py
#   tyrK         16 GB state present
set -uo pipefail
REM=/home/cvlab20/project/chan
CHAN=/mnt/dataset1/chan
REPO_DIR=${REPO_DIR:-/mnt/dataset1/chan/repo}
CARDS=${CARDS:-"0 1 2 3"}
cd "$REPO_DIR" || exit 1
. $REM/cvlab20-server/env.sh
export ALPAMAYO_REPO="$REPO_DIR"
mkdir -p "$CHAN/logs"
say() { echo "$(date -u '+%F %T') CHAIN | $*"; }

# ---- rebuild tyr_u40_r if it is not here ---------------------------------------------
if [ ! -s outputs/slim_tyr_u40_r/slim_state.pt ]; then
  free=$(df -BG --output=avail /mnt/dataset1 | tail -1 | tr -dc '0-9')
  [ "$free" -lt 40 ] && { say "ABORT: /mnt/dataset1 has ${free}G, need ~16G + margin"; exit 1; }
  gpu=""
  until [ -n "$gpu" ]; do
    for g in $CARDS; do
      u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null)
      [ -n "$u" ] && [ "$u" -le 16 ] && { gpu=$g; break; }
    done
    [ -n "$gpu" ] || { say "no empty card for the build, waiting"; sleep 300; }
  done
  say "rebuilding slim_tyr_u40_r on gpu$gpu"
  .venv/bin/python experiments/head_analysis/make_slim.py \
    --config tyr_u40_r --out outputs/slim_tyr_u40_r \
    --importance importance_v1 \
    --tyr-supernet tyr_supernet_u40 \
    --tyr-config tyr_search_u40/final_config.json \
    --gpu "$gpu" >>"$CHAN/logs/build_tyr_u40_r.log" 2>&1
  rc=$?
  say "build exit=$rc -- $(head -2 outputs/slim_tyr_u40_r/summary.txt 2>/dev/null | tr '\n' ' ')"
  [ "$rc" -eq 0 ] || { say "ABORT: rebuild failed"; exit 1; }

  # the rebuild must reproduce the arm the published numbers came from
  .venv/bin/python experiments/head_analysis/verify_supernet_u40.py \
    --supernet outputs/tyr_supernet_u40 \
    --levels outputs/tyr_search_u40/final_config.json \
    --reference outputs/slim_tyr_u40_r/slim_meta.json \
    >>"$CHAN/logs/verify_tyr_u40_r.log" 2>&1
  say "kept-set check exit=$? (0 = reproduces)"
fi

# ---- the three arms, one after another ------------------------------------------------
for arm in dual_u40_v2 tyr_u40_r tyrK; do
  say "evaluating $arm on cuda:$CARDS"
  ARM="$arm" CARDS="$CARDS" NSH=4 MODEL="outputs/slim_$arm" \
    bash $REPO_DIR/cvlab20-server/eval_arm_pred.sh >>"$CHAN/logs/eval_${arm}_pred.out" 2>&1
  rc=$?
  say "eval $arm exit=$rc"
  for s in test indist oodval; do
    n=$(ls "$CHAN/outputs/${arm}_pred_${s}"/*_s*of*.json 2>/dev/null | wc -l)
    say "  $arm $s: $n row files"
  done
done
say "collision arms done"
