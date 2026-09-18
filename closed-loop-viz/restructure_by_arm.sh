#!/bin/bash
# Lay the visualisations out per arm: video/<arm>/<suite>/{camera,topdown}/
#
# Under an arm folder every file has to be that arm, so the top-down here is rendered
# single-panel rather than reusing the multi-arm comparisons -- a six-panel video filed
# under `dual/` would claim to be something it is not. The multi-arm renders keep their own
# place and are not deleted:
#
#   video/dual/origin150/{camera,topdown}   dual alone, the 10 scenes closest to GT
#   video/dual/hard100/{camera,topdown}     same, on the hard100 suite
#   video/matrix150/                        6 arms, all 150 scenes  (unchanged)
#   video/matrix_hard100/                   4 arms, all 98 renderable hard100 scenes
#
# This script ran once, when the per-arm layout was introduced, and its two source
# directories (gt_closest10, gt_closest10_hard) no longer exist -- so the camera moves below
# are now no-ops and only the top-down render would do anything on a re-run. It also used to
# park the multi-arm hard100 top-downs in a holding pen; those 10 scenes are a subset of the
# 98 in matrix_hard100, which renders them at full arm coverage, so that step is gone.
#
# Camera videos are moved, not re-rendered: they are already dual-only.

set -uo pipefail

REPO=/home/cvlab21/project/chan/alpamayo-model-compression
WT=$REPO/.claude/worktrees/closed-loop-viz
ALPASIM=/home/cvlab21/project/chan/alpasim
RUNS=/home/cvlab21/project/chan/alpasim-runs
V=$REPO/closed-loop-viz/video
T=/home/cvlab21/project/chan/.claude/jobs/285e9a74/tmp
LOG=$T/restructure.log
ARM=${ARM:-dual}
WORKERS=${WORKERS:-4}

log() { echo "[$(date -u -d '+9 hours' '+%m-%d %H:%M KST')] $*" | tee -a "$LOG"; }
: > "$LOG"

# wait for the hard100 batch if it is still going -- its camera output is an input here
if [ -f "$T/gt10_hard_render.log" ] && ! grep -q "완료:" "$T/gt10_hard_render.log"; then
  log "hard100 배치 대기 중"
  for _ in $(seq 1 120); do
    grep -q "완료:" "$T/gt10_hard_render.log" && break
    sleep 30
  done
  grep -q "완료:" "$T/gt10_hard_render.log" || { log "대기 실패"; exit 1; }
fi
log "hard100 배치 완료 확인"

mkdir -p "$V/$ARM/origin150/camera" "$V/$ARM/origin150/topdown" \
         "$V/$ARM/hard100/camera" "$V/$ARM/hard100/topdown"

# ---- 1. camera: move what already exists ----------------------------------------------
moved=0
for f in "$V/gt_closest10/camera/"*.mp4; do
  [ -e "$f" ] || continue
  mv -n "$f" "$V/$ARM/origin150/camera/" && moved=$((moved + 1))
done
for f in "$V/gt_closest10_hard/camera/"*.mp4; do
  [ -e "$f" ] || continue
  mv -n "$f" "$V/$ARM/hard100/camera/" && moved=$((moved + 1))
done
log "카메라 $moved개 이동"

# ---- 2. top-down: single arm, both suites ---------------------------------------------
render_one() {
  local scene="$1" suite="$2" run="$3"
  local out="$V/$ARM/$suite/topdown/$scene.mp4"
  [ -s "$out" ] && { echo "skip  $suite $scene" >> "$LOG"; return 0; }
  cd "$ALPASIM" || return 1
  if CUDA_VISIBLE_DEVICES="" uv run python "$WT/closed-loop-viz/render_replay.py" \
       --scene "$scene" --arm "$ARM=$run" --out "$out" >>"$LOG.$scene" 2>&1; then
    echo "ok    $suite $scene  $(du -h "$out" | cut -f1)" >> "$LOG"
    rm -f "$LOG.$scene"
  else
    echo "FAIL  $suite $scene -- $LOG.$scene" >> "$LOG"
  fi
}
export -f render_one
export V ALPASIM WT ARM LOG

for spec in "origin150:$T/gt10.txt:$RUNS/m2601_merged_slim_dual_u40_v2" \
            "hard100:$T/gt10_hard.txt:$RUNS/h100_merged_slim_dual_u40_v2"; do
  suite=${spec%%:*}; rest=${spec#*:}; list=${rest%%:*}; run=${rest#*:}
  log "top-down 렌더: $suite ($ARM 단독)"
  xargs -a "$list" -P "$WORKERS" -I{} bash -c 'render_one "$1" "$2" "$3"' _ {} "$suite" "$run"
done

# ---- 3. retire the old directories once empty -----------------------------------------
for d in "$V/gt_closest10" "$V/gt_closest10_hard"; do
  find "$d" -type d -empty -delete 2>/dev/null
  [ -d "$d" ] && log "남아 있음(비지 않음): $d"
done

ok=$(grep -c '^ok' "$LOG"); bad=$(grep -c '^FAIL' "$LOG")
log "완료: top-down ok $ok / FAIL $bad"
for s in origin150 hard100; do
  log "  $ARM/$s  camera $(ls "$V/$ARM/$s/camera" 2>/dev/null | wc -l)  " \
      "topdown $(ls "$V/$ARM/$s/topdown" 2>/dev/null | wc -l)"
done
