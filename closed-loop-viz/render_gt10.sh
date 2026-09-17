#!/bin/bash
# The ten scenes where `dual` drove closest to alpasim's GT: four-camera replay + top-down.
#
# Selection comes from pick_gt_close.py, which ranks by the sim's own dist_to_gt_trajectory
# but only among scenes the ego actually drove -- a rollout that stops early sits at d2gt
# ~ 0 without ever having had the chance to leave the GT curve, and half of an unguarded
# top-10 was exactly that.
#
# The camera side needs a renderer; the top-down side does not, and the six-arm top-down
# for all 150 scenes already exists under video/matrix150, so those are gathered rather
# than re-rendered (identical content, ~15 min of CPU saved).
#
# SCENESET matters: the renderer serves only what its glob matched, and sceneset
# 6f937b0c... is the one holding all 150 of the matrix's usdz. A per-shard sceneset holds
# 38 and would silently miss most of these.

set -uo pipefail

REPO=/home/cvlab21/project/chan/alpamayo-model-compression
WT=$REPO/.claude/worktrees/closed-loop-viz
ALPASIM=/home/cvlab21/project/chan/alpasim
T=/home/cvlab21/project/chan/.claude/jobs/285e9a74/tmp

GPU=${GPU:-1}
PORT=${PORT:-16007}
CONFIG=${CONFIG:-slim_dual_u40_v2}
SCENESET=${SCENESET:-6f937b0c67258133d8e372901b8f2aa8}
OUT=${OUT:-$REPO/closed-loop-viz/video/gt_closest10}
SCENES_FILE=${SCENES_FILE:-$T/gt10.txt}
LOG=$T/gt10_render.log

mkdir -p "$OUT/camera" "$OUT/topdown"
: > "$LOG"
log() { echo "[$(date -u -d '+9 hours' '+%m-%d %H:%M KST')] $*" | tee -a "$LOG"; }

mapfile -t SCENES < "$SCENES_FILE"
log "씬 ${#SCENES[@]}개, arm $CONFIG, GPU $GPU, 씬셋 $SCENESET"

# ---- 1. top-down: gather the existing six-arm renders ---------------------------------
missing=0
for s in "${SCENES[@]}"; do
  src=$REPO/closed-loop-viz/video/matrix150/$s.mp4
  if [ -s "$src" ]; then
    cp -n "$src" "$OUT/topdown/$s.mp4"
  else
    log "top-down 없음: $s"
    missing=$((missing + 1))
  fi
done
log "top-down $(( ${#SCENES[@]} - missing ))/${#SCENES[@]} 수집 (matrix150 의 6패널 렌더)"

# ---- 2. renderer ----------------------------------------------------------------------
GPU=$GPU PORT=$PORT SCENESET=$SCENESET bash "$WT/closed-loop-viz/start_renderer.sh" >>"$LOG" 2>&1
for _ in $(seq 1 60); do
  docker logs chan_nre_replay 2>&1 | grep -q "Serving on" && break
  sleep 5
done
docker logs chan_nre_replay 2>&1 | grep -q "Serving on" || { log "렌더러 기동 실패"; exit 1; }
avail=$(docker logs chan_nre_replay 2>&1 | grep -o "clipgt-[0-9a-f-]*" | sort -u | wc -l)
log "렌더러 기동, 서빙 가능한 씬 $avail 개"

# every pick must be servable, or the loop fails one scene at a time for a reason the
# log makes look like a render error rather than a missing artifact
for s in "${SCENES[@]}"; do
  docker logs chan_nre_replay 2>&1 | grep -q "$s" || log "경고: 렌더러 목록에 없음 -> $s"
done

# ---- 3. camera ------------------------------------------------------------------------
ok=0
for s in "${SCENES[@]}"; do
  out=$OUT/camera/$s.mp4
  if [ -s "$out" ]; then
    log "skip $s (이미 있음)"
    ok=$((ok + 1))
    continue
  fi
  cd "$ALPASIM" || exit 1
  if CUDA_VISIBLE_DEVICES="" uv run python "$WT/closed-loop-viz/replay_camera.py" \
       --scene "$s" --config "$CONFIG" --camera all \
       --endpoint "localhost:$PORT" --out "$out" >>"$LOG.$s" 2>&1; then
    log "ok   $s  $(du -h "$out" | cut -f1)"
    rm -f "$LOG.$s"
    ok=$((ok + 1))
  else
    log "FAIL $s -- $LOG.$s 참조"
  fi
done

docker rm -f chan_nre_replay >/dev/null 2>&1
log "렌더러 정리, GPU $GPU 반납"
log "완료: 카메라 $ok/${#SCENES[@]},  top-down $(( ${#SCENES[@]} - missing ))/${#SCENES[@]}"
echo "-> $OUT"
