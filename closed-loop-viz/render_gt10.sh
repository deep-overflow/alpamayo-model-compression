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
# Resolve the script directory from $0 rather than naming a worktree. These lived in
# .claude/worktrees/closed-loop-viz while the work was in flight, and a worktree goes away
# with its session -- after which every `$HERE/...` here pointed at nothing.
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ALPASIM=/home/cvlab21/project/chan/alpasim
T=/home/cvlab21/project/chan/.claude/jobs/285e9a74/tmp

GPU=${GPU:-1}
PORT=${PORT:-16007}
CONFIG=${CONFIG:-slim_dual_u40_v2}
SCENESET=${SCENESET:-6f937b0c67258133d8e372901b8f2aa8}
OUT=${OUT:-$REPO/closed-loop-viz/video/gt_closest10}
SCENES_FILE=${SCENES_FILE:-$T/gt10.txt}
LOG=${LOG:-$T/gt10_render.log}
# Top-down comes from one of two places. GATHER_FROM copies an existing six-arm render
# (matrix150 already holds all 150, so re-rendering them would be pure waste); TOPDOWN_ARMS
# renders instead, which is what a suite with no prior batch needs. Set exactly one.
GATHER_FROM=${GATHER_FROM:-$REPO/closed-loop-viz/video/matrix150}
TOPDOWN_ARMS=${TOPDOWN_ARMS:-}

mkdir -p "$OUT/camera" "$OUT/topdown"
: > "$LOG"
log() { echo "[$(date -u -d '+9 hours' '+%m-%d %H:%M KST')] $*" | tee -a "$LOG"; }

mapfile -t SCENES < "$SCENES_FILE"
log "씬 ${#SCENES[@]}개, arm $CONFIG, GPU $GPU, 씬셋 $SCENESET"

# ---- 1. top-down ----------------------------------------------------------------------
missing=0
if [ -n "$TOPDOWN_ARMS" ]; then
  tdargs=()
  for a in $TOPDOWN_ARMS; do tdargs+=(--arm "$a"); done
  for s in "${SCENES[@]}"; do
    out=$OUT/topdown/$s.mp4
    [ -s "$out" ] && continue
    cd "$ALPASIM" || exit 1
    if CUDA_VISIBLE_DEVICES="" uv run python "$HERE/render_replay.py" \
         --scene "$s" "${tdargs[@]}" --out "$out" >>"$LOG.td.$s" 2>&1; then
      rm -f "$LOG.td.$s"
    else
      log "top-down FAIL $s -- $LOG.td.$s 참조"
      missing=$((missing + 1))
    fi
  done
  log "top-down $(( ${#SCENES[@]} - missing ))/${#SCENES[@]} 렌더"
else
  for s in "${SCENES[@]}"; do
    src=$GATHER_FROM/$s.mp4
    if [ -s "$src" ]; then
      cp -n "$src" "$OUT/topdown/$s.mp4"
    else
      log "top-down 없음: $s"
      missing=$((missing + 1))
    fi
  done
  log "top-down $(( ${#SCENES[@]} - missing ))/${#SCENES[@]} 수집 ($GATHER_FROM)"
fi

# ---- 2. renderer ----------------------------------------------------------------------
GPU=$GPU PORT=$PORT SCENESET=$SCENESET bash "$HERE/start_renderer.sh" >>"$LOG" 2>&1
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
  if CUDA_VISIBLE_DEVICES="" uv run python "$HERE/replay_camera.py" \
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
