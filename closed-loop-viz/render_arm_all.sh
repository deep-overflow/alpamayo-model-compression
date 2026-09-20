#!/bin/bash
# Render every scene of one suite for one arm: top-down (whole + zoom) and four-camera.
#
# Files are named `<score>_<scene>.mp4` with the score to three decimals, so listing a
# directory sorts by score. The score is the SCENE score -- the mean over the scene's
# rollouts, the number the tables use -- not the score of the worse rollout the video
# happens to show; those differ on a split scene.
#
# Top-down needs no GPU. Camera needs a renderer, and the renderer serves only the scenes
# its --artifact-glob matched, so SCENESET has to be one that holds the whole suite.
#
# Existing files are skipped, so an interrupted run resumes and a suite already done costs
# nothing.

set -uo pipefail

REPO=/home/cvlab21/project/chan/alpamayo-model-compression
WT=$REPO/.claude/worktrees/closed-loop-viz
ALPASIM=/home/cvlab21/project/chan/alpasim
# Overridable: the default is the job dir this was first written under, and a job dir goes
# away with its job. A caller running from a later session passes its own.
T=${T:-/home/cvlab21/project/chan/.claude/jobs/285e9a74/tmp}

ARM=${ARM:-dual}
SUITE=${SUITE:-origin150}
RUN=${RUN:?RUN 이 필요합니다 (alpasim-runs 아래 이름 또는 절대경로)}
SCENESET=${SCENESET:?SCENESET 이 필요합니다}
GPU=${GPU:-1}
PORT=${PORT:-16007}
NAME=${NAME:-chan_nre_$SUITE}
ZOOM=${ZOOM:-45}
TD_WORKERS=${TD_WORKERS:-3}
DO_TOPDOWN=${DO_TOPDOWN:-1}
DO_CAMERA=${DO_CAMERA:-1}

# Resolve RUN to an absolute path HERE. Both renderers expand a bare name under the
# m2601_merged_ prefix, so passing `m2601_merged_slim_dual_u40_v2` produced
# `m2601_merged_m2601_merged_...` and every scene failed instantly -- and a hard100 run
# name would have been prefixed with the wrong suite entirely. Normalising once removes
# the rule the caller would otherwise have to remember.
case "$RUN" in
  /*) ;;
  *) RUN=/home/cvlab21/project/chan/alpasim-runs/$RUN ;;
esac
[ -f "$RUN/aggregate/results-summary.json" ] || {
  echo "RUN 이 잘못되었습니다: $RUN"; exit 1; }

OUT=$REPO/closed-loop-viz/video/$ARM/$SUITE
LOG=$T/all_${ARM}_${SUITE}.log
LIST=$T/scenes_${ARM}_${SUITE}.tsv

mkdir -p "$OUT/camera" "$OUT/topdown"
: > "$LOG"
log() { echo "[$(date -u -d '+9 hours' '+%m-%d %H:%M KST')] $*" | tee -a "$LOG"; }

"$REPO/.venv/bin/python" "$WT/closed-loop-viz/scene_scores.py" --run "$RUN" --out "$LIST" \
  >>"$LOG" 2>&1 || { log "씬 목록 생성 실패"; exit 1; }
n=$(wc -l < "$LIST")
log "$ARM / $SUITE: 씬 $n개, zoom ±${ZOOM}m, 씬셋 $SCENESET"

# ---- top-down: CPU only ----------------------------------------------------------------
td_one() {
  local score="${1%%$'\t'*}" scene="${1##*$'\t'}"
  local out="$OUT/topdown/${score}_${scene}.mp4"
  [ -s "$out" ] && return 0
  cd "$ALPASIM" || return 1
  if CUDA_VISIBLE_DEVICES="" uv run python "$WT/closed-loop-viz/render_replay.py" \
       --scene "$scene" --arm "$ARM=$RUN" --zoom "$ZOOM" --out "$out" \
       >>"$LOG.td.$scene" 2>&1; then
    echo "td ok   $score $scene" >> "$LOG"; rm -f "$LOG.td.$scene"
  else
    echo "td FAIL $score $scene" >> "$LOG"
  fi
}
export -f td_one
export OUT ALPASIM WT ARM RUN ZOOM LOG

if [ "$DO_TOPDOWN" = 1 ]; then
  log "top-down 시작 (워커 $TD_WORKERS)"
  xargs -a "$LIST" -d '\n' -P "$TD_WORKERS" -I{} bash -c 'td_one "$1"' _ {}
  log "top-down 완료: ok $(grep -c '^td ok' "$LOG") / FAIL $(grep -c '^td FAIL' "$LOG")"
fi

# ---- camera: one renderer, scenes served serially ---------------------------------------
if [ "$DO_CAMERA" = 1 ]; then
  GPU=$GPU PORT=$PORT NAME=$NAME SCENESET=$SCENESET \
    bash "$WT/closed-loop-viz/start_renderer.sh" >>"$LOG" 2>&1
  for _ in $(seq 1 60); do
    docker logs "$NAME" 2>&1 | grep -q "Serving on" && break
    sleep 5
  done
  docker logs "$NAME" 2>&1 | grep -q "Serving on" || { log "렌더러 기동 실패"; exit 1; }
  log "렌더러 $NAME (GPU $GPU), 서빙 씬 $(docker logs "$NAME" 2>&1 |
      grep -o 'clipgt-[0-9a-f-]*' | sort -u | wc -l)개"

  # Read the list on fd 3, and give the renderer /dev/null for stdin. On fd 0 the python
  # child consumed part of the list, so `read` later returned half a line: the scene came
  # back empty and the score field held the whole record. It failed loudly only because
  # replay_camera rejects an empty --scene.
  while IFS=$'\t' read -r score scene <&3; do
    [ -n "$scene" ] || { echo "cam SKIP 빈 줄" >> "$LOG"; continue; }
    out="$OUT/camera/${score}_${scene}.mp4"
    [ -s "$out" ] && continue
    cd "$ALPASIM" || exit 1
    if CUDA_VISIBLE_DEVICES="" uv run python "$WT/closed-loop-viz/replay_camera.py" \
         --scene "$scene" --config "$RUN" --camera all \
         --endpoint "localhost:$PORT" --out "$out" \
         </dev/null >>"$LOG.cam.$scene" 2>&1; then
      echo "cam ok   $score $scene" >> "$LOG"; rm -f "$LOG.cam.$scene"
    else
      echo "cam FAIL $score $scene -- $LOG.cam.$scene" >> "$LOG"
    fi
  done 3< "$LIST"

  docker rm -f "$NAME" >/dev/null 2>&1
  log "렌더러 정리, GPU $GPU 반납"
  log "camera 완료: ok $(grep -c '^cam ok' "$LOG") / FAIL $(grep -c '^cam FAIL' "$LOG")"
fi

log "완료: $OUT  (topdown $(ls "$OUT/topdown" | wc -l), camera $(ls "$OUT/camera" | wc -l))"
