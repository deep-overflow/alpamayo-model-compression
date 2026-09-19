#!/bin/bash
# Fill video/<arm>/<suite>/{camera,topdown} for every arm that has a run on both suites.
#
# `render_arm_all.sh` does one arm on one suite; this queues all of them. Two phases rather
# than one loop over whole jobs, because the two halves want different hardware: top-down is
# pure CPU and camera needs a GPU renderer, so running a job end to end would leave a card
# idle through its top-down half.
#
#   phase td   -- DO_CAMERA=0,  JOBS_PAR jobs at a time x TD_WORKERS each
#   phase cam  -- DO_TOPDOWN=0, one worker per GPU in GPUS, each with its own renderer
#
# Cards: the default is the Blackwell half (0-3). Rendering is not evaluation, and the rule
# that evaluation stays on Ada exists so a kernel difference cannot reach a reported number
# -- keeping the replay off 4-7 leaves those free for whoever needs them. Replay pixels
# cannot change the driving anyway: the poses come out of rollout.asl already decided.
#
# Every worker re-checks its card is actually EMPTY (<=16 MiB) right before it starts a
# renderer, and waits rather than piling onto someone else's job. "Has room" is not empty.
#
# Existing files are skipped by render_arm_all.sh, so this is resumable: re-running it after
# an interruption costs only the scenes that are missing.
#
# PHASE=td|cam|both  GPUS="0 1 2 3"  JOBS_PAR=4  TD_WORKERS=4  ARMS="tyr coc ..."
#
# To stop it: kill the process GROUP, not this script. `pkill -f render_arms_queue` leaves
# the render_arm_all children and their python running.
#   ps -eo pid,pgid,cmd | grep render_arm_all   then   kill -9 -<PGID>
# and then remove the renderers:  docker rm -f chan_nre_q0 chan_nre_q1 ...

set -uo pipefail

REPO=/home/cvlab21/project/chan/alpamayo-model-compression
WT=$REPO/.claude/worktrees/closed-loop-viz
RUNS=/home/cvlab21/project/chan/alpasim-runs
SOOWON=/mnt/nvme1n1/ad_vla/outputs/soowon/alpasim-analysis/runs_root
T=${T:-/home/cvlab21/project/chan/.claude/jobs/34d28911/tmp}

SS_ORIGIN=6f937b0c67258133d8e372901b8f2aa8   # the 150 matrix usdz
SS_HARD=chan_hard100_all                     # the 100 hard usdz

PHASE=${PHASE:-both}
GPUS=${GPUS:-"0 1 2 3"}
JOBS_PAR=${JOBS_PAR:-4}
TD_WORKERS=${TD_WORKERS:-4}
ZOOM=${ZOOM:-45}
ARMS=${ARMS:-"tyr coc traj wanda tyrK llm-pruner baseline"}

Q=$T/arms_queue
mkdir -p "$Q"
MLOG=$Q/queue.log
log() { echo "[$(date -u -d '+9 hours' '+%m-%d %H:%M KST')] $*" | tee -a "$MLOG"; }

# ---- the arm table ----------------------------------------------------------------------
# One row per arm: the run name under each suite's prefix. llm-pruner is the exception --
# only its hard100 run is under our prefix; the 150-scene one lives in soowon's runs_root,
# which is why grepping only alpasim-runs once made it look like hard100-only.
run_for() {
  local arm="$1" suite="$2"
  case "$arm" in
    tyr)        n=slim_tyr_u40_r ;;
    tyrK)       n=slim_tyrK ;;
    coc)        n=slim_coc_u40_v2 ;;
    traj)       n=slim_traj_u40_v2 ;;
    wanda)      n=slim_wanda_u40_v2 ;;
    dual)       n=slim_dual_u40_v2 ;;
    baseline)   n=baseline ;;
    llm-pruner) [ "$suite" = origin150 ] && { echo "$SOOWON/lp_r50"; return; }; n=lp_r50 ;;
    *) echo ""; return ;;
  esac
  case "$suite" in
    origin150) echo "$RUNS/m2601_merged_$n" ;;
    hard100)   echo "$RUNS/h100_merged_$n" ;;
  esac
}

: > "$Q/jobs.tsv"
for arm in $ARMS; do
  for suite in origin150 hard100; do
    run=$(run_for "$arm" "$suite")
    if [ -z "$run" ] || [ ! -f "$run/aggregate/results-summary.json" ]; then
      log "건너뜀 (완료된 런 없음): $arm / $suite"
      continue
    fi
    ss=$SS_ORIGIN; [ "$suite" = hard100 ] && ss=$SS_HARD
    printf '%s\t%s\t%s\t%s\n' "$arm" "$suite" "$run" "$ss" >> "$Q/jobs.tsv"
  done
done
njobs=$(wc -l < "$Q/jobs.tsv")
log "작업 $njobs개: $(cut -f1 "$Q/jobs.tsv" | sort -u | tr '\n' ' ')"
[ "$njobs" -gt 0 ] || { log "할 일이 없습니다"; exit 1; }

# ---- claim one job atomically -----------------------------------------------------------
# A plain `head -n $((++i))` race gives two workers the same job; both then render the same
# files and the second wastes a card. flock on the cursor makes the read-and-bump atomic.
claim() {
  local cur next
  exec 9>"$Q/cursor.lock"
  flock 9
  cur=$(cat "$Q/cursor" 2>/dev/null || echo 0)
  next=$((cur + 1))
  echo "$next" > "$Q/cursor"
  flock -u 9
  exec 9>&-
  [ "$next" -le "$njobs" ] && sed -n "${next}p" "$Q/jobs.tsv"
}

gpu_free() {  # empty, not merely roomy
  local used
  used=$(nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
  [ -n "$used" ] && [ "$used" -le 16 ]
}

one_job() {   # phase, line, gpu, port, name
  local phase="$1" line="$2" gpu="$3" port="$4" name="$5"
  local arm suite run ss
  IFS=$'\t' read -r arm suite run ss <<< "$line"
  local do_td=1 do_cam=0
  [ "$phase" = cam ] && { do_td=0; do_cam=1; }

  if [ "$do_cam" = 1 ]; then
    local waited=0
    until gpu_free "$gpu"; do
      [ "$waited" = 0 ] && log "GPU $gpu 사용 중 -- 비워질 때까지 대기 ($arm/$suite)"
      waited=1; sleep 60
    done
  fi

  log "시작 [$phase] $arm / $suite  (GPU $gpu)"
  ARM="$arm" SUITE="$suite" RUN="$run" SCENESET="$ss" \
  GPU="$gpu" PORT="$port" NAME="$name" ZOOM="$ZOOM" \
  TD_WORKERS="$TD_WORKERS" DO_TOPDOWN="$do_td" DO_CAMERA="$do_cam" \
  T="$Q/$phase" \
    bash "$WT/closed-loop-viz/render_arm_all.sh" >>"$Q/$phase.$arm.$suite.out" 2>&1
  local rc=$?
  log "끝   [$phase] $arm / $suite  rc=$rc  $(tail -1 "$Q/$phase/all_${arm}_${suite}.log" 2>/dev/null)"
}

run_phase() {
  local phase="$1" par="$2"; shift 2
  local slots=("$@")
  mkdir -p "$Q/$phase"
  echo 0 > "$Q/cursor"
  log "=== phase $phase, 동시 $par ==="
  local pids=()
  local i=0
  for s in "${slots[@]}"; do
    (
      while :; do
        line=$(claim)
        [ -n "$line" ] || break
        one_job "$phase" "$line" "$s" "$((16020 + s))" "chan_nre_q$s"
      done
    ) &
    pids+=($!)
    i=$((i + 1))
    [ "$i" -ge "$par" ] && break
  done
  for p in "${pids[@]}"; do wait "$p"; done
  log "=== phase $phase 완료 ==="
}

# top-down needs no card, so its slots are just a count. The slot still carries a number
# because one_job builds a port and a container name from it, and DO_CAMERA=0 means neither
# is ever used -- one code path rather than two.
td_slots=$(seq 0 $((JOBS_PAR - 1)))
cam_slots=$GPUS
ncam=$(echo $GPUS | wc -w)

case "$PHASE" in
  td)   run_phase td "$JOBS_PAR" $td_slots ;;
  cam)  run_phase cam "$ncam" $cam_slots ;;
  both)
    run_phase td "$JOBS_PAR" $td_slots
    run_phase cam "$ncam" $cam_slots
    ;;
  *) log "PHASE 는 td|cam|both"; exit 1 ;;
esac

log "전체 완료"
for arm in $ARMS; do
  for suite in origin150 hard100; do
    d=$REPO/closed-loop-viz/video/$arm/$suite
    [ -d "$d" ] || continue
    log "  $arm/$suite  topdown $(ls "$d/topdown" 2>/dev/null | wc -l)  camera $(ls "$d/camera" 2>/dev/null | wc -l)"
  done
done
