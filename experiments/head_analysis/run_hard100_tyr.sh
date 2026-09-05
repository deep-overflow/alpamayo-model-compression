#!/bin/bash
# hard100 후속 arm: slim_tyr_u40_r. run_hard100.sh(baseline -> dual)가 끝나면 이어받는다.
#
# 실행 중인 run_hard100.sh 의 CONFIGS 를 고치지 않는 이유: bash 는 스크립트를 증분 파싱하므로
# 돌고 있는 파일을 편집하면 실행이 깨진다. 그래서 별도 프로세스로 대기시킨다.
#
# 로그도 분리한다 — 두 프로세스가 같은 파일에 append 하면 줄이 섞일 수 있다.
#
# 취소:  touch /home/cvlab21/project/chan/alpasim-runs/hard100/ABORT_TYR
# 진행:  tail -f /home/cvlab21/project/chan/alpasim-runs/hard100/run_tyr.log
set -u

DIR=/home/cvlab21/project/chan/alpasim-runs/hard100
REPO=/home/cvlab21/project/chan/alpamayo-model-compression
RUNS=/home/cvlab21/project/chan/alpasim-runs
LOG=$DIR/run_tyr.log
PREV_LOG=$DIR/run.log
CFG=slim_tyr_u40_r
GPUS="4 5 6 7"
FREE_MIB=2048
POLL=60
CONSEC=3
WAIT_PREV_H=${WAIT_PREV_H-12}     # 앞선 런(dual)이 끝나기를 기다리는 한도
GPU_WAIT_H=${GPU_WAIT_H-12}       # 그 뒤 카드가 비기를 기다리는 한도
N_SCENES=100
N_ROLLOUTS=2

log() { echo "[$(date -u -d '+9 hours' '+%m-%d %H:%M KST')] $*" >> "$LOG"; }
aborted() { [ -f "$DIR/ABORT_TYR" ]; }

log "=== tyr_r 후속 대기 시작 (pid $$)"
log "    트리거: run.log 의 '전체 완료' + Ada $GPUS 유휴 확인"
log "    계획: $CFG, 씬 $N_SCENES, 씬당 $N_ROLLOUTS rollout"

# --- 1) 앞선 런(baseline+dual) 종료 대기
deadline=$(( $(date +%s) + WAIT_PREV_H * 3600 ))
while ! grep -q "전체 완료" "$PREV_LOG" 2>/dev/null; do
  if aborted; then log "ABORT_TYR 감지 -- 종료"; exit 130; fi
  if grep -q "FATAL" "$PREV_LOG" 2>/dev/null; then
    log "앞선 런이 FATAL 로 끝났습니다 -- tyr_r 을 실행하지 않고 종료"
    exit 1
  fi
  if [ "$(date +%s)" -gt "$deadline" ]; then
    log "FATAL: ${WAIT_PREV_H}h 안에 앞선 런이 끝나지 않음 -- 종료"; exit 1
  fi
  sleep "$POLL"
done
log "앞선 런(baseline+dual) 완료 확인"

# --- 2) GPU 유휴 확인 (컨테이너가 완전히 내려갈 때까지)
deadline=$(( $(date +%s) + GPU_WAIT_H * 3600 ))
free_streak=0
while true; do
  if aborted; then log "ABORT_TYR 감지 -- 종료"; exit 130; fi
  if [ "$(date +%s)" -gt "$deadline" ]; then
    log "FATAL: ${GPU_WAIT_H}h 동안 GPU 가 비지 않음 -- 종료"; exit 1
  fi
  busy=0
  for g in $GPUS; do
    used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    [ -z "$used" ] && used=999999
    [ "$used" -ge "$FREE_MIB" ] && busy=1
  done
  if [ "$busy" -eq 0 ]; then
    free_streak=$((free_streak + 1))
    log "GPU 유휴 확인 $free_streak/$CONSEC"
    [ "$free_streak" -ge "$CONSEC" ] && break
  else
    [ "$free_streak" -gt 0 ] && log "다시 사용 중 -- 카운터 초기화"
    free_streak=0
  fi
  sleep "$POLL"
done
log "Ada $GPUS 유휴 확인 -- $CFG 시작"

start=$(date +%s)
SUITE=public_2601_hard100 \
PARENT_SUITE=public_2601 \
PREFIX=h100_ \
DRIVER_OMP_THREADS=8 \
SCENES_CSV=$DIR/hard100_suite.csv \
  bash "$DIR/launch_alpasim_shards.sh" "$CFG" "$N_SCENES" "$N_ROLLOUTS" "$GPUS" >> "$LOG" 2>&1
rc=$?
mins=$(( ($(date +%s) - start) / 60 ))
if [ "$rc" -ne 0 ]; then
  log "FATAL: 런처가 rc=$rc 로 종료 (${mins}분)"; exit "$rc"
fi
log "$CFG 런 완료 (${mins}분) -- 샤드 병합"

if ! "$REPO/.venv/bin/python" "$REPO/experiments/head_analysis/merge_alpasim_shards.py" \
    --runs-root "$RUNS" \
    --shards "h100_${CFG}_sh0" "h100_${CFG}_sh1" "h100_${CFG}_sh2" "h100_${CFG}_sh3" \
    --out "h100_merged_${CFG}" --expect-scenes "$N_SCENES" >> "$LOG" 2>&1; then
  log "FATAL: 병합 실패"; exit 1
fi
log "$CFG 병합 완료 -> $RUNS/h100_merged_${CFG}"
log "=== tyr_r 전체 완료"
