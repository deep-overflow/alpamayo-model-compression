#!/bin/bash
# hard100 폐루프 실행 — `dual` 세션의 신호(GO 파일)를 기다렸다가 시작한다.
#
# 왜 GPU 유휴만으로 판단하지 않는가: 사용자님 작업이 두 개 줄 서 있어서, 단순히 "Ada 4-7이
# 비면 시작"하면 두 실험 사이의 빈 틈에 잘못 발사된다. 그래서 명시적 바통을 쓴다.
#   1) $DIR/GO 파일이 생길 때까지 대기      <- `dual` 세션이 자기 실험을 마치고 touch 한다
#   2) 그 다음 Ada 4-7이 연속 $CONSEC 회 비어 있는지 확인  <- GO가 조금 일러도 충돌 방지
#   3) baseline -> 병합 -> slim_dual_u40_v2 -> 병합
#
# 시작 신호:  touch /home/cvlab21/project/chan/alpasim-runs/hard100/GO
# 취소:      touch /home/cvlab21/project/chan/alpasim-runs/hard100/ABORT
#            (또는 kill $(cat /home/cvlab21/project/chan/alpasim-runs/hard100/scheduler.pid))
# 진행 확인:  tail -f /home/cvlab21/project/chan/alpasim-runs/hard100/run.log
set -u

DIR=/home/cvlab21/project/chan/alpasim-runs/hard100
REPO=/home/cvlab21/project/chan/alpamayo-model-compression
RUNS=/home/cvlab21/project/chan/alpasim-runs
LOG=$DIR/run.log
GPUS="4 5 6 7"
FREE_MIB=2048          # 유휴 카드도 수백 MiB를 쓸 수 있어 0이 아닌 문턱
POLL=60
CONSEC=3               # GO 이후 3분 연속 비어 있어야 시작
MAX_WAIT_H=${MAX_WAIT_H-36}
N_SCENES=100
N_ROLLOUTS=2
CONFIGS="baseline slim_dual_u40_v2"

log() { echo "[$(date -u -d '+9 hours' '+%m-%d %H:%M KST')] $*" >> "$LOG"; }

aborted() { [ -f "$DIR/ABORT" ]; }

log "=== hard100 예약 대기 시작 (pid $$)"
log "    트리거: $DIR/GO 파일 + Ada $GPUS 유휴 확인"
log "    계획: 씬 100개, config [$CONFIGS], 씬당 $N_ROLLOUTS rollout, 최대 대기 ${MAX_WAIT_H}h"

deadline=$(( $(date +%s) + MAX_WAIT_H * 3600 ))

# --- 1) GO 바통 대기
while [ ! -f "$DIR/GO" ]; do
  if aborted; then log "ABORT 감지 -- 실행하지 않고 종료"; exit 130; fi
  if [ "$(date +%s)" -gt "$deadline" ]; then
    log "FATAL: ${MAX_WAIT_H}h 안에 GO 신호가 오지 않았습니다 -- 실행하지 않고 종료"
    exit 1
  fi
  sleep "$POLL"
done
log "GO 신호 감지 -- GPU 유휴 확인으로 넘어갑니다"

# --- 2) GPU 유휴 확인 (GO가 일러도 남의 작업을 밟지 않도록)
free_streak=0
while true; do
  if aborted; then log "ABORT 감지 -- 실행하지 않고 종료"; exit 130; fi
  if [ "$(date +%s)" -gt "$deadline" ]; then
    log "FATAL: GO 이후에도 GPU가 비지 않았습니다 -- 실행하지 않고 종료"
    exit 1
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
log "Ada $GPUS 유휴 확인 -- 실행 시작"

# --- 3) config를 순서대로 (baseline이 없으면 dual 결과를 읽을 수 없다)
for cfg in $CONFIGS; do
  if aborted; then log "ABORT 감지 -- $cfg 를 시작하지 않고 종료"; exit 130; fi
  log "--- $cfg 시작"
  start=$(date +%s)
  SUITE=public_2601_hard100 \
  PARENT_SUITE=public_2601 \
  PREFIX=h100_ \
  DRIVER_OMP_THREADS=8 \
  SCENES_CSV=$DIR/hard100_suite.csv \
    bash "$DIR/launch_alpasim_shards.sh" "$cfg" "$N_SCENES" "$N_ROLLOUTS" "$GPUS" \
    >> "$LOG" 2>&1
  rc=$?
  mins=$(( ($(date +%s) - start) / 60 ))
  if [ "$rc" -ne 0 ]; then
    log "FATAL: $cfg 런처가 rc=$rc 로 종료 (${mins}분). 이후 config는 실행하지 않습니다."
    exit "$rc"
  fi
  log "$cfg 런 완료 (${mins}분) -- 샤드 병합"

  if ! "$REPO/.venv/bin/python" \
      "$REPO/experiments/head_analysis/merge_alpasim_shards.py" \
      --runs-root "$RUNS" \
      --shards "h100_${cfg}_sh0" "h100_${cfg}_sh1" "h100_${cfg}_sh2" "h100_${cfg}_sh3" \
      --out "h100_merged_${cfg}" --expect-scenes "$N_SCENES" >> "$LOG" 2>&1; then
    log "FATAL: $cfg 병합 실패 -- 이후 config는 실행하지 않습니다."
    exit 1
  fi
  log "$cfg 병합 완료 -> $RUNS/h100_merged_${cfg}"
done

log "=== 전체 완료. 분석 명령:"
log "    cd /home/cvlab21/project/chan/alpasim && uv run python \\"
log "      $REPO/experiments/head_analysis/analyze_alpasim.py \\"
log "      --runs-root $RUNS --out $REPO/outputs/hard100_eval"
