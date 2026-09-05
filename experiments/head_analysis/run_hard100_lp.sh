#!/bin/bash
# hard100 후속 arm: soowon 님의 LLM-Pruner 체크포인트. run_tyr.sh 가 끝나면 이어받는다.
#
# 공정성을 위해 우리 arm 들과 완전히 같은 조건으로 돌린다 — 같은 hard100 스위트 CSV,
# 씬당 2 rollout, 샤드당 4서비스를 한 카드에, DRIVER_OMP_THREADS=8. soowon 님의 150씬 런은
# 드라이버와 렌더러를 두 카드에 나눴지만, 여기서 맞춰야 할 대상은 그 런이 아니라 이 hard100
# 매트릭스의 baseline/dual/tyr 이다.
#
# 체크포인트는 soowon 님 원본과 하드링크(같은 inode)라 그분이 평가한 것과 동일하다.
#
# 예산은 정확히 맞지 않는다: LP 2,768,240,640 (25.0%) vs 우리 2,657,452,032 (24.0%).
# LP 쪽이 111M 더 잘라내므로 LP 에 불리한 방향이고, 그래도 동률이면 결론은 보수적이다.
#
# 사용:  bash run_lp.sh "lp_r50_dual"          # 한 arm
#        bash run_lp.sh "lp_r50 lp_r50_dual"   # 둘 다 (순차)
# 취소:  touch /home/cvlab21/project/chan/alpasim-runs/hard100/ABORT_LP
# 진행:  tail -f /home/cvlab21/project/chan/alpasim-runs/hard100/run_lp.log
set -u

DIR=/home/cvlab21/project/chan/alpasim-runs/hard100
REPO=/home/cvlab21/project/chan/alpamayo-model-compression
RUNS=/home/cvlab21/project/chan/alpasim-runs
LOG=$DIR/run_lp.log
PREV_LOG=$DIR/run_tyr.log
PREV_DONE="tyr_r 전체 완료"
CONFIGS=${1:-lp_r50_dual}
GPUS="4 5 6 7"
FREE_MIB=2048
POLL=60
CONSEC=3
WAIT_PREV_H=${WAIT_PREV_H-24}
GPU_WAIT_H=${GPU_WAIT_H-24}     # 다른 랩 멤버가 카드를 오래 잡는 일이 실제로 있었다
N_SCENES=100
N_ROLLOUTS=2

log() { echo "[$(date -u -d '+9 hours' '+%m-%d %H:%M KST')] $*" >> "$LOG"; }
aborted() { [ -f "$DIR/ABORT_LP" ]; }

log "=== LP 후속 대기 시작 (pid $$)  config: $CONFIGS"
log "    트리거: run_tyr.log 의 '$PREV_DONE' + Ada $GPUS 유휴 확인"

# --- 실행 전 점검: 드라이버가 실제로 있는지
for cfg in $CONFIGS; do
  if [ ! -f "/mnt/nvme1n1/ad_vla/data/alpasim/drivers/$cfg/slim_state.pt" ]; then
    log "FATAL: 드라이버 $cfg 의 slim_state.pt 가 없습니다"; exit 1
  fi
done
log "    드라이버 확인 OK"

# --- 1) 앞선 런(tyr) 종료 대기
deadline=$(( $(date +%s) + WAIT_PREV_H * 3600 ))
while ! grep -q "$PREV_DONE" "$PREV_LOG" 2>/dev/null; do
  if aborted; then log "ABORT_LP 감지 -- 종료"; exit 130; fi
  if grep -q "FATAL" "$PREV_LOG" 2>/dev/null; then
    log "앞선 런(tyr)이 FATAL 로 끝났습니다 -- LP 를 실행하지 않고 종료"; exit 1
  fi
  if [ "$(date +%s)" -gt "$deadline" ]; then
    log "FATAL: ${WAIT_PREV_H}h 안에 tyr 가 끝나지 않음 -- 종료"; exit 1
  fi
  sleep "$POLL"
done
log "앞선 런(tyr_r) 완료 확인"

for cfg in $CONFIGS; do
  aborted && { log "ABORT_LP 감지 -- $cfg 시작 안 함"; exit 130; }

  # --- GPU 유휴 확인 (남의 작업을 밟지 않는다)
  deadline=$(( $(date +%s) + GPU_WAIT_H * 3600 ))
  free_streak=0
  while true; do
    aborted && { log "ABORT_LP 감지 -- 종료"; exit 130; }
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
  log "Ada $GPUS 유휴 확인 -- $cfg 시작"

  start=$(date +%s)
  SUITE=public_2601_hard100 \
  PARENT_SUITE=public_2601 \
  PREFIX=h100_ \
  DRIVER_OMP_THREADS=8 \
  SCENES_CSV=$DIR/hard100_suite.csv \
    bash "$DIR/launch_alpasim_shards.sh" "$cfg" "$N_SCENES" "$N_ROLLOUTS" "$GPUS" >> "$LOG" 2>&1
  rc=$?
  mins=$(( ($(date +%s) - start) / 60 ))
  if [ "$rc" -ne 0 ]; then
    log "FATAL: $cfg 런처가 rc=$rc 로 종료 (${mins}분)"; exit "$rc"
  fi
  log "$cfg 런 완료 (${mins}분) -- 샤드 병합"

  if ! "$REPO/.venv/bin/python" "$REPO/experiments/head_analysis/merge_alpasim_shards.py" \
      --runs-root "$RUNS" \
      --shards "h100_${cfg}_sh0" "h100_${cfg}_sh1" "h100_${cfg}_sh2" "h100_${cfg}_sh3" \
      --out "h100_merged_${cfg}" --expect-scenes "$N_SCENES" >> "$LOG" 2>&1; then
    log "FATAL: $cfg 병합 실패"; exit 1
  fi
  log "$cfg 병합 완료 -> $RUNS/h100_merged_${cfg}"
done
log "=== LP 전체 완료"
