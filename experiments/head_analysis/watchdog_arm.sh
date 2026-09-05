#!/bin/bash
# hard100 arm 하나를 감시한다. tyr 전용이던 watchdog3 을 인자화한 것.
#
# 상태를 먼저 나눈다 (1판이 이걸 안 해서, 다른 랩 멤버가 카드를 잡은 6시간짜리 정당한 대기를
# 45분 만에 고장으로 오탐했다 — 09-05 03:33):
#   실행 중(우리 컨테이너 있음)  -> 새 rollout 이 STALL_MIN 분 없으면 고장
#   대기 중 + 카드 비어 있음     -> START_GRACE 분째 시작 못 하면 고장
#   대기 중 + 남이 카드 사용     -> 정상 대기, 알람하지 않는다
#
# 사용: bash watchdog_arm.sh <config> <logfile> <done-string> [waiter-script]
#   예: bash watchdog_arm.sh lp_r50 run_lp.log "LP 전체 완료" run_lp.sh
set -u
DIR=/home/cvlab21/project/chan/alpasim-runs/hard100
RUNS=/home/cvlab21/project/chan/alpasim-runs
CFG=${1:?config}
LOGF=$DIR/${2:?logfile}
DONE=${3:?done-string}
WAITER=${4-}
GPUS="4 5 6 7"
FREE_MIB=2048
STALL_MIN=45          # 실행 중. hard100 최장 씬(724 m)이 20분 이상 걸린다
START_GRACE=20        # 카드가 빈 뒤 시작까지 (게이트 3분 + 기동 여유)
DISK_MIN_GB=15
POLL=180
free_since=0

alarm() {
  echo "=== $(date -u -d '+9 hours' '+%m-%d %H:%M KST')  [$CFG] 이상 감지: $1"
  echo; echo "--- $LOGF 끝"; tail -10 "$LOGF" 2>/dev/null
  echo; echo "--- 진행"
  for d in "$RUNS"/h100_${CFG}_sh*/; do
    [ -d "$d" ] || continue
    echo "  $(basename "$d"): $(find "$d" -name metrics.parquet 2>/dev/null | wc -l) rollout"
  done
  echo "--- 우리 컨테이너"; docker ps --format '{{.Names}}\t{{.Status}}' | grep "^h100_${CFG}" | head -5
  echo "--- GPU 점유자"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | while read -r p m; do
    echo "    pid ${p%,} $m $(ps -o cmd= -p "${p%,}" 2>/dev/null | cut -c1-60)"
  done
  echo "--- 디스크"; df -h /mnt/nvme1n1 | tail -1
  echo "--- launch 로그 에러"
  for f in "$RUNS"/h100_${CFG}_sh*.launch.log; do
    [ -f "$f" ] || continue
    e=$(grep -iE "error|traceback|refused|deadline" "$f" | tail -2)
    [ -n "$e" ] && { echo "  $(basename "$f"):"; echo "$e" | sed 's/^/    /'; }
  done
  exit 0
}

gpus_free() {
  for g in $GPUS; do
    u=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    [ -z "$u" ] && return 1
    [ "$u" -ge "$FREE_MIB" ] && return 1
  done
  return 0
}

echo "[$(date -u -d '+9 hours' '+%m-%d %H:%M KST')] 워치독 시작: $CFG (완료 문자열 '$DONE')"
while true; do
  if grep -q "$DONE" "$LOGF" 2>/dev/null; then
    echo "$CFG 정상 완료 감지 — 워치독 종료 (이상 없음)"; exit 0
  fi
  grep -q "FATAL" "$LOGF" 2>/dev/null && alarm "로그에 FATAL"
  if [ -n "$WAITER" ] && ! pgrep -f "bash $DIR/$WAITER" >/dev/null 2>&1; then
    alarm "$WAITER 가 완료 기록 없이 종료됨"
  fi

  free_gb=$(df -BG --output=avail /mnt/nvme1n1 | tail -1 | tr -dc '0-9')
  [ "${free_gb:-999}" -lt "$DISK_MIN_GB" ] && alarm "디스크 여유 ${free_gb}GB"

  n_ours=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -c "^h100_${CFG}" || true)
  if [ "$n_ours" -gt 0 ]; then
    free_since=0
    newest=$(find "$RUNS"/h100_${CFG}_sh* -name metrics.parquet -printf '%T@\n' 2>/dev/null \
             | sort -n | tail -1)
    if [ -n "${newest:-}" ]; then
      age=$(( ($(date +%s) - ${newest%.*}) / 60 ))
      [ "$age" -ge "$STALL_MIN" ] && alarm "실행 중인데 새 rollout 이 ${age}분째 없음"
    fi
  elif gpus_free; then
    [ "$free_since" -eq 0 ] && free_since=$(date +%s)
    waited=$(( ($(date +%s) - free_since) / 60 ))
    [ "$waited" -ge "$START_GRACE" ] && alarm "GPU 가 ${waited}분째 비어 있는데 시작하지 않음"
  else
    free_since=0        # 남이 카드를 쓰는 중 -- 정상 대기
  fi
  sleep "$POLL"
done
