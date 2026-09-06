#!/bin/bash
# 폐루프 arm 하나를 처음부터 끝까지 돌린다 (체크포인트 대기 -> 드라이버 하드링크 ->
# 도커 네트워크 정리 -> Ada 유휴 대기 -> 샤드 실행 -> 병합 -> 분석).
#
# `2026-09-06_calibration-size-closedloop.html` 의 두 arm 을 이 스크립트로 돌렸다. 매트릭스
# 런처와 달리 arm 하나만 받고, 그 arm 을 남의 작업 위에 발사하지 않도록 카드가 CONSEC 회
# **연속** 비어 있을 때만 시작한다 (잠깐 비는 틈을 유휴로 오인하지 않기 위해).
#
# 이 스크립트가 안고 있는 두 교훈:
#   - 시작 전에 죽은 도커 네트워크를 쓸어낸다. 런마다 브리지 네트워크가 4개씩 남고 주소 풀은
#     32개뿐이라, 차면 샤드가 컨테이너도 못 띄우고 `fully subnetted` 로 죽는다. GPU 문제처럼
#     보이지만 아니다. 우리 런 접두사만 지워서 남의 스택은 건드리지 않는다.
#   - 완료 판정을 프로세스 종료가 아니라 **산출물**로 한다. 샤드가 하나라도 따로 재기동되면
#     런처의 exit 는 더 이상 완료 신호가 아니다.
#
# Usage:
#   bash run_calibsize_arm.sh <config>              # 예: slim_dual_st2000b
#   DIR=<로그디렉터리> CONFIGS_CMP="baseline slim_dual_u40_v2" bash run_calibsize_arm.sh <config>
#
# 취소:  touch $DIR/ABORT     진행:  tail -f $DIR/run.log
set -u

REPO=/home/cvlab21/project/chan/alpamayo-model-compression
ALPASIM=/home/cvlab21/project/chan/alpasim
RUNS=/home/cvlab21/project/chan/alpasim-runs
DRIVERS=/mnt/nvme1n1/ad_vla/data/alpasim/drivers
CFG=${1:?usage: run_calibsize_arm.sh <config> (예: slim_dual_st2000b)}
DIR=${DIR-$RUNS/$CFG}
LOG=$DIR/run.log
mkdir -p "$DIR"
GPUS="4 5 6 7"
FREE_MIB=2048
POLL=60
CONSEC=3
CKPT_WAIT_H=${CKPT_WAIT_H-3}
GPU_WAIT_H=${GPU_WAIT_H-24}
N_SCENES=150
N_ROLLOUTS=2

log() { echo "[$(date -u -d '+9 hours' '+%m-%d %H:%M KST')] $*" >> "$LOG"; }
aborted() { [ -f "$DIR/ABORT" ]; }

log "=== dual@st2000b 폐루프 스케줄러 시작 (pid $$)"
log "    config=$CFG  씬 ${N_SCENES}개 x ${N_ROLLOUTS} rollout  GPU [$GPUS]"

# --- 1) 체크포인트 대기
deadline=$(( $(date +%s) + CKPT_WAIT_H * 3600 ))
SRC=$REPO/outputs/$CFG
while true; do
  aborted && { log "ABORT 감지 -- 종료"; exit 130; }
  if [ -f "$SRC/slim_state.pt" ] && [ -f "$SRC/summary.txt" ]; then
    log "체크포인트 확인: $SRC ($(du -h "$SRC/slim_state.pt" | cut -f1))"; break
  fi
  [ "$(date +%s)" -gt "$deadline" ] && { log "FATAL: ${CKPT_WAIT_H}h 안에 체크포인트 미완성 -- 종료"; exit 1; }
  sleep "$POLL"
done

# --- 2) 드라이버 하드링크
if [ ! -f "$DRIVERS/$CFG/slim_state.pt" ]; then
  mkdir -p "$DRIVERS/$CFG"
  for f in config.json slim_meta.json slim_state.pt summary.txt; do
    ln -f "$SRC/$f" "$DRIVERS/$CFG/$f" || { log "FATAL: 하드링크 실패 ($f)"; exit 1; }
  done
  a=$(stat -c %i "$SRC/slim_state.pt"); b=$(stat -c %i "$DRIVERS/$CFG/slim_state.pt")
  [ "$a" = "$b" ] || { log "FATAL: 하드링크가 아니라 사본 (inode $a != $b)"; exit 1; }
  log "드라이버 준비 완료 (inode 공유 확인)"
fi

# --- 3) 도커 네트워크 정리
# 런마다 브리지 네트워크를 4개 남기고 도커의 주소 풀은 32개뿐이다. 09-06 새벽 sh2/sh3 가
# "all predefined address pools have been fully subnetted" 로 컨테이너도 못 띄우고 죽었다.
# 컨테이너가 붙어 있지 않은 *우리* 런의 네트워크만 지운다 (남의 것은 이름으로 배제).
before=$(docker network ls -q | wc -l)
gone=0
for n in $(docker network ls --format '{{.Name}}' | grep -E '^(h100_|m2601_|cl150_|fmp_)'); do
  [ "$(docker network inspect "$n" --format '{{len .Containers}}' 2>/dev/null)" = "0" ] &&
    docker network rm "$n" >/dev/null 2>&1 && gone=$((gone+1))
done
log "도커 네트워크 정리: $gone개 제거 ($before -> $(docker network ls -q | wc -l))"

# --- 4) Ada 유휴 확인
deadline=$(( $(date +%s) + GPU_WAIT_H * 3600 ))
free_streak=0
while true; do
  aborted && { log "ABORT 감지 -- 종료"; exit 130; }
  [ "$(date +%s)" -gt "$deadline" ] && { log "FATAL: ${GPU_WAIT_H}h 안에 GPU 미확보 -- 종료"; exit 1; }
  busy=""
  for g in $GPUS; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null || echo 99999)
    [ "$used" -gt "$FREE_MIB" ] && busy="$busy $g(${used}MiB)"
  done
  if [ -z "$busy" ]; then
    free_streak=$((free_streak + 1)); log "Ada $GPUS 유휴 ($free_streak/$CONSEC)"
    [ "$free_streak" -ge "$CONSEC" ] && break
  else
    [ "$free_streak" -gt 0 ] && log "GPU 재점유 --$busy, 연속 카운터 초기화"
    free_streak=0
  fi
  sleep "$POLL"
done

# --- 5) 샤드 실행
log "실행 시작: $CFG, ${N_SCENES}씬 / ${N_ROLLOUTS} rollout"
DRIVER_OMP_THREADS=8 bash "$REPO/experiments/head_analysis/launch_alpasim_shards.sh" \
    "$CFG" "$N_SCENES" "$N_ROLLOUTS" "$GPUS" >> "$LOG" 2>&1
log "샤드 런처 종료 (rc=$?)"

# --- 6) 산출물 기준으로 완료 판정 (프로세스 종료가 아니라)
for i in 0 1 2 3; do
  f="$RUNS/m2601_${CFG}_sh${i}/aggregate/results-summary.json"
  [ -f "$f" ] || { log "FATAL: sh$i 가 aggregate 를 쓰지 않았습니다 -- 병합하지 않습니다"; exit 1; }
done
SHARDS=""
for i in 0 1 2 3; do SHARDS="$SHARDS m2601_${CFG}_sh${i}"; done
rm -rf "$RUNS/m2601_merged_${CFG}"
"$REPO/.venv/bin/python" "$REPO/experiments/head_analysis/merge_alpasim_shards.py" \
    --runs-root "$RUNS" --shards $SHARDS --out "m2601_merged_${CFG}" \
    --expect-scenes "$N_SCENES" >> "$LOG" 2>&1
mrc=$?
log "병합 종료 (rc=$mrc)"
[ "$mrc" -ne 0 ] && { log "FATAL: 병합 실패 -- 분석 생략"; exit 1; }

# --- 7) 4-arm 분석
cd "$ALPASIM" && uv run python "$REPO/experiments/head_analysis/analyze_alpasim.py" \
    --runs-root "$RUNS" --prefix m2601_merged_ \
    --configs ${CONFIGS_CMP-baseline slim_dual_u40_v2 slim_dual_st2000} "$CFG" \
    --out "$REPO/outputs/${CFG}_eval" >> "$LOG" 2>&1
log "분석 종료 (rc=$?) -> $REPO/outputs/${CFG}_eval"
log "=== 완료: dual@st2000b 폐루프 150씬"
