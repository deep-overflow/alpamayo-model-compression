# MoOP l0keep 제거 비율 스윕 (u35 / u45 / u50, A100)

날짜: 2026-11-01. 상태: **사용자 지시로 진행** ("MoOP keep l0 u35, u45, u50 진행해줘", 평가 test2000·OOD-Val·LingoQA).

## 목적
- MoOP l0keep(`dual_u40_l0keep_v2`)의 기준·0번 층 보존 규칙은 그대로 두고 **층당 절단 비율만** 바꿨을 때
  주행(test_2000, OOD-Val262)과 언어(LingoQA)가 어떻게 변하는지, u40 주변의 예산 곡선을 A100 표 안에서 잰다.

## 설정
- config `dual_u{35,45,50}_l0keep_v2` (`make_slim.py`, importance_v2 = calib_100, NEURON 사본도 sha256 동일).
  층당 Q head와 MLP 채널을 round(N% × 유닛 수)만큼 dual 순위로 자르고, 0번 층 Q head는 전부 남기고 그만큼을
  모든 층의 MLP 채널로 갚는다(u40에서 13 head → 1,109 채널).
- 선택만 하는 레시피(`--no-state`): A100 평가 때 base 가중치에서 그대로 다시 만든다.
- 비교 기준: 같은 A100 표의 MoOP l0keep(u40), MoOP(u40), dense. 같은 비율의 l0keep 아닌 MoOP는 재지 않는다(사용자 범위).

## 절차
1. **빌드 (NEURON, A100 1장):** `cvpr27/code/build_l0keep_ratios.sbatch`, 이 트랙 코드 사본 `/scratch/e1997a06/project/amc-tyr`.
   로컬 GPU 8장이 모두 사용 중이라 NEURON에서 빌드한다.
2. **게이트:** 같은 잡에서 `dual_u40_l0keep_v2`를 다시 빌드해 기존 `outputs/slim_dual_u40_l0keep_v2`와 유닛 단위로 같은지 확인.
   각 비율의 제거 파라미터 수를 기록.
3. **평가:** `eval_a100.sbatch` arm `l0keep_u35/45/50`, `a100_queue.sh arm --lingo ...`(작업 21개), 2장 홀더.
   `tyr_u40_r` 7개와 같은 큐를 쓴다.
4. **보고:** `analyze_a100.py --stage table`(REFS: moop_l0keep, moop), 리포트, README, main 커밋.

## 비용
- 빌드 약 20분(A100 1장). 평가는 비율당 약 5–6 A100-시간, 세 비율 약 16 A100-시간.
