# Tyr 공식 재빌드판(`tyr_off`) 폐루프: s600, GPU 4–7 (2026-11-01)

## 0. 목적

- **출발점.** A100 개루프(`cvpr27/reports/2026-11-01_tyr-official-a100.md`)에서 `tyr_off`는 MoOP와 같은 제거량이다.
  - 주행: dense 수준이고, MoOP보다 좋았다(−0.075*).
  - LingoQA: 38.0%로 무너졌다.
- **질문.** 개루프 주행 우위가 폐루프에서도 나오는가. MoOP·LLM-Pruner와 같은 600씬에서 짝지어 본다.
- **사용자 지시 (2026-11-01).** GPU 4–7에서 돌린다.

## 1. 체크포인트

- `outputs/cvpr27/slim_tyr_off`(다른 세션이 cvlab21에서 빌드, 가중치 포함, 16.8 GB)를 드라이버 폴더 `slim_tyr_off`에 하드링크했다.
- **구성.**
  - 제거량 2,657,452,032개로 MoOP와 같다. k/v와 expert는 손대지 않았다.
  - 층마다 유지하는 Q head·MLP 채널 수가 다르다. 탐색으로 정한 할당이다.
- **확인.** 논문 Tyr(`slim_tyr_u40_r`)와 VLM 선택이 다르다. meta의 `config: tyr_u40_r`는 빌드 설정 이름을 그대로 쓴 것뿐이다.

## 2. 실행

- **스크립트:** `cvpr27/code/run_tyr_off_s600.sh` (`run_dual_s600.sh`에 `GPUS="4 5 6 7"`).
- **이름:** TAG `tyo`. 결과는 `outputs/cvpr27/alpasim_tyr_off_s600/`, 런은 `alpasim-runs/tyo_merged_slim_tyr_off`.
- **씬 배정.**
  - `shard` = dense 기준선 GPU 번호를 4로 나눈 나머지. GPU 목록이 4–7이므로, 기준선이 Ada였던 300씬은 dense와 **같은 물리 카드**에서 돈다.
  - 기준선이 Blackwell이었던 300씬은 Ada에서 돈다. 이 300씬은 `card_mismatch_vs_dense.csv`에 적는다.
  - MoOP l0keep과 카드 종류가 다른 씬은 429씬이다(`card_mismatch_vs_l0keep.csv`).
- **smoke.** 기준선이 Ada였던 smoke 4씬을 카드당 1씬씩, 기준선과 같은 카드에서 돌린다. 게이트는 G1·G2다.
- **본 런.** 596씬, 카드당 약 150씬. 약 15–20시간 걸린다.
  - CoC 길이에 따라 달라진다. LingoQA가 무너진 판이라 생성이 길어질 수 있다.
- **동시 실행.** GPU 0–3에서는 LLM-Pruner 두 판의 Blackwell 절반이 돌고 있다. wizard가 동시에 뜨지 않으므로 포트 충돌 위험은 없다.

## 3. 분석 (자동)

| 비교 | 600씬 | 같은 카드 |
|---|---|---|
| − 출하본 MoOP (주 비교) | O | 300씬 |
| − MoOP l0keep | O | 171씬 |
| − dense | O | 300씬 |
| − lp_m24 (분석 시점에 lp_m24가 끝나 있으면) | O | 300씬 |

- 리포트: `cvpr27/reports/<날짜>_tyr-off-closedloop-s600.md`. 개루프 A100 결과와 나란히 놓는다.

## 4. git

- 기존 Tyr 기준선 라인의 후속이라 main에 바로 커밋한다.
