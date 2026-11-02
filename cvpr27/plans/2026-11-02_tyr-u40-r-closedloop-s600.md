# 논문 Tyr 행(`tyr_u40_r`) 폐루프: s600, Wanda-sp 뒤 8장 (2026-11-02)

## 0. 목적

- 논문 표의 Tyr 행은 `tyr_u40_r`(이 레포의 간이 포트, 2026-08 빌드)이다.
  - 폐루프는 패치 전 하네스의 Origin150(`m2601_`)과 Hard100(`h100_`)에서만 쟀다.
  - s600(패치된 하네스)에서는 잰 적이 없다.
- 같은 600씬에서 MoOP, 공식 재빌드판 `tyr_off`, LLM-Pruner, Wanda-sp와 짝지어 비교한다.
- **사용자 지시 (2026-11-02).** Wanda-sp가 끝나면 GPU 8장으로 진행한다.

## 1. 체크포인트

- 드라이버 폴더 `slim_tyr_u40_r`(8월 21일 빌드)를 그대로 쓴다.
  - 가중치 `slim_state.pt`가 있다. Tyr는 OSSCAR 재구성으로 가중치를 다시 쓰므로 가중치가 꼭 필요하다.
  - 제거량은 2,657,452,032개로 MoOP와 같다. 빌드 smoke NLL은 0.610이다.
  - 과거 Origin150·Hard100 폐루프에 쓴 것과 같은 폴더다.

## 2. 실행

- **스크립트:** `cvpr27/code/run_tyr_u40_r_s600.sh`
  - Wanda-sp 스크립트(`run_wandasp_s600.sh`)가 끝나고 GPU 8장이 모두 빌 때까지 기다린다.
  - 그 뒤 smoke(8씬) → 게이트 G1·G2 → 본 런(592씬) → 짝 분석으로 이어진다.
- **카드:** GPU 0–7. 각 씬은 dense 기준선의 카드에서 돈다(`run_dual_s600.sh` 기본 목록).
- **이름:** TAG `t40`. 결과는 `outputs/cvpr27/alpasim_tyr_u40_r_s600/`, 로그는 `outputs/cvpr27/tyr_u40_r_s600.log`.
- **시간:** 약 7.5시간.

## 3. 분석 (자동)

| 비교 대상 | 600씬 | 같은 카드 |
|---|---|---|
| 출하본 MoOP | O | 600씬 전부 |
| MoOP l0keep | O | 471씬 |
| dense | O | 600씬 전부 |
| tyr_off | O | 517씬 |
| lp_m24 | O | 600씬 전부 |
| wanda_sp (분석 시점에 끝나 있으면) | O | 600씬 전부 |

- 리포트: `cvpr27/reports/<날짜>_tyr-u40-r-closedloop-s600.md`. 기존 Origin150·Hard100 결과(패치 전 하네스)도 참고로 붙인다.

## 4. git

- 기준선 비교 라인의 후속이라 main에 바로 커밋한다.
