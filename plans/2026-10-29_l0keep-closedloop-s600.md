# MoOP 0번 층 보존판 폐루프: s600에서 dense·출하본 MoOP와 짝지은 비교 (2026-10-29)

## 0. 목적

- **무엇을 재나.** 0번 층 Q head를 자르지 않는 MoOP(`dual_u40_l0keep_v2`, VLM −24.0%)의 폐루프 성능을 같은 600씬에서 잰다.
  - dense 기준선(`s600_merged_baseline`, 0.729)과 씬별로 짝짓는다.
  - 출하본 MoOP(`d600_merged_slim_dual_u40_v2`, 0.766)와도 씬별로 짝짓는다.
- **왜.**
  - 10-13 결과: 0번 층의 Q head 선택이 개루프 붕괴와 시드·추출 분산을 만든다. 0번 층을 보존하면 test500 SD가 0.337 → 0.015로 줄고,
    예산을 맞추는 비용은 +0.004다. 그래서 현행 권장 레시피가 이 판이다.
  - 하지만 폐루프는 한 번도 재지 않았다(CLAUDE.md: "nothing closed loop is re-measured").
  - 출하본 MoOP는 폐루프에서 dense보다 +0.037이었고, 그 이득은 페이스였다. 이 페이스가 0번 층 절단에서 오는지가 열린 질문이다.
    10월 분석에서 VLM을 덜 자를수록 dense처럼 인계 감속을 이어 갔는데, 그 버그는 지금 고쳐져 있다.
- **가설 검정은 아니다.** 두 짝지은 차이와 CI를 보고한다. 짝지은 차이의 검출 한계는 약 0.035다.

## 1. 사용자 결정 (2026-10-29)

- "moop 첫번째 레이어를 제외한 버전으로 진행"을 0번 층 보존판으로 이해했다.
- 이 판은 dual 폐루프를 계획할 때 선택지로 제시했던 판이다.

## 2. 설계 (출하본 MoOP 런과 같음, 체크포인트만 다름)

- **체크포인트.** `make_slim.py --config dual_u40_l0keep_v2 --importance importance_v2`를 가중치 포함으로 다시 빌드했다
  (`outputs/slim_dual_u40_l0keep_v2_state`).
  - 기존 레시피(`outputs/slim_dual_u40_l0keep_v2`)와 VLM·expert 메타가 같다. smoke NLL도 0.2245로 같다.
  - driver 폴더에 하드링크했다(`drivers/slim_dual_u40_l0keep_v2`).
  - 제거 파라미터는 2,657,447,936개로, 출하본보다 4,096개 적다.
- **씬과 카드.** `outputs/alpasim_dual_s600/scenes_d600.csv` 600씬 × 1을 그대로 쓴다. `shard` 열로 씬마다 dense·출하본 MoOP와 같은
  카드에서 돈다.
- **실행.** `run_dual_s600.sh`를 `CFG/TAG/OUT` 환경변수로 parametrize했다(기본값은 출하본 MoOP 그대로).
  - 접두사는 `l600smoke_` / `l600_`, 병합 결과는 `alpasim-runs/l600_merged_slim_dual_u40_l0keep_v2`.
- **smoke 게이트.** 출하본 MoOP와 같다(8/8 채점, slim 인식, 인계 \|차 − 사람\| < 0.1 m/s, 포화 0).
  - 통과하면 본 런이 smoke 8씬을 건너뛰고, 끝난 뒤 합친다.
- **시간.** 약 7시간 15분 예상(출하본 MoOP 실측).

## 3. 분석

- `analyze_dual_s600.py`를 `--base/--arm/--out` 인자로 parametrize했다. 기본값으로 다시 돌린 출하본 MoOP 결과는 이전과 글자 단위로 같다.
  1. `--arm l600_merged_slim_dual_u40_l0keep_v2 --arm-label "MoOP l0keep" --out outputs/alpasim_l0keep_s600 --no-pre-fix`
  2. 같은 arm에 `--base d600_merged_slim_dual_u40_v2 --base-label MoOP --out outputs/alpasim_l0keep_s600/vs_moop`
- 지표, 중단 씬, 부분 집합, CoC 퇴화율은 출하본 MoOP 분석과 같다.
- 리포트: `reports/evaluation/<날짜>_l0keep-closedloop-s600.md`

## 4. git

- 브랜치 `l0keep-closedloop-s600`. `dual-closedloop-s600`(PR #92, 미머지) 위에 쌓는다.
- **PR #92를 먼저 머지할 것.** 이 PR의 base 브랜치를 지우면 스택된 PR이 닫힌다(`--delete-branch` 금지).
