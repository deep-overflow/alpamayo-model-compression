# CE-only 폐루프 (0번 층 보존판): s600, GPU 0–3 (2026-10-30)

## 0. 목적

- **무엇을 재나.** CE-only(CoC NLL Taylor 기준 단독)의 폐루프 성능을 같은 600씬에서 잰다.
- **0번 층 보존 (사용자 지시, 2026-10-30).**
  - 0번 층 Q head 32개를 모두 남기고, 그만큼을 MLP 채널 1,109개로 갚는다. VLM −24.0%는 다른 arm과 같다.
  - 그러면 FM-only 가드판, MoOP 0번 층 보존판과 0번 층 조건이 같다. 세 arm의 차이는 1–34번 층의 선택 기준이다.
  - CE-only는 35번 층 가드가 필요 없다. `I_CoC`는 36개 층 모두 평평하지 않다. 0이거나 평평한 층은 Q·MLP 모두 없다.
  - 그래서 35번 층은 세 arm 모두 CoC 점수로 고른 셈이다. FM-only는 가드로, MoOP는 max 합집합에서 궤적 항이 0이라서다.

## 1. 체크포인트 (2026-10-30 빌드·검증 완료)

- `make_slim.py --config coc_u40_l0keep_v2 --importance importance_v2` → `outputs/slim_coc_u40_l0keep_v2`, 가중치를 포함한다.
  - 10-15 빌드에는 메타만 있었다(`--no-state`).
  - driver 폴더 `/mnt/nvme1n1/ad_vla/data/alpasim/drivers/slim_coc_u40_l0keep_v2`에 하드링크했다.
- 검증
  - `slim_meta.json`이 10-15 빌드와 바이트 단위로 같다. 유닛 선택이 같다는 뜻이다.
  - 0번 층 Q 32/32 유지(13개 복원), 제거량 2,657,447,936개(24.0%). MoOP·FM-only 0번 층 보존판과 같다.
  - 빌드 smoke 클립 CoC NLL 0.2249(10-15 빌드 0.2238. 유닛이 같으니 수치 차이로 본다).

## 2. 실행

- **설정**은 기준선과 같다: 패치된 alpasim, `pair_a15`, `DRIVER_OMP_THREADS=8`.
- **카드**: GPU 0–3(Blackwell). FM-only와 같은 이유로 4번 카드는 다른 사용자가 쓰고 있다.
  - 씬 목록과 카드 배정은 FM-only와 같다(`scenes_t600.csv`, `scenes_smoke4.csv`를 `outputs/alpasim_cocl0_s600/`로 복사).
  - 따라서 CE-only − FM-only는 600씬 모두 같은 카드다.
- **smoke**: 4씬, 게이트 G1·G2를 통과해야 본 런을 자동으로 시작한다.
- **본 런**: 596씬, 약 13.5시간 예상(FM-only 실측).
  ```
  CFG=slim_coc_u40_l0keep_v2 TAG=c600 OUT=outputs/alpasim_cocl0_s600 GPUS="0 1 2 3"
  SMOKE_CSV=.../scenes_smoke4.csv MAIN_CSV=.../scenes_t600.csv STAGE=smoke|main
  ```

## 3. 분석 (`experiments/head_analysis/finish_coc_s600.sh`, 본 런 뒤 자동)

| 비교 대상 | 600씬 | 같은 카드 |
|---|---|---|
| MoOP 0번 층 보존판 (주 비교) | O | 429씬 (`card_mismatch_vs_l0keep.csv` 제외) |
| FM-only 가드판 | O | 600씬 전부가 같은 카드라 따로 내지 않는다 |
| dense | O | 300씬 (`card_mismatch.csv` 제외) |
| 출하본 MoOP | O | 300씬 |

- 리포트: `reports/evaluation/<날짜>_coc-closedloop-s600.md`
  - 세 기준(FM-only, CE-only, MoOP)을 0번 층 보존 조건에서 한 표로 묶는다.
  - FM-only 리포트의 CoC 퇴화–충돌 사분위 분석도 CE-only에 적용한다.

## 4. git

- 브랜치 `coc-closedloop-s600`. `traj-closedloop-s600`(PR #97) 위에 쌓았다. `run_dual_s600.sh`의 GPU 인자가 아직 main에 없어서다.
- #97이 머지되면 main을 base로 PR을 연다. 커밋 대상은 계획서, 분석 스크립트, 리포트다.

## 5. 변경 (2026-10-30, 사용자 요청): 8장으로 이어 돌리기

- **바꾼 이유.** 4번 카드가 비면서 GPU 4–7이 모두 비었다. 사용자가 빈 카드를 모두 써서 빨리 끝내라고 요청했다.
- **멈춘 런.** 4장 본 런은 03:17 UTC에 시작해 04:58 UTC에 멈췄다(KST 12:17–13:58).
  - 끝난 62씬(shard 0–3에서 14/16/16/16)은 `alpasim-aggregation`으로 채점해 살렸다. smoke 4씬과 합쳐 66씬이다.
  - 진행 중이던 rollout은 버렸다. 그 씬은 이어 돌리기에서 처음부터 다시 돈다.
- **이어 돌리기.** 남은 534씬(`scenes_resume534.csv`)을 GPU 0–7에서 돌린다. 각 씬은 **dense 기준선과 같은 카드**에 놓는다.
  - 카드당 66–68씬이다. 8장 동시 처리량은 dense 런 기준 Blackwell 11.7, Ada 10.5 씬/시간이다.
  - prefix `c600b_`, 로그 `outputs/alpasim_cocl0_s600/resume.log`
- **카드 종류의 결과.**
  - dense·출하본 MoOP와는 거의 모든 씬이 같은 카드다. 다른 것은 4장 단계에서 Blackwell로 돈 기준선 Ada 씬뿐이다.
  - FM-only(전부 Blackwell)와는 이어 돌리기에서 Ada로 간 씬이 어긋난다. 그래서 FM-only 대비도 같은 카드 비교를 따로 낸다.
- **분석.** `finish_coc_s600.sh`를 고쳤다.
  - 세 묶음을 600씬으로 병합한다.
  - 실제로 쓴 카드에서 비교 대상별 불일치 목록을 만든다(`card_mismatch_vs_{dense,l0keep,fmonly}.csv`).
  - 네 비교 대상마다 600씬과 같은 카드 결과를 낸다.

## 6. 결과 (2026-10-30)

- 8장 이어 돌리기: 04:59–13:01 UTC, 534/534 씬. 정리 완료.
- 병합 600/600, 중단 8씬(다른 압축 arm들과 같은 씬).
- 카드 종류 불일치: dense·출하본 MoOP 대비 31씬, MoOP 0번 층 보존판 대비 160씬, FM-only 대비 269씬.
- 결과 요약
  - − MoOP 0번 층 보존판: score −0.048 [−0.075, −0.021], 같은 카드 440씬 −0.042. progress −0.038, 과실 충돌 +0.029
  - − FM-only: −0.024 [−0.051, +0.003]
  - − dense: −0.003 (같음)
- 리포트: `reports/evaluation/2026-10-30_coc-closedloop-s600.md`
