# FM-only 폐루프 (35번 층 가드판): s600, GPU 0–3 (2026-10-29)

## 0. 목적

- **무엇을 재나.** FM-only(궤적 flow-matching Taylor 기준 단독)의 폐루프 성능을 같은 600씬에서 잰다.
  - dense(`s600_merged_baseline`)와 씬별로 짝짓는다.
  - 출하본 MoOP(`d600_merged_slim_dual_u40_v2`)와도 씬별로 짝짓는다.
- **35번 층 가드를 넣는 이유 (사용자 결정, 2026-10-29).**
  - 논문 FM-only arm(`traj_u40_v2`)은 35번 층의 `I_traj`가 Q·MLP 모두 정확히 0이다. 그 층의 `o_proj`/`down_proj`는 expert가 읽는
    KV 캐시에 닿지 않는다.
  - 그래서 남길 유닛을 `argsort`의 동점 정리 순서가 정한다: Q는 13–31번, MLP는 임의의 7,390개.
  - 이 판(`trajfix_u40_v2`)은 그 층만 `I_CoC`로 고른다. 0–34번 층은 출하본 FM-only와 유닛 단위로 같다.

## 1. 사용자 결정

- 체크포인트: FM-only 출하본 레시피에 35번 층 가드만 추가한다(0번 층은 출하본처럼 자른다).
- GPU: 0–3(Blackwell 4장)만 쓴다.

## 2. 체크포인트 (2026-10-29 빌드·검증 완료)

- `make_slim.py --config trajfix_u40_v2 --importance importance_v2` → `outputs/slim_trajfix_u40_v2`, driver 폴더에 하드링크.
- 검증 결과
  - VLM 36개 층 중 35개(0–34)가 `slim_traj_u40_v2`와 같다. 35번 층만 다르고, expert도 같다.
  - 제거 파라미터 2,657,452,032개(24.0%)로 출하본과 같다.
  - 35번 층 겹침: 출하본 FM-only와 Q 13/19, MLP 4,471/7,390. 출하본 MoOP와 Q 15/19, MLP 5,626/7,390.
  - 빌드 smoke 클립(1개) CoC NLL: 출하본 FM-only 0.795 → 가드판 0.468.

## 3. 실행

- **설정**은 기준선과 같다: 패치된 alpasim, `pair_a15`, `DRIVER_OMP_THREADS=8`.
- **카드 배정** (`outputs/alpasim_traj_s600/scenes_t600.csv`)
  - 기준선에서 Blackwell 0–3에 있던 300씬은 같은 GPU에 고정한다.
  - Ada 4–7에 있던 300씬은 GPU (원래 번호 − 4)로 옮긴다. 카드당 150씬이다.
  - 카드 종류가 어긋나는 300씬은 `card_mismatch.csv`에 기록한다.
- **smoke**: 4씬(카드당 1씬, 기준선과 같은 카드), 게이트는 이전 arm과 같다. 통과하면 본 추정에 포함한다.
- **본 런**: 596씬. 약 13–14시간 예상이다.
- **실행 방법**: `run_dual_s600.sh`에 `GPUS`, `SMOKE_CSV`, `MAIN_CSV` 환경변수를 추가했다(기본값은 이전 동작 그대로).
  ```
  CFG=slim_trajfix_u40_v2 TAG=t600 OUT=outputs/alpasim_traj_s600 GPUS="0 1 2 3"
  SMOKE_CSV=.../scenes_smoke4.csv MAIN_CSV=.../scenes_t600.csv STAGE=smoke|main
  ```

## 4. 분석

`analyze_dual_s600.py`로 네 가지를 낸다.
1. FM-only 가드판 − dense
2. FM-only 가드판 − dense, `--exclude card_mismatch.csv`(같은 카드 300씬)
3. FM-only 가드판 − 출하본 MoOP
4. FM-only 가드판 − 출하본 MoOP, 같은 카드 300씬

- 리포트: `reports/evaluation/<날짜>_traj-closedloop-s600.md`
- 후속(별도): 가드판의 개루프(test500) 재측정. 논문 FM-only 개루프 수치와 일관성을 맞추고, 35번 층 가드의 효과를 직접 보기 위해서다.

## 5. git

- 브랜치 `traj-closedloop-s600` (main 기준)
- 커밋 대상: `make_slim.py` trajfix, 실행 스크립트 GPU 인자, 게이트 스크립트(shard 수 가변), 계획서, 리포트
- 끝나면 PR을 연다.

## 6. 변경 (2026-10-29, 사용자 결정): 0번 층 보존 + 35번 층 가드

- **바꾼 이유.** 위 판(`trajfix_u40_v2`)은 0번 층을 출하본처럼 자른다. 0번 층 Q head 선택은 기준과 무관하게 결과를 흔든다(10-13: 같은 레시피
  8 arm의 test500 SD 0.337 → 0.015). FM-only와 MoOP의 기준 차이만 보려면 두 arm의 0번 층이 같아야 한다.
  - 짝 비교 대상은 이미 돌린 MoOP 0번 층 보존판(`l600_merged_slim_dual_u40_l0keep_v2`)이다.
  - 그러면 두 arm은 0번 층(둘 다 보존)과 35번 층(둘 다 CoC 기준)이 같고, 1–34번 층의 기준만 다르다.
- **중단한 런.** `trajfix_u40_v2` 본 런은 12:47 UTC에 시작해 약 20분 만에 멈췄다.
  - smoke 4씬과 본 런 shard 로그 폴더(`t600*_slim_trajfix_u40_v2*`)는 버린다. 레시피가 달라서다.
- **체크포인트.** `make_slim.py --config trajfix_u40_l0keep_v2 --importance importance_v2` → `outputs/slim_trajfix_u40_l0keep_v2`.
  - `traj_u40_l0keep_v2`(기존 FM-only 0번 층 보존 레시피)와 36개 층 중 35개가 같고, 35번 층만 다르다.
  - 0번 층 Q는 32/32를 유지한다. 제거량은 2,657,447,936개로, MoOP 0번 층 보존판과 같다.
  - 빌드 smoke 클립 CoC NLL: `traj_u40_l0keep_v2` 0.700 → 가드판 0.418.
- **실행.**
  - `CFG=slim_trajfix_u40_l0keep_v2 TAG=f600 OUT=outputs/alpasim_trajl0_s600`
  - 씬 목록과 카드 배정은 위와 같다(GPU 0–3, 카드 종류 불일치 300씬).
  - smoke → 게이트 → 본 런을 자동으로 이어 돌린다.
- **분석.**
  1. 주 비교: − MoOP 0번 층 보존판 (600씬, 같은 카드 300씬)
  2. − dense
  3. − 출하본 MoOP

## 7. 결과 (2026-10-30)

- 본 런: 2026-10-29 13:24 UTC – 10-30 02:53 UTC (13.5시간), 600/600 씬, 중단 8씬(자른 arm들과 같은 씬), 정리 완료.
- smoke 4씬: G1·G2 통과.
- 분석: `experiments/head_analysis/finish_traj_s600.sh` → `outputs/alpasim_trajl0_s600/vs_*`
  - 계획의 4개에 더해 MoOP 0번 층 보존판 대비 2개(600씬, 같은 카드 429씬)를 냈다.
- 결과 요약
  - − MoOP 0번 층 보존판: score −0.024 [−0.049, +0.000] (같은 카드 −0.014 n.s.), progress 동일, 충돌 +0.032, CoC 퇴화 3.6% → 9.9%
  - − dense: +0.020 [−0.005, +0.046] (Wilcoxon p=0.0097), progress +4.0 pp
  - − 출하본 MoOP: −0.016 n.s.
- 리포트: `reports/evaluation/2026-10-30_traj-closedloop-s600.md`
