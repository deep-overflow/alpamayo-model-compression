# 76개 선별 씬 × rollout 4회 폐루프: MoOP l0keep vs tyr_off vs lp_m24 vs dense (2026-11-05)

## 0. 목적

- s600(rollout 1회)에서 다음 조건을 만족하는 76씬을 골랐다(`outputs/cvpr27/moop_clean_scenes/selected76_tyroff_lpm24.csv`).
  - MoOP l0keep은 과실 충돌(앞·옆)과 offroad가 없었다.
  - tyr_off 또는 lp_m24는 과실 충돌이나 offroad가 있었다. 뒤 충돌은 세지 않는다(사용자 규칙).
  - 구성: tyr_off만 33씬, lp_m24만 38씬, 둘 다 5씬.
- **질문.** 이 차이가 같은 씬에서 rollout을 반복해도 유지되는가.
  - rollout 1회는 우연이 크다. 그리고 고른 rollout 자체를 다시 쓰면 기준(MoOP l0keep)에 유리하게 치우친다(평균으로의 회귀).
  - 그래서 **새로 돌린 4회 rollout만** 쓴다.
- **사용자 결정 (2026-11-05).** arm은 MoOP l0keep, tyr_off, lp_m24, dense. GPU는 0–3만 쓴다.

## 1. 실행

- **스크립트:** `cvpr27/code/run_rollout4_76.sh`. 네 arm을 차례로 돌린다(MoOP l0keep → tyr_off → lp_m24 → dense).
  - arm마다 GPU 0–3이 빌 때까지 기다린다.
  - 그 뒤 `launch_alpasim_shards.sh <driver> 76 4 "0 1 2 3"`(씬당 rollout 4회) → 컨테이너 정리 → 병합 순서로 진행한다.
- **씬 배정:** `outputs/cvpr27/rollout4_76/scenes76_gpu0-3.csv`
  - 카드당 19씬이다. 같은 씬은 모든 arm에서 같은 카드에서 돈다.
  - 원래 기준선이 Ada였던 33씬도 Blackwell에서 돈다. 이 실험 안에서는 arm 간 카드가 같으므로 비교에는 영향이 없다.
- **설정:** 패치된 AlpaSim, `pair_a15`, `DRIVER_OMP_THREADS=8`, CoC 상한 128토큰. s600과 같다. smoke는 생략한다(네 드라이버 모두 s600에서 게이트를 통과했다).
- **이름:** prefix `r4l0_`, `r4ty_`, `r4lp_`, `r4dn_`. 병합 런은 `r4*_merged_<driver>`다.
- **시간 (4장).** arm당 304 rollout이다.
  - MoOP l0keep·tyr_off·dense: 각 약 6.5시간
  - lp_m24: CoC가 길어 약 9시간
  - 합계 약 28시간

### 경위

- 01:35 KST에 8장으로 시작했다가, 사용자 요청으로 5분 만에 멈추고 GPU 0–3으로 다시 띄웠다. 끝난 rollout은 없었다. 멈춘 런의 폴더는 지웠다.

## 2. 분석 (`cvpr27/code/analyze_rollout4_76.py`, 큐 마지막에 자동)

- **사건:** rollout마다 과실 충돌(앞·옆) 또는 offroad. 뒤 충돌은 참고로만 센다.
- **arm별:** 사건 비율(rollout 단위), 과실 충돌·offroad·뒤 충돌 비율, scene score 평균, 4회 모두 무사고인 씬 수.
- **MoOP l0keep과의 짝 차이:** 씬별 사건 비율과 score의 차이, 씬 부트스트랩 20,000회 CI. 선택 그룹(tyr_off/lp_m24/둘 다)별로도 낸다.
- **산출물:** `outputs/cvpr27/rollout4_76/{per_scene.csv, metrics.json, summary.txt, event_rate.png}`.
  - 분석 코드는 s600 런으로 미리 돌려 끝까지 실행되는 것을 확인했다.
- **리포트:** `cvpr27/reports/<날짜>_rollout4-76scenes.md`

## 3. git

- 폐루프 비교 라인의 후속이라 main에 바로 커밋한다.
