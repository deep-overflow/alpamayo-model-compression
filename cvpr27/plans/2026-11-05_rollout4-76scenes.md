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

## 4. 변경 (2026-11-05 02:5x KST, 사용자 지시): tyr_off는 GPU 4–7에서 동시에

- GPU 4–7이 비어서, tyr_off를 그쪽에서 바로 돌린다. 나머지 arm은 GPU 0–3에서 차례로 돈다.
- **진행 스크립트 교체.** `cvpr27/code/run_rollout4_76_split.sh`가 이어받는다.
  - 기존 큐(`run_rollout4_76.sh`)의 bash만 멈췄다. 그 큐가 띄운 MoOP l0keep 런처는 GPU 0–3에서 계속 돈다.
  - 새 스크립트가 그 런처가 끝나기를 기다렸다가, 정리·병합을 하고 lp_m24 → dense를 GPU 0–3에서 잇는다.
- **씬 배정.** 같은 `scenes76_gpu0-3.csv`를 쓴다. shard i는 GPU 4+i가 되어, 씬이 같은 자리를 지킨다.
- **카드 종류 차이.** tyr_off만 Ada에서 돌고 나머지 세 arm은 Blackwell이다.
  - s600에서는 카드 종류가 결론을 바꾸지 않았다. 다만 이 비교에서는 tyr_off − MoOP l0keep에 카드 종류가 섞인다. 리포트에 적는다.
- **예상 종료.** GPU 0–3 쪽(MoOP l0keep → lp_m24 → dense)이 마지막이다. 약 08:10 + 9 h + 6.5 h로, 11-05 23:40 KST 전후다.

## 5. 경위 (2026-11-05 23:10 – 11-06 11:01 KST)

- **lp_m24.** 23:10 KST에 끝났다(GPU 0–3).
  - 실행 중 rollout 103회가 드라이버 RPC CANCELLED로 실패했고, 런타임이 자동으로 다시 돌렸다(`max_rollout_retries=2`).
  - 실패는 14–16시, 18–19시 KST에만 몰렸다. 서버가 다른 작업으로 붐빈 시간대다. MoOP l0keep·tyr_off 런에서는 0회였다.
  - 완료 뒤 실패로 처리돼 다시 돌려진 1회(`31d24a5c`)가 그 씬의 다섯 번째 rollout으로 남았다. 분석에서는 뺀다(`EXCLUDE`).
- **dense.** 23:10부터 대기했다. 다른 사용자의 작업이 GPU 0·2·3에 올라와 있었다.
  - 11-06 10:51 KST에 서버가 재부팅되면서(커널 5.15.0-198 → 202) 큐가 꺼졌다.
  - 재부팅 직후 GPU 0–3이 모두 비어 있어 11:01 KST에 `cvpr27/code/run_rollout4_76_dense.sh`로 다시 띄웠다. 설정은 같다.
  - 큐는 `setsid`로 띄워 세션이 끝나도 남는다. 로그는 `outputs/cvpr27/rollout4_76/queue_dense.log`.

## 6. 추가 arm (2026-11-06 사용자 지시): FM-only·CE-only, 둘 다 0번 층 Q head 보존판

- **드라이버.** 둘 다 s600에서 돌린 것과 같다.
  - FM-only: `slim_trajfix_u40_l0keep_v2` (s600 런 `f600_merged_…`)
  - CE-only: `slim_coc_u40_l0keep_v2` (s600 런 `c600_merged_…`)
  - prefix는 `r4fm_`, `r4ce_`.
- **실행.** `cvpr27/code/run_rollout4_76_fmce.sh`. dense 큐가 끝나기를 기다렸다가 GPU 0–3에서 FM-only → CE-only 순으로 돈다.
  - MoOP l0keep·lp_m24·dense와 같은 Blackwell 카드, 같은 씬 배정이다.
  - arm마다 CoC 퇴화율을 계산하고, 마지막에 분석을 다시 돌린다.
- **예상.** arm당 약 6.5시간. dense가 11-06 18:30 KST쯤 끝나면 FM-only는 11-07 01:00, CE-only는 07:30 KST 전후다.
- **분석.** `analyze_rollout4_76.py`의 ARMS에 두 arm을 추가했다. 기준은 그대로 MoOP l0keep이다.
  - MoOP l0keep − FM-only, MoOP l0keep − CE-only를 짝 비교한다.
  - s600에서 본 "합집합이 두 반쪽보다 낫다"가 rollout을 반복해도 유지되는지를 본다.
