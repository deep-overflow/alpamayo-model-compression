# Experiment: u50 (29.98% 제거) closed loop on Hard100 — 95% CI와 MoOP 대비 짝 차이

- 날짜: 2026-11-15 ~ 2026-11-18 (KST)
- 이전 보고서: Origin150 결과는 `reports/2026-11-12_u50-closedloop-origin150-ci.md`.
- 계획: `plans/2026-11-15_u50-closedloop-hard100.md`
- 코드:
  - 실행: `code/run_u50_hard100.sh <cfg>`
  - 집계: `code/u50_closedloop_ci.py --set hard100`
- 결과 위치:
  - 표: `outputs/iclr27-rebuttal/u50_hard100/ci/{metrics,table}_seed{0,1}.*`
  - 병합된 런: `alpasim-runs/u50h_merged_<cfg>`
  - gate 결과: `outputs/iclr27-rebuttal/u50_hard100/gates_<cfg>.txt`

## Purpose
- Origin150(쉬운 추출)에서 MoOP u50은 모든 u50 baseline과 ICLR dense를 score로 앞섰다.
- 이 실험은 그 순서가 논문이 함께 보고하는 어려운 세트 Hard100에서도 유지되는지 확인한다.

## Setup

| 항목 | 값 |
|---|---|
| scene | Hard100: `outputs/scene_difficulty/hard100_suite.csv`, ICLR Hard100 런과 같은 목록이다. 100 scene × 2 rollout으로 arm당 200 rollout |
| 하네스 | Origin150 u50 arm과 같다. 인계 감속 수정이 들어간 AlpaSim, pair_a15, driver 스레드 8, ASL 이미지 생략 |
| 카드 | MoOP, Tyr, LLM-Pruner, FM-only는 Ada(4–7). Wanda와 CE-only는 Blackwell(0–3). 사용자 결정(11-10)으로 카드 종류는 고려하지 않았고, 카드는 기록만 해 두었다 |
| dense | `h100_merged_baseline`: ICLR판 dense, 논문 표 2·8의 Hard100 Baseline. **인계 감속 수정 전 하네스**에서 돌았다 |
| arm | Origin150과 같은 빌드다. MoOP `dual_u50_v2`, Tyr `tyr_u50_r`, LLM-Pruner `lp_r60`, FM-only `traj_u50_v2`, CE-only `coc_u50_v2`, Wanda `wanda_u50_v2` |
| 지표 | scene score, 과실 충돌률, offroad율, progress(`progress_clipped_rel`). 사건 지표는 rollout에 사건이 한 번이라도 있으면 1이다(논문 표 2·8 규약) |
| 중단 rollout | 시뮬레이터의 경로 정합성 검사("route folds back on itself", 이웃 구간 사이 각도 150° 초과)에 걸린 rollout이다. AlpaSim 규약대로 score 0, 사건 없음, progress 0으로 200개에 포함했다(논문과 같다) |
| 통계 | scene별 평균을 낸 뒤 scene을 다시 뽑는 부트스트랩, 20만 회. 모든 arm이 같은 재표집을 공유한다. 시드 1에서도 유의성이 같다 |
| gate | 모든 arm이 통과했다. G1h는 "충돌·offroad 외 실패가 전부 경로 정합성 검사인가", G2는 "인계 감속 수정 유지"다. 원래 G1은 Hard100에서 구조적으로 실패하므로 G1h로 대체했다 |

## Results (Facts)

### 절댓값 (95% CI)

| Method | 카드 | 중단 | Scene score ↑ | At-fault coll. ↓ | Offroad ↓ | Progress ↑ |
|---|---|---|---|---|---|---|
| Baseline (dense, ICLR) | — | 4 | 0.510 [0.435, 0.586] | 0.030 [0.005, 0.065] | 0.265 [0.185, 0.350] | 0.553 [0.495, 0.611] |
| Wanda u50 | Blackwell | 4 | 0.266 [0.213, 0.322] | 0.025 [0.005, 0.055] | 0.290 [0.205, 0.380] | 0.324 [0.284, 0.365] |
| LLM-Pruner u50 | Ada | 4 | 0.454 [0.383, 0.525] | 0.055 [0.020, 0.095] | 0.210 [0.135, 0.290] | 0.466 [0.414, 0.519] |
| Týr-the-Pruner u50 | Ada | 4 | 0.614 [0.539, 0.687] | 0.020 [0.000, 0.045] | 0.140 [0.080, 0.210] | 0.622 [0.559, 0.683] |
| FM-only u50 | Ada | 4 | 0.289 [0.241, 0.339] | 0.070 [0.030, 0.115] | 0.240 [0.165, 0.320] | 0.300 [0.267, 0.335] |
| CE-only u50 | Blackwell | 4 | 0.153 [0.120, 0.188] | 0.070 [0.030, 0.120] | 0.345 [0.260, 0.435] | 0.195 [0.173, 0.221] |
| **MoOP u50** | Ada | 5 | **0.651 [0.574, 0.725]** | **0.010 [0.000, 0.025]** | 0.195 [0.125, 0.270] | **0.676 [0.619, 0.732]** |

### MoOP u50 대비 짝 차이 (95% CI)

부호는 다음과 같다. 양수면 MoOP가 낫고, `*`는 CI가 0을 벗어난다는 뜻이다.
- score, progress: MoOP − method
- 사건 비율: method − MoOP

| Method | Scene score | At-fault coll. | Offroad | Progress |
|---|---|---|---|---|
| Baseline (dense, ICLR) | **+0.140 [+0.070, +0.212]*** | +0.020 [−0.005, +0.050] | +0.070 [−0.005, +0.145] | **+0.123 [+0.075, +0.174]*** |
| Wanda u50 | **+0.385 [+0.298, +0.470]*** | +0.015 [−0.010, +0.045] | +0.095 [−0.005, +0.195] | **+0.353 [+0.293, +0.412]*** |
| LLM-Pruner u50 | **+0.197 [+0.112, +0.282]*** | **+0.045 [+0.005, +0.090]*** | +0.015 [−0.065, +0.095] | **+0.210 [+0.153, +0.268]*** |
| Týr-the-Pruner u50 | +0.037 [−0.037, +0.111] | +0.010 [−0.015, +0.040] | −0.055 [−0.115, +0.005] | +0.054 [+0.000, +0.110] |
| FM-only u50 | **+0.362 [+0.277, +0.445]*** | **+0.060 [+0.015, +0.110]*** | +0.045 [−0.050, +0.140] | **+0.377 [+0.316, +0.436]*** |
| CE-only u50 | **+0.498 [+0.421, +0.573]*** | **+0.060 [+0.015, +0.110]*** | **+0.150 [+0.055, +0.245]*** | **+0.482 [+0.423, +0.539]*** |

- Tyr의 progress 하한은 시드 0에서 +0.000, 시드 1에서 −0.000이다. 0에 걸쳐 있으므로 유의로 세지 않는다.

### Origin150과 Hard100의 MoOP u50 대비 score 차이

| Method | Origin150 (150 scene) | Hard100 (100 scene) |
|---|---|---|
| Baseline (dense, ICLR) | +0.116* | +0.140* |
| Wanda u50 | +0.516* | +0.385* |
| LLM-Pruner u50 | +0.232* | +0.197* |
| Týr-the-Pruner u50 | +0.124* | +0.037 (n.s.) |
| FM-only u50 | +0.315* | +0.362* |
| CE-only u50 | +0.587* | +0.498* |

두 세트의 score 순서는 다음과 같다.
- Origin150: MoOP 0.866 > dense 0.750 ≈ Tyr 0.742 > LLM-Pruner 0.634 > FM-only 0.551 > Wanda 0.350 > CE-only 0.279
- Hard100: MoOP 0.651 > Tyr 0.614 > dense 0.510 > LLM-Pruner 0.454 > FM-only 0.289 > Wanda 0.266 > CE-only 0.153

### 참고: u40과 u50의 Hard100 score 차이

u40 값은 ICLR Hard100 런(인계 감속 수정 전 하네스)을 같은 코드로 다시 계산한 것이다.

| Method | u40: MoOP u40 − arm | u50: MoOP u50 − arm |
|---|---|---|
| Baseline (dense, ICLR) | +0.085* | +0.140* |
| Wanda | +0.189* | +0.385* |
| LLM-Pruner | +0.020 | +0.197* |
| Týr-the-Pruner | +0.010 | +0.037 |
| FM-only | +0.049* (하한 +0.003) | +0.362* |
| CE-only | +0.180* | +0.498* |

- u40 절댓값: MoOP 0.595, 중단 8개. 같은 하네스에서 돈 arm이 아니므로 u40과 u50 절댓값은 직접 비교하지 않는다.

### 중단 rollout

| scene | 해당 arm | 언제 | 내용 |
|---|---|---|---|
| `4bad2f63` (waypoint 83), `adb899bd` (waypoint 1) | 모든 arm, 각 2개 | 시작 시 | 녹화 궤적 자체가 뒤로 꺾여 있다. 주행 기록이 없다 |
| `ceb3d0d5` (waypoint 237) | MoOP u50만, rollout 1개 | 주행 중 | 지도 차선으로 경로를 연장하다 걸렸다. 원래 시도와 재시도 2번이 모두 같은 지점에서 실패했다 |

### 실행 기록

- **scene 파일:** 11-15에 확인해 보니 Hard100 usdz 27개가 런처의 scene 캐시에 없었다. 이것이 soowon 런이 11-09에 실패한 원인이다. 같은 파일시스템의 `ad_vla/data/alpasim/nre-artifacts/all-usdzs`에서 하드링크해 채웠다.
- **11-17 10:30 KST 디스크 정리:** 메인 venv(`venvs/alpamayo-mc`)와 HF hub 캐시 대부분이 지워졌다.
  - 이미 돌고 있던 Wanda와 LLM-Pruner는 rollout을 끝까지 마쳤다. 병합만 실패해서 alpasim venv로 다시 병합했다.
  - CE-only와 FM-only는 driver 초기화 실패로 4번 다시 띄웠다. 원인은 순서대로 venv, Cosmos-Reason2-8B, Qwen3-VL-8B, Qwen3-VL-2B였고, 마지막 두 시도에서는 포트 충돌도 함께 있었다.
  - 세 저장소는 가중치를 뺀 설정·토크나이저 파일만 이전과 같은 리비전으로 복원했다. driver 이미지 안에서 초기화 경로 전체를 오프라인으로 검증한 뒤 다시 실행했다.
  - 실패한 시도들에서 채점된 rollout은 0개라 잃은 결과는 없다. 실패 디렉토리는 `alpasim-runs/failed_16{17,24,32}/`에 있다.

## Analysis (Opinion)

1. **Hard100에서도 MoOP u50은 Tyr를 뺀 모든 비교 대상보다 score가 유의하게 높다.**
   - dense, Wanda, LLM-Pruner, FM-only, CE-only 다섯 개가 해당한다. 크기와 순서도 Origin150과 거의 같다.
   - 과실 충돌은 LLM-Pruner, FM-only, CE-only가 MoOP보다 유의하게 많다(+0.045 ~ +0.060).
2. **Tyr와의 차이는 Hard100에서 재현되지 않는다(+0.037, n.s.).**
   - Origin150에서는 +0.124*였다.
   - Hard100에서는 Tyr의 offroad가 오히려 낮고(0.140 대 0.195, n.s.), progress 차이도 0에 걸친다.
   - 따라서 "u50 폐루프에서 MoOP가 Tyr보다 낫다"는 Origin150에서만 성립한다. 두 세트를 합친 250 scene(세트 층화) 분석이 이 판정을 정할 다음 단계다.
3. **score 차이의 대부분은 progress(페이스)에서 온다.**
   - dense +0.140 중 progress가 +0.123이다. LLM-Pruner·FM-only·CE-only도 score 차이와 progress 차이가 거의 같다.
   - 사건 지표 차이는 score 차이보다 훨씬 작다. 그래서 "더 안전하게 운전한다"보다 "멈추지 않고 경로를 더 많이 진행한다"가 정확한 설명이다.
4. **dense와의 비교에는 하네스 차이가 섞여 있다.**
   - dense는 인계 감속 수정 전 하네스에서 돌았다.
   - Origin150에서는 수정 후 하네스로 dense를 다시 돌리자 score가 +0.030, progress가 +0.045 올랐다. 그래서 Hard100의 +0.140 중 일부도 하네스 효과일 수 있다.
   - 수정 후 하네스로 돌린 Hard100 dense 런은 없다.
5. **중단 rollout은 MoOP에 불리하게 작용한다.**
   - MoOP만 주행 중 중단을 1번 받았다. scene 평균으로는 약 −0.005다.
   - 이 rollout을 중단 직전까지의 기록으로 다시 채점하거나 해당 scene을 빼도 판정이 바뀔 비교는 없다. u40에서도 같은 처리로 판정이 바뀐 적은 없었다.
6. **u40과 비교하면 압축 강도가 커질수록 방법 사이 격차가 벌어진다.**
   - LLM-Pruner(+0.020 → +0.197*)와 FM-only(+0.049 → +0.362*)가 두드러진다. Origin150과 같은 경향이다.
   - Tyr만 Hard100에서 u40·u50 모두 MoOP와 구분되지 않는다.

다음 단계로 할 만한 것:
- Origin150 + Hard100을 합친 250 scene, 세트 층화 paired 분석. 특히 Tyr와의 비교를 판정한다.
- 수정 후 하네스로 Hard100 dense 실행. dense 비교에서 하네스 차이를 없앤다.
- 필요하면 `ceb3d0d5`의 MoOP rollout을 중단 직전 기록으로 다시 채점.
