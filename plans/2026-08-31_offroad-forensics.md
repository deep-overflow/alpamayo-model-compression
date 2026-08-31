# 재구성은 왜 폐루프에서 지는가 — offroad·차선 유지의 법의학 (rollout-free)

날짜: 2026-08-31. 브랜치: `offroad-forensics` (승인 후). 상태: **완료** (2026-08-31 20:10 KST).

## 왜

폐루프에서 재구성 계열이 선택-only `dual`에 지는 패턴이 세 번 반복됐다(dualr 0.792, tyr_r 0.786,
dualr_wl 0.783 vs dual 0.828, baseline 0.750). `dualr_wl`(2026-08-31)이 그 원인에서 **reasoning을
제거**했다: LingoQA 41.8 → 72.6, CoC 붕괴 0.045 → 0.027로 고쳤는데도 점수는 dualr과 같았고
(wl − dualr −0.009 [−0.038, +0.020]), 대신 **offroad가 0.077 → 0.097로 악화**되며 개선분을 상쇄했다.
남은 후보는 차선 유지·횡방향 행동이다. 이미 저장된 rollout 시계열(10 Hz, `metrics.parquet`)과 ASL
로그로 GPU 없이, 재시뮬레이션 없이 답할 수 있다.

## 가설

- **H1 (횡방향 여유)**: 재구성 arm은 차선 경계에 더 가깝게 달린다 —
  `min_distance_to_lane_boundary_m`의 시간 분포가 dual보다 왼쪽으로 이동. offroad는 그 꼬리다.
- **H2 (GT 추종 ↔ 차선)**: 재구성 arm의 낮은 d2gt(2.82–2.97 vs dual 3.32)와 낮은 차선 여유가 같이
  간다 — 재구성이 캘리브레이션 클립의 GT 궤적을 더 충실히 재현하고, 그 궤적은 *현재 씬의* 차선
  기하와 어긋난다(캘리브레이션 과적합의 폐루프 발현). 씬 단위로 d2gt와 lane-margin의 상관을 본다.
- **H3 (누적 vs 순간)**: offroad 이탈은 (a) 서서히 밀려나는 표류인가 (b) 특정 조작(회전·차선변경)의
  실패인가 — 이탈 직전 5 s의 lane margin 기울기와 heading 변화율로 구분.

## 계측 (전부 저장된 산출물에서)

`analyze_longitudinal.py`의 구조를 그대로 따른다(rollout → scene 평균 → baseline/dual 대비 paired
부트스트랩·Wilcoxon). arm: baseline, dual, dualr, tyr_r, dualr_wl (5 × 150씬 × 2 rollout).

1. **차선 여유 분포**: `min_distance_to_lane_boundary_m` 시계열에서 중앙값·10퍼센타일·
   `<0.5 m`/`<0.2 m` 시간 비율·최소값. offroad 프레임은 제외하고 재는 것과 포함해 재는 것 둘 다.
2. **wrong_lane 시간 비율**과 offroad 에피소드 수·길이(연속 프레임 런), 이탈 시점 분포(rollout 초반 vs 후반).
3. **이탈 전조**: 각 offroad 에피소드의 t−5 s 창에서 lane margin 선형 기울기, 속도, `plan_deviation`,
   `min_ade@2.5s(gt)`. 같은 씬에서 이탈하지 않은 arm의 같은 창을 대조로.
4. **H2 상관**: 씬별 (d2gt, lane margin 10퍼센타일, offroad 여부)의 Spearman — arm 내부와 arm 간 모두.
5. **CoC 대조**: 이탈 창의 CoC 붕괴율(있다면) — wl에서 이미 낮으므로 reasoning 잔여 효과를 배제하는 확인.

## 사전 등록 게이트

- **G1**: 재구성 arm(dualr·tyr_r·wl)의 lane margin 10퍼센타일이 dual보다 유의하게 작다(씬 페어드 CI).
  통과 → offroad는 꼬리가 아니라 분포 전체의 이동이다.
- **G2**: 씬 단위 Spearman(d2gt, lane margin) < −0.3 (arm 내). 통과 → H2(GT 추종과 차선 여유의 교환).
- **G3**: 이탈 에피소드의 t−5 s 창에서 lane margin 기울기가 유의하게 음수 → 표류형(H3a);
  기울기 0 근처에서 급변 → 조작 실패형(H3b). 어느 쪽이든 보고.
- **G4 (음성 결과의 정의)**: 세 재구성 arm 사이에 일관된 차이가 없고 dual과도 구별되지 않으면,
  offroad 차이는 150씬 수준의 잡음이며(offroad 0.070–0.097 = 씬 10.5–14.6개) 폐루프 열세의 설명은
  다른 곳(예: progress·정지 행동)에 있다 — 그 경우 progress·속도 프로파일로 축을 옮긴다.

## 비용

GPU 0. alpasim venv에서 parquet 읽기 5 arm × 300 rollout ≈ 1,500 파일 → 10–20분.
코드 `experiments/head_analysis/analyze_offroad.py` (신규, `analyze_longitudinal.py` 재사용),
보고서는 기존 dualrwl 보고서에 절 추가 또는 짧은 별도 보고서.

## 결과 — G1·G2·G3 모두 음성, 원인은 offroad가 아니라 **progress(속도)**

**차선 유지(H1) 기각.** 5 arm의 차선 여유 분포가 사실상 같다 (margin p10: dual 0.157, dualr 0.157,
tyr_r 0.146, wl 0.170; <0.5 m 시간 비율 0.69–0.70; wrong_lane 0.052–0.060). dual 대비 씬 페어드
CI가 어떤 arm에서도 margin p10·median·frac<0.5 m에서 0을 배제하지 않는다. offroad **시간 비율**은
오히려 재구성 arm이 더 낮다(dualr −0.0089\* vs dual). 즉 보고된 offroad **rate** 차이(0.067 → 0.097)는
소수 씬의 이진 이벤트이지 상시 차선 여유의 차이가 아니다.

**H2(GT 추종 ↔ 차선 여유) 기각**: Spearman(d2gt, margin p10)이 −0.10 ~ −0.21로 전 arm 비슷하고
baseline이 가장 강하다 — 재구성 특유의 교환이 아니다.

**진짜 축은 progress.** 점수 = 게이트 × progress인데, dual 대비 잃은 씬을 분해하면 게이트보다
**progress에서 더 많이** 잃는다: dualr 38씬 중 게이트 14 / progress-only 24, tyr_r 34 → 11/23,
wl 31 → 13/18. 씬 페어드로도 progress_clipped_rel만 전 arm 유의하게 음수
(dualr −0.029\*, tyr_r −0.033\*, wl −0.033\*)이고 offroad·충돌은 CI가 0을 포함한다.
**게이트를 아무도 건드리지 않은 깨끗한 씬만 봐도** progress 격차가 남는다(dualr −0.025 [−0.051, −0.002],
tyr_r −0.029, wl −0.019).

기전은 **속도**다: 평균 속도 dual 8.95 m/s vs dualr 8.45(−0.50\*), wl 8.37(−0.58\*), tyr_r 8.71(−0.24\*),
baseline 8.04(−0.92\*). 정지 시간 비율도 wl +0.013\*, 이동 거리 dual 대비 −4.7 ~ −11.6 m\*.
**재구성은 모델을 무압축 쪽으로 되돌린다 — 그리고 무압축은 dual보다 느리고 폐루프 점수가 낮다**
(baseline 0.750 < dual 0.828). dual의 폐루프 우위는 "덜 조심스러워져서 더 나아간다"는 성질이고,
재구성은 그 성질을 되돌리는 것이다. 이것이 세 arm 공통 패턴의 설명이다.

게이트 판정: G1 음성, G2 음성, G3(이탈 전조: 기울기 −0.05 ~ −0.16 m/s로 표류형이나 arm 간 차이 없음),
**G4의 조건이 발동** → 축을 progress·속도로 옮겼고 그 축에서 답이 나왔다.

코드 `experiments/head_analysis/analyze_offroad.py`, 산출물 `outputs/alpasim_offroad/`
(`metrics.json`의 `decomposition`, `speed_tables.json`, `plots/offroad_margins.png`).
