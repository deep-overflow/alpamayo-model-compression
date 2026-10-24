# 왜 traj-only와 CoC-only는 실패하고 dual은 성공하는가 — 샘플 단위 분석 (2026-09-24)

저장된 기록만 사용(GPU 0). arm: `baseline_ada` / `coc_u40_v2` / `traj_u40_v2` / `dual_u40_v2`, 같은 예산(층당 Q head 13 + MLP 4,898 제거).
open loop: test500 + OOD 1,533클립(페어드, K=8; OOD는 GT CoC teacher-forced minADE·NLL도 저장됨). closed loop: alpasim 150씬 + hard100(씬당 2 rollout).
출력: `outputs/arm_failure_cases_v1/{summary.txt,metrics.json,plots/}`, 스크립트 `paper-analysis/scripts/failure_cases/`, 그림
`paper-analysis/figures/supplementary/fig7_failure_channels.png`. 기존 상황별 분석(`outputs/failure_by_situation_v1`)의 후속.

## 한 줄 결론

두 단일 기준은 **서로 다른 채널**에서 실패한다. CoC-only는 궤적 유닛을 버려서 생기는 **행동 채널 손상**(GT CoC를 넣어 줘도 +0.31 m)에
자기 CoC가 무너지는 **언어 채널 손상**(+0.23 m, 퇴화 클립이 손해의 30–46%)이 얹히고, traj-only는 **행동 채널만**(건강한 클립에서 +0.15 m,
언어 채널 +0.03 n.s.) 잃되 감속·정지와 어려운 회전에서 잃는다. dual은 두 채널 모두 작다(+0.05 / +0.04). 샘플 단위로도 상보적이다:
CoC-only의 최악 클립에서 traj-only는 멀쩡하고, traj-only의 최악 클립은 모두가 힘든 클립인데 dual만 손해를 절반(open loop)~0(closed loop)으로 줄인다.

## 사실 (숫자)

### 1. 두 채널 분해 (OOD 1,533, 평균 d minADE [95% CI])

| arm | 전체 | 행동 채널 (GT CoC teacher-forced) | 언어 채널 (own − teacher-forced) | 건강한 CoC 클립: 행동 / 언어 | 퇴화 클립 (n): 행동 / 언어 |
|---|---:|---:|---:|---:|---:|
| CoC-only | +0.537 [+0.466, +0.614] | +0.306 [+0.263, +0.351] | +0.231 [+0.161, +0.309] | +0.302 / +0.100 | (161) +0.336 / +1.346 |
| traj-only | +0.218 [+0.168, +0.267] | +0.154 [+0.116, +0.194] | +0.064 [+0.013, +0.115] | +0.150 / +0.029 (n.s.) | (48) +0.272 / +1.131 |
| dual | +0.086 [+0.044, +0.128] | +0.046 [+0.018, +0.073] | +0.040 [−0.002, +0.083] | +0.050 / +0.026 | (39) −0.109 / +0.573 |

행동 채널을 기동별 dense minADE 대비로 보면(그림 b): traj-only는 decel_stop 39% / turn 20% / cruise 14% / accel 2%, CoC-only는 decel_stop 43% /
cruise 39% / turn 38% / accel 19%, dual은 2–10%.

### 2. 최악 클립(상위 10%)의 성격

| | traj-only 꼬리 | CoC-only 꼬리 |
|---|---|---|
| CoC 퇴화율 (test / OOD) | 8% / 16% (전체 2.6 / 3.1%) | **58% / 41%** (전체 15 / 10.5%) |
| GT CoC를 넣으면 (OOD, 중앙값) | +1.50 → +0.96 (−36%) | +2.79 → +1.15 (**−59%**) |
| 기동 | decel_stop ×1.5–1.7 (OOD p=0.002), accel ×0.6 | cruise ×1.37 (test p=0.002), decel_stop ×0.15, accel ×0.27 |
| baseline 난이도 (minADE 중앙값, 꼬리 vs 나머지) | 0.69–0.76 vs 0.46–0.47 (p≤0.001) | test 0.43 vs 0.48 (n.s.), OOD 0.69 vs 0.47 |
| GT 운동학 (test) | 초기 속도 12.7 vs 8.8 m/s (p=0.009), 누적 yaw 12.7° vs 5.6° (p=0.001) | 속도 22.9 vs 8.3 m/s (p=1e−13), 경로 148 vs 48 m, yaw 차이 없음 |
| OOD cluster / CoC 키워드 | complex intersection ×1.9, road debris ×2.4; "slow" ×1.4, "lane change" ×1.8 | cyclists ×1.9, other-longtail ×2.2; "lane change" ×2.35 (p=8e−7), "stop" ×0.46 |
| 그 꼬리에서 다른 arm (OOD 중앙값) | CoC-only +1.40, **dual +0.46** | **traj-only +0.64, dual +0.28** |
| 그 꼬리에서 다른 arm (alpasim 150 최악 15씬, d score 중앙값) | CoC-only −0.46, **dual 0.00 (67% 정상)** | **traj-only 0.00 (73%), dual 0.00 (73%)** |

겹침: Spearman(d_traj, d_coc) test +0.37 / OOD +0.45; 최악 10% Jaccard 0.11 / 0.26 (우연 0.05). 부분적으로 겹치지만(모두가 힘든 클립),
대부분은 기준 고유의 실패다.

### 3. 추론 드리프트와 행동 손상의 관계 (건강한 클립, CoC 머리 어절이 바뀐 클립 vs 같은 클립)

- traj-only: 무관 (test p=0.94, OOD p=0.72; decel_stop p=0.96 / 0.09). CoC 본문은 68–85% 바뀌고 NLL은 폭증하지만 궤적 손상은 그와 무관.
- CoC-only: 결합 (OOD 전체 +0.58 vs +0.33, p=0.018; decel_stop +1.07 vs +0.42, p=0.025).
- 손해의 분해(양의 d 합): CoC-only는 퇴화 클립(11–15%)이 30–46%; traj-only는 퇴화 클립(3%)이 14–16%, 나머지는 텍스트 상태와 무관하게 퍼짐; dual은 "같은 텍스트" 클립이 34–60%.

### 4. closed loop

| | 150씬 d score [CI], Wilcoxon | hard100 d score [CI] | 실패 양상 (rollout 비율, 150씬) |
|---|---|---|---|
| traj-only | +0.033 [−0.012, +0.081], p=0.07 | +0.036 [−0.016, +0.090], p=0.25 | at-fault 0.033 (base 0.040), offroad 0.087 (0.070), plan_dev 0.41 (0.32) |
| CoC-only | **−0.089 [−0.150, −0.029], p=0.008** | **−0.095 [−0.171, −0.018], p=0.017** | **at-fault 0.073**, offroad 0.090, **plan_dev 0.50** (hard100 0.67) |
| dual | **+0.079 [+0.036, +0.123], p=1e−4** | **+0.085 [+0.031, +0.143], p=0.002** | at-fault 0.023, offroad 0.067, plan_dev 0.36 |

- CoC-only 최악 씬: 150씬에서 at-fault 0.50(base 0.00), hard100에서 offroad 0.50·plan_dev 1.00, **빠른 직진 씬**(hard100: v 29.3 vs 10.6 m/s p=0.005, yaw 16° vs 82° p=0.011). traj-only는 그 씬들에서 정상(73–80%).
- traj-only 최악 씬(150): at-fault 0.30·offroad 0.30, **빠르고 회전이 크고 어려운 씬**(hard_score 0.94 vs −0.59 p=0.04, v 12.9 vs 7.2, yaw 13° vs 2.4°). CoC-only도 거기서 −0.46, dual은 0.00.
- 최악 씬 겹침: Jaccard 0.15 (150) / 0.05 (hard100) — 거의 서로소.

### 5. 유닛 수준 (calib_100 클립별 gradient × 출하 유지집합)

- `I_traj`의 클립당 가중은 turn 2.8×, decel_stop 0.42×, cruise 0.64× (FM loss가 움직임 크기에 비례; 10클립이 질량의 49%). `I_CoC`는 0.9–1.2×로 균등.
- traj arm의 선택은 turn 클립 중요도에는 거의 최적(제거 질량이 자기 순위 하한보다 Q 0.7–1.3 pp 위)이지만 decel_stop 중요도에는 덜 최적(2.7 pp Q, 3.5–6.3 pp MLP 위).
- **그런데 dual도 decel_stop 궤적 질량을 traj arm만큼 제거한다**(Q 0.172 vs 0.172 trunk, 0.210 vs 0.198 late). dual이 decel_stop을 고치는 것은 궤적 유닛을 더 남겨서가 아니다. dual과 traj arm의 차이는 `I_CoC` 질량이다: late에서 traj arm은 `I_CoC` 질량의 29.6%(Q)/34.0%(MLP)를 제거하고 dual은 19.7%/29.4%; trunk 22.7% vs 19.7%.
- 반대쪽: CoC arm은 `I_traj` 질량을 late 29–34%, trunk 18–20% 제거(dual 20–28% / 16–18%).

## 해석

1. **CoC-only가 실패하는 이유.** 첫째, `I_CoC`로 자르면 궤적 유닛이 떨어져 나간다(late `I_traj` 질량 29–34% 제거). 이 손상은 GT CoC를 넣어 줘도 남는 행동 채널 손상(+0.31 m)이고 네 기동 모두에 걸친다. 둘째, teacher-forced NLL로 고른 유닛은 "좋은 prefix 다음 토큰"을 지키지만 자유 생성의 종료·자기일관성은 지키지 못해(인과 리포트: 종료 실패 8–13%, `<|cot_end|>`는 gradient의 5.5%) 10–15% 클립에서 CoC가 무너지고, 무너진 CoC는 cache V를 통해 expert로 들어가 궤적을 망친다(퇴화 클립 +1.68 m의 80%가 언어 채널). 이 붕괴는 빠른 직진·차선 변경 클립에 몰려서(cruise ×1.37, lane change ×2.35, hard100 최악 씬 v 29 m/s) closed loop에서 plan deviation과 at-fault 충돌로 나타난다.
2. **traj-only가 실패하는 이유.** 언어는 드리프트하지만(NLL +240%, 텍스트 68–85% 변경) 거의 무너지지 않고(3%), 궤적 손상은 추론 드리프트와 무관하다. 즉 순수한 행동 채널 손상이고 감속·정지와 빠르고 어려운 회전에 집중된다. 유닛 수준에서 `I_traj`는 소수 turn 클립이 지배하는 편향된 추정(turn 2.8×, decel_stop 0.42×)이라 decel_stop 유닛 선택이 덜 최적이지만, **dual도 그 유닛을 더 남기지 않으므로** 이것만으로 dual의 회복을 설명할 수 없다. dual이 추가로 남기는 것은 `I_CoC`가 높은 유닛이고, 이 유닛들이 GT CoC teacher-forcing 아래에서도 궤적을 지킨다는 것은 **궤적 기능이 언어 쪽 점수가 높은 유닛에 의존한다**는 뜻이다(expert가 읽는 cache의 CoC·text 자리 표현을 만드는 유닛). 인과 리포트의 trunk-only 결과(traj-only trunk FM +26.1% / minADE +2.64 m vs dual trunk +4.8% / +0.06 m)가 같은 방향이다. 1차 Taylor 점수는 FM gradient가 작은 자리(trunk text 몫 7%)의 유닛을 낮게 보지만 기능은 거기에 걸려 있다.
3. **dual이 성공하는 이유.** 합집합은 각 목적함수의 상위 유닛을 남기고, 각 목적함수의 상위 유닛에는 **다른 기능이 필요로 하지만 다른 점수는 볼 수 없는 유닛**이 들어 있다: `I_traj` 상위에는 CoC-only가 버리는 궤적 유닛(행동 채널), `I_CoC` 상위에는 traj-only가 버리는, 언어를 지키고 궤적도 떠받치는 유닛(언어 채널 + 행동 채널의 일부). 그래서 dual은 CoC-only의 꼬리(빠른 직진, CoC 붕괴)에서 traj-only처럼 멀쩡하고, traj-only의 꼬리(어려운 감속·회전)에서 유일하게 손해를 절반~0으로 줄이며, closed loop에서 두 세트 모두 baseline을 이기는 유일한 arm이다.

## 검증 제안 (GPU 필요, 승인 시)

`run_token_ablation.py`로 held-out 100클립에서 두 유닛 집합만 제거: (i) trunk에서 dual은 남기고 traj arm은 버린 유닛(`I_CoC` 상위), (ii) dual은 남기고 CoC arm은 버린 유닛(`I_traj` 상위). 예측: (i)는 GT CoC teacher-forcing 아래에서도 decel_stop·turn FM/minADE를 망치고 NLL도 올린다, (ii)는 NLL은 거의 안 움직이고 FM/minADE만 망친다. 이게 맞으면 해석 2·3이 인과적으로 닫힌다.


## 검증 1차 결과 (2026-09-24 실행): dense 모델에서는 가설 (A)가 성립하지 않는다

플랜 `plans/2026-09-24_dual-saves-trunk.md`, 출력 `outputs/dual_saves_trunk_v1/summary.txt`, 브랜치 `worktree-dual-saves-trunk`.
held-out 100클립(indist_500 앞 100개, dense 모델의 rollout을 teacher-forcing)에서 trunk(0–21층)의 `Straj`(dual은 남기고
traj arm은 버린 유닛, 층당 Q head 2.1개 + MLP 채널 709개)를 dense 모델에서 제거했다. 대조군은 궤적 점수가 같고 CoC 점수만
낮은 `MStraj`와 크기 맞춤 무작위 3개.

| 제거한 집합 (두 축 동시, 0–21층) | dFM (dense 대비) | dminADE | dNLL |
|---|---:|---:|---:|
| Straj (dual 남김, traj 버림) | +0.5% [−0.4, +1.3] | −0.032 [−0.092, +0.015] | **+4.5% [+2.3, +6.8]** |
| MStraj (궤적 점수 맞춤 대조군) | +0.3% | −0.022 | +1.2% |
| 무작위 ×3 | +0.5–2.4% | −0.02–+0.03 | +1.3–2.7% |
| Scoc (dual 남김, CoC 버림) | +0.2% | −0.006 | +0.6% |

- 축별(Q만 / MLP만, 0–21과 6–21 두 밴드)로도 같다: `Straj`의 dFM +0.2–0.3%, dminADE CI가 0 포함, 무작위 집합이 오히려 FM을 더 망친다(Q 최대 +2.7%).
- 움직이는 것은 언어뿐이다: 두 축 동시 `Straj`의 NLL +4.5%는 대조군(+1.2%)과 무작위(+1.3–2.7%)보다 크다(p = 0.016).
- 사전 등록 판정 G1–G4 전부 FAIL.

**해석.** dense 모델 안에서는 dual이 traj arm보다 더 남기는 trunk 유닛이 언어 유닛이고 측정 가능한 궤적 기능이 없다. 이는 그
유닛들의 1차 궤적 점수가 낮다는 사실과 일치한다(1차 점수는 dense 모델에서의 한계 효과를 맞게 예측했다). 따라서 위 "해석 2"의
(A) — 궤적 점수가 못 보는 궤적 유닛 — 은 dense 맥락에서는 기각이다. 남은 가능성은 두 가지다. (i) 그 유닛의 기능은 arm의
다른 유닛들이 잘려 나간 뒤에야 드러난다(중복 파트너 효과; 2026-09-21 라운드에서 엣지를 하나씩 막으면 대체 가능한 엣지를
놓친다고 확인한 것과 같은 구조). trunk-only 밴드 결과(traj-only trunk FM +26.1% vs dual trunk +4.8%)에서 두 trunk 선택은
`Straj` ↔ (traj는 남기고 dual은 버린 유닛)의 교체만 다르므로, 이 교체가 그 차이를 만드는지는 **pruned 맥락의 add-back**
(`trajtrunk+Straj` vs `trajtrunk+MStraj`)으로 바로 잴 수 있다. (ii) 그 유닛은 어디서도 궤적 기능이 없고 dual의 주행 우위는
언어 보호와 trunk–late 상호작용에서 온다. 어느 쪽이든 add-back 결과로 닫힌다(플랜 Part 2, 승인 대기).


## 검증 2차 결과 (2026-09-25, add-back): 유닛의 정체는 주행에 중요하지 않고, 단일 기준의 선택 자체가 취약하다

출력 `outputs/dual_saves_addback_v1/summary.txt`. 같은 held-out 100클립에서 단일 기준 arm 안에 유닛을 되살렸다. 되살린 수는 전부 같다
(+46 head, +15,592 채널 = arm이 trunk에서 버린 것의 약 1/10). dense 대비 페어드 손해 (FM 상대 / minADE / NLL 상대):

| 맥락 | arm 그대로 | + S (dual이 남긴 유닛) | + M (점수 맞춤 대조군) | + 무작위 |
|---|---|---|---|---|
| traj arm (출하본) | +6.6% / +0.061 / +241% | +3.1% / −0.003 / +92% | +2.5% / −0.011 / +132% | +2.8–4.0% / −0.03–+0.02 / +114–143% |
| traj trunk-only | +26.1% / +2.64 / +51% | +2.7% / +0.003 / +10.5% | +2.1% / −0.017 / +22.9% | +3.4% / +0.012 / +23.0% |
| CoC arm (출하본) | +15.8% / +0.564 / +67% | +2.8% / +0.033 / +37% | +3.9% / +0.086 / +16% | +3.0–4.1% / +0.03–0.05 / +37–51% |
| CoC trunk-only | +17.0% / +0.459 / +28% | +2.4% / +0.021 / +12.6% | +3.9% / +0.267 / +5.4% | +3.5% / +0.026 / +12.5% |

- **A1·A2 FAIL (두 맥락 모두).** `Straj`를 되살려도 FM·minADE 회복이 대조군·무작위와 같다 (FM vs 대조군 p = 0.76 / 0.78, minADE p = 0.56 / 0.77).
- **A3: `Straj`는 언어 집합이다.** NLL 손해를 62%(arm) / 79%(trunk-only) 회복하는데 다른 집합은 45–55% (p = 7.5e−15 / 1.2e−8).
- **A4 FAIL.** `Scoc`는 trunk-only에서만 대조군보다 FM을 더 회복하고(p = 0.0028) 무작위보다는 0.01 기준으로 못 이긴다(p = 0.033).
- **가설을 대체하는 발견.** 어떤 유닛이든 trunk에서 버린 것의 1/10만 되살리면 단일 기준 arm의 주행 손해가 거의 사라진다: traj trunk-only FM
  +26.1% → +2–3%, minADE +2.64 → 0.00 (Straj·대조군·무작위 모두), CoC trunk-only +17.0% → +2–4%; 출하본에서는 FM 손해의 절반(traj)~4/5(coc)와
  minADE 손해의 거의 전부. 즉 단일 기준의 주행 손해는 특정 유닛을 잃어서가 아니라 **선택 전체의 취약성**(절벽)이고, dual의 선택은 같은
  층별 예산에서 절벽 밖에 있다(trunk-only FM +4.8%, `armheld_v1`). 살아남은 Q head가 없는 KV 그룹 수로는 설명되지 않는다
  (trunk 0–21층: traj 19, CoC 12, dual 13).

**해석 갱신.** 위 "해석 2"의 (A)는 dense 맥락(1차)과 pruned 맥락(2차) 모두에서 기각이다. 논문에 쓸 수 있는 문장은 두 가지다.
(a) dual이 traj arm보다 더 남기는 trunk 유닛은 언어 유닛이고, 그것을 되살리는 것이 traj arm의 언어 채널을 고친다 — 인과적으로 확인됨.
(b) 행동 채널에서 합집합은 "다른 점수가 보는 특정 유닛"을 구하는 것이 아니라, 단일 선택이 취약한 곳에서 **취약하지 않은 선택**을 만든다.
단일 선택이 왜 취약한지(임의의 유닛 1/10을 되살리면 무엇이 풀리는지)는 열린 문제다. 후보: 단일 점수는 층마다 같은 종류의 유닛을
잘라 손상이 깊이에 걸쳐 누적되는 반면 합집합은 종류를 섞는다 — 층별 add-back이나 깊이별 누적 측정으로 검증 가능.
