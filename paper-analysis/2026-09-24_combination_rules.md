# CE·FM 중요도 결합 규칙 — rank-max의 대안 실험 정리

2026-09-24. 출하 기준 `dual` = `max(rank_ℓ(I_traj), rank_ℓ(I_CoC))`(층내 rank 정규화 후 최댓값, 예산 uniform 0.3985632694,
VLM Q head·MLP 채널만). 아래는 같은 예산·같은 calibration(`importance_v2`, `calib_100`)·같은 축에서 **결합 규칙만** 바꾼
arm들의 결과다. 모두 이미 측정된 것이며 새 실험은 없다.

## 1. 결합 연산자 (2개 손실, 같은 항)

| 규칙 | 정의 | dual과 유지집합 겹침 (Q / MLP) | 개루프 ΔminADE@6 vs dual (중앙값 [CI]) | 판정 |
|---|---|---|---|---|
| **max (dual, 출하)** | max(rank traj, rank CoC) | — | — | 기준 |
| rank-sum (= rank-mean, Borda) `dualsum` | rank traj + rank CoC (2항이라 rank-mean과 유지집합 동일) | 91.4% / 93.6% | test500 **−0.0010 [−0.0105, +0.0069]** — 동급 | max 유지 (비열등, 우세 아님) |
| rank-product `dualprod` | rank traj × rank CoC | 88.5% / 90.2% | test500 **+0.1423 [+0.0998, +0.1769]**\*, FDE +0.485\* | **기각** |
| z-score 평균 `meandual` | mean(z traj, z CoC) — rank가 아니라 z-score 정규화 | — | val500 vs dualfix **+0.0058 (p=0.50)** — 동급 | 동급 |
| raw 합 | I_traj + I_CoC | traj-only와 95.2% / 95.6% 일치 | (arm 불필요 — `traj_u40_v2`가 곧 그 결과) | **traj 단일 기준으로 붕괴** |
| raw 곱 | I_traj × I_CoC | — | 35층에서 I_traj ≡ 0이라 선택 무정의 | **제외** |

- rank-sum·rank-product가 max와 갈리는 unit은 정확히 준-specialist다: max가 지키고 sum이 버리는 unit의 (좋은 rank, 나쁜 rank) = (0.63, 0.13), 반대는 (0.51, 0.42) — 즉 sum/product는 한쪽에서만 상위인 unit을 양쪽 중간 unit으로 바꾼다. product는 그 교환이 더 강해(0.68, 0.13 ↔ 0.50, 0.39) 손해가 유의하게 난다.
- raw 스케일은 FM이 CE의 중앙값 11.7배(Q) / 11.9배(MLP)라 rank 정규화 없이는 어떤 결합도 traj 단일 기준이 된다. **rank 정규화가 "둘 다 쓴다"를 성립시키는 장치**다.
- CoC 퇴화율: dualsum 3.4%, dualprod 4.8% (dual 6.0%, 2×dual 미만).

## 2. 항의 수 × 연산자 (2×2: max/mean × 2항/11항)

11항 = CoC + FM 10 denoising step 각각.

| 규칙 | val / test / OOD-val ΔminADE vs dualfix | 판정 |
|---|---|---|
| max · 2항 (dualfix) | 기준 | |
| mean · 2항 (`meandual`) | +0.0058 (p=0.50) | 동급 |
| max · 11항 (`maxstep11`) | −0.0033 / −0.0115 / +0.0011 (전부 n.s.) | 동급 — **max에 항을 더 넣어도 무해** |
| mean · 11항 (`znorm11`) | **+0.14 ~ +0.20** (dual_ada 대비; 프루닝 자체 비용의 2.5–3.3배) | **기각** |

- 어느 요인 단독도 해가 없는데 둘을 합치면 +0.173 → 순수 상호작용. 기전: FM 10 step은 서로 0.92로 비슷해 평균에서는 공유 성분이 10/11의 무게를 가져 CoC 몫이 1/2 → 1/11로 희석되고, max는 가중이 없어 중복 항이 CoC에 비용을 지우지 않는다.
- 주의: 이 2×2의 max 행은 rank_norm, mean 행은 z-score라 "mean vs z-score"는 아직 분리되지 않았다(`mean of 11 rank_norm` arm이 필요).
- maxstep11의 초판 "꼬리 이득" 주장은 dualfix 자신의 점수로 부분집합을 고른 선택 편향이라 철회됨.

## 3. 균형 노브 (동등 가중이 임의적인가)

가법형 `max(rank traj, rank CoC − δ)` (δ=+1/0/−1이 traj/dual/coc를 정확히 재현).

| δ | val500 / test500 / OOD-val ΔminADE vs dual | 판정 |
|---|---|---|
| +0.10 (`dualfm10`) | +0.0001 / −0.0023 / +0.0086 (여섯 CI 전부 0 포함) | 구별 불가 |
| +0.20 (`dualfm20`) | val +0.0102 [+0.0005, +0.0240] (p=0.04), test·OOD 미실행 | 미재현 |

- n=100에서 노브는 학습 불가: 서로소 추출 11개의 잡음 바닥이 유지집합의 9.0% / 14.6%인데 |δ|=0.1이 움직이는 것은 5.2% / 3.8%뿐이고, leave-one-out 최적 δ의 최빈값이 양 축 모두 **0**. 잡음은 n^−0.43으로 줄어 Q는 n≈460, MLP는 n≈3,000에서야 노브가 잡음을 넘는다.
- 곱셈형 λ 파라미터화는 다이얼이 아니라 스위치(rank가 균등분포라 λ∈[0.4, 0.6] 밖에서 포화).
- 결론: δ=0(동등 가중)이 경험적으로 지지되는 기본값.

## 4. 항 자체를 바꾸거나 더하기

| 변형 | 결과 | 출처 |
|---|---|---|
| CoC 항을 J-lens(rollout 불필요)로 교체 `j_traj` = max(rank traj, rank J) | 개루프 val: dual보다 +0.0234 (중앙값), 평균 +0.042 [+0.014, +0.069], p=1.7e−6 — 약간 나쁨; 폐루프 150씬 0.791 (dual 0.828), baseline 대비 +0.042 [−0.005, +0.088] n.s. | `2026-08-09_criterion-oneshot`, `2026-08-11_criterion-closedloop`, `alpasim-all` |
| 세 번째 max 항 추가 (J-lens) | 2항 유지집합의 12.7% / 13.7%를 옮김(FM 10 step은 7.0% / 9.7%); 변위 크기는 성능을 예측하지 못하므로 그 자체로는 판단 근거 아님 | `2026-09-08_criterion-augmentation` §C |
| 클립 재가중 `dualsafe` (w = exp(−d/5)) | val500 **+0.1945 (p=4.7e−32)** — 유효표본 100→70.8 축소 효과, 다른 100클립 추출과 구별 불가 | `2026-09-08_criterion-augmentation` |
| 분산 가중 `dual2nd` (E[g²]) | test500 **+0.027 (p=0.00063)** — 곡률이 아니라 클립 간 분산(점수의 69–81%)이 선택을 움직임 | CLAUDE.md, LLM-Pruner 절 |
| 상수 half 가드 `dualfix` (35층 I_traj ≡ 0 → −inf) | 세 세트에서 정확히 0 (85–87% 클립 CoC 비트 동일) | `2026-09-03_criterion-aggregation` |

## 5. 논문에 쓸 수 있는 요약

1. **rank 정규화는 필수**: raw로 합치면 FM 스케일(≈12×)이 CE를 지워 traj 단일 기준이 된다(95% 일치).
2. **연산자는 max ≈ sum ≈ mean(2항), product만 나쁘다**: 준-specialist(한쪽만 상위)를 보존하는 규칙은 동급이고, 양쪽 중간을 선호하는 product는 +0.14.
3. **max의 장점은 항을 더해도 무너지지 않는 것**(maxstep11 n.s. vs znorm11 +0.17) — 합집합 성질.
4. **동등 가중은 n=100에서 튜닝 불가이며 최빈 최적값이 0** — 임의 선택이 아니다.
5. 두 번째 항을 rollout-free J-lens로 바꿀 수는 있으나 CoC-NLL이 약간 낫다(개루프 +0.023, 폐루프 −0.037 절대).

이 결과들이 말하지 않는 것: "max가 최적"(동급 규칙이 둘 있다), "3항 이상이 유익"(무해할 뿐), 폐루프에서의 연산자 비교(dualsum·dualprod·meandual·znorm11은 개루프만 측정).

## 출처

- 연산자 ablation: `plans/2026-08-20_combination-operator-ablation.md`, `reports/evaluation/2026-08-20_openloop-results-tables.html` §3, `experiments/head_analysis/compare_combine_ops.py` (CPU, importance_v2), `outputs/dual{sum,prod}_u40_v2_test`
- 2×2: `reports/evaluation/2026-09-04_union-step-criterion.html`, `2026-09-03_criterion-aggregation.html`; `outputs/{meandual,maxstep11,znorm11,dualfix}_u40_v2_*`
- 균형 노브: `reports/evaluation/2026-09-09_criterion-balance.html`, `2026-09-09_balance-openloop.html`; `outputs/balance_stability`, `outputs/balance_openloop`
- J-lens·세 번째 항·dualsafe: `reports/evaluation/2026-08-09_criterion-oneshot.html`, `2026-09-08_criterion-augmentation.html`, `2026-09-10_alpasim-all.html`

## 6. 왜 rank-sum(= rank-mean)이 max와 동급인가 (2026-09-24 추가)

출하 예산에서 세 규칙의 유지집합을 CPU로 비교했다(`importance_v2`). "강한 specialist" = 한쪽 rank ≥ 0.9, 다른 쪽 ≤ 0.2:
Q head 8개, MLP 채널 2,480개.

| 규칙 | 강한 specialist 탈락 (Q / MLP) | 컷 임계값 | max가 지키고 이 규칙이 버리는 unit의 좋은 쪽 rank |
|---|---|---|---|
| max | 0 / 0 | max-rank ≈ 0.55 | — |
| rank-sum | 1 / 0 | 평균 rank ≈ 0.42 | 중앙값 0.61–0.62, 최대 0.86–0.90 |
| rank-product | 4 / 1,652 | 곱 ≈ 0.15 | 중앙값 0.67–0.68, ≥ 0.9가 1,652개 |

지켜진 중요도 질량은 max와 sum이 0.5 pp 안에서 같다(Q late `I_traj` 0.804 vs 0.802, `I_CoC` 0.801 vs 0.805; MLP 0.724 vs 0.727,
0.706 vs 0.710). 이유: 60%를 남기면 평균 rank 컷이 ≈0.42라 (0.9, 0.0)의 평균 0.45도 통과한다 — 2항에서는 sum도 합집합
성질을 사실상 가진다. 두 규칙이 갈리는 unit은 컷 경계의 약한 semi-specialist(0.6, 0.1) ↔ 범재(0.5, 0.4)뿐이라 질량도
기능도 같다. product는 곱이라 한쪽 0.15가 다른 쪽을 지워 강한 specialist의 2/3를 버린다(+0.14 m의 원인; 잃는 질량은
2 pp뿐이라 질량으로는 설명되지 않는다).

**일반화.** sum ≈ max는 2항 + 넉넉한 예산의 산물이다. 항이 11개면 한 항의 specialist는 평균 1/11이라 탈락한다
(znorm11 붕괴 / maxstep11 무해의 이유). 남기는 비율이 ~45% 아래로 내려가면 평균 컷이 0.5를 넘어 (0.9, 0.0)도 떨어지므로
sum과 max가 갈릴 것이다(미측정). max의 장점은 "2항에서 sum보다 낫다"가 아니라 **항 수·예산에 무관하게 specialist를
지킨다**는 것이다.
