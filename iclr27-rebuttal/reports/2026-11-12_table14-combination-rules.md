# Experiment: 논문 표 14 — 결합 규칙 4가지의 95% CI (0번 층 보존 전 레시피, Ada)

2026-11-12 · RTX 5880 Ada 4–7번, 22:02–23:56 KST
계획 `iclr27-rebuttal/plans/2026-11-08_comb-rules-pre-l0keep-test500.md` (11-10 · 11-12 개정)
코드 `code/build_recipes_cpu.py`(CPU 빌드), `code/check_comb_pre_l0keep_builds.py`(B0/B1), `code/run_comb_pre_l0keep.sh`(평가), `code/table14_comb_rules.py`(표), `code/analyze_comb_pre_l0keep.py`(보조 분석)
결과 `outputs/iclr27-rebuttal/table14_comb_rules/`(표 · CI · LaTeX 행), `outputs/iclr27-rebuttal/comb_pre_l0keep/`(빌드 · 런 · 로그), `outputs/iclr27-rebuttal/comb_pre_l0keep_analysis/`

## Purpose

- 논문 표 14("Combination rules with 95% confidence intervals")의 빈칸 9개를 채운다. 행은 Mean-rank · Max-raw · Mean-raw, 열은 LingoQA · Test500 · OOD-Val262다.
- 표 14의 Max-rank(MoOP) 행이 논문 본 표와 같은 설정(0번 층도 자르는 `dual_u40_v2`, Ada)이다. 그래서 나머지 행도 같은 설정으로 쟀다.

## Setup

| 항목 | 값 |
|---|---|
| 규칙 | Max-rank = `dual_u40_v2`(MoOP), Mean-rank = `dualsum_u40_v2`, Max-raw = `rawabsmean_u40_v2`, Mean-raw = `rawabsavg_u40_v2`. raw 규칙의 FM 점수는 10스텝 \|Taylor\|의 평균(`importance_stepvlm_v1`)이다. CoC 점수와 중요도 파일은 모두 `importance_v2` |
| 공통 | u40: 층마다 Q head 13개 + MLP 4,898채널 절단(0번 층 포함). 제거 파라미터 2,657,452,032개. expert · KV는 그대로 |
| 빌드 | GPU가 모두 폐루프에 쓰이고 있어 CPU에서 레시피(`--no-state`)로 만들었다. make_slim의 함수를 그대로 쓰고 1클립 스모크만 뺐다. B0: Mean-rank CPU 재빌드 = 9월 GPU 빌드(유닛 단위) PASS. B1: 세 규칙 모두 MoOP와 같은 제거량 · 층별 개수 · expert PASS |
| 기존 행 | Max-rank는 논문 런을 그대로 썼다(`dual_u40_v2_pred_test`, `_pred_oodval`, `lingo_vqa_slim_dual_u40_v2`) |
| 개루프 | Test500 500클립, OOD-Val262 262클립(rollout만, `--no-tf`). k = 8, seed 42, max_gen 256. minADE@6 · minFDE@6 |
| LingoQA | 500문항 = 100세그먼트, greedy, max_gen 128, Lingo-Judge |
| CI | 표의 Max-rank 칸과 같은 추정 방식. 개루프는 클립 부트스트랩, LingoQA는 세그먼트 부트스트랩, 둘 다 200,000회, 시드 0. 짝 차이도 같은 재표집 |
| 재현 확인 | Mean-rank Test500 재실행이 9월 런과 500/500클립 비트 단위로 같다 |

## Results (Facts)

### 표 14 (평균 [95% CI], 6.4 s)

**2026-11-13 갱신:** 논문 표가 각 세트에 minFDE@6 열을 더한 배치로 바뀌어 그에 맞췄다. 값은 같은 런 · 같은 부트스트랩(200,000회, 시드 0)이다.

| Rule | LingoQA (%) ↑ | Test500 minADE@6 (m) ↓ | Test500 minFDE@6 (m) ↓ | OOD-Val262 minADE@6 (m) ↓ | OOD-Val262 minFDE@6 (m) ↓ |
|---|---|---|---|---|---|
| Max-rank (MoOP) | 68.8 [64.4, 73.0] | 0.950 [0.869, 1.034] | 2.546 [2.276, 2.832] | 1.119 [0.958, 1.307] | 2.851 [2.362, 3.410] |
| Mean-rank | 67.0 [62.6, 71.4] | 0.960 [0.874, 1.051] | 2.607 [2.322, 2.907] | 1.160 [0.991, 1.355] | 2.967 [2.454, 3.546] |
| Max-raw | 59.2 [54.6, 63.6] | 0.974 [0.889, 1.065] | 2.629 [2.345, 2.930] | 1.173 [1.004, 1.367] | 2.961 [2.456, 3.534] |
| Mean-raw | 14.0 [10.8, 17.4] | 1.216 [1.102, 1.337] | 3.248 [2.896, 3.619] | 1.373 [1.167, 1.604] | 3.533 [2.903, 4.232] |

LaTeX 행 (논문 배치):

```
Max-rank (MoOP) & 68.8 {\scriptsize [64.4, 73.0]} & 0.950 {\scriptsize [0.869, 1.034]} & 2.546 {\scriptsize [2.276, 2.832]} & 1.119 {\scriptsize [0.958, 1.307]} & 2.851 {\scriptsize [2.362, 3.410]} \\
Mean-rank & 67.0 {\scriptsize [62.6, 71.4]} & 0.960 {\scriptsize [0.874, 1.051]} & 2.607 {\scriptsize [2.322, 2.907]} & 1.160 {\scriptsize [0.991, 1.355]} & 2.967 {\scriptsize [2.454, 3.546]} \\
Max-raw & 59.2 {\scriptsize [54.6, 63.6]} & 0.974 {\scriptsize [0.889, 1.065]} & 2.629 {\scriptsize [2.345, 2.930]} & 1.173 {\scriptsize [1.004, 1.367]} & 2.961 {\scriptsize [2.456, 3.534]} \\
Mean-raw & 14.0 {\scriptsize [10.8, 17.4]} & 1.216 {\scriptsize [1.102, 1.337]} & 3.248 {\scriptsize [2.896, 3.619]} & 1.373 {\scriptsize [1.167, 1.604]} & 3.533 {\scriptsize [2.903, 4.232]} \\
```

minFDE@6 짝 차이(규칙 − Max-rank, 6.4 s):

| Rule | Test500 | OOD-Val262 |
|---|---|---|
| Mean-rank | +0.061 [−0.058, +0.189] | +0.115 [−0.109, +0.358] |
| Max-raw | +0.083 [−0.059, +0.234] | +0.109 [−0.126, +0.359] |
| Mean-raw | +0.702 [+0.434, +0.992]* | +0.681 [+0.236, +1.171]* |

### 짝 차이: 규칙 − Max-rank (같은 클립 · 문항, * = CI가 0을 벗어남)

| Rule | LingoQA (pp) | Test500 dminADE@6 6.4 s | OOD-Val262 dminADE@6 6.4 s |
|---|---|---|---|
| Mean-rank | −1.8 [−6.6, +2.8] | +0.010 [−0.029, +0.053] | +0.041 [−0.047, +0.134] |
| Max-raw | **−9.6 [−14.6, −4.6]*** | +0.024 [−0.021, +0.074] | +0.054 [−0.033, +0.140] |
| Mean-raw | **−54.8 [−60.0, −49.4]*** | **+0.266 [+0.179, +0.362]*** | **+0.254 [+0.106, +0.415]*** |

### 전 구간 (Test500 / OOD-Val262, 규칙 − Max-rank)

- **Mean-rank:** 두 세트의 1.6 · 3.2 · 6.4 s, ADE · FDE 모두 0을 포함한다.
- **Max-raw:** 0을 벗어나는 칸은 OOD 3.2 s minFDE 하나뿐이다(+0.080 [+0.006, +0.151]).
- **Mean-raw:** 두 세트 모든 칸에서 유의하게 나쁘다. 예: Test500 1.6 s minADE +0.019\*, 6.4 s minFDE +0.702\*.
- 전체 표는 `outputs/iclr27-rebuttal/table14_comb_rules/summary.txt`에 있다.

### 보조 수치 (Test500)

| Rule | CoC 퇴화율 | 6.4 s dminADE@6 cruise / decel_stop / accel / turn | 0번 층 Q head (Max-rank와 공유) |
|---|---|---|---|
| Max-rank | 3.0% | — | 19/19 |
| Mean-rank | 3.4% | +0.017 / −0.021 / −0.042 / +0.058 | 17/19 |
| Max-raw | 5.4% | +0.038 / +0.103 / −0.032 / −0.033 | 18/19 |
| Mean-raw | 9.4% | +0.318 / −0.032 / +0.122 / +0.460 | 17/19 |

- 0번 층 Q head 비교(Max-rank 대비 차이):
  - Mean-raw는 7번 · 30번을 버리고 20번 · 29번을 남긴다.
  - Mean-rank는 7번 · 26번을 버린다.
  - Max-raw는 26번을 버린다.
- LingoQA 항상-"yes" 기준선은 37.6%다. Mean-raw는 이보다 23.6 pp 낮다.

## Analysis (Opinion)

- **표 14가 말하는 것.**
  - 순위 공간의 두 규칙(Max-rank · Mean-rank)은 세 지표 모두에서 구분되지 않는다.
  - 원래 크기(raw)로 합치면 언어를 잃는다: Max-raw −9.6 pp\*, Mean-raw −54.8 pp\*.
  - 주행은 Max-raw에서 유의하지 않고, Mean-raw에서만 크게 나빠진다.
  - 그래서 "순위 정규화가 필수"는 이 설정에서도 지지된다. "max가 mean보다 낫다"는 지지되지 않는다(Mean-rank −1.8 pp, n.s.).
- **Mean-raw의 붕괴는 결합 규칙만의 효과가 아닐 수 있다.**
  - 0번 층을 보존한 판(11-05, A100 test_2000)의 Mean-raw는 LingoQA 64.4%, 주행 +0.056\*로 Max-raw와 비슷했다.
  - 여기서는 LingoQA가 14.0%로 무너졌다. 손실은 cruise(+0.318)와 turn(+0.460)에 몰렸고, CoC 퇴화율도 9.4%로 올랐다.
  - 이것은 10-13에 확인한 "0번 층 Q head 조합에 따른 붕괴"의 모양과 같다. 같은 규칙인데 0번 층을 자르느냐만 다르다. 그래서 붕괴의 상당 부분이 Mean-raw가 0번 층에서 버린 head(7번 · 30번) 때문일 가능성이 크다.
  - 이 설명은 가설이다. 확인하려면 Mean-raw의 0번 층만 Max-rank와 같게 맞춘 arm을 하나 더 재면 된다(Ada 4장 약 40분).
- **논문 문구.**
  - 표 14는 논문 본 표와 같은 레시피라 그대로 쓸 수 있다.
  - Mean-raw 행에는 "0번 층 head 선택에 민감한 레시피에서 무너졌다"는 각주를 붙이고, 0번 층 보존판 결과(64.4%, +0.056\*)를 함께 적는 편이 공정하다.
  - 결론 "rank normalisation is essential"은 두 판 모두에서 성립한다. 다만 크기(−10 pp 대 −55 pp)는 판에 따라 크게 다르다.
