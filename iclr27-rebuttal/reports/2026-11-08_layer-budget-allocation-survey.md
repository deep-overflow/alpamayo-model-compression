# 레이어 예산 배분 실험 정리 — dual 기준에서 균일 배분을 이긴 방식이 있었나

2026-11-08 정리 (조사는 2026-11-07) · 새 실험 없음, 기존 계획서·보고서의 수치를 원문과 대조해 모음

## Purpose

MoOP(`dual_u40_v2`)는 모든 층에서 같은 수를 자른다(층당 Q head 13개 + MLP 4,898채널, 전체 24.0%). 이 배분을 정한 근거와,
이후 다른 배분 방식을 시험한 결과를 한곳에 모은다. 리뷰에서 "왜 균일 배분인가"를 물으면 쓸 근거다.

## Setup

- 범위:
  - 층(깊이) 사이의 예산 배분: 고정 배분 프로파일, Tyr 탐색 배분, 범위 제한·예외
  - 층 안의 head↔MLP 분할(관련 축)
  - 연속 층 제거
- 판정 기호:
  - 개루프는 minADE@6 짝 차이(m)이고 **+이면 나쁨**이다.
  - 폐루프는 AlpaSim scene score 차이이고 **+이면 좋음**이다.
  - `*`는 95% CI가 0을 포함하지 않는다는 뜻이다.
- 대부분은 출하본(0번 층 head를 자르는 판) 기준이다. 0번 층 보존판(`l0keep`)이 나온 뒤의 실험은 따로 표시했다.

## Results (Facts)

### A. 고정 배분 프로파일

**A1. 기준 × 배분 2×4 격자** (2026-07-29, `reports/archive/2026-07-29_criterion-vs-allocation-grid.html`, `outputs/grid_v1`)
- 설정: 제거 39.8–40.0%. 개루프 7월 split val 50클립 × K=8. 중요도는 7월 50클립 세트(`importance_v1`).

| 배분 | CoC 퇴화 traj / dual | minADE 중앙값 Δ traj / dual |
|---|---|---|
| uniform | 0.080 / **0.060** | +0.187 / +0.098 |
| late (후반 집중) | **0.820 / 0.300** | +0.604 / +0.106 |
| agree (목적 불일치 비례) | 0.120 / 0.040 | +0.146 / +0.179 |
| depthprior (깊이 기울기) | 0.160 / 0.120 | +0.327 / +0.092 |

- 주 효과: 배분 +0.500, 기준 +0.165(CoC 퇴화). CoC 붕괴의 원인은 기준이 아니라 후반 집중 배분이다.
- 규칙: "후반 층에 예산을 몰지 말 것". agree와 depthprior도 uniform보다 낫지 않다.

**A2. 질량 임계 배분 dualmass** (2026-09-11, `plans/2026-09-11_mass-threshold-allocation.md`, 개루프, 출하본 대비)

| arm | val500 | test500 | OOD-Val |
|---|---|---|---|
| dualmass | +0.1989 [+0.110, +0.295]* | +0.3358 [+0.230, +0.465]* | +0.2868 [+0.149, +0.445]* |
| dualmass_f25 | +0.0509 [+0.008, +0.096]* | +0.1064 [+0.046, +0.171]* | +0.0575 [−0.027, +0.140] |

- CoC 퇴화: 출하본 1.4/2.6/5.7% → dualmass 9.8/13.6/11.5%.
- 판정: "uniform이 이긴다".

### B. 탐색 배분 (Tyr 계열)

| 실험 | 결과 | 출처 |
|---|---|---|
| Tyr 탐색 − Tyr 균일 (재구성 포함) | test −0.010*, val −0.006*, OOD −0.016* | `plans/2026-08-20_tyr-baseline.md` |
| **dual + 전역 배분 탐색(dualg) / dual + Tyr 배분 이식** | test 0.978 (+0.001) / 0.958 (+0.002). 판정 "전역 할당 탐색은 dual에 이득 0" | `plans/2026-08-21_dual-global.md` |
| Tyr 공식판(tyr_off)의 배분 | 층당 Q 4–31개(29번 층 4, 0번 층 19, 15·17·24번 층 31). 1–3번 층 MLP 862–2,206채널 | `cvpr27/reports/2026-11-01_tyr-official-a100.md` |
| tyr_vt (Hessian V:T 1:1) | Q head가 0–8층 184→161, 9–17층 225→185, 18–26층 172→205, 27–35층 103→133으로 이동. LingoQA +24.8 pp*, 주행 +0.049*. 배분 이동이 원인인지는 가설 | `cvpr27/reports/2026-11-02_tyr-vt-equal-hessian.md` |
| 폐루프 Tyr − dual | Origin150 −0.0419 [−0.0848, −0.0004]* (p=0.0989, 경계). s600에서는 Tyr 논문 행 − MoOP +0.017 n.s. | `plans/2026-08-20_tyr-baseline.md`, `cvpr27/reports/2026-11-03_tyr-u40-r-closedloop-s600.md` |

- tyr_off는 선택·재구성까지 MoOP와 달라서, 배분 효과만 분리할 수는 없다.

### C. 범위 제한과 예외

| 실험 | 결과 | 출처 |
|---|---|---|
| **범위 맞춤 dual**: 4–33층만, 층당 head 15 + MLP 5,928/5,929 | 출하본 대비 val +0.0236*, test +0.0389*, OOD +0.0379 (n.s.). 폐루프 미실행 | `plans/2026-08-22_scope-matched-dual.md` |
| **0번 층 Q 보존 (l0keep)** | arm 간 SD 0.337 → 0.015. 출하본 대비 +0.006 n.s., 예산 보충 비용 +0.004 n.s. s600 폐루프 +0.008 n.s. | `reports/evaluation/2026-10-13_layer0-exemption.html`, `cvpr27/reports/2026-11-02_s600-closedloop-all-arms.md` |
| **1–4번 층 MLP 추가 절단** (l14m95/90/80, 예산 +2.3–3.0%p, l0keep 위) | test_2000 +0.006 / +0.010 / +0.016 (n.s.). s600 폐루프 −0.017 / −0.002 / −0.014 (n.s.). LingoQA −3.6 / −4.0 / −6.0 pp | `cvpr27/reports/2026-10-30_l0keep-l14-mlp-cut.md`, `2026-10-31_l14m-closedloop-s600.md` |
| **깊이 인지 dual (dualv3)**: 22층부터 CoC 점수만 | test500 +0.0430 [+0.0006, +0.0873]*, val500 +0.0869 [+0.0398, +0.1430]*, CoC 퇴화 6.8 / 7.0% | `reports/evaluation/2026-10-15_raw-max-criterion.html` |

- 1–4번 층을 더 자를 수 있는 배경: 1–4번 층 MLP는 Taylor 질량이 극도로 몰려 있다(Gini 0.89–0.97). 균일 절단은 그 층들 질량의 0.1–1.8%만 지운다(`cvpr27/reports/2026-10-29_vlm-mlp-layer-concentration.md`).

### D. 층 안의 head ↔ MLP 분할 (관련 축)

| 실험 | 개루프 | 폐루프 | 출처 |
|---|---|---|---|
| qcut4 − maxstep11 (max11 기준) | test500 −0.0223 [−0.0373, −0.0081]*, CoC 퇴화 0.026 → 0.004 | — | `reports/evaluation/2026-09-06_axis-allocation.html` |
| **dual+h4** (같은 24%, head 4개만 자르고 나머지는 MLP) | val −0.0387, test −0.0390, OOD −0.0848 (모두 n.s.), CoC 퇴화 0% | Origin150 **−0.091 [−0.134, −0.050]*** (p=3.1e−6), offroad 6.7 → 9.3% | `reports/evaluation/2026-09-10_head-vs-mlp-budget.html` |
| q13m1861 (u20 예산 + head 13개) | — | u20 대비 +0.0660*, u30과 +0.0136 n.s. | `reports/evaluation/2026-09-12_mlp-width-dissociation.html` |

- 폐루프 점수를 정하는 것은 예산보다 자른 head 수다. head↔MLP 재배분은 개루프만으로 판정할 수 없다.

### E. 연속 층 제거 (무압축 모델, 층 우회, teacher forcing)

- 출처: `reports/evaluation/2026-10-15_interval-landscape.html`
- 8–16층 우회: minADE +1.743 m(3층 블록 합의 3.59배). 26–34층 우회: minADE +0.022 m지만 CoC NLL +4.892 nat.
- 두 출력이 모두 버티는 가장 긴 창은 4층이다.
- 가까운 층끼리는 손상이 가산적이지 않아서, 이어진 구간 제거는 층별 점수 합으로 추정할 수 없다.

## Analysis (Opinion)

1. **dual 기준에서 균일 배분을 이긴 배분 방식은 없다.** 후반 집중은 CoC를 무너뜨리고, 질량 배분·범위 제한·깊이 인지 dual은 개루프가 유의하게 나쁘다. 전역 탐색은 이득이 0이다.
2. **쓸모가 확인된 변형은 예외 하나다.** 0번 층 Q head를 모두 남기는 `l0keep`은 비용 없이 보정 데이터에 따른 붕괴 분산을 없앤다. 1–4번 층 MLP 추가 절단은 주행에는 공짜지만 LingoQA를 잃는다.
3. **Tyr의 탐색 배분은 재구성과 함께일 때만 약간 이득이다.** dual 선택에 이식하면 효과가 없다. 그래서 MoOP와 Tyr의 차이는 배분보다 선택 기준과 재구성에서 온다.
4. **head↔MLP 분할은 개루프와 폐루프가 반대로 나온다.** 리뷰 답변에서 "MLP를 더 자르면 된다"는 주장은 피해야 한다.

**주의할 점**
- **`dual_w4`/`dual_w8`은 배분 실험이 아니다.** 프루닝에 4/8-bit 가짜 양자화를 더한 체크포인트다(`outputs/slim_dual_u40_w{4,8}/config.json`의 `"kind": "slim + baked fake quantization"`). 09-10 폐루프 표 스크립트(`experiments/evaluation/fill_alpasim_all_report.py:34-35`)가 "층당 Q 4/8개 제한 배분"이라고 잘못 표기했다.
- **0번 층 재확인이 없다.** A1 격자와 A2 dualmass는 0번 층 head 선택 혼동(2026-10-13 발견)을 재확인하지 않았다. 배분마다 0번 층에서 자르는 head 수가 다르므로, 일부 차이가 그 효과일 수 있다.
- **A1 격자는 파일럿 중요도로 했다.** 7월 50클립 세트(`importance_v1`)를 썼다(`2026-11-08_reviewer-q3-pilot-calib50.md`).
- **폐루프 근거가 적다.** 폐루프까지 본 배분 실험은 dual_uniform, Tyr, l0keep, l14m, dual+h4뿐이다. 패치 전·후 하네스가 섞여 있다.
- **제안만 하고 안 돌린 실험:** 예산을 고정한 후반→초반 재배분(`cvpr27/plans/2026-10-30_followup-experiments.md` E1).
