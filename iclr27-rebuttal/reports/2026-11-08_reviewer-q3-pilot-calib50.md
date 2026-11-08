# 리뷰 질문 3 — "earlier 50-clip calibration set"(test 클립 12개 포함)이 무엇에 쓰였나

2026-11-08 정리 (조사는 2026-11-07) · 새 실험 없음, 저장된 config·매니페스트·`slim_meta.json` 대조
관련 부록: 논문 Appendix A.3.1 (p.14) — "We exclude 12 test clips belonging to an earlier 50-clip calibration set"

## Purpose

리뷰 질문 3:

> Which decisions, if any, were made using the earlier 50-clip calibration set that contained 12 test clips—including the
> architecture, scoring formulation, unit types, pruning ratios, or hyperparameters? If any final design choice used those
> clips, can the corresponding selection and final evaluation be repeated without test-set exposure?

이 문서는 세 가지를 확인한다. 50클립 세트가 무엇에 쓰였는가. 최종 모델과 평가가 그 세트에 노출됐는가. 그 세트로 정한
설계 결정은 무엇인가.

## Setup (확인한 근거)

| 근거 | 경로 |
|---|---|
| 50클립 세트 정의 | `outputs/split.json["calib"]` (7월 자체 split, `notebooks/clip_ids.parquet` 1,181클립, seed 20260721: train 900 / val 131 / test 150 / calib 50 = train 앞 50개) |
| 공식 split 구성 | `experiments/evaluation/SAMPLING.md` 6절: 공식 train 22 / val 16 / test 12 (그중 OOD 2) |
| 파일럿 중요도 | `outputs/importance_v1/config.json` (`num_clips` 50, 위 50클립) |
| 교체 | `outputs/importance_v2/config.json` (`calib_manifest: calib_100`, 100클립, 2026-08-08). `calib100_sampling.md`, `test500_sampling.md` |
| 논문 arm 레시피 | 각 `outputs/slim_*/slim_meta.json`의 `importance_from` |
| 교집합 | `outputs/eval_sets/*.parquet`의 `clip_id`로 계산 |

## Results (Facts)

### 1. 50클립 세트의 용도

- 7월 파일럿(2026-07-21 ~ 08-06)에서 dual-objective Taylor 중요도(CoC NLL + FM MSE)를 재는 보정 세트였다(`importance_v1`).
  이 점수가 제거할 Q head와 MLP 채널을 정했다.
- 공식 split을 쓰기 전에 뽑혀 공식 test 클립 12개가 섞였다. 2026-08-08에 공식 train에서만 뽑은 `calib_100`으로 바꿨다.
- Test500을 뽑을 때(2026-08-07) 이 12개를 후보에서 뺐다(61,599 → 61,587).

### 2. 논문의 최종 arm은 모두 `calib_100`

| 논문 행 | 빌드 | 중요도·보정 데이터 | 근거 |
|---|---|---|---|
| MoOP | `slim_dual_u40_v2` | `importance_v2` (calib_100) | `slim_meta.json` |
| MoOP + expert | `slim_dualexp_u40_em93p75` | VLM `importance_v2`; expert `importance_stepexp_znorm` ← `stepimp_fm_perstep_v2`·`importance_v2_ada` (둘 다 calib_100) | `slim_meta.json`, 각 `config.json` |
| FM-only / CE-only | `slim_traj_u40_v2` / `slim_coc_u40_v2` | `importance_v2` | `slim_meta.json` |
| Tyr | `slim_tyr_u40_r` | `importance_v2`; supernet·탐색 calib_100 | `run_tyr_supernet.py:76`, `run_tyr_search.py:428` 기본값, `plans/2026-08-20_tyr-baseline.md:50` |
| Wanda | `slim_wanda_u40_v2` | 활성값 `wanda_v1` (calib_100) | `outputs/wanda_v1/config.json` |
| LLM-Pruner | soowon `lp_r50` | calib_100 | `experiments/llm_pruner/lp_prune.py:212`. 이 코드로 다시 빌드한 `lp_r50`이 soowon 원본과 유닛 단위로 같음 |

- `importance_v1`을 쓰는 빌드는 `slim_tyrK` 하나뿐이고, 논문에 없다.

### 3. 평가셋과의 교집합

| 50클립 세트 ∩ | Test500 | Test2000 | Val500 | **OOD-Val262** | Calib100 |
|---|---|---|---|---|---|
| 클립 수 | 0 | 0 | 0 | **2** (2ab96e64, f4d0d2ce) | 0 |

7월 split 전체(1,181클립)의 다른 부분도 확인했다.

| 7월 split ∩ | Test500 | Test2000 | Val500 | OOD-Val262 |
|---|---|---|---|---|
| train (900) | 1 | 5 | 7 | 14 |
| val (131) | 0 | 0 | 0 | 2 |
| test (150) | 0 | 0 | 1 | 1 |

### 4. 50클립 세트로 정한 설계 결정

- **층별 균일 예산과 운영점(VLM q/o+MLP 40% 제거 = 전체 24.0%).**
  - 근거 실험은 2026-07-29 기준 × 배분 2×4 격자다(`reports/archive/2026-07-29_criterion-vs-allocation-grid.html`, `outputs/grid_v1`).
  - `importance_v1`을 썼고, 7월 split의 val 50클립 × K=8로 평가했다.
  - 이 평가 클립 50개는 Test500·Test2000·Val500과 0개, OOD-Val262와 **1개** 겹친다.
- **단위 종류(Q head + MLP 채널, KV 유지)** 는 구조에서 나왔다. expert가 VLM KV 캐시를 읽기 때문이다.
- **`calib_100`으로 다시 한 것:** 기준 비교와 FM-only/CE-only 행(08-09 one-shot, 08-11 폐루프), 순위 결합 규칙 ablation(08-20), expert 축 결정(08-21 ~ 08-28, 주로 val500).

### 5. 50클립 세트 말고 test split을 쓴 곳

| 세트 | 정체 | 용도 | 최종 모델에 영향 |
|---|---|---|---|
| `calib_test100` | Test500의 앞 100클립 | 기울기 해부 재현 분석(`gradanat_test100`, `paper-analysis/2026-09-24_heldout_replication.md`)만 | 없음(마스크·모델을 만들지 않음) |
| `calib_val100` | Val500의 100클립 | 보정 출처 실험(`outputs/calib_source`). 평가는 Test500·OOD-Val262에만 | 논문 arm 아님 |
| `test_cruise120` | Test500의 120클립 | 0번 층 head 교체 분석 → `l0keep` 레시피(cvpr27 트랙) | ICLR 논문의 MoOP(`dual_u40_v2`)와는 무관 |

## Analysis (Opinion)

1. **보고된 모델과 평가에는 test 노출이 없다.**
   - 논문의 7개 행은 모두 `calib_100`(공식 train만)으로 중요도·보정을 다시 쟀다.
   - 50클립 세트의 test 12개는 Test500·Test2000에 없다.
2. **답해야 할 부분은 파일럿 단계의 설계 결정이다.**
   - 리뷰어가 짚은 "pruning ratios"와 층별 배분 방식은 50클립 중요도로 정했다. 그 평가에 test 클립은 쓰지 않았다.
   - 따라서 "어떤 결정에도 안 썼다"가 아니라 "파일럿 격자의 중요도 측정에 썼고, 최종 모델은 모두 다시 쟀다"로 답해야 한다.
3. **OOD-Val262 겹침 2개(격자 평가 클립까지 합하면 3개)는 공개하는 편이 안전하다.** 논문의 "OOD-Val262 is clip-disjoint from Calib100 and Test500"은 맞지만, 파일럿이 그 클립을 봤다.
4. **재현 가능성.** 의존을 없애는 방법은 간단하다.
   - `calib_100` 중요도로 2×4 격자를 val 클립만으로 다시 돌린다(개루프 8칸 × 50클립).
   - 저장된 클립별 결과에서 두 클립을 빼고 OOD-Val260 수치를 다시 낸다(GPU 불필요).

**답변 초안 (영문, `[ ]`는 위 두 확인 뒤 채움)**

> The earlier 50-clip set was the calibration set of our pilot study, drawn before we adopted the official PhysicalAI-AV
> splits (22 train / 16 val / 12 test clips). It was used only to compute Taylor importance scores in pilot experiments. Two
> design choices were fixed in that phase: the uniform per-layer budget and the 40% VLM attention/MLP operating point (24% of
> total parameters). Both came from a 2×4 allocation × criterion grid, evaluated on 50 pilot-validation clips that do not
> overlap Test500. Unit types follow the architecture, because the action expert reads the VLM KV cache. Every reported model
> was rebuilt with importance measured on Calib100 (official train only): MoOP, CE-only, FM-only, the expert step
> importance, Tyr's reconstruction and search, Wanda's activations, and LLM-Pruner's gradients. The 12 test clips are
> excluded from Test500, so no evaluated clip contributed to any importance measurement. To remove the remaining
> dependence, we re-ran the allocation grid with Calib100 importance on validation clips only: [result]. We also note that
> 2 of the pilot clips belong to OOD-Val262; excluding them changes OOD results by [value].

**다음 단계**
- 격자 재실행: GPU 필요. 빈 카드와 시간 확인 후.
- OOD-Val260 재계산: 저장된 클립별 결과로 바로 가능.
