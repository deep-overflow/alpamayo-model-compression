# 결합 연산자 ablation — dualsum / dualprod (2026-08-20)

## 가설과 배경

dual의 `max(rank I_traj, rank I_CoC)`에서 max()는 사전 등록된 결합 연산이지만, 대안
(rank-sum = Borda, rank-product)보다 낫다는 직접 증거는 없다 — 지금 근거는 간접적이다
(single-criterion arm들의 붕괴 + KI 관점의 specialist 보호 논리). 리뷰어가 "왜 max인가"를
물을 것이므로 operator ablation 행을 만든다.

CPU 사전 분석 (importance_v2, 동일 예산, `experiments/head_analysis/compare_combine_ops.py`):

| 결합 | dual과 kept-set 겹침 (Q / MLP) | 교환되는 유닛 |
|---|---|---|
| rank-sum | 92.0% / 93.6% | (0.63, 0.13) 준-specialist → (0.51, 0.42) generalist |
| rank-product | 88.6% / 90.2% | (0.68, 0.12) → (0.50, 0.39), 더 강한 anti-specialist |
| raw 덧셈 | — | traj 스케일이 CoC의 ~12배라 traj-only와 95.2% 일치 → **arm 불필요** (traj_u40_v2가 그 결과) |
| raw 곱셈 | — | L35의 traj 구조적 0으로 선택 무정의 → **제외** (G-CAL 0/4로 cardinal 신뢰도 기각) |

**H-OP**: max는 rank-sum·rank-product에 비열등하다 (specialist 손실이 양방향 모두
재앙적이라는 single-arm 증거의 연장: traj-only LingoQA 73.2→37.0, coc-only 폐루프 −0.089*).

## 설계 (one-factor)

새 config `dualsum_u40_v2` / `dualprod_u40_v2`: u40_v2 패밀리와 예산(uniform
0.3985632694), 축(VLM Q/MLP만, expert·KV 불변), calibration(importance_v2, calib_100)
전부 동일 — 레이어 내 rank 결합 연산만 max → sum / product로 바꾼다.

- `make_slim.build_masks`의 uni 분기에 op 디스패치 추가. 기본은 `np.maximum` 그대로 —
  기존 config들의 비트 동일 재생성이 깨지면 안 된다.
- 빌드는 `--no-state` (recipe만; `load_slim`이 로드 시 재구성). 디스크 추가 ~0.
- 평가: 고정 프로토콜의 **test_500만** (u55/composition arm과 같은 ablation-행 취급).
  rollout-only, K=8 per-sample 배열 저장, 보고는 minADE@6·minFDE@6 평균, Ada 카드,
  paired seed(clip_seed).

## 사전 등록 게이트

- **R0-OP (무결성)**: 제거 파라미터 == 2,657,452,032 정확 일치, 레이어별 컷 13/32 Q ·
  4898/12288 MLP, slim_meta kept-set의 dual 대비 겹침이 CPU 분석과 일치.
- **G-OP1 (주 판정)**: test_500 paired ΔminADE@6 (arm − dual_u40_v2), bootstrap
  median CI.
  - CI가 0을 포함하거나 전체 > 0 → **max 유지** (예상 결과; 논문에 ablation 행으로 수록).
  - CI 전체 < 0 (sum/prod 유의 우세) → 채택 전에 val_500 · OOD-val · LingoQA 후속 검증 필수.
- **G-OP2 (추론 채널)**: coc_degenerate가 dual(test ~2%대)의 2배를 넘으면 언어 손상
  신호로 기록.
- 부기록: paired Δ vs baseline_ada_ps_test, minFDE@6.

## 실행 / 예산 (Ada 2장, run_retry_host)

1. `make_slim --config dual{sum,prod}_u40_v2 --importance importance_v2 --no-state`
   — GPU 4·5 병렬, 각 ~30–40분 (스모크 포함)
2. `run_baseline --set test --model outputs/slim_dual{sum,prod}_u40_v2` — 이어서 각 ~1.2 h
3. R0-OP 검증 + paired 분석 + 게이트 판정, `paper_numbers.py` ARMS 추가

합계 ~2 h wall-clock. 산출물: `outputs/slim_dual{sum,prod}_u40_v2` (recipe),
`outputs/dual{sum,prod}_u40_v2_test` rows, 게이트 판정 보고.

## 상태

2026-08-20 사용자 승인("계획으로 정리해서 진행해줘"), 실행 개시.

**완료 — H-OP 확인, max 유지.** R0-OP: 두 arm 모두 −2,657,452,032 정확 일치, 레이어별
19/7390 보존, dual 대비 겹침 dualsum Q 91.4%/MLP 93.6%, dualprod Q 88.5%/90.2% (CPU
예측 재현). G-OP1 (test_500, paired ΔADE@6 vs dual): dualsum **−0.0010 [−0.0105,+0.0069]**
— CI가 0 포함, 동급; dualprod **+0.1423 [+0.0998,+0.1769]*** — 유의 열세, FDE도
+0.4851*. G-OP2: degen dualsum 0.034 / dualprod 0.048, 둘 다 2×dual(0.060) 미만.
절대치: dualsum 0.9597/2.6066, dualprod 1.2381/3.4210 (dual 0.9498/2.5459).
결과 보고: `reports/evaluation/2026-08-20_openloop-results-tables.html` §3.
