# Wanda 베이스라인 — 폐루프(alpasim) 평가 (2026-08-26)

## 목적

T2(`tab:criteria`)의 폐루프 열은 기준별로 채워져 있다(baseline 0.750, traj 0.783,
coc 0.660, j 0.536, dual 0.828, jtraj 0.791). gradient-free 베이스라인 Wanda만 빠져
있어, 같은 프로토콜(150 scenes × 2 rollouts, public_2601)로 그 칸을 채운다.

개루프에서 Wanda는 이미 붕괴했으므로(test ADE@6 2.975, degen 87.6%; 현재 최하인
j-단독 2.158보다 나쁨) 폐루프 점수도 j의 0.536 아래일 것으로 예상된다 — 즉 이 실험은
가설 검정이라기보다 **베이스라인 표를 완성하기 위한 측정**이다. 4장 × ~8h의 비용을
그 목적에 쓴다는 점을 명시해 둔다.

**H-AC**: Wanda의 폐루프 score는 baseline 대비 유의하게 낮고(paired CI < 0), 지금까지
측정된 모든 기준 arm보다 낮다.

## 설정 (기존 프로토콜 그대로)

- 체크포인트: `slim_wanda_u40_v2`를 **state 포함 재빌드**(개루프용 `--no-state` 빌드는
  drivers 하드링크가 불가). 동일 점수(`wanda_v1`)·동일 예산이므로 선택은 재현되어야 한다.
- 드라이버 등록: `outputs/slim_wanda_u40_v2/` → `/mnt/nvme1n1/ad_vla/data/alpasim/
  drivers/slim_wanda_u40_v2/` 하드링크(같은 마운트라 추가 용량 0).
- 실행: `DRIVER_OMP_THREADS=8 bash launch_alpasim_shards.sh slim_wanda_u40_v2 150 2
  "4 5 6 7"` — 시프된 `m2601_merged_*` 전부와 같은 경로(샤드당 driver/renderer/physics/
  trafficsim 동일 카드, Ada 4장, OMP 8). 이후 `merge_alpasim_shards.py
  --expect-scenes 150` → `m2601_merged_slim_wanda_u40_v2`.
- 분석: alpasim venv에서 `analyze_alpasim.py`(per-rollout → per-scene → baseline 대비
  paired delta, bootstrap CI + Wilcoxon). degen은 `coc_stats`.

## 사전 등록 게이트

- **A0 (무결성)**: 재빌드 마스크가 개루프에서 평가한 `--no-state` 빌드와 **동일**
  (kept 인덱스 36/36 레이어 일치, 제거 2,657,452,032). 150 scenes 전부 완주,
  merge는 `aggregate/` 없는 샤드를 거부.
- **A1 (주 판정)**: paired d_score(vs baseline) bootstrap 95% CI. 부기록으로 Wilcoxon.
  N=150에서 해상도 0.080이므로 그보다 작은 차이는 미검출로 보고한다.
- **A2 (부기록)**: 폐루프 CoC degen, at-fault 충돌률, progress, 그리고 기존 arm들
  (j 0.536 최하) 대비 위치.

## 비용

재빌드 ~1.5h(1장, 16.8 GB — /mnt/nvme1n1 잔여 111G 확인 완료) + 폐루프 ~8h(4장)
+ merge·분석 ~10분.

## 상태

2026-08-26 사용자 지시("wanda 모델로 alpasim eval 진행")로 실행 개시.

**완료 (2026-08-27) — H-AC 절반만 성립.** 150/150 완주(fail=0), 300 rollouts,
merge `m2601_merged_slim_wanda_u40_v2`.

| config | score | pass% | col@fault | progress | CoC degen | d_score vs baseline |
|---|---|---|---|---|---|---|
| baseline | 0.750 | 89.0 | 0.040 | 0.747 | 0.006 | — |
| dual | 0.828 | 91.0 | 0.023 | 0.828 | 0.027 | +0.079 [+0.036,+0.123] |
| jtraj | 0.791 | 88.7 | 0.043 | 0.804 | 0.008 | +0.042 [−0.005,+0.088] |
| traj | 0.783 | 88.0 | 0.033 | 0.805 | 0.050 | +0.033 [−0.012,+0.079] |
| coc | 0.660 | 83.7 | 0.073 | 0.685 | 0.020 | −0.089 [−0.151,−0.029] |
| **wanda** | **0.621** | **76.0** | **0.147** | **0.692** | **0.458** | **−0.128 [−0.200,−0.058]** |
| j | 0.536 | 70.7 | 0.173 | 0.638 | 0.014 | −0.214 [−0.281,−0.145] |

- **A0** PASS (사전 검증 완료). **A1** d_score −0.128, CI가 0 제외, Wilcoxon p=0.0023,
  W/L/T 44/67/39 → baseline 대비 유의 열세 **확인**.
- **H-AC의 두 번째 절반은 기각**: wanda(0.621)는 j-단독(0.536)보다 **높다**. 개루프는
  wanda가 j보다 훨씬 나쁜데(ADE@6 2.975 vs 2.158, LingoQA 9.2 vs 32.2%) 폐루프 순서가
  뒤집힌다.
- **A2**: at-fault 충돌 0.147(baseline 0.040, j 0.173 다음으로 높음), 폐루프 CoC degen
  0.458 — 그 중 empty 0.406으로 대부분 **빈 출력**(Tyr의 붕괴 양상과 동일), 길이 138.
  개루프 test degen 0.876보다 낮다.

**해석**: `tail-safety-independent-of-reasoning` 메모의 "CoC 건강은 안전 프록시가 아니다"가
반대 방향으로 다시 확인됐다 — 추론이 거의 완전히 무너진 wanda가 추론은 멀쩡한(degen 1.4%)
j-단독보다 주행은 낫다. 표에는 wanda를 gradient-free 베이스라인 행으로 싣되, "개루프 순위가
폐루프 순위를 함의하지 않는다"는 사례로 함께 언급한다.
