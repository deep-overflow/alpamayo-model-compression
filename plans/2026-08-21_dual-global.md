# dual × Tyr 요인 분해 — 전역 할당 탐색·재구성을 dual에 얹으면? (2026-08-21)

## 배경과 질문

Tyr 베이스라인(tyr-eval 세션) test 500 @6: `tyr_r`(OSSCAR 선택 + 재구성 + 전역 탐색)
0.950 / degen 0.8%, `dual_u40_v2` 0.950 / 3.0% — paired Δ −0.001 [−0.027, +0.017] (동률),
중앙값·degen은 tyr_r 우세. `tyr_uniform_r`(탐색 없음) 0.999 / 4.8%, T2(tyr_r − tyr_uniform_r)
−0.010* → 전역 탐색은 Tyr 자신의 uniform을 작지만 유의하게 개선.

질문: Tyr의 두 장치 — **전역 할당 탐색**과 **OSSCAR 재구성** — 를 dual에 얹으면 dual이
올라가는가, 아니면 dual의 동급 성능은 기준(Taylor dual) 자체에서 오는가?

## 요인 설계 (선택 기준 × 재구성 × 할당)

| 셀 | 선택 | 재구성 | 할당 | arm | 상태 |
|---|---|---|---|---|---|
| 1 | dual | 없음 | uniform | `dual_u40_v2` | 있음 (0.950 / 3.0%) |
| 2 | OSSCAR | 없음 | uniform | `tyr_sel_uniform` | 있음 (붕괴 2.11 / 74%) |
| 3 | OSSCAR | OSSCAR | uniform | `tyr_uniform_r` | 있음 (0.999 / 4.8%) |
| 4 | OSSCAR | OSSCAR | 탐색 | `tyr_r` | 있음 (0.950 / 0.8%) |
| **A** | dual | 없음 | **탐색** | `dualg_u40` | 신규 — "dual-global" |
| **B1** | dual | **OSSCAR** | uniform | `dualr_u40` | 신규 — 재구성만 |
| **B2** | dual | **OSSCAR** | **탐색** | `dualgr_u40` | 신규 — 완전 융합 |

A−1 = 전역 탐색 이득(dual에서), B1−1 = 재구성 이득(dual에서), B2 vs 4 = 융합 vs Tyr
정면 비교, B2−A = 탐색된 할당 위에서의 재구성 이득. 예산은 전부 타입-보존 레벨로
−2,657,452,032 동일.

## 설계 세부

- **레벨·탐색**: Tyr와 동일(레이어당 Q 13±l head, MLP 4898±256·l, 9레벨, 타입-보존 변이,
  20세대×32 offspring, 다단 선택 4/16/48 클립, 적합도 = KL_coc/KL₀ + MSE_vf/MSE₀,
  teacher 캐시 `tyr_teacher_u40` 재사용). 레이어 내 선택은 **dual 랭킹**
  (`max(rank I_traj, rank I_CoC)`, importance_v2) 상위 유지 — level-0이 dual_u40_v2와
  비트 동일해야 함(무결성 게이트).
- **A (dual-global)**: 재구성이 없으므로 supernet은 마스크만 — `run_tyr_search.py`에
  mask-supernet 모드 추가(후보 = 레벨 벡터 → dual 랭킹 마스크를 `mask_lib.PruneMasks`로
  즉시 적용, 가중치 파일 불필요). 최종 config → `make_slim` 마스크 빌드(`--no-state` 가능).
- **B1/B2 (dual + 재구성)**: `run_tyr_supernet.py`에 `--selection dual` 추가 — 레벨별
  kept-set을 dual 랭킹으로 정하고 OSSCAR `local_prune_core`의 "이미 절단된 그룹 재구성"
  경로(`remaining_to_prune<=0`)로 inv(H_kk)·G_k만 수행. damping 업스트림 1e-2
  (Tyr 결과에서 1.0은 유해로 확인). error accumulation 동일. state 저장 빌드 필수.
- B2는 B1의 supernet 위에서 탐색 → 추가 supernet 불필요.
- 보조 probe(선택, 비용 ~0): "dual 선택 + tyr_r의 탐색된 할당 이식" — 할당이
  선택과 무관하게 전이되는지 한 번에 확인.

## 사전 등록 게이트 (test_500, paired ΔminADE@6 median CI; val·OOD-val 부기록)

- **G0 무결성**: 세 arm 모두 제거 −2,657,452,032; A·B1·B2의 level-0 마스크가
  dual_u40_v2 kept-set과 비트 동일(B1은 마스크 동일, 가중치만 재구성).
- **G-A**: dualg − dual. CI < 0 → 전역 탐색이 dual을 개선. 0 포함 → dual의 uniform은
  이미 충분(그리드 결과와 일관). > 0 → 탐색이 해로움(적합도 한계 — degen 부기록 필수).
- **G-B1**: dualr − dual. CI < 0 → 재구성이 dual을 개선(Tyr 동급의 출처 = 재구성).
- **G-B2**: dualgr − tyr_r (정면), dualgr − dual, dualgr − dualg.
- 부기록: degen(추론 채널), minFDE@6, Δ vs baseline, 탐색 레이어별 분배 vs tyr_r 분배
  비교, 탐색 적합도 궤적.
- 이득이 확정되면(G-A/B1/B2 중 하나라도 CI<0) 후속: LingoQA·폐루프 150 scenes.

## 비용 (Ada 4장, Tyr 평가 종료 후)

1. A: mask-supernet 탐색 ~50분(3워커) → 빌드 → 3셋 평가 ~2.5h
2. B1: dual-selection supernet(H 수집+재구성, 9레벨) ~50분 → state 빌드 → 3셋 ~2.5h
3. B2: B1 supernet 위 탐색 ~50분 → state 빌드 → 3셋 ~2.5h
병렬 배치 시 전체 ~6–7h. 디스크: B1 supernet ~41GB (nvme1n1 여유 있음).

## 위험

- 적합도(teacher-forced KL + field MSE)는 rollout degeneration을 직접 보지 못함 — Tyr에서는
  우연히 degen이 낮아졌으나 보장은 없음. dual은 선택 자체에 CoC 보호가 있어 위험 낮음.
- 재구성은 kept 가중치를 30–60%(일부 수 배) 바꾼다 — Tyr 선택에서는 유효했지만 dual
  선택에서의 상호작용은 경험적으로 확인해야 함(B1이 그 답).
- 탐색 잡음: 세대별 적합도가 미니배치에 따라 ±0.3 흔들림. 최종 config의 전체-calib
  적합도를 따로 기록.

## 상태

승인(2026-08-21 "일단 모두 진행해줘", 4장 최대 활용) — 실행 개시. A·B1·B2 + 보조 probe(dual 선택 + tyr_r 할당) 전부 진행.

## 결과 (2026-08-21, 3셋 + LingoQA 완료)

| arm | test | val | OOD-val | LingoQA |
|---|---|---|---|---|
| dual | 0.950 | 0.890 | 1.119 | 68.8 |
| dualg (A) | 0.978 (+0.001) | 0.897 (+0.003) | 1.049 (−0.001) | — |
| dualr (B1) | 0.860 (−0.052*) | 0.814 (−0.034*, baseline 동급) | 1.106 (−0.028*) | 41.8 (−27.0pp*) |
| dualgr (B2) | 0.864 (−0.037*) | 0.816 (−0.037*) | 1.103 (−0.035*) | 42.2 (−26.6pp*) |
| probe (dual + tyr_r 할당) | 0.958 (+0.002) | val/OOD-val 동급 | | — |

(minADE@6 평균, 괄호는 paired Δ vs dual 중앙값, * = CI가 0 제외.)

- **G-A 불통과**: 전역 할당 탐색은 dual에 이득 0 (자체 탐색·tyr_r 할당 이식 모두 동급).
- **G-B1 통과(3셋)**: 재구성은 궤적을 크게 개선(val은 baseline 동급) — 단 **LingoQA −27pp**,
  단일-기준 붕괴 수준. 재구성 = 주행 프리필 분포로의 암묵적 미세조정 → VLM 지식 소실.
  지식 ↔ 궤적 트레이드오프이지 공짜 이득 아님.
- **G-B2 = B1**: 재구성 위의 탐색은 ADE 이득 0, degen만 1.4→0.8%.
- 귀속: Tyr 동급 성능의 출처는 OSSCAR 재구성; 전역 탐색 아님; OSSCAR 선택은 단독 붕괴.
  dual(선택-만)은 궤적·언어·폐루프를 모두 지키는 유일한 config.
- 후속 후보(미승인): 부분 재구성(MLP만/attention만), 혼합 calibration H, dualr 폐루프.
