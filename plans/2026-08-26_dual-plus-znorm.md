# dual + znorm: 두 타워의 검증된 절단을 한 체크포인트로 (`dualexp_u40_e25`)

날짜: 2026-08-26. 브랜치: `dual-plus-znorm`.

## 가설

개별 검증된 두 절단 —

- **VLM**: `dual_u40_v2` (dual = max(rank I_traj, rank I_CoC), uniform 0.3985632694,
  −2,657,452,032 / 24.0%; 스텝 축은 현행 sum — VLM znorm은 +0.79 REGRESSION으로 닫힘)
- **expert**: znorm r25 (스텝별 레이어 내 z-정규화 집계, uniform 0.25,
  −532,021,248 / 4.8%; 개루프 무압축과 구별 불가 +0.0003)

— 를 합치면 **−3,189,473,280 (28.79%)** 이고, expert 몫은 dual 위에서도 "공짜"다.

진짜 미지수는 상호작용이다: **프루닝된 VLM의 KV 캐시를 프루닝된 expert가 읽는 첫
config**다. 지금까지 expert znorm은 dense VLM의 캐시에 대해서만 측정·평가됐다.

## 설계 (one-factor)

- 새 config `dualexp_u40_e25`: `dual_u40_v2`의 VLM 절반 + `expert_u25`의 expert 절반을
  글자 그대로 합친 것. KV 무접촉. `make_slim.py`에 분기 추가
  (미지명은 else=cocsafe로 조용히 떨어지므로 명시 분기 필수).
- importance는 **Blackwell-anchored drop-in** `importance_stepexp_bw_znorm`
  (`make_stepexp_importance.py --ref importance_v2 --aggs znorm --prefix importance_stepexp_bw`).
  이유: 출시된 `slim_dual_u40_v2`는 Blackwell `importance_v2`로 빌드됐고, Ada 본
  (`importance_stepexp_znorm`이 상속한 `importance_v2_ada`)과의 VLM kept-set 겹침은
  Q 0.9649 / MLP 0.9721 — Ada 본을 쓰면 기존 dual arm과의 비교가 이중 요인이 된다.
  expert znorm 키는 `stepimp_fm_perstep_v2`에서 나와 ref와 무관하므로 재생성에 손해가 없다.
- `--no-state` 빌드 (선택-only, slim_meta.json으로 bit-identical 복원; 디스크 99%).
- 평가: 고정 프로토콜 (rollout-only, K=8 → minADE@6/minFDE@6, Ada 4–7, 클립 유도 시드).
  기존 대조군 재사용: `baseline_ada_ps_indist` (0.8236), `dual_u40_v2_ps_indist` (0.8904).
  신규 2건: combined (`dualexp_u40_e25_ps_indist`), expert 단독
  (`expert_znorm_r25_ps_indist` — znorm r25의 첫 고정 프로토콜 수치, 가산성 검사용).

## 사전 등록 게이트

- **G0 (무결성, hard)** — 빌드 직후 slim_meta 비교:
  `vlm` == `slim_dual_u40_v2`의 것과 bit-identical, `expert` == `slim_expert_znorm_r25`의
  것과 bit-identical, removed == 3,189,473,280. VLM이 다르면 Ada importance 누출 → 중단.
- **G1 (CoC 동일성, sanity)** — VLM 가중치·시드가 dual arm과 동일하므로 per-clip
  `gen_coc` / `nll_self`는 `dual_u40_v2_ps_indist`와 일치해야 정상 (같은 Ada 아키텍처).
  다르면 프로토콜 드리프트 조사. fallback 게이트: combined `coc_degenerate` ≤ dual + 1pp.
- **G2 (주 게이트)** — paired Δ minADE@6 (combined − dual), val500, 95% bootstrap CI
  [lo, hi]. 문턱 **+0.013** = dual의 VLM 비용 0.0668 × 파라미터 비율 532/2657
  (비례-비용 예측):
  - ADOPT ("공짜"): hi < +0.013
  - INCONCLUSIVE: CI가 +0.013을 걸침 ("회귀 미검출"로만 보고)
  - REJECT: lo ≥ +0.013 (상호작용 페널티 실재)
- **G3 (가산성, 부차)** — per-clip 상호작용
  `I_c = (combined − dual) − (expert_only − baseline)`의 평균 CI, 기대 0;
  lo > +0.01이면 초가산 플래그. Δ(expert_only − baseline)도 함께 보고
  (사전값: 200클립에서 +0.0003).
- **G4 (vs baseline, 부차)** — Δ(combined − baseline), 기대 ≈ +0.067. 게이트 없음
  (헤드라인 표용; VLM 변화를 가로지르는 대비라 ±0.075 해상도를 그대로 물려받는다).

## Stage 2 (G2 = ADOPT일 때만)

test500 (대조 `dual_u40_v2_ps_test`, 게이트 hi < +0.022 = 0.1081×0.200) +
OOD-val 262 (`dual_u40_v2_ps_ood` 클립 교집합, hi < +0.024 = 0.1197×0.200).

## 범위에서 제외 (결정 사항)

폐루프 alpasim: N=150의 해상도는 0.080인데 기대 효과 ~0, expert znorm 단독도 이미
판정 불가(−0.0297 [−0.065,+0.005]), 그리고 21 GB `slim_state.pt`가 99% 디스크에
필요하다. 이 arm이 개루프에서 ADOPT면 폐루프는 znorm-vs-magnitude 재대결
(2026-08-21 리포트 §12.1)과 함께 별도 결정.

## 비용

빌드 ~0.5 GPU-h + 평가 2 × ~1.1 GPU-h (Ada 4·5 병렬) + stage 2 ~1.7 GPU-h.
디스크 < 100 MB (state 없음).
