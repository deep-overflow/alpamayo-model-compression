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

## Addendum (2026-08-26): 조건부 expert 중요도 — pruned VLM 위에서 다시 재기

(e10/e15 비율 sweep은 사용자 지시로 보류; 대신 아래를 먼저 검증한다.)

**가설.** expert znorm의 kept set은 dense VLM의 캐시를 읽는 상태에서 잰 기울기로 골랐다.
결합 config에서 expert는 **프루닝된 VLM의 캐시**를 읽으므로 입력 분포가 달라졌고, e25
REJECT(+0.0688)의 일부는 이 낡은 선택 때문일 수 있다. `slim_dual_u40_v2` 위에서 expert
Taylor를 다시 재면(조건부 중요도) 이를 직접 검증할 수 있다.

**절차.**
1. 측정: `run_step_importance.py --mode fm --noise-mode per_step
   --model outputs/slim_dual_u40_v2 --exp-id stepimp_fm_perstep_dualvlm`
   (calib_100, seed 42 — 원본 `stepimp_fm_perstep_v2`와 동일 클립·시드라 paired).
   `--model` 플래그는 이번에 추가 (`load_slim` 경유; 게이트는 o_proj/down_proj
   forward-pre-hook이라 slim 바인딩과 무관하게 작동).
2. 집계: `make_stepexp_importance.py --stepimp stepimp_fm_perstep_dualvlm
   --ref importance_v2 --aggs znorm --prefix importance_stepexp_dv`
   → VLM 절반은 여전히 Blackwell `importance_v2` (G0의 VLM bit-identity 유지).
3. **C1 (무료 go/no-go)**: r25에서 조건부 vs 기존 znorm 선택의 kept-overlap.
   **≥ 0.98 (Q·MLP 모두) → 조건화는 no-op, 빌드·평가 생략** 하고 종료.
   (스케일 참조: sum vs znorm 겹침이 0.889/0.945였다.)
4. 빌드: `--config dualexp_u40_e25 --importance importance_stepexp_dv_znorm
   --out outputs/slim_dualexp_u40_e25_cond --no-state`.
   G0': vlm == slim_dual_u40_v2 bit-identical, removed == 3,189,473,280
   (expert kept가 기존과 다른 것이 요점).
5. 평가: val500, 4-way 샤드 GPU 4–7, exp-id `dualexp_cond_ps_indist`.

**게이트.**
- **C2 (주)**: cond-combined − dual, mean Δ CI vs **+0.013** (G2와 동일 문턱·동형).
- **C3**: cond-combined − stale-combined (`dualexp_u40_e25_ps_indist`) paired —
  재보정 효과 그 자체. hi < 0이면 재보정이 유의하게 회복.
- 기대 관리: median 가산성(+0.0248 ≈ +0.0228)은 상호작용이 작다고 시사하므로,
  재보정이 회복할 수 있는 상한은 mean의 꼬리 성분(~+0.04)이다. C1에서 겹침이 1에
  가까우면 그조차 선택 문제가 아니라는 뜻이다.

## Addendum 2 (2026-08-26): 조건부 중요도로 e10/e15 sweep — 공짜 상한

C2 REJECT 이후 사용자 지시로 재개. importance는 **조건부** `importance_stepexp_dv_znorm`
(C3에서 stale보다 유의하게 나음이 확인된 선택), 절차·프로토콜·대조군은 본문과 동일.

- 빌드: `dualexp_u40_e10` / `dualexp_u40_e15` + `--importance importance_stepexp_dv_znorm`
  → `outputs/slim_dualexp_u40_e{10,15}_cond`, `--no-state`.
- 그리드 주의: `select_mask`는 레이어당 round(16×r)개 Q head를 제거하므로 e10·e15 모두
  **Q 2개/레이어(12.5%)로 동일**하고 MLP(826 vs 1238/레이어)만 다르다. 예상 제거:
  e10 220,446,720 / e15 311,574,528 (빌드 후 slim_meta로 확정).
- 무결성: VLM은 셋 다 slim_dual_u40_v2와 bit-identical; expert kept 중첩
  kept(e10) ⊇ kept(e15) ⊇ kept(e25_cond) (동일 argsort의 접두사 구조).
- **게이트 (S-series, G2와 동형)**: 문턱_eN = 0.0668 × removed_eN / 2,657,452,032
  → e10 ≈ +0.0055, e15 ≈ +0.0078. mean Δ(combined_eN − dual) CI [lo,hi]:
  FREE hi < 문턱 / REJECT lo ≥ 문턱 / 그 외 INCONCLUSIVE.
- **검정력 한계를 사전에 인정**: mean CI 반폭이 ~0.03이라 e10 문턱(+0.0055)의 FREE는
  증명 불가능에 가깝다. median CI(반폭 ~0.015)를 부차 판독으로 함께 보고하고,
  INCONCLUSIVE는 "회귀 미검출"로만 서술한다 (공짜 단정 금지).

**판정 (2026-08-26 실측)**: C1 GO (kept-overlap Q 0.9444 / MLP 0.9784, 문턱 0.98 미달).
C2 **REJECT** — cond-combined − dual = +0.0544 [+0.0283, +0.0808] (median +0.0250),
CI 하한이 문턱 +0.013 초과. C3 **ACCEPT** — cond − stale = **−0.0144 [−0.0287, −0.0003]**
(median −0.0066 [−0.0127, −0.0011]): 재보정은 실재하는 개선이지만 격차의 ~21%만 회복.
나머지 ~+0.054는 expert r25 절단의 고유 비용. 산출물: `stepimp_fm_perstep_dualvlm`,
`importance_stepexp_dv_znorm`, `slim_dualexp_u40_e25_cond`, `dualexp_cond_ps_indist`,
분석 `dualexp_cond_arms_val` / `dualexp_cond_vs_stale`.
