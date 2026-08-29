# 캐시 표적 재구성 — expert가 민감한 깊은 층 캐시만 지키는 VLM 프루닝 (`dualrc`)

날짜: 2026-08-29. 브랜치: `cache-targeted-recon`. 상태: **승인, 구현 중** (2026-08-29 20:24 KST).

## 출발점 — 캐시 참조 지도(`2026-08-29_cache-use-map.html`)가 설계에 주는 것

| 보고서 결과 | 설계에 들어가는 방식 |
|---|---|
| 프루닝 캐시 + dense expert(A10−A00) = +0.07 ≈ dual 전체 비용 +0.058 (cache-shift) | 캐시 보존이 회수할 수 있는 상한 = dual의 개루프 궤적 비용 거의 전부 |
| 단위 이동당 **민감도**는 깊은 층에 몰림(층 24–35 49%, 0–15 27%); 목표 Σ 민감도×이동²의 92%가 층 20–35 | **층 20–35의 캐시만 표적**. 층 0–15는 지키지 않는다(이동도 작고 민감도도 낮음) |
| 차단 의존도 R은 상류 위치를 잴 뿐(민감도~R +0.06); 어텐션 질량·readout 몫은 인과의 대리가 아님(G1); 셀 차단은 분포 밖(G4) | 목표 함수 어디에도 R·어텐션 통계·차단값을 쓰지 않는다. 가중치는 Stage C 민감도 `maps_swap.npz:sensitivity` (층, 그룹) |
| expert는 10스텝 전부 캐시를 읽고, 스텝별 구간 구성은 일정(vision 0.72–0.75, text 0.15–0.17) | 목표는 스텝-무관(VLM 쪽 양). 캘리브레이션 **위치 가중**은 Stage A 구간 질량으로: vision 0.73 / text 0.16 / hist 0.04 / sink 0.01 / CoC 0.02 |
| GQA: 그룹 g의 캐시는 층 l+1의 k/v_proj 행 g가 h_{l+1}에서 만든다 | 캐시 오차 = 잔차 오차의 W_k,g / W_v,g 사영. 재구성 refit은 출력 metric에 불변이므로 **민감도는 선택 오차(OSSCAR 그룹 제거 판정)에만** 들어가고, refit은 위치 가중 Hessian으로 |
| `dualr`(dual 선택 + 전 층 OSSCAR 재구성)은 val500 +0.003 [−0.007, +0.019] — 재구성 = 캐시 보존은 이미 증명 | 새 것은 "전 층·prefill-only" → "깊은 층만·expert 가중·decode 혼합". dualr이 직접 대조군 |
| 그러나 dualr은 폐루프에서 dual보다 나쁨(−0.036\*, 충돌 +60%), Tyr은 CoC 붕괴(prefill-only Hessian이 decode 경로를 망침, racfit +0.0197) | decode 스트림을 Hessian 혼합에 유지(racfit 최적 혼합); 폐루프 게이트를 배포 주장 전 필수로 |

## 가설

**H1**: 재구성을 층 20–35에만 두고(층 0–15는 선택만), prefill 위치를 expert 어텐션으로 가중하며 decode
스트림을 섞은 Hessian으로 refit하면, val500 궤적은 dualr 수준(≈ 무압축)으로 회복되면서 CoC는 dual과
같은 수준을 유지한다(Tyr/dualr의 decode 손상 회피).
**H2**: 폐루프에서 dualr의 역전(GT 경로 추종↑·충돌↑)이 재현되지 않는다 — 그 역전이 "전 층 재구성 =
캘리브레이션 과적합"에서 왔다면, 절반 층 재구성은 그것을 줄인다. (H2가 기각되면 캐시 보존 자체가
폐루프 안전과 상충한다는 뜻이고, 그때는 선택-only 경로(C)로 돌아간다.)

## 설계

- 선택: **dual 그대로** (`dual_u40_v2` kept set과 bit-identical) → dual·dualr과 one-factor.
- 재구성 대상 층 L_R: arm 두 개 — `dualrc_16` (층 16–35), `dualrc_24` (층 24–35). 상류 절단의 누적을
  잔차 수준에서 되돌리는 것이므로 민감 층보다 조금 앞에서 시작하는 16–35가 본명, 24–35는 하한 확인.
- Hessian (`StreamHessianHook` → `mix_hessians`): 스트림 가중치 = Stage A 구간 질량(vision 0.73,
  text 0.16, hist 0.04, sink 0.01) + decode 스트림 가중치는 racfit 최적 혼합의 decode 비율. refit은
  `reconstruct_levels(damp=1e-2)`(upstream 값; 1.0은 퇴행).
- 선택 오차 metric(선택적 2단계, `dualrc_16m`): OSSCAR 그룹 제거 오차를 ‖M_l ΔY‖²로, M_l = 층 l+1의
  [√s_{l+1,g} W_k,g ; √s_{l+1,g} W_v,g] 스택(s = 민감도). 선택이 dual에서 벗어나므로 별도 arm.
- 저장: 가중치를 고쳐 쓰므로 `slim_state.pt` 필수(`--no-state` 금지, make_slim 가드 확장). 디스크 17 GB/arm.

## 사전 등록 게이트

- **G0 recipe**: VLM kept set == dual_u40_v2 (bit-identical); 층 0–15 가중치 == dual(재구성 안 함);
  L_R 밖 변경 0.
- **G1 캐시 대리 지표** (cachediff 기계, val500 앞 200클립, ~40분/arm): (i) 민감도 가중 캐시 이동
  Σ s·‖ΔV‖²가 dual 대비 ≥ 50% 감소, (ii) A10형 Δ(프루닝 캐시 + dense expert, 같은 CoC)가 +0.07 →
  CI가 0 포함. dualr에서도 같은 지표를 재서 "깊은 층만"의 손실을 정량화.
- **G2 val500 궤적**: Δ vs baseline 중앙값 CI가 dualr(+0.003)과 겹침; dual(+0.058) 대비 유의 개선.
- **G3 CoC/decode**: gen_coc 붕괴율 ≤ dual, LingoQA ≥ dual(바닥 37% 초과), racfit decode 재구성 오차
  ≤ dual 수준(Tyr의 +0.0197 악화 없음).
- **G4 폐루프** (150씬, Ada, dual·dualr·baseline과 페어드): score ≥ dual, 충돌 ≤ dual. dualr의
  d2gt↓·충돌↑ 분리가 재현되는지 별도 보고.

## 실행 순서와 비용

1. 민감도 가중치·스트림 가중치 파일 고정(이미 있음: `outputs/cacheuse_v1/maps_swap.npz`, Stage A 질량).
2. `make_slim.py` 분기 `dualrc_<start>` + Hessian 수집(`run_racfit`류, calib_100) — 빌드 ~1.5 h/arm, 2 arm.
3. G1 대리 지표 2 arm + dualr (3 × 40분, 2 GPU) → 여기서 H1의 캐시 부분이 갈린다.
4. G2·G3 val500 (2 arm × ~1 h, Ada).
5. G4는 G1–G3 통과 시에만 (8 h, Ada 4장).

## 이 계획이 답하지 않는 것
- expert 쪽 적응(캐시에 맞춰 expert LoRA)은 별도 — 상한이 같은 +0.07이라 VLM 쪽이 먼저.
- 층 20–35 캐시가 왜 민감한지(어떤 위치·어떤 정보)는 지도의 다음 단계이지 이 계획의 대상이 아니다.

## 구현 노트 (2026-08-29)

- `run_cache_recon.py`: dual 마스크를 **전 층에 건 채** forward하며 층 ≥ start의 o_proj/down_proj 입력에
  **토큰 가중 Hessian** H = Σ w_t x_t x_tᵀ를 모은다(prefill 구간별 expert 어텐션 몫 vision 0.722 /
  text 0.166 / hist 0.042 / sink 0.011을 클립 내 토큰 평균으로 정규화, own CoC는 `racfit_v1/rollouts.json`의
  K=4 시드를 decode 몫 0.16(racfit d10 코너)으로). 블록 순차 오차 누적은 하지 않는다 — 마스크된 상류가
  비관적 대리. 층은 서로 독립으로 refit되므로 start=16 supernet 하나가 `dualrc_u40_s16`과
  `dualrc_u40_s24`를 모두 만든다. 100클립 × 4 forward ≈ 5분 + refit.
- `make_slim.py` `dualrc_u40_s<N>`: dual 선택(allocations uniform + dual_scores, dual_u40_v2와 동일 경로)
  + supernet의 층 ≥ N 가중치 기록, 0-열에서 유도한 마스크가 dual과 일치해야 함(assert). `--no-state` 금지.
- `run_cacheproxy.py` (G1): dense + slim 두 모델을 한 프로세스에 올려 같은 TF 텍스트의 캐시를 비교
  (rel_v per (l,g), 민감도 가중 합)하고 dense expert로 A00/A10을 denoise(K=6, 클립 유도 시드).
- 예비 결과로 dual·dualr에도 같은 G1 지표를 잰다(세 arm 페어드).

### 첫 빌드의 결함 (2026-08-29 21:30 KST) — 마스크된 자기 입력은 refit을 항등으로 만든다

첫 supernet은 dual 마스크를 **전 층에** 건 채 Hessian을 모았다. 대상 층의 o_proj/down_proj 입력에서
제거 열이 이미 0이면 H의 제거 행·열이 0이라 최소제곱 해가 W_kept = H_kk⁻¹ H_kk W_k = W_k, 즉 **항등**이다.
G1 프록시가 dualrc_s16·s24를 dual과 소수 4자리까지 동일하게(이동 비 1.000, 비용 차 +0.0000) 잡아냈다.
수정: Tyr/dualr supernet과 같은 **블록 순차** — 대상 층 l의 forward는 층 < l에만 마스크(이미 refit된 층은
0-열 가중치를 써 넣음), 층 l 자신의 입력은 마스크 해제. refit의 상대 변화량을 로그로 남긴다.
start=16·24 supernet을 각각 따로 빌드한다(s24는 층 16–23이 refit 없이 프루닝된 상류를 봐야 하므로).

## 결과 — G1 (2026-08-29 22:40 KST, val500 앞 200클립, K=6, 클립 페어드)

| arm | refit 층 | A10−A00 중앙값 [CI] | 평균 | 민감도 가중 이동 (dual 비) | ‖ΔV‖/‖V‖ 층 24–35 |
|---|---|---|---|---|---|
| dual | — | +0.0766 [+0.0166, +0.1322] | +0.106 | 0.972 (1.00) | 0.512 |
| dualrc_s24 | 24–35 | +0.0808 [+0.0318, +0.1331] | +0.101 | 0.928 (**0.957**) | 0.498 |
| dualrc_s16 | 16–35 | +0.0270 [+0.0016, +0.0659] | +0.089 | 0.736 (**0.757**) | 0.437 |
| dualr | 0–35 | +0.0015 [−0.0096, +0.0227] | −0.024 | 0.537 (**0.544**) | 0.383 |

**G1 기각** (두 dualrc 모두: 이동 −50% 미달, 비용 CI가 0 제외). 그러나 **용량-반응**이 뚜렷하다:
refit 시작 층이 이를수록(24 → 16 → 0) 캐시 이동과 캐시 비용이 단조로 내려간다. s16은 dual 대비
비용 −0.025 [−0.075, −0.002]로 유의하게 좋지만 0에 못 미치고, s24는 dual과 같다.

**기전**: 깊은 층 캐시 이동은 상류 절단이 누적된 입력 이동이고, 층별 OSSCAR refit은 옮겨진 자기
입력 위에서 자기 출력만 맞춰 그 층 아래에서 온 이동은 되돌리지 못한다. 민감도 지도는 *무엇을*
지킬지 말하지만, 지키려면 *그 캐시를 흔드는 절단이 있는 층 전부*(첫 절단 층부터) refit해야 한다.

## 결과 — G2 val500 (2026-08-29 23:13 KST) 과 결론

| arm | minADE@6 | ΔminADE@6 vs 무압축 (중앙값 [CI]) | 평균 | CoC 붕괴율 |
|---|---|---|---|---|
| dual | 0.8904 | +0.0581 [+0.0361, +0.0829] | +0.067 | 0.014 |
| dualrc_s24 | 0.9153 | +0.0554 [+0.0347, +0.0827] | +0.092 | 0.016 |
| dualrc_s16 | 0.8852 | **+0.0249 [+0.0065, +0.0490]** | +0.062 | 0.016 |
| dualr | 0.8143 | +0.0030 [−0.0059, +0.0186] | −0.009 | 0.010 |

**G2 기각**(dualr 수준 회복 아님) — 그러나 G1의 용량-반응이 궤적에서도 그대로: 시작 층 24 → 16 → 0에
따라 +0.055 → +0.025 → +0.003. test500·OOD-val·G4는 계획 순서대로 돌리지 않았다.

**결론**: "민감한 캐시가 있는 층만 refit"은 틀린 문제 설정이었다. 캐시 이동은 상류 절단의 누적이라
보존의 단위는 *층*이 아니라 *체인*이고, refit을 첫 절단 층부터 하지 않으면 그 위에서 온 이동을
못 되돌린다. 민감도 가중 Hessian은 이 실험에서 분리 검증되지 않았다(refit 범위와 교락).
보고서 `reports/evaluation/2026-08-29_cache-targeted-reconstruction.html`.

남는 길: (a) dense-target refit(교차 모멘트 Σ x_p x_dᵀ, 층 l이 상류 오차까지 보정) — 과적합 위험,
(b) **dualr_w**: 전 층 refit 유지 + expert 가중·own-CoC Hessian만 바꾼 one-factor(dualr의 진짜 문제였던
CoC/폐루프를 겨냥), (c) 상류 절단을 "깊은 캐시를 얼마나 흔드는가"로 고르는 선택 기준(층별 기여 진단 선행).
