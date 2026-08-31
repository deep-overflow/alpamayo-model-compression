# dualr_w — 전 층 refit은 그대로, Hessian만 바꾼다 (2×2: prefill 가중 × decode 몫)

날짜: 2026-08-30. 브랜치: `dualr-weighted`. 상태: **완료** (2026-08-30 19:30 KST).

## 왜

- `dualr`(dual 선택 + 전 층 OSSCAR refit)은 **캐시와 궤적은 지키지만 reasoning을 잃는다**:
  val500 +0.003(무압축 동률), test +0.015\*, OOD-val +0.034\*, 그런데 LingoQA **41.8**(dual 68.8,
  바닥 37.0) — CoC가 붕괴하진 않지만(degen 0.010) 내용이 사라진다. 폐루프는 0.792(+0.043\* vs
  baseline, −0.036\* vs dual, 충돌 +60%).
- 원인의 후보는 Hessian이다. dualr의 H는 prefill 토큰 균등, **decode(CoC) 토큰 0** — racfit이
  "prefill-only refit은 decode 경로를 mask-only보다도 악화시킨다"(o_proj +0.035)고 밝힌 바로 그
  설정이고, own-CoC를 16% 섞은 `d10` 혼합은 held-out decode 오차를 0.16–0.18 줄이면서 prefill
  오차는 거의 안 올렸다. 그 결과가 실제 모델의 LingoQA로 이어지는지는 아직 아무도 안 봤다.
- `dualrc` 실험(2026-08-29)은 "어디를 refit하느냐"가 문제였고 Hessian 가중은 분리 검증되지 않았다.
  이번엔 refit 범위를 dualr과 똑같이(전 층) 고정하고 **Hessian만** 바꾼다.

## 가설

- **H1 (decode)**: own-CoC를 decode 몫 0.16으로 섞으면 LingoQA가 dualr(41.8)에서 dual(68.8) 쪽으로
  유의하게 회복된다 — 캐시 보존(G1)과 궤적(G2)은 dualr 수준을 유지한 채.
- **H2 (expert 가중)**: prefill을 expert 어텐션 몫으로 가중하는 것은 캐시 보존을 더 좋게 하거나
  최소한 해치지 않는다. (캐시 참조 지도에서 "어텐션은 인과 참조의 대리가 아니다"가 나왔으므로
  기대는 낮다 — 그래서 별도 인자로 분리한다.)
- **H3 (폐루프, 탐색적)**: dualr의 폐루프 역전(GT 경로 추종↑·충돌↑)이 decode 경로 손상과 함께
  가는지 — H1이 서면 G4에서 본다. 기전이 다른 것(캘리브레이션 과적합)일 가능성을 열어 둔다.

## 설계 — 2×2 one-factor 격자

| arm | prefill 가중 | decode 몫 | 비고 |
|---|---|---|---|
| `dualr_rep` | 균등 | 0 | dualr **재현**(G0). 새 파이프라인이 dualr의 supernet과 같은 가중치를 내는지 |
| `dualr_d` | 균등 | **0.16** | H1 단독 — 본명 |
| `dualr_e` | **expert** (vision 0.722 / text 0.166 / hist 0.042 / sink 0.011, 클립 내 토큰 평균 정규화) | 0 | H2 단독 |
| `dualr_w` | expert | 0.16 | 둘 다 |

- 선택은 네 arm 모두 dual_u40_v2와 bit-identical(제거 −2,657,452,032). refit은 층 0–35 전부,
  블록 순차, damp 1e-2, `reconstruct_levels` — dualr supernet 빌더와 같은 절차.
- own-CoC: `racfit_v1/rollouts.json`의 무압축 모델 rollout, K=2 시드(decode 몫은 가중치이므로
  K는 다양성용). racfit은 K=4였다; 필요하면 K=4로.
- 코드: `run_cache_recon.py`에 `--start 0 --prefill-weights {uniform,expert} --decode-share {0,0.16}`
  (uniform 옵션만 추가), `make_slim.py`는 `dualrc_u40_s0` 경로 그대로(별칭 `dualr_<x>_u40`).
  상태 저장 필수(17 GB × 4 = 68 GB; 디스크 164 GB 여유 → 끝나면 rep는 삭제).
- **held-out decode 오차(무료 G3a)**: 빌드 중 클립을 해시로 A/B 폴드로 나눠, 각 층에서 B 폴드의
  decode 토큰 Hessian을 따로 쌓고 refit 후 `recon_error(W, W_hat, H_D_eval)`를 기록한다(racfit과
  같은 정의). 네 arm의 층별 decode/prefill held-out 오차가 표로 나온다 — LingoQA 전에 기전을 본다.

## 사전 등록 게이트

- **G0 재현**: `dualr_rep`의 모듈별 refit 가중치가 `slim_dualr_u40/slim_state.pt`의 kept 열과
  상대 Frobenius 차 < 1e-2 (bf16 저장·누적 순서 차이 허용). 실패하면 파이프라인 문제 — 진행 중단.
- **G1 캐시**(`run_cacheproxy.py`, val500 앞 200클립, dual·dualr과 페어드): dualr_d·dualr_w의
  민감도 가중 이동이 dualr 대비 [0.9, 1.1] 안, A10−A00 CI가 0 포함. 캐시 보존을 잃지 않을 것.
- **G2 궤적**(val500, Ada): Δ vs 무압축 중앙값 CI가 dualr(+0.003)과 겹치고, arm − dualr paired
  CI 상한 < +0.01.
- **G3 reasoning — 주 판정**:
  (a) held-out decode 재구성 오차(빌드 중): dualr_d·dualr_w < dualr_rep, 층 페어드 부트스트랩 CI가
  0 제외(racfit G2와 같은 형식);
  (b) **LingoQA**(`eval_lingo_arm.sh`, 500문항, Lingo-Judge, Ada): dualr_d − dualr paired 정확도
  차 CI > 0, 그리고 바닥(37.0) 대비 유의. dual(68.8)까지의 회복 비율을 보고;
  (c) CoC 붕괴율 ≤ dualr(0.010), OOD-val nll_self 부기록.
- **G4 폐루프**(150씬 × 2, Ada 4장, 8 h; G1–G3 통과 arm 하나만): score와 충돌률을 dual·dualr·
  baseline과 페어드. H3의 답.

판정 규칙: H1은 G3(a)+(b)로, H2는 dualr_e − dualr_rep(G1·G2·G3)로, 상호작용은 dualr_w −
(dualr_d + dualr_e − dualr_rep)로.

## 비용·순서

1. 4 supernet 병렬(GPU 0–3, 36블록 × 91 s ≈ 55분) → G0 → 4 빌드(각 3분).
2. G1 프록시 3 arm(GPU 0–2, 30분) ‖ val500 3 arm(Ada, 6 job ≈ 1.3 h) ‖ LingoQA 3 arm
   (Ada, arm당 ≈1 h; 워커 비는 대로).
3. 분석·보고서 → G4 결정. 개루프까지 **약 4 h**, G4는 별도 8 h.

## 이 계획이 답하지 않는 것
- 선택 기준 변경(상류 층 절단이 깊은 캐시를 얼마나 흔드는지) — 별도.
- dense-target refit — 별도.

### G0 (2026-08-30 15:00 KST): 가중치 공간 재현은 FAIL — 식별 불가 문제, 함수 공간으로 재정의

`dualr_rep` supernet vs `slim_dualr_u40`의 kept 열 상대 차: down_proj 중앙값 0.13(층 0–2에서 0.007–0.03,
깊어질수록 커짐 = 누적), **o_proj 중앙값 0.32, 층 1은 58.7배**. o_proj 입력은 헤드 간 공선성이 커서
최소제곱 해가 가중치 공간에서 식별되지 않는다 — 같은 출력을 내는 W′가 여럿이고 미세한 H 차이(fp32
누적 순서, K=2 pass, bf16 write-back)가 해를 크게 바꾼다. 따라서 G0를 **함수 공간**으로 재정의한다:
(i) `check_dualr_rep_fn.py` — 몇 개 층에서 ‖(W′_rep − W′_dualr)X‖/‖WX‖ (dense 입력 H, 10클립),
(ii) rep 체크포인트의 캐시 프록시·val500이 dualr과 CI 안에서 일치. 그리고 **rep를 파이프라인 내
기준 arm으로** 삼아 d/e/w를 rep와 비교한다(dualr은 2차 기준). 사전 등록 G0(가중치 1e-2)는 기각으로
기록하고, 원인은 식별 불가로 적는다.

### 두 번째 결함 (15:40 KST): 산발적 최소제곱 폭주 → safe_refit

G0 함수 공간 검사에서 rep의 **층 1 o_proj** 자기 오차가 1.55(dualr 0.038), 가중치 노름 4168(dualr 71)로
실제로 터진 해였다. G1 프록시도 rep가 망가졌음을 잡았다(CoC NLL +8.4, 얕은 층 캐시 이동 0.21 vs 0.03).
각 arm의 빌드 로그에서 "refit 오차 > mask-only 오차"인 모듈을 세니 rep 1개(층 1 o_proj), w 1개(층 7 o_proj),
d·e 0개 — 근-특이 H_kk의 fp32 해가 산발적으로 폭주하는 것이고, dualr 자체도 층 35 o_proj에서 같은 증상
(kept 열 상대 변화 6.6, max |w| 46)이 있었다. 반면 e(expert 가중, decode 0)는 dualr을 함수 공간에서
사실상 재현했다(캐시 비용 +0.001 vs +0.0015, 이동 비 0.530 vs 0.533, 층 띠 동일).

**수정**: `safe_refit` — 감쇠된 최소제곱은 fit 목표에서 mask-only 가중치보다 나쁠 수 없으므로(그 해가
feasible), 나쁘면 수치 실패로 보고 damp를 ×10, ×100으로 올려 재시도, 끝내 안 되면 mask-only 유지.
모듈별 damp와 fit 목표를 기록. rep·w supernet을 이 규칙으로 재빌드(d·e는 실패 모듈 0개라 변경 없음).
G1 프록시(200클립)가 두 번 연속 빌드 결함을 잡아냈다 — 이 프록시를 재구성 빌드의 표준 검사로 둔다.

## 결과 (2026-08-30 19:30 KST) — H1 기각(반대 방향), H2 중립, 예상 밖의 발견 하나

| arm | Hessian | G1 캐시 비용 A10−A00 | 이동 비 (rep) | val500 Δ | LingoQA | Δ vs rep |
|---|---|---|---|---|---|---|
| dualr (기존) | 균등, decode 0 | +0.002 | 1.001 | +0.003 | **41.8** | −10.4\* |
| **dualr_rep** | 균등, decode 0 (safe_refit) | +0.006 | 1.000 | +0.008 [−0.000, +0.022] | **52.2** | — |
| dualr_d | 균등, decode 0.16 | +0.012 | 1.043 | +0.018 [+0.007, +0.032] | **34.0** | **−18.2\*** |
| dualr_e | expert, decode 0 | +0.001 | 0.996 | +0.003 [−0.003, +0.014] | 49.6 | −2.6 n.s. |
| dualr_w | expert, decode 0.16 | +0.009 | 1.026 | +0.011 [+0.000, +0.030] | 49.0 | −3.2 n.s. |
| dual | (선택만) | +0.077 | 1.84 | +0.058 | 68.8 | +16.6\* |

- **G0**: 가중치 공간 FAIL(식별 불가), 함수 공간 PASS(층 0–24 ≤0.03; 층 35는 dualr 쪽이 터진 모듈).
  G1에서 rep = dualr 소수 3자리 일치.
- **G3a**: d·w의 in-sample decode 재구성 오차는 rep의 절반(o_proj 0.53 → 0.24, down_proj 0.41 → 0.13), rep의
  decode 오차는 mask-only보다 나쁨(racfit 재현). 그러나 **G3b LingoQA는 반대**: d는 −18.2pp(바닥 근처),
  w·e는 n.s. → in-sample decode fit은 암기(rank-deficient 3k 토큰의 정확 보간)이지 generalization이 아님.
- **예상 밖**: safe_refit만 적용한 rep가 기존 dualr보다 LingoQA **+10.4pp**(41.8 → 52.2), 캐시·궤적 동일.
  dualr의 reasoning 손실 중 열 포인트는 층 35 o_proj의 터진 최소제곱(max|w| 46) 탓. 남은 −16.6pp(vs dual)가
  재구성 자체의 비용.
- G4 미실행(G3 통과 arm 없음).

**결론**: 전 층 refit에서 Hessian 가중은 답이 아니다 — decode 토큰 혼합은 해롭고 expert 가중은 무효.
재구성의 reasoning 비용을 줄인 유일한 것은 **최소제곱을 제대로 푸는 것**(safe_refit)이었고, 그래도
dual보다 16.6pp 낮다. 보고서 `reports/evaluation/2026-08-30_dualr-weighted-hessian.html`.
