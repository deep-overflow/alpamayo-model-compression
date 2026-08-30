# dualr_w + LingoQA train — QA 텍스트를 refit Hessian에 넣으면 reasoning이 돌아오는가

날짜: 2026-08-30. 브랜치: `dualr-lingo`. 상태: **완료** (2026-08-31 05:30 KST).

## 왜

2×2(`plans/2026-08-30_dualr-weighted-hessian.md`)에서 own-CoC 16% 혼합은 LingoQA를 무너뜨렸다(d 34.0,
rep 52.2). 기전은 (1) own-CoC가 ≈3k 토큰뿐이라 4096·12288차원 H에서 rank-deficient → 해가 그 토큰을
정확히 보간(암기), (2) 주행 CoC 분포가 QA 텍스트로 옮겨가지 않음. LingoQA **train**(148,506 QA, 3,508
세그먼트)의 답변 토큰을 decode 스트림으로 쓰면 둘 다 없어진다 — 토큰이 충분해 보간이 불가능하고,
분포가 평가와 같다. 선택 기준 쪽에서는 LingoQA-train 문맥이 무효였지만(`vqa` 66.4 / `trajvqa` 71.2 vs
dual 68.8, n.s.), refit에서는 Hessian의 토큰 구성이 결정적임을 2×2가 보였다.

## 가설

- **H1**: dualr_w의 Hessian에 LingoQA train 샘플(VQA prefill + 참조 답변)을 더하면 LingoQA가 w(49.0)
  및 rep(52.2)보다 유의하게 오르고, 캐시·궤적은 rep 수준을 유지한다.
- **H1′ (대조)**: 같은 LingoQA 샘플을 expert 가중·own-CoC 없이(rep의 Hessian에) 더해도 같은 효과가
  나면, 효과는 QA 데이터이지 w의 나머지 요소가 아니다.

## 설계

| arm | 주행 prefill | own-CoC | LingoQA prefill | LingoQA 답변 | 역할 |
|---|---|---|---|---|---|
| `dualr_wl` | expert 가중 0.68 | 0.04 | 0.16 (vision/text/sink 몫) | **0.12** | 본명 (w + LingoQA) |
| `dualr_rl` | 균등 0.68 | 0 | 0.16 | **0.16** | 대조 (rep + LingoQA) |
| (기존) rep, w, dualr, dual, baseline | | | | | 비교 기준 |

- 몫은 샘플 내 토큰 평균 정규화(2×2와 같은 규약), decode 총 몫은 racfit이 검증한 0.16 유지.
- 선택 dual 고정, 층 0–35 블록 순차 OSSCAR refit, **safe_refit**, damp 1e-2.
- LingoQA 샘플: train에서 해시로 300 세그먼트 × 2문항 = **600 QA**(`run_vqa_importance.load_train_manifest`
  규약; 평가셋 val 500문항과 세그먼트 서로소 — 다른 이미지 트리). 입력은 `helper.create_vqa_message`
  (4프레임, 궤적 자리표시자 없음) + 참조 답변 teacher-forcing(`<|answer_end|>` 포함), `vqa_nll`과 동일.
  VQA prefill 구간: vision = 이미지 토큰, sink = 첫 토큰, text = 나머지(hist 없음 → text에 합산).
- 주행: calib_100, own-CoC K=1(비용). 블록당 forward ≈ 100 + 600 → ≈ 8분 → **36블록 ≈ 4.9 h** (두 arm 병렬).
- 코드: `run_cache_recon.py`에 `--lingo-questions N --lingo-prefill-share --lingo-answer-share` 추가
  (VQA 샘플 preload + 구간 가중치), 나머지 파이프라인 재사용.

## 평가 (사용자 요청: 네 벤치마크 전부)

- **G1 캐시 프록시**(200클립): 민감도 가중 이동이 rep 대비 [0.9, 1.1], A10−A00 CI가 0 포함.
- **G2 개루프 세 세트** val500 / test500 / OOD-val(262): 고정 프로토콜, Ada. wl·rl과 함께 **rep·w의
  test500·OOD-val도 채운다**(지금은 val500만 있음). 판정: arm − rep paired CI 상한 < +0.01(val500),
  세 세트 부호 일관성.
- **G3 LingoQA**(평가셋 500문항, rep 기준 paired): **wl − w CI > 0**(H1), rl − rep CI > 0(H1′),
  wl − rep, dual까지의 회복 비율. LingoQA train은 캘리브레이션에만 쓰였고 평가셋과 세그먼트 서로소임을
  보고서에 명시.
- G3a: 빌드 중 스트림별(주행 prefill / own-CoC / LingoQA 답변) in-sample 재구성 오차.
- G4 폐루프: G1–G3 통과 arm 하나만, 별도.

## 비용·순서

1. supernet 2개 병렬(GPU 0·1, ≈4.9 h) → safe_refit 로그 확인 → 체크포인트 2개 → G1 프록시(GPU 0·1, 30분).
2. Ada 큐: wl·rl × 3세트 + rep·w × (test, oodval) ≈ 10.8 GPU-h / 4장 ≈ 2.7 h; LingoQA wl·rl(13분씩).
3. 분석·보고서. 지금 시작하면 supernet ~01:30 KST, 평가 ~04:30 KST, 보고서는 아침.

## 확정 (2026-08-30 22:15 KST, 사용자 결정)
- **`dualr_wl`만** 빌드(대조 `dualr_rl`은 생략). 재빌드 가능한 산출물 삭제(dualrc s16/s24, dualr_d/e 체크포인트,
  결함 supernet 3개) → 디스크 38 → 113 GB.
- GPU: **Ada 4·5**(Blackwell은 다른 작업 중). 4번 = supernet(≈4.3 h) → 체크포인트 → 프록시 → 평가 워커 →
  LingoQA; 5번 = 그동안 rep·w의 test500·OOD-val(8 job, 최종 표용) → 이어서 wl 평가 워커.
- smoke(층 34–35, 3클립 + 3 세그먼트×2문항): VQA 프롬프트 ≈750 토큰, 답변 ≈23 토큰; 혼합 몫 0.68/0.04/0.16/0.12
  확인, L 스트림 오차 기록, safe_refit 승격 동작(damp 0.1·1.0). 본 빌드: 100 클립(K=1) + 600 VQA → 블록당
  ≈430 s.

## 결과 (2026-08-31 05:30 KST) — H1 채택: LingoQA 격차가 사실상 닫혔다

| arm | LingoQA | val500 Δ | test500 Δ | OOD-val Δ | 캐시 비용 | 가중 이동(rep비) |
|---|---|---|---|---|---|---|
| dualr (기존) | 41.8 | +0.003 | +0.015\* | +0.034\* | +0.002 | 1.00 |
| dualr_rep | 52.2 | +0.008 | +0.014\* | +0.029\* | +0.006 | 1.000 |
| dualr_w | 49.0 | +0.011\* | +0.027\* | +0.029\* | +0.009 | 1.026 |
| **dualr_wl** | **72.6** | +0.010\* [+0.001, +0.029] | +0.018\* [+0.005, +0.037] | +0.030\* [+0.004, +0.055] | +0.002 | 1.155 |
| dual | 68.8 | +0.058\* | +0.052\* | +0.078\* | +0.077 | 1.84 |
| 무압축 | 73.2 | — | — | — | — | — |

- **LingoQA**: wl 72.6 = w +23.6pp [+18.8, +28.4], rep +20.4pp [+15.4, +25.6]; dual(68.8)을 넘고
  **무압축(73.2)과 통계적으로 동률**(baseline − wl +0.6pp n.s.). 같은 decode 몫 16%가 own-CoC
  3k 토큰으로는 −18pp, QA 답변 9.7k 토큰으로는 +20pp — 2×2의 rank·분포 진단 그대로.
- **개루프 세 세트**: wl은 rep·w와 같은 수준(부호 일관, CoC 붕괴율 0.006–0.019). 가중 캐시 이동은
  rep의 1.155배(±10% 게이트 미달; LingoQA 몫 28%가 주행 prefill을 0.84→0.68로 줄인 대가)지만
  캐시 비용 +0.002·궤적 비용으로는 이어지지 않았다.
- 사다리: Tyr(붕괴) → dualr 41.8 → rep(safe_refit) 52.2 → **wl 72.6 ≈ 무압축 73.2**.
  재구성의 reasoning 비용은 고유한 것이 아니라 Hessian이 그 능력의 데이터를 본 적이 없어서였다.
- 한계: LingoQA train↔평가셋은 같은 벤치마크의 다른 split(세그먼트 서로소) — 분포 내 회복.
  G4 폐루프 미실행. 운영 사고 2건(공유 큐 phantom-increment race → claim 수정; 실행 중 스크립트
  편집으로 워커 bash 재파싱 사망)은 본문·메모리에 기록.
보고서 `reports/evaluation/2026-08-31_dualr-lingo-hessian.html`.
