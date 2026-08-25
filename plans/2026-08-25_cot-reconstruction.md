# CoT 재구성 (RAC) 이식 — 레이어별 출력 보존 한계 (2026-08-25)

## 배경

*Reasoning Models Can be Accurately Pruned Via Chain-of-Thought Reconstruction*
(arXiv:2509.12464). 주장은 한 줄이다 — 표준 LLM 프루닝은 레이어 출력을 **프롬프트 활성값**
에서만 재구성하는데 reasoning은 decode-dominated이므로, 재구성 캘리브레이션에 모델 자신의
**on-policy CoT 활성값**을 함께 넣어야 한다. 수식은 단순 열 결합이고, 따라서 Hessian은 가법:

```
min_Ŵ ‖(W − Ŵ)[X^P X^D]‖_F²   ⇒   H^RAC = H_P + H_D
```

이 가법성이 설계의 축이다. 스트림별로 H를 따로 쌓아두면 forward를 다시 돌리지 않고
임의 혼합비를 사후에 만들 수 있다.

### 이 저장소는 이미 논문이 지목한 쪽에 서 있다

`tyr_lib.py`가 레이어별 최소제곱 재구성을 이미 구현해 두었고(`HessianHook`,
`prune_levels`, `reconstruct_levels`), `run_tyr_supernet.py:142`가 자기 metadata에
`"hessian_tokens": "full fused prompt prefill, no labels"`라고 적는다. 100클립 ×
3,086토큰 = **308,600 프리필 포지션, 그중 93%가 vision**. 정확히 "input reconstruction"이다.
RAC 이식은 새 알고리즘이 아니라 **H를 쌓는 토큰 집합을 바꾸는 1-factor 변경**이다.

### 저장소가 이미 남긴 지지 증거

| 기록 | 수치 | 함의 |
|---|---|---|
| 커밋 `69efd68` | "reconstruction buys trajectory, costs language" | prefill-only 재구성은 vision→trajectory를 보존하고 언어를 희생 |
| Tyr T1 (`plans/2026-08-20_tyr-baseline.md:149-170`) | **미달** — `dual` 대비 궤적 이득 없음 | 재구성이 사는 값이 궤적이 아님 |
| LingoQA (`:230-265`) | `tyr_uniform_r` 20.8%, `tyr_r` 34.2% vs 퇴화 하한 **37.0%** (`dual` 68.8%) | prefill-H 재구성 후 언어는 하한 아래로 붕괴 |
| 폐루프 (`reports/evaluation/2026-08-23_tyr-closedloop.html`) | Týr degen 5.9% (empty 0.059 / soup 0.000) | 붕괴 양상이 "빈 출력" |
| 같은 리포트 §4 | "reconstruction is fitted to restore outputs on the AV prefill distribution … drifts and goes silent" | 저장소가 스스로 RAC의 문제 진술에 도달 |
| `plans/2026-08-21_dual-global.md:98` | "혼합 calibration H" — 미승인 후속 | 이 계획이 그 항목 |

### 아직 아무도 재보지 않은 것

Tyr는 OSSCAR 2차 목적값(`prune_loss`)을 계산하고 **버린다**(`tyr_lib.py:61`). 디스크에 남은
관련 산출물은 `outputs/tyr_hdiag.json`(4개 레이어 × level-0 damping 진단)뿐이다.
**"레이어별 재구성 오차 vs 희소도" 곡선은 이 저장소에서 계산된 적이 없다.**

### 결정적 사실: Alpamayo는 R1처럼 decode-dominated가 아니다

`outputs/wanda_txt_v1/records.json` 실측 (calib_100):

| 스트림 | 토큰/클립 | 100클립 합 | 비중 |
|---|---:|---:|---:|
| vision + traj-history + sink (**V**) | ~2,929 | ~292,900 | 94.9% |
| 프롬프트 텍스트 (**T**) | 157 | 15,700 | 5.1% |
| 모델 자신의 CoC rollout (**D**) | **15.4** (median 15, max 123) | **1,540** | **0.5%** |

논문은 CoT가 최대 8,192토큰이라 단순 concat만으로 H가 뒤집힌다. Alpamayo는 정반대여서
**그냥 concat하면 H가 0.5%만 바뀌는 no-op**이다. 충실한 이식은 concat이 아니라
**토큰수 정규화 혼합**이어야 한다:

```
H(w) = w_V·(H_V/N_V) + w_T·(H_T/N_T) + w_D·(H_D/N_D)
```

`w ∝ N`이면 현행 Tyr(= 논문의 naive concat)이고 `w_V = 0`이면 이미 기각된
`wanda --tokens text` 프로토콜이다. 텍스트만의 H는 조건수로 탈락한 이력이 있으므로
(`plans/2026-08-20_tyr-baseline.md:110-112`, `down_proj` tiny_eigs 11,825/12,288)
**혼합**이 미시도 영역이고, V가 조건수를 지탱해 주는 것이 혼합의 존재 이유다.

## 가설

- **H1 (메커니즘)** prefill-only H로 맞춘 재구성은 decode(CoC) 포지션에서 일반화하지
  못한다. D 스트림을 섞으면 held-out decode 재구성 오차가 유의하게 준다.
- **H2 (배분)** 레이어별 출력 보존 한계는 균일하지 않고, 혼합비에 따라서도 달라진다.

## 사전등록 게이트

| 게이트 | 판정 대상 | 통과 조건 | 실패 시 |
|---|---|---|---|
| **G0** 재현 | `nat`, damp 1e-2, layer 0, `--folds 1` | `slim_tyr_uniform_u40_recon`의 layer-0 재구성 가중치와 상대 Frobenius 차 < 1e-3 | 코드 경로가 기존 Tyr과 다름 → 수정 후 재실행 |
| **G1** 전제 | `diag H̄_V` vs `diag H̄_D` | Spearman ρ < 0.95 **AND** 상위-512 고유공간 에너지 중첩 `tr(H̄_D P_V)/tr(H̄_D)` < 0.9 | 두 스트림이 사실상 동일 → RAC 여지 없음, 음성 보고 후 종료 |
| **G2** 본안 | u40 레벨, `err_D(nat) − err_D(D10)` | 36층 paired bootstrap 95% CI가 0 배제, 중앙값 개선 > 2% | RAC가 Alpamayo에 이득 없음 |
| **G3** 선택 변화 | 혼합 간 kept-set 중첩 | Q < 0.860, MLP < 0.782 (calib_100 50:50 잡음 바닥) | 선택은 불변 — 효과는 순수 재구성분 |
| **G4** 비용 | `err_V(D10) − err_V(nat)` | 악화의 CI 상한 < +2% | vision 경로 손상 → 트레이드오프로 보고 |
| **G5** 배분 | ε=0.10에서 per-layer r*(ℓ) | 레이어간 표준편차 > 0.05 | 프로파일 평평 → uniform이 이미 최적 |

G1 실패도 결과다: "Alpamayo의 CoC는 프롬프트의 0.5%이고 통계적으로도 프롬프트와 구분되지
않아, 긴-CoT 모델에서 통하는 RAC가 짧은-CoC VLA에는 이식되지 않는다."

## 설계

- **스코프**: VLM 텍스트 타워 36층만. expert는 VLM KV cache를 읽으므로 VLM 레이어 입력에
  diffusion 토큰이 원리적으로 등장하지 않는다 → V/T/D 3스트림이 VLM에 대해 완결적이다.
  expert·KV 불변 (`u40_v2` 계열·Tyr 베이스라인과 동일 스코프).
- **모듈**: `self_attn.o_proj` (Q head 32그룹, group_size 128, `update_iter=1`),
  `mlp.down_proj` (채널 12,288그룹, group_size 1, `update_iter=16`).
- **혼합 6점**: `nat`(w ∝ N) · `VT`(w_D=0) · `D1`/`D10`/`D100`(D를 자연 비중의 1/10/100배) ·
  `Donly`(진단). 저장된 스트림별 H의 선형 결합으로 사후 생성.
- **keep 레벨 8점**: Q `{32,29,26,23,19,16,13,10}`, MLP
  `{12288,11264,10240,9216,7390,6144,4096,2048}`. **19/7390이 u40_v2 지점**.
- **선택**: 주 = OSSCAR 그리디(`prune_levels`). 부 = `dual` 고정 선택 + 재구성만
  (`reconstruct_levels`), u40 레벨에서만 — 선택 효과와 재구성 효과 분리.
- **damping**: 1e-2 (Tyr 최종 판정. `--damp 1.0`은 회귀). 민감도는 L02/L05/L17/L35만.
- **데이터**: `calib_100`, 2-fold(클립 해시 50:50) — held-out 오차만 보고. Tyr은 fit=eval이라
  이 구분이 없었다.
- **D 스트림**: 클립당 K=4 시드 rollout (temperature 0.6, 시드 `clip_seed(seed*1000+j, clip)`),
  ≈ 6,160 decode 포지션. 토큰 id는 `rollouts.json`에 한 번 캐시.

### 지표

```
E²(Ŵ, H_eval) = tr( (W − Ŵ) H_eval (W − Ŵ)ᵀ )
rel_err        = sqrt( E² / tr( W H_eval Wᵀ ) )
```

held-out fold의 스트림별 `H_eval`(토큰수 정규화)에 대해 `err_V / err_T / err_D`. 비교
기준으로 **재구성 없음**(kept 열만 남기고 0)도 같이 낸다 → "재구성이 사 준 양".

### 산출물

1. `err[ℓ, module, mix, level, eval_stream, {recon, mask}]` 격자 (`racfit.npz`).
2. `r*(ℓ, module | ε, stream)` — ε ∈ {0.05, 0.10, 0.20}의 레이어별 최대 제거율.
3. u40 예산(제거 2,657,452,032)에 물-채우기로 맞춘 **비균일 배분** `(rq, rm)` —
   `run_grid.allocations()`와 같은 표현(길이 36 float, prune fraction).
4. 혼합 간 kept-set 중첩, 조건수/유효랭크 진단.

## 실행

```bash
# 스모크
bash experiments/head_analysis/run_retry_host.sh 5 \
    experiments/head_analysis/run_racfit.py --gpu 0 \
    --exp-id racfit_smoke --num-clips 2 --blocks 2 --num-levels 3 --k-seeds 1
# 본실행 (~1.5 h, 카드 1장; 개루프 평가가 아니므로 Blackwell 0-3 사용)
bash experiments/head_analysis/run_retry_host.sh 240 \
    experiments/head_analysis/run_racfit.py --gpu 0 --exp-id racfit_v1
.venv/bin/python experiments/head_analysis/analyze_racfit.py --exp-id racfit_v1
```

## 범위 밖 (게이트 통과 후 별도 승인)

`run_tyr_supernet.py --streams/--mix` → error-accumulation 포함 supernet 재생성 →
`make_slim.py` `rac_u40` 브랜치(+ `--no-state` 금지 목록) → `launch_arms.sh`로
indist/test/OOD-val + LingoQA. 비교 상대는 `tyr_uniform_u40_r`, `tyr_u40_r`, `dual_u40_v2`.
핵심 판정 지표는 **LingoQA가 37.0% 퇴화 하한을 넘는가**.

## 상태

계획 승인 2026-08-25. 구현 진행 중.
