# J-space를 structured pruning에 적용

## Context

Anthropic, *Verbalizable Representations Form a Global Workspace in Language Models*
(transformer-circuits, 2026-07-06). Jacobian lens(J-lens)로 찾은 저차원 부분공간 J-space가
활성 분산의 10% 미만인데, 이를 제거하면 **얕은 태스크(MMLU/SQuAD/감성분류)는 무손상**인 반면
**multi-hop reasoning은 near-zero, free-form 생성은 Haiku 4.5 이하로 붕괴**한다.

이 서명은 우리 `integrated_mag`(−29.3%) 실패 양상과 구조적으로 일치한다:

| J-space ablation | integrated_mag |
|---|---|
| 분산 <10% 부분공간 손상 | 궤적 목적함수 기준 = 분산/크기 주도 |
| 얕은 태스크 무손상 | in-dist open-loop 무손상, closed-loop scene score Δ+0.002 (p=0.80) |
| multi-hop·free-form 붕괴 | CoC 퇴화 86.3%, OOD rollout ΔminADE +0.499 |
| — | at-fault 충돌 2→8, 100% 전방 |

**가설 H**: 궤적 손실만 보는 Taylor importance는 분산이 작은 J-space 방향을 우선적으로
제거하며, 이것이 CoC 퇴화 → 선행차 거리 판단 상실 → 전방 충돌의 표현공간 차원 메커니즘이다.

목표는 두 가지다. (1) **진단**: 현재 현상 기술뿐인 결과에 메커니즘 설명을 붙인다.
(2) **기준**: J-projection 기반의 라벨-프리 pruning 기준이 기존 CoC-NLL Taylor 항을
대체·보강할 수 있는지 검증한다.

## Key facts (검증 완료)

**논문 방법**
- `J_ℓ = E[∂h_final,t' / ∂h_ℓ,t]`, 소스 위치 t와 **모든 미래 위치 t' ≥ t**, ~1000 프롬프트 평균.
- J-lens 연산: `softmax(W_U · norm(J_ℓ · h_ℓ))`. J-lens 벡터 = `W_U J_ℓ`의 행(어휘당 1개),
  n_vocab > d_model이므로 **overcomplete basis**(직교 아님).
- J-space = J-lens 벡터들의 sparse nonnegative 결합, `min ||h − Vc||² s.t. ||c||₀ ≤ k, c ≥ 0`,
  k ≤ 25, gradient pursuit.
- 분산 기여 <10% (레이어별 상이). Ablation은 top-k=10 벡터를 잔차 스트림에서 사영 제거.
- 레이어 밴드: sensory(깊이 0–38%) / **workspace(38–92%)** / motor(92–100%), kurtosis + CKA로 식별.
- **n=10 프롬프트가 n=1000과 거의 동등, n=1도 "respectable"** → 캘리브레이션 비용이 낮다.

**우리 모델 (`importance.npz` 배열 shape에서 확인)**
- VLM: 36층, Q head 32, head_dim 128, d_model **4096**, intermediate **12288**,
  `model.vlm.lm_head` 존재 (`prune_lib.coc_nll`이 사용) → **J-lens 직접 적용 가능**.
- Expert: 36층, Q head 16, hidden **2048**, intermediate **8256**, **unembedding 없음**
  → J-lens 미적용, A-lens 필요(Stage D).
- KV group 8개. VLM 32/8 = 4 heads/group, expert 16/8 = 2 heads/group.
- 기존 점수: `outputs/importance_v1_n30/importance.npz`,
  `{coc,traj}_{vlm_q,vlm_mlp,kv_k,kv_v,exp_q,exp_mlp}`, 캘리브레이션 클립 50개.

**연산 구조 (핵심)**
- 출력 방향 u 하나에 대한 VJP 1회 = **모든 레이어 ℓ, 모든 소스 위치 t**에 대한 `∂/∂h_ℓ,t`를
  동시에 산출. 즉 `d_model`번의 backward로 36개 레이어의 `J_ℓ`이 전부 나온다.
  `torch.autograd.grad(..., is_grads_batched=True)`로 벡터화.
- `J_ℓ` (4096×4096, fp32 2.4 GB 전체) 을 한 번 만들어 두면 **유닛 점수는 전부 matmul**:
  - MLP 채널 c의 write 방향 = `down_proj[:, c]` (고정) → `W_U J_ℓ D_ℓ` 청크 계산.
  - Q head h의 write 부분공간 = `o_proj[:, 128h:128(h+1)]` (rank 128).
  - 유닛별 JVP를 돌릴 필요가 없다.

## 설계 결정

1. **소스 위치는 텍스트/CoC 스팬으로 제한.** Alpamayo 프롬프트는 vision token이 지배적이고
   이들은 verbalizable 표현의 담지자가 아니다. `analysis_lib`가 이미 vision / traj-history /
   sink / text 스팬을 식별하므로 이를 사용한다. (평균에 vision을 섞으면 J-lens가 희석된다.)
2. **Gate 통과 전에는 전체 J를 만들지 않는다.** r=256 랜덤 사영 스케치로 G2를 먼저 통과시킨다.
   상관계수 판정에는 JL 왜곡 범위에서 충분하다.
3. **레이어 배분 confound를 건드리지 않는다.** `cocsafe`와 `integrated_mag`는 기준과 레이어
   배분이 **둘 다** 다르므로(메모리 `dual-objective-claim-confound`), 두 config의 레이어
   프로파일 비교는 H의 증거로 쓸 수 없다. Stage C는 레이어별 정규화 후 기술 통계로만 보고한다.

## 파일

**신규**
1. `experiments/head_analysis/jlens_lib.py` — `build_jacobian(model, prompts, layers, rank=None,
   span="text")` (스케치/전체), `jlens_vectors(J, W_U, topk)`, `jspace_decompose(h, V, k)`
   (gradient pursuit), `unit_jscores(J, W_U, layers)` → `(L,32)` Q head + `(L,12288)` MLP.
2. `experiments/head_analysis/run_jlens.py` — Stage A/B 러너. `outputs/jlens_v1/`.
3. `experiments/head_analysis/analyze_jspace.py` — G2 상관 분석 + G3 회고 진단, 플롯.
4. `experiments/head_analysis/jspace_report_template.html` — 보고서 템플릿.

**수정 없음**: `prune_lib.py` / `mask_lib.py`는 Stage E 전까지 손대지 않는다.

## Stages

### Stage A — J-lens 구축 + 타당성 검증 (게이트 G1)
1. 캘리브레이션 클립 2개(`importance_v1_n30`의 clip_ids 앞 2개, train 소속 → test 청결 유지)로
   스케치 `J̃_ℓ` (r=256) 계산. 텍스트/CoC 스팬만 소스로 사용.
2. 각 레이어에서 J-lens readout 상위 토큰을 덤프한다.
3. 레이어 밴드 구조: 레이어별 excess kurtosis + CKA 블록 구조를 계산해 sensory/workspace/motor
   전이점을 찾는다. 논문 비율(0–38 / 38–92 / 92–100%)을 36층에 대입하면 workspace ≈ **층 14–33**.

**G1 통과 조건**: readout 상위 토큰이 주행 관련 어휘(제동/보행자/차선/합류 등)로 해석 가능하고,
kurtosis 곡선에 중간층 고원이 나타난다. 둘 다 실패하면 이 모델에 J-space가 (측정 가능한 형태로)
없다는 뜻이므로 **여기서 중단하고 그 자체를 negative result로 기록**한다.

### Stage B — 유닛별 J-projection 점수 (게이트 G2, 방향 전체의 생사)
1. `unit_jscores`로 VLM Q head `(36,32)` / MLP channel `(36,12288)` 점수 산출.
2. 레이어별 Spearman 상관을 계산: J-score vs {magnitude, `traj_vlm_*`, `coc_vlm_*`}.

**G2 통과 조건 (사전 등록)**
- (a) **신규성**: magnitude와의 상관 median |ρ| < 0.9. 0.95 이상이면 새 정보가 없으므로 중단.
- (b) **방향성**: `ρ(J, coc) > ρ(J, traj)` — J-score가 궤적 목적함수보다 CoC 목적함수에 가깝다.
  이것이 가설 H의 직접 예측이며, 뒤집히면 H는 기각이다.

### Stage C — 회고 진단 (기존 체크포인트, 재학습 없음)
`slim_cocsafe_r20` / `slim_cocsafe_r30` / `slim_integrated_mag`의 keep mask를 읽어, 제거된
유닛들이 보유하던 **J-space mass의 비율**을 config별로 계산. 총 write mass 대비로 정규화한다.

**예측**: 총 write mass 손상은 integrated ≤ cocsafe인데도 J-space mass 손상은 integrated ≫ cocsafe.
성립하면 "왜 composite score가 안전 붕괴를 가렸는가"에 대한 표현공간 차원의 답이 된다.
(설계 결정 3에 따라 레이어 배분 confound를 명시하고 레이어별 정규화 수치로 보고.)

### Stage D — Expert용 A-lens (신규 기여)
Expert에는 unembedding이 없으므로 `J^act_ℓ = E[∂(action_out_proj 출력) / ∂h^exp_ℓ]`를 정의한다.
출력 차원이 64×2 = **128차원뿐**이라 야코비안이 작고 전체 계산이 저렴하다(어휘 사영 불필요).
그러면 우리 기준이 `max(rank_norm(I_traj), rank_norm(I_CoC))` 휴리스틱에서
**"두 부분공간을 모두 보존하라"**는 기하 진술로 바뀐다.
KV 캐시가 두 타워를 잇는 실제 매체이므로, 캐시로 전달되는 J-space 성분의 잔존량도 함께 측정한다.
(단, "KV 캐시 = global workspace"는 가설을 만드는 유비이지 확립된 결과가 아니다. 그렇게 서술한다.)

### Stage E — 신규 기준 + mask 스윕 (G1–G3 전부 통과 시에만)
`score_u = ||P_J(Δh_u)||` 를 `mask_lib.select_mask`에 투입, 기존 r20/r30 비율에서
`run_ablation` 경로로 스윕. 비교 대상은 magnitude / traj-only / cocsafe / **J-score** /
**J+A 결합**. 물리적 수술과 폐루프 평가는 open-loop에서 cocsafe를 이긴 경우에만 진행한다.

## 산출물

`outputs/jlens_v1/{config.json, metrics.json, summary.txt, plots/}` (레포 규약),
보고서는 `reports/2026-07-XX_jspace-pruning.html` (`build_report.py`).

## Risks

| Risk | Mitigation |
|---|---|
| 연산 비용 (4096 backward × n) | 스케치 r=256로 G2 선통과, n=2에서 시작 (논문: n=1도 respectable) |
| Vision token이 J-lens를 희석 | 소스 위치를 text/CoC 스팬으로 제한 (설계 결정 1) |
| 주행 파인튜닝 모델에 J-space 부재 | G1이 정확히 이걸 검사, 실패 시 negative result로 종료 |
| J-score ≈ magnitude (기여 없음) | G2(a) 사전 등록, 하루 안에 판정 |
| 레이어 배분 confound | 설계 결정 3, 2×3 그리드 완료 전까지 기술 통계로만 |
| 스쿠프 (논문 3주 전 공개) | 우리 해자는 방법이 아니라 폐루프 안전 증거 (추론 손상 → 실제 충돌). Stage A–C를 먼저 끝낸다 |
| ICLR 일정 잠식 | Stage A–C(진단)까지가 기본, Stage D/E는 G2 통과 시에만 승격 |

## Verification

- G1/G2/G3 각 게이트의 사전 등록 조건.
- `metrics.json`에 레이어별 상관 행렬, config별 J-space mass 손상률, kurtosis/CKA 밴드 경계 수록.
- Stage B는 하루 내 판정 가능해야 한다. 초과 시 스케치 rank를 낮추고 재시도.

## 실행

```bash
bash experiments/head_analysis/run_retry.sh 20 experiments/head_analysis/run_jlens.py --gpu N ...
```
