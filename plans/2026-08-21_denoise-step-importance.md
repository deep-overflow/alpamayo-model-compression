# Denoising-step별 중요도 분해 — action expert의 Taylor 기준이 magnitude에 지는 이유 (2026-08-21)

## 0. 관측된 사실 (재확인)

`outputs/expert_abl_summary/eval_summary.txt` (n=80 클립, K-sample minADE, 구 프로토콜),
expert tower만 마스킹, 레이어 내 균일 비율:

| ratio / scope | traj Taylor ΔminADE | magnitude ΔminADE | random ΔminADE |
|---|---|---|---|
| all r10 | **+0.0176** | −0.0021 | +0.0952* |
| all r25 | **+0.0776*** | +0.0091 | +0.4386* |
| all r40 | +0.1505* | +0.1671* | +0.6433* |
| all r50 | +0.4980* | +1.2935* | +2.0364* |
| early r25 | +0.0159 | +0.0090 | +0.1023* |
| early r50 | +0.1044* | +0.0517 | +0.5188* |

(* = 95% bootstrap CI가 0을 배제)

즉 **r10–r25에서 magnitude가 traj Taylor보다 낫고**(r25에서 Taylor만 유의한 회귀),
r50에서만 역전된다. random보다는 둘 다 훨씬 낫다. 사용자가 말한 "Taylor가 magnitude보다
못하다"는 이 구간의 관측이다. **같은 기준 계열이 VLM tower에서는 정반대**다: 매칭 예산 u40_v2 셀에서
activation-aware magnitude(Wanda)는 test_500 minADE 2.7706 / CoC degen 0.876으로 붕괴하고
dual Taylor는 0.7426 / 0.032이다 (`outputs/wanda_u40_v2_test`, `outputs/dual_u40_v2_test`).
즉 **타워를 바꾸면 기준의 우열이 뒤집힌다** — 원인은 기준 자체가 아니라 expert tower의
구조적 특성일 가능성이 높고, 그 중 가장 큰 구조적 차이가 "VLM은 1회, expert는 서로 다른 t로
10회 실행된다"는 점이다.

## 1. 코드 실사 — 현행 expert Taylor의 정확한 정의

`prune_lib.expert_fm_grads` + `run_importance.process_clip`을 읽어 확정한 사실:

1. **스텝 루프**: `for s in range(fm_steps=10)`, `t_val = (s+0.5)/10` ∈ {0.05, …, 0.95},
   스텝마다 새 noise ε_s를 뽑아 `x_t = (1-t)ε_s + t·x1`, 목표 `v* = x1 − ε_s`,
   `loss = MSE(v̂, v*)` → `loss.backward()`.
2. **게이트 grad는 스텝 간 zero되지 않는다.** 10번의 backward가 같은 게이트 텐서에
   누적되고, 루프가 끝난 뒤 `expert_gates.q_scores()`가 `|grad|`를 읽는다. 따라서
   유닛 u의 점수는

   ```
   I_u = | Σ_{s=0}^{9} ∂L_s/∂g_u |          ← 스텝 축은 "합의 절댓값"
   ```

   반면 **클립 축은** `acc["traj"]["exp_q"] += vlm/expert_gates.q_scores()`로
   클립마다 절댓값을 취한 뒤 더한다:

   ```
   I_u = Σ_{clips} | Σ_{steps} ∂L_s/∂g_u |   ← 두 축의 집계 규칙이 비대칭
   ```

   → **스텝 간 부호가 엇갈리면 상쇄되어 0으로 수렴한다.** 클립 축에는 없는 소거 경로가
   스텝 축에만 존재한다. (동일 비대칭이 `traj_vlm_*`·`traj_kv_*`에도 있다. 캐시 leaf의
   `.grad`가 10스텝 누적된 뒤 단 한 번의 VLM backward를 태우므로 역시 `|Σ_s|`이다.)
3. **측정 경로 ≠ 추론 경로**:
   - 측정: GT 직선 경로 위의 `x_t = (1−t)ε + t·x1`, t 격자 {0.05,…,0.95}, 스텝마다 독립 noise.
   - 추론(`FlowMatching._euler`): `x ← x + dt·v̂`, t 격자 {0.0, 0.1, …, 0.9}, x는 **모델 자신의
     반복값**(오차가 누적된 궤적), noise draw는 x_0 한 번뿐.
   → x_t 분포 불일치 자체가 Taylor를 나쁘게 만드는 **별개의 원인 후보**다.
4. magnitude 기준(`mask_lib.magnitude_scores`)은 `‖W_o[:, head]‖_F`, `‖W_down[:, c]‖_2` —
   스텝·데이터와 무관하므로 위 두 병리(부호 상쇄, 분포 불일치)에 면역이다. 이것이
   "왜 하필 magnitude가 이기는가"에 대한 정합적인 설명이 된다.

## 2. 가설

- **H0 (사용자 가설, 스텝 이질성)**: denoising 스텝마다 중요한 유닛이 다르다. 정적 마스크
  하나가 10 스텝 전부를 감당해야 하므로, 스텝 축을 뭉갠 집계는 기준으로서 손해다.
- **H1 (부호 상쇄)**: `|Σ_s g_s| ≪ Σ_s |g_s|`인 유닛이 상당수이고, 두 집계의 선택 결과가
  크게 다르다. 현행 점수는 "중요하지만 스텝별 부호가 다른 유닛"을 저평가한다.
- **H2 (스케일 지배)**: 스텝별 중요도 질량 `Σ_u |g_{u,s}|`가 s에 따라 수배 이상 차이 나고,
  합산 랭킹이 최대 질량 스텝의 랭킹과 거의 같다.
- **H3 (분포 불일치)**: 추론 궤적 위에서 잰 스텝별 중요도가 FM 직선 경로 위에서 잰 것과
  다르며, 전자가 실제 ΔminADE를 더 잘 예측한다.
- **H4 (실증 민감도)**: 같은 마스크를 **스텝 s에만** 적용했을 때 ΔminADE(s)가 s에 따라
  유의하게 다르다 — 스텝 민감도가 실재한다.
- **H5 (수리 가능)**: 스텝 축을 올바로 집계(`Σ_s|·|`, 스텝 랭크의 max, 스텝-정규화)하면
  매칭 예산에서 현행 합산보다 낫고 magnitude와의 격차가 줄어든다.

H1·H2는 "스텝 이질성이 왜 손해로 이어지는가"의 **구체적 기전**이고, H4는 기전과 무관하게
스텝 이질성 자체를 재는 gradient-free 실증이다. H5가 이 분석의 payoff다.

## 3. 측정 설계

공통: `calib_100` 100클립 (선택 신호는 항상 여기서만 측정), 클립 유래 시드
`sc.clip_seed(42, clip_id)`, 캐시된 샘플(`pre_processed/calib`), Ada 카드,
`run_retry_host.sh`로 런치. expert tower만 대상 (Q head 16/layer, MLP 8256/layer, 36 layer).

### A. per-step FM Taylor (현행 기준의 스텝 분해) — `stepimp_fm_v1`

`prune_lib`에 **새 함수** `expert_fm_grads_stepwise()`를 추가한다(기존 `expert_fm_grads`는
importance_v2 재현성을 위해 손대지 않는다). 스텝마다 backward 직후 게이트 grad를 읽고
zero → 스텝별 부호 있는 grad `g_{u,s}`를 그대로 저장.

- 저장: `step_importance.npz`
  - `g_signed_exp_q (S,L,H)`, `g_signed_exp_mlp (S,L,I)` — 클립 평균의 **부호 있는** 합
  - `g_abs_exp_q`, `g_abs_exp_mlp` — 클립 축 Σ|·| (스텝별 절댓값)
  - `kv_k (S,L,KV)`, `kv_v` — 캐시 leaf grad에서 스텝별로 (VLM backward 불필요, 공짜)
  - `loss (S,)`, per-clip 배열은 q/kv/mlp 전부 **fp32** (100×10×36×8256×4B ≈ 1.2 GB,
    nvme1n1 여유 324 GB 확인)
  - **fp16으로 저장했다가 실패했다 (2026-08-21)**: 이 기울기는 ~1e-7 크기라 fp16의
    subnormal 영역(min normal 6.1e-5)에 들어가고, MLP 항목의 **74.7%가 정확히 0으로
    underflow**했다. 재집계 시 상대오차 중앙값 0.22. 그 파일로 계산한 per-clip 통계
    (split-half 바닥, 클립 집중)는 MLP에 대해 전부 무효였다. Q head는 fp32라 무사.
    → `analyze_step_importance.perclip_fidelity()`가 이제 재집계 일치를 먼저 검사하고,
    통과 못한 유닛 타입은 per-clip 통계에서 자동 제외한다.
- **noise 모드 2종** (`--noise-mode`):
  - `per_step` (기존과 동일, 스텝마다 독립 ε) → G0 무결성 검증용
  - `shared` (클립당 ε 하나를 10개 t에 공유) → 스텝 비교에서 noise draw 분산을 제거한
    **짝지은(paired)** 비교. 본 분석의 주 측정.
- VLM backward 없음 → 클립당 ~3–4 s 예상 (기존 dual 측정이 5 s/클립).

### B. per-step Taylor on the inference trajectory — `stepimp_infer_v1`

배포 경로에서의 진짜 중요도. 자체 Euler 루프(‎`model.diffusion.sample`은 `@torch.no_grad`라
사용 불가)를 grad 켠 채로 돌리고, 스텝별로 다른 게이트를 태운다(`StepGates`: hook이
`current_step`을 보고 `gates[s]`를 곱함). 10스텝 그래프를 유지한 뒤 **한 번의 backward**로
모든 스텝의 grad를 얻는다.

- 목적함수: `task` — 최종 액션 → `action_to_traj` → xy, `MSE(xy, gt_xy)` (minADE와 정합,
  `action_to_traj`는 cumsum/cos/sin으로만 이루어져 미분 가능).
  - **GT-프리 대응물은 1차 Taylor로 만들 수 없다**: 게이트 값이 정확히 1이므로
    `MSE(x1_pruned, x1_dense.detach())`는 게이트에서 값도 기울기도 0이다. 출력 보존 기준을
    GT 없이 세우려면 Jacobian norm `‖∂x1/∂g_u‖`가 필요하고, 이는 랜덤 투영
    (r~N(0,I)에 대해 `⟨r, x1⟩` backward를 R회) 추정이 필요해 비용이 R배다.
    이번 범위에서는 제외하고, D 단계에서 필요해지면 별도 제안한다.
    - 용어 주의(2026-08-22 정정): 이 트랙에서 "label-free"는 세 가지 다른 것을 뭉뚱그려 왔다.
      **사람 주석**(어떤 기준도 안 씀), **모델 생성**(CoC-NLL Taylor는 롤아웃 필요, J-lens는
      불필요), **GT 데이터**(`I_traj`만 GT 액션 필요). 여기서 문제 삼는 것은 세 번째다.
- noise draw K=4개 평균(클립 유래 시드), 스텝별 부호 있는 grad와 |grad| 모두 저장.
- 메모리 추정: 스텝당 expert 활성값 ≈ 140 MB × 10 ≈ 1.4 GB 추가. `--reserve-gb 34`로 시작.

### C. 스텝-한정 마스킹 (gradient-free 실증) — `stepmask_v1`

기울기 가정 없이 H4를 직접 잰다. `PruneMasks`를 denoise 루프 안에서 스텝마다 갈아끼운다
(step_fn 래퍼가 t로 스텝 인덱스를 판정).

- 평가 클립: **`indist_500`의 앞 60클립** (calib이 아니다 — 선택/평가 분리)
- 마스크: r=0.40, 기준 2종 (현행 sum-Taylor `traj_exp_*`, magnitude)
- config: `only_s`(스텝 s에만 마스크) 10개 × 2기준 + `except_s`(s만 dense) 10개 × 1기준
  + baseline + full-mask 2개 = 33개, K=4 샘플
- **지표 2종** (구현 중 검정력 문제를 발견해 추가, 2026-08-21):
  - `dev` (**주 지표**): 같은 노이즈 시드의 **마스크 없는 경로와의 waypoint 평균 거리**.
    한 스텝만 마스킹하면 ΔminADE가 전체 마스크의 ~1/10(≈0.015 m)이라 n=60의 해상도
    (±0.03)에 묻힌다. `dev`는 같은 섭동을 자기 자신의 대조군에 대고 재므로 같은 n에서
    스텝 곡선이 분해된다. baseline의 `dev`는 구성상 정확히 0.
  - `minADE`/`minFDE` (부지표): 과제 지표. 부호와 크기는 보고하되 판정은 `dev`로 한다.
- 산출: dev(s)·ΔminADE(s) 곡선 각 2종 + except 곡선. 곡선이 평평하면 H4 기각.

### D. 집계 연산자 비교 (payoff, **H1/H2/H4 중 하나라도 확인될 때만 실행**) — `stepagg_*`

expert-only 마스크 arm 비교. 매칭 예산 r ∈ {0.25, 0.40}, 레이어 내 균일, expert 외 축 불변.

**2026-08-21 측정 결과를 반영해 arm 목록을 개정한다.** A/C 결과가 지목한 것은 부호 상쇄가
아니라 (1) 스텝 이질성과 (2) **클립 평균의 이상치 지배**이므로, arm은 두 축 각각을 겨냥한다.

| arm | 무엇을 바꾸나 | 근거 |
|---|---|---|
| `sum` | — (현행 `traj_exp_*`) | 기준선 |
| `trimclip` | **클립 축**: 평균 → 10% 절사평균 | step 0에서 1클립이 질량의 49%; 절사 시 선택 14% 변화 (겹침 0.858) — 스텝과 무관한 1줄 수정, 가장 높은 기대값 |
| `znorm` | **스텝 축**: 스텝별 레이어 내 z-정규화 후 평균 | 스텝 질량 max/min 7.7배 → 합산이 소수 스텝의 의견 |
| `maxrank` | **스텝 축**: 스텝별 레이어 내 랭크의 max | 최악 스텝 보호 (dual max-rank와 같은 철학); 스텝 겹침 0.82가 남기는 여지를 직접 겨냥 |
| `trimclip+znorm` | 두 축 동시 | 두 효과가 가산적인지 확인 |
| `damagewt` | 스텝 축: 스텝 s를 G4 실측 손상 `dev(s)`로 가중 | 스텝 9의 섭동이 최소 스텝의 5.2배로 출력에 실림 (기준 무관, 모양 r=0.990) |
| `sumabs` | 스텝 축: `Σ_s|g_s|` | 값싼 대조군. G1이 겹침 0.93을 줬으므로 **거의 변화 없을 것으로 예측** — 예측이 맞는지 자체가 검증 |
| `infer` | 측정 B의 추론-경로 점수 | G3 결과에 따라 포함 |
| `magnitude` | — | 참조(현재의 승자) |

- 평가: **고정 프로토콜** — rollout-only, K=8 per-sample 저장 → minADE@6/minFDE@6,
  `indist_500` 200클립으로 1차 판정, 승자만 test_500 전체로 확인.
- 클립당 VLM forward 1회를 모든 arm이 공유(`run_expert_ablation`과 같은 구조)하므로
  arm 수가 늘어도 비용은 denoise 수에만 비례한다.
- **주의**: 이 arm들은 전부 `outputs/stepimp_fm_perstep`의 per-clip 배열에서 계산되므로
  **새 GPU 측정이 필요 없다** — 점수 재집계 + 마스크 평가만 하면 된다.

## 4. 사전 등록 게이트

- **G0 (무결성, 통과 못하면 이후 전부 무효)**: `--noise-mode per_step`으로 잰
  `|Σ_s g_s|`의 클립 평균이 원본 `run_importance.py`의 `traj_exp_q`/`traj_exp_mlp`와 일치 —
  상대오차 median < 1e-3, 레이어별 Spearman ρ > 0.999. (bf16 autocast 비결정성으로 bitwise는
  요구하지 않는다.)
  - **비교 대상은 `importance_v2`가 아니라 `importance_v2_ada`**: 원본은 Blackwell에서
    측정됐고 이번 분해는 Ada에서 돈다. 아키텍처가 다르면 3–4% 클립의 CoC 텍스트가 달라져
    캐시가 달라지므로, 그 차이가 구현 오차로 오독된다. 같은 100클립·같은 시드로 원본을
    Ada에서 한 번 더 돌려 같은 아키텍처 기준선을 만든 뒤 비교한다.
    `importance_v2`와의 대조는 아키텍처 드리프트의 크기를 부기록하는 용도로만 쓴다.
- **G1 (H1 부호 상쇄)**: 상쇄비 `C_u = |Σ_s g_s| / Σ_s |g_s|`의 분포를 보고한다.
  - median C < 0.5 **이고** `sum` vs `sumabs`의 r=0.40 kept-set 겹침 < 0.90 → H1 채택.
  - median C > 0.9 → H1 기각(상쇄는 기전이 아니다).
- **G2 (H0/H2 스텝 이질성)**: 스텝 간 Spearman ρ(s,s′)의 median을, **같은 스텝 안에서의
  클립 split-half ρ**(잡음 바닥, 20회 랜덤 분할)와 비교.
  - across-step median ρ + 95% CI가 split-half ρ의 CI 아래에 완전히 놓이면 → 스텝 이질성이
    표집 잡음을 넘는다(H0 채택). 겹치면 기각.
  - 부기록: 스텝별 질량비 max/min, `sum` 랭킹과 최대 질량 스텝 랭킹의 kept 겹침(H2).
- **G3 (H3 경로 불일치)**: 측정 A(shared noise)와 측정 B(task)의 유닛 랭킹 상관.
  ρ < 0.7이면 경로 불일치가 실재. 어느 쪽이 C의 ΔminADE(s) 곡선을 더 잘 예측하는지
  (스텝별 예측-실측 상관)로 우열을 가른다.
- **G4 (H4 실증)**: 주 지표 `dev` 기준으로, 두 기준 중 **어느 하나라도** `only_s` 곡선에서
  max_s / min_s ≥ 2 이고 최댓값 스텝과 최솟값 스텝의 paired 차이 CI가 0을 배제 → 스텝
  민감도 실재. ΔminADE 곡선은 같은 형식으로 병기한다.
  - **해석 주의**: Euler 체인에서는 늦은 스텝의 섭동일수록 뒤에서 교정될 기회가 적으므로,
    `dev(s)`가 s에 따라 커지는 것 자체는 유닛 이질성이 아니라 **체인의 구조적 성질**일 수
    있다. "스텝마다 중요한 유닛이 다른가"는 G2가 답하고, G4는 "스텝이 서로 교환 가능한가"만
    답한다. 두 게이트를 함께 읽어야 한다.
- **G5 (H5 payoff, D 단계)**: r=0.40에서 `sumabs`/`maxrank`/`znorm`/`infer` 중 최소 하나가
  - `sum` 대비 paired ΔminADE@6 < 0 (95% CI가 0 배제), **그리고**
  - magnitude와의 격차를 50% 이상 축소 → 기준 트랙에 "스텝 축 집계" 규칙으로 반영.
  - 아무것도 통과 못하면: expert에서는 magnitude를 정직하게 기준으로 채택하고, 그 사실을
    보고서에 명시(리뷰어 관점에서도 유의미한 음성 결과).

## 5. 파일 구성

| 파일 | 상태 | 역할 |
|---|---|---|
| `experiments/head_analysis/prune_lib.py` | 수정(추가만) | `expert_fm_grads_stepwise()`, `StepGates` 추가. 기존 `expert_fm_grads`는 불변 |
| `experiments/head_analysis/run_step_importance.py` | 신규 | 측정 A/B (`--mode fm|infer`, `--noise-mode`) → `outputs/stepimp_*/step_importance.npz` |
| `experiments/head_analysis/analyze_step_importance.py` | 신규 | G0–G3 판정, ρ 행렬·kept 겹침 행렬·스텝별 질량·레이어 프로파일·상쇄비 분포 플롯 |
| `experiments/head_analysis/run_step_mask.py` | 신규 | 측정 C (스텝-한정 마스킹 sweep) |
| `experiments/head_analysis/verify_step_euler.py` | 신규 | 측정 B의 Euler 재구현이 공식 `FlowMatching.sample`과 같은 궤적을 내는지 검증 (같은 x₀를 양쪽에 주입) |
| `experiments/head_analysis/run_expert_agg.py` | 신규(D 단계) | 집계 연산자 arm 평가, 고정 프로토콜 |
| `experiments/head_analysis/stepimp_report_template.html` | 신규 | 보고서 템플릿 |
| `reports/evaluation/2026-08-2X_denoise-step-importance.html` | 산출 | 최종 보고 |

## 6. 비용과 실행 순서 (Ada 1장 기준, 현재 4장 모두 개루프 평가 중 → `run_retry_host.sh`)

1. `run_step_importance --mode fm --noise-mode per_step --num-clips 100` (~10 분) → **G0 판정**
2. `--mode fm --noise-mode shared` (~10 분) → G1/G2 판정
3. `--mode infer --num-clips 100 --k 4` (~30–40 분) → G3
4. `run_step_mask` 60클립 × 33 config × K=4 (~40–60 분) → G4
5. (게이트 통과 시) `run_expert_agg` val 200클립 × ~10 arm × K=8 (~2–3 시간)
6. 분석 + 보고서

합계 1 GPU-day 미만. 새 체크포인트(15–17 GB)는 만들지 않는다 — 전부 마스크 수준이고,
`mask_lib` 문서대로 마스킹은 제거와 기능적으로 동일하다. D에서 승자가 나오면 그때
`make_slim.py`로 물리 제거 체크포인트를 별도 승인 후 만든다.

## 7. 반증되면 다음 후보

G1·G2·G4가 모두 기각되면 스텝 축은 기전이 아니다. 그때의 다음 후보(별도 계획으로):

1. **게이트 위치**: expert의 o_proj/down_proj 입력 게이트가 잡는 1차 항이, 폭이 좁은
   타워(2048)에서 2차 항 대비 부정확할 가능성 → Hessian/OBS 계열(Tyr 트랙과 합류).
2. **GT 의존**: FM 목적이 GT 액션을 쓰는데 expert는 다봉(multi-modal) 분포를 학습한다.
   GT는 그 중 한 mode일 뿐이므로 "GT mode 재현에 쓰이는 유닛"만 높게 평가될 수 있다
   → 측정 B의 `self`(dense 출력 보존) 목적이 이 축의 대조군이다.
3. **클립 축 집계**: expert는 VLM보다 클립별 분산이 클 수 있다 → CVaR 트랙 재적용.

---

# D 단계 사전 등록 (2026-08-21, A~C 결과 확인 후 승인)

## 무엇이 D를 정당화하나

A~C가 확인한 것과 기각한 것:

| 발견 | 수치 | D에서 겨냥하는 arm |
|---|---|---|
| 스텝마다 유닛 랭킹이 다름 (G2 ACCEPT) | Q head 스텝 간 ρ 0.717 vs 잡음 바닥 0.929; 각 스텝 keep-set의 45.8%만 전 스텝 생존 | `znorm`, `maxrank` |
| 스텝 질량 불균형 | max/min 7.68 (Q head), peak = step 0 | `znorm` |
| 클립 평균의 이상치 지배 (사전 등록 외) | step 0에서 1클립이 질량의 49%; 10% 절사 시 선택 14% 변화 | `trimclip` |
| 손상이 스텝 위치에 비대칭 (G4 ACCEPT) | `only_s` dev max/min 5.15, 두 기준 간 모양 r=0.990 | `damagewt` |
| 부호 상쇄는 무해 (G1 INCONCLUSIVE) | C 중앙 0.637이나 겹침 0.925 | `sumabs` (음성 예측 검증) |
| 훈련/추론 경로는 대체로 일치 (G3 REJECT) | 집계 ρ 0.799 / 0.941, 단 s=0에서 0.50 | `infer` |

## Arm 정의 (전부 재집계, 새 GPU 측정 불필요)

레이어 내 균일 비율, expert tower만, KV·VLM 불변. 유닛 점수는 항상 레이어 내에서 argsort된다.

| arm | 점수 정의 |
|---|---|
| `sum` | `mean_clips |Σ_s g|` — 현행 `traj_exp_*`. **재구성이 원본과 일치해야 함(G5a)** |
| `sumabs` | `mean_clips Σ_s |g|` |
| `trimclip` | 클립 축 10% 절사평균 of `|Σ_s g|` (유닛별로 상위 10% 클립 제외) |
| `znorm` | 스텝별 레이어 내 z-정규화 후 스텝 평균 |
| `maxrank` | 스텝별 레이어 내 랭크 정규화 후 스텝 최대 |
| `trimclip_znorm` | 클립 절사로 스텝별 점수를 만든 뒤 `znorm` |
| `damagewt` | `Σ_s w_s · a_s`, `w_s ∝ dev(s)` (stepmask_v1 `only_s` traj 곡선, 합 1로 정규화) |
| `infer` | 측정 B(`stepimp_infer_v2`)의 스텝 합 — 배포 경로 |
| `magnitude` | `‖W_o[:,head]‖_F` / `‖W_down[:,c]‖_2` — 참조(현재의 승자) |

## 평가 프로토콜

- 비율 **r ∈ {0.25, 0.40}**. r25는 expert_abl에서 traj−magnitude 격차가 가장 큰 구간
  (+0.0776* vs +0.0091), r40은 두 기준이 비슷해지는 구간 — 두 지점을 모두 봐야 "언제
  뒤집히는가"를 말할 수 있다.
- **`indist_500`의 61~260번째 클립 200개.** 앞 60개는 `stepmask_v1`이 썼고 `damagewt`의
  가중치가 거기서 나오므로, 그대로 쓰면 그 arm만 평가셋에서 조율된 셈이 된다 (가중치가
  10개 숫자뿐이라 누수는 작지만 피하는 비용이 0이다).
- rollout-only, K=8 per-sample 배열 저장 → **minADE@6 / minFDE@6**, 호라이즌 1.6/3.2s 병기.
- 클립당 rollout 1회 + TF VLM forward 1회를 모든 arm이 공유 (expert 마스크는 VLM·CoC에
  영향이 없다). CoC NLL은 클립당 1회만 기록.
- 부지표 `dev_k`: 같은 시드의 마스크 없는 경로와의 waypoint 평균 거리 — arm 간 섭동 크기를
  낮은 분산으로 비교.

## 사전 등록 게이트

- **G5a (무결성)**: 재집계한 `sum` 점수가 `importance_v2_ada`의 `traj_exp_q`/`traj_exp_mlp`와
  일치 (레이어별 Spearman ρ > 0.999, kept-set 겹침 = 1.000). 실패 시 D 전체 무효.
  per-clip 배열은 fp64 집계본을 재현해야 한다 (fp16 underflow 사고 재발 방지, 상대오차 중앙값 < 1e-3).
- **G5b (주 판정, r=0.25)**: `sum` 대비 paired ΔminADE@6.
  - `trimclip` / `znorm` / `maxrank` / `trimclip_znorm` / `damagewt` / `infer` 중 최소 하나가
    **CI 전체 < 0** → 집계 개선이 실재. 기준 트랙에 반영을 제안한다.
  - 아무것도 통과 못하면 → **expert에서는 magnitude를 정직한 기준으로 채택**하고 그 사실을
    보고한다 (리뷰어 관점에서도 유의미한 음성 결과).
- **G5c (격차 축소)**: 승자와 `magnitude`의 paired ΔminADE@6 격차가 `sum`−`magnitude` 격차의
  50% 이상 줄어야 "설명이 맞았다"고 말할 수 있다. 줄지 않으면 집계는 부분적 원인일 뿐이다.
- **G5d (음성 예측 검증)**: `sumabs`는 `sum`과 사실상 같아야 한다 (kept 겹침 0.925에서 예측).
  크게 다르면 G1 해석이 틀린 것이므로 재검토.
- **G5e (부기록)**: r=0.40 동일 표, kept-set 겹침 행렬, `dev` 기준 arm 순위, 버킷별 분해.

## 비용

19 config (2 비율 × 9 arm + baseline) × K=8 ≈ 81 s/클립, 200 클립을 GPU 6·7 2-way 샤딩 →
카드당 약 2.3 시간.
