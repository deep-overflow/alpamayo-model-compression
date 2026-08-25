# VLM 쪽 스텝 축 집계 수정 — 출시 config의 절반을 고칠 수 있는가 (2026-08-22 KST)

선행: `plans/2026-08-21_denoise-step-importance.md`,
`reports/evaluation/2026-08-21_denoise-step-importance.html`

## 0. 왜 이걸 하나

D 단계가 expert tower에서 확인한 것: 현행 trajectory Taylor 점수는 스텝 축을
`|Σ_s ∂L_s/∂g|`로 합산하는데, 스텝별 중요도 질량이 7.7배까지 차이 나므로 이 합이 소수 스텝의
의견이 된다. 스텝별로 레이어 내 z-정규화만 하고 합치면(`znorm`) r=0.25에서
ΔminADE@6이 **+0.0977 → +0.0003**으로 떨어지고 magnitude(+0.0601)를 추월했다.

**같은 결함이 VLM 쪽에도 있다.** `run_importance.process_clip`은 expert backward가 10스텝
동안 캐시 leaf에 누적한 grad를 **한 번의** VLM backward에 seed로 태운다:

```python
fm_loss, grads, leaves = pl.expert_fm_grads(...)   # grads = Σ_s dL_s/dcache
torch.autograd.backward(ts, gs)                     # → Σ_s dL_s/dg_vlm, 단 한 번
acc["traj"]["vlm_q"] += vlm_gates.q_scores()        # |Σ_s|
```

역전파는 seed에 대해 선형이므로 이 한 번의 backward가 주는 것은 정확히
`Σ_s (dL_s/dg_vlm)`이고, `q_scores()`가 절댓값을 취하므로 결과는 `|Σ_s|` — expert와 동일한
비대칭이다. `traj_kv_k`/`traj_kv_v`도 같다.

그리고 **이 `traj_vlm_*`가 출시된 `*_u40_v2` 패밀리 기준의 절반이다**
(`make_slim.build_masks`):

```python
dual_q = np.maximum(rank_norm(imp["traj_vlm_q"]), rank_norm(imp["coc_vlm_q"]))
```

`dual` / `j_traj` / `cocsafe` / `integrated_mag` 전부 이 `I_traj`를 쓴다. expert는 출시
config에서 건드리지 않는 축이었지만(u40_v2는 VLM only), VLM은 주력 config 그 자체다.
따라서 expert에서 확인된 수정이 여기서도 통한다면 **파급이 훨씬 크다**.

CoC 쪽 절반은 영향이 없다: `coc_vlm_*`는 diffusion 스텝이 없는 단일 손실
(`nll.backward(retain_graph=True)`)이므로 스텝 축 자체가 존재하지 않는다. 즉 이 수정은
`max(rank I_traj, rank I_CoC)`의 **한쪽만** 바꾸는 깨끗한 one-factor다.

## 1. 오늘 실측한 제약 (계획 수립 전 측정, `tmp/probe_vlm_bwd.py`)

expert는 스텝 분해가 공짜였다(기울기가 이미 스텝별이었고 루프가 더하기만 멈추면 됐다).
VLM은 아니다 — 스텝별 점수를 얻으려면 **스텝마다 VLM backward를 한 번씩** 해야 한다.
calib 클립 1개에서 실측:

| 항목 | 실측값 |
|---|---|
| VLM forward | 0.4 s |
| expert 10스텝 forward+backward | 1.6 s |
| VLM backward **1회** (현행) | 0.66 s |
| VLM backward **10회** (스텝별) | 6.58 s (정확히 10.0배) |
| 추가 비용 | **+5.9 s/클립** → 100클립 ≈ +10분 |

**우려했던 것보다 훨씬 싸다.** 클립당 총 ~13–14 s(롤아웃 포함), 100클립에 약 25분.

두 가지 실측된 함정:

1. **메모리.** 10스텝의 캐시 grad를 동시에 들고 있으면 peak가 **45.0 GB**까지 오른다
   (47.4 GB 카드에서 클립 2에서 OOM으로 죽었다). 현행 importance 런은 40.5 GB다. 해결책은
   저장이 아니라 **인터리브**: 스텝 s의 expert backward → 즉시 그 스텝의 VLM backward →
   게이트 점수(작다)만 남기고 캐시 grad는 버림. 이러면 캐시 grad 사본이 항상 1개뿐이라
   현행과 같은 ~41 GB로 돌아온다. 구현이 이 순서를 지켜야 한다.
2. **bf16 정밀도 바닥.** `|Σ_s g_s|`(스텝별 10회 backward를 더한 것)와 현행(합친 seed로
   1회 backward)의 상대차 중앙값이 **5e-03**이다. 수학적으로는 선형성에 의해 동일해야
   하지만 깊고 넓은 VLM 그래프를 bf16 autocast로 통과하기 때문이다. expert의 6e-05보다
   80배 크다. **무결성 게이트 문턱을 이에 맞춰 잡아야 한다** (1e-3은 통과 불가능).

## 2. 오늘 실측한 두 가지 근거 (가설을 강화/약화하는 쪽 모두)

**(a) 강화 — VLM의 스텝 질량 불균형이 expert보다 훨씬 심하다.**

| tower | 스텝별 중요도 질량 max/min (100클립 평균) |
|---|---|
| expert Q head | 7.68 |
| **VLM Q head** | **19.5** |

VLM 정규화 질량 프로파일: `1.000 0.363 0.224 0.127 0.103 0.075 0.060 0.051 0.053 0.051`
— step 0이 사실상 전부이고 뒤쪽 스텝은 사라진다. 현행 합산은 **거의 t≈0.05의 의견만
반영하는 셈**이다. expert에서 7.7배 불균형이 0.097 m의 효과를 냈으므로 여지가 있다
(단 효과 크기는 선형이 아니고, 아래 (b) 때문에 상한이 있다).

> **정정 (2026-08-22, 본 측정 후).** 계획 초안은 이 값을 **60.1**로 적었는데 그것은
> 프로브의 **단일 클립** 수치였다. 100클립 평균은 **19.5**다 (expert의 7.68도 100클립
> 평균이므로 이제 사과 대 사과 비교). 결론의 방향은 같다 — VLM의 불균형이 expert보다
> 2.5배 심하다 — 이지만 "8배"가 아니라 "2.5배"다.

**(b) 약화 — `max(rank, rank)` 가드레일이 변화의 절반을 흡수한다.**

`traj` 절반이 `max()`에서 실제로 binding인 비율을 측정했다: Q head **54.3%**, MLP **49.2%**.
즉 `I_traj`를 아무리 고쳐도 `dual`의 선택은 **최대 절반의 유닛에서만** 움직인다. 이 트랙에는
전례가 있다 — J-lens 8클립 선택 churn 25%를 `max(rank, rank)`가 흡수해 체크포인트를 다시
만들지 않았다(`jlens-32clip-noise-floor`). 그래서 아래 설계는 **순수 `traj` 기준을 민감도
프로브로 먼저 쓰고**, 그것이 움직여야만 `dual`로 간다.

**(c) 비용에 영향 — 아키텍처 드리프트가 선택을 2–3% 바꾼다.**

출시 config는 Blackwell에서 잰 `importance_v2`로 만들어졌고 이번 측정은 Ada다. 두 런의
u40_v2 선택 kept-overlap을 쟀다:

| | Q head | MLP |
|---|---|---|
| `traj` 단독 | 0.9810 | 0.9701 |
| `coc` 단독 | 0.9839 | 0.9733 |
| `dual = max()` | 0.9722 | 0.9729 |

**2–3%는 무시할 수 없다.** expert에서 znorm이 선택을 11% 바꿔 0.097의 효과를 냈으므로,
3% 드리프트는 그 4분의 1 규모의 교란이 된다. 따라서 **출시된 Blackwell 평가 행을 대조군으로
재사용할 수 없다** — 대조군도 Ada에서 만들어야 한다. 다행히 `importance_v2_ada`(같은 100클립,
같은 시드, 같은 레시피)가 이미 있으므로 **새 측정은 필요 없고 빌드+평가만** 하면 된다.

## 3. 가설

- **HV1 (주 가설)**: VLM의 `I_traj`도 스텝 축 합산 결함을 갖고 있으며, 스텝별 정규화로
  고치면 매칭 예산에서 개선된다. 질량 불균형이 60배로 expert(7.7배)보다 심하므로 효과가
  더 클 수 있다.
- **HV2 (가드레일 흡수)**: `max(rank I_traj, rank X)` 때문에 `dual`에서의 효과는 순수
  `traj`에서의 효과보다 작다. binding 비율 ~50%가 그 상한이다.
- **HV3 (싼 근사)**: 스텝별 VLM backward 10회 없이, **캐시 grad 질량으로 가중한 seed 1회**로
  znorm을 근사할 수 있다. 선형성에 의해 `Σ_s w_s (dL_s/dcache)`를 seed로 한 backward 1회가
  정확히 `Σ_s w_s (dL_s/dg)`를 주고, 캐시 grad는 expert 쪽에서 공짜로 나온다.
  오늘 실측한 `corr(캐시 grad 질량, VLM 게이트 질량) = 0.9861`이 근거다.
  참이면 앞으로의 VLM 재측정이 10배 싸진다.
- **HV4 (KV 축은 공짜)**: `traj_kv_*`도 같은 결함이 있고, 스텝별 캐시 grad는 이미
  `stepimp_fm_perstep_v2`에 저장돼 있으므로 **추가 측정 0**으로 고칠 수 있다. 단 KV는
  u40_v2 패밀리가 쓰지 않으므로(VLM only) `cocsafe`/`j_traj`/`integrated_mag`용 부기록이다.

## 4. 측정 설계

공통: `calib_100` 100클립(선택 신호는 항상 여기서만), 클립 유래 시드 `sc.clip_seed(42, clip_id)`,
`pre_processed/calib`, **Ada GPU 6–7만**, `run_retry_host.sh`.

### A. 스텝별 VLM 중요도 — `importance_stepvlm_v1`

`run_importance.py`를 건드리지 않고 새 러너를 만든다(출시 재현성 보존). 클립당:

1. 롤아웃 + teacher-forced VLM forward, 캐시 grad retain (현행과 동일)
2. CoC NLL backward → `coc_vlm_*`, `coc_kv_*` (스텝 축 없음, 현행과 동일하게 1회)
3. **스텝 s = 0..9 인터리브 루프**:
   - expert forward/backward (스텝 s) → 캐시 leaf grad = `dL_s/dcache`
   - 그 grad를 seed로 **VLM backward 1회** (`retain_graph=True`)
   - `vlm_gates.q_signed()` / `mlp_signed()` 읽고 **zero**, 캐시 grad도 zero
   - 스텝별 KV 점수는 leaf에서 바로 (공짜)
4. 저장: 스텝별 부호 있는 grad — **fp32** (fp16 subnormal underflow 사고 반복 금지,
   `expert-taylor-step-aggregation` 참조). 크기: q (100,10,36,32) 46 MB,
   mlp (100,10,36,12288) **1.8 GB**, kv 작음. nvme1n1 여유 확인 필요.
5. 부산물: 스텝별 캐시 grad 질량(HV3용 가중치), 스텝별 손실

**추가로 같은 런에서 HV3의 싼 근사도 잰다**: 각 클립에서 인터리브 루프가 끝난 뒤
가중 seed `Σ_s w_s (dL_s/dcache)` (w_s ∝ 1/캐시 grad 질량)로 backward를 **1회 더** 돌려
`seedznorm` 점수를 직접 얻는다. 정확한 znorm과 나란히 저장해 비교한다 (+0.66 s/클립).

### B. 재집계 → 드롭인 importance 파일 — `make_stepvlm_importance.py`

**`make_slim.py`를 수정하지 않는다.** 대신 기존 코드가 그대로 읽는 `importance.npz` 형식의
드롭인을 만든다. `importance_v2_ada`를 복사한 뒤 `traj_vlm_q`/`traj_vlm_mlp`(그리고 옵션으로
`traj_kv_*`)만 재집계 값으로 교체한다:

| 산출 디렉터리 | traj_vlm_* 집계 |
|---|---|
| `importance_stepvlm_sum` | `mean_clips \|Σ_s g\|` — 무결성 검증용 (V0) |
| `importance_stepvlm_znorm` | 스텝별 레이어 내 z-정규화 후 평균 (주 arm) |
| `importance_stepvlm_seedz` | 가중 seed 1회 근사 (HV3) |
| `importance_stepvlm_trimz` | 클립 10% 절사 + znorm (부기록) |

그러면 arm 빌드는 **코드 변경 0**:

```bash
python make_slim.py --config dual_u40_v2 --importance importance_stepvlm_znorm --no-state
```

이 방식은 이미 검증된 패턴이다 — CLAUDE.md에 따르면 `dual_u40_v2 --importance importance_v1`이
출시 `slim_dual_uniform`을 bit-identical하게 재현한다.

### C. 평가 arm (단계적, 게이트 통과 시에만 다음 단계)

예산·축은 u40_v2 셀 고정: uniform **0.3985632694**, VLM only, expert·KV 불변,
제거 파라미터 정확히 2,657,452,032. `--no-state`로 빌드(17 GB 체크포인트 불필요,
`load_slim()`이 base weights에서 재구성).

**Stage 2 — 순수 `traj` 쌍 (민감도 프로브, val500만)**

| arm | 빌드 |
|---|---|
| `traj_ada_ctl` | `--config traj_u40_v2 --importance importance_v2_ada` |
| `traj_znorm` | `--config traj_u40_v2 --importance importance_stepvlm_znorm` |

가드레일이 없으므로 수정 효과가 최대로 드러난다. 여기서 안 움직이면 `dual`에서는 확실히
안 움직인다(binding 50%).

**Stage 3 — `dual` 쌍 (배포 질문, 3셋 전부)** — Stage 2 통과 시에만

| arm | 빌드 |
|---|---|
| `dual_ada_ctl` | `--config dual_u40_v2 --importance importance_v2_ada` |
| `dual_znorm` | `--config dual_u40_v2 --importance importance_stepvlm_znorm` |

평가는 **고정 프로토콜**: rollout-only(`--no-tf`), K=8 per-sample 배열 저장 →
minADE@6 / minFDE@6 평균 우선(중앙값 병기), 호라이즌 1.6/3.2 s 병기,
**val 500 · test 500 · OOD-val 262**, Ada, paired seed.

## 5. 사전 등록 게이트

- **V0 (무결성)**: `importance_stepvlm_sum`이 `importance_v2_ada`의 `traj_vlm_q`/`traj_vlm_mlp`와
  일치. 문턱은 **오늘 실측한 bf16 바닥에 맞춰** 상대오차 중앙값 **< 2e-2**, 레이어별
  Spearman ρ **> 0.99**, u40_v2 kept-overlap **> 0.98**. (expert의 1e-3은 VLM에서 물리적으로
  불가능하다 — 1절 참조.) 실패 시 이후 전부 무효.
- **V1 (go/no-go, 무료)**: znorm이 선택을 실제로 바꾸는가.
  - `traj`: kept-overlap(sum, znorm) — **< 0.97**이면 진행. ≥ 0.99면 **중단**하고
    "VLM에서는 스텝 축이 선택을 바꾸지 못한다"로 보고(GPU 낭비 방지).
  - `dual`: 같은 수치를 부기록. HV2의 상한(≈0.5 binding)과 대조.
- **V2 (Stage 2 판정, val500)**: paired ΔminADE@6 (`traj_znorm` − `traj_ada_ctl`).
  - CI 전체 < 0 → HV1 채택, Stage 3 진행.
  - CI가 0 포함 → 순수 기준에서도 효과 없음. Stage 3 **취소**하고 음성 결과로 보고.
    (expert와 VLM에서 결론이 갈리는 것 자체가 보고 가치가 있다.)
- **V3 (Stage 3 주 판정, 3셋)**: paired ΔminADE@6 (`dual_znorm` − `dual_ada_ctl`).
  - test500에서 CI 전체 < 0 → 기준 트랙에 반영 제안. **단 폐루프 확인 전에는 배포 주장 금지**
    (개루프가 폐루프 실패를 가린 전례: `2026-08-11_criterion-closedloop`).
  - degen rate, minFDE@6, 버킷별 분해 병기.
- **V4 (HV3 싼 근사)**: kept-overlap(znorm, seedznorm) **> 0.98**이면 앞으로 VLM 재측정에
  10회 backward 불필요로 기록. < 0.95면 근사 폐기.
- **V5 (부기록)**: KV 축 재집계 결과(HV4), `trimz` arm의 선택 변화, 스텝별 질량·깊이
  프로파일, `traj` 절반 binding 비율의 znorm 전후 변화.

## 6. 파일 구성

| 파일 | 상태 | 역할 |
|---|---|---|
| `experiments/head_analysis/run_step_importance_vlm.py` | 신규 | 측정 A (인터리브 루프, fp32 저장, seedznorm 동시 측정) |
| `experiments/head_analysis/make_stepvlm_importance.py` | 신규 | 측정 B (재집계 → 드롭인 `importance.npz`) |
| `experiments/head_analysis/analyze_stepvlm.py` | 신규 | V0/V1/V4/V5 판정, 플롯 |
| `experiments/head_analysis/prune_lib.py` | 수정(추가만) | 인터리브 backward 헬퍼. 기존 함수 불변 |
| `experiments/evaluation/analyze_arms.py` | 재사용 | V2/V3 (기존 3-arm 게이트 스크립트) |
| `experiments/head_analysis/stepvlm_report_template.html` | 신규 | 보고서 템플릿 |
| `make_slim.py` | **수정 없음** | 드롭인 importance로 처리 |

## 7. 비용과 실행 순서 (Ada **6–7번만**, 2장)

| 단계 | 작업 | 비용 |
|---|---|---|
| 1 | `run_step_importance_vlm` smoke 2클립 (메모리 41 GB 확인) | 5분 |
| 2 | 본 측정 100클립 | ~25분, 1장 |
| 3 | 재집계 + **V0 / V1 판정** | 무료 (CPU) |
| 4 | Stage 2: 2 arm 빌드(`--no-state`) | ~40분 × 2, 2장 병렬 |
| 5 | Stage 2: val500 평가 2 arm | ~1.5 h, 2장 병렬 |
| 6 | **V2 판정** | 무료 |
| 7 | Stage 3(통과 시): 2 arm 빌드 + 3셋 평가 | ~4 h, 2장 병렬 |
| 8 | 보고서 | — |

**V1에서 멈추면 약 30분**, Stage 2까지면 약 3시간, 전부 가면 약 7–8시간.
새 17 GB 체크포인트는 만들지 않는다(`--no-state`) — 이것이 중요한 이유는 `/mnt/nvme1n1`이
**97% (여유 250 GB)** 이기 때문이다. 디스크 추가 소요는 스텝별 mlp 배열 **~1.8 GB**뿐.
(비교: 체크포인트를 만들었다면 arm당 17 GB × 4 = 68 GB.)

## 8. 반증되면 다음 후보

V1 또는 V2에서 멈추면 "스텝 축 집계 결함은 expert 고유이고 VLM에는 영향이 없다"가 결론이며,
그 자체가 보고 가치가 있다(두 타워가 같은 코드 결함을 공유하는데 결과가 갈리는 이유는
VLM이 CoC 절반과 `max()`로 묶여 보호받기 때문이라는 설명이 된다). 이후 후보:

1. **폐루프 우선.** expert 쪽 `znorm` 승리를 alpasim에서 확인하는 것이 VLM을 더 파는 것보다
   가치가 높을 수 있다 — 개루프 minADE가 폐루프 실패를 가린 전례가 있다.
2. **expert `znorm`의 물리 제거 체크포인트 + 3셋 전체 평가.** 현재 D 결과는 마스크 수준,
   200클립, r∈{0.25,0.40}이다. 고정 프로토콜 3셋으로 승격해야 논문 표에 들어간다.
3. **r=0.40에 남은 격차** (expert에서 두 축을 다 고쳐도 magnitude와 동률):
   1차 Taylor 근사의 한계(Hessian/OBS, Tyr 트랙 합류) 또는 GT 단일 mode 의존.
