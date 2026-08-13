# 중요도 기반 양자화: 두 타워의 비트 배분 (2026-08-11)

> rev2 (2026-08-11) — 리뷰 반영. 주요 변경: 점수를 텐서 단위로 재정의, 1차 근사임을 명시하고
> finite-perturbation 검증을 Q2 진입 게이트로 승격, H1을 weight-space/task-space로 분리,
> H2에 등급형 CoC 지표와 메커니즘 프로브 추가, 배분을 rank에서 multiple-choice knapsack으로 교체,
> 예산을 실효 저장 바이트로 정의, `head`/`vit` 독립 arm 추가.

## 가설

프루닝 트랙의 결론은 "**추론(CoC)이 먼저 무너지고 주행은 버틴다**"였다
(`dual_uniform` −24%가 폐루프에서 baseline을 이겼고, CoC 건강도는 안전 대리변수가 아니었다).

양자화는 **반대 방향**일 것으로 예상한다.

| | 가설 | 지지 증거의 종류 |
|---|---|---|
| **H1a** | 동일 비트폭·그룹에서 expert 가중치의 RTN 재구성 오차가 VLM보다 크다 | weight-space. **아래 실측으로 이미 지지됨** |
| **H1b** | expert 양자화가 **절약 바이트당** 궤적을 더 크게 악화시킨다 | task-space. **미검증** |
| **H2** | 비트를 낮출 때 **궤적이 CoC보다 먼저** 무너진다 | task-space. 미검증 |
| **H3** | 동일 실효 저장 예산에서 태스크 손실 기반 비트 배분이 균일 RTN보다 낫다 | 미검증 |

**H1a와 H1b는 독립 축이다.** weight-space 재구성 오차가 크다고 downstream 태스크가 더 상하는
것은 아니다 — AWQ/GPTQ 계열이 activation-aware objective를 쓰는 이유가 정확히 이것이다.
논문 수준의 주장은 **H1b가 성립할 때만** 한다.

H1b·H2가 맞으면 최종 주장은

> **압축 연산자가 고장 양식을 결정한다. 구조적 제거는 CoC 추론을, 유한 정밀도 섭동은
> 연속 행동 생성을 우선적으로 파괴한다.**

가 되고, 그 위에 "태스크 인지 혼합 정밀도가 고정 저장 예산에서 손상된 능력을 부분 복원한다"(H3)를
얹는다. 이 대조는 프루닝 단독으로는 주장할 수 없다.

## 점수 정의

### 무엇을 재는가

프루닝 Taylor `|∂L/∂g|`는 **제거**에 대한 민감도다. 양자화에서 흔히 쓰는 논법 — "δ가 평균 0이라
1차항이 상쇄되므로 Hessian이 필요하다" — 는 δ를 확률적 라운딩 노이즈로 볼 때의 이야기다.
RTN 오차 `ε_b = Q_b(W) − W`는 **확률변수가 아니라 W의 결정론적 함수**이므로 그 논법이 적용되지
않고, 실제 섭동 방향을 그대로 probing할 수 있다.

텐서 `t`, 비트폭 `b`, 목적함수 `o`에 대해

```
S^o_{t,b} = | ⟨ ∇_{W_t} L_o , ε_{t,b} ⟩ | ,    ε_{t,b} = Q_b(W_t) − W_t
```

**정확한 성격 규정** (과대주장 금지):

> `S_{t,b}`는 정확한 결정론적 RTN 섭동에 대한 **태스크 인지 국소 방향 민감도**다. 확률적 양자화
> 노이즈를 가정하지 않는다는 장점이 있으나, 유한 양자화 효과에 대한 **1차 근사**로 남는다.

실제 유한 섭동은 `ΔL_b = ⟨∇L, ε_b⟩ + ½ ε_bᵀ H ε_b + …` 이고, ε가 결정론적이라고 해서 2차항이
사라지지는 않는다. **W3/W4처럼 섭동이 큰 영역에서 1차항이 유한 손실을 잘 순위화한다는 보장이
없으므로, 이 가정 자체를 Q2 진입 게이트로 검증한다** (Gate Q2-0).

### 단위: 텐서

Q2의 결정 변수가 `(tower, layer, module)` 텐서당 비트폭이므로, **섭동과 점수의 단위도 텐서여야
한다.** 그룹 단위 점수를 만들어놓고 텐서 단위로 배분하면 결정 입도와 민감도 입도가 어긋난다.
텐서 단위로 두면 누적기는 `#텐서 × #비트 × #목적함수` = 622×4×2 ≈ 5천 개 스칼라로 사실상 공짜다
(rev1의 652 MB 누적기는 불필요해진다).

구현상 축약 축만 설정으로 남겨둔다(`sum()` → `sum(-1)` 한 줄). 후속에서 per-channel
outlier 보호로 확장할 때 재설계가 필요 없도록 하되, **이번 계획에서는 텐서 단위만 사용한다.**

### 비용

α=0이 공통 작동점이므로 **하나의 목적함수에 대해서는 한 번의 backward로 모든 비트폭 점수가
동시에** 나온다 (서로 다른 b의 섭동이 작동점을 옮기지 않는다).

**목적함수는 별개다.** `S_traj`와 `S_CoC`는 서로 다른 gradient field `∇L_traj`, `∇L_CoC`를
요구하므로 **backward가 목적함수당 1회씩, 총 2회** 필요하다. `L_traj + L_CoC`를 한 번에
backward하면 각각을 복원할 수 없다. (`run_importance.py`가 이미 이 2회 구조다.)

### GPTQ/AWQ와의 차이

GPTQ는 레이어별 `‖WX − ŴX‖²`를 최소화하며, 그 오차가 궤적을 망치는지 추론을 망치는지 모른다.
우리 점수는 실제 두 태스크 손실에서 나온다. 이 구분이 기여 축이다. `jlens_lib`의 활성값 2차
모멘트를 쓰는 label-free 트윈도 가능하지만 이 계획 밖이다.

## 실측 근거 (2026-08-11)

### 양자화 표면

| pool | 텐서 | 파라미터 | bf16 | g64 패딩 필요 |
|---|---:|---:|---:|---:|
| VLM text Linear | 252 | 6.9458 B | 13.89 GB | 0 |
| expert Linear | 252 | 2.2791 B | 4.56 GB | 0 |
| lm_head | 1 | 0.6377 B | 1.28 GB | 0 |
| ViT Linear | 117 | 0.5742 B | 1.15 GB | 27 |
| **합** | **622** | **10.4367 B** | **20.87 GB** | 27 |

`embed_tokens`(0.6377 B)는 룩업이라 GEMM이 없어 제외한다. 622개 텐서 × 4개 비트폭은 정확한
DP로 풀기에 사소한 규모다.

### 그룹 크기와 실효 비트

행 방향(`in_features` 축) 그룹 양자화 기준으로 g64에서 나눠떨어지지 않는 것은 **ViT
`linear_fc2` (in=4304) 27개뿐**이다. 8256이 문제되는 지점은 정확히 말해 **expert `down_proj`의
양자화 축**이다 (`down_proj`는 (2048, 8256)이라 in=8256; `gate/up_proj`는 (8256, 2048)이라
in=2048로 g128도 나눠떨어진다). g64는 8256을 정확히 나눈다.

fake-quant의 zero-padding은 absmax/symmetric scale에서 거의 무해하지만, Q3의 packed kernel에서는
activation 측 layout까지 영향을 주므로 **fake-quant 패딩과 real-kernel 패딩은 별개 구현 문제로
취급한다.**

bf16 scale(그룹당 16비트)과 꼬리 패딩을 포함한 실효 비트 — **예산은 이 값으로 잡는다**:

| nominal | g64 실효 | g128 실효 |
|---:|---:|---:|
| 3 | **3.250** | 3.127 |
| 4 | **4.251** | 4.127 |
| 6 | **6.251** | 6.129 |
| 8 | **8.251** | 8.130 |

g64의 scale 오버헤드는 가중치당 +0.25비트로, W3에서 +8.3%다. **무시할 수 없으므로 H3 비교에
반드시 포함한다.**

### H1a의 근거 (weight-space만)

RTN 상대 재구성 MSE:

| 텐서 | kurtosis | W8g128 | W4g128 | W4g64 |
|---|---:|---:|---:|---:|
| VLM L00 q_proj | 3.5 | 4.3e-5 | 1.41e-2 | 1.18e-2 |
| VLM L17 gate_proj | 3.4 | 4.5e-5 | 1.48e-2 | 1.22e-2 |
| VLM L35 down_proj | 10.9 | 5.5e-5 | 1.79e-2 | 1.42e-2 |
| EXP L00 down_proj | 22.5 | 1.3e-4 | 4.33e-2 | 3.06e-2 |
| EXP L17 gate_proj | **76.1** | 2.9e-4 | **7.37e-2** | 4.55e-2 |
| EXP L35 gate_proj | **393.1** | 1.9e-4 | 4.26e-2 | 2.73e-2 |
| lm_head | **324.9** | 3.0e-4 | 5.80e-2 | 3.14e-2 |

VLM은 거의 가우시안(kurt 3.3~4.7)이고 깊이에 무관하게 균일하다. expert MLP는 kurt 8~393의
heavy-tail이라 2.5~5배 어렵다. lm_head도 outlier가 심하다. **이는 H1a를 지지할 뿐 H1b와는
무관하다.**

### 사전 sanity (완료)

- `tie_word_embeddings: False`. `embed`와 `lm_head`는 서로 다른 샤드에 저장되고 byte-identical이
  아니다(max abs diff 0.783). → **`head-only`는 실재하는 독립 개입이다.** 그래도 런타임
  `data_ptr()` alias 검사를 startup assertion으로 넣는다.

### 환경

양자화 라이브러리는 없다. torch 2.8.0+cu128이 `_weight_int4pack_mm`, `_weight_int8pack_mm`,
`_scaled_mm`을 노출하지만 **underscore가 붙은 private API이므로 "네이티브 지원 보장"이라고 쓰지
않는다.** Q3 착수 전 별도 microbenchmark로 shape/group/dtype 제약, layout 변환 비용,
compile 호환성을 확인하고, pack 시간이 1회성 로드 비용인지 추론 경로 비용인지 분리한다.

## 설계

`mask_lib → slim_lib` 2단 구조를 복제한다. **Q1/Q2는 fake-quant** (quantize–dequantize를 bf16에서
수행). 가중치를 한 번 반올림하면 그 뒤로는 평범한 bf16 모델이므로 런타임·결정성이 baseline과
동일하고, 정확도 실험에 인프라 리스크가 0이다.

> **논리적 절약 ≠ 물리적 절약.** fake-quant는 dequantize된 bf16을 들고 있으므로 실제 GPU 메모리는
> 줄지 않는다. 모든 플롯의 축과 표 헤더는 **`projected weight storage saved (GB)`** 로 표기하고,
> 실제 메모리·지연은 **Q3에서만** 주장한다.

### Q1 — 민감도 사다리 (중요도 불필요)

| 축 | 값 |
|---|---|
| 비트폭 | W8 / W6 / W4 / W3 (g64, symmetric, per-group bf16 scale) |
| 범위 | `vlm` / `exp` / **`head`** / `both` / `both+head` / (`vit`) |
| 고정 | activation bf16 유지(W-only). `embed` 미양자화 |

`head`를 독립 arm으로 반드시 넣는다. lm_head는 kurt 324.9 / W4 MSE 5.8e-2로 **H2의 강한
교란 요인**이다 — `both+head`에서 CoC가 무너지면 원인이 VLM 추론 타워가 아니라 lm_head의
극단 outlier일 수 있다. lm_head의 한계 효과는 `Δ(both+head) − Δ(both)`로 분리한다.
`vit`는 여력이 되면 추가한다(prefill 전용이라 지연 메커니즘도 expert와 다르다).

activation 양자화는 범위 밖이다. expert가 메모리 대역폭 바운드라 W-only가 속도 이득의 대부분을
가져가고, A-quant는 보정 복잡도만 늘린다.

### 능력 판독 (Q1과 프로브 공용)

타워 개입과 능력 판독을 **직교하게** 분리한다. 이래야 "궤적이 먼저 무너진다"가 expert-only arm을
포함시켰기 때문에 자명하게 나온 결과라는 비판을 피할 수 있다 — 진짜 검증은 **`vlm`-only에서
CoC가 상하고 궤적은 상대적으로 보존되는가**이다.

```
개입:  vlm-only | exp-only | both
판독:  (a) CoC:  ΔNLL, baseline 토큰 top-1 일치율, degeneracy(중증 지표)
       (b) expert: 1-step FM 오차, Euler step별 오차 누적
       (c) 최종:  minADE/minFDE@8
```

**CoC 판독은 등급형이어야 한다.** degeneracy 0.05는 완전 붕괴에 가까운 지점이고, 그 전에 NLL
증가·토큰 일치율 감소·logit margin 축소가 먼저 온다. degeneracy는 **중증 지표로만** 쓴다.

측정 프로토콜: baseline이 생성한 CoC 토큰열을 양자화 모델에 **teacher-force**하여 같은 위치에서
per-token NLL과 argmax 일치를 잰다. 페어드이고 샘플링 노이즈가 없다.

### 메커니즘 프로브 (Q1 이후, Q2 이전)

두 가설의 **메커니즘 증거**를 만든다. 비용이 헤드라인 대비 거의 없다.

**P1 — CoC logit margin vs 섭동.** baseline에서 생성 위치마다 top-1/top-2 logit 간격
`m_t = z_(1) − z_(2)`를 저장하고, 양자화 모델의 같은 위치 logit 섭동 `|Δz_t|`와 비교한다.
H2가 맞다면 W8→W4 구간에서 한동안 `|Δz_t| < m_t`라 토큰 정체성이 유지되다가 어느 지점에서
flip이 급증해야 한다. 이러면 "이산화가 오차를 흡수한다"가 추측이 아니라 측정이 된다.

**P2 — Euler step 오차 누적.** 각 denoise step k에서 `e_k = ‖x_k^Q − x_k^FP‖`를 기록한다.
H1의 누적 메커니즘이 맞으면 `e_1 < e_2 < … < e_10` 또는 최소한 말단 증폭이 보여야 한다.
**첫 expert forward에서 이미 대부분의 오차가 나고 이후 커지지 않으면 원인은 적분 누적이 아니라
단순 회귀자 민감도이므로, 그 경우 H1의 서술을 고친다.**
`exp`-only arm에서 재면 VLM 캐시가 bit-identical이라 순수 expert 효과가 분리된다.

### Q2 — 예산 인지 이산 최적화 (Gate Q1-C, Q2-0 통과 시에만)

**사전 등록 — Q2 텐서 풀은 `vlm + expert` (504텐서, 9.225 B, bf16 18.45 GB).** 두 타워 비트
배분이 H1b/H3의 가장 깨끗한 형태이기 때문이다. `lm_head`와 ViT는 secondary extension으로만
붙인다(ViT는 prefill 전용이라 지연 메커니즘 자체가 다르다).

`rank(S)` 기반 배분을 쓰지 않는다. 혼합 정밀도는 텐서마다 `b_t ∈ {3,4,6,8}`을 고르는
**multiple-choice knapsack**이고, 필요한 정보는 "텐서 t가 중요한가"가 아니라 **"t를 3→4로 올릴 때
얼마나 이득인가, 6→8은?"** 이다. 단순 rank 배분은 텐서 크기 차이, 비트별 비선형 민감도,
scale 메타데이터 비용을 전부 버린다.

```
minimize   Σ_t D_{t,b_t}
subject to Σ_t C_{t,b_t} ≤ B
```

- `C_{t,b}` = 실효 저장 바이트 = `N_t·b + N_groups,t·16 + padding` (위 표의 실효 비트 정의)
- `D_{t,b}` = `S_{t,b}` (목적함수별 정규화 후)
- 622(주 풀은 504) 텐서 × 4 선택지이므로 **예산을 이산화한 정확 DP**로 푼다. greedy 한계효용
  `U = (D_{t,b1} − D_{t,b2}) / (C_{t,b2} − C_{t,b1})` 은 텐서별 D–C 곡선이 볼록할 때만 최적이라
  **DP를 주로 쓰고 greedy는 sanity 대조로만** 둔다.

**목적함수 결합.** knapsack은 랭크가 아니라 **크기**를 요구하므로 `max(rank, rank)`가 구조적으로
맞지 않는다. 주 방식은 목적함수별 robust 정규화 후 max:

```
D^dual_{t,b} = max( S^traj_{t,b} / Z_traj ,  S^CoC_{t,b} / Z_CoC ) ,   Z = 90th percentile
```

| arm | 배분 | 역할 |
|---|---|---|
| `uniform` | Q1 무릎점 비트폭, 전 텐서 동일 | **대조군** |
| `dual_mag` | 위 정규화 damage로 DP | **primary** |
| `traj` | `S_traj`만으로 DP | secondary |
| `dual_rank` | `max(rank, rank)` 유사 배분 | 프루닝 연속성 ablation (appendix) |

### Q3 — 실커널 / latency (이번 계획 범위 밖)

**Q2의 비트 집합과 Q3의 커널 지원 비트 집합은 일치하지 않는다.** `{3,4,6,8}` 혼합이 이기더라도
W4 중심의 packed kernel로 그대로 구현되지 않을 수 있다. 이번 계획은 **science-first** — Q2는
fake-quant로 **배분 원리**를 검증하고, Q3는 지원 비트로의 *projection*이다. "Q2 결과가 Q3로
그대로 이어진다"고 쓰지 않는다.

expert는 FLOP의 5.5%인데 wall-clock의 22~28%인 step-bound 구간이므로 weight-only의 이득이 가장
클 것으로 예상하지만, 이는 Q3에서만 주장한다. `bind_identity()`가 프루닝에서 했던 역할(공정한
페어드 latency)에 해당하는 장치가 필요하다.

## 평가 프로토콜

`run_baseline.py`에 `--quant` 분기만 추가하면 나머지가 전부 재사용된다.

| 단계 | 셋 | n |
|---|---|---:|
| Q1 스크리닝 | val 500의 **버킷 층화 200** | 200 |
| 헤드라인 | val 500 + test 500 + OOD 1,533 | 2,533 |

- 스크리닝 200은 `outputs/baseline_indist`에 이미 기록된 시나리오 버킷으로 층화하고, 버킷 내에서
  `sha256(clip_id)` 순으로 비례 배분해 결정론적으로 뽑는다(decel_stop 70 / turn 77 / accel 72 /
  cruise 281 → 28/31/29/112). **매니페스트 선두 200은 쓰지 않는다** — greedy prefix가 무작위보다
  대표적이라는 보장이 없고, 오히려 특정 희소 버킷을 초기에 과대표집할 수 있다.
- 스크리닝은 **screening subset only; no inferential claim is made from it.** 모든 게이트 판정은
  전량에서만 한다.
- 페어드 시드 `sha256(f"{seed}:{clip_id}")[:4]`, k=8, seed 42, `MODEL_REV=7aba8293…` 고정
- 이미 측정된 baseline(val 0.690 / test 0.777 / OOD 0.842)과 직접 비교 가능

캘리브레이션은 `calib_100` (`importance_v2`/`jlens_v2`와 동일 100클립, 동일 revision).

**통계.** 주 분석은 클립별 페어드 대비 `d_i`의 **중앙값 + 페어드 부트스트랩 95% CI**
(`eval_lib.paired_bootstrap_ci`). Wilcoxon signed-rank는 대칭성 가정이 있고 중앙값 차이를 직접
검정하지 않으므로 **secondary p-value**로만 보고한다.

## 사전 등록 게이트

**Gate Q1-A (타워 비대칭)** — 두 비교를 병렬로 본다. 비율 하나만 쓰지 않는 이유는 `ΔL(C)`가
압축량에 선형일 이유가 없고 4.56 GB와 13.89 GB는 작동점이 아주 다르기 때문이다.

- *Primary (mechanistic)*: **동일 비트폭**에서 `exp`-only vs `vlm`-only의 페어드 median ΔminADE.
  "같은 수치 정밀도 섭동에서 어느 타워가 더 민감한가."
- *Efficiency-normalized*: W8/W6/W4/W3 네 점으로 타워별 **(projected saved GB, ΔminADE) 곡선**을
  그리고 국소 기울기 / Pareto frontier를 비교. **곡선 자체가 결과물**이고 `Δ/GB`는 보조 요약이다.

`vlm ≥ exp`이면 **H1b 기각** — expert는 프루닝에도 양자화에도 여유가 있다는 뜻이고, Q2의 배분
축이 타워에서 레이어 깊이로 바뀐다.

**Gate Q1-B (능력 비대칭, H2)** — 비트폭을 낮추며 어느 임계를 먼저 넘는지:

| 능력 | 임계 |
|---|---|
| 궤적 | median ΔminADE > **+0.05 m** (이 프로토콜의 분해능 하한, 깊이 ablation Gate D / 기준 비교 Gate C1과 동일) |
| CoC 충실도 | ΔNLL 또는 baseline 토큰 top-1 일치율 임계 (Q1 첫 결과에서 baseline 분산을 보고 확정, **데이터를 보기 전에 고정**) |
| CoC 중증 | degeneracy > 0.05 (baseline 0.006~0.008) |

궤적이 먼저 넘으면 H2 지지. **판정은 `vlm`-only arm에서 한다** — `exp`-only에서 궤적이 먼저
상하는 것은 자명하므로 검정력이 없다.

**Gate Q1-C (무릎점)** — **`vlm+expert` 범위 균일 양자화** 기준, val 500에서 median ΔminADE
≤ +0.05 m를 만족하는 가장 낮은 비트폭. 이것이 Q2의 실효 저장 예산이 된다. 무릎점이 W8이면
(= W4가 이미 깨지면) Q2의 여지가 작으므로 계획을 재검토한다.

**Gate Q2-0 (점수 타당성) — Q2 진입 필수 조건.** 1차 근사가 유한 RTN 효과를 순위화하는지 직접
검증한다. `S_{t,b}` 분위수별로 텐서 50~100개를 뽑아 **하나씩만** 실제 W4/W3 fake-quant하고
실제 `ΔL_{t,b}`를 잰 뒤 Spearman ρ를 본다.

- **ρ ≥ 0.7** → 통과. 이 결과 자체가 독립적인 기여다.
- **ρ < 0.7** → Q2 중단. 이 경우 H3 실패는 "적응적 비트 배분이 나쁘다"가 아니라
  **"1차 proxy가 유한 RTN을 예측하지 못했다"** 이므로, 두 결론을 반드시 분리해 보고한다.
  대안은 2차항 포함(레이어별 Hessian-vector product) 또는 layerwise 재구성 기준으로의 후퇴.

**Gate Q2-1 (캘리브레이션 안정성)** — `calib_100`을 50/50 분할하여 텐서 민감도 순위의
split-half Spearman ρ와 top-k overlap을 측정. **ρ > 0.8**이면 진행, 미달이면 캘리브레이션
클립 수를 늘린다. `max(·,·)` 결합은 한 목적함수의 노이즈가 큰 텐서를 과보호할 수 있어 특히
중요하다. (참고: `jlens_v2`의 split-half는 0.98/0.95였다.)

**Gate Q3 (배분의 이득, H3)** — **primary는 `dual_mag` vs `uniform` 하나로 미리 고정**한다.
동일 실효 저장 예산에서 클립별 페어드 대비 `d_i = ΔminADE^dual − ΔminADE^uniform`에 대해

```
median(d) ≤ −0.05 m   AND   페어드 부트스트랩 95% CI 상한 < 0
```

`traj` arm은 secondary이며 Holm 보정 후 보고한다. 미달이면 "이 아키텍처에서 텐서 단위 비트
배분은 균일 대비 이득이 없다"가 결론이고, 이는 보고할 가치가 있는 **음성 결과**다 — 프루닝에서
allocation이 criterion보다 지배적이었던 것과의 대조가 된다.

## 구현

### 0. 사전 sanity (착수 즉시)

- `lm_head.weight.data_ptr() != embed.weight.data_ptr()` assertion (config상 tied 아님은 확인됨)
- 모든 Linear의 실제 `(out_features, in_features)`와 g64 나눗셈 확인 → ViT `linear_fc2` 27개만 패딩
- fake-quant 단위 테스트: (a) W8→W6→W4→W3 재구성 오차 단조 증가, (b) quantize→dequantize
  결정론성(같은 입력 → bit-exact 같은 출력), (c) 패딩 경로가 비패딩 텐서와 동일 결과

### 1. `experiments/head_analysis/quant_lib.py` (신규)

```
quantize_dequantize(W, bits, group, sym=True)   # 꼬리 그룹 zero-pad
apply_quant(model, spec)                        # spec: 모듈 패턴 -> (bits, group) / None
effective_bytes(model, spec)                    # N·b + N_groups·16 + padding, 논리 저장량
quant_report(model, spec)                       # 텐서별 상대 MSE, 실효 비트, projected saved GB
```

`mask_lib`처럼 in-place. fake-quant는 결정론적 함수라 spec만 있으면 재현되므로 체크포인트 저장이
불필요하다. 슬림 체크포인트 위에도 그대로 걸린다.

### 2. `run_baseline.py`에 `--quant` 추가

로드 직후 `apply_quant`. `config.json`에 spec, `effective_bytes`, `quant_report`를 기록.

### 3. `experiments/head_analysis/run_qsens.py` (신규)

섭동 스코어링. **메모리가 유일한 실질 리스크다** — `run_importance.py`의 peak는 중앙값 40.5 GB /
최대 41.4 GB(48 GB 카드)라, 가중치 grad를 그 위에 얹는 것은 불가능하다(VLM만 bf16 grad 13.9 GB).

해법: `ε`를 저장하지 않고 backward에서 **재계산**하는 커스텀 `autograd.Function`.

```
forward(x, W, alpha, specs):  return F.linear(x, W)     # alpha = 0
backward(gy):
    gW = gy^T @ x                                        # (O, I), 레이어 1개분만 존재
    for b in specs:
        eps = Q_b(W) - W                                 # 재계산, 저장 없음
        g_alpha[b] = (gW * eps).sum()                    # 텐서당 스칼라
```

추가 메모리는 레이어 1개분 `gW`+`eps`(최대 12288×4096 bf16 ≈ 200 MB)뿐, 누적기는 스칼라 5천 개.
한 backward에 모든 비트폭이 나오고, 목적함수마다 backward 1회.

착수 직후 **1클립 메모리 프로브**로 실측하고, 40 GB를 넘으면 타워별 분리 패스로 강등한다.

### 4. `experiments/head_analysis/probe_quant_mech.py` (신규)

P1(logit margin)과 P2(Euler 오차 누적). baseline 참조량(토큰열, `m_t`, `x_k^FP`)을 클립당 한 번
계산해 저장하고 arm마다 재사용한다.

### 5. `analyze_quant.py`

`analyze_baseline.py --compare` 재사용 + 비트폭×범위 격자 곡선
(x = **projected weight storage saved (GB)** 또는 실효 평균 비트, y = ΔminADE / ΔNLL / 토큰 일치율).

### 6. `alloc_lib.py` (Q2)

실효 바이트 비용 테이블 + damage 테이블 → 이산화 예산 DP. greedy 한계효용은 대조로만.

## 실행 순서

1. **사전 sanity** (§구현 0) — GPU 불필요, 즉시
2. **Q1** — `vlm`/`exp`/`head`/`both`/`both+head` × W8/W6/W4/W3, 층화 200 스크리닝 →
   무릎점 후보를 val 500으로 확정. Gate Q1-A/B/C
3. **메커니즘 프로브** — P1, P2. 이 둘이 pruning-vs-quantization 서사를 가장 크게 강화한다
4. **Q2 점수 검증** — `run_qsens.py` → Gate Q2-0 (Spearman), Gate Q2-1 (split-half)
5. **Q2 배분** — DP, primary `dual_mag` vs `uniform`
6. **헤드라인** — val 500 + test 500 + OOD 1,533. **전량 실행 이후에만** H1b/H2/H3 주장

```bash
# Q1 (fake-quant는 런타임 오버헤드 0이므로 baseline과 동일 속도)
bash experiments/head_analysis/run_retry_host.sh 20 \
    experiments/evaluation/run_baseline.py --set indist --exp-id quant_w4g64_exp \
    --quant "expert:4:64" --subset strat200 --shard 0 --n-shards 4 --gpu 0

# Q2 점수 (calib_100, 100클립)
bash experiments/head_analysis/run_retry_host.sh 20 \
    experiments/head_analysis/run_qsens.py --exp-id qsens_v1 --gpu 0
```

## 리스크

1. **GPU 가용성** — 2026-08-11 기준 8장 전부 사용 중(Blackwell 0–3 만재, Ada 4–7 28~34 GB).
2. **아키텍처 혼입** — baseline은 Blackwell에서 측정됐고 결정성은 아키텍처 내부에서만 성립한다
   (관측 예 0.286 vs 0.291). 양자화 arm도 **Blackwell**에서 돌려야 vs-baseline 비교가 깨끗하다.
   Ada로 밀리면 baseline을 Ada에서 재측정하거나 arm 간 비교만 주 판정으로 쓴다.
3. **`run_qsens.py` 메모리** — 재계산 설계로 O(1 레이어)까지 내렸으나 미검증. 실패 시 타워별
   분리 패스(3배 비용)로 강등.
4. **Gate Q2-0 실패 가능성** — 1차 근사가 W3/W4에서 깨질 수 있다. 이것이 **가장 가능성 높은 실패
   지점**이며, 그래서 별도 게이트로 분리했다. 실패해도 Q1 + 메커니즘 프로브 결과는 온전하다.
5. **W8이 bf16보다 정밀할 여지** — group scale이 붙은 int8은 좁은 동적 범위에서 bf16보다 정확할
   수 있다. 실측 W8g128 상대 MSE 4.3e-5는 bf16 자체(~1.3e-6)보다 크므로 실제로는 섭동이 맞지만,
   W8에서 baseline보다 **좋아지는** 결과가 나오면 노이즈가 아니라 이 효과를 의심한다.
6. **범위 확대 압력** — GPTQ/AWQ 재구현, activation 양자화, QAT/LoRA 복구는 전부 이 계획 밖이다.

## 이후 파급

프루닝 산출물은 무효화되지 않는다. fake-quant는 로드된 모델에 거는 후처리라 `slim_*` 체크포인트
위에 그대로 얹히고, **프루닝 × 양자화 결합 압축 곡선**이 자연스러운 후속이 된다. 다만 두 축이
모두 lossy이므로 결합 정확도는 별도 검증이 필요하고, 계획서 수준에서는 **"compression axes can be
composed conceptually"** 이상으로 주장하지 않는다 — 선행 연구의 결합 검증은 주로 KV-cache
양자화 축이고 우리는 weight 양자화라, 직접 인용 가능한 근거인지는 실제 문헌 조사 후에 판단한다.

폐루프(alpasim) 검증은 개루프에서 살아남은 config에 한해 그 다음이다.
