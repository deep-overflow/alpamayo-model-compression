# QVLA-CoC: 추론 손실만으로 유도한 VLM 전용 양자화 (2026-08-13)

## 목적

QVLA(ICLR 2026)의 기계장치를 Alpamayo-1.5에 그대로 이식하되, 민감도 기준을 **CoC NLL 하나로만**
바꾼다. 궤적 손실은 어느 단계에서도 쓰지 않는다. 양자화 대상은 **VLM 타워뿐**이고 expert는
bf16으로 남긴다.

이전 계획(`2026-08-11_importance-guided-quantization.md`)은 dual-objective + 두 타워 + knapsack
구조였는데, 문헌 조사 결과 그 축이 이미 선점되어 있어 폐기한다. 아래 "선행 연구 위치" 참조.

## 선행 연구 위치 (2026-08-13 조사)

| 논문 | 이미 한 것 |
|---|---|
| QVLA (ICLR 2026) | action-space 민감도 → 채널 단위 비트 배분 `{0,2,4,8,16}`, greedy demotion, 0-bit로 프루닝 통합. OpenVLA-OFT W4A4급 96.0 vs FP 97.1 |
| Mix-QVLA | layer 단위 혼합 정밀도 + model-size/BitOps 예산 제약 최적화. OpenVLA-OFT 15.4→4.1 GB |
| QuantVLA | 최초 VLA PTQ, DiT action head 포함, FP 초과 |
| HoloQ-VLA | SVD·Hadamard 회전 + per-step DiT activation scaling, 균일 W4A4. **π0.5 98.0 vs FP16 97.1** |

**선점된 것**: 혼합 정밀도 비트 배분 자체, 한계비용 greedy(`ρ = Δs/Δb`), 0-bit 통합,
diffusion/flow action head 양자화, "action head가 가장 민감하다"는 모듈 분석.

**네 편 전부가 하지 않은 것**: 전수 검색 결과 **언어/추론 출력 품질을 평가한 논문이 하나도 없다**
(perplexity/BLEU/reasoning accuracy 류 지표가 QVLA related work의 한 줄 언급 외 전무). 전부
LIBERO task success rate와 action MSE뿐이다. Alpamayo의 CoC는 중간 산물이 아니라 **그 자체가
산출물**(AD의 검증·해석 채널)이므로, 여기가 남은 자리다.

**주의**: HoloQ-VLA는 회전 적용 후 π0.5 expert 쪽이 0.04–0.07, LLM 쪽이 0.07–0.13으로 **expert가
더 쉽다**고 보고한다. 우리가 잰 "expert가 RTN에서 2.5~5배 어렵다"는 회전 없는 조건의 성질이다.
이번 계획은 expert를 건드리지 않으므로 이 쟁점을 우회한다.

## 왜 CoC-only인가

**1. expert에는 CoC 기울기가 정확히 0이다.** 이건 선택이 아니라 아키텍처다.

```python
# prune_lib.py — CoC 손실 경로에 model.expert가 등장하지 않는다
def vlm_forward_with_grad(model, seq_tf, tokenized_data, use_cache):
    out = model.vlm.model(input_ids=..., pixel_values=..., ...)
    return out.last_hidden_state, out.past_key_values, out.rope_deltas

def coc_nll(model, hidden, seq_tf, coc_start, coc_end):
    logits = model.vlm.lm_head(hidden[:, coc_start-1:coc_end-1]).float()
    return F.cross_entropy(logits[0], seq_tf[0, coc_start:coc_end])
```

`run_importance.py`도 `acc["coc"]`에 `vlm_q / vlm_mlp / kv_k / kv_v`만 누적하고 `exp_*`는 없다.
CoC-only 점수를 greedy demotion에 넣으면 expert 전 채널이 `s=0`이라 0-bit까지 강등되어 타워가
통째로 삭제된다. **따라서 expert 미개입은 CoC-only 기준의 논리적 귀결이다.**

**2. label-free다.** `coc_nll`은 모델 **자기 rollout**을 teacher-force한 NLL이라 GT 궤적이 필요
없다. 궤적 Taylor는 GT future path를 요구한다. 즉 이 기준은 **라벨 없는 주행 로그만으로**
캘리브레이션된다. repo의 기존 label-free 라인(J-lens, `j_traj`)과 이어진다.

## 설계

### 범위

| | 텐서 | 파라미터 | bf16 | 처리 |
|---|---:|---:|---:|---|
| VLM text Linear | 252 | 6.9458 B | 13.89 GB | **양자화** |
| lm_head | 1 | 0.6377 B | 1.28 GB | **양자화** |
| ViT Linear | 117 | 0.5742 B | 1.15 GB | **양자화** |
| **풀 합계** | **370** | **8.1577 B** | **16.32 GB** | |
| expert Linear | 252 | 2.2791 B | 4.56 GB | bf16 유지 |
| embed_tokens | 1 | 0.6377 B | 1.28 GB | bf16 유지 (룩업, GEMM 없음) |
| action_in/out_proj, norm | — | 0.0040 B | 0.01 GB | bf16 유지 |

풀 정의는 QVLA를 따른다 — *"all linear and convolutional layers in both the vision and language
backbones"*. `embed_tokens`는 선형 연산이 아니라 제외.

**압축 상한을 정직하게 적어둔다**: expert가 bf16으로 남으므로 총량 22.16 GB 중 5.84 GB는 어떤
경우에도 줄지 않는다. 풀 평균 4비트면 16.32→4.08 GB, 총 **22.16→9.9 GB (−55%)**. 풀이 0비트여도
바닥은 5.84 GB(원본의 26%)다.

### 기준

텐서 `l`의 출력채널(행) `c`, 비트폭 `b ∈ {0,2,4,8,16}`:

```
S^(b)_{l,c} = | ⟨ ∇_{W_{l,c}} L_CoC , ε^(b)_{l,c} ⟩ | ,    ε^(b)_{l,c} = Q_b(W_{l,c}) − W_{l,c}
```

`L_CoC`는 모델 자기 rollout의 NLL. α=0이 공통 작동점이므로 **한 번의 backward로 모든 비트폭
점수가 동시에** 나온다. 목적함수가 하나뿐이므로 backward도 클립당 1회다(궤적 backward를 돌리지
않으므로 `run_importance.py`보다 오히려 싸다).

이것은 유한 양자화 효과의 **1차 근사**다 — QVLA도 같은 위치에 1차 Jacobian 프록시를 두고
(`‖ΔA‖ ≈ ‖J‖·‖ΔX‖`) 상위 채널만 실측으로 보정한다. 우리는 내적 형태라 노름 곱보다 직접적이지만,
근사인 것은 같다. 검증은 Gate G0.

### 비트 배분

QVLA Eq. 8 그대로. 전 채널을 16비트에서 시작해 단계적으로 강등(16→8→4→2→0)하며, 각 단계에서

```
ρ_{l,c} = ( S^(b_lo)_{l,c} − S^(b_hi)_{l,c} ) / ( b_hi − b_lo )
```

가 낮은 순으로 내린다. 예산 `B̄`(풀 평균 실효 비트) 도달 시 정지. 0-bit는 그 행을 제거(프루닝).

### scale 레이아웃 — QVLA에서 유일하게 벗어나는 지점

QVLA는 행당 scale+zero-point 하나(in_features 방향 그룹화 없음)를 쓴다. 실측 결과 이 레이아웃은
4비트에서 손해가 크다:

| 텐서 | per-row 2b | per-row 4b | per-row 8b | **g64 4b** |
|---|---:|---:|---:|---:|
| VLM L17 gate_proj | 4.97e-1 | 2.26e-2 | 7.9e-5 | 8.6e-3 |
| VLM L35 down_proj | 6.86e-1 | 4.61e-2 | 1.6e-4 | 9.5e-3 |
| **lm_head** | 6.09e-1 | **4.62e-1** | 2.8e-3 | **1.6e-2** |

per-row 4비트는 g64 대비 2.5배 나쁘고, **lm_head는 4.62e-1로 파괴 수준**(29배 차이) — 155697행
× 4096의 outlier를 행당 scale 하나로 감당하지 못한다. 2비트는 어떤 레이아웃에서도 RTN으로는
사실상 사용 불가(0.19~0.69).

**scale 입도와 비트 배분 입도는 직교하므로**, 결정 변수는 QVLA대로 **행당 비트폭**으로 두고 scale만
**(행, g64)** 로 둔다. 한 행의 비트폭은 그 행의 모든 그룹에 균일하므로 QVLA의 배분 알고리즘은
그대로 성립한다. 실효 비트에 그룹 scale 오버헤드(+0.25 bit/weight)를 포함해 예산을 계산한다.
원본 충실도 확인용으로 per-row scale 버전을 ablation 1건 남긴다.

### arm

| arm | 배분 기준 | 역할 |
|---|---|---|
| `qvla_coc` | `S_CoC` + greedy demotion | **주 실험** |
| `uniform` | 없음, 풀 전체 동일 비트폭 | 대조군 |
| `qvla_coc_prow` | 위 + per-row scale | 충실도 ablation (1건) |

> **범위 밖으로 명시**: 궤적 기준 arm(`qvla_act`)은 이번 계획에서 만들지 않는다. 그것이 있어야
> "QVLA의 action-space 정렬 테제가 이 아키텍처에서도 필요한가"를 1요인으로 검정할 수 있으나,
> 궤적 손실 미사용이 이번 실험의 제약이다. 따라서 이번에 답할 수 있는 것은
> **"CoC 기준이 균일 배분을 이기는가"** 까지이고, QVLA 테제와의 직접 대결은 후속으로 남긴다.

## 핵심 판독 — 궤적 손상의 2경로 분해

VLM만 양자화하면 궤적은 **정확히 두 경로**로만 상한다. expert 가중치가 그대로이기 때문이다.

```
(i) 텍스트 경로 : CoC 문장이 바뀜        → 추론 문맥이 달라짐
(ii) 캐시 경로  : KV 캐시 수치가 바뀜    → 문장이 같아도 expert가 읽는 값이 달라짐
```

이 둘을 **분리 측정할 수 있다**:

| 측정 | 조건 | 잡히는 것 |
|---|---|---|
| `ade_rollout` | 양자화 VLM이 자기 CoC를 생성 | (i) + (ii) |
| `ade_tf_base` | **baseline의 CoC를 teacher-force** | (ii)만 |
| 차이 | | (i) |

`run_baseline.py`가 이미 baseline의 `gen_coc`를 클립마다 저장하고 있고, OOD 경로에 teacher-forcing
분기(`gt_coc_seq`)가 이미 있으므로 재사용된다.

**이 분해는 네 편 어느 것도 갖고 있지 않다.** OpenVLA 계열은 action head가 LLM의 마지막 은닉을
읽고 중간 텍스트가 없어서 (i)/(ii) 구분 자체가 성립하지 않는다.

## 사전 등록 게이트

**Gate G0 (점수 타당성) — 배분 진입 필수.** 1차 근사가 유한 RTN 효과를 순위화하는지 검증한다.
`S` 분위수별로 행 200개를 뽑아 **그 행 하나만** 실제 4비트로 양자화하고 실제 `ΔL_CoC`를 측정,
Spearman ρ를 본다. **ρ ≥ 0.7이면 통과.** 미달이면 배분을 중단하고, 이후 결과 해석에서
"1차 프록시가 예측에 실패"와 "배분이 무용"을 반드시 분리해 보고한다.

**Gate G1 (기준의 이득)** — 동일 실효 예산에서 `qvla_coc` vs `uniform`의 클립별 페어드 대비
`d_i`. **주 판정은 median(d)와 페어드 부트스트랩 95% CI**(Wilcoxon은 secondary).
CoC 지표(ΔNLL)와 궤적 지표(ΔminADE) **둘 다** 보고한다 — CoC로 배분해놓고 CoC만 재면 순환이다.

**Gate G2 (능력 비대칭)** — 예산을 낮추며 어느 쪽이 먼저 임계를 넘는가:

| 능력 | 임계 |
|---|---|
| 궤적 | median ΔminADE > +0.05 m (이 프로토콜의 분해능 하한) |
| CoC 충실도 | ΔNLL / baseline 토큰 top-1 일치율 — Q1 첫 결과에서 baseline 분산을 보고 **데이터 보기 전에** 고정 |
| CoC 중증 | degeneracy > 0.05 (baseline 0.006~0.008) |

프루닝 트랙은 "CoC가 먼저 무너진다"였다. **VLM만 양자화했을 때도 같은 순서인지**가 이 게이트의
질문이고, 순서가 뒤집히면 "압축 연산자가 고장 양식을 결정한다"의 첫 증거가 된다.

**Gate G3 (경로 분해)** — `ade_rollout − ade_tf_base`로 텍스트 경로의 기여를 정량화. 궤적 손상이
대부분 (ii) 캐시 경로면 "CoC 품질은 궤적의 대리변수가 아니다"가 되고, 이는 폐루프 트랙의 기존
결론(CoC 건강도 ≠ 안전 대리변수)과 독립적으로 일치하는지 확인할 수 있다.

**Gate G4 (프루닝 연결, sanity)** — 0-bit 끝단은 프루닝 트랙의 `coc` 기준과 같은 대상이다.
`importance_v2`의 `coc_vlm_mlp` 랭킹과 이번 행 단위 0-bit 선택의 중첩률을 확인한다. 크게 어긋나면
둘 중 하나가 잘못된 것이므로 착수 초기에 잡는다.

## 평가 프로토콜

기존 것을 그대로 쓴다. `run_baseline.py`에 `--quant` 분기만 추가.

| 단계 | 셋 | n |
|---|---|---:|
| 스크리닝 | val 500의 버킷 층화 200 | 200 |
| 헤드라인 | val 500 + test 500 + OOD 1,533 | 2,533 |

- 층화 200은 `outputs/baseline_indist`에 기록된 시나리오 버킷으로 비례 배분(decel_stop 28 /
  turn 31 / accel 29 / cruise 112), 버킷 내 `sha256(clip_id)` 순. **스크리닝은 방향 탐지용이며
  게이트 판정은 전량에서만 한다.**
- 페어드 시드 `sha256(f"{seed}:{clip_id}")[:4]`, k=8, seed 42, `MODEL_REV=7aba8293…` 고정
- 캘리브레이션 `calib_100` (`importance_v2`와 동일 100클립·동일 revision)
- baseline 기준선: val 0.690 / test 0.777 / OOD 0.842 (Blackwell 측정) — **양자화 arm도
  Blackwell에서** 돌려야 비교가 깨끗하다

## 구현

### 0. 사전 sanity (GPU 불필요)

- `lm_head.weight.data_ptr() != embed.weight.data_ptr()` (config상 `tie_word_embeddings: False` 확인됨)
- expert / embed / action_proj가 양자화 대상에서 제외되는지 spec 단위 테스트
- fake-quant 단위 테스트: 비트폭에 따른 오차 단조성, quantize→dequantize 결정론성, 패딩 경로 동치
  (풀 안에서 g64 패딩이 필요한 것은 ViT `linear_fc2` 27개뿐)

### 1. `quant_lib.py`

```
quantize_dequantize(W, bits, group, asym=True)   # bits=0 -> 행 제거, 16 -> 무변환
apply_quant(model, spec)                          # 행 단위 비트 벡터를 텐서별로 받음
effective_bits(spec)                              # N·b + N_groups·16 + padding
quant_report(model, spec)                         # 텐서별 상대 MSE, 실효 비트, projected saved GB
```

fake-quant(bf16에 되담기)이므로 런타임·메모리·결정성이 baseline과 동일하다. 논리적 절약과 물리적
절약을 반드시 구분해 표기한다 — 플롯 축은 `projected weight storage saved (GB)`.

### 2. `run_qsens_coc.py`

`run_importance.py`에서 **궤적 절반을 들어내고** CoC backward만 남긴 것. 행 단위
`∂L/∂α`를 커스텀 `autograd.Function`으로 읽는다. ε는 저장하지 않고 backward에서 재계산한다:

```
forward(x, W, alpha, bits):  return F.linear(x, W)      # alpha = 0
backward(gy):
    gW = gy^T @ x                                        # (O, I), 레이어 1개분만
    for b in bits:
        eps = Q_b(W) - W                                 # 재계산, 저장 없음
        g_alpha[b] = (gW * eps).sum(-1)                  # (O,) 행별
```

누적기는 `Σ_t O_t × 5비트 × 1목적함수` ≈ 수백만 스칼라(수십 MB). 메모리 여유가 이전 계획보다
크다 — 궤적 backward와 `retain_graph`가 없으므로 `run_importance.py`의 peak 40.5 GB보다 낮을
것으로 예상하나, **1클립 프로브로 실측 후 진행**한다.

### 3. `alloc_lib.py`

QVLA greedy demotion (Eq. 8). 단계별 정렬 후 강등, 예산 도달 시 정지. `O(C log C)`.
과다 프루닝 방지를 위해 2→0 단계에 임계 규제(QVLA도 동일한 heuristic을 둔다).

### 4. `analyze_qvla_coc.py`

`analyze_baseline.py --compare` 재사용 + 예산 곡선(x = projected saved GB / 실효 평균 비트,
y = ΔminADE, ΔNLL, 토큰 일치율, degeneracy) + 경로 분해 플롯 + 레이어별 비트 배분 히트맵.

## 실행 순서

1. 사전 sanity — 즉시, GPU 불필요
2. **Q0 스모크**: 풀 균일 W8 / W4를 층화 200에 걸어 무릎점이 어디인지 먼저 본다 (반나절).
   문헌상 W4가 거의 공짜일 가능성이 있고, 그러면 예산 격자를 더 낮은 쪽으로 옮겨야 한다
3. `run_qsens_coc.py` → Gate G0 (Spearman), Gate G4 (프루닝 연결)
4. greedy demotion으로 예산 3~4점 배분 → `qvla_coc` vs `uniform`, 층화 200
5. 헤드라인 2,533 클립 → Gate G1 / G2 / G3

## 리스크

1. **W4가 이미 거의 무손실일 가능성** — HoloQ-VLA가 π0.5를 W4A4에서 FP 초과(98.0 vs 97.1)로
   보고한다. 그러면 `qvla_coc`가 `uniform`을 이길 여지가 작다. Q0를 2번째 단계에 둔 이유가 이것이고,
   무릎점이 낮으면 예산을 2~3비트 영역으로 내려 격차가 벌어지는 구간에서 비교한다.
2. **2비트 RTN이 사실상 사용 불가** (per-row 0.41~0.69, g64 0.19~0.23). greedy가 저민감 채널에만
   2비트를 주므로 원리상 문제는 아니나, 예산을 낮추면 2비트 배정 비율이 커져 급격히 무너질 수 있다.
   배분 결과의 비트 히스토그램을 매 예산마다 기록한다.
3. **Gate G0 실패** — 1차 프록시가 4비트에서 깨질 수 있다. 가장 가능성 높은 실패 지점이라 별도
   게이트로 분리했다. 실패해도 Q0의 균일 사다리와 경로 분해(G3)는 온전하다.
4. **아키텍처 혼입** — baseline은 Blackwell 측정. 결정성은 아키텍처 내부에서만 성립한다
   (관측 예 0.286 vs 0.291).
5. **범위 확대 압력** — 회전(Hadamard), GPTQ/AWQ, activation 양자화, expert 양자화는 전부 범위
   밖이다. 다만 논문화 시점에는 GPTQ/AWQ 대조가 요구될 것이므로, `quant_lib`의 인터페이스는
   나중에 보정 알고리즘을 끼울 수 있게 열어둔다.

## 실행 기록 (2026-08-13)

계획 대비 바뀐 것과 그 이유. 다음 라운드가 같은 함정을 다시 밟지 않도록 남긴다.

**1. ViT 기울기가 0이었다 (버그).** 첫 배분이 ViT를 통째로 0비트로 잘랐다. 원인은 ViT 306,544행의
점수가 **정확히 0**이었던 것 — `pixel_values`는 grad를 요구하지 않고 ViT 가중치는 frozen이라
autograd가 비전 서브그래프를 순회하지 않았다. `enable_input_require_grads()`는 텍스트 임베딩만
연결한다. "비전 인코더는 추론에 안 중요하다"로 읽힐 뻔했지만 **측정 자체가 안 된 것**이었다.
풀 가중치에 `requires_grad_(True)`를 주어 해결(우리 Function은 W 자리에 None을 반환하므로 grad
메모리는 0). 첫 클립 후 전 텐서의 점수 유무를 검사하는 assertion을 넣었다.

**2. 그래서 OOM → gradient checkpointing.** ViT 그래프가 들어오자 46.9/47.4 GiB로 OOM(ViT 단독
패스도 동일). 체크포인팅을 켜되 **rollout 구간만 꺼야** 했다 — train 모드가 `use_cache=False`를
강제해 증분 디코딩이 `get_rope_index`에서 깨진다(`run_importance.py`가 체크포인팅을 못 쓰는 이유와
같은 문제). teacher-forced 패스는 원래 `use_cache=False`라 문제없다. peak 41.2 → **24.2 GB**,
클립당 7 → 9초. 실제 소요 100클립 14.4분.

**3. 평균 8비트는 staged greedy에서 퇴화한다.** 16→8 단계가 끝나는 순간 예산이 충족되어 전 채널이
8비트, 즉 균일 W8과 bit-identical이 된다. 그래서 예산을 **균일 W8의 실제 저장 비용**으로 잡고
라그랑주 MCKP로 재분배했다(`alloc_lib.allocate`). 같은 바이트, 다른 분포.

**4. 0비트를 메뉴에서 뺐다.** lm_head 행별 점수의 중앙값이 **1.9e-8** — 155,697개 vocab 행 중
캘리브레이션 100클립이 실제로 사용한 것만 기울기를 받는다. 0비트를 허용하면 어휘 대부분이 잘려
캘리브레이션 지지집합 밖 토큰을 못 만드는 모델이 되는데, 이는 압축 결과가 아니라 캘리브레이션
인공물이다. 기본 메뉴 `{2,4,8,16}`, `--allow-prune`으로 QVLA 원본 메뉴 복원.

**5. 실제 배분** (`outputs/quant_specs/qvla_coc_b8.json`, 예산 100.00% 사용):

| 부분 | 실효 비트 |
|---|---:|
| ViT | 13.14 |
| VLM text | 8.54 |
| lm_head | 2.37 |

행 분포 2b 12.7% / 4b 3.5% / 8b 66.1% / 16b 17.7%. CoC 기준은 **비전 인코더를 보호하고 출력
헤드를 강하게 압축**하는 쪽을 골랐다. lm_head 2.37비트는 이 실험의 가장 큰 위험 지점이고,
`uniform_w8` 대조군이 그것을 분리해준다.

**6. 압축 상한.** 풀 bf16 16.31 GB → 8.54 GB(projected). expert 4.56 GB + embed 1.28 GB가
그대로이므로 총 **22.16 → 14.4 GB (−35%)**. fake-quant이므로 실제 GPU 메모리는 줄지 않는다.

**7. 남은 것.** Gate G0(Spearman 검증)과 G4(프루닝 트랙 연결)는 아직 돌리지 않았다. G3(경로 분해)는
`ade_tf_base` 조건을 `run_baseline.py`에 추가해야 하며, 현재 실행은 own-rollout 조건만 담는다.

## 이후

- **Gate G0 / G3 / G4** — 위 7번.
- **`qvla_act` 대조** — 궤적 기준 arm. QVLA 테제와의 직접 1요인 대결. 궤적 손실 사용이 필요하다.
- **회전 + GPTQ 강baseline** — HoloQ-VLA가 보인 대로 회전이 heavy-tail을 해결하므로, RTN만으로는
  논문 대조군이 약하다.
- **expert 축** — 이번에 손대지 않은 20.6%. 압축 상한이 −74%에 걸려 있는 이유다.
- **폐루프(alpasim)** — 개루프에서 살아남은 config에 한해.
