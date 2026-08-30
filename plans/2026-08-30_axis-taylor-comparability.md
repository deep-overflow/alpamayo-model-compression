# Q head vs MLP channel Taylor 중요도의 축 간 비교 — expert에서 검증하고 VLM으로 확장

날짜: 2026-08-30. 브랜치: `axis-taylor-comparability` (worktree, 승인 후 구현).
선행: `reports/evaluation/2026-08-28_expert-axis.html` (expert에서 Q-only가 MLP-only보다
비싸다 — 6배 적게 지우고도, 파라미터를 맞춰도), `plans/2026-08-28_expert-axis-ablation.md`.

## 0. 질문

expert-axis 보고서는 두 축을 **따로 잘랐을 때의 비용**이 다르다는 것을 보였지만, 두 축의
**Taylor 중요도 자체**를 나란히 놓은 적은 없다(§6의 "질량" 곡선은 축 *안에서의* 분포다).
그래서 순서대로:

1. Q head와 MLP channel의 Taylor 중요도는 **수학적으로 비교 가능한가?** (아래 §1, 실험 전 답)
2. 가능하다면 expert에서 **비교 실험** — raw 점수의 축 간 비교 + 1차 근사가 두 축에서
   똑같이 정확한지의 검증 (Stage 1).
3. VLM에서 dual 중요도의 두 원재료를 축 간 비교하고 (Stage 2a, 계산만),
   Q head만 / MLP channel만 따로 제거하는 one-factor 실험 (Stage 2b).

## 1. 수학적 비교 가능성 (분석)

### 1.1 두 점수는 같은 것을 재고 있다 — 비교 가능

두 축 모두 `prune_lib.UnitGates`의 **곱셈 게이트** $g_u = 1$ 위의 1차 Taylor다:

- Q head $h$: 게이트는 `o_proj` 입력의 head slice — $y_h \leftarrow g_h\, y_h$ ($y_h \in \mathbb{R}^{128}$)
- MLP channel $c$: 게이트는 `down_proj` 입력 채널 — $a_c \leftarrow g_c\, a_c$ ($a_c \in \mathbb{R}$)

둘 다 "그 유닛이 residual stream에 쓰는 기여를 통째로 끄는" 스칼라이고, `mask_lib.PruneMasks`가
실제로 적용하는 0/1 마스크와 **같은 연산**이다. 유닛 $u$를 제거한 손실 변화는

$$\Delta L_u = L(g_u{=}0) - L(g_u{=}1) = -\frac{\partial L}{\partial g_u} + \tfrac12 \frac{\partial^2 L}{\partial g_u^2} - \cdots$$

저장된 점수 $I_u = \mathrm{mean}_{\text{clips}}\left|\sum_s \partial L_s/\partial g_u\right|$ 는 이 식의
**1차 항의 크기**이고, 두 축이

- 같은 손실 (expert: FM MSE, VLM: FM MSE 또는 CoC NLL),
- 같은 100 클립 (`calib_100`), 같은 시드,
- 같은 집계 규칙 (`mean_clips |Σ_s|`)

을 공유한다. 즉 **raw 점수는 같은 단위(손실 단위)의 같은 양**이고, "head 하나를 끄면 손실이
얼마나 변하나"와 "channel 하나를 끄면 얼마나 변하나"를 직접 비교할 수 있다. 파라미터당으로
나누는 것도 잘 정의된다: head 1개 = 524,288 (expert) / 1,048,576 (VLM), channel 1개 = 6,144 /
12,288 — 두 타워 모두 **85.3배** (= 2·128 / 3).

### 1.2 비교 **불가능**한 것 — 실제로 배포된 점수

- `znorm` (`importance_stepexp_znorm`의 `traj_exp_*`): 스텝별 **축별·층별 z-score** 후 평균.
  각 축이 자기 층 안에서 평균 0·표준편차 1로 정규화되므로 축 간 스케일은 정의상 사라진다
  (실제 파일의 전역 평균이 1.7e-17 / −3.6e-19).
- `dual` = `max(rank_norm(I_traj), rank_norm(I_CoC))`: 축별·층별 **순위** [0, 1]. 마찬가지.

따라서 축 간 비교는 **raw 배열**로 해야 한다 — expert는 `importance_stepexp_sum`
(`= importance_v2_ada`, `mean_clips|Σ_s|`), VLM은 `importance_v2` (u40 family의 원천).
배포된 선택(znorm / dual)이 고른 유닛의 비용은 그 유닛의 raw 점수를 읽으면 되므로,
"선택은 정규화 점수로, 비용 비교는 raw로"가 가능하다 (§2의 미리보기가 정확히 그것이다).

### 1.3 비교가 *정량적으로* 유효하려면 — 축마다 다를 수 있는 것 (실험이 필요한 부분)

1차 항은 같은 단위지만, **1차 근사가 두 축에서 똑같이 정확하다는 보장은 없다**:

- (a) **섭동 크기.** head 제거는 128차원 벡터를 끄는 것이고 channel 제거는 스칼라 하나다.
  2차 이상 항의 크기가 축마다 다를 수 있어, 1차 항 대비 실제 $\Delta L$의 비율(기울기)이
  축별로 다르면 raw 비교가 체계적으로 편향된다. **G3 초선형(헤드 2배 → 비용 5배)** 이
  head 축에서 고차항이 크다는 힌트다.
- (b) **집합 가산성.** 배포는 유닛 하나가 아니라 집합 $S$를 지운다. 1차 예측
  $\sum_{u \in S}$ 는 유닛 간 교차항을 무시하고, 4 head의 교차항 구조와 341 channel의
  교차항 구조는 다르다.
- (c) **부호.** 집합 동시 제거의 1차 예측은 **signed 합** $-\sum_{u\in S}\partial L/\partial g_u$ 인데,
  저장된 per-clip 값은 $|\cdot|$ 뿐이다(`importance_perclip.npz`). 축 간 비교의 정확한 형태를
  얻으려면 signed grad를 다시 재야 한다 (probe에서 1 backward로 얻음).
- (d) **스텝 상쇄 (expert).** `|Σ_s|` 규칙의 상쇄비 $C = \sum|\Sigma_s| / \sum\Sigma_s|\cdot|$ 가
  Q 0.665 / MLP 0.609 — 거의 같아서 축 간 비교를 크게 비틀지는 않는다. `sumabs`
  ($\sum_s |\partial L_s/\partial g|$, `stepimp_fm_perstep_v2`)를 병기해 확인한다.

**결론: 차원적으로는 비교 가능(1차, 같은 손실 단위)하다. 그러나 (a)(b)가 축마다 다를 수
있으므로, "예측 $\Delta L$ vs 실측 $\Delta L$"의 기울기가 두 축에서 일치하는지를 실험으로
확인한 뒤에야 축 간 비교를 정량적으로 신뢰할 수 있다.** 그 확인이 Stage 1이다.

## 2. 미리보기 — raw 점수의 축 간 비교 (계산만, GPU 없음, 2026-08-30)

`/home/cvlab21/project/chan/.claude/jobs/b868fb43/tmp/preview.py` (Stage 1 분석 스크립트로
승격 예정). 선택은 배포 기준(expert: znorm, VLM: dual)으로, 비용은 raw 단위로 읽었다.

### expert (`importance_stepexp_sum`, FM MSE 단위)

| 양 | Q head | MLP channel | 비 |
|---|---|---|---|
| 유닛당 중앙값 $I$ | 7.83e-3 | 1.65e-7 | **47,400×** (평균 634×) |
| 파라미터당 중앙값 $I/p$ | 1.49e-8 | 2.69e-11 | **555×** |
| 층 질량 합 (36층 평균) | 0.194 | 0.158 | 1.23 (파라미터 비 0.165) |
| 가장 싼 head 1개 vs 최하위 85 channel (같은 파라미터) | 2.35e-3 | 3.25e-6 | **644×** |

| arm (znorm 선택) | 제거 파라미터 | $\sum_{S} I$ (raw) | $\sum_s\lvert\cdot\rvert$ | $I$/param |
|---|---|---|---|---|
| q25 (4 heads/층) | 75.5M | **0.884** | 1.242 | 1.17e-8 |
| q50 (8 heads/층) | 151.0M | 2.125 | 3.113 | 1.41e-8 |
| m_pm (341 ch/층) | 75.4M | 0.00062 | 0.00112 | 8.3e-12 |
| m25 (2064 ch/층) | 456.5M | 0.0057 | 0.0101 | 1.3e-11 |
| m50 (4128 ch/층) | 913.0M | 0.0158 | 0.0275 | 1.7e-11 |

raw 1차 Taylor는 **이미 축 비대칭을 예측한다**: q25(75.5M)의 1차 질량이 파라미터가 같은
m_pm(75.4M)의 1,420배이고, **자기보다 12.1배 많은 파라미터를 지우는** m50(913.0M)의 56배다.
실측 순서(q25 +0.022 > m50 +0.0001 ≈ m25 ≈ m_pm ≈ 0)와
정성적으로 일치한다. Stage 1은 이 일치가 정량적인지(기울기·가산성)를 묻는다.

### VLM (`importance_v2`)

| 양 | traj: Q / MLP | CoC: Q / MLP |
|---|---|---|
| 유닛당 중앙값 비 | 29.5× | 32.5× |
| **파라미터당** 중앙값 비 | **0.35×** | **0.38×** |
| 층 질량 합 비 Q/MLP (파라미터 비 0.222) | 0.14 | 0.08 |

`dual_u40_v2`의 두 절반을 raw로 읽으면 (선택은 dual rank, 비용은 raw):

| 절반 | 제거 파라미터 | $\sum_S I_{traj}$ | $\sum_S I_{CoC}$ |
|---|---|---|---|
| Q 절반 (13/32 heads/층) | 490.7M | 1.79 | 0.203 |
| MLP 절반 (4898/12288 ch/층) | 2,166.7M | 18.1 | 3.32 |
| 파라미터 매칭 MLP (1109 ch/층) | 490.6M | 3.16 | 0.620 |

**expert와 반대 방향의 예측**: VLM에서는 파라미터를 맞춰도 MLP 축이 Q 축보다 1.8× (traj) /
3.1× (CoC) 비싸고, dual_u40_v2의 비용은 1차 예측상 90%가 MLP 절반 몫이다. 이것이 Stage 2b의
사전 등록 예측이다 — 두 타워가 반대로 나오면 "축 효과"가 타워 구조(step-특화 head vs
프롬프트-문맥 head)에 따른 것이지 head/channel의 일반 성질이 아니라는 뜻이 된다.

## 3. Stage 1 — expert: 1차 근사의 축별 정확도 (probe)

### 가설
- **H1**: 두 축의 "예측 $\Delta L$ → 실측 $\Delta L$" 기울기가 같다 → raw 비교가 정량적으로
  유효하고, §2의 축 간 배수를 그대로 읽어도 된다.
- **H1′ (기각 시)**: 기울기가 다르다 → 축 간 비교에는 축별 보정 계수(기울기 비)가 필요하며,
  그 계수를 보고한다. 어느 쪽이든 §2의 순서(q25 ≫ m50)가 뒤집힐 만큼 크지는 않을 것으로
  예상하지만, 그것도 측정한다.

### 설계 (`experiments/head_analysis/run_taylor_probe.py`, 신규)

- 클립: `calib_100`의 앞 **32개** (`sample_cache.calib_samples`), `run_importance`와 같은
  클립-유도 시드. 근사 정확도를 재는 것이므로 중요도를 잰 클립을 그대로 쓰는 것이 맞다
  (일반화가 아니라 근사를 검증한다).
- 클립당 1회: CoC rollout → VLM forward → prefill cache (run_importance와 동일 경로).
  expert 손실 $L_{FM}$은 `expert_fm_grads`의 t-grid $(s{+}0.5)/10$, **고정 ε 1개를 10 스텝이
  공유**(`noise_mode="shared"`) → probe 간 완전 paired, 잡음 없음.
- 클립당 1 backward (`UnitGates`, expert만, 같은 고정 ε): **signed** $\partial L/\partial g_u$ →
  probe 집합 $S$의 1차 예측 $\widehat{\Delta L}(S) = -\sum_{u\in S}\partial L/\partial g_u$.
- probe (층 $l$마다, `PruneMasks`로 마스크 교체, forward만):

  | probe | 개수/층 | 목적 |
  |---|---|---|
  | head 1개 | 16 (전부) | 유닛 단위 예측-실측 (Q 축) |
  | **channel 1개** (raw 순위 0/25/50/75/90/99% 위치) | **6** | **유닛 단위 예측-실측 (MLP 축)** — 없으면 "축 효과"와 "한 번에 85개를 자른 효과"가 섞인다 |
  | channel 85개 블록 (= head 1개 파라미터의 99.6%) | 8 — raw 순위 0/10/25/50/75/90/95/99% 위치의 연속 블록 | 같은 파라미터의 MLP 섭동 (S1b) |
  | arm 집합: q25-층 (znorm 하위 4 head), q50-층 (8), m_pm-층 (341), m25-층 (2064), m50-층 (4128) | 5 | 집합 가산성 + 실제 arm의 층별 실측 |

  = 35 probe/층 × 36층 + **전 층 arm 마스크 5개** = **1,265 probe**, 클립당 ≈ 1,265 × 10 expert step.
  전 층 arm 5개에는 k=6 denoise의 minADE@6도 기록해 probe 스케일을 보고 지표와 연결하고,
  층별 arm 36개의 합 vs 전 층 arm으로 **층 간 가산성**을 직접 잰다.

  실측 비용(스모크): **1265 probe ≈ 12분/클립**. 계획의 2 h 추정은 probe 1,044개·0.2 s 가정이었고
  실제는 0.58 s/probe다. 그래서 32클립을 **2-way 샤딩**한다(`--shard i --n-shards 2`,
  샤드당 16클립 ≈ 3.3 h). 샤드 0은 GPU 7에서 즉시 시작했고, 샤드 1은 평가 큐가 카드를 비우는 대로
  올린다. 샤드 0만으로도 head probe 576×16 = 9,216점이라 판정은 가능하다.
- 산출: `outputs/taylor_probe_expert/{config.json, rows.jsonl(클립×probe: $L$, $\widehat{\Delta L}$,
  ADE), metrics.json, summary.txt, plots/}`; 분석 `analyze_taylor_probe.py`.

### 사전 등록 게이트

- **S0 무결성**: 마스크 없는 probe의 $L$이 baseline과 bitwise 동일; 클립당 signed grad의
  $|\cdot|$ 합이 `stepimp_fm_shared_v2`/`importance_v2_ada`의 해당 클립 값과 부호 없는
  집계에서 일치(같은 ε 규칙일 때); 16 head probe의 예측 합 = 층 전체 Q 예측.
- **S1 기울기 (주 판정)**: 축별 OLS $\Delta L_{meas} = \beta_{axis}\,\widehat{\Delta L}$ (절편 0,
  클립×probe 점 위, 클립 부트스트랩 CI). **$\beta_Q / \beta_{MLP}$ 의 95% CI가 [0.5, 2] 안**
  → H1 채택 (raw 비교 유효). 밖 → H1′, 보정 계수 = 그 비.
  병기: $R^2$ 축별, log-log 기울기(비선형성).
- **S2 집합 가산성**: arm 집합의 $\Delta L_{meas}(S) / \widehat{\Delta L}(S)$ 를 축별로 —
  q25·q50 vs m_pm·m25·m50. 비율이 1에서 멀어지는 방향과 크기(초선형이면 >1)를 보고.
- **S3 순서**: 층별 arm 집합 5개의 예측 순위와 실측($\Delta L$, ΔADE) 순위의 Spearman;
  전체 합에서 q25 vs m50의 예측 배수(56×) vs 실측 배수.
- **S4 sumabs 강건성**: 예측을 `sumabs`로 바꿔도 S1 판정이 유지되는지.

### 자원
Ada 1장, probe당 ≈ 0.1–0.2 s (64 토큰 × 10 step, prefill 재사용) → 클립당 ≈ 3 min + rollout
→ 32 클립 **≈ 2 h**. 체크포인트 없음, 디스크 무부담.

## 4. Stage 2 — VLM

### 2a. raw 중요도 축 간 비교 (계산만)
§2의 VLM 표를 `analyze_taylor_probe.py`가 함께 산출(층별 곡선 포함: 층 질량 비 Q/MLP,
파라미터당 중앙값 비, dual 절반별 $\sum I$). dual 자체는 순위라 비교하지 않고(§1.2) 그
두 원재료 $I_{traj}$, $I_{CoC}$를 각각 비교한다. 새 측정 없음 — `importance_v2`가 이미
dual의 재료다.

### 2b. VLM 축 one-factor (expert-axis 설계의 VLM판)

VLM만 자르고 expert·KV는 무접촉. 기준은 **dual**(`tyr.dual_scores`, `importance_v2`),
비율은 u40 family의 **0.3985632694**, `select_mask_ratios`. `slim_dual_u40_v2`의 마스크가
`vq = select_mask_ratios(sq, rq)`, `vm = select_mask_ratios(sm, rm)` 로 축별 독립이므로
**dualq ∪ dualm == dual_u40_v2 가 정확히 성립**하고, 가산성은 기존 `dual_u40_v2_ps_*`로
공짜다. `--no-state` 순수 선택 config.

| arm | config | 층당 제거 | 제거 파라미터 | 전체 대비 |
|---|---|---|---|---|
| **vq** | `dualq_u40_v2` | 13 / 32 heads | **490,733,568** | 4.43% |
| **vm** | `dualm_u40_v2` | 4898 / 12288 ch | **2,166,718,464** | 19.56% |
| **vm_pm** | `dualm_c1109` | 1109 ch (파라미터 매칭, 차이 −147,456 = 0.030%) | **490,586,112** | 4.43% |
| (기존) both | `dual_u40_v2` = vq ∪ vm | 13 heads + 4898 ch | 2,657,452,032 | 23.99% |

**파라미터 수는 출시 가중치에서 직접 확인했다** (`check_vlm_axis.py`, safetensors 헤더만 읽음,
GPU·모델 로드 없음). 두 타워 모두 head 1개 = q_proj 행 + o_proj 열, channel 1개 = gate/up 행 +
down 열이고 q_norm/k_norm은 `(128,)`로 헤드 간 공유라 헤드 제거로 사라지는 norm 파라미터는 없다
(= `make_slim.expected_removed`가 kv-only 층 밖에서 norm 항을 더하지 않는 이유).

| tower | hidden | heads×dim | inter | head 1개 | channel 1개 | 비 | Q축 합 | MLP축 합 |
|---|---|---|---|---|---|---|---|---|
| VLM | 4096 | 32×128 | 12288 | **1,048,576** | **12,288** | 85.3× | 1,207,959,552 | 5,435,817,984 |
| expert | 2048 | 16×128 | 8256 | **524,288** | **6,144** | 85.3× | 301,989,888 | 1,826,095,104 |

가중치 총합 11,078,526,194가 `slim_dual_u40_v2`의 `params.full`과 일치하고,
490,733,568 + 2,166,718,464 = 2,657,452,032 = 그 체크포인트의 `params.removed`와 정확히 같다.
`vm_pm`은 균일 1109 ch/층을 유지해서 −0.030%가 남는다(정확 매칭은 12개 층만 1110으로 바꿔야 하고,
그러면 "균일 할당" 인자가 깨진다 — expert의 `expertm_c341`도 같은 이유로 −0.1%를 남겼다).

평가: 고정 프로토콜(val500 / test500 / OOD-val 262, rollout-only, minADE@6·minFDE@6, k=8,
클립 유도 시드), **Ada 4–7**, `launch_arms.sh` 큐. 대조군 `baseline_ada_ps_*`;
`dual_u40_v2_ps_{indist,test}` 그대로, OOD-val은 `dual_u40_v2_ps_ood`(전체 1,533)를
`split=='val'`로 축소해 페어링(시드가 클립 유도라 호환 — 2026-08-19 프로토콜 메모).
VLM 절단은 CoC를 바꾸므로 **CoC 퇴화율·gen_coc 동일성도 부차 지표로 보고**
(expert-axis와 달리 gen_coc 동일성은 게이트가 아니라 관찰).

사전 등록 게이트 (val500 paired 중앙값 CI, 평균 병기, Wilcoxon; §2 예측 병기):

- **V0 무결성**: 제거 파라미터 정확 일치; expert·KV·반대 축 kept = 전부; vq ∪ vm ==
  `slim_dual_u40_v2` bit-identical.
- **V1 비율 매칭**: vq − vm. §2 예측: **음수**(MLP 절반이 비싸다). CI < 0 → 예측 채택;
  CI > 0 → expert와 같은 방향(head가 비싸다), 1차 Taylor의 VLM 예측 기각.
- **V4 파라미터 매칭**: vq − vm_pm. §2 예측: **음수**, 1차 예측 배수 1.8× (traj).
- **V2 가산성**: dual − (vq + vm) 의 CI.
- **V5 예측 정량**: 세 arm의 실측 Δ 순서 vs $\sum_S I_{traj}$ 순서(3개라 순서만).
- 세트 간 일관성: test500·OOD-val에서 V1·V4 부호 유지.

자원: 빌드 3 × ~20 min(`--no-state`), 평가 3 arm × 1,262 클립 ≈ **9.6 GPU-h** → Ada 4장
≈ 2.5–3 h (현재 4–7 전부 유휴, Blackwell 0–3은 타인 사용 중). 디스크 무부담
(`/mnt/nvme1n1` 127 GB 여유, 99%).

### 2c (선택, 후순위). VLM probe
Stage 1과 같은 probe를 VLM에 (teacher-forced VLM forward가 probe당 필요, ≈ 1 s) — 36층 ×
(8 head + 8 채널블록 + 3 arm 집합) × 16 클립 ≈ **3–4 GPU-h**. 2b 결과가 §2 예측과
어긋날 때만 돌린다(어긋남이 1차 근사 실패인지 확인하려고). 기본은 생략.

## 5. 결과의 쓰임

- S1 채택이면 "Q head 1개 ≈ MLP channel 수백~수천 개"라는 raw 배수를 그대로 쓸 수 있어,
  **축 간 예산 배분을 파라미터당 중요도로 통합**하는 다음 단계(expert: MLP로, VLM: Q로 예산
  이동)의 근거가 된다. S1′이면 보정 계수를 붙여 같은 일을 한다.
- 2b가 예측대로(VLM은 MLP가 비쌈)면 dual_u40_v2의 예산을 Q 쪽으로 옮긴 config
  (예: Q 50% + MLP 30%, 같은 2.66B)가 자연스러운 후보다. 반대면 두 타워가 같은 방향이고
  "head는 어디서나 비싸다"가 된다.

## 6. 파일

| 파일 | 변경 |
|---|---|
| `experiments/head_analysis/run_taylor_probe.py` (신규) | Stage 1 probe 러너 (`run_expert_agg.py`의 클립 루프 + `expert_fm_grads`의 t-grid 재사용) |
| `experiments/head_analysis/analyze_taylor_probe.py` (신규) | S0–S4 판정 + §2 raw 비교 표/곡선 (expert·VLM), `outputs/taylor_probe_expert/`, `outputs/axis_taylor_analysis/` |
| `experiments/head_analysis/make_slim.py` | 분기 추가 `^dual(q\|m)_u40_v2$`, `^dualm_c(\d+)$` (반대 축·expert·KV identity). **공유 파일** |
| `experiments/evaluation/analyze_vlm_axis.py` (신규) | V0–V5 판정 (`analyze_expert_axis.py` 구조 재사용, OOD-val 필터 포함), `outputs/vlm_axis_analysis/` |
| `experiments/evaluation/paper_numbers.py` | ARMS에 `dualq40`/`dualm40`/`dualm_pm` 추가. **공유 파일 — 주 checkout에 미커밋 변경 있음(dualr-weighted), 충돌 주의** |
| `experiments/head_analysis/axis_taylor_report_template.html` (신규) | 보고서 → `reports/evaluation/2026-08-30_axis-taylor-comparability.html` (Stage 1 + 2a + 2b 한 보고서) |

## 7. 실행 순서

1. Stage 1 probe (Ada 1장, ~2 h) ‖ 동시에 2b 빌드 3개(Ada 1장 순차, ~1 h) → 2b 평가 큐
   (Ada 4장, ~3 h). 총 wall-clock ≈ 3–4 h.
2. `analyze_taylor_probe.py` → S 게이트; `analyze_vlm_axis.py` → V 게이트.
3. 보고서 1개, plan에 결과 추기, PR.

## 실행 기록 (2026-08-30)

- **승인**: 2026-08-30, "계획대로 진행 + VLM 파라미터 수를 제대로 확인 후 엄밀하게".
- **V0 사전 검증 통과** (`check_vlm_axis.py`, `outputs/vlm_axis_check/`): 가중치에서 읽은
  단위 파라미터, union == `slim_dual_u40_v2` 36층 전부 kept 인덱스 일치, 제거 파라미터 합 일치.
- **빌드 완료** 06:45 KST: 세 recipe 모두 `surgery` 로그의 실측 제거량이 설계값과 정확히 일치
  (−490,733,568 / −2,166,718,464 / −490,586,112). `--no-state`, 각 30초.
- **평가**: `launch_arms.sh` 큐(2 shard × 3 세트 × 3 arm = 18 job), Ada 4·5·6 워커 3장.
  GPU 7은 probe 전용으로 남겼다.
  - **주의**: `init`이 커서를 0으로 되돌렸는데도 워커가 이전 세션 값(10)에서 시작해 앞 10개
    job이 건너뛰어졌다. 실행 중이던 3개를 제외한 15개로 큐를 다시 쓰고 커서를 0으로 복구했다
    (flock 사용). 전체 18개가 정확히 한 번씩 실행된다.
  - **teacher-forced 조건**: `--no-tf`를 주지 않는다. 2026-08-19 프로토콜은 TF를 *보고*하지
    않는다는 것이고, 대조군 `baseline_ada_ps_oodval`과 expert-axis arm이 전부 TF를 계산한
    경로로 측정됐다. 같은 경로를 유지해야 paired 비교의 bitwise 결정성이 보장되므로 계산은
    하되 보고하지 않는다.
- **Stage 1 스모크에서 이미 나온 신호**(1클립·4층·121 probe, 확정 아님):
  S0는 완벽 통과(gate pass 손실이 mask 경로 baseline과 **정확히** 일치, rel gap 0).
  Q축 head probe는 slope 1.10 / r 0.80으로 1차가 맞고, MLP 85채널 블록은 예측이 측정보다
  1–2자릿수 작고 **부호가 반대**(r −0.97)다. 이것이 단일 채널 probe를 추가한 이유다.
- **교차 확인**: `analyze_taylor_probe.py`의 raw 표가 출시 보고서의 expert 하위 50% 질량
  (Q 24.96% / MLP 0.44%)을 소수 둘째 자리까지 재현한다. VLM은 Q 21.13% / MLP 27.39%(traj)로
  **집중 구조가 없다** — expert의 "MLP 공짜"를 만든 편중이 VLM에는 없다는 뜻.

## 승인 요청 사항 (해소됨)

- Stage 1 클립 수 32 / probe 1,044 (≈ 2 h). 16 클립으로 줄이면 1 h, CI가 넓어진다.
- 2b는 세 세트 전부(9.6 GPU-h) vs val500만 먼저(3.2 GPU-h). 기본은 전부(expert-axis와 동일).
- 2c(VLM probe)는 기본 **생략**, 2b가 예측과 어긋날 때만.
