# Dual + expert MLP pruning: 주장과 피겨 연결

2026-09-22. 저장된 분석 결과를 재정리했으며 새 모델 평가를 실행하지 않았다.
[전체 피겨 미리보기](index.html) · [파일 목록·재현](README.md) ·
[상세 분석과 방법론 설명](../paper/2026-09-22_dual-expert-method-narrative.md).

Depth·토큰 분석을 연결한 [논문 문안과 수치 정의](2026-09-22_analysis-flow-feedback.md)를
함께 참고한다. 토큰별 비율과 objective 내부 기여 비중을 구분해 서술했다.
이를 Dual의 rank-max 선택으로 연결한 [본문용 초안](2026-09-22_dual-analysis-to-method.md)도 제공한다.

## 1. 논문의 중심 주장

> VLM의 trajectory와 CoC 출력에 필요한 구조를 두 중요도의 결합으로 선택하고,
> trajectory expert에서는 구조별 압축 민감도에 따라 Q heads를 보존하면서
> MLP 채널을 추가 압축한다. 이 선택의 근거는 중요도 분포 관찰과 실제 구조 제거
> 실험이며, 최종 효과는 open-loop·closed-loop·reasoning 성능과 압축률로 검증한다.

설명 순서는 **관찰 → 토큰·출력 경로 분석 → held-out 구조 제거 → Dual 선택 규칙 →
expert 구조 선택 → 결합 모델의 성능·비용**이 자연스럽다.
추가 보고서 세 개를 작성 순서대로 소개할 필요는 없다.

| 주장 | 피겨·표 | 방법론으로 연결하는 문장 |
|---|---|---|
| 두 중요도는 동일한 구조를 동일하게 평가하지 않는다 | Depth, layer rank agreement | 하나의 objective 점수만으로 VLM 전체의 보존 대상을 결정하지 않는다 |
| 점수 차이는 토큰 구성과 gradient가 전달되는 출력 경로에 크게 영향을 받는다 | Same-token Q/MLP, random-readout MLP | 서로 다른 두 출력 경로에서 얻은 선택 신호를 함께 고려한다 |
| 두 신호의 결합은 실제로 유리한 구조 선택을 제공한다 | **대역별 FM·NLL 개입** | 같은 예산에서 Dual의 trunk는 trajectory 손상이 작고, 후반 CoC 신호는 언어 관련 구조를 보존한다 |
| Expert의 Q와 MLP는 같은 제거율에 동일하게 반응하지 않는다 | Step heatmap, raw mass, Q/MLP 제거 비교 표 | Q heads를 보존하고 여러 step을 집계한 점수로 MLP 채널을 선택한다 |
| 결합 방법은 추가 압축을 가능하게 한다 | Main 성능 표, expert 압축 사다리 표 | 높은 압축률에서 얻는 이득과 closed-loop 비용을 함께 보고한다 |

**기존 해석의 수정:** 순위가 크게 갈라지는 후반층에 Dual의 주행 이득이 집중된다고
설명하지 않는다. 후속 개입에서는 **순위 상관이 높은 공유 trunk(0–21층)**에서 큰
trajectory 차이가 관찰됐다. 높은 순위 상관이 두 선택 기준의 교환 가능성을 뜻하지 않는다.

## 2. 본문에 배치할 피겨와 캡션

아래 A–D는 논문의 개념적 묶음이다. 파일명은 기존 보고서와 추적할 수 있도록 유지했다.
개별 패널의 캔버스는 모두 3.5 × 2.8 inch다.

### A. 두 objective를 함께 고려해야 하는 관찰

파일: [MLP depth](fig1_depth_mlp.pdf), [Q-head depth](fig1_depth_q_head.pdf),
[layer rank agreement](fig1_rank_agreement.pdf).

**전달할 내용:** 두 objective의 중요도는 깊이에 따른 분포와 동일 레이어 내 unit 순위에서
차이를 보인다. 이것은 두 기준을 비교할 동기이며 성능 이득의 직접 증명은 아니다.

**영문 캡션 초안:**

> Layer-wise importance profiles and within-layer rank agreement for trajectory and
> CoC objectives. Each depth profile is normalized by its own maximum across layers;
> Spearman correlation compares unit rankings within the same layer. The profiles
> reveal different relative depth preferences and reduced rank agreement in late layers.

주의: 서로 다른 loss의 절대 크기를 비교하는 그림이 아니다. Q와 MLP의 layer score는
각 레이어의 unit importance를 합산한 값이다. 마지막 레이어의 trajectory score는
output gate에서 expert가 읽는 후속 cache로 가는 경로가 없어 0이다. 이 레이어의
상수 rank는 유효한 Spearman 값으로 해석하지 않는다. Late VLM MLP 상관 하락의
일부는 calibration 측정 신뢰도 저하이며, expert MLP의 step 안정성과 구분한다.

### B. 그 차이는 어디에서 생기는가

파일: [same-token Q heads](fig4_same_token_q_head.pdf),
[same-token MLP channels](fig4_same_token_mlp.pdf),
[random-readout MLP ratio](fig5_probe_ratio_mlp.pdf).

**전달할 내용:** token support를 맞추면 일부 불일치가 줄고, 같은 출력 경로의 무작위
readout에서도 depth 차이가 재현된다. 두 점수를 각각 순수한 주행 회로·언어 회로의
지도라고 부르기보다, 공유 계산을 다른 경로에서 측정한 중요도라고 설명한다.

**Same-token 캡션 초안:**

> Ceiling-corrected cross-objective rank agreement of VLM Q heads and MLP channels after restricting
> both importance measurements to the same token group. Vision-restricted agreement
> exceeds pooled agreement in late layers. Text tokens include prompt and generated
> CoC positions. Layer-wise correction divides cross agreement by the geometric mean
> of the two self-agreements; layers with either self-agreement ≤ 0.2 are omitted.

Late(22–34층)의 레이어별 보정 상관 평균:

| VLM 구조 | All tokens | Vision tokens | Text tokens |
|---|---:|---:|---:|
| Q head | 0.382 | 0.786 | 0.531 |
| MLP channel | 0.541 | 0.715 | 0.591 |

MLP도 동일한 vision 토큰에 제한하면 상관이 높아지지만, 불일치가 사라지지는 않는다.
각 조건에서 13개 late 레이어 모두 self-agreement > 0.2 기준을 통과했다.
Q와 MLP에 동일한 축 범위(−0.3–1.1)·보정 방식·색·서체를 적용했다.
Q-head prompt-only의 0.81은 별도 분석값이며 이 그림의 Text tokens 곡선 값이 아니다.
보정 상관과 기존 A의 raw Spearman을 그대로 비교하지 않는다.

**Random-readout 캡션 초안:**

> VLM MLP importance ratios for the actual objectives and random linear readouts
> at the expert output, LM-head input, and cache entries. Each ratio is divided by its mean
> over layers 0–21 and plotted on a logarithmic scale. Random readouts through the
> same pathways reproduce the depth transition, supporting a substantial pathway
> contribution to the observed importance profiles.

검정 곡선은 실제 Traj/CoC, 파랑은 random expert/random LM-head,
노랑은 random cache/random LM-head 비율이다. 여기서 파랑·노랑은 각각의 단독 loss가
아닌 **readout 경로 비교**다. 모델의 학습된 가중치·입력·FM 상태는 유지했으므로
“task 정보가 없다”거나 “아무 loss를 써도 같다”는 결론은 아니다.

**무작위 readout의 뜻과 그래프 읽는 방법:**

Readout은 모델의 벡터를 스칼라 값으로 읽어내는 함수다. 이 대조에서는 무작위 벡터
\(r\)와 중간 표현 \(h\)의 내적 \(L_{probe}=r^\top h\)를 계산하고,
그 스칼라를 gate까지 역전파해 중요도를 측정했다. LM-head probe는 CoC 예측 위치의
**LM head 직전 hidden state**, expert probe는 같은 FM 상태에서의 **expert 출력**,
cache probe는 **각 레이어의 K/V tensor**를 읽는다.

| 그래프 곡선 | 비교하는 layer importance 비율 |
|---|---|
| Actual objectives | 실제 trajectory FM loss / 실제 CoC NLL |
| Expert / LM head | Expert 출력의 random readout / LM-head 입력의 random readout |
| Cache / LM head | K/V cache의 random readout / LM-head 입력의 random readout |

각 비율은 0–21층 평균으로 나누었으므로 초기 구간이 대략 1이 된다.
실제 objective와 random expert/head 비율 모두 22–23층 근처에서 하락한다.
즉 과제의 정답을 사용하는 loss를 random readout으로 바꿔도 같은 출력 위치에서
역전파하면 depth 차이가 유사하게 남는다. 이 결과는 점수가 gradient 전달 경로의
영향을 크게 받는다는 근거다. Expert 경로만이 변화의 유일한 원인이라는 결론은 아니다.

발표용 문장: **“중요도의 차이에는 과제의 목표뿐 아니라 gradient를 측정하는 출력
경로가 반영된다. 따라서 두 출력 경로의 점수를 함께 고려하고 실제 제거로 검증한다.”**

### C. Dual의 가장 직접적인 검증: 실제 구조 제거

파일: **[FM loss increase](fig5_damage_by_band_fm.pdf)**,
**[CoC NLL increase](fig5_damage_by_band_nll.pdf)**.

**전달할 내용:** 같은 pruning 예산에서 선택 기준을 바꾸면 실제 기능 손상이 달라진다.
이 두 패널을 본문의 핵심 분석으로 권한다.

**영문 캡션 초안:**

> Held-out effects of applying equal-budget VLM pruning masks to the trunk
> (layers 0–21), late layers (22–35), or all layers (0–35). Bars report mean paired
> absolute changes from dense in FM loss and CoC NLL; error bars are 95% bootstrap
> confidence intervals across 100 held-out clips. All arms use the same dense CoC
> sequence under teacher forcing and matched stochastic conditions. Dual reduces
> trunk trajectory damage, while including CoC importance protects late-layer
> language modeling relative to trajectory-only selection.

CI는 clip 단위 paired 차이를 5,000회 bootstrap한 percentile 구간이며 RNG seed는 0이다.

| 적용 대역 | 기준 | FM 변화, dense 대비 | NLL 변화, dense 대비 |
|---|---|---:|---:|
| Trunk 0–21 | Traj | +26.1% | +51.2% |
| Trunk 0–21 | CoC | +17.0% | +28.3% |
| Trunk 0–21 | Dual | **+4.8%** | +25.2% |
| Late 22–35 | Traj | −0.1% | +65.5% |
| Late 22–35 | CoC | +0.1% | **+6.3%** |
| Late 22–35 | Dual | +0.1% | +10.1% |

**그래프 y축은 절대 loss 차이이고 위 표는 상대 변화율**이다. Dense 기준은 FM
0.242487, NLL 0.170763이다. 이 실험의 teacher-forced minADE@8에서 trunk-only
Dual의 평균 변화는 +0.06 m [−0.05, +0.15], Traj-only는 +2.64 m [1.81, 3.54]였다.
이를 자유 생성 minADE@6 주 평가표와 합쳐서 보고하지 않는다.

Dual이 모든 대역·지표에서 최고인 것은 아니다. Late NLL은 CoC-only가 더 낮으며,
전체 mask에서 Dual–Traj의 FM 차이는 유의하지 않았다(p=0.33).
이 결과로 “후반층은 주행에 불필요하다”거나 “Dual은 언제나 우월하다”고 쓰지 않는다.

**방법론으로 잇는 문장:**

> 중요도 상관만으로 선택의 효용을 판단하지 않고, 동일 예산의 실제 제거 실험을 통해
> 두 중요도를 함께 사용하는 근거를 확인했다. 이에 각 레이어와 구조 종류 내부에서
> 두 중요도의 rank를 결합해 고정 개수의 unit을 유지한다.

실제 VLM 선택 규칙:

\[
S^{VLM}_{\ell,u}=\max\{R_u(I^{traj}_{\ell,u}),R_u(I^{CoC}_{\ell,u})\}.
\]

Layer·구조별 normalized rank를 결합하고 top-k를 유지한다. Rank는 loss scale 차이를
없애며, max는 한 기준에서 높은 unit의 점수가 다른 기준의 낮은 점수와 평균되지 않게
한다. 두 top-k 집합의 단순 합집합이 아니며 rank-max의 최적성을 입증한 실험도 아니다.

### D. Expert는 왜 MLP를 압축하는가

파일: [MLP step agreement](fig2_steps_mlp.pdf), [Q-head step agreement](fig2_steps_q_head.pdf),
[raw importance mass](fig2_mass_curve.pdf), **[실제 Q/MLP 제거 비교](tables/table_02_expert_axis.pdf)**.

**전달할 내용:** MLP의 안정성과 중요도 집중도는 구조 선택의 동기다. 실제 Q/MLP
pruning 대조와 최종 조합 평가를 붙여 압축 효과를 뒷받침한다.

**Step·mass 캡션 초안:**

> Expert structure sensitivity across ten sampled flow-matching times. MLP channel
> rankings are more consistent across steps than Q-head rankings. The cumulative
> mass plot reports cumulative raw importance fractions removed within each layer,
> averaged across layers. These diagnostics motivate MLP-focused compression; empirical pruning
> comparisons assess its effect on trajectory quality.

| 관찰·대조 | 결과 | 해석 범위 |
|---|---|---|
| Step 간 평균 비대각 Spearman | MLP 0.874, Q 0.668 | 현재 calibration에서 MLP 순위가 상대적으로 안정적 |
| Nominal 90% 제거 raw importance | MLP 1.8%, Q 68.5% | 구조별 분포 비교. Q 실제 제거는 14/16=87.5% |
| Q50 − MLP50, 동일 unit 제거율 | 평균 ΔminADE@6 +0.2459 m [0.1694, 0.3168] | Q 제거의 손상이 큼. 제거 파라미터 수는 다름 |
| Q25 − MLP4.13, 약 75.5M/75.4M 제거 | 평균 +0.0317 m [−0.0355, 0.0761]; 중앙값 +0.0217 m [0.0103, 0.0375] | 파라미터를 맞춘 대조는 평균 CI가 0을 포함함 |

Expert의 최종 선택 점수는 clip별 절대 gate gradient의 평균을 step·layer 내부에서
z-score로 표준화한 뒤 10개 step에 걸쳐 평균한 값이다:

\[
A_{s,\ell,u}=\frac{1}{N}\sum_c\left|\frac{\partial L_{c,s}}{\partial g^E_{\ell,u}}\right|,
\qquad S^E_{\ell,u}=\frac1{10}\sum_s\operatorname{zscore}_{u}(A_{s,\ell,u}).
\]

Population std(ddof=0), std=0이면 분모 1, 최종 음수 값에 abs를 다시 적용하지 않는다.
100 calibration clips, 10개 FM midpoint time `(s + 0.5) / 10`, step별 새 noise로
측정했다. 실제 순차 rollout의 연속 상태를 측정한 heatmap이 아니다.
Mass curve는 raw step 합의 분포로, 최종 z-score 평균의 질량과 다르다.
각 레이어에서 제거한 raw 중요도 비율을 계산한 뒤 레이어 평균을 낸다.
Heatmap도 step 쌍마다 레이어 내부 unit 순위의 Spearman을 구한 뒤 레이어 평균을 낸다.
작은 raw importance 질량이 같은 비율의 실제 loss 손상을 뜻하지 않는다.

## 3. 최종 성능으로 주장을 마무리하는 방법

주 평가표: [open-loop](tables/table_01_open_loop.pdf),
[closed-loop](tables/table_01_closed_loop.pdf), [LingoQA](tables/table_01_lingoqa.pdf).
추가 압축: [expert open-loop](tables/table_03_expert_open_loop.pdf),
[expert closed-loop](tables/table_03_expert_closed_loop.pdf).
표는 `ppt-fig`의 기존 결과를 그대로 복사했다. CSV도 같은 폴더에 있다.

| 지표 | Dual | Dual + expert MLP93.75 |
|---|---:|---:|
| 전체 모델 파라미터 제거율 | 24.0% | 39.4% |
| Expert MLP 유지 채널 / layer | 8,256 | 516 |
| Test500 minADE@6 (m) | 0.9498 | 0.9506 |
| Test500 minFDE@6 (m) | 2.5459 | 2.5386 |
| OOD-val262 minADE@6 (m) | 1.1192 | 1.1176 |
| OOD-val262 minFDE@6 (m) | 2.8514 | 2.8440 |
| Closed-loop scene score | 0.8283 | 0.7928 |
| Collision@fault | 2.3% | 3.0% |
| Offroad | 6.7% | 8.3% |
| Progress | 0.8276 | 0.8163 |

**권장 결론:** “Expert MLP를 추가 압축해 전체 제거율을 39.4%까지 높였고, open-loop
변화는 작았다. Closed-loop에서는 성능 비용이 관찰되어 압축률에 따른 절충을 보고한다.”
Scene-score paired Δ는 −0.0355, 95% CI [−0.0726, −0.0005]다.
“성능 손실 없는 압축”이나 “안전성까지 개선”으로 표현하지 않는다.
87.5% 제거 결과도 압축 사다리에 함께 제시하며, 그 설정이 최적이라고 단정하지 않는다.
주 조합은 plain Dual이며 `dualrwl` reconstruction의 결과와 섞지 않는다.

## 4. 보충 피겨의 용도와 캡션

| 파일 | 답하는 질문 | 해석·캡션에 필요한 조건 |
|---|---|---|
| [Dual retained units](supplementary/fig5_dual_saves.pdf) | Traj-only가 놓친 unit이 실제로 필요한가? | `Q vs Traj`는 Dual이 유지하고 Traj가 제거하는 late Q heads. 제거 시 NLL +8.6%, matched random +1.3–3.2%. 선택 set error bar는 95% bootstrap CI, random은 **3개 set의 min–max**. 초록은 선택 set, 회색은 random. 네 그룹을 모두 보여주며 MLP까지 같은 강도의 효과를 주장하지 않음 |
| [Token handoff](supplementary/fig4_token_handoff_mlp.pdf) | 중요도에서 vision의 기여는 어디서 줄어드는가? | Signed token 기여를 원래 gate score로 분해한 share. 두 objective 모두 vision에서 text-side 중심으로 이동. Attention probability 비율이 아님 |
| [Cache contribution](supplementary/fig4_itraj_ports_mlp.pdf) | Traj depth profile의 급격한 감소는 무엇과 연결되는가? | Cache 경로별 additive gradient 기여, 원래 profile 최대값으로 정규화. 21→23층 점수 하락의 약 89.5%가 cache 22–23 경유분 소실과 연결. Cache 정보의 필수성 또는 knockout 손상과 동일시하지 않음 |
| [Vision interaction](supplementary/fig5_vv_window_profile.pdf) | 공유 trunk의 vision 계산은 실제 trajectory와 연결되는가? | 50 validation clips의 누적 vision–vision 차단 손상 차분과 vision-token Q importance. 각 profile을 자체 최대값으로 정규화. 다른 레이어도 차단된 조건의 차분으로 가법적 독립 기여가 아님 |

선택적 추가 캡션:

> Removing late-layer units retained by Dual but discarded by a single-objective
> criterion. Green bars show the selected unit sets, with 95% bootstrap confidence
> intervals across clips. Gray bars show the average over three matched-size random
> sets, with their minimum-to-maximum range. The clearest language-modeling effect
> is observed for Q heads retained over trajectory-only selection.

## 5. 15분 발표에서 사용할 순서

| 시간 | 내용 | 화면 |
|---|---|---|
| 0–1분 | VLM 두 출력과 trajectory expert의 압축 문제 | 모델 구조·문제 정의 |
| 1–3분 | 왜 두 objective가 필요한가 | A의 depth 하나 + rank; B의 readout은 한 문장 또는 backup |
| 3–6분 | **같은 예산의 실제 개입에서 무엇이 보존되는가** | C의 FM/NLL 두 패널 |
| 6–7분 | 실제 Dual 선택 규칙 | Layer 내부 rank-max 식 |
| 7–10분 | Expert MLP 선택의 근거와 점수 | D의 heatmap 두 개 + mass 또는 Q/MLP 대조 |
| 10–14분 | 주요 benchmark와 추가 압축의 절충 | 주 성능 표 + expert open/closed-loop |
| 14–15분 | 핵심 주장과 적용 범위 | 전체 제거율·보존 성능·closed-loop 비용 |

토큰 세부분해, cache별 bar, 단위 집합 개입, 모든 교차 mask 조합은 질의응답 자료로 둔다.
발표 슬라이드 제목은 “두 중요도가 다르다”보다
**“두 중요도의 결합이 공유 trunk의 trajectory 기능을 더 잘 보존한다”**가 핵심 결과를
명확하게 전달한다. 이때 held-out 고정 텍스트 개입이라는 조건을 함께 표시한다.

## 6. 출처

- [Gradient anatomy 보고서](../reports/evaluation/2026-09-20_gradient-anatomy.html)
- [Importance causal validation 보고서](../reports/evaluation/2026-09-21_importance-causal-validation.html)
- [Why importance differs 보고서](../reports/evaluation/2026-09-21_why-importance-differs.html)
- [분석 수치](../figures/fig5_causal_validation_stats.json), [anatomy·경로 수치](../figures/fig3_why_differs_stats.json)
- [기존 6개 피겨 재현 정의](../paper/2026-09-09_figure-reproducibility.md)
- [Expert 중요도 구현 정의](../paper/2026-09-09_dualrwl-expert-mlp-importance.md)
- [입력·산출물 SHA-256 manifest](manifest.json)
