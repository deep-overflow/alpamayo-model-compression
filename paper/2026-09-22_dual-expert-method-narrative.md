**Dual + expert MLP pruning: 추가 분석을 반영한 방법론 설명**

2026-09-22. 아래 세 보고서와 저장된 통계·실험 코드를 대조했다.
새 GPU 실험은 수행하지 않았다. 원본 figure를 유지하면서 추가 분석의 논문용 사본을
[paper-fig](../paper-fig/README.md)에 모았다. [주장·캡션 모음](../paper-fig/CLAIMS.md)과
[전체 미리보기](../paper-fig/index.html)를 함께 제공한다.

- [Gradient anatomy](../reports/evaluation/2026-09-20_gradient-anatomy.html): gate 점수를 토큰 위치와 cache 경로로 분해.
- [Why importance differs](../reports/evaluation/2026-09-21_why-importance-differs.html): 변화점·측정 신뢰도·토큰·경로 분석을 통합.
- [Importance causal validation](../reports/evaluation/2026-09-21_importance-causal-validation.html): 무작위 readout, held-out unit 제거, 대역별·교차 mask, vision interaction 개입.

앞선 두 보고서의 일부 방법론 해석은 세 번째 보고서에서 정정됐다.
최종 설명에는 그 정정까지 반영한다. 대표 방법은 **plain Dual + expert MLP pruning**이며,
VLM reconstruction을 추가한 `dualrwl`의 효과와 구분한다.

**핵심 주장과 가장 중요한 변화**

> 두 출력 경로를 갖는 VLM을 한 목적의 국소 중요도로 압축하면 다른 경로에 필요한 구조를
> 충분히 보존하지 못할 수 있다. Dual은 trajectory와 CoC의 중요도를 결합하고,
> 실제 개입 실험에서는 공유 trunk의 주행 성능과 후반층의 언어 기능을 더 잘 보존하는
> 선택을 제공했다. 이후 trajectory expert에서는 head와 MLP의 압축 민감도를 구분해,
> Q heads를 유지하고 step별로 정규화한 점수로 MLP 채널을 추가 압축한다.

핵심 논거를 **“두 중요도의 순위가 다르다”에서 “두 중요도를 결합했을 때 실제로
더 나은 구조가 남는다”로 확장**할 수 있다. 특히 Dual의 주행 측면 이득이 뚜렷한 곳은
순위가 가장 크게 갈라지는 후반층만이 아니라 **공유 trunk, 0–21층**이다.
후반층에서는 CoC 기준이 언어 기능을 보존하는 역할에 대한 근거가 강하다.

**1. 기존 설명에서 무엇을 바꾸면 좋은가**

| 기존에 하기 쉬운 설명 | 추가 분석을 반영한 설명 | 방법론에 주는 의미 |
|---|---|---|
| Traj는 주행 회로, CoC는 언어 회로를 각각 찾는다 | 두 점수는 공유 계산을 서로 다른 토큰 구성과 출력 경로를 통해 측정한다 | 두 점수를 독립된 기능 지도처럼 취급하지 않고 상호 보완적인 선택 신호로 결합한다 |
| 후반층 Spearman이 낮으므로 Dual의 이득도 후반에 집중된다 | 후반의 불일치는 확실하지만 주행 측면의 큰 개입 효과는 trunk에서 관찰됐다 | 높은 순위 상관만으로 두 기준 중 하나를 생략하지 않는다 |
| CoC-only는 주행만, Traj-only는 언어만 잃는다 | 고정 텍스트 NLL에서는 비대칭이 있으나 자유 생성·LingoQA에서는 두 단일 기준 모두 손상된다 | 특정 loss의 보존과 전체 출력 능력의 보존을 구분한다 |
| Expert MLP의 작은 importance가 성능 보존을 보장한다 | 안정성·집중도는 MLP를 압축할 동기이며 실제 축별 pruning과 결합 평가로 효과를 확인한다 | First-order 점수와 실제 제거 효과를 연결하되 동일시하지 않는다 |

보고서에 있는 `union`은 본문에서 **고정 예산의 rank-max 선택**으로 풀어 쓰는 편이 정확하다.
실제 구현은 두 top-k 집합을 그대로 합쳐 모두 남기는 방식이 아니다.

**2. 중요도 차이의 원인: 두 점수는 무엇을 보고 있는가**

현재 depth와 Spearman figure 뒤에 gradient anatomy를 붙이면 관찰의 원인을 설명할 수 있다.

- Gate gradient는 토큰 위치에 대한 합이다: \(\partial L/\partial g_u=\sum_p x_u(p)\,\partial L/\partial x_u(p)\).
  따라서 같은 head/channel도 어느 토큰의 기여가 많이 들어오는지에 따라 순위가 달라진다.
- 두 점수 모두 대략 17–24층에서 vision 중심의 기여에서 text 중심의 기여로 넘어간다.
  “FM gradient는 끝까지 vision에만 남는다”는 예측은 지지되지 않았다.
- 후반 Q heads에서 pooled 두 점수의 천장 보정 상관은 약 0.38이지만,
  같은 vision 토큰으로 제한하면 0.79, 같은 prompt 토큰에서는 0.81이다.
  생성된 CoC 토큰만 보면 0.41로 여전히 낮다. 모든 토큰에서 일치한다고 쓰지는 않는다.
- [VLM MLP도 같은 토큰 조건으로 비교](../paper-fig/fig4_same_token_mlp.pdf)하면,
  후반 22–34층의 보정 상관 평균이 pooled 0.541에서 vision-only 0.715로 높아진다.
  Text-side(prompt + CoC)는 0.591이다. 토큰 구성은 차이의 일부를 설명하지만
  전체 불일치를 없애지는 않는다.
- FM gradient는 cache를 통해 VLM unit에 전달된다. Layer l의 output gate는 m > l인
  cache에만 영향을 줄 수 있다. Cache 22–23 경유분의 소실은 unit layer 21→23의
  trajectory 점수 하락 중 MLP 약 90%, Q 약 73%를 설명한다.

이 결과는 **왜 곡선이 갈라지는가**에 답한다. Gradient가 많이 들어오는 cache layer를
expert의 필수 정보가 가장 많이 들어 있는 cache layer와 동일시하지 않는다.
실제 cache knockout과 upstream gate-gradient 분해는 서로 다른 개입·민감도를 측정한다.

후속 무작위 readout은 이 해석을 더 직접적으로 뒷받침한다. 같은 입력과 모델에서
trajectory FM residual 대신 expert 출력의 무작위 선형 readout을 사용해도,
VLM layer 중요도 프로파일과 23층의 변화점이 재현됐다. Expert 경로 probe와 실제
trajectory 점수의 Q-head 순위 상관은 보정 후 약 0.99였고,
두 출력 경로의 probe끼리도 trunk 0.79 → late 0.35로 갈라졌다.

따라서 **중요도 차이는 loss의 의미뿐 아니라 gradient를 읽는 출력 경로의 영향도 크게
받는다**고 설명할 수 있다. 입력과 학습된 모델은 유지했으므로 “trajectory 정보가 전혀
없다”거나 “어떤 loss를 써도 동일하다”로 일반화하지 않는다.

방법론으로 잇는 문장:

> 하나의 출력 경로에서 계산한 중요도만으로 전체 VLM의 구조를 선택하기보다,
> LM-head와 trajectory expert를 통해 측정한 두 중요도를 함께 고려한다.

이것은 Dual의 두 측정 경로를 정당화한다. 두 gradient가 서로 반대 방향이어서 충돌한다는
설명은 지지하지 않는다. Residual gradient cosine은 순위가 비슷한 trunk에서도 거의 0이었다.

**3. 가장 강한 근거: 실제로 구조를 제거했을 때 무엇이 보존되는가**

핵심 그림은 [대역별 FM 손상](../paper-fig/fig5_damage_by_band_fm.pdf)과
[대역별 NLL 손상](../paper-fig/fig5_damage_by_band_nll.pdf)이다.

조건은 held-out 100 clips, 동일한 dense 생성 CoC에 teacher forcing, 동일 noise/seed다.
0–21층에 각 arm의 mask만 적용하고 22–35층과 expert는 dense로 유지했다.
세 방법은 해당 대역에서 같은 구조별 제거 개수를 사용한다.
모든 수치는 dense 대비이며, 이 minADE@8을 main table의 자유 생성 minADE@6와 섞지 않는다.

| Trunk에만 적용한 선택 기준 | FM loss 변화 | CoC NLL 변화 | 평균 ΔminADE@8 (m), 95% CI |
|---|---:|---:|---:|
| Traj-only | +26.1% | +51.2% | +2.64 [1.81, 3.54] |
| CoC-only | +17.0% | +28.3% | +0.46 [0.29, 0.64] |
| Dual | +4.8% | +25.2% | +0.06 [−0.05, 0.15] |

**같은 대역·예산에서 실제 선택을 바꾼 개입 비교**라는 점이 중요하다.
Trunk에서 두 기준의 순위 상관이 높더라도, 순위의 작은 차이를 반영한 선택은 큰 기능
차이를 낼 수 있었다. 특히 trajectory 기준만으로 선택한 trunk가 trajectory 평가에서도
가장 좋은 것은 아니었다. 이는 CoC 정보를 함께 반영하는 것이 언어 지표 외에도
trajectory 경로의 보존에 도움이 될 수 있다는 직접적인 경험적 근거다.

후반 22–35층만 pruning한 비교에서는 다음과 같다.

| Late에만 적용한 선택 기준 | FM loss 변화 | CoC NLL 변화 |
|---|---:|---:|
| Traj-only | −0.1% | +65.5% |
| CoC-only | +0.1% | +6.3% |
| Dual | +0.1% | +10.1% |

여기서 CoC 신호가 하는 역할은 언어 기능의 보존으로 드러난다.
더 직접적으로, **Traj-only가 버리고 Dual이 유지한 후반 Q heads**를 제거하면 NLL이
8.6% 증가했고, 크기를 맞춘 무작위 세트의 증가는 1.3–3.2%였다.
이때 FM 변화는 작았다. [Dual이 보존한 unit 개입](../paper-fig/supplementary/fig5_dual_saves.pdf).
이 결과를 모든 MLP 채널에도 동일하게 일반화하지 않는다. 해당 단위 수준의 검정은
Q head의 언어 측면에서 가장 분명했고, MLP에는 같은 강도의 근거가 없었다.

현재 범위에서 주장할 수 있는 내용:

> Dual은 공유 trunk에서는 두 단일 기준보다 작은 trajectory 손상을 보이는 구조를
> 선택하고, 후반층에서는 trajectory 단독 선택이 놓치는 언어 관련 head를 보존한다.

후반층 전체가 모든 조건에서 주행에 불필요하다는 뜻은 아니다. 위 결론은 정해진 mask와
제거율, dense 텍스트에 조건화한 평가 범위에서의 결과다.
전체 mask의 held-out FM 차이에서는 Dual과 Traj-only의 차이가 유의하지 않았다(p=0.33).
Trunk-only의 큰 차이를 전체 모델의 같은 크기 성능 향상으로 옮겨 말하지 않는다.

**4. 왜 단일 점수의 국소 분석만으로 충분하지 않은가**

교차 mask 실험은 구조 제거 효과가 대역 간 상호작용을 가진다는 점을 보여준다.
Traj-only trunk의 FM 손상은 후반층이 dense일 때 +26.1%지만,
같은 크기의 무작위 후반 mask를 더하면 +8.8%로 줄었다.
후반부를 추가 제거했는데 trajectory 지표가 개선된 것이다.

이 관찰은 두 대역의 손상을 단순 합산할 수 없음을 보여준다.
보고서가 제안한 “trunk 손상이 후반층을 거쳐 late cache에서 증폭된다”는 설명은
가능한 해석이다. 해당 표현을 실제로 측정한 cache 오염 또는 특정 head의 인과 경로가
완전히 규명된 것처럼 쓰지는 않는다.

발표 본문에서는 이를 한 문장으로 요약하는 편이 좋다.

> Pruning은 유한한 구조 변경이고 레이어 간 상호작용도 있으므로,
> 하나의 first-order 점수에 대한 높은 순위만으로 기능 보존이 보장되지는 않는다.

이는 Dual의 empirical 선택 이점을 설명하는 배경이며, max 연산의 수학적 최적성 증명은 아니다.
세부 교차 mask 표는 질의응답용으로 둔다.

공유 trunk의 역할을 보여줄 때는 [vision interaction 개입](../paper-fig/supplementary/fig5_vv_window_profile.pdf)을
보조 근거로 사용할 수 있다. Vision–vision attention을 누적 차단한 손상의 깊이 프로파일은
trajectory의 vision-token 중요도와 상관 0.85(Q)/0.83(MLP)을 보였고,
NLL 손상 프로파일과도 0.86의 상관이 있었다. 이는 두 출력의 공유 시각 계산이라는 해석을
뒷받침한다. 누적 차단의 차분은 다른 층도 차단된 조건에서의 기여이며 가법 기여가 아니다.
관찰된 attention mass 자체는 같은 중요도 지표로 사용할 근거가 약했다.

**5. 이 분석을 실제 Dual 규칙과 연결하는 방법**

실제 선택은 layer·구조 종류별 normalized rank를 결합한다.

\[
S^{\mathrm{VLM}}_{\ell,u}
=\max\{R_u(I^{\mathrm{traj}}_{\ell,u}),
       R_u(I^{\mathrm{CoC}}_{\ell,u})\}.
\]

Rank는 서로 다른 loss의 scale을 제거하고, max는 한 목적에서 높은 점수를 받은 unit을
다른 목적의 낮은 점수 때문에 평균에서 낮추지 않는 규칙이다. 해당 점수의 top-k를
고정 예산으로 유지한다. 대표 recipe는 layer당 Q 19/32, MLP 7,390/12,288 유지다.

추가 분석을 반영한 설계 설명은 다음 세 단계가 자연스럽다.

1. 서로 다른 출력 경로에서 중요도를 각각 측정한다.
2. 두 경로 중 하나에서라도 높게 평가된 unit을 고려하는 rank-max로 구조를 선택한다.
3. 실제 held-out 구조 제거 대조로 두 단일 기준보다 유리한 선택인지 확인한다.

현재 추가 실험은 **두 점수를 함께 사용하는 선택의 효용**을 강화한다.
Rank-max가 다른 모든 결합 규칙보다 낫다는 결론이나, early/late별 다른 allocation이
이미 적용된 방법이라는 설명은 포함하지 않는다. 후반층을 CoC-only, trunk를 Dual로
선택하는 새로운 hybrid도 가능한 후속 연구이지 현재 방법의 이름을 바꾼 것은 아니다.

**6. Expert MLP pruning으로 넘어가는 자연스러운 연결**

세 추가 보고서의 주된 검증 대상은 VLM의 중요도와 구조 선택이다.
Expert MLP의 중복성·안정성을 새로 입증한 보고서로 소개하지 않는다.
Expert에 대해서는 기존 step 분석과 실제 Q/MLP pruning 대조를 별도의 근거로 연결한다.

연결 문장:

> VLM에서는 두 출력에 필요한 표현을 함께 보존하도록 선택했다.
> 추가 압축은 trajectory expert 내부의 구조별 민감도를 확인한 뒤 수행한다.
> Step 간 순위가 상대적으로 안정적이고 중요도가 소수 채널에 집중된 MLP를 대상으로,
> 여러 step의 정보를 집계해 채널을 제거하고 Q heads는 유지한다.

| 설계 질문 | 직접적인 근거 | 적용한 방법 |
|---|---|---|
| VLM에서 어떤 unit을 남길까? | 두 출력 경로의 중요도 분석 + trunk/late의 실제 mask 개입 | Traj/CoC의 layer 내부 rank-max |
| Expert의 어떤 구조를 압축할까? | Step heatmap, raw mass 분포, Q vs MLP 실제 제거 비교 | Q heads 보존, MLP channels 제거 |
| Expert MLP 점수를 step 간 어떻게 합칠까? | Step별 척도를 맞춰 상대적 중요도를 집계하는 설계 | Step·layer별 z-score 후 step 평균 |
| 두 압축을 결합하면 무엇을 얻나? | Plain Dual parent와 MLP 사다리의 open/closed-loop·메모리 비교 | 추가 압축률에 따른 성능·비용 절충 보고 |

Expert의 최종 점수는:

\[
A_{s,\ell,u}=\frac1N\sum_c
\left|\frac{\partial L_{c,s}}{\partial g^E_{\ell,u}}\right|,
\qquad
S^E_{\ell,u}=\frac1{10}\sum_{s=0}^{9}
\operatorname{zscore}_{u}(A_{s,\ell,u}).
\]

식의 의미는 절대 gate gradient → clip 평균 → step·layer별 channel 표준화 → step 평균이다.
표준편차는 population std이고 0이면 분모를 1로 처리한다. 최종 음수 점수에 절댓값을
다시 취하지 않는다. Dense 측정 점수를 재사용하며, pruning 후 재보정한 점수가 아니다.

Step heatmap의 평균 상관은 MLP 0.874, Q 0.668이다.
Nominal 90% 제거 지점에서 raw 중요도 비율은 MLP 1.8%, Q 68.5%다
(Q의 실제 unit 비율은 14/16=87.5%). Mass curve는 \(\sum_s A_s\)의 분포로,
실제 선택 점수 \(\mathrm{mean}_s z(A_s)\)의 누적 질량과 다르다.
안정성과 집중도가 pruning 내성의 원인이라는 주장이나 `znorm`의 최적성은 별도 대조가 필요하다.

93.75% expert MLP 제거에서는 8,256 → 516 channels/layer를 유지하고,
전체 모델 제거율이 24.0% → 39.4%가 된다. Test500 minADE@6는 0.9498 → 0.9506 m로
변화가 작지만, closed-loop scene score는 0.8283 → 0.7928이며 paired Δ의
95% CI는 [−0.0726, −0.0005]다. 추가 압축을 “모든 지표의 성능 손실이 없는 압축”으로
표현하지 않는다. Expert MLP에 대한 최종 근거는 이 실제 조합 평가까지 포함해야 한다.

근거: [expert 점수 정의](2026-09-09_dualrwl-expert-mlp-importance.md),
[Q/MLP 축별 대조](../ppt-fig/table_02_expert_axis.csv),
[조합 평가와 CI](../ppt-fig/meeting_table_data.json).

**7. 본문과 15분 발표에서 우선할 그림**

| 역할 | 우선 자료 | 전달 문장 |
|---|---|---|
| 관찰 | 기존 depth + [rank agreement](../paper-fig/fig1_rank_agreement.pdf) | 두 기준의 중요도는 깊이와 구조에 따라 다르다 |
| 원인 설명 | [같은 토큰의 순위](../paper-fig/fig4_same_token_q_head.pdf) + [무작위 readout 비율](../paper-fig/fig5_probe_ratio_mlp.pdf) | 차이에는 토큰 구성과 출력 경로가 크게 작용한다 |
| **Dual의 핵심 검증** | **[대역별 FM](../paper-fig/fig5_damage_by_band_fm.pdf) + [대역별 NLL](../paper-fig/fig5_damage_by_band_nll.pdf)** | Trunk에서는 trajectory, 후반에서는 언어 기능 보존으로 선택의 차이가 드러난다 |
| 선택적 보강 | [Dual이 보존한 heads](../paper-fig/supplementary/fig5_dual_saves.pdf) | 단일 기준이 놓친 일부 head가 실제 언어 기능에 기여한다 |
| Expert 압축 동기 | 기존 Q/MLP step heatmap + raw mass curve | Expert는 구조별 압축 민감도가 다르다 |
| 결합 효과 | Dual + expert MLP 성능·메모리 표 | 추가 압축의 이득과 closed-loop 비용을 함께 확인한다 |

15분에는 port map, 전체 실패 가설 목록, 교차 mask의 모든 조합을 본문에 넣기보다
보충 자료로 둔다. 새 자료 중 하나만 추가할 수 있다면 **대역별 FM/NLL 개입 비교**를 고른다.
이 자료가 “분석이 흥미롭다”에서 “그래서 이 선택 규칙을 쓴다”로 넘어가는 연결을 가장 직접적으로 만든다.

추천 설명 순서는 **관찰 → 원인 분해 → 실제 구조 제거 → Dual 규칙 → expert 구조 선택 →
조합 평가**다. 연구 수행 날짜 순서로 보고서 세 개를 소개할 필요는 없다.

**8. 논문 표현의 범위**

- 기존 Spearman figure의 raw 상관과 anatomy의 천장 보정 상관을 같은 숫자처럼 섞지 않는다.
  Late VLM MLP 불일치의 일부는 측정 신뢰도 하락이다. 이는 expert MLP step 안정성과 별개다.
- 점수 변화점이 여러 calibration 조건에서 재현된다는 것과 최종 pruning 성능이
  calibration에 강건하다는 것은 다른 주장이다.
- Teacher-forced minADE@8 개입 결과, 자유 생성 minADE@6 주 표, closed-loop scene score를
  구분한다. CoC-only의 자유 생성 퇴화는 별도의 evidence로 제시한다.
- 사전 등록 검정이 모두 통과한 것은 아니다. 특히 late의 양방향 기능 분리, Dual의
  모든 dense 프로파일 보존, held-out full-mask FM에서의 Dual–Traj 차이는 지지되지 않았다.
- 초기 보고서가 제시한 “CoC-only의 후반 vision blind spot이 주행 손상의 원인” 및
  “Dual의 이득은 hand-over 이후에 집중”이라는 설명은 후속 개입 결과에 맞춰 수정한다.
- 최종 선택은 현재 코드와 일치시킨다. Rank의 상수 입력 처리 등 기존 구현 검토 사항은
  [audit](../reports/evaluation/2026-09-10_dual-code-audit.md)에 별도로 기록돼 있다.

**논문용 영문 문단**

> Our analysis links Dual pruning to the VLM's two output pathways. Token-wise
> gradient decomposition and random-readout probes show that differences between
> trajectory and CoC importance are strongly shaped by token contributions and the
> pathway through which gradients are measured. We therefore assess these scores
> through held-out structural interventions. Under matched pruning budgets, Dual
> preserves the shared trunk with smaller trajectory degradation than either
> single-objective criterion, while retaining late-layer heads whose removal harms
> language modeling. These results support combining both importance rankings within
> a fixed budget, without establishing rank-max as an optimal aggregation rule.
> We then compress the trajectory expert using separate evidence about structural
> sensitivity: attention heads are retained, while MLP channels are selected using
> step-normalized Taylor scores. The combined method enables additional parameter
> reduction with small open-loop changes, with closed-loop performance evaluated
> explicitly as a compression trade-off.

수치 대조 데이터: [causal validation stats](../figures/fig5_causal_validation_stats.json).
Held-out protocol: [run_token_ablation.py](../experiments/head_analysis/run_token_ablation.py).
