# Dual 분석 흐름: 피드백과 논문용 수정 문안

2026-09-22. Depth·raw rank는 `outputs/importance_v2/importance.npz`, 토큰별 비율은
`outputs/gradanat_v1/anatomy.npz`, 기여 비중·보정 상관은 같은 run의
`metrics_analysis.json`을 대조했다. 기존 결과만 분석했으며 모델 평가를 추가하지 않았다.

## 먼저 수정할 해석

1. **중요도 비율과 절대 중요도를 구분한다.** `I_traj / I_CoC`의 상승은 분자의 상승,
   분모의 하락 또는 두 변화의 조합일 수 있다. 실제로 vision 비율은 Q에서는 소폭
   증가하지만 MLP에서는 감소한다. 두 구조 모두 “후반 vision의 Traj 중요도가 커진다”로
   요약할 수 없다.
2. **Traj도 후반에는 text 기여가 커진다.** CoC가 text에 훨씬 강하게 집중하고,
   Traj가 vision·ego-history의 기여를 더 유지한다는 표현이 데이터에 맞는다.
   “CoC는 text만 처리한다” 또는 “Traj는 모든 modality를 잘 처리한다”는 기능적
   해석은 이 중요도 분석만으로 확인되지 않았다.
3. **비율 figure와 상관 figure를 구분한다.** 이번 첨부의 이미지 4–7은 모두 y축이
   `I_traj / I_CoC`인 비율 figure이며, 5·6·7은 같은 MLP 이미지다. 토큰 조건별
   순위 상관을 설명할 때는 [Q-head](fig4_same_token_q_head.pdf)와
   [MLP](fig4_same_token_mlp.pdf) 상관 figure를 배치한다.
4. **토큰 종류와 파라미터 종류는 일대일 대응하지 않는다.** 같은 head/channel이
   여러 토큰 위치에 적용된다. 따라서 “vision 전용 파라미터”보다
   “vision 토큰 위치에서의 연산이 중요도에 기여한다”가 정확하다.
   Text tokens는 prompt + CoC이며 CoC tokens와 겹치는 집합이다.
5. **Rank agreement는 같은 레이어 내부 unit들의 순위 상관이다.** 레이어들 자체의
   중요도 순위를 비교하는 분석과 구분한다.

## 논문에 바로 사용할 수 있는 문안

두 objective의 레이어별 중요도를 비교하면, 서로 다른 깊이에서 상대적 중요도가
최대가 된다. MLP의 경우 trajectory와 CoC 중요도의 peak는 각각 18층과 27층이며,
Q head에서는 각각 19층과 24층이다. 두 곡선은 각 objective의 최대값으로 독립적으로
정규화했으므로, 이 결과는 objective별 깊이 분포의 차이를 나타낸다. 동일 레이어
내부의 unit 순위를 비교한 평균 Spearman 상관도 0–21층에서 22–34층으로 이동할 때
Q head는 0.782에서 0.335로, MLP는 0.753에서 0.339로 낮아졌다. 따라서 두 objective는
후반층에서 서로 다른 unit 선택 신호를 제공한다.

이 차이가 어떤 토큰 위치에서 나타나는지 확인하기 위해, 토큰 종류별로 제한한
importance의 비율 `I_traj / I_CoC`를 분석했다. 0–21층과 23–34층을 비교하면,
text-side(prompt + CoC)의 비율은 Q head에서 8.98 → 1.86, MLP에서 6.29 → 2.10으로
감소했다. Ego-history에서는 각각 25.87 → 43.11, 23.20 → 35.49로 증가했다.
Vision 비율은 Q head에서 16.73 → 18.25로 소폭 증가했지만 MLP에서는
14.97 → 12.40으로 감소했다. 따라서 공통된 관찰은 후반에 vision의 Traj 중요도가
일괄적으로 증가한다는 것이 아니라, **text-side에서 두 objective의 상대적 중요도
비율이 크게 감소하는 반면 vision·ego-history에서는 높은 비율이 유지된다**는 것이다.
이 수치는 각 구간의 레이어별 비율을 기하평균한 값이며, 절대적인 비율 크기는
loss의 척도에 의존한다.

각 objective 내부에서 토큰 위치별 기여를 가법적으로 분해하면 이러한 차이를 더
직접적으로 볼 수 있다. 6–17층에서 vision의 기여 비중은 Q head의 Traj/CoC에서
각각 72.1%/62.6%, MLP에서 90.1%/74.9%로 높았다. 27–34층에서는 CoC의 text-side
기여가 Q head와 MLP 모두 약 98.2%에 달했다. Traj 역시 후반 text-side 기여가
각각 48.6%와 66.0%로 커졌지만, vision의 기여를 35.3%/26.4%, ego-history의 기여를
16.1%/7.5% 유지했다. 이는 **두 objective 모두 깊이에 따라 text-side 기여가 커지되,
CoC의 집중이 더 강하고 trajectory 중요도는 상대적으로 더 다양한 토큰 위치의
기여를 유지함**을 보여준다. 이 기여 비중은 비율 figure만에서 읽은 값이 아니라
별도의 signed-gradient 가법 분해에서 얻었다.

같은 토큰 종류에 제한한 순위 상관 분석도 이 해석을 보강한다. 22–34층에서
천장 보정 상관은 pooled/vision/text-side 순서로 Q head가 0.382/0.786/0.531,
MLP가 0.541/0.715/0.591이었다. 특히 vision 토큰으로 조건을 맞추면 pooled 조건보다
상관이 높아지므로, 후반의 전체 순위 불일치에는 토큰별 기여 구성의 차이가 반영된다.
다만 동일 토큰 조건에서도 상관이 완전히 회복되지는 않아, 토큰 구성만으로 모든
차이를 설명할 수는 없다. 이 보정 상관은 앞서 제시한 raw Spearman과 구분한다.

이 결과는 trajectory와 CoC 중요도가 공유 구조를 서로 다른 관점에서 평가한다는
것을 시사하며, 두 순위를 함께 고려하는 Dual 선택 규칙의 동기를 제공한다.
최종 방법은 layer·구조 종류별로 두 normalized rank의 최댓값을 취하고 고정 예산에서
top-k를 유지한다. 이 결합이 실제 기능 보존에 유리한지는 중요도 분석에 이어
동일 예산의 trunk/late pruning 개입으로 검증한다.

## 수치·집계 정의

서로 다른 분석 구간을 하나의 “early/late”로 묶지 않고 본문과 캡션에 명시한다.
기여 비중은 기존 anatomy 분석의 6–17/27–34층, rank는 0–21/22–34층,
비율은 변화점 전후의 0–21/23–34층을 사용한다.

### Depth와 raw rank

| 구조 | Traj peak | CoC peak | 평균 raw ρ, 0–21층 | 평균 raw ρ, 22–34층 |
|---|---:|---:|---:|---:|
| Q head | 19 | 24 | 0.782 | 0.335 |
| MLP channel | 18 | 27 | 0.753 | 0.339 |

Depth는 unit별 clip 평균 절대 gate gradient를 레이어 안에서 합산한 뒤
objective별로 최대 레이어 값으로 나눈다. Rank는 각 레이어 내부 unit 간 Spearman을
계산한 뒤 구간의 레이어 평균을 낸다. 35층은 Traj가 구조적으로 0이므로 rank에서 제외한다.

### 토큰별 중요도 비율

| 토큰 조건 | Q: 0–21 → 23–34층 | MLP: 0–21 → 23–34층 |
|---|---:|---:|
| Pooled | 14.30 → 3.14 | 12.88 → 2.79 |
| Vision | 16.73 → 18.25 | 14.97 → 12.40 |
| Text-side (prompt + CoC) | 8.98 → 1.86 | 6.29 → 2.10 |
| CoC only | 7.59 → 1.96 | 5.80 → 3.56 |
| Ego-history | 25.87 → 43.11 | 23.20 → 35.49 |

레이어별 raw layer-sum 비율의 **기하평균**이다. Text-side는 prompt와 CoC의 signed
gradient를 먼저 합한 뒤 abs를 취한다. 두 부분의 절대 중요도를 더한 값과는 다르다.
Pooled 역시 모든 토큰 종류의 signed gradient를 먼저 합한 뒤 abs를 취한다.
이 그림은 loss 척도가 다른 두 점수의 비율이므로, 수치가 10이라는 이유로 주행에서
해당 토큰이 언어보다 10배 중요하다는 의미로 해석하지 않는다.

### 후반 27–34층에서의 가법적 기여 비중

| 구조·objective | Vision | Text-side (prompt + CoC) | Ego-history |
|---|---:|---:|---:|
| Q · Traj | 35.3% | 48.6% | 16.1% |
| Q · CoC | 1.6% | 98.2% | 0.2% |
| MLP · Traj | 26.4% | 66.0% | 7.5% |
| MLP · CoC | 1.8% | 98.2% | 0.1% |

각 레이어에서 원래 pooled 점수를 signed gradient의 가법적 기여로 분해한 뒤,
레이어별 비중을 구간 평균했다. Sink 항은 매우 작아 표에서 생략했다.
`E|G_type| / Σ_type E|G_type|` 정규화 비중과는 다른 정의다.

구체적으로 token group k의 signed gradient를 \(G_{c,k,\ell,u}\)라 할 때,

\[
C_{k,\ell}=\frac1N\sum_{c,u}G_{c,k,\ell,u}
\operatorname{sign}\!\left(\sum_jG_{c,j,\ell,u}\right),\qquad
f_{k,\ell}=\frac{C_{k,\ell}}{\sum_j C_{j,\ell}}.
\]

분모는 원래 pooled layer score이며, token group별 기여는 합하면 이를 정확히 재구성한다.
Signed 기여이므로 일반적으로 음수도 가능하며 attention probability 또는 처리 정확도가 아니다.

### 같은 토큰 조건의 천장 보정 상관

| 구조 | 조건 | 0–21층 | 22–34층 |
|---|---|---:|---:|
| Q | Pooled | 0.851 | 0.382 |
| Q | Vision | 0.956 | 0.786 |
| Q | Text-side | 0.720 | 0.531 |
| MLP | Pooled | 0.922 | 0.541 |
| MLP | Vision | 0.976 | 0.715 |
| MLP | Text-side | 0.847 | 0.591 |

Cross-objective split-half 상관을 두 objective의 self-agreement 기하평균으로
나눈 후 레이어 평균한다. Self-agreement가 어느 한쪽이라도 0.2 이하이면 제외한다.
이 조건들에서는 각 13개 late 레이어 모두 포함됐다. VLM MLP의 raw rank 하락 중 일부는
측정 신뢰도 하락이므로 raw와 corrected를 구분한다.

## 권장 figure 순서와 주장 범위

1. **Depth + pooled raw rank:** 어떤 차이가 존재하는가.
2. **토큰별 비율:** 그 차이가 어떤 토큰 위치에서 두드러지는가.
3. **Token contribution/share:** 각 objective의 중요도에 어떤 토큰 위치가 얼마나 기여하는가.
   현재 모음에는 [MLP vision-share figure](supplementary/fig4_token_handoff_mlp.pdf)가 있다.
   위 후반 기여 표를 함께 제시하면 ratio로 share를 추정하는 문제를 피할 수 있다.
4. **Same-token corrected rank:** 토큰 조건을 맞추면 불일치가 얼마나 줄고 얼마나 남는가.
5. **[FM](fig5_damage_by_band_fm.pdf)·[NLL](fig5_damage_by_band_nll.pdf) 개입:**
   두 점수를 결합한 선택이 실제 손상을 줄이는가.

본문 결론은 “CoC는 언어만, Traj는 모든 modality를 처리한다”보다
**“공유 unit의 중요도에 대한 토큰별 기여 구성과 unit 순위가 objective에 따라 달라지므로,
두 기준을 결합한 선택을 고려하고 실제 제거 실험으로 검증한다”**가 타당하다.

## 출처

- [Depth·raw rank 계산](../experiments/paper/fig_criterion.py)
- [토큰별 ratio 계산](../experiments/paper/fig_why_differs.py)
- [가법 분해·보정 상관 계산](../experiments/head_analysis/analyze_gradient_anatomy.py)
- [Token anatomy 통계](../outputs/gradanat_v1/metrics_analysis.json)
- [원본 layer/unit 중요도](../outputs/importance_v2/importance.npz)
- [원본 token별 중요도](../outputs/gradanat_v1/anatomy.npz)
- [Dual와 expert 방법론 연결](CLAIMS.md)
