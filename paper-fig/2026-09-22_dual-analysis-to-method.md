# Objective별 중요도 분석에서 Dual pruning으로의 연결

논문 본문용 초안. 수치의 집계 구간·계산 정의는
[분석 흐름 피드백과 수치표](2026-09-22_analysis-flow-feedback.md)에 정리했다.
본문의 figure 번호는 최종 배치에 맞춰 삽입한다.

## 분석과 방법론 문단

Trajectory 예측과 CoC 생성에 필요한 구조를 하나의 중요도 기준으로 선택할 수 있는지
확인하기 위해, 두 objective가 VLM의 Q heads와 MLP channels를 어떻게 평가하는지
분석하였다. Objective별 최대값으로 정규화한 레이어 중요도는 서로 다른 깊이에서
peak를 보였다. MLP의 trajectory와 CoC 중요도는 각각 18층과 27층에서,
Q head는 19층과 24층에서 최대가 되었다. 각 레이어 내부의 unit 순위를 비교한
평균 Spearman 상관 역시 0–21층에서 22–34층으로 이동할 때 Q head는
0.782에서 0.335로, MLP는 0.753에서 0.339로 감소하였다. 이러한 결과는
두 objective가 깊이별 중요도 분포뿐 아니라 동일 레이어에서 우선시하는 unit에도
차이를 보인다는 것을 나타낸다.

이 차이를 해석하기 위해 중요도를 토큰 위치별로 분해하였다. Text-side(prompt + CoC)의
trajectory-to-CoC 중요도 비율은 후반층에서 크게 감소하는 반면, vision과 ego-history에서는
상대적으로 높은 비율이 유지되었다. 예를 들어 MLP의 0–21층과 23–34층을 비교하면,
text-side 비율의 기하평균은 6.29에서 2.10으로 감소하고 ego-history는
23.20에서 35.49로 증가하였다. Vision 비율은 14.97에서 12.40으로 낮아졌지만,
후반 text-side보다 높은 값을 유지하였다. 이 비율은 loss의 척도에 영향을 받으므로,
그 절대 크기보다는 깊이에 따른 변화와 토큰 조건 사이의 차이에 초점을 맞춘다.

각 objective 내부의 중요도를 토큰별 가법적 기여로 분해하면 두 기준의 차이가
더 명확해진다. MLP의 6–17층에서는 vision이 trajectory 중요도의 90.1%,
CoC 중요도의 74.9%를 차지하였다. 27–34층에서는 CoC 중요도의 98.2%가
text-side에 집중되었다. Trajectory 중요도 역시 text-side의 기여가 66.0%로
증가했으나, vision과 ego-history의 기여를 각각 26.4%와 7.5% 유지하였다.
따라서 두 objective 모두 깊이에 따라 text-side의 기여가 커지지만, CoC는
후반층에서 text-side에 더 강하게 집중하고 trajectory는 상대적으로 다양한
토큰 위치의 기여를 유지한다. 이는 공유 unit의 중요도가 어떤 토큰 위치에서
발생한 연산의 기여를 더 크게 반영하는지에 관한 차이이다.

같은 토큰 조건에서의 순위 비교는 토큰별 기여 구성의 차이가 전체 순위 불일치에
반영됨을 뒷받침한다. 22–34층에서 두 objective의 천장 보정 상관은 pooled 조건에서
Q head 0.382, MLP 0.541이었으나, vision 토큰에 제한하면 각각 0.786과 0.715로
높아졌다. 그럼에도 같은 토큰 조건에서 순위가 완전히 일치하지는 않았다.
이 결과는 두 objective가 공유 구조를 서로 다른 토큰 기여 구성과 unit 순위로
평가한다는 해석을 지지한다. 따라서 한 objective의 중요도만을 기준으로 선택하면,
다른 objective에서 높게 평가되는 unit을 충분히 고려하지 못할 수 있다.

이러한 관찰에 기반하여, 우리는 두 objective의 선택 신호를 고정된 pruning 예산
안에서 함께 반영하는 **Dual objective pruning**을 사용한다. 각 레이어의 Q heads와
MLP channels에 대해, 전체 토큰 기여를 합산해 얻은 trajectory 및 CoC 중요도를
각각 레이어·구조 내부의 normalized rank로 변환한 뒤 두 rank의 최댓값을 결합 점수로
정의한다.

\[
S^{\mathrm{Dual}}_{\ell,u}
=\max\left\{
R_{\ell,a}\!\left(I^{\mathrm{traj}}_{\ell,u}\right),
R_{\ell,a}\!\left(I^{\mathrm{CoC}}_{\ell,u}\right)
\right\},\qquad u\in\mathcal U_{\ell,a}.
\]

여기서 \(a\)는 Q head 또는 MLP channel을, \(\mathcal U_{\ell,a}\)는 해당 레이어와
구조의 unit 집합을 의미한다. \(R_{\ell,a}\)는 중요도가 클수록 높은 값을 갖는
\([0,1]\) 범위의 순위 변환이다. Rank 변환은 서로 다른 loss의 척도를 맞추고,
max 결합은 한 objective에서 높은 순위를 갖는 unit의 결합 점수가 다른 objective의
낮은 순위 때문에 줄어들지 않도록 한다. 이후 각 레이어와 구조에서
\(S^{\mathrm{Dual}}_{\ell,u}\)가 큰 상위 \(k_{\ell,a}\)개 unit을 유지한다.
이로써 두 objective 중 어느 하나에서 높게 평가되는 구조를 함께 고려하면서
정해진 압축 예산을 만족시킨다.

## 실제 개입 검증으로 이어지는 후속 문단

위 분석이 제공하는 설계 동기를 검증하기 위해, 동일한 구조별 제거 예산에서
trajectory-only, CoC-only, Dual의 mask를 서로 다른 레이어 대역에 적용하였다.
Dense 모델의 CoC를 고정한 held-out 100 clips 평가에서 공유 trunk(0–21층)에만
mask를 적용했을 때 FM loss 증가는 trajectory-only 26.1%, CoC-only 17.0%에 비해
Dual에서 4.8%로 작았다. 후반 22–35층만 제거했을 때 CoC NLL 증가는
trajectory-only 65.5%, CoC-only 6.3%, Dual 10.1%였다. 이 결과는 두 점수의 결합이
공유 trunk의 trajectory 기능 보존에 도움이 될 수 있으며, CoC 신호의 포함이
후반층의 언어 기능 보존에 기여함을 보여준다. 특히 큰 trajectory 차이가 관찰된
구간은 순위 불일치가 가장 큰 후반층이 아니라 공유 trunk였으므로,
중요도 상관과 실제 제거 효과를 함께 평가할 필요가 있다.

## 편집 시 유지할 조건

- 첫 문단의 raw Spearman과 네 번째 문단의 split-half 천장 보정 상관은 다른 통계다.
  Figure·caption에서 각각 표시한다.
- 기여 비중은 signed-gradient 가법 분해 결과다. Attention mass나 처리 정확도로
  설명하지 않는다. Text-side는 prompt와 CoC를 포함한다.
- 토큰 비율은 0–21/23–34층의 기하평균, 기여 비중은 6–17/27–34층의 레이어 평균,
  상관은 0–21/22–34층의 레이어 평균이다.
- Depth 분석은 unit 선택의 동기다. 현재 대표 recipe는 레이어별 제거율이 동일하며,
  peak가 높은 레이어에 더 많은 파라미터를 배정하는 방법으로 서술하지 않는다.
- Rank-max에서 top-k를 선택하므로, 두 단일 기준의 top-k 집합을 전부 보존한다는
  의미는 아니다. 다른 결합 연산보다 최적이라는 주장도 이 분석만으로 도출하지 않는다.
- 개입 문단은 고정 텍스트의 평가 조건을 유지한다. 전체 mask FM에서 Dual–Traj의
  차이는 유의하지 않았고(p=0.33), late NLL은 CoC-only가 더 낮다.
- 최종 구현에서 동점·상수 중요도의 rank 처리는 별도의 구현 세부사항이다.
  여기서는 실제 저장된 recipe의 선택 규칙을 설명하며 이상적인 동점 처리를 가정하지 않는다.

## Figure와 구현 대응

| 문단 | 자료 |
|---|---|
| Depth·unit rank 차이 | [MLP depth](fig1_depth_mlp.pdf), [Q depth](fig1_depth_q_head.pdf), [raw rank](fig1_rank_agreement.pdf) |
| 토큰별 비율 | [Q ratio](../figures/fig4_ratio_by_token5_q_head.pdf), [MLP ratio](../figures/fig4_ratio_by_token5_mlp.pdf) |
| 토큰별 기여 | [MLP vision contribution](supplementary/fig4_token_handoff_mlp.pdf), [수치표](2026-09-22_analysis-flow-feedback.md) |
| 같은 토큰 조건 | [Q](fig4_same_token_q_head.pdf), [MLP](fig4_same_token_mlp.pdf) |
| Rank-max 구현 | [rank_norm와 결합](../experiments/head_analysis/run_cocsafe.py), [대표 recipe](../recipes/README.md) |
| 실제 개입 | [FM loss](fig5_damage_by_band_fm.pdf), [CoC NLL](fig5_damage_by_band_nll.pdf) |
