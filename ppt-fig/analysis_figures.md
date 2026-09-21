# 발표용 분석 figure

기존 `figures/`의 논문용 그림을 확인하고 `ppt-fig/`에도 동일한 PNG·PDF·SVG를 모았다.
아래 PNG 링크와 같은 이름의 PDF·SVG도 사용할 수 있다.

Closed-loop의 쉬운·어려운 장면에 대한 새 분석 figure 4개와 발표 설명은
[난이도 분석](difficulty_figures.md)에 별도로 정리했다.
Hard100 평가의 동기는 [static difficulty 검증](fig_08_static_difficulty_validation.png)으로
설명할 수 있다.

| 순서 | 그림 | 설명할 내용 |
|---|---|---|
| 1 | [Q-head layer importance](fig1_depth_q_head.png) / [MLP layer importance](fig1_depth_mlp.png) | Trajectory와 CoC가 중요하게 보는 레이어 깊이가 다르다 |
| 2 | [레이어별 Spearman](fig1_rank_agreement.png) / [유지집합 overlap](fig1_kept_overlap.png) | 같은 레이어에서도 순위와 최종 유지 구조가 달라진다 |
| 3 | [Q-head step heatmap](fig2_steps_q_head.png) / [MLP step heatmap](fig2_steps_mlp.png) | Expert MLP의 중요도 순위는 denoising step 간 더 안정적이다 |
| 4 | [누적 importance mass](fig2_mass_curve.png) | MLP에는 raw importance가 작은 채널이 많이 분포한다 |
| 보충 | [Step 안정성과 하위 importance mass](fig2_stability_vs_mass.png) | Layer 단위의 안정성과 중요도 집중도 관계를 확인한다 |

15분 발표에서는 1·2·3을 각각 한 장에 두 그림씩 배치한다. 4는 expert pruning 동기를
설명할 시간이 있으면 사용하고, 마지막 scatter는 질의응답용으로 둔다.

## Dual objective

발표 문장: “두 objective는 중요하게 보는 깊이뿐 아니라 같은 레이어 안의 구조 선택도
다릅니다. 따라서 고정된 예산 안에서 두 중요도를 함께 반영하는 Dual을 비교했습니다.”

- Layer importance는 각 레이어의 구조 점수를 합한 뒤 **각 objective 곡선을 자신의 최댓값으로
  정규화**한다. Q-head의 peak는 trajectory 19 / CoC 24, MLP는 18 / 27이다.
  이 그림에서 objective 간 절대 gradient 크기를 비교하지 않는다.
- Spearman은 레이어 내부의 전체 구조 순위 관계다. Overlap은 각 objective로 top-k를 선택했을 때
  `|K_traj ∩ K_CoC| / k`이며 Jaccard가 아니다. Q는 19/32, MLP는 7,390/12,288을 유지한다.
- Layer 22–34 평균 overlap은 Q 67.6%, MLP 68.8%다. 독립 무작위 선택의 기대값은 각각
  59.4%, 60.1%다. Score가 상수인 마지막 레이어는 순위·선택 해석이 불가능해 제외한다.
- Overlap의 차이는 두 기준이 서로 다른 구조를 선택한다는 근거다. 성능 향상 여부는 별도 실험 표로 설명한다.

## Expert MLP

발표 문장: “Step 간 순위 상관은 MLP가 더 높고, 작은 raw importance를 가진 채널도 많습니다.
이 관찰을 바탕으로 expert MLP의 추가 압축을 실험했습니다.”

- Heatmap은 각 expert layer에서 step pair의 Spearman을 계산한 뒤 layer 평균한 것이다.
  대각선을 제외한 평균은 Q 0.668, MLP 0.874다. 두 그림은 같은 0–1 색 범위를 쓴다.
- Mass curve는 step별 raw score를 합하고, layer 안에서 오름차순 정렬·누적 정규화한 뒤
  layer 평균한다. Nominal 90% 제거 위치의 raw mass는 Q 68.5%, MLP 1.8%다.
  Q의 실제 위치는 head 개수 때문에 14/16 = 87.5%다.
- 이 mass curve는 최종 pruning에 쓰는 step·layer별 z-score 평균의 누적 질량이 아니다.
  최종 expert MLP importance와 그래프 정의를 구분한다.
- Scatter의 각 점은 하나의 layer다. 각 구조 축 내부의 상관은 Q −0.451, MLP −0.436이다.
  안정성이 pruning 내성을 일으킨다는 인과 결론까지 제시하지 않는다.

원본 생성 코드: [fig_criterion.py](../experiments/paper/fig_criterion.py).
입력: `outputs/importance_v2/importance.npz`, `outputs/stepimp_fm_perstep_v2/step_importance.npz`.
Calibration·target·gate·정규화의 상세 정의는
[figure 재현 설명](../paper/2026-09-09_figure-reproducibility.md)에 있다.

```bash
MPLCONFIGDIR=/tmp/alpamayo-mpl .venv/bin/python experiments/paper/fig_criterion.py
MPLCONFIGDIR=/tmp/alpamayo-mpl .venv/bin/python experiments/paper/make_meeting_tables.py
```

첫 명령은 논문 그래프를 생성하고, 둘째 명령은 발표용 표 생성과 분석 figure 복사를 수행한다.
