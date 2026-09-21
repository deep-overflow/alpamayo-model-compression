# 논문용 figure — 2026-09-21

지정한 6개 그림의 축 제목은 12 pt, 눈금·범례·수치 주석은 10 pt로 설정했다.
모든 선은 실선이다.
데이터와 중요도·상관계수 계산은 기존과 동일하다.

| 그림 | PNG |
|---|---|
| MLP layer importance | [fig1_depth_mlp.png](fig1_depth_mlp.png) |
| Q-head layer importance | [fig1_depth_q_head.png](fig1_depth_q_head.png) |
| Layer rank agreement | [fig1_rank_agreement.png](fig1_rank_agreement.png) |
| Cumulative importance | [fig2_mass_curve.png](fig2_mass_curve.png) |
| MLP denoising-step agreement | [fig2_steps_mlp.png](fig2_steps_mlp.png) |
| Q-head denoising-step agreement | [fig2_steps_q_head.png](fig2_steps_q_head.png) |

각 파일과 같은 이름의 PDF·SVG도 저장했다. 논문 배치에는 PDF를 사용할 수 있다.

| 그래프 | X축 | Y축 | 컬러바 |
|---|---|---|---|
| 레이어별 중요도 | VLM layer index | Normalized layer importance | — |
| 레이어별 순위 상관 | VLM layer index | Spearman’s ρ | — |
| 누적 중요도 곡선 | Fraction of units removed | Removed importance fraction | — |
| 스텝 간 상관 히트맵 | Denoising step | Denoising step | Spearman’s ρ |

레이어별 중요도는 각 objective의 레이어별 최대값으로 정규화했다.
누적 중요도 곡선의 Y축은 제거한 유닛의 중요도 합이 전체 중요도에서 차지하는 비율이다.

| 항목 | 원본 | 변경 |
|---|---:|---:|
| 축 제목 | 8 pt | 12 pt |
| 눈금·범례 | 7 pt | 10 pt |
| Mass curve 수치 주석 | 7.5 pt | 10 pt |
| Heatmap 색상 막대 제목 | 7.5 pt | 12 pt |
| Heatmap 색상 막대 눈금 | 7 pt | 10 pt |

폰트는 DejaVu Sans이며 math 기호도 같은 upright 폰트를 사용한다.
원래 크기인 3.5 × 2.8 inch와 가로:세로 비율 5:4를 유지한다.
캔버스 안에서 여백과 범례 배치를 조정했다.
PNG는 600 dpi, 2100 × 1680 px이며 PDF는 TrueType font 포함, SVG는 glyph path다.
Heatmap은 10 × 10 값을 그대로 표시하고 축 눈금 글자는 0·3·6·9에 표시한다.

- Traj: `#618BC8` = RGB(97, 139, 200).
- CoC: `#ECAE3C` = RGB(236, 174, 60).
- Q-head/MLP 비교선도 각각 위의 파랑/노랑을 사용한다.
- 기준선·peak 선·범례까지 모두 실선이다.
- Heatmap은 동일한 `viridis`, 0–1 색상 범위를 사용한다.

재현:

```bash
MPLCONFIGDIR=/tmp/alpamayo-mpl .venv/bin/python experiments/paper/fig_criterion.py --paper-style
```

이 옵션은 기본적으로 위 6개만 `paper-fig/`에 저장한다.
`--panels NAME ...`으로 저장 대상을, `--out-dir PATH`로 저장 폴더를 선택할 수 있다.
`--paper-style` 없는 기존 명령은 기존 스타일과 `figures/` 경로를 사용한다.

생성 코드: [fig_criterion.py](../experiments/paper/fig_criterion.py).
입력: `outputs/importance_v2/importance.npz`,
`outputs/stepimp_fm_perstep_v2/step_importance.npz`.
계산 정의는 [figure 재현 설명](../paper/2026-09-09_figure-reproducibility.md)을 참고한다.
