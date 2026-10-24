# 논문용 figure 모음 — 2026-09-22

**[전체 피겨 미리보기](index.html)** · **[주장·캡션·발표 구성](CLAIMS.md)** ·
[상세 방법론 설명](../paper/2026-09-22_dual-expert-method-narrative.md)

**[논문 본문용: 분석에서 Dual 방법론으로 연결](2026-09-22_dual-analysis-to-method.md)**

[분석 흐름 피드백·수치가 포함된 논문 문안](2026-09-22_analysis-flow-feedback.md):
깊이별 분포 → 토큰별 비율 → 가법적 기여 비중 → 같은 토큰의 순위 상관을 연결한다.

기존 분석 피겨 6개를 유지하고 추가 분석 9개를 같은 논문 스타일로 재생성했다.
현재 분석 피겨는 본문 후보 11개, 보충 4개이며, 기존 성능 표 6개도 별도로 모았다.
모두 저장된 실험 결과를 사용했고 새 모델 평가를 수행하지 않았다.
파일명 `fig1`–`fig5`는 원래 분석과의 대응을 위한 것이며 논문 최종 번호는 아니다.

## 본문에서 사용할 순서

| 역할 | 파일 | 핵심 내용 |
|---|---|---|
| 관찰 | [MLP depth](fig1_depth_mlp.pdf), [Q depth](fig1_depth_q_head.pdf), [rank agreement](fig1_rank_agreement.pdf) | 두 objective의 깊이별 분포와 unit 순위 차이 |
| 원인 | [same-token Q](fig4_same_token_q_head.pdf), [same-token MLP](fig4_same_token_mlp.pdf), [random readout MLP](fig5_probe_ratio_mlp.pdf) | 토큰 구성과 출력 경로가 중요도에 미치는 영향 |
| **핵심 개입 검증** | **[FM loss](fig5_damage_by_band_fm.pdf)**, **[CoC NLL](fig5_damage_by_band_nll.pdf)** | 같은 예산의 trunk·late·전체 mask 비교 |
| Expert 압축 동기 | [MLP step](fig2_steps_mlp.pdf), [Q step](fig2_steps_q_head.pdf), [raw mass](fig2_mass_curve.pdf) | 구조별 step 안정성과 중요도 집중도 |

본문에서는 여러 패널을 3–4개 figure로 묶는 것을 권한다.
캔버스 비율을 유지하도록 개별 파일로 제공한다. 자세한 주장과 영문 캡션은 [CLAIMS.md](CLAIMS.md)에 있다.

## 보충 자료

| 파일 | 용도 |
|---|---|
| [supplementary/fig5_dual_saves.pdf](supplementary/fig5_dual_saves.pdf) | Dual이 보존한 unit의 실제 NLL 기여 |
| [supplementary/fig4_token_handoff_mlp.pdf](supplementary/fig4_token_handoff_mlp.pdf) | 토큰별 중요도 기여의 깊이 변화 |
| [supplementary/fig4_itraj_ports_mlp.pdf](supplementary/fig4_itraj_ports_mlp.pdf) | Traj gradient의 cache 경로 분해 |
| [supplementary/fig5_vv_window_profile.pdf](supplementary/fig5_vv_window_profile.pdf) | 누적 vision interaction 차단과 중요도 비교 |

## 성능 표

`tables/`에는 다음 표를 `ppt-fig/`에서 변경 없이 복사했다.
각 표는 PNG/PDF/SVG/CSV를 포함한다. 표의 원래 크기·서체 크기는 그대로다.

- [Open-loop](tables/table_01_open_loop.pdf): Test500, OOD-val262의 minADE@6/minFDE@6.
- [Closed-loop](tables/table_01_closed_loop.pdf): scene score, collision@fault, offroad, progress.
- [LingoQA](tables/table_01_lingoqa.pdf): reasoning benchmark.
- [Expert Q/MLP 제거 대조](tables/table_02_expert_axis.pdf): 동일 unit 비율·파라미터 수 비교.
- [Expert open-loop 압축 사다리](tables/table_03_expert_open_loop.pdf).
- [Expert closed-loop 압축 사다리](tables/table_03_expert_closed_loop.pdf).

## 스타일과 재현

모든 분석 피겨는 축 이름 12 pt, 눈금·범례·주석 10 pt, 실선,
DejaVu Sans, 3.5 × 2.8 inch(5:4), PNG 600 dpi(2100 × 1680 px)다.
논문에는 같은 이름의 PDF 또는 SVG를 사용할 수 있다.
Traj `#618BC8`, CoC `#ECAE3C`, Dual `#009E73`를 사용한다.
Token·cache·probe 구분을 위한 별도 색은 범례로 표시하며,
probe 비율 그래프의 파랑/노랑은 단독 loss가 아닌 출력 경로 비교다.

추가 9개 피겨·표 복사·미리보기·manifest 재현:

```bash
MPLCONFIGDIR=/tmp/alpamayo-mpl .venv/bin/python experiments/paper/make_claim_figure_pack.py
```

기존 6개 파일이 먼저 있어야 하며 이 명령은 그 파일들의 hash가 바뀌지 않았는지 확인한다.
원본 `figures/`, `ppt-fig/`, 공유 `outputs/`는 수정하지 않는다.
[manifest.json](manifest.json)은 사용한 입력과 산출물의 SHA-256을 기록한다.

추가 renderer: [make_claim_figure_pack.py](../experiments/paper/make_claim_figure_pack.py).
분석 계산과 원본 plotting 함수: [fig_why_differs.py](../experiments/paper/fig_why_differs.py).
원본 계산을 그대로 재사용하며 축 명칭·범례 길이·서체·색·실선 스타일만 조정했다.
Same-token MLP 패널은 같은 저장 통계에서 Q-head와 동일한 천장 보정·신뢰도 기준으로 추가했다.

## 기존 6개 figure 상세 설정

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
