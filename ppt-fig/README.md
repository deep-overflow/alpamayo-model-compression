# 발표용 표와 그림

15분 발표 구성과 해석은 [미팅 가이드](../paper/2026-09-15_dual-expert-mlp-meeting.md)에 정리했다.

| 파일 | 용도 |
|---|---|
| [table_01_open_loop.png](table_01_open_loop.png) | 본문: Test500·OOD-val262 각각의 minADE@6 / minFDE@6 |
| [table_01_closed_loop.png](table_01_closed_loop.png) | 본문: scene score / collision@fault / offroad / progress |
| [table_01_lingoqa.png](table_01_lingoqa.png) | 본문: LingoQA accuracy |
| [table_01_main_results.png](table_01_main_results.png) | 보충: 위 세 평가의 모든 지표를 포함한 통합 표 |
| [table_02_expert_axis.png](table_02_expert_axis.png) | 본문: expert Q-head와 MLP의 같은 비율·파라미터 수 비교 |
| [table_03_expert_open_loop.png](table_03_expert_open_loop.png) | 본문: expert MLP 추가 압축, Test500·OOD-val262 minADE / minFDE, 절대값 (Dual 대비 차이) |
| [table_03_expert_closed_loop.png](table_03_expert_closed_loop.png) | 본문: expert MLP 추가 압축, scene score·충돌·offroad·progress, 절대값 (Dual 대비 차이) |
| [table_03_dual_expert_mlp.png](table_03_dual_expert_mlp.png) | 보충: 위 expert MLP open-loop·closed-loop 통합 표. 기존 파일을 갱신 |
| [table_04_main_all_splits.png](table_04_main_all_splits.png) | 보충: 통합 표에 val500 minADE@6 / minFDE@6도 포함 |
| [fig_05_calibration_closedloop.png](fig_05_calibration_closedloop.png) | 본문: calibration 선택에 따른 closed-loop 결과와 95% CI |
| [table_06_hard100_closed_loop.png](table_06_hard100_closed_loop.png) | Hard100: Dense·LLM-Pruner·Týr·Týr-K·Dual, 기존 보고서의 점수·사건 비율 집계 유지 |
| [table_06_hard100_common_valid.png](table_06_hard100_common_valid.png) | Hard100 보충: Dense·LLM-Pruner·Týr·Dual의 공통 96 scenes × 2 rollouts 비교 |

분석 figure 8개도 이 폴더에 PNG·PDF·SVG로 모았다.
[분석 figure 목록과 발표 문장](analysis_figures.md)에서 각 그림의 역할과 해석을 확인할 수 있다.

Closed-loop 난이도 분석 figure 4개도 추가했다.
[난이도 분석과 발표 문장](difficulty_figures.md)에 static difficulty의 정확한 정의,
763 scenes에서의 검증, Easy100/Hard100 선별, 방법별 난이도 분석을 정리했다.
이미지는 `fig_07_static_difficulty_correlation`, `fig_08_static_difficulty_validation`,
`fig_09_static_difficulty_metrics`, `fig_10_pruning_by_static_difficulty`의 PNG·PDF·SVG다.
아래 별도 생성 명령을 사용하며 원본 결과를 읽기만 한다.

```bash
MPLCONFIGDIR=/tmp/alpamayo-mpl .venv/bin/python experiments/paper/make_difficulty_figures.py
```

새 난이도 그림은 PNG 400 dpi이며, 집계값과 원본 SHA-256은
`difficulty_analysis_data.json`에, scene별 수치는 `difficulty_*csv`에 저장한다.

동일 이름의 PDF·SVG도 제공한다. 표에는 CSV도 있다.
PNG 본문 표 폭은 3,999px, 주 비교 통합 표는 6,000px, expert MLP 통합 표는 6,300px,
전체 split 통합 표는 7,050px, calibration 그래프는 2,850px이며 모두 300 dpi다.
기존 분석 figure는 2,100 × 1,680px, 600 dpi다.
DejaVu Sans를 사용하며 PDF는 font 포함, SVG는 glyph path로 저장해 font 대체를 방지한다.

```bash
MPLCONFIGDIR=/tmp/alpamayo-mpl .venv/bin/python experiments/paper/make_meeting_tables.py
```

생성 코드는 공유 `outputs/`와 기존 LLM-Pruner·Alpasim 결과를 읽기만 한다.
`meeting_table_data.json`에는 집계 수치와 source file SHA-256을 저장한다.
Open-loop clip ID, closed-loop scene ID의 집합 일치, 중복 및 유한 점수를 검증한다.
Open-loop는 각 clip에서 첫 6개 샘플 중 ADE·FDE의 최소값을 각각 구한 뒤 clip 평균한다.
Main 및 expert MLP의 collision@fault와 offroad는 전체 300 rollouts 중 발생 비율(%)이며,
progress는 `progress_clipped_rel`의 전체 rollout 평균(ratio)이다.
세 평가 표 모두 Dense, Wanda, LLM-Pruner, Týr, Traj, CoC, Dual 순서로 표시한다.
`table_01_open_loop`와 `table_01_closed_loop`에는 마지막에 `Dual + MLP93.75`도 추가했다.
이 행은 plain Dual VLM에 expert MLP 93.75% 제거를 적용한 모델이며,
layer당 516 channels와 모든 expert Q heads를 유지한다. 전체 모델 제거율은 39.4%다.

`table_03_*` expert MLP 표의 괄호는 **plain Dual 대비 부호 있는 차이**이며,
상대 변화율(%)이나 CI가 아니다.
minADE·minFDE는 m, scene score·progress는 원래 단위의 차이, 충돌·offroad는 percentage points(pp)다.
Dual 행도 절대값을 표시하고 `(ref.)`로 비교 기준임을 나타낸다.
기존 paired 95% CI는 JSON의 `composition`에 보존하며 minFDE의 CI도 추가했다.
Clip bootstrap은 5,000회, scene bootstrap은 10,000회, seed 0이다.

Hard100에는 **경로 검사 실패로 세부 지표가 누락된 rollout**이 있다.
Dual은 8/200, 나머지 네 방법은 각각 4/200이다.
`table_06_hard100_closed_loop`의 scene score는 실패 시 저장된 0점을 포함한 100-scene 평균이다.
충돌·offroad는 기존 보고서와 같은 `기록된 사건 수 / 200`으로, 미측정 rollout을 무사고라고
확인한 비율은 아니다. Progress는 측정값만 평균해 Dual 192개, 나머지는 196개를 사용한다.
분모와 누락 수를 이미지 하단에도 표시했다.
`table_06_hard100_common_valid`는 어느 방법에서든 누락된 4 scenes를 모두 제외하여,
모든 열을 동일한 96 scenes × 2 rollouts에서 계산한다. 두 표의 모집단을 구분해서 사용한다.
Hard100은 주 비교 150 scenes와 서로소임을 suite CSV와 scene ID로 검증했다.
Wanda·Traj·CoC·Dual + expert MLP의 완료된 Hard100 결과는 확인되지 않아 행을 만들지 않았다.

LLM-Pruner는 CoC `param_first`인 `lp_r50`이며 약 25% 제거다.
다른 주 비교 arm의 약 24%와 정확히 같은 예산이라고 해석하지 않는다.
사용자 요청으로 모든 성능 표의 LLM-Pruner Removed 표시를 24.0으로 통일했다.
Open-loop·closed-loop·LingoQA·통합 표·전체 split·Hard100 표의 PNG·PDF·SVG·CSV에 적용한다.
원본 메타데이터와 성능 측정값은 유지한다.
Týr는 searched allocation + reconstruction인 `slim_tyr_u40_r`로 통일했다.
Dual 강조색은 발표 대상 표시이며, 통계적 유의성을 나타내는 표시는 아니다.
