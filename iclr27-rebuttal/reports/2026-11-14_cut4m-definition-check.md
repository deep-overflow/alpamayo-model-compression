# cut4m 항(4 m 규칙 손실)의 정의 확인: rebuttal 문장 (a)·(b) 점검

- 날짜: 2026-11-14 (KST)
- 대상:
  - 논문 Table 6의 cut4m 열(MoOP: Origin150 0.039, Hard100 0.067)
  - `iclr27-rebuttal/reports/2026-11-07_closedloop-gap-100scenes.md`
- 코드:
  - rollout 하나: `experiments/evaluation/analyze_closedloop_gap.py`
  - 세트 평균: `cvpr27/code/closedloop_gap_100.py`
  - progress 지표: alpasim `src/eval/src/eval/scorers/ground_truth.py`
- 재현: `.venv/bin/python iclr27-rebuttal/code/cut4m_check.py` (저장된 `outputs/closedloop_gap_v1/rollouts.parquet`만 읽음, GPU 불필요)

## 점검한 문장

> (a) For each such rollout, we approximate the score lost to the 4 m rule by the score that remained available along the reference route from the point where scoring stopped.
>
> (b) Even if all the score lost to the 4 m rule were credited back, MoOP would reach 0.867 on Origin150 and 0.662 on Hard100.

## 요약

- **(a)는 코드와 다르다.**
  - 코드는 "채점이 멈춘 지점에서 남아 있던 점수(1 − P)"를 세지 않는다.
  - 대신 "채점이 멈춘 뒤 차가 기준 경로를 따라 실제로 더 진행한 만큼"을 센다. 끝점은 rollout 끝 또는 첫 충돌·off-road 중 먼저 오는 쪽이다.
  - 그리고 alt-route rollout만이 아니라 4 m 규칙으로 멈춘 모든 rollout에 매긴다.
- **(b)의 숫자는 코드 값과 맞다.** 0.8283 + 0.0389 = 0.8672이고, 0.5955 + 0.0670 = 0.6625이다.
  - Hard100은 반올림하면 **0.663**이다. 0.662는 반올림한 값끼리(0.595 + 0.067) 더해서 나온 것이다.
  - 다만 이 값은 "4 m 규칙 손실을 전부 돌려준 점수"가 아니다(5절).

## 1. rollout 하나의 cut4m 계산

```python
cut4m = P(i_g) - P(i_ship)     # analyze_closedloop_gap.py:280  "cut4m": p_gate - p_ship
```

| 기호 | 정의 | 위치 |
|---|---|---|
| P(i) | `pscore`: min(clip(progress_rel_to_total[i], 0, 1) / 0.8, 1). 기록 경로가 5 m 미만이면 1 | 184–187행 |
| progress_rel_to_total | ego 중심을 기록된 기준 경로에 투영한 정규화 위치. 누적 최댓값이 아니라 매 시점의 위치라서 뒤로 갈 수 있다 | alpasim `ground_truth.py:55` (`full_gt_linestring.project(..., normalized=True)`) |
| i_g | 첫 `collision_any`(후방 충돌 포함) 또는 off-road 시점. 없으면 rollout 마지막 시점(20 s) | 189–192행 |
| i_4 | i_g까지 중 기준 경로에서 4 m 이상 벗어난 첫 시점 | 193–194행 |
| i_ship | i_4가 있으면 i_4, 없으면 i_g. 실제로 채점이 멈춘 시점 | 195행 |

- `p_gate`, `p_ship`은 200행에서 계산한다.
- 4 m 규칙으로 멈추지 않은 rollout은 i_ship = i_g이므로 cut4m = 0이다.

## 2. 어느 rollout에 매기나

- 4 m 규칙으로 멈춘 **모든** rollout에 같은 식을 쓴다. 분류(alt-route 등, 215–225행)는 계산식에 들어가지 않는다.
- route end와 ≥80% done은 이미 P(i_ship) = 1이라서 0이 된다.

MoOP(u40)에서 세트 평균 cut4m을 분류별 기여로 나누면 다음과 같다. 기여 = 그 분류의 cut4m 합 / 세트 rollout 수.

| 4 m 종료 분류 | Origin150 n | Origin150 기여 | Hard100 n | Hard100 기여 |
|---|---:|---:|---:|---:|
| alt-route | 17 | 0.0125 (32%) | 25 | 0.0233 (35%) |
| excursion (다시 4 m 안으로 복귀) | 8 | 0.0105 (27%) | 14 | 0.0248 (37%) |
| hidden failure (멈춘 뒤 off-road·과실 충돌) | 25 | 0.0158 (41%) | 24 | 0.0189 (28%) |
| stalled | 1 | 0.0000 | 0 | — |
| route end / ≥80% done | 113 / 12 | 0 | 26 / 6 | 0 |
| **4 m 규칙으로 멈춘 전체** | **176** | **0.0389** | **95** | **0.0670** |

따라서 cut4m에서 alt-route가 차지하는 몫은 3분의 1 정도다.

## 3. 평균 단위

- **rollout 평균이다.** Origin150은 300개, Hard100은 200개다(`closedloop_gap_100.py:127`, `a[c].mean()`).
- 모든 scene에 rollout이 정확히 2개다(117–118행 assert). 그래서 scene 평균과 소수점까지 같다(Origin150 0.0389 = 0.0389, Hard100 0.0670 = 0.0670).
- Hard100의 200개에는 중단된 rollout 8개가 들어 있다. 논문 관례대로 이들은 score 0이고 cut4m 0이다(`fill_aborted`, 43–52행).

## 4. off-road·충돌 gate가 걸린 rollout

- **채점 구간 안에서** 과실 충돌이나 off-road가 나서 점수가 0이 된 rollout은 cut4m = 0이다. 이는 구조상 그렇다.
  - 그 사건이 i_ship 이전이면 i_gate ≤ i_ship ≤ i_g = i_gate이므로 i_ship = i_g이고, 차이는 0이 된다(201·204행).
  - MoOP에서 이런 rollout은 Origin150 27개, Hard100 46개이고, 모두 cut4m이 0이다.
- 4 m 규칙으로 멈춘 **뒤에** off-road·충돌이 난 hidden failure는 점수가 0이 되지 않는다(score = P(i_ship)). 이 경우 그 사건 시점까지 더 진행한 만큼이 cut4m에 들어간다.
- 후방 충돌은 cut4m을 세는 끝점(i_g)은 되지만 점수를 0으로 만들지는 않는다. 0으로 만드는 것은 전방·측방 충돌과 off-road뿐이다(201행).

## 5. score + cut4m은 "전부 돌려준 점수"인가

**1을 넘는지:**
- rollout별 score + cut4m은 점수가 0이 되지 않은 rollout에서 P(i_g)이고, gate로 0점이 된 rollout에서 0이다.
- P는 187행에서 1로 잘리므로 rollout별 값은 1을 넘지 않는다. 관측 최댓값도 1.000이므로 **따로 자를 필요는 없다.**
- cut4m이 음수인 rollout이 Origin150 1개, Hard100 2개 있다. 모두 alt-route이고, 투영 위치가 뒤로 밀린 경우다. 평균에 주는 영향은 −0.0006 / −0.0009다.

**해석:** "4 m 규칙 손실을 돌려준다"는 말은 정의에 따라 값이 다르다.

| 정의 | Origin150 | Hard100 | 코드 |
|---|---:|---:|---|
| 실제 점수 (score) | 0.828 | 0.595 | |
| 4 m 규칙을 실제로 끈 점수: 뒤늦은 off-road·충돌이 0점이 된다 | 0.785 | 0.612 | `score_no4m`, 283행 |
| score + cut4m: 멈춘 뒤 실제로 더 간 만큼만 돌려주고, 그 뒤의 실패는 무시 | **0.867** | **0.663** | (b)의 값 |
| score + L_cut: 4 m 규칙으로 끝난 rollout의 남은 점수(1 − score)를 전부 돌려줌 | 0.893 | 0.710 | 282행, `L_cut` |

- (b)의 0.867은 hidden failure 몫(Origin150 0.016)을 포함한다. 그런데 이 몫은 4 m 규칙을 실제로 끄면 오히려 0점이 되는 rollout에서 나온다.
- 따라서 0.867은 "규칙을 끈 점수"도 아니고 "남은 점수를 전부 돌려준 상한"도 아니다. 그 중간이다.

## (a) 수정 문장

**A안: Table 6의 cut4m(0.039 / 0.067)을 그대로 쓴다.**

> For each rollout stopped by the 4 m rule, we approximate the score lost to the rule by the additional progress score the vehicle accrued along the reference route after scoring stopped, measured up to the end of the rollout or its first collision or off-road event, whichever comes first.

이 경우 (b)는 "all the score lost" 대신 다음처럼 쓰는 것이 정확하다. Hard100은 0.663이다.

> Even if the progress made after every 4 m stop were credited back, MoOP would reach 0.867 on Origin150 and 0.663 on Hard100.

**B안: (a)의 원래 의미("remained available")를 살린다.** 이 경우 숫자를 L_cut 기준으로 바꾼다.

> For each rollout stopped by the 4 m rule, we take the score lost to the rule to be all the score that remained available, i.e. one minus its score.

이 경우 (b)는 "MoOP would reach 0.893 on Origin150 and 0.710 on Hard100"이 된다.

## 판단 (의견)

- (b)가 "다 돌려줘도 이 정도"라는 상한 논증이라면 B안(L_cut)이 문장의 뜻과 정확히 맞다. 가장 관대한 가정이기 때문이다.
- A안을 쓰면 두 문장을 모두 위처럼 "progress made after the stop"으로 맞춰야 리뷰어가 Table 6과 대조했을 때 어긋나지 않는다.
- 어느 쪽이든 "For each **such** rollout"이 alt-route rollout을 가리키면 틀린 문장이 된다. cut4m의 3분의 2는 alt-route가 아닌 rollout(excursion, hidden failure)에서 나온다.
