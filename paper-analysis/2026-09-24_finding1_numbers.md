# Finding 1 개정용 수치 (2026-09-24)

요청받은 수치 전부. 출처는 `outputs/importance_v2/importance.npz`(Fig. 1의 원천, Blackwell, calib_100)와
`outputs/gradanat_v1/`(같은 100클립의 gradient anatomy, Ada). 계산: `paper-analysis/scripts/finding1_numbers.py`
(원출력은 맨 아래 부록). **몫의 계산 규칙은 모두 동일**: 층마다 가법 분해 몫(5개 타입 합 = 1)을 구한 뒤
구간의 층들에 대해 평균(= 기존 22–34 값과 review_numbers.py가 쓴 규칙). text = prompt + 생성 CoC.
`I_traj`는 35층에서 0이라 22–34 구간은 두 점수 모두 22–34층(13개 층)으로 통일.

## 1. ¶4 — 새로 계산: 0–21층 몫

| 대상 | 점수 | vision | text | (prompt + CoC) | ego-history |
|---|---|---:|---:|---:|---:|
| **MLP** | `I_traj` | **0.74** | **0.15** | 0.09 + 0.07 | 0.10 |
| **MLP** | `I_CoC` | **0.59** | **0.38** | 0.19 + 0.19 | 0.03 |
| Q head (괄호용) | `I_traj` | 0.58 | 0.26 | 0.16 + 0.10 | 0.16 |
| Q head (괄호용) | `I_CoC` | 0.47 | 0.48 | 0.24 + 0.23 | 0.05 |

**주의 — 0–21에는 0–5층이 들어간다.** 0–6층은 두 점수 모두 자리가 섞여 있어(MLP `I_traj` vision 0.57 / text 0.18 /
hist 0.25, `I_CoC` vision 0.46 / text 0.47) 0–21 평균이 지금 본문의 6–17 값(0.90 / 0.75)보다 꽤 낮다.
"trunk에서는 둘 다 주로 vision 자리에서 번다"를 0–21 값으로 쓰면 `I_CoC`는 0.59 대 0.38이라 "주로"가 약해진다.
같은 규칙의 7–21 값은 MLP `I_traj` vision 0.83 / text 0.14 / hist 0.03, `I_CoC` vision 0.65 / text 0.34 / hist 0.01
(Q: 0.68 / 0.24 / 0.08, 0.56 / 0.41 / 0.03). ¶2가 0–21을 쓰므로 구간을 맞추는 쪽(0–21)이 일관되고,
그 경우 문장은 "mainly at vision positions (74% / 59%)" 정도로 두고 `I_CoC`의 text 38%를 trunk 불일치의
출처로 쓰는 편이 자연스럽다.

## 2. ¶2 — 확인: 순위 일치도 (raw, 보정 전)

| 출처 | 축 | 0–21 | 22–34 |
|---|---|---:|---:|
| importance_v2 (Fig. 1 원천) | Q head | 0.782 | 0.335 |
| importance_v2 (Fig. 1 원천) | MLP | 0.753 | 0.339 |
| gradanat_v1 typed sum (같은 클립, Ada) | Q head | 0.776 | 0.334 |
| gradanat_v1 typed sum (같은 클립, Ada) | MLP | 0.752 | 0.341 |

- **0–21: 0.78 = Q head, 0.75 = MLP** (본문의 "0.75–0.78"은 이 둘).
- **22–34: 0.33 (Q) / 0.34 (MLP)는 raw가 맞다.** 100클립 평균 점수 두 벡터의 층내 Spearman을 층 평균한 값,
  천장 보정 없음. 보정값은 Q 0.38, MLP 0.54(4절 표 pooled 행).

## 3. ¶5 — 확인: 22–34층 몫 (1절과 같은 규칙으로 재계산, 전부 일치)

| 대상 | 점수 | vision | text | (prompt + CoC) | ego-history |
|---|---|---:|---:|---:|---:|
| MLP | `I_traj` | 0.28 | 0.66 | 0.34 + 0.31 | 0.07 |
| MLP | `I_CoC` | 0.04 | 0.96 | 0.69 + 0.26 | 0.00 |
| Q head | `I_traj` | 0.34 | 0.53 | 0.20 + 0.33 | 0.13 |
| Q head | `I_CoC` | 0.04 | 0.96 | 0.40 + 0.56 | 0.00 |

본문의 28 / 66 / 7%, 4 / 96%, Q 34 / 53 / 13%, text 분할 34 + 31 / 20 + 33이 그대로 나온다. ¶4와 같은 규칙.

## 4. (b) 선택 시 — 같은 자리로 제한한 순위 일치도, raw 값

raw = 타입별 |gradient|의 100클립 평균 두 벡터의 층내 Spearman, 층 평균(¶2의 0.33과 같은 규칙, 보정 없음).
corrected = 기존 split-half 천장 보정값(재계산, 50/10회 분할이라 저장값과 ±0.01 몬테카를로 차이).

| 축 | 자리 | raw 0–21 | **raw 22–34** | corrected 22–34 |
|---|---|---:|---:|---:|
| Q head | pooled (출하 점수) | 0.78 | **0.33** | 0.38 |
| Q head | vision | 0.89 | **0.72** | 0.79 |
| Q head | prompt-text | 0.76 | **0.75** | 0.81 |
| Q head | ego-history | 0.96 | **0.86** | 0.94 |
| Q head | 생성 CoC | 0.42 | **0.36** | 0.41 |
| MLP | pooled (출하 점수) | 0.75 | **0.34** | 0.54 |
| MLP | vision | 0.81 | **0.47** | 0.71 |
| MLP | prompt-text | 0.68 | **0.49** | 0.74 |
| MLP | ego-history | 0.81 | **0.52** | 0.83 |
| MLP | 생성 CoC | 0.63 | **0.24** | 0.38 |

- **Q head는 raw로도 그대로 성립**: 0.33 → vision 0.72 / prompt 0.75 / ego-history 0.86, CoC 자리만 0.36.
  (b)를 고르면 본문 문장은 "0.33 → 0.72 at vision, 0.75 at prompt-text and 0.86 at ego-history positions
  for Q heads, except at the generated-CoC positions (0.36)".
- **MLP는 raw로는 약하다**: 0.34 → 0.47 / 0.49 / 0.52. MLP 점수의 split-half 신뢰도가 낮아(pooled 0.46–0.49)
  raw 값이 천장에 눌린다. 지금 본문처럼 이 문장을 Q head 기준으로 두면 (a)/(b) 어느 쪽이든 문제 없고,
  MLP까지 raw로 적으면 "same position → agree" 주장이 MLP에서는 약하게 읽힌다.
- CoC 자리의 불일치는 모든 깊이에서 raw로도 유지(Q 0.42 / 0.36).

## 5. 참고 — 규칙 2(층 합산 후 비율)로 계산하면

층별 몫의 평균 대신 구간 층들의 가법 부분을 합산해 비율을 내면(질량 가중) 모든 셀이 규칙 1과 0.06 이내로
같다(예: MLP `I_traj` 0–21 vision 0.77 vs 0.74, 22–34 text 0.67 vs 0.66; MLP `I_CoC` 0–21 text 0.40 vs 0.38).
결론이 바뀌는 셀은 없다. 기존 값이 모두 규칙 1이므로 규칙 1로 통일.

## 재현

```bash
cd /home/cvlab21/project/chan/alpamayo-model-compression/paper-analysis/scripts
../../.venv/bin/python finding1_numbers.py
```

## 부록 — 스크립트 원출력

## A. Token shares by band (rule 1 = mean over the band's layers of the per-layer additive share; the rule behind every existing 22-34 number)

| axis | score | band | vision | text (= prompt + CoC) | prompt | CoC | ego-history |
|---|---|---|---:|---:|---:|---:|---:|
| mlp | I_traj | 0-21 | 0.74 | 0.15 | 0.09 | 0.07 | 0.10 |
| mlp | I_traj | 22-34 | 0.28 | 0.66 | 0.34 | 0.31 | 0.07 |
| mlp | I_CoC | 0-21 | 0.59 | 0.38 | 0.19 | 0.19 | 0.03 |
| mlp | I_CoC | 22-34 | 0.04 | 0.96 | 0.69 | 0.26 | 0.00 |
| q | I_traj | 0-21 | 0.58 | 0.26 | 0.16 | 0.10 | 0.16 |
| q | I_traj | 22-34 | 0.34 | 0.53 | 0.20 | 0.33 | 0.13 |
| q | I_CoC | 0-21 | 0.47 | 0.48 | 0.24 | 0.23 | 0.05 |
| q | I_CoC | 22-34 | 0.04 | 0.96 | 0.40 | 0.56 | 0.00 |

### Rule 2, reference only: pooled mass (sum of the typed additive parts over the band's layers / sum of the layer totals)

| axis | score | band | vision | text | prompt | CoC | ego-history |
|---|---|---|---:|---:|---:|---:|---:|
| mlp | I_traj | 0-21 | 0.77 | 0.18 | 0.09 | 0.08 | 0.05 |
| mlp | I_traj | 22-34 | 0.26 | 0.67 | 0.35 | 0.32 | 0.06 |
| mlp | I_CoC | 0-21 | 0.58 | 0.40 | 0.20 | 0.21 | 0.02 |
| mlp | I_CoC | 22-34 | 0.03 | 0.97 | 0.73 | 0.24 | 0.00 |
| q | I_traj | 0-21 | 0.62 | 0.28 | 0.17 | 0.11 | 0.10 |
| q | I_traj | 22-34 | 0.31 | 0.59 | 0.21 | 0.38 | 0.10 |
| q | I_CoC | 0-21 | 0.50 | 0.46 | 0.24 | 0.22 | 0.04 |
| q | I_CoC | 22-34 | 0.05 | 0.95 | 0.42 | 0.53 | 0.00 |

## B. Rank agreement (raw within-layer Spearman between the two 100-clip mean scores, band mean)

| source | axis | 0-21 | 22-34 |
|---|---|---:|---:|
| importance_v2 (Fig. 1 source) | q | 0.782 | 0.335 |
| importance_v2 (Fig. 1 source) | mlp | 0.753 | 0.339 |
| gradanat_v1 typed sum (same clips, Ada) | q | 0.776 | 0.334 |
| gradanat_v1 typed sum (same clips, Ada) | mlp | 0.752 | 0.341 |

## D. Same-position agreement, 22-34 (and 0-21): raw = Spearman between the 100-clip means of E|G_type| (uncorrected, same rule as B); corrected = split-half ceiling-corrected (analyzer rule, layers with either self-reliability <= 0.2 dropped)

| axis | positions | raw 0-21 | raw 22-34 | corrected 22-34 (layers used) | half-sample cross 22-34 |
|---|---|---:|---:|---:|---:|
| q | pooled (shipped score) | 0.78 | 0.33 | 0.38 (13/13) | 0.33 |
| q | vision | 0.89 | 0.72 | 0.79 (13/13) | 0.69 |
| q | prompt | 0.76 | 0.75 | 0.81 (13/13) | 0.70 |
| q | CoC | 0.42 | 0.36 | 0.41 (13/13) | 0.34 |
| q | hist | 0.96 | 0.86 | 0.94 (13/13) | 0.82 |
| mlp | pooled (shipped score) | 0.75 | 0.34 | 0.54 (13/13) | 0.25 |
| mlp | vision | 0.81 | 0.47 | 0.71 (13/13) | 0.38 |
| mlp | prompt | 0.68 | 0.49 | 0.74 (13/13) | 0.34 |
| mlp | CoC | 0.63 | 0.24 | 0.38 (13/13) | 0.21 |
| mlp | hist | 0.81 | 0.52 | 0.83 (13/13) | 0.40 |
