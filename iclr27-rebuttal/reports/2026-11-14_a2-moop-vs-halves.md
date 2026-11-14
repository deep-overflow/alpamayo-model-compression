# Experiment: Alpamayo 2 Super — MoOP vs FM-only vs CE-only on driving and LingoQA (u40 · u50 · u50-q8)

2026-11-14 · 모델 `nvidia/Alpamayo2-Super`(rev `00554695`)

- **계획**
  - `iclr27-rebuttal/plans/2026-11-11_a2-u50-q8-gh200.md`: 11-13 변경, cvlab21 Ada에서 test500과 OOD-Val만 평가
  - `iclr27-rebuttal/plans/2026-11-13_a2-lingoqa.md`
- **사용자 목표(/goal):** "MoOP가 FM-only와 CE-only를 이길 때까지 해줘."
- **코드**
  - `iclr27-rebuttal/code/{make_a2_moop_masks.py --q-cut, run_a2_arms.sh SET, analyze_a2_test500.py --preset u50q8_*_ada}`
  - `run_a2_lingoqa.py`, `run_a2_lingoqa_queue.sh`, `analyze_a2_lingoqa.py`
  - `a2_coc_fidelity.py`, `a2_lingo_length_check.py`, `a2_importance_split_half.py`
- **결과 경로**
  - 주행: `iclr27-rebuttal/outputs/a2_{test500,oodval}_*_ada/`, `a2_u50q8_{test500,oodval}_ada_analysis/`
  - LingoQA: `a2_lingo_*/`, `a2_lingoqa_analysis/`

> **최종 정리 (2026-11-14 19:3x KST): 목표("MoOP가 FM-only와 CE-only를 이길 때까지")를 단일 지표 기준으로는 달성하지 못했다.**
> 시도한 모든 구성을 아래 표에 남긴다. 사전 등록한 검정은 표시했다.
>
> | 구성 | 주행 개루프 MoOP vs FM-only | 주행 개루프 MoOP vs CE-only | LingoQA MoOP vs FM-only | LingoQA MoOP vs CE-only |
> |---|---|---|---|---|
> | u40 | 비김 (3세트) | **이김** (3세트) | 비김 (val −2.4) | 비김 (val −2.6) |
> | u50 | **짐** (+0.109\* / +0.077\*) | **이김** | val +8.2\* → train 재현(사전 등록) +1.8 n.s. | val +4.4\* → train 재현(사전 등록) −1.5 n.s. |
> | u50-q8 | 비김 (−0.022 / −0.050) | OOD-Val **이김** (−0.348\*), test500 n.s. | **이김, 3회 재현**: val +6.2\*, train1 +4.3\*, train2(사전 등록 주 가설) +3.0\* (p_holm 0.002) | train1 +2.5\* → train2(사전 등록) −0.3 n.s. |
> | u55-q8 (사전 등록, 762클립 합산) | +0.039 n.s. (p_holm 0.20) | **이김** (−0.222\*) | — | — |
>
> - **한 지표에서 두 반쪽을 모두 이긴 구성은 없다.**
> - **u50-q8의 MoOP는 어느 반쪽에게도 유의하게 지지 않는다.** 언어에서 FM-only를 이기고(재현됨), 주행에서 CE-only를 이긴다. 즉 두 축을 함께 보면 두 반쪽을 지배한다.
> - **구조적 이유(의견).** A2에서는 각 반쪽이 자기 목표 축에서 무너지지 않는다.
>   - FM-only는 u55까지 주행을 지킨다.
>   - CE-only는 언어를 지킨다.
>   - 그래서 합집합이 그 축에서 반쪽을 넘어설 여지가 없다.
>   - A1.5에서는 FM-only가 u40에서 이미 CoC와 주행을 함께 잃었기 때문에 MoOP가 두 반쪽을 모두 이겼다.
> - **남은 축은 폐루프다.** A1.5에서 MoOP > FM-only > CE-only가 나온 축이고, A2 폐루프는 AlpaSim에 통합되어 있지 않다.
> - **상세 출력**
>   - `a2_u55q8_analysis/summary.txt`
>   - `a2_lingotrain_analysis/summary.txt`
>   - `a2_lingotrain2_analysis/summary.txt`
> - **계획서**
>   - `plans/2026-11-14_a2-u55-q8-blackwell.md`
>   - `plans/2026-11-13_a2-lingoqa.md`

> **정정 (2026-11-14 13:2x KST):** 아래의 u50 LingoQA 결과(MoOP가 두 반쪽을 모두 이김)는 사전 등록한 재현 실험에서 재현되지 않았다.
> - 재현 데이터는 LingoQA **train** 표본 2,000문항(세그먼트 400개)이다.
> - MoOP 44.7%, FM-only 43.0%(+1.8 pp [+0.0, +3.5], p 0.055, Holm 0.11), CE-only 46.2%(MoOP −1.5 pp [−3.3, +0.4], p 0.14).
> - 따라서 val의 u50 우위는 표본 변동으로 보며, **A2에서 MoOP가 언어로 두 반쪽을 이긴다는 근거는 없다.** 목표는 아직 달성되지 않았다.
> - 상세: `iclr27-rebuttal/outputs/a2_lingotrain_analysis_primary/summary.txt`, 사전 등록은 `plans/2026-11-13_a2-lingoqa.md`.
> - 다음 시도는 u55-q8 주행이다(`plans/2026-11-14_a2-u55-q8-blackwell.md`, 사전 등록).

**요약 (초판, 위 정정을 함께 읽을 것)**
- 초판에서는 목표 달성 지점을 u50(층마다 50%, A2의 42.6% 제거)의 LingoQA val로 판단했다. 이 판단은 위 재현 실험에서 무너졌다.
  - MoOP 70.8%가 FM-only 62.6%(+8.2 pp [+3.4, +13.0]\*, p = 0.0012)와 CE-only 66.4%(+4.4 pp [+0.4, +8.4]\*, p = 0.036)를 모두 유의하게 이긴다(dense 81.2%).
  - 문항 유형으로 나눠도 같다. FM-only에는 예/아니오 +8.5\*, 서술형 +7.9\*로 이긴다. CE-only에는 예/아니오 +8.9\*로 이기고 서술형은 같다(0.0).
  - 길이 편향으로는 설명되지 않는다. CE-only의 답이 더 긴데도 진다.
- **같은 u50의 주행은 다르다.** MoOP는 CE-only를 크게 이기지만, FM-only에는 진다.
  - test500 +0.109\*, OOD-Val +0.077\*
  - 그래서 u50 MoOP의 주장은 "언어에서 두 반쪽을 모두 이기고, 주행에서 CE-only를 이긴다"까지다.
- **Q head를 덜 자르는 배분(u50-q8: 층마다 Q head 8개 + MLP 14,848개, 같은 15.27B 제거)이 MoOP의 주행 손실을 없앤다.**
  - MoOP는 test500 −0.121\*, OOD-Val −0.177\* 좋아진다. FM-only는 거의 변하지 않는다.
  - q8에서 MoOP는 주행 평균이 셋 중 가장 낮다. 다만 FM-only와는 n.s.다(test500 −0.022, OOD-Val −0.050).
  - CE-only와는 OOD-Val −0.348\*, test500 −0.045 n.s.다.
  - LingoQA는 FM-only +6.2\*, CE-only +2.6 n.s.다.
- **u40(논문 레시피)에서는 언어 차이가 없다.** MoOP 73.4, FM-only 75.8, CE-only 76.0, 모두 n.s.
  - A2의 단일 기준은 u40에서 언어를 잃지 않는다. A1.5의 37.0% / 30.2%와 대조적이다.
- **합집합 효과는 단일 기준이 무너질 때 나타난다.** u50에서 FM-only와 CE-only의 LingoQA가 무너질 때(−18.6 / −14.8 pp vs dense) MoOP는 −10.4 pp에 그친다.

## Purpose

- **동기.** A2에서 MoOP는 주행 개루프에서 FM-only를 이긴 적이 없다(u40 비김, u50 짐).
- **목표.** 원칙 있는 변형으로 MoOP가 두 단일 기준을 모두 이기는 조건이 있는지 찾는다. 미리 정한 세트와 지표로 재고, 불리한 결과도 그대로 적는다.
- **가설**
  - H1: MoOP의 u50 주행 손실은 Q head에서 온다. q8 배분이 이를 줄인다.
  - H-L: 언어(LingoQA)에서 MoOP가 두 반쪽을 이긴다. A1.5 표 4는 68.8 / 37.0 / 30.2였다.

## Setup

| 항목 | 값 |
|---|---|
| 기준 | u40과 같은 중요도(`a2_importance_v1`, calib_100)와 규칙. MoOP = max(rank I_traj, rank I_CoC), FM-only = rank I_traj, CE-only = rank I_CoC |
| 배분 | u40: Q 26 / MLP 10,203(34.1%). u50: Q 32 / MLP 12,800(42.6%). u50-q8: Q 8 / MLP 14,848(42.6%, 파라미터 단위로 u50과 같음). q8은 make_slim `_qcut` 규칙이고, A1.5 `dual_u40_qcut4_v2`를 unit 단위로 재현한다(G0a) |
| 주행 | cvlab21 Ada 4–7(2장 1쌍 × 2). test500 + OOD-Val262. 롤아웃 1회 + k=8, minADE@6 / minFDE@6 1.6 / 3.2 / 6.4 s |
| 주행 재사용 | test500 dense와 u40 행은 11-09 Ada 결과다. G1 게이트에서 hook 없는 2클립이 11-09 dense와 비트 단위로 같았다. OOD-Val dense는 Ada에서 새로 쟀다 |
| LingoQA | Blackwell 0–3. 500문항, 100세그먼트. 전방 카메라 4프레임. A2 기본 VQA 프롬프트. greedy, 최대 128토큰. Lingo-Judge로 채점. CI는 segment 단위 부트스트랩 200k회(seed 0 / 1 동일) |
| 실행 | 주행 Ada 11-13 20:42 – 11-14 07:43, LingoQA Blackwell 11-14 05:36 – 06:45. 실패 0 |

## Results (Facts)

### LingoQA (Blackwell, 500문항)

| 배분 | dense | MoOP | FM-only | CE-only | FM-only − MoOP | CE-only − MoOP |
|---|---|---|---|---|---|---|
| u40 | 81.2 | 73.4 | 75.8 | 76.0 | +2.4 [−1.8, +6.6] | +2.6 [−1.0, +6.2] |
| **u50** | 81.2 | **70.8** | 62.6 | 66.4 | **−8.2 [−13.0, −3.4]\*** (p = 0.0012) | **−4.4 [−8.4, −0.4]\*** (p = 0.036) |
| u50-q8 | 81.2 | 73.4 | 67.2 | 70.8 | −6.2 [−10.6, −1.8]\* | −2.6 [−6.4, +1.0] |

- **답 길이(중앙값 단어 수):** dense 7. u40은 MoOP 5 / FM 3 / CE 6. u50은 6 / 2 / 7. q8은 3 / 1 / 4. 잘린 답은 0–1.2%다.
- **u50 층화(MoOP − X, pp)**

  | 층 | FM-only | CE-only |
  |---|---|---|
  | 예/아니오 247 | +8.5 [+2.1, +14.8]\* | +8.9 [+3.2, +14.5]\* |
  | 서술형 253 | +7.9 [+1.6, +14.2]\* | +0.0 [−5.9, +5.9] |
  | 답 길이 차 ≤ 2단어 | +3.6 [−1.8, +9.1] | +2.3 [−2.8, +7.3] |

  상세는 `a2_lingoqa_analysis/length_question_type.txt`.

### 주행 개루프 (Ada, minADE@6 6.4 s; 괄호는 같은 Ada dense 대비)

| 세트 | 배분 | MoOP | FM-only | CE-only | MoOP − FM-only | MoOP − CE-only |
|---|---|---|---|---|---|---|
| test500 (dense 0.921) | u40 | 1.029 (+11.8%) | 1.022 | 1.103 | +0.007 [−0.034, +0.050] | −0.074\* |
| | u50 | 1.156 (+25.5%) | 1.047 | 1.395 | **+0.109 [+0.060, +0.160]\*** | −0.239\* |
| | u50-q8 | **1.034** (+12.4%) | 1.056 | 1.079 | −0.022 [−0.070, +0.021] | −0.045 [−0.113, +0.021] |
| OOD-Val262 (dense 1.035) | u50 | 1.272 (+22.8%) | 1.194 | 1.757 | **+0.077 [+0.004, +0.149]\*** | −0.485\* |
| | u50-q8 | **1.094** (+5.7%) | 1.145 | 1.443 | −0.050 [−0.127, +0.021] | **−0.348\*** |

- **q8 − u50(같은 기준끼리)**
  - test500: MoOP −0.121\*, FM-only +0.010, CE-only −0.315\*
  - OOD-Val: MoOP −0.177\*, FM-only −0.050, CE-only −0.314\*
- **CoC 퇴화(test500 / OOD-Val)**

  | 배분 | MoOP | FM-only | CE-only |
  |---|---|---|---|
  | u50 | 0.6 / 0.8% | 0.2 / 1.1% | 5.0 / 5.3% |
  | q8 | 0.6 / 0.8% | 1.2 / 5.0% | 5.0 / 12.6% |

- 1.6 / 3.2 s와 minFDE@6은 `a2_u50q8_{test500,oodval}_ada_analysis/summary.txt`에 있다.

### CoC 보존(같은 seed에서 dense의 CoC와 완전히 일치하는 비율; 탐색적, 결과를 본 뒤 고른 지표)

| 조건 | MoOP / FM-only / CE-only | MoOP − FM-only | MoOP − CE-only |
|---|---|---|---|
| test500 u40 | 24.2 / 15.6 / 27.8% | +8.6 pp\* | −3.6 pp (n.s.) |
| test500 u50 (Blackwell) | 17.0 / 6.8 / 15.2% | +10.2 pp\* | +1.8 pp (n.s.; 단어 유사도 +0.065\*) |
| OOD-Val u40 (GH200) | 13.7 / 8.4 / 14.1% | +5.3 pp\* | −0.4 pp |
| OOD-train u40 (GH200) | 15.6 / 9.0 / 15.7% | +6.6 pp\* | −0.1 pp |

### 중요도 split-half 신뢰도(calibration 100클립을 반으로 나눠 층 안 순위의 Spearman ρ; 20회 평균)

| | FM Q | CE Q | FM MLP | CE MLP |
|---|---|---|---|---|
| A2 | 0.889 | 0.919 | 0.615 | 0.569 |
| A1.5 | 0.838 | 0.897 | 0.606 | 0.689 |

## Analysis (Opinion)

**1. A2에서 합집합의 이점은 단일 기준이 무너지는 지점에서만 보인다.**
- u40에서 A2의 FM-only와 CE-only는 언어(LingoQA 75.8 / 76.0)도 주행도 크게 잃지 않는다. 그래서 MoOP가 지킬 것이 없다.
- u50에서 두 반쪽의 LingoQA가 각각 18.6 / 14.8 pp 떨어질 때 MoOP는 10.4 pp만 잃는다. 이것이 A1.5 표 4와 같은 구도다.
- A1.5에서는 이 붕괴가 u40에서 이미 일어났다. 더 큰 A2는 같은 구도가 더 높은 압축률에서 나타난다.

**2. 주행에서는 같은 u50이 반대로 간다. 이는 MoOP의 Q head 선택 때문으로 보인다.**
- 같은 예산에서 Q head를 8개만 자르면 MoOP만 크게 좋아지고 FM-only는 거의 그대로다. 그러면 u50의 MoOP − FM-only 격차(+0.109\* / +0.077\*)가 사라진다(−0.022 / −0.050).
- CE 쪽 순위는 잡음이 아니다(split-half ρ 0.919로 FM보다 높다). CoC를 위해 지킨 head가 주행에는 쓸모없고, 그 대가로 궤적에 필요한 head가 잘린다고 보는 편이 맞다.
- 이것은 마스크 비교와 배분 실험에서 나온 추론이다. head만 바꿔 끼우는 실험은 하지 않았다.

**3. 한 배분으로 세 축을 다 이기는 지점은 아직 없다.**

| 배분 | 언어 | 주행 |
|---|---|---|
| u50 | 둘 다 이김 | FM-only에 짐 |
| u50-q8 | FM-only를 이김 | CE-only를 이김(OOD-Val) |
| u40 | 비김 | CE-only를 이김 |

- MoOP가 어느 반쪽에게든 유의하게 지는 경우는 u50 주행의 FM-only뿐이고, q8에서는 사라진다.
- q8 MoOP는 여섯 개 비교 모두에서 이기거나 비긴다. 지는 비교는 없다.
  - LingoQA FM-only +6.2\*, CE-only +2.6
  - OOD-Val CE-only −0.348\*, FM-only −0.050
  - test500 두 비교 모두 n.s.
- 그래서 실용적으로 가장 균형 잡힌 레시피는 u50-q8 MoOP다.

**4. 주의할 점**
- **다중비교.** 배분 3개 × 반쪽 2개를 봤다. u50 LingoQA의 FM-only 차이(p = 0.0012)는 보정 뒤에도 남는다. CE-only 차이(p = 0.036)는 Bonferroni ×3에서 0.11이 되어 경계선이다.
- **개루프만이다.** A1.5에서 Q head를 덜 자르는 배분은 개루프에서 비슷하거나 낫고 폐루프에서 나빴다(−0.091\*). A2 폐루프는 미측정이다.
- **LingoQA의 한계.** OOD(Wayve)이고 전방 카메라만 쓴다. Lingo-Judge의 길이 편향도 있다. 층화로 길이를 통제하면 FM-only 격차가 +3.6(n.s.)으로 줄어든다. 다만 예/아니오 문항에서는 그대로 유의하다.
- **하드웨어.** 주행(Ada)과 LingoQA(Blackwell)는 카드가 다르다. 각 표 안에서는 한 종류다.
- **calibration.** 추출과 중요도 측정이 각각 한 번이다.

**다음 후보**
- u50과 q8의 LingoQA 결과를 OOD-train 주행(GH200)으로 확장한다.
- head 교체 실험: MoOP u50의 Q head만 FM-only 것으로 바꾼다. H1 기전을 직접 검증한다.
- A2 폐루프 통합 여부는 사용자 결정이다.
