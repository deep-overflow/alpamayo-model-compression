# Finding 1 리뷰에 대한 답변

2026-09-24. 대상: `paper-analysis/analysis_v2.md`의 Finding 1(순위 불일치 → 자리별 분해 → 같은 자리에서는 일치)과
그 그림(`fig1_rank_agreement`, `fig4_token_share_{traj,coc}_mlp`). 아래 수치는 모두 저장된 run에서 계산했다
(`outputs/importance_v2`, `outputs/gradanat_v1`, `outputs/gradanat_probes_v1`; 계산 스크립트 `scripts/review_numbers.py`).
새 GPU 실험은 없다.

## 요약

| # | 지적 | 판정 | 본문 수정 |
|---|---|---|---|
| 1 | "decomposes exactly"가 정말 exact인가 | **반박** — 부호 보존 가법 분해라 unit·층 단위로 정확히 합쳐진다 | 정의 수식 한 문장과 상쇄 지수 추가 |
| 2 | 후반 결과는 구조적으로 당연한 것 아닌가 | **부분 동의** — 방향은 구조가 정하고, 위치(22층)와 크기·trunk 일치는 경험적. 무작위 readout probe가 이를 분리해 준다 | "구조 → 가설 → 검증" 구조로 재서술, 35층 제외 명시 |
| 3 | 같은 자리 일치는 activation 항의 공유 때문일 수 있다 | **동의** — 실제로 그렇다. gradient 코사인 ≈ 0, 무작위 readout 두 개가 trunk 0.79 / late 0.35를 재현 | 문장 하나 + probe baseline 수치 추가; "value the same units"를 점수 언어로 |
| 4 | Spearman 불일치 → 제거 집합 차이는 별개 | **동의** — 출하 예산에서 교차 objective 질량 손실을 계산했다: 후반에서 단일 기준은 다른 objective의 질량을 자기 기준의 약 2배 제거 | 수치 추가, dual-axis 평가 전제 명시 |
| 5 | 대역이 cherry-pick처럼 보이고 "increasingly with depth"가 안 맞는다 | **동의** — 0–6 / 7–21 / 22–34 세 체제이며 MLP는 7층과 22층 두 계단 | 대역 규약 하나로 통일, 0–6 체제 한 문장, "increasingly" 삭제 |
| 6 | 신뢰도·CI·"calibration noise" | **동의** — 부트스트랩 CI와 split-half 신뢰도를 보고하면 된다 | 표기 정리, 대역 평균 CI, 추출 강건성 한 문장 |
| 7 | 그림–본문 불일치(4-way vs 3-way, 토큰 수, 34층, 라벨) | **동의** | text 몫의 prompt/CoC 분할 수치, per-token 언급, 34층 edge 주석, 라벨 정리 |
| 8 | Finding 상자가 길다 | **동의** | 제안된 형태로 축약 |

---

## 1. 분해는 exact다 — 단, 정의를 써야 한다

**점수.** unit $u$, 클립 $c$에 대해 gate gradient는 자리별 합이다:
$G_{c,u}=\sum_p a_{c,p,u}\,g_{c,p,u}$ ($a$ = unit 출력, $g$ = 그 출력에 대한 loss gradient).
출하 점수는 **abs를 클립 안에서, 평균을 밖에서** 취한다: $I_u=\mathbb{E}_c\,|G_{c,u}|$.

**분해.** 타입 $t$의 부분합 $G_{c,t,u}=\sum_{p\in t}a_{c,p,u}\,g_{c,p,u}$는 **부호를 보존**한 채 두고, 기여를
$C_{t,u}=\mathbb{E}_c\big[G_{c,t,u}\cdot\operatorname{sign}(G_{c,u})\big]$
로 정의한다. 그러면 $\sum_t C_{t,u}=\mathbb{E}_c[G_{c,u}\operatorname{sign}(G_{c,u})]=\mathbb{E}_c|G_{c,u}|=I_u$가
항등식으로 성립한다. 층 몫은 $\sum_u C_{t,u}/\sum_u I_u$이고 타입 합은 정확히 1이다. 리뷰어가 의심한
$\sum_p|a_pg_p|$ 형태가 **아니다** — 그림의 몫이 [0,1]인 것은 부호 보존 기여가 이 데이터에서 음수가 되지 않았기
때문이지(최솟값 sink −0.000), 절댓값을 씌워서가 아니다.

**수치 검증.** 타입별 gradient를 합쳐 출하 점수와 비교하면 최대 상대오차 6.0e-06(클립 3개 × 역전파 13회).
**상쇄 지수**(리뷰어가 요구한 $\sum_t\mathbb{E}|G_t|\,/\,\mathbb{E}|G_{\text{full}}|$, 타입 수준): 27–34층에서
CE 1.21 / FM 1.37 (Q), 1.22 / 1.45 (MLP). 즉 타입 간 부호 상쇄는 있으나 1.2–1.5배 수준이다. 자리(position) 단위의
$\sum_p|c_p|$는 저장하지 않아 보고할 수 없다(타입 단위 합만 저장).

**수정.** Finding 1에 정의 한 문장을 넣는다:
> "A unit's score is $I_u=\mathbb{E}_c|\sum_p a_{c,p,u}\,g_{c,p,u}|$; we split the inner sum by position type and credit each part with the sign of the clip's full gradient, $C_{t,u}=\mathbb{E}_c[G_{c,t,u}\,\mathrm{sign}(G_{c,u})]$, so that $\sum_t C_{t,u}=I_u$ exactly (signed shares; the cross-type cancellation index $\sum_t\mathbb{E}|G_{t}|/\mathbb{E}|G|$ is 1.2–1.5 in the late band)."

## 2. 구조적 예측과 경험적 부분을 나눠 쓴다

리뷰어의 구조 논증은 맞다: CoC loss는 LM head가 CoC 자리의 최종 residual만 읽으므로, 층 $l$의 vision 자리 출력이
CoC loss에 닿는 경로는 "$>l$ 층의 text query가 그 K/V를 읽는 것"뿐이고, expert는 모든 자리의 per-layer KV를 읽는다.
이것이 **방향**을 정한다. 우리는 이것을 `gradanat_probes_v1`에서 직접 확인했다 — loss 내용을 없앤 **무작위 readout**을
같은 두 문(LM head / expert)으로 흘리면:

| 비교 | 값 |
|---|---|
| expert-door probe의 깊이 프로파일 vs `I_traj` | Pearson 0.980 (Q) / 0.964 (MLP) |
| head-door probe vs `I_CoC` | 0.871 / 0.936 |
| expert/head probe 비율의 변화점 | 23층, 계단 5.4× (실제 `I_traj/I_CoC`: 23층, 4.6×) |
| probe의 vision 몫 6–17 → 27–34층 | head 0.41 → 0.05, expert 0.60 → 0.38 (실제 CoC 0.53 → 0.05, traj 0.61 → 0.33; Q) |
| Q head 순위: head probe vs `I_CoC` / expert probe vs `I_traj` (천장 보정) | 0.965 / 0.990 |
| **두 probe끼리** (objective 없음) trunk / late | **0.786 / 0.353** (실제 두 objective: 0.85 / 0.37) |

즉 후반 불일치와 자리 hand-over는 두 read-out 인터페이스의 성질이다. 그래서 리뷰어 제안대로
"구조에서 도출되는 가설 → 검증"으로 쓰는 것이 더 강하다. 다만 **구조가 정하지 않는 것** 세 가지는 남겨야 한다:
- **떨어지는 위치.** 구조는 끝점(35층에서 vision 몫 0)만 정한다. 실제 하락은 17–24층에 몰려 있고(CoC vision 몫 22층 0.20, 25층 0.04 — 아직 10개 층이 남았을 때), 이 위치는 expert의 포트 집중(cache 21–23)에서 온다.
- **`I_traj`도 후반에서 text 자리로 옮겨 간다**(Q 53%, MLP 66%). "expert는 모든 자리를 읽는다"만으로는 예측되지 않고, expert가 CoC 자리의 cache entry에 크게 의존하기 때문이다(포트 맵: CoC 토큰 16개가 FM 질량의 29–32%, 토큰당 prompt의 12–15배).
- **trunk에서 둘 다 vision 자리에서 번다**는 것도 구조가 강제하지 않는다.

**35층.** `I_traj`는 35층에서 항등적으로 0이다(35층의 출력은 최종 hidden → LM head로만 가고, expert가 읽는 cache 35는
34층 출력에서 계산됨). 그래서 순위 상관은 22–34층, pruning 대역은 22–35층이다. 본문에 한 절로 명시한다.

**수정.** 문단 3 앞에 두 문장: *"The two outputs read the backbone through different interfaces — the LM head reads the final residual at CoC positions only, the action expert reads every layer's K/V at every position — so late-layer CoC importance must concentrate at text positions while trajectory importance need not. We test whether this shows up as the rank disagreement, and where."* 끝에: *"Layer 35 is excluded from the correlations because its outputs never reach the cache the expert reads (I_traj ≡ 0 there)."*

## 3. activation 교락은 실재하고, 이미 잰 baseline이 있다

동의한다. 점수는 $\sum_p a_p g_p$이고 같은 자리로 제한하면 $a_p$를 공유한다. 우리 데이터가 그 우려를 **확인**한다:
- 두 loss의 residual-stream gradient 코사인은 모든 깊이·모든 타입에서 ≈ 0 (+0.019 ~ −0.001). 즉 $g$는 직교한다.
- 무작위 방향 readout(위 표): 각 문의 probe가 실제 objective의 순위를 0.965 / 0.990으로 재현하고, **두 probe끼리의 일치도가 trunk 0.786 / late 0.353**으로 실제 두 objective(0.85 / 0.37)와 거의 같다. → trunk의 ≈0.8 일치는 "어떤 loss든 두 문으로 흘리면 나오는 값", 즉 공유 activation·인터페이스 항이다.
- 텍스트 자리 write norm과의 순위 상관: `I_CoC` 0.50 → 0.77 (late), `I_traj` 0.24 → 0.25. 0–5층 MLP는 순위가 곧 write norm 순서(ρ 0.88–0.97)라 두 objective가 0.93–0.98로 일치한다(5번 항목의 0–6 체제).

**결론에 미치는 영향은 없다.** Finding 1의 주장은 "두 *점수*가 후반에서 다른 unit을 고른다"이고 pruning은 점수를
쓴다. 바꿔야 할 것은 해석 문장이다 — "at any one kind of position the two objectives value nearly the same units"는
"…the two *scores* agree, largely through the activation factor they share"로.

**수정.** 문단 4에 한 문장: *"This same-position agreement is largely carried by the activation factor the two scores share: the two losses' residual-stream gradients are nearly orthogonal at every depth (cosine ≤ 0.02), and two content-free random readouts through the same two interfaces agree at 0.79 in the trunk and 0.35 in the late band — almost exactly the real objectives' 0.85 and 0.37."*

## 4. 예산 수준의 차이 — 교차 objective 질량 손실

출하 예산(층당 head 13/32, 채널 4,898/12,288 제거)에서 각 objective의 bottom-k를 제거했을 때:

| | 제거집합 Jaccard (traj vs coc) | `I_CoC` 질량 손실: 자기 순위 / **traj 순위로** | `I_traj` 질량 손실: 자기 순위 / **coc 순위로** | 무작위 |
|---|---|---|---|---|
| Q 0–6 | 0.56 | 0.16 / 0.21 | 0.13 / 0.16 | 0.41 |
| Q 7–21 | 0.58 | 0.20 / 0.23 | 0.15 / 0.19 | 0.41 |
| **Q 22–34** | **0.37** | 0.16 / **0.30** | 0.17 / **0.29** | 0.41 |
| MLP 0–6 | 0.83 | 0.07 / 0.07 | 0.06 / 0.07 | 0.40 |
| MLP 7–21 | 0.52 | 0.26 / 0.29 | 0.22 / 0.27 | 0.40 |
| **MLP 22–34** | **0.36** | 0.27 / **0.34** | 0.26 / **0.34** | 0.40 |

(무작위 40% 부분집합 둘의 기대 Jaccard는 0.25.) 후반에서 한 objective로 자르면 다른 objective의 질량을 자기 기준의
**약 두 배**(Q 0.16 → 0.30, 0.17 → 0.29) 제거하고, MLP는 무작위(0.40)와 거의 다르지 않다(0.34). trunk에서는 3–4 pp
차이뿐이다. 이것이 "discard units the other objective values"의 정량화이며, 불일치가 중간 rank가 아니라 bottom tail에도
있음을 보인다. **dual-axis 전제**: CoC가 value하는 unit을 지켜야 한다는 것은 추론 품질이 평가 축일 때 성립한다는
절을 넣는다("reasoning soundness ≠ driving safety" 결과와 일관되게). 참고로 주행만 볼 때도 합집합이 trunk에서 필요하다는
제거 실험(traj-only trunk FM +26.1% vs union +4.8%)이 있으나 현재 본문 밖이다(`removed_finding2_band_ablation.md`).

**수정.** 문단 4 끝에: *"At the shipped budget this is a difference in what gets pruned, not only in rank order: in layers 22–34 the units the trajectory score discards carry 30% (Q heads) and 34% (MLP) of the CoC score's mass, against 16% and 27% for the CoC score's own selection, and the two removed sets overlap at Jaccard 0.37 / 0.36 (0.58 / 0.62 in the trunk; 0.25 for random sets)."* + 전제 절: *"…which matters whenever reasoning quality is an evaluation target alongside driving."*

## 5. 대역: 세 체제, 두 계단

한 규약(0–6 / 7–21 / 22–34)으로 다시 낸 값:

| | 0–6 | 7–21 | 22–34 |
|---|---|---|---|
| raw ρ, Q / MLP | 0.78 / 0.93 | 0.78 / 0.67 | 0.33 / 0.34 |
| 보정 ρ (pooled), Q / MLP | 0.83 / 0.98 | 0.86 / 0.90 | 0.38 / 0.54 |
| `I_traj` vision / text / hist (MLP) | 0.57 / 0.18 / 0.25 | 0.83 / 0.14 / 0.03 | 0.28 / 0.66 / 0.07 |
| `I_CoC` vision / text / hist (MLP) | 0.46 / 0.47 / 0.07 | 0.65 / 0.34 / 0.01 | 0.04 / 0.96 / 0.00 |
| `I_traj` (Q) | 0.37 / 0.31 / 0.32 | 0.68 / 0.24 / 0.08 | 0.34 / 0.53 / 0.13 |
| `I_CoC` (Q) | 0.28 / 0.62 / 0.09 | 0.56 / 0.41 / 0.03 | 0.04 / 0.96 / 0.00 |

- 리뷰어의 관찰이 맞다: MLP는 **7층과 22층 두 계단**(0.93 → 0.67 → 0.34)이고 7층의 하락은 vision 몫 급등(0.57 → 0.83)과 겹친다. Q head는 계단이 22층 하나다(0.78 / 0.78 / 0.33).
- 0–6층은 별개 체제다: 1–5층 MLP는 중요도가 소수 채널에 몰려 있고(Gini 0.88) 순위가 write norm 순서(ρ 0.88–0.97)라 두 objective가 거의 같은 unit을 고르며, `I_traj`는 ego-history 자리 몫이 0.25–0.32로 높다.
- "increasingly so with depth"는 삭제한다.

**수정.** 모든 수치를 이 세 대역으로 통일하고, 0–6층을 한 문장으로 설명한다: *"Layers 0–6 form a separate regime in which importance sits in a few high-norm channels and both scores rank them alike (ρ 0.93 for MLP channels); the disagreement appears with the vision-dominated regime from layer 7 (ρ 0.67) and widens from layer 22 (0.34)."* 6–17 / 27–34 인용은 모두 7–21 / 22–34로 바꾼다.

## 6. 통계 보고

- **층별 ρ의 잡음.** Q head(층당 32개)의 층별 ρ를 클립 부트스트랩(B=300)하면 95% CI 반폭 중앙값 0.11–0.12. 후반의 층 간 진동(|Δρ| 중앙값 0.24)은 CI보다 크므로 **전부 잡음은 아니다** — 예: 26층 0.64 [0.52, 0.71], 28층 −0.06 [−0.13, 0.10], 33층 −0.11 [−0.22, 0.06]. 다만 개별 층을 해석하지 않고 대역 평균만 쓴다: 22–34층 0.335 [0.285, 0.387], 7–21층 0.784 [0.672, 0.828], 0–6층 0.779 [0.679, 0.817]. 그림에는 대역 평균선 또는 CI 띠를 넣는다.
- **신뢰도.** split-half 자기 일치도(traj / coc), 22–34층: pooled Q 0.81 / 0.89, MLP 0.46 / 0.49; vision Q 0.84 / 0.90; prompt Q 0.83 / 0.90; **CoC 자리 Q 0.79 / 0.87**(낮지 않다 — 0.41은 신뢰도 문제가 아님), MLP CoC 0.45 / 0.66. 보정값과 함께 표(부록)로 싣는다.
- **"0.38 → 0.79"의 뜻.** 둘 다 천장 보정된 22–34층 값이고, pooled 점수 → vision 자리로 제한한 점수다. MLP도 넣는다: 0.54 → 0.72 (vision) / 0.74 (prompt) / 0.83 (ego-history) / 0.37 (CoC).
- **"calibration noise" → "sampling noise of the 100-clip estimate".** 추출 도메인 강건성은 따로 있다: 서로소 100클립 추출 3개(rd100_a/b/c)를 별도 run으로 재면 보정 일치도가 같은 값(Q 0.85 → 0.37, MLP 0.90 → 0.53)이고, 변화점 23층은 12개 run(서로소 추출, n=100–4,000, OOD, self-anchored target) 전부에서 재현된다. 한 문장으로 넣는다.

## 7. 그림–본문 정리

- **text 몫의 분할.** 22–34층 `I_traj`의 text 몫 = prompt + CoC: MLP 0.34 + 0.31, Q 0.20 + 0.33. CoC 자리가 절반 이상이며, 이는 traj가 생성된 CoC에 conditioning된다는 리뷰어 추측과 맞다. 본문에 괄호로 넣고 4-way 패널(`fig4_token_handoff5_*`)은 보충 자료로.
- **토큰 수 정규화.** 토큰 하나당 중요도(`fig4_pertoken_*`): 22–34층 `I_traj`에서 ego-history 토큰 하나가 vision 토큰 하나의 23배(MLP) / 29배(Q), CoC 토큰은 229배 / 220배. 캡션 또는 본문에 한 문장으로 넣고 토큰 수(2,880 / 157 / ~16 / 48)를 병기한다.
- **34층 vision 반등**(MLP 0.32 → 0.52, Q 0.39 → 0.65). 34층의 출력은 expert가 읽는 cache 중 마지막(35)에만 닿고 35층은 경로가 없으므로 34–35층은 edge다. 캡션에 "layers 34–35 are edge layers (see text)"로 표시한다.
- **라벨.** `fig:vlm_rank_correlation` → 파일 `fig1_rank_agreement`에 맞춰 `fig:rank_agreement`, `fig:token_handoff` → `fig:token_share`로 통일.

## 8. Finding 상자

리뷰어 제안을 세 대역 규약으로 맞춘 안:
> **Finding 1.** In the late layers the CoC and trajectory objectives rank units differently (ρ ≈ 0.33 in layers 22–34 against 0.67–0.78 in layers 7–21). The gap follows from where each output reads the backbone: CoC importance is earned almost only at text positions, whereas trajectory importance stays spread over vision, text and ego-history positions.

---

## 반영 순서

1. 문단 3에 정의 수식과 상쇄 지수 (1번).
2. 문단 2·3을 "인터페이스 → 가설 → 검증"으로 재배열하고 35층 제외 명시 (2번).
3. 문단 4에 activation 문장 + probe 수치 (3번), 예산 수준 수치 + dual-axis 전제 (4번).
4. 모든 수치를 0–6 / 7–21 / 22–34로 통일, 0–6 문장, "increasingly" 삭제 (5번).
5. 신뢰도 표(부록), 대역 평균 CI, 추출 강건성 문장, 용어 교체 (6번).
6. 캡션·라벨·34층 주석·per-token 문장 (7번), 상자 축약 (8번).

## 출처

- 1: `outputs/gradanat_v1/summary.txt` G0 (a), P1 cancellation index; 정의는 `experiments/head_analysis/prune_lib.py` `TypedUnitGates`.
- 2·3: `outputs/gradanat_v1/summary.txt` residual-gradient cosine; `outputs/gradanat_probes_v1/summary.txt` (D1–D4); `outputs/portmap_v1/summary.txt`.
- 4·5·6: `scripts/review_numbers.py` (importance_v2 + importance_perclip + gradanat_v1 metrics_analysis).
- 7: `figures/supplementary/fig4_token_handoff5_*`, `fig4_pertoken_*`.
