# `sec2_rank_analysis.tex` 피드백 (2026-09-24)

## 0. 총평

수치는 전부 맞다(`2026-09-24_finding1_numbers.md`와 대조: 0.78 / 0.33, 0.75→0.34, 58 / 47 / 48 / 26, 96 / 34 / 53 / 13, 33 of 53,
0.72 / 0.75 / 0.86 / 0.36, NOTE의 MLP 값까지 12개 모두 일치; 각 밴드의 vision + text + ego-history = 1.00). 흐름(갈라짐 →
출처 → 같은 자리로 읽으면 복원)도 짧고 분명하다. 고칠 것은 셋: **메소드로 이어지는 마지막 문장이 없다**, "as do all results
below"가 MLP의 raw 위치 제한 일치도에는 과하다, ¶2의 text 몫 차이 문장이 ρ=0.78과 연결되지 않는다.

## 1. 반드시 고칠 것

**(a) 메소드로 가는 다리가 없다.** 섹션이 CoC 자리 예외로 끝난다. 예산 수준 결과 한 문장을 마지막에 두면 ρ가 프루닝 결과로
번역되고 "그래서 둘 다 쓴다"가 나온다(`importance_v2`, review_numbers.py (4); MLP는 34 vs 27%, 34 vs 26%):

> At the pruning budget of Sec.~X (13 of 32 heads per layer), ranking by the trajectory score alone removes 30\% of the
> CoC score's mass in layers 22--34, against 16\% when ranking by the CoC score itself (29\% vs.\ 17\% in the other
> direction; a random cut removes 41\%). A single score therefore discards late-layer heads that the other objective
> values, which is why Sec.~X scores with both.

**(b) "as do all results below (supplement)"는 완화.** MLP의 위치 제한 일치도는 raw로 0.34 → 0.47–0.52라 "agree again"이
아니다(NOTE에 적힌 대로 신뢰도 0.46–0.49 때문). 본문은 raw만 쓰기로 했으니 supplement에서 raw와 corrected를 나란히 두고
본문은:

> MLP channels show the same drop ($0.75\to0.34$) and the same position shares; their position-restricted agreement
> rises less in raw terms because channel scores are noisier (supplement).

**(c) ¶2 마지막 문장.** "This difference in text share is the main difference between the two scores in these layers"는
무엇의 차이인지 불명확하고 ρ=0.78과 연결되지 않는다. raw 0–21 값(vision 0.89, prompt 0.76, ego-history 0.96, CoC 0.42)으로
이으면 ¶4의 논리가 early에서도 성립함을 보인다(혼합 차이가 작으면 0.78 vs 0.89, 크면 0.33 vs 0.72):

> This difference in text share accounts for the residual disagreement in these layers: restricted to vision positions,
> the two rankings agree at $\rho=0.89$.

## 2. 표현·용어

- **"process language positions" / "read the scene"**: 몫은 head의 *출력이 놓인 자리*(그 출력이 loss에 들어가는 자리)이지
  head가 무엇을 읽는지가 아니다(vision 자리의 Q head 출력은 어느 자리든 attend 할 수 있다). 앞서 정한 "importance earned at
  positions" 해석과 맞추기: "the CoC score favors heads whose outputs at language positions matter to the loss, whereas the
  trajectory score also values their outputs at vision, ego-history and generated-CoC positions."
- **"part" 동사 3회**(문단 제목 포함) → 일부는 "diverge".
- **"different heads in the late layers"**: ρ=0.33은 약한 양의 상관. "largely different" 또는 "only weakly related".
- **밴드 용어 통일**: method_v2.md는 *trunk* / *late band*, 이 초안은 *early and middle* / *late*. 하나로(TODO에 추가).
- **35층 제외 각주**: "Layer 35 is excluded: the trajectory score is identically zero there because that layer's outputs never
  reach the expert's cache." 그림 (a)의 x축도 0–34.
- **기호 병기 한 번**: "the CoC score $I^{\mathrm{CoC}}$ ... the trajectory score $I^{\mathrm{traj}}$" → method의 기호와 연결.
- **"33 of 53 points"** → "33 of its 53 percentage points".
- **CoC 자리 예외는 깊이와 무관**: "(0.42 in layers 0--21 as well)" 한 마디면 late 특유 현상이 아님이 드러난다. CoC 자리 점수의
  split-half 신뢰도는 Q 0.79 / 0.87로 높아 잡음이 아니라는 점은 supplement.
- **held-out 재현 한 문장**(오늘 결과): "The same pattern holds on validation, OOD and test clips (supplement)." ¶1 끝 또는 각주.
- (선택) 왜 CoC 자리에서 갈라지는지는 본문에 안 넣어도 되지만 reviewer가 물을 수 있다. 각주 한 줄: 같은 자리의 출력이라도
  CE에는 다음 토큰 예측으로, FM에는 expert가 읽는 cache로 들어가므로 두 loss가 쓰는 정보가 다르다.

## 3. 그림 계획

- **(b), (c) Q head 몫 패널은 이미 있다**: `figures/supplementary/fig4_token_share_{traj,coc}_q_head.*` (논문 스타일, 오늘
  생성). 단 Q head 선은 층별 지터가 크다(클립 부트스트랩 95% CI 반폭 ≈ 0.11; 22층·27층 스파이크는 실제). 본문이 밴드 평균만
  인용하므로 밴드 평균 수평선을 겹치거나(analysis 스타일 `fig4_token_handoff5_q_head_bands` 방식) CI 밴드를 넣기를 권장.
  대안은 (b, c)를 밴드별 누적 막대(0–21 / 22–34 × 3 자리)로 바꾸는 것 — 지터가 사라지고 본문 숫자와 1:1이지만 21층 부근
  교대라는 시각 증거를 잃는다.
- **(a)** Q-only + 밴드 평균 수평선: `paper_rank_agreement.py`에서 MLP 선을 빼고 0–21 / 22–34 평균선을 얹으면 된다.
- **(d)** 막대 5개(0.33 / 0.72 / 0.75 / 0.86 / 0.36)는 새로 그려야 한다. 데이터는 `finding1_numbers.py`에 있다.
- **배치**: 2×2에서 (a, d) 위 / (b, c) 아래는 본문 순서(¶1 → ¶2–3 → ¶4)와 어긋난다. 한 줄 4패널이 본문 순서와 맞고, 2×2면
  (a, b) / (c, d).
- **캡션**: (d)에 "raw Spearman, mean over layers 22--34"; 음영 설명에 35층 제외 언급.
