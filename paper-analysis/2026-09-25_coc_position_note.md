# CoC 토큰 자리의 순위 불일치 — 원인·인과 정리와 논문 문장 (2026-09-25, 판 3: Run A·B 반영)

상세(영문, 수치·게이트 전부): `2026-09-25_coc_position_disagreement.md` (Test 1–4). 계획·결과 원본: 브랜치 `worktree-dual-saves-trunk`의
`plans/2026-09-25_coc-position-disagreement.md`, `plans/2026-09-25_coc-position-content.md`. 출력 `outputs/coc_posanat_v1/` (Run A),
`outputs/coc_posabl_v1/` (Run B), `outputs/coc_census_v1/`, `outputs/coc_pathsplit_v1/`.
스크립트 `experiments/head_analysis/{run_coc_position_anatomy,analyze_coc_position_anatomy,make_coc_position_masks,run_coc_position_ablation,analyze_coc_position_ablation,run_coc_census,analyze_coc_census,run_coc_path_split,analyze_coc_path_split}.py`.
그림(논문 스타일, `figures/supplementary/`): **`fig9_coc_position_split.{png,pdf,svg}`** (CoC 안의 토큰 가중 + 같은 구간 일치도),
**`fig10_coc_position_dissociation.{png,pdf,svg}`** (이중 해리), `fig8b_coc_attention_bars.*` (attention census) —
`scripts/paper_coc_position_split.py`, `paper_coc_position_dissociation.py`, `paper_coc_attention_bars.py`.

**표기 규칙.** 순위 일치도는 본문과 같이 **raw** Spearman(100클립 평균 점수, 층내, 밴드 평균)을 기준으로 쓰고 split-half 천장 보정값은
괄호에 병기한다. 해리 실험의 p값은 사전 등록한 **단측** 페어드 Wilcoxon, census의 p값은 head 단위 Mann–Whitney **양측**이다.

## 한 줄 결론

CoC 자리에서 두 점수가 갈리는 것은 **같은 토큰에서 다른 내용을 원해서가 아니라 서로 다른 토큰을 읽어서**다. `I_CoC`는 추론의 **단어**에서
(CE 손실이 거기서 평가됨), `I_traj`는 추론을 닫고 행동 구간을 여는 두 경계 토큰 **`<|cot_end|>`·`<|traj_future_start|>`**에서
(expert가 추론을 거기서 읽음) 점수를 번다. 단어로만 제한하면 일치도가 다른 자리 수준(0.78–0.82 raw / 0.90–0.92 보정)으로 돌아오고,
두 독자는 인과적으로 분리된다(0–21층: `I_traj` 쪽 head를 CoC 자리에서 끄면 행동만, `I_CoC` 쪽 head를 끄면 텍스트만 망가짐).
판 2의 본문 문장 *"even at the same tokens, the two scores favor different units"*는 **틀렸으므로 교체**해야 한다.

## 질문

같은 자리로 제한한 22–34층 순위 일치도(raw)는 vision 0.72, prompt-text 0.75, ego-history 0.86인데 생성 CoC 자리만 0.36이고
(보정값 0.79 / 0.81 / 0.94 / 0.41), 0–21층에서도 CoC 자리는 0.42(보정 0.47)로 낮다. 왜 이 자리만 갈리고, 거기서 각 점수가 고르는 head는
무엇을 하는가.

## 사실

### 배제된 원인 (calib_100 anatomy 저장 데이터 + 경로 분해, 판 2)

잡음 아님(보정해도 0.41 / 0.47), 모달리티 아님(prompt-text도 텍스트인데 0.75), 클립 혼합 아님(가중 재조정 0.42 → 0.44–0.46, 클립 하나 안에서도
0.23), gradient 방향 아님(토큰별 코사인 ≈ 0), **손실 경로 아님**(own-token 경로가 `I_CoC`@CoC의 87%를 차지하지만 두 경로가 head를 같은 순서로
세우고 — raw 0.88 / 0.86 — `I_traj`@CoC와의 일치도는 전체 0.42 / 0.36, own-token 0.41 / 0.34, cross 0.44 / 0.42로 차이 없음).

### 원인: CoC 안의 토큰 가중 혼합 (Run A, `run_coc_position_anatomy.py`, calib_100 중 99클립 — 124토큰 CoC 하나는 버퍼 초과로 제외)

CoC 자리마다 typed gate를 하나씩 두어(5 + n_coc 타입) CE backward 한 번, full-seed FM backward 한 번으로 **Q head별·CoC 위치별** 부호 있는
기여를 얻었다. 위치 합은 anatomy의 CoC 행을 재현한다(층별 Spearman 최소 1.000 / 0.999, A0 통과). 생성 CoC는 99클립 전부 `<|cot_end|>`
`<|traj_future_start|>`로 끝나고(길이 8–47, 중앙값 16 → 두 토큰이 14%), 하위 구간은 head clause(앞 세 단어), 나머지 문장, 두 경계 토큰.

**A1 토큰 가중** — CoC 자리 점수(|Q-head 기여|)의 구간별 몫, 층·클립 평균, 0–21 / 22–34 (괄호는 토큰 몫):

| 점수 | head clause (0.23) | 나머지 문장 (0.63) | `<\|cot_end\|>` (0.07) | `<\|traj_future_start\|>` (0.07) |
|---|---:|---:|---:|---:|
| `I_CoC` | 0.27 / 0.22 | 0.73 / 0.78 | 0.00 / 0.00 | 0.00 / 0.00 |
| `I_traj` | 0.12 / 0.11 | 0.25 / 0.24 | 0.32 / 0.21 | 0.31 / 0.43 |

- CE의 경계 토큰 몫이 0인 것은 **구조적**이다: `<|traj_future_start|>` 자리의 상태는 아무것도 예측하지 않고(손실이 거기서 끝남), `<|cot_end|>`
  자리의 상태는 결정적인 `<|traj_future_start|>`를 예측한다.
- FM은 정확히 거기에 몰린다: 층별 63% / 64%, 밴드 층을 합치면 71% / 75%(클립 중앙값 0.76, 클립의 94–95%가 절반 초과); 토큰 하나당 경계
  토큰은 단어의 19배(중앙값), residual gradient norm은 15배(7–21층, IQR 9.8–19.1). residual gradient로 봐도 같다(끝에서 두 위치가 FM의
  0.32 + 0.35, 그 앞은 위치당 ≤ 0.03; CE는 끝 두 위치 0.00).
- 사전 예측은 반만 맞았다: FM이 끝에 몰리는 것은 맞지만 **경계 토큰**이지 head clause가 아니고, head clause는 오히려 CE보다 덜 본다(0.12 vs 0.27).

**A2 같은 구간 일치도** (Q head, raw / 보정, 0–21 | 22–34):

| 제한 | 0–21 | 22–34 |
|---|---:|---:|
| CoC 전체 (출하 행) | 0.42 / 0.46 | 0.34 / 0.39 |
| head clause | **0.91 / 0.98** | **0.85 / 0.95** |
| 나머지 문장 | 0.74 / 0.89 | 0.84 / 0.92 |
| `<\|cot_end\|>` | 0.76 / 0.95 | 0.55 / 0.84 (CE 자기신뢰도 0.72 / 0.56) |
| `<\|traj_future_start\|>` | 정의 불가 (CE = 0) | 정의 불가 |
| **단어만** (head + 나머지) | **0.78 / 0.92** | **0.82 / 0.90** |
| `<\|traj_future_start\|>`만 제외 | 0.50 / 0.55 | 0.63 / 0.68 |

사전 등록 판정: **토큰 혼합**(최고 구간 보정 0.95 ≥ 0.70). 단어만의 일치도(late 0.82 raw / 0.90 보정)는 vision(0.72 / 0.79)·prompt(0.75 / 0.81)·
ego-history(0.86 / 0.94) 자리 수준이고 meta-action head clause는 그 전부보다 높다. `<|traj_future_start|>` 하나만 빼면 부족하다(0.63 / 0.68):
`<|cot_end|>`도 FM의 21–32%를 들고 있기 때문. MLP는 3구간만 저장돼 `<|cot_end|>`가 "나머지"에 섞여 있다(head clause 0.77 / 0.96 | 0.57 / 0.78,
나머지 0.58 / 0.73 | 0.33 / 0.43) — 4구간 재실행이면 정리된다.

**A3 단일 위치** (클립 하나·층 하나·위치 하나에서 head 간 Spearman, 평균): 0.56 | 0.59. 같은 클립에서 CoC 위치를 전부 합치면 0.23 | 0.24, 한
구간 안에서만 합치면 head clause 0.53 | 0.57, 나머지 0.44 | 0.47. 클립 하나 안에서도 **CoC 전체를 합치는 것이 일치도를 반으로 줄인다**(단일
클립의 천장은 모르므로 절대값은 하한, 집계 수준 간 비교가 요점).

### 인과: 이중 해리 (Run B, `run_coc_position_ablation.py`, held-out `indist_500` 앞 100클립, K = 8)

층마다 rank(`I_traj`@CoC) − rank(`I_CoC`@CoC) 상위 8개(T, `I_traj` 쪽)·하위 8개(C, `I_CoC` 쪽)·무작위 8개 ×3(R)을 **CoC 자리에서만** 끈다
(typed gate = 0; 다른 자리는 그대로). 같은 rollout 텍스트·노이즈·seed로 unmasked 모델과 페어링. dense NLL 0.1708, FM 0.2425, minADE 0.837.

| 집합 (층당 8 head, CoC 자리에서만 off) | dNLL | dFM | dminADE |
|---|---|---|---|
| **T, 0–21층** | +0.0028 [+0.0002, +0.0060], +1.7% (중앙값 +0.0001) | **+0.0285 [+0.0170, +0.0401], +11.7%** (+0.0187) | **+0.195 m [+0.049, +0.349]**, +23% (+0.068) |
| **C, 0–21층** | **+0.0282 [+0.0184, +0.0392], +16.5%** (+0.0140) | −0.0004 [−0.0017, +0.0007], −0.2% (0.0000) | −0.021 [−0.046, +0.000] (−0.001) |
| R ×3, 0–21층 | +2.6 … +5.8% | −0.9 … +1.3% | +0.014 [−0.028, +0.064] |
| T, 0–34층 | +0.4221 [+0.3490, +0.4972], **+247%** (+0.351) | +0.0284, +11.7% (+0.0183) | +0.210 [+0.052, +0.369] |
| C, 0–34층 | +0.1003 [+0.0768, +0.1285], +58.7% (+0.049) | −0.0006, −0.3% | −0.018 [−0.041, +0.001] |
| R ×3, 0–34층 | +10.4 … +16.7% | −1.3 … −0.1% | −0.014 [−0.101, +0.051] |

- **0–21층: D1 통과**(C > T p = 9.1e−10, C > 무작위 셋 다 최대 p = 8.9e−07), **D2 통과**(T > C p = 3.3e−08, T > 무작위 최대 p = 5.1e−08),
  D3 minADE T > C p = 0.0013. 교차 이중 해리다 — `I_traj` 쪽 head는 행동만(FM +11.7%, minADE +0.20 m = dense 오차의 1/4, decel_stop에서
  +37–39%로 최대), `I_CoC` 쪽 head는 텍스트만(NLL +16.5%, 무작위의 3–6배). 각 점수는 **자기 목적함수에 대해서만** 옳다.
- **0–34층: D2 통과, D1 실패** — C가 못해서가 아니라(C > 무작위 p = 8.6e−13) **T가 텍스트를 더 망가뜨려서**다. late T head를 더해도 행동은
  그대로(+11.7% → +11.7%: expert는 그 출력을 읽지 않음, 22–34층의 FM 무감각과 일치)인데 텍스트는 +1.7% → **+247%**, C(+59%)의 4배·무작위의
  15–24배. `I_traj`가 CoC 자리에서 높게 세우고 `I_CoC`는 낮게 세운 late head가 텍스트에 필수라는 뜻이다 — 1차 점수는 작은 스케일링을
  기술하는데 head를 모든 추론 단어에서 끄는 것은 작지 않고, 포화된 기여(로짓 마진 큼, gradient 작음)가 바로 점수가 과소평가하는 곳. dual-saves
  결과(`I_traj`를 통해 dual이 남긴 unit이 언어 unit)의 위치 제한판이며 union이 그것들을 지켜야 하는 또 하나의 이유. 기전은 미정(census상
  late `I_traj` 쪽 head가 prompt를 약간 더 읽음 0.154 vs 0.136; sink형 상수 출력 제거가 RMSNorm 스케일을 바꾸는 것도 후보).
- `I_CoC` 쪽 집합은 어디서도 행동을 건드리지 않는다(dFM −0.2 / −0.3%, minADE −0.02 n.s.): 텍스트 이어 쓰기 내용은 expert가 읽는 것이 아니다.

### 독자의 정체 (CoC 질의 attention census, 판 2, 100클립)

`I_CoC` 쪽 head는 **이전 CoC 토큰**을 읽고(0–21층 0.187 vs 0.066, p = 5e−17; 22–34층 0.087 vs 0.039, p = 6e−5), `I_traj` 쪽 head는
**ego-history**를 읽는다(0.047 vs 0.019, p = 9e−10; 0.030 vs 0.011, p = 7e−9); vision은 같다(p = 0.68 / 0.88; late는 모든 head가 CoC
질의 attention의 약 60%를 장면에 둠). 연속 지표(전체 head, 층내 Spearman): 이전-CoC 질량 ↔ `I_CoC`@CoC +0.45 / +0.48, ego-history 질량 ↔
`I_traj`@CoC +0.22 / +0.38, vision 질량 ↔ 순위 차이 +0.01 / −0.01. 사전 등록 게이트 G-B(prompt + 이전 CoC)는 0–21층 통과(p = 2e−4),
G-A(vision + ego-history)는 vision이 갈라지지 않아 실패 — ego-history 결과는 사후(post hoc)로 표기.

## 해석

추론 토큰의 표현을 읽는 독자는 둘이고 **읽는 자리가 다르다**. 뒤따르는 CoC 토큰(텍스트 이어 쓰기)은 단어 하나하나에서 읽고, expert는 추론을
닫고 행동 구간을 여는 두 경계 토큰 — prefix의 마지막 두 자리, diffusion 토큰 바로 옆 — 에서 요약처럼 읽는다. 출하 CoC 행은 단어 가중 점수와
경계 가중 점수를 더한 것이고 그것이 예외의 전부다: **토큰을 맞추면 두 목적함수는 어디서나 같은 head를 세운다**, 추론 토큰에서도. 그 위에서
census는 독자의 정체를(경계 토큰에 쓰는 head는 ego-history를, 단어에 쓰는 head는 이전 추론을 읽음), 해리 실험은 필요성을(0–21층에서 각 집합이
자기 목적함수만 담당) 준다. late 층에서는 `I_traj`가 세운 head가 텍스트에 필수라는 점이 union의 근거를 하나 더 보탠다.

## 논문 문장 (English)

**본문 — sec2 ¶4의 CoC 예외 문장 자리 (판 2의 교체안을 대체):**

```latex
The exception is the generated-CoC positions ($0.36$; $0.42$ in layers 0--21),
and it is a token mixture one level down rather than a disagreement about the
same tokens. The CE loss is evaluated at the words of the reasoning, whereas
the FM loss reaches the reasoning where the expert reads it: through the two
tokens that close it and open the action segment, $\langle$cot\_end$\rangle$
and $\langle$traj\_start$\rangle$, which carry $63$--$64\%$ of $I_{\mathrm{FM}}$
at CoC positions and, by construction, none of $I_{\mathrm{CE}}$. Restricted to
the words, the two scores agree at $0.78$--$0.82$ ($0.90$--$0.92$ after ceiling
correction), the level of the other position types, and at the meta-action
head clause at $0.85$--$0.91$ ($0.95$--$0.98$). The two readers are separately
necessary: switching the heads $I_{\mathrm{FM}}$ favors there off at the CoC
positions of layers 0--21 raises the FM loss by $11.7\%$ and minADE by
$0.20$\,m and leaves the text unchanged ($+1.7\%$), while switching off the
heads $I_{\mathrm{CE}}$ favors raises the NLL by $16.5\%$ and leaves the
action unchanged ($-0.2\%$; random sets $\le 5.8\%$ and $\le 1.3\%$,
Appendix~\ref{app:coc_positions}).
```

**Appendix 단락 (raw 기준, 보정값 괄호; `app:coc_positions`, 판 2 단락을 대체):**

```latex
\paragraph{Why the two scores disagree only at generated-CoC positions.}
With both scores restricted to one kind of position, the within-layer rank
agreement of Q heads in layers 22--34 is $0.72$ at vision, $0.75$ at
prompt-text and $0.86$ at ego-history positions but $0.36$ at generated-CoC
positions ($0.79$ / $0.81$ / $0.94$ / $0.41$ after split-half ceiling
correction), and $0.42$ ($0.47$) in layers 0--21. The exception is not sampling
noise (split-half reliabilities $0.79$ and $0.87$), not modality (prompt text
agrees), not a clip-mixture effect (reweighting either score's calibration
clips to the other's moves it from $0.42$ to at most $0.46$, and it is present
within single clips, $0.23$ against $0.46$ at vision positions), and not the
loss path: detaching the keys and values of every CoC position during the CE
backward splits $I_{\mathrm{CE}}$ at these positions exactly into an own-token
part ($87\%$) and a cross-position part, and the two rank the heads alike
($\rho = 0.88$ / $0.86$) and agree with the FM score no better than the whole
($0.41$ / $0.34$ and $0.44$ / $0.42$). It is a token mixture inside the CoC. With
one typed gate per CoC position (99 calibration clips; the summed gates
reproduce the pooled row, per-layer $\rho \ge 0.999$), the CE score is earned
entirely at the words of the reasoning ($27$--$22\%$ at the three-word head
clause, the rest of the sentence otherwise), while $63$--$64\%$ of the FM score
at CoC positions falls on the two special tokens that end every generated
reasoning, $\langle$cot\_end$\rangle$ and $\langle$traj\_start$\rangle$ ($14\%$
of the tokens; $71$--$75\%$ when a band's layers are pooled, above one half in
$94$--$95\%$ of clips, $19\times$ the per-token mass of a word). The CE gradient
is zero there by construction, since the last state predicts nothing and the
one before it predicts a deterministic token. Restricted to the words, the two
scores agree at $0.78$ in layers 0--21 and $0.82$ in 22--34 ($0.92$ / $0.90$
corrected), and at the head clause at $0.91$ / $0.85$ ($0.98$ / $0.95$); the
same holds within single clips, where the agreement at one CoC position is
$0.56$--$0.59$ and falls to $0.23$--$0.24$ once a clip's positions are summed.
The two readers are separately necessary. Switching the eight most
FM-favored heads per layer off at the CoC positions only (100 held-out clips,
paired with the unmasked model on the same text, noise and seeds) raises the
FM loss by $11.7\%$ [$7.0$, $16.5$] and minADE by $0.20$\,m [$0.05$, $0.35$] in
layers 0--21 and leaves the CoC NLL at $+1.7\%$, while the eight most
CE-favored heads raise the NLL by $16.5\%$ [$10.8$, $23.0$] and leave the FM loss
at $-0.2\%$ (three random sets: NLL $+2.6$ to $+5.8\%$, FM $-0.9$ to $+1.3\%$;
one-sided paired Wilcoxon $p \le 8.9\times10^{-7}$ for every pre-registered
comparison). Extending the sets to layers 22--34 changes the action by nothing
($+11.7\%$ in both cases, as the expert does not read those outputs) but raises
the NLL to $+247\%$ for the FM-favored heads against $+59\%$ for the CE-favored
ones ($+10$ to $+17\%$ random): the late heads the FM score ranks high at the
reasoning tokens are ones the text cannot do without, which the first-order CE
score, a description of small scalings, does not flag. What the two sets of
heads read is what the readers need: in an attention census at the CoC
queries, the CE-favored heads attend to the preceding CoC tokens ($0.19$ vs.\
$0.07$ in layers 0--21, $0.09$ vs.\ $0.04$ in 22--34) and the FM-favored heads
to the ego history ($0.05$ vs.\ $0.02$, $0.03$ vs.\ $0.01$; two-sided
Mann--Whitney $p \le 6\times10^{-5}$), while both attend to the scene alike
($p = 0.68$ and $0.88$); within layers, attention to the preceding CoC
correlates with the CE score ($\rho = 0.45$ / $0.48$) and attention to the ego
history with the FM score ($0.22$ / $0.38$). The reasoning tokens are thus read
twice, word by word by the text that continues them and as a summary at their
boundary by the action expert, and the two scores each see one reading. For MLP
channels the exception is confined to layers 22--34 ($0.63$ in 0--21, $0.24$
in 22--34; $0.77$ and $0.37$ corrected) and follows the same pattern at the
head clause ($0.77$ / $0.57$; $0.96$ / $0.78$).
```

## 다음 단계 제안 (C, 승인 시 진행)

A의 63–75%는 gradient 몫이다. 위치 수준 knockout 둘이면 인과가 된다: (i) **읽는 쪽** — expert ← cache knockout(`run_pathway.py`)에
경계 두 자리 / CoC 단어 / 무작위 단어 2개 config를 추가, val 50클립·K = 8·Ada 한 장 약 15분(예측: 경계 두 자리 차단 ≈ CoC 전체 차단,
단어 차단 ≈ 대조); (ii) **쓰는 쪽** — Run B의 T(0–21) 집합을 경계 토큰에서만 / 단어에서만 끄기(예측: FM 손상 +11.7%는 경계 조건에,
C의 NLL 손상은 단어 조건에). 둘 다 config 추가가 먼저이고 실행은 B와 같은 패턴.

**추가 (2026-09-26).** 질의 위치별 census(`outputs/coc_census_pos_v1/`)로 독자 이야기가 닫힌다: ego-history 읽기는 단어에서 편평(0.03)하고 두 경계 토큰에서 뛰며(`<|traj_future_start|>`: FM 쪽 head 0.193 vs CE 쪽 0.027, 0–21층), 두 special 토큰이 FM 쪽 head의 ego-history attention 43–47%를 차지한다. 즉 Run A의 FM gradient, Run B의 행동 손상, census의 ego-history 읽기가 모두 같은 자리 — 추론의 경계 토큰 — 에 있다. 표·문장은 `2026-09-25_coc_token_importance.md` 추가 2, 그림 `figures/supplementary/fig12_coc_query_ego_history_by_position.png`.
