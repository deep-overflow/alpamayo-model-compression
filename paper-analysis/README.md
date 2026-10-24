# 논문 분석 절 — 그림과 해석 정리

2026-09-23. 이 폴더는 논문의 Analysis 절(`analysis_v2.tex`)과 그것이 이어지는 Method 절
(`method_v2.tex`)에 쓰일 그림, 각 그림에서 끌어내는 분석, 그리고 그 근거 수치를 한곳에 모은 것이다.
분석의 목적은 하나다: **CE loss와 FM loss로 각각 잰 Taylor importance를 rank-max로 합치는
dual 기준이 왜 필요한가**를 점수 수준의 관찰(Finding 1)과 제거 실험(Finding 2)으로 보이는 것.

```
paper-analysis/
  README.md            이 문서 — 그림별로 무엇을 그렸고 어떤 문장을 지지하는지
  analysis_v2.tex      Analysis 절 (Finding 1–3)
  method_v2.tex        Method 절 (각 결정 옆에 근거 finding)
  figures/main/        본문 후보 패널 (.png 미리보기 + .pdf)
  figures/supplementary/  보조 패널
  scripts/             이번에 새로 만든 패널의 재현 스크립트
```

## 1. 분석 절의 흐름

| 절 | 무엇을 보이나 | 근거 |
|---|---|---|
| **Finding 1** VLM backbone | 두 objective의 중요도는 backbone의 다른 부분에 있다. 21층까지는 둘 다 vision 자리의 출력에서 중요도를 벌어 순위가 거의 같고, 22층부터는 언어 자리의 출력에 집중되면서(CoC는 거의 전부, trajectory는 부분적으로) 순위가 갈라진다. 같은 자리로 읽으면 다시 일치하고, 생성 CoC 자리만 예외 | 점수 분해 (1차 민감도) |
| **Finding 2** 대역별 ablation | 후반(22–35)에서는 CoC 중요도만 제거 손상과 대응한다(FM은 무반응, traj-only 후반 마스크 NLL +65%). trunk(0–21)에서는 두 loss가 어느 점수든 높게 보는 unit에 의존하므로 합집합이 어느 한쪽보다 훨씬 낫다(FM +4.8% vs +17.0 / +26.1%). `I_traj`만으로는 action조차 안 된다(깊이 분할 규칙 +13.5% / +0.98 m) | 제거 실험 (held-out 100클립) |
| **Finding 3** action expert | MLP 채널은 denoising step에 걸쳐 순위가 안정적이고(0.874 vs Q 0.668) 중요도가 상위 10%에 집중된다(90% 제거 시 1.8%) | step별 점수 |
| **Method** | 두 점수 분리 ← F1; 층내 rank 후 max ← F2; 층별 고정 예산 ← F1+F2; expert 고정 채널·attention 유지 ← F3 | |

Finding 1은 **점수**(gradient, 무한소)에 대한 진술이고 Finding 2는 **제거**(유한 섭동)에 대한
진술이다. 두 절은 동사를 다르게 쓴다 — F1은 "importance is earned at / the score is sensitive to",
F2는 "the loss depends on … once units are removed". 후반 `I_traj`의 언어 자리 몫은 F1에서는 실재하고
F2에서는 기능이 없는데, 이것은 모순이 아니라 "어느 점수를 어느 깊이에서 믿을지는 점수만으로는
모른다"는 F2의 존재 이유다.

## 2. 핵심 주장: 왜 두 loss를 같이 써야 하는가

1. **두 점수는 같은 것이 아니다.** 층내 순위 상관 Q 0.78 → 0.34, MLP 0.75 → 0.34 (0–21 → 22–34층).
   후반 유지집합은 68%만 겹친다(층당 head 약 6개).
2. **후반층에서는 CE 점수가 없으면 언어가 무너진다.** 후반에 어떤 마스크를 씌워도 FM은 ±0.6% 이내인데
   NLL은 traj-only +65.5% vs CoC-only +6.3% vs union +10.1%. traj가 버리고 union이 지킨 head만 빼도
   +8.6%(무작위 +1.3~3.2%).
3. **trunk에서는 합집합이 어느 한쪽보다 훨씬 낫다.** FM +4.8% (union) vs +17.0% (CoC-only) vs
   +26.1% (traj-only); minADE +0.06 / +0.46 / +2.64 m. 두 점수가 0.85–0.96으로 일치하는 구간에서 난
   차이다.
4. **`I_traj`만으로는 action도 안 된다.** 전체 마스크의 근사 동률(+6.6 vs +5.5%)은 두 대역을 동시에
   망가뜨린 상쇄다. trunk는 traj·후반은 CoC로 나누면 +13.5% / +0.98 m, 후반을 union으로 두고 trunk만
   traj로 바꿔도 +7.7% / +0.26 m (union +5.5% / +0.01 m).

| 예상 반론 | 답 (수치) |
|---|---|
| 후반 불일치는 MLP 잡음 | Q head는 천장이 평평(보정 0.85 → 0.37); 후반 선택에 따라 NLL이 +65.5 vs +6.3%로 갈림 |
| CE만 쓰면 되지 않나 | trunk에서 action을 잃음: FM +17.0%, minADE +0.46 m; 전체 +15.8% / +0.56 m |
| FM만 쓰고 후반은 안 자르면 | 예산이 달라지고 trunk 선택 자체가 나쁨(+26.1%, +2.64 m) |
| trunk는 FM, 후반은 CE로 나누면 | +13.5% / +0.98 m vs union +5.5% / +0.01 m |
| 두 loss를 더해 점수 하나로 | FM 원값이 CE의 6–10배라 traj 단일 기준이 됨(유지 head의 93.9%를 FM이 결정); rank 정규화가 합집합을 만드는 장치 |
| max가 최적인가 | 주장하지 않음. 두 z-score 평균은 dual과 구별 불가; max의 장점은 항을 더해도(FM 10스텝) 안 무너진다는 것뿐 |

## 3. 그림별 분석

각 항목: **파일** → 무엇을 그렸나 → 읽을 수치 → 지지하는 문장(tex) → 해석 경계.

### Finding 1 — 문단 1: 깊이 프로파일

**`main/fig1_depth_mlp`, `main/fig1_depth_q_head`** (기존 `fig/fig3_vlm_analysis` 안)
- 층별 중요도 합을 objective별 최댓값으로 정규화. 파랑 `I_traj`, 주황 `I_CoC`.
- peak: MLP 18 / 27, Q 19 / 24.
- 문장: *"trajectory importance peaks in intermediate layers … and drops rapidly, whereas CoC importance keeps rising and peaks later."*
- 경계: 정규화가 곡선별이라 두 곡선의 높이는 비교 불가. 모양만 읽는다.

**`main/fig3_ratio_step`** (추가 후보)
- 원값 비율 `I_traj / I_CoC`의 층별 값. 0–21과 23–34에서 각각 평평, 사이에서 4.6배 하락(변화점 23, R² 0.93 / 0.86).
- 문장: *"The two profiles are not two independent shapes: the ratio … is nearly constant within layers 0–21 and again within layers 23–34 … The different peaks therefore follow from a single change of regime at layers 22–23."*
- 이 그림이 trunk / late 구분의 근거다. 비율의 **수준**(12.9×)은 loss 단위가 달라 해석하지 않는다.

### Finding 1 — 문단 2: 층내 순위

**`main/fig1_rank_agreement`** (기존 wrapper 안)
- 층 안에서 두 점수의 unit 순위 Spearman. Q 0.78 → 0.34, MLP 0.75 → 0.34.
- 문장: *"the within-layer Spearman correlation between the two rankings falls from 0.78 … to 0.34 for Q heads and from 0.75 to 0.34 for MLP channels."*

**`supplementary/fig3_noise_ceiling_q_head`, `_mlp`**
- 100클립을 반으로 나눠 같은 점수끼리 낸 자기 일치도(천장)와 objective 간 일치도.
- Q: 천장 평평(0.85 → 0.82, 0.90 → 0.89), 보정 0.85 → 0.37 — 진짜 불일치. MLP: 천장 자체가 내려가(0.69 → 0.46, 0.80 → 0.50) raw 하락의 약 1/3이 잡음, 보정 0.92 → 0.54.
- 문장: *"For Q heads this is a genuine disagreement … For MLP channels the ceiling itself falls in late layers, and about one third of the raw drop is measurement noise."*
- 경계: 보정은 Pearson 감쇠 공식을 Spearman에 적용한 근사. 믿는 근거는 세 경로(반분, 서로소 추출 3개, 별도 run)가 같은 값을 준다는 점.

### Finding 1 — 문단 3: 어느 자리의 출력에서 벌리는가

**`main/fig4_token_handoff5_mlp`** (새 패널, 2행: 위 `I_traj`, 아래 `I_CoC`)
- 층 점수 중 vision / prompt-text / CoC / ego-history 자리에서 unit이 쓴 출력이 벌어들인 몫(부호 보존 가법 분해, 합 = 1).
- 6–17층: vision 몫 `I_traj` 0.90, `I_CoC` 0.75. text 쪽 몫 `I_CoC` 0.24 (prompt 0.10 + CoC 0.14) vs `I_traj` 0.07. 27–34층: `I_CoC` text 0.98 (prompt 0.70, CoC 0.28), `I_traj` text 0.66에 vision 0.26·history 0.08 유지.
- 문장: *"In layers 6–17 both objectives earn their importance mainly from the units' outputs at vision positions … They do not agree completely, and the remainder is already the language side: I_CoC earns 24% (MLP) and 34% (Q heads) of its trunk importance at text-side positions, against 7% and 18% for I_traj … In layers 27–34, I_CoC is earned almost entirely (98%) by the units' outputs at text-side positions, whereas I_traj is earned there only partially (49% for Q heads, 66% for MLP channels) and keeps a substantial part at vision (35% and 26%) and ego-history (16% and 8%) positions."*
- 경계 (중요): y축은 **unit이 그 자리에 쓴 출력**이 objective에 얼마나 중요한가이지, 그 자리에서 **무엇을 읽었는가**가 아니다. vision 자리는 causal mask 때문에 vision 데이터만 담지만, text·CoC 자리는 vision 정보를 읽어 합치는 곳이다(경로 지도: 18–26층 CoC←vision 차단 NLL +18.2%, CoC←text +0.9%). "후반은 언어 자리의 출력에 집중된다"까지가 점수가 허락하는 문장이고, "언어 데이터를 처리한다"는 아니다. 또한 몫은 1차 점수의 구성이라 후반 `I_traj`의 몫은 기능이 없는 질량의 구성이다(F2).

**`main/fig4_token_handoff5_q_head_bands`** (Q head; 층별 선은 흐리게, 대역 평균 굵게)
- Q head는 층당 head 32개라 층별 몫이 출렁인다. 클립 부트스트랩으로 확인하면 층당 CI 반폭 ≈0.05로 잔진동은 잡음이고 큰 봉우리(`I_traj` CoC 몫 27층 0.73 [0.67, 0.79], 22층 0.53)는 실제다. 평활 대신 대역 평균(0–5 / 6–17 / 18–21 / 22–26 / 27–34)을 얹었다.
- 대역 값: `I_traj` vision 0.36 / 0.72 / 0.50 / 0.32 / 0.35, CoC 0.13 / 0.05 / 0.19 / 0.36 / 0.32, history 0.32 / 0.10 / 0.08 / 0.07 / 0.16; `I_CoC` vision 0.27 / 0.63 / 0.31 / 0.08 / 0.02, prompt 0.33 / 0.16 / 0.35 / 0.46 / 0.36, CoC 0.30 / 0.18 / 0.30 / 0.46 / 0.62.
- 보조: `supplementary/fig4_token_handoff5_q_head` (원 층별 선), `_ci` (95% 부트스트랩 띠).

**`supplementary/fig4_token_handoff_mlp`, `_q_head`** — 원래의 2선(vision 몫만) 버전. `paper-fig/supplementary/`에 논문 스타일 export가 있다.

**`supplementary/fig4_pertoken_mlp`, `_q_head`** — 토큰 **하나당** 중요도(타입별 층 합 ÷ 토큰 수, 클립별 계산). 두 loss 모두 모든 깊이에서 CoC 토큰 하나가 vision 토큰 하나의 19–64배(trunk). trunk의 vision 90% 몫은 2,880개의 개수 효과다. 후반에서 vision 토큰 하나의 값이 약 1/10로 떨어지는 것이 hand-over의 토큰당 표현. 비율 `I_traj / I_CoC`는 토큰 수로 나눠도 불변이므로 비율 그림의 토큰 수 정규화판은 없다.

### Finding 1 — 문단 4: 같은 자리로 읽으면

**`main/fig4_same_token5_q_head`** (새 패널) — 결정적 그림
- 두 점수를 같은 토큰 자리로 제한한 뒤 층내 순위 일치도(천장 보정). pooled / vision / prompt / CoC / ego-history.
- 22–34층: pooled 0.38 → vision 0.79, prompt 0.81, history 0.94; CoC 0.41. 0–21층: pooled 0.85 → vision 0.96; CoC 0.47.
- 문장: *"When both scores are read at the same kind of position, the ceiling-corrected agreement in layers 22–34 rises from 0.38 … to 0.79 at vision positions, 0.81 at prompt-text positions, and 0.94 at ego-history positions … The exception is the generated CoC (0.41 …): among the units whose outputs at the reasoning tokens matter, the two objectives rank different ones. For Q heads this holds at every depth (0.47 in the trunk), and it is the source of the trunk's residual disagreement as well."*
- 읽기: CoC 자리의 낮은 값은 "CoC 자리에 출력을 쓰는 unit 중 어느 것이 중요한가의 순위가 objective에 따라 다르다"는 뜻이다. Q head에서는 깊이와 무관하므로 후반층의 성질이 아니라 추론 토큰 자리의 성질이다.

**`main/fig4_same_token5_mlp`** — MLP 판. 순서는 같다(history 0.83 > prompt 0.74 ≈ vision 0.72 > pooled 0.54 > CoC 0.37). Q와 다른 점: CoC 자리가 trunk에서는 0.78이었다가 후반에 0.37로 내려간다(깊이 의존). 천장이 낮으므로(late 0.43–0.66) 보정값으로만 읽는다.

**`supplementary/fig4_same_token_q_head`** — 원래의 3선(pooled / vision / text-side) 버전. text-side 0.53은 prompt 0.81과 CoC 0.41의 중간값이었다. `paper-fig/`에 논문 스타일 `_q_head`, `_mlp`가 있다.

**`supplementary/fig4_ratio_by_token5_mlp`, `_q_head`** — 비율 `I_traj / I_CoC`를 토큰 타입별로. vision·ego-history에는 계단이 없고(1.2× / 0.9×, history는 오히려 상승 23 → 35×) text 쪽에만 있다(3.0× / 4.8×). R1의 계단이 text 자리의 현상임을 보이는 보조 그림. 문단 1의 "단일 계단"과 문단 3의 hand-over를 잇는다.

### Finding 2 — 대역별 ablation

**`main/fig5_damage_by_band_fm`, `main/fig5_damage_by_band_nll`** (표 1의 막대판; `paper-fig/`에 논문 스타일)
- 세 선택(traj / CoC / union)을 trunk만 / 후반만 / 전체에 마스크로 적용, held-out 100클립, dense CoC 텍스트 teacher-forcing.
- 후반: FM −0.1 / +0.1 / +0.1%, NLL +65.5 / +6.3 / +10.1%. trunk: FM +26.1 / +17.0 / +4.8%, NLL +51.2 / +28.3 / +25.2%, minADE +2.64 / +0.46 / +0.06 m. 전체: FM +6.6 / +15.8 / +5.5%, NLL +240.6 / +67.2 / +32.6%, minADE +0.06 / +0.56 / +0.01 m.
- 문장 (후반): *"the FM loss does not depend on the units' outputs at this budget … The importance the trajectory objective earns there … is therefore a first-order quantity with no counterpart in removal: damage to the FM loss per unit of I_traj falls from 0.011 in the trunk to zero in the late band. The CoC NLL, by contrast, does depend on those outputs."*
- 문장 (trunk): *"both losses depend on the same units — and on more of them than either score alone identifies … keeping a unit whenever either objective's importance is high preserves the action better than the action objective's own ranking."*
- 경계: 단일 dose(24%)만 측정. 물리적 제거가 아닌 마스크(기능적으로 동일). 텍스트를 dense의 것으로 고정했으므로 자유 생성의 붕괴(CoC-only 종료 실패 8–13%)는 여기 안 나온다.

**`main/fig5_band_mix`** (새 패널) — "`I_traj`만으로는 안 되나"의 답
- trunk를 traj로 고정하고 후반을 traj / union / CoC / 무작위 / 그대로 둔 마스크의 FM·minADE.
- union 전체 +5.5% / +0.01 m; traj 전체 +6.6% / +0.06 m; traj trunk만 +26.1% / +2.64 m; + union 후반 +7.7% / +0.26 m; + CoC 후반 +13.5% / +0.98 m; + 무작위 후반 +8.8% / +0.42 m.
- 문장: *"That near-parity is an artefact of the trajectory-only rule damaging both bands at once … A depth-split rule that uses I_traj where the two scores agree and I_CoC where they disagree is therefore not a substitute: it costs FM +13.5% and minADE +0.98 m … The union is needed in the trunk itself, for the action."*
- 읽기: traj-only의 trunk 손상은 온전한 후반층이 증폭한 것이라, 후반을 어떻게든 솎으면 상쇄되어 전체 마스크에서 union과 비슷해 보인다. 상쇄이지 동등이 아니다.

**`supplementary/fig5_dual_saves`** — traj-only가 버리고 union이 지킨 후반 head만 제거: NLL +8.6% vs 같은 크기 무작위 +1.3–3.2%. FM은 어느 세트도 안 움직임. 문단의 "8.6%" 문장.

**`supplementary/fig5_dissociation_q_head`, `_mlp`** — 점수가 고른 소집합(k=6 head)의 FM vs NLL 손상. 예측한 이중 분리는 실패(action 쪽은 어떤 세트도 못 움직임), 언어 쪽만 성립. "후반은 언어 채널"의 세트 수준 근거.

**`supplementary/fig5_matched_fm`, `_nll`** — 전체 마스크 세 arm의 클립별 FM·NLL 분포(dense 대비). 표의 "all layers" 행의 분포판.

### Finding 3 — action expert

**`main/fig2_steps_q_head`, `main/fig2_steps_mlp`, `main/fig2_mass_curve`** (기존 `fig/fig4_expert_analysis` 안)
- step 간 순위 상관 heatmap(45쌍 평균 Q 0.668, MLP 0.874)과 누적 중요도 곡선(Q head 14/16 제거 = 68.5%, MLP 90% 제거 = 1.8%).
- 문장: *"MLP channels exhibit greater ranking stability across steps … Q-head importance is broadly distributed and MLP-channel importance is strongly concentrated in the top 10% of channels."*

## 4. 넣지 않은 것과 그 이유

- **인터페이스 설명**(무작위 readout이 같은 계단·같은 불일치를 재현, `paper-fig/fig5_probe_ratio_mlp.pdf`): "왜 갈라지는가"의 답이지만 dual의 필요성 논증에는 필요 없어 분석 절에서 뺐다. 리뷰어가 "왜"를 물으면 한 문장으로 답할 수 있다.
- **주행 상황별·LingoQA·종료 실패**(물리적으로 자른 arm, 2,533클립): 프로토콜이 달라 실험 절 몫. traj-only는 cruise·감속/정지·회전에서 union의 1.2–2.8배 minADE, CoC-only는 LingoQA 30.2(dual 68.8), 종료 실패 8–13%.
- **후반 `I_traj`의 포트 경로**(cache 21–23이 주 통로): 점수의 경로이지 기능이 아니므로 분석 절과 무관.

## 5. 수치 출처

| 내용 | 위치 |
|---|---|
| 깊이 프로파일, raw 순위 | `outputs/importance_v2/importance.npz` (`experiments/paper/fig_criterion.py`) |
| 비율 계단, 천장 | `experiments/paper/fig_why_differs.py`, `figures/fig3_why_differs_stats.json` |
| 토큰별 몫, 같은 자리 일치도, 토큰당 값 | `outputs/gradanat_v1/{anatomy.npz, anatomy_perclip_q.npz, anatomy_perclip_mlp.npz, metrics_analysis.json, metrics.json}` |
| 대역별 ablation 표 | `outputs/armheld_v1/summary.txt` (iii), dense NLL 0.1708 / FM 0.2425 / minADE 0.8368 |
| 섞은 대역 | `outputs/armmix_v1/{summary.txt, metrics.json}` |
| 세트 제거, dual saves | `outputs/tokabl_v1/summary.txt` |
| expert step | `outputs/stepimp_fm_perstep_v2/step_importance.npz` |
| 경로 지도 (CoC←vision 등) | `outputs/pathway_e_v1/summary.txt` |
| 상황별 손상 | `outputs/failure_by_situation_v1/summary.txt` |
| 보고서 | `reports/evaluation/2026-09-21_why-importance-differs.html`, `2026-09-21_importance-causal-validation.html` |

## 6. 상태와 재현

- `figures/main`, `figures/supplementary`의 파일은 저장소 `figures/`의 복사본이다. 그중 **`*5_*`, `_bands`, `_ci`, `pertoken`, `band_mix`는 이번에 만든 패널**로, `fig_why_differs.py`가 만들지 않고 `figures/README.md`에도 없다(git untracked). 논문에 쓰기로 정하면 스크립트에 패널로 올려야 재현된다.
- `scripts/`의 파이썬 파일이 그 패널들을 다시 만든다. 저장소 `.venv`로 실행하며 `experiments/paper/fig_criterion.py`·`fig_why_differs.py`의 스타일 함수를 import한다. 출력은 각 스크립트 옆의 `figs/`에 떨어진다.
  ```bash
  cd /home/cvlab21/project/chan/alpamayo-model-compression
  MPLCONFIGDIR=/tmp/alpamayo-mpl .venv/bin/python paper-analysis/scripts/same_token5.py q     # 또는 mlp
  MPLCONFIGDIR=/tmp/alpamayo-mpl .venv/bin/python paper-analysis/scripts/token_handoff5.py
  MPLCONFIGDIR=/tmp/alpamayo-mpl .venv/bin/python paper-analysis/scripts/token_handoff5_q_variants.py
  MPLCONFIGDIR=/tmp/alpamayo-mpl .venv/bin/python paper-analysis/scripts/ratio_by_token5.py
  MPLCONFIGDIR=/tmp/alpamayo-mpl .venv/bin/python paper-analysis/scripts/per_token_importance.py
  MPLCONFIGDIR=/tmp/alpamayo-mpl .venv/bin/python paper-analysis/scripts/band_mix.py
  ```
  `handoff_q_check.py`는 Q head 몫의 클립 부트스트랩 검사만 출력한다(그림 없음).
- 스타일: `figures/`의 패널은 분석 스타일, `paper-fig/`의 pdf는 논문 스타일(12 pt 라벨)이다. 새 패널을 본문에 쓰려면 같은 스타일로 다시 뽑아야 한다.

## 7. 논문 스타일 패널 (2026-09-23 추가)

Finding 1의 본문 그림 두 장을 `fig_criterion.py --paper-style`과 같은 규격으로 만들었다
(고정 3.5 × 2.8 in, 축 라벨 12 pt·본문 10 pt, 실선, 팔레트 #618BC8 / #ECAE3C, x축 "VLM layer index").

| 파일 (`figures/main/`) | 내용 | 비고 |
|---|---|---|
| `fig1_rank_agreement.{png,pdf,svg}` | 층내 순위 일치도 (Q head / MLP channel), y축 −0.25~1.0, 범례 축 위 한 줄 | `scripts/paper_rank_agreement.py` (`fig_criterion.py`의 계산을 그대로 쓰되 y 범위만 바꿈; `paper-fig/` 판은 −1~1) |
| `fig4_token_share_traj_mlp.{png,pdf,svg}` | `I_traj` 몫: Vision / Text (prompt + CoC 합산) / Ego history, MLP 채널 | `scripts/paper_token_share.py` |
| `fig4_token_share_coc_mlp.{png,pdf,svg}` | `I_CoC` 몫, 같은 구성 | 위와 동일 |

스타일 공통부는 `scripts/paper_style.py`(export 규칙·팔레트)에 있다. Text 선은 prompt-text와 생성 CoC 자리의 가법 몫을 더한 것(가법 분해라 정확). 범례는 "Vision / Text / Ego history"로 두 패널 모두 축 위 한 줄에 놓았고(패널 안 `I_traj`/`I_CoC` 표기는 없음 — (a)/(b) subcaption으로 구분),
"text = prompt + generated CoC"는 캡션에서 정의한다. 세 번째 색 #C8615F(벽돌색)는 논문 팔레트에 없어서 새로 정했다.

## 8. held-out 클립에서의 재현 (2026-09-24 추가)

Finding 1의 두 패널이 calibration 클립(`calib_100`, 공식 train)에서만 잰 것이라, 모델이 캘리브레이션에
쓰지 않은 세 세트에서 같은 모양이 나오는지 확인했다: in-distribution val(`calib_val100`), long-tail
OOD(`calib_ood_100`), held-out 공식 test(`calib_test100`, `test_500`의 앞 100행, 새로 생성). 세 세트
각각에 gradient-anatomy run을 새로 돌렸고(`outputs/gradanat_{val100,ood,test100}/`, 클립만 다르고
나머지는 `gradanat_v1`과 동일), rank agreement 패널은 같은 run의 full-seed gradient에서 읽었다
(anatomy의 typed-sum이 곧 출하 점수; val100·OOD는 자기 importance run과 층별 Spearman ≥ 0.999로 일치).

**결과: 세 세트 모두 재현.** 사전 등록한 창(밴드 평균이 calib_100 값의 ±0.10 안)에 token share 24/24
셀(MLP·Q head), rank agreement 17/18 셀이 들어온다. 유일한 예외는 val100의 MLP 7–21층 0.77(calib 0.67,
+0.10)로, held-out 쪽이 trunk에서 *더* 일치하는 방향이라 주장을 약화시키지 않는다. 표 전체·수치·재현
명령은 `2026-09-24_heldout_replication.md`, 계획과 판정 기준은 `plans/2026-09-24_token-share-replication.md`.

| 셀 (MLP) | calib_100 | val100 | OOD | test100 |
|---|---:|---:|---:|---:|
| `I_traj` 6–17 vision / 27–34 vision | 0.90 / 0.26 | 0.90 / 0.29 | 0.93 / 0.30 | 0.92 / 0.27 |
| `I_traj` 27–34 text / hist | 0.66 / 0.08 | 0.66 / 0.05 | 0.66 / 0.04 | 0.68 / 0.05 |
| `I_CoC` 6–17 vision / text | 0.75 / 0.24 | 0.75 / 0.24 | 0.72 / 0.26 | 0.71 / 0.27 |
| `I_CoC` 27–34 text | 0.98 | 0.98 | 0.98 | 0.98 |
| rank ρ Q 0–6 / 7–21 / 22–34 | 0.78 / 0.77 / 0.33 | 0.81 / 0.81 / 0.33 | 0.77 / 0.79 / 0.40 | 0.78 / 0.80 / 0.35 |
| rank ρ MLP 0–6 / 7–21 / 22–34 | 0.93 / 0.67 / 0.34 | 0.95 / 0.77 / 0.39 | 0.94 / 0.73 / 0.41 | 0.94 / 0.74 / 0.39 |

| 파일 (`figures/supplementary/`) | 내용 |
|---|---|
| `fig1_rank_agreement_{val100,ood,test100}.{png,pdf,svg}` | 순위 일치도, 논문 스타일 (`scripts/paper_rank_agreement.py <run> <suffix>`) |
| `fig4_token_share_{traj,coc}_mlp_{val100,ood,test100}.{png,pdf,svg}` | 토큰 자리 몫, 논문 스타일 (`scripts/paper_token_share.py <run> <suffix>`) |

비교표는 `scripts/token_share_compare.py gradanat_v1 gradanat_val100 gradanat_ood gradanat_test100`이 만든다.
논문에는 "the same shapes appear on validation, OOD and test clips (supplement)" 한 문장과 supplement
한 figure(4열 × 3행 또는 rank 패널 3장 + share 패널 6장)로 넣으면 된다.

한눈에 보는 12패널 시트(3행 × 4열, 미리보기용): `figures/supplementary/heldout_replication_sheet.png`.

**Q head 버전 (2026-09-24 추가).** 같은 스크립트에 세 번째 인자 `q`를 주면 Q head 몫을 그린다
(`scripts/paper_token_share.py <run> <suffix> q`). 네 세트 모두 `figures/supplementary/`에 있다:
`fig4_token_share_{traj,coc}_q_head.*`(calib_100), `..._q_head_{val100,ood,test100}.*`. Q head 몫은
층별 지터가 크지만(클립 부트스트랩 95% CI 반폭 ≈ 0.11) 밴드 평균은 네 세트에서 ±0.10 안이다(위 비교표의
"Q heads" 표). 미리보기: `figures/supplementary/heldout_replication_sheet_q.png`.

**head별 지배 자리 지도 (2026-09-24 추가).** `figures/main/fig6_head_token_map_q_head.*`(`scripts/head_token_map.py <run> <suffix>`):
36층 × 32 head 격자에 head의 지배 자리를 색으로, 층 내 중요도 순위(왼쪽) 또는 지배 몫(가운데)을 진하기로. held-out 세 세트에서도
head별 지배 자리가 유지된다(일치 75–86%, κ 0.61–0.79, calib 반쪽 천장 이상): `2026-09-24_head_type_stability.md`,
지도 `figures/supplementary/fig6_head_token_map_q_head_{val100,ood,test100}.*`.

**CoC 자리 순위 불일치 (2026-09-25 추가).** 원인 배제(잡음·모달리티·클립 혼합·gradient 방향·손실 경로)와 기전(CoC 질의 attention census: `I_CoC` 쪽 head는 이전 CoC, `I_traj` 쪽 head는 ego-history를 읽음, vision은 동일)을 `2026-09-25_coc_position_note.md`(한글 요약 + 논문 문장)와 `2026-09-25_coc_position_disagreement.md`(영문 상세)에 정리. 그림 `figures/supplementary/fig8_coc_query_attention.png`, `fig8b_coc_query_attention_bars.png`, `fig8c_coc_path_split.png`.

**CoC 자리 순위 불일치 — 원인 확정 (2026-09-25 판 3 추가).** Run A(CoC 위치별 typed gate, calib 99클립)로 예외가 CoC 안의 **토큰 가중 혼합**임을 확정: `I_CoC`는 단어에서, `I_traj`는 두 경계 토큰 `<|cot_end|>` `<|traj_future_start|>`에서(FM 몫 63–64%, CE는 구조적으로 0) 점수를 벌고, 단어만으로 제한하면 일치도 0.78–0.82 raw(0.90–0.92 보정)로 다른 자리 수준. Run B(held-out 100클립, CoC 자리에서만 head off)로 0–21층 교차 이중 해리 통과(`I_traj` 쪽: FM +11.7%·minADE +0.20 m·NLL +1.7%; `I_CoC` 쪽: NLL +16.5%·FM −0.2%), late 층은 `I_traj` 쪽 head가 텍스트에 필수(+247% vs +59%). 판 2의 "같은 토큰에서 다른 unit" 문장은 철회. `2026-09-25_coc_position_note.md`(판 3, 논문 문장 교체안), `2026-09-25_coc_position_disagreement.md`(Test 3–4), 그림 `figures/supplementary/fig9_coc_position_split.*`, `fig10_coc_position_dissociation.*`, 스크립트 `scripts/paper_coc_position_split.py`, `paper_coc_position_dissociation.py`.

**CoC 토큰 단위 범주별 importance (2026-09-25 추가, 논문 에이전트 요청).** Run A 데이터로 CoC 위치의 head 기여를 단어로 병합해 decision / scene / spatial / other / 두 special 토큰으로 나눔(`experiments/head_analysis/analyze_coc_token_importance.py`, `outputs/coc_tokimp_v1/`). 요청 정규화(special 포함)에서는 FM의 decision enrichment가 CE보다 낮고(질량 72–75%가 special 토큰), 단어만 보면 FM이 decision을 1.5배(CE 1.0–1.2배, 차이 +0.32 / +0.47), Q1 안에서는 +0.13 / +0.16; FM 쪽 head는 I_FM의 86%를 special 토큰에 씀. 정리 `2026-09-25_coc_token_importance.md`, 그림 `figures/supplementary/fig11_coc_token_category_shares.png`, `fig11b_coc_token_category_enrichment.png`. 6번(expert attention 마스킹)은 제안만.

**ego-history를 읽는 CoC 질의 위치 (2026-09-26 추가).** census를 질의 위치별로 재측정(`run_coc_census.py --per-query`, `analyze_coc_query_census.py`, `outputs/coc_census_pos_v1/`): ego-history 읽기는 단어 위치에서 편평(0.03)하고 두 경계 토큰에서 뛴다 — `<|traj_future_start|>`에서 FM 쪽 head 0.193 vs CE 쪽 0.027(0–21), 두 special 토큰이 FM 쪽 head의 ego-history attention 43–47%. 단어 중에는 속도·조향 동사(down/turn/slow/adapt/nudge/yield)가 조금 높음. 정리 `2026-09-25_coc_token_importance.md` 추가 2, 그림 `figures/supplementary/fig12_coc_query_ego_history_by_position.png`.

**CoC 단어 그림 (2026-09-26 추가).** `figures/supplementary/fig11c_coc_words.*` — I_traj가 무겁게 보는 CoC 단어(결정 절: accelerate/stop/nudge/yield/keep/turn/adapt; 원인 절: slowly/light/cut-in/curve/stopped/red/clear/sign/green/intersection)의 발생당 몫, I_CoC 병기. 스크립트 `scripts/paper_coc_words.py`, 데이터 `outputs/coc_tokimp_v1/metrics.json`.

**I_FM이 보는 텍스트 토큰 — prompt·ego-history·CoC 위치별 (2026-09-26 추가, Run D).** vision을 뺀 모든 위치에 typed gate(`run_text_position_anatomy.py`, `analyze_text_position_anatomy.py`, `outputs/textpos_v1/`). I_FM의 non-vision 질량은 언어가 아니라 숫자·경계에 있다: 0–21층 ego-history 토큰 37%(토큰 몫의 1.7배, 최신 timestep일수록 커서 마지막 8/처음 8 = 1.78), `<|vision_end|>` 마커 20%(1.4배), CoC 경계 토큰 11%(12배); 시스템·지시문·카메라 레이블·CoC 단어는 토큰 몫 이하. I_CE는 반대(CoC 단어 29% 4.8배, `<|cot_start|>` 12.5배, 턴 마커 1.5배). 어느 역할에서도 같은 토큰의 head 순위는 일치(보정 ≥ 0.93 / ≥ 0.86). 정리 `2026-09-26_text_token_importance.md`, 그림 `figures/supplementary/fig13_text_tokens.*`, `fig13_text_role_shares_analysis.png`, `fig13b_prompt_position_share_analysis.png`.

**head-token map 확장 (2026-09-26 추가).** `figures/main/fig6b_head_token_map_mlp.{png,pdf}` — fig6의 MLP 채널판(`scripts/head_token_map_mlp.py`; 층마다 채널을 자기 중요도 순으로 정렬, 점선 = 출하 절단 7,390/12,288): 0–21층은 두 점수 모두 거의 전부 vision 지배(traj 95%, coc 83%), 22–34층에서 갈림(traj: prompt 47% / CoC 29% / vision 22%; coc: prompt 88% / CoC 12%), 같은 지배 타입 0.84 → 0.51. `figures/main/fig6c_head_map_plain.{png,pdf}` — 토큰 타입 없이 층내 순위만(`scripts/head_map_plain.py`; Q head는 인덱스, MLP는 I_traj 순위로 정렬) + rank(I_traj) − rank(I_CoC) 열: 평균 |순위 차| Q 0.148 → 0.268, MLP 0.142 → 0.261 (0–21 → 22–34), |차| > 0.5인 unit 2.8% → 17.8% (Q), 3.2% → 14.5% (MLP).

**색 통일 (2026-09-26).** fig6(Q head)·fig6b(MLP)·held-out fig6 세 판의 토큰 타입 색을 fig4_same_position과 같게 바꿈: vision 파랑 #618BC8, prompt text 노랑 #ECAE3C, CoC 보라 #8E6BB3, ego history 벽돌색 #C8615F (`scripts/head_token_map.py`, `head_token_map_mlp.py`의 TYPES). fig6c는 목적함수 색(I_traj 파랑 / I_CoC 노랑)만 쓰므로 그대로.

**dominant share 단독 배치 (2026-09-26).** `figures/main/fig6d_head_token_map_dom_q_head.{png,pdf}`, `fig6d_head_token_map_dom_mlp.{png,pdf}` — I_FM | I_CE 두 패널, 색 = 지배 토큰 타입(fig4 팔레트), alpha = 지배 몫, 범례 하단 (`scripts/head_token_map_dom.py q|mlp [run] [suffix]`). MLP판은 층마다 자기 순위로 정렬, 점선 = 출하 절단.

**합본 (2026-09-26).** `figures/main/fig6e_head_map_combined.{png,pdf}` (`scripts/head_map_combined.py`) — 위 행 Q head, 아래 행 MLP; 열 = [지배 타입 I_FM] [지배 타입 I_CE] [층내 순위 I_FM] [층내 순위 I_CE], 순위 차 열 제외. MLP 지배 타입 패널은 자기 순위 정렬(점선 = 출하 절단), 순위 패널은 I_FM 순위 정렬. 타입 색 fig4 팔레트, 순위는 viridis 공용 콜로바.

**합본 갱신 (2026-09-26).** `fig6e_head_map_combined` — 범례 제목 제거, 다섯째 열 rank(I_FM) − rank(I_CE) 추가(노랑 = CE가 높게, 파랑 = FM이 높게; MLP는 I_FM 순위 정렬), 콜로바 둘(순위 viridis / 차이 diverging).

**합본 4열판 (2026-09-26).** `figures/main/fig6e_head_map_combined_nodiff.{png,pdf}` — 순위 차이 열을 뺀 4열 버전(범례 제목 없음). `scripts/head_map_combined.py gradanat_v1 "" nodiff`.

**층 순서 뒤집기 + Q head 정렬판 (2026-09-26).** fig6d 두 장과 fig6e 합본은 이제 위가 35층, 아래가 0층. `fig6e_head_map_combined_sortq.{png,pdf}`, `fig6e_head_map_combined_nodiff_sortq.{png,pdf}` — Q head 행의 순위·차이 패널에서 head를 층마다 I_FM 순위로 정렬(MLP 행과 같은 방식; KV 그룹 선 제거). `scripts/head_map_combined.py gradanat_v1 "" [nodiff] [sortq]`.

**합본 파일명 정리 (2026-09-26).** 기본은 순위 차이 열 없는 4열: `figures/main/fig6e_head_map_combined.{png,pdf}`(Q head 인덱스 순), `fig6e_head_map_combined_sortq.{png,pdf}`(Q head 행 순위 패널을 I_FM 순위로 정렬). 5열(차이 열 포함)은 `_diff` 접미사(`fig6e_head_map_combined_diff`, `fig6e_head_map_combined_sortq_diff`)로만 보관. 이전 `_nodiff` 파일은 삭제. 모두 위가 35층. `scripts/head_map_combined.py gradanat_v1 "" [sortq] [diff]`.

**raw로 통일 (2026-09-26).** 순위 일치도 그림은 raw Spearman(100클립 평균 점수의 층내 순위 상관)을 기본으로: `fig4_same_position_q_head`·`fig4_same_position_mlp`(y축 "Rank agreement ρ"; `scripts/paper_same_position.py [run] [suffix] [q|mlp] [raw|corrected]`, 기본 raw)과 `fig8c_coc_path_split`(`scripts/paper_coc_path_split.py "" raw`). 보정판은 `_corrected` 접미사로만 보관. Fig. 1(`experiments/paper/fig_criterion.py`)은 원래 raw. 전체 0–34층 raw 일치도: Q head 0.61, MLP 0.60 (0–21층 0.78 / 0.75, 22–34층 0.33 / 0.34).

**MLP 인덱스 순서판 (2026-09-26).** `figures/main/fig6e_head_map_combined_idxmlp.{png,pdf}` — 아래 행 MLP를 채널 인덱스 순서로(정렬·절단선 없음). `scripts/head_map_combined.py gradanat_v1 "" idxmlp`.

**합본 글씨 크기 (2026-09-26).** fig6e 전 변종: 축 라벨 15 pt, 눈금 12 pt, 콜로바 라벨 15 / 눈금 12 pt, 하단 범례 15 pt, 패널 제목 15 pt (캔버스 12.6 × 7.0 in 기준; `head_map_combined.py`의 LABEL/TICK/LEGEND/TITLE).

**합본 라벨·캔버스 (2026-09-26).** 글씨를 키우면서 라벨을 줄임: 제목 "I_FM: position type" / "I_FM: rank in layer", y축 "Layer (Q heads)" / "Layer (MLP)", x축 "MLP ch. (own rank)" / "MLP ch. (I_FM rank)" / "Q head (I_FM rank)", 콜로바 "Rank in layer"; 캔버스 14.7 × 8.1 in(4열) / 17.9 × 8.1 in(5열).

**타입 패널 인덱스 순 (2026-09-26).** fig6e의 MLP 위치-타입 두 패널은 이제 기본이 채널 인덱스 순(x축 "MLP ch. index", 절단선 없음); 순위 정렬판은 `typerank` 플래그(`_typerank` 접미사). 순위 패널은 그대로 I_FM 순위.

**토큰 몫 그림, Text 분리판 (2026-09-26).** `fig4_token_share_{traj,coc}_{mlp,q_head}_split.{png,pdf,svg}` (MLP는 `figures/main/`, Q head는 `figures/supplementary/`): Text를 Prompt text(노랑)와 CoC(보라, fig4 팔레트)로 나눈 네 곡선. `scripts/paper_token_share.py gradanat_v1 "" q|mlp split`. 수치(Q head, 6–17층 | 27–34층): I_traj prompt 0.13 / CoC 0.05 | 0.17 / 0.32; I_CoC prompt 0.16 / CoC 0.18 | 0.36 / 0.62. MLP: I_traj 0.04 / 0.03 | 0.37 / 0.29; I_CoC 0.09 / 0.14 | 0.70 / 0.28.

## 9. 그림 폴더 정리 (2026-09-26)

`figures/main`·`figures/supplementary`를 세 폴더로 재편: **`figures/1_main`** (논문에 들어간 분석 그림 7장: `fig4_same_position_q_head`, `fig4_token_share_{traj,coc}_q_head_split`, `fig6e_head_map_combined_sortq`, `fig2_mass_curve`, `fig2_steps_{mlp,q_head}`), **`figures/2_supplementary`** (서플 후보: MLP 쌍, held-out 재현 val100/OOD/test100, CoC 자리 fig9/10/13, census fig8b, fig12, fig11c, dual-saves·해리·matched fig5, depth profile fig1, hand-over fig4, ratio step fig3, method 패널 fig5_removed_mass/layer_dispersion — 확정 필요), **`figures/3_unused`** (보정판·구판·변형·분석 스타일). 전체 목록과 한 줄 설명은 `figures/MANIFEST.md`. 이전 노트의 `figures/main/...` 경로는 이 매핑으로 읽을 것. 렌더 스크립트 출력은 그대로 `scripts/paperfigs/`, `scripts/figs/`.
