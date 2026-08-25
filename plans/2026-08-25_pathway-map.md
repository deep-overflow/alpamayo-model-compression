# 정보 경로 지도 (Pathway Map) — attention knockout으로 본 토큰 구간별 인과 기여

- 상태: 승인됨 (2026-08-25, 사용자 "분석 계획 세운 후에 진행해줘")
- 선행: `reports/archive/2026-07-20_head-pruning-attention-analysis-results.html` §3.3, §4 (미실행 후속 항목)
- 선행 문헌: Map the Flow (arXiv 2510.13251), VLA-Pruner (arXiv 2511.16449)
- 브랜치: `worktree-pathway-map`

## 1. 배경

07-20 리포트는 expert가 VLM 캐시의 각 구간에 배분하는 attention **질량**을 측정했다
(vision 72.6% / prompt text 16.7% / traj-history 3.9% / 생성 CoC 2.3% / sink 1.1%).
그리고 같은 리포트가 곧바로 **"질량은 중요도의 대리 지표가 아니다"**를 보였다
(layer 내 Taylor vs vision mass ρ = +0.078; 각 layer에서 attention 최대 head가 Taylor
최상위인 경우는 36개 중 2개).

따라서 위 질량표는 "expert가 어디를 보는가"이지 "expert가 무엇을 필요로 하는가"가 아니다.
후자는 한 번도 측정된 적이 없다. 07-20의 다음 단계 목록에 다음 항목이 있었으나 실행되지 않았다:

> "CoC 구간을 마스킹한 denoising으로 CoC가 trajectory에 미치는 실제 인과적 기여 측정
> (attention 경로 vs cache 갱신 경로 구분)"

이 계획은 그 항목을 Map the Flow의 attention-knockout 방법론으로 일반화해 실행한다.

## 2. 이 저장소에 대해 이 실험이 갖는 위치

이 프로젝트는 한 달 동안 **유닛 축**(Q head / MLP channel / KV group)에서
"궤적에 중요한 것과 추론에 중요한 것이 다르다"를 확립했다. 이 실험은 같은 질문을
**토큰·경로 축**으로 옮긴다. 07-27 축 조사에서 vision token 축을 1순위(−16~22% e2e)로
꼽았고, 가장 가까운 선행연구인 VLA-Pruner의 "semantic-action gap"이 정확히 이 질문이다.
Map the Flow는 단일 tower VideoLLM에서 readout 하나(정답 확률)로 경로 지도를 그렸다.
우리는 **readout이 둘(CoC NLL / 궤적)** 이고 **tower가 둘**이라 그들이 그릴 수 없는
지도를 그릴 수 있다.

## 3. 구조적 사실 — 왜 X3가 완전한 ablation인가

VLM prefill은 causal이다. 따라서 프롬프트 위치의 K/V는 **뒤이어 생성된 CoC의 영향을 받지 않는다.**
CoC가 캐시에 남기는 것은 오직 `coc_start … coc_end` 구간의 K/V뿐이다.
expert는 캐시를 통해서만 VLM과 통신한다.

⇒ **`prefix_mask[coc_start:coc_end] = 0`은 CoC가 궤적에 미치는 영향을 100% 제거한다.**

07-20이 제기한 두 가설 중 (b) "CoC의 영향이 attention 경로가 아니라 cache 자체가 갱신된 것을
통해 간접 전달된다"는 causal masking 하에서 **구조적으로 불가능**하다. 이 실험은 그 사실을
전제로 (a)의 크기를 직접 잰다.

부수 confound 하나: `offset = prefill`은 CoC 길이를 포함하므로, CoC 위치를 막아도 diffusion
토큰의 RoPE 위치는 CoC 길이만큼 밀려 있다. 동일 길이 무작위 구간 차단(G0)이 이 confound를 흡수한다.

## 4. 가설

- **H1 (경로 필요성).** expert가 읽는 구간마다 인과적 필요도가 다르며, 그 순서는 attention
  질량 순서와 다르다.
- **H2 (추론 경로).** 생성 CoC 구간을 차단하면 궤적이 유의하게 나빠진다.
  기각되면 "CoC는 궤적에 attention 경로로 기여하지 않는다"가 되고, §3에 의해
  **CoC는 궤적에 전혀 기여하지 않는다**가 된다 — 이 프로젝트의 안전 서사에 직접 영향.
- **H3 (이중 readout 분기).** VLM 내부 edge 중 CoC NLL과 궤적 손상의 부호·크기가 갈리는
  edge가 존재한다. 없으면 dual 논지는 토큰 축으로 이전되지 않는다.

## 5. 설계

### 공통

- 데이터: `outputs/split.json`의 val 구간 클립. Stage 1은 50 클립.
- **teacher-forced CoC**: 클립마다 무손상 모델이 CoC를 1회 생성하고, 모든 config가
  **같은 토큰열**로 forward한다. 텍스트 변화와 경로 변화가 섞이지 않는다 (P2 프로토콜).
- 클립당 VLM forward 1회 → 모든 X config가 같은 캐시를 재사용 (expert_per_clip 최적화).
- 궤적: 공유 캐시 위에서 K=8 denoise, 시드는 config 간 동일 → 전 config paired.
- 판정: 클립 단위 paired, bootstrap 95% CI 주 판독, Wilcoxon 보조.
  minADE 델타는 heavy-tailed이므로 중앙값 병기.
- 하드웨어: Ada (GPU 4–7). 아키텍처 내에서만 결정성이 성립하므로 전 arm 동일 카드군.

### Stage 1 — expert ← 구간 (X 계열), 전 layer

`denoise_with_cache(..., prefix_mask)`가 이미 캐시 위치별 마스크를 받고
`_build_expert_pos_ids_and_attn_mask`가 `prefix_mask == 0`을 `finfo.min`으로 바꾼다.
**신규 수술 코드 없음.**

| id | 차단 구간 | 질량(참고) |
|---|---|---|
| X0 | 없음 (baseline) | — |
| X1 | vision (~2,880 tok) | 72.6% |
| X2 | traj-history (48 tok) | 3.9% |
| X3 | **생성 CoC** | 2.3% |
| X4 | prompt text | 16.7% |
| X5 | sink (pos 0) | 1.1% |
| X6 | vision 제외 전부 | — |
| X1c/X2c/X3c/X4c | **G0 통제**: 동일 개수 무작위 위치 | — |

무작위 통제는 클립별로 시드 고정해 재현 가능하게 뽑는다.

### Stage 2 — VLM 내부 edge (E 계열), layer window

Map the Flow대로 중심 `l`의 k=9 layer window에 additive mask를 걸어 (query group ← key group)
edge를 끊는다. 신규 코드: VLM text layer의 self_attn forward pre-hook.

| id | query ← key | 비고 |
|---|---|---|
| E1 | vision ← 같은 카메라 이전 프레임 | cross-frame (시간) |
| E2 | vision ← 다른 카메라 | cross-camera (공간, 우리 고유) |
| E3 | traj-history ← vision | |
| E4 | instruction text ← vision | |
| E5 | CoC ← vision | |
| E6 | CoC ← traj-history | ego-motion 경로 |
| E7 | CoC ← instruction text | |
| E8 | 전체 ← sink | sink head 181개와 대조 |

- readout 2종: CoC NLL(teacher-forced) + 궤적.
- 스윕은 K=4로 비용을 낮추고, 유의한 셀만 K=8로 재측정한다.
- 층 중심은 4층 간격 9개 (l = 0, 4, 8, …, 32).

Stage 2는 Stage 1 결과를 보고 착수한다.

## 6. 사전 등록 게이트

| 게이트 | 조건 | 실패 시 해석 |
|---|---|---|
| **G0** (방법 타당성) | 각 구간 차단 효과가 **동일 크기 무작위 차단**보다 유의하게 클 것 | 통과 못 하면 "무엇을 막아도 같다" → 지도가 무의미. Map the Flow의 통제군과 동일 |
| **G1** (질량≠인과) | 구간별 손상 순위가 attention 질량 순위와 **다를 것** (Spearman < 0.9) | 같으면 질량표로 충분했다는 뜻이고 knockout의 부가가치가 없음 |
| **G2** (H2, 주 판정) | X3 − X0의 minADE paired CI가 0을 배제하는가 | 배제하면 CoC의 인과 기여 확정. 포함하면 §3에 의해 "CoC는 궤적에 기여하지 않는다"로 읽고, 이 저장소의 추론-안전 서사를 개루프 범위에서 재서술해야 함 |
| **G3** (검정력) | X1(vision) 차단이 유의하게 파국적일 것 | 이것조차 검출 안 되면 표본이 부족한 것이므로 G2의 null을 해석하면 안 됨 |

G3는 **양성 통제**다. G2가 null로 나왔을 때 "효과 없음"과 "검정력 없음"을 가르는 장치이며,
이 저장소가 07-22 회전 버킷 n=3 사건에서 배운 교훈을 반영한 것이다.

## 7. 비용

- Stage 1: 클립당 rollout 1회(~1.4s) + TF forward 1회(~0.4s) + 10 config × 8 denoise × 0.38s
  ≈ 32s → **50 클립 ≈ 27분**, GPU 1장.
- Stage 2: 8 edge × 9 중심 = 72 config, 각각 VLM forward 필요.
  K=4 기준 클립당 ≈ 72 × (0.4 + 4×0.38) ≈ 138s → 50 클립 ≈ 1.9h, 4장 샤딩 시 ~30분.

## 8. 산출물

- `experiments/head_analysis/run_pathway.py` — Stage 1/2 러너
- `experiments/head_analysis/analyze_pathway.py` — paired 통계, 게이트 판정, 플롯
- `outputs/pathway_x_v1/` — `config.json`, `metrics.json`, `summary.txt`, `plots/`
- 최종: `reports/evaluation/2026-08-25_pathway-map.html`

## 9. 한계 (사전 명시)

- **key 차단은 softmax 질량을 재분배한다.** head 제거와 달리 불가피하며 Map the Flow도 같은
  조건이다. "knockout 효과 = 제거 효과"가 아니다.
- Stage 1은 **전 layer 일괄 차단**이라 layer 해상도가 없다. layer별 분해는 Stage 2 방식의
  hook이 필요하다.
- CoC 구간이 매우 작다(~14–16 tok vs vision 2,880). 작은 섭동이므로 G3 양성 통제가 필수다.
- 개루프 전용. 이 저장소의 반복된 교훈상 폐루프 판정은 별도다.
