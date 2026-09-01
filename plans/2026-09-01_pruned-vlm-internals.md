# 프루닝된 VLM의 내부 — expert attention 분포와 VLM 잔차 스트림

날짜: 2026-09-01. 브랜치: `pruned-vlm-internals`.
보고서: `reports/evaluation/2026-09-01_pruned-vlm-internals.html`.

## 가설

지금까지의 dense-vs-pruned 비교는 전부 **KV 캐시**만 봤다(`cache-shift`, `cache-proxy`).
`2026-08-28_cache-shift.html`은 (layer, head) 수준에서 expert attention mass 변화가 1 pp
미만이라고 보고했다(vision −0.6 pp). 두 가지를 묻는다.

**H1**: 그 "거의 안 변한다"는 결론은 **집계 해상도의 산물**이다. 스텝·query 토큰까지 풀고
분포 거리로 재면 span 안에서의 재배치가 드러난다.
**H0**: 고운 축에서도 평평하다 → 캐시가 이동해도 expert의 읽기 패턴은 바뀌지 않는다.

**H2**: 잔차 스트림 비교는 저장소에 전혀 없다. 층 안의 세 지점(`h_in`, `h_mid`, `h_out`)에서
dense와 pruned를 비교하면 **층 안 어디서 어긋나는지**와 **변환 양상 자체가 바뀌는지**가 갈린다.

## 설계

`dual_u40_v2`(VLM만 절단, expert 16/16 온전) vs dense. **모델 하나 + `mask_lib.PruneMasks`**
계측 — 체크포인트 두 개는 가중치만 36.3 GiB이고 캡처까지 더하면 48 GB 카드를 넘긴다.
마스크면 20.6 GiB이고 가중치·커널·코드 경로가 고정돼 측정 차이가 전부 프루닝 효과다.

teacher-forcing: dense가 한 번 rollout → 그 `seq_tf`를 두 모델에 동일하게. val500(=`indist_500`)
앞 32클립, 시드 `clip_seed(42, clip_id)`, Ada 고정, 결정성 설정 포함.

**탭**: `output_hidden_states=True`는 **쓸 수 없다** — DeepStack이 텍스트 층 0·1·2 뒤에
vision feature를 제자리 대입해 기록 텐서를 사후 변조하고(실측 최대 26.1), 마지막 항목은
RMSNorm을 거쳐 층 35의 raw 출력을 얻을 수 없다. 36개 decoder layer의 forward hook
(`args[0]=h_in`, `output=h_out`, 훅 안에서 clone) + `post_attention_layernorm` pre-hook(`h_mid`).

**지표**: 잔차는 세 지점의 cos·rel(모델 간)과 전이 cos·상대 쓰기 크기(모델 안).
attention은 TV(주), KL 양방향, JS, span mass, 엔트로피 — 모두 (layer 36, head 16, step 10,
query 64) 전 축 유지.

## 사전 등록 게이트와 결과

| 게이트 | 기준 | 실측 | 판정 |
|---|---|---|---|
| G0 정렬 | 캐시 층0 ΔK = 0, 층0 `h_in` rel = 0 | 0.0 / 0.0 | PASS |
| G1 잔차 분해 | `h_out == (h_in+a).add(m)` 36층 bitwise | 1.000 | PASS |
| G2 정규화 | bf16 행합 편차 | 2.98e-07 | PASS |
| G3-1 재실행 바닥 | 마스크 off ×2 → 0 | TV max 1.55e-07 | PASS |
| G3-2 bf16 바닥 | fp32 재구성 대비 | rel 2.23e-03 | 기록 |

## 결과 (32클립, 클립 중앙값 [95% CI])

### expert attention — H1 채택

| 지표 | 값 |
|---|---|
| **TV** | **+0.0836 [+0.0818, +0.0893]** |
| — span 사이 | +0.0250 [+0.0240, +0.0265] |
| — **span 안** | **+0.0616 [+0.0579, +0.0644]** = **71.2%** |
| JS | +0.0125 [+0.0121, +0.0133] |
| KL(dense‖pruned) | +0.0653 / KL(pruned‖dense) +0.0535 → 비대칭 **+0.0119** |
| Δ 정규화 엔트로피 | +0.0040 |

span mass 변화(pp): vision **−0.600**, text +0.717, hist +0.273, sink −0.439, coc −0.052.
**vision −0.6 pp는 `cache-shift`의 보고값을 독립 구현이 그대로 재현한 것**이다.

즉 **질량은 제자리인데 분포는 8.4% 움직이고, 그 71%가 span 경계를 넘지 않는다.**
층 축이 지배적(층 0–5 TV 0.02 → 층 20–35 0.17), query 축은 뒤쪽 waypoint가 더 크다
(0.083 → 0.090). 층별 span mass는 ±4 pp까지 흔들리며 서로 상쇄된다 — 집계가 숨긴 두 번째 층위.

### VLM 잔차 스트림 — H2

| 지점 | rel | cos |
|---|---|---|
| `h_in` | +0.2650 | +0.9633 |
| `h_mid` | +0.2716 | +0.9623 |
| `h_out` | +0.2760 | +0.9617 |

층 안에서 attention이 벌린 몫 **+0.0066**, MLP가 **+0.0046** (1.4배). 층별 곡선은
**층 22에서 0.46으로 정점을 찍고 층 25 이후 0.37로 되돌아온다** — 후반 층의 부분적 자기 교정.

**변환 양상 자체는 보존된다** (dense → pruned):

| 전이 | dense | pruned | 변화 |
|---|---|---|---|
| cos(h_in, h_mid) | 0.9904 | 0.9912 | 거의 동일 |
| **‖a‖/‖h_in‖** | 0.1182 | 0.1128 | **−4.6%** |
| cos(h_mid, h_out) | 0.9808 | 0.9819 | 거의 동일 |
| ‖m‖/‖h_mid‖ | 0.4272 | 0.4244 | −0.7% |

두 축에서 각각 ~40%를 지웠는데 쓰기 크기는 attention 4.6% / MLP 0.7%만 줄었다 —
**MLP가 훨씬 잘 흡수한다**. `2026-08-30_axis-taylor-comparability.html`의 개루프 결과(같은
파라미터에서 Q +0.030 vs MLP 0.000)와 같은 방향이고, 그 기전을 표현 수준에서 보여준다.

## 결론

1. "expert는 프루닝된 캐시를 거의 같게 읽는다"는 **span 해상도에서만 참**이다. 분포로 보면
   8.4%가 재배치되고 71%가 span 안에서 일어난다.
2. KL 비대칭이 양수 → dense가 보던 키를 pruned가 **버리는** 쪽이 우세하고, 동시에 엔트로피가
   올라 분포가 평평해진다.
3. 잔차는 방향(cos 0.96)을 지키고 크기·세부에서 어긋난다. 층 22 정점 후 되감김.
4. 블록의 상대적 쓰기 프로파일은 보존되고, 크기만 attention에서 선택적으로 줄어든다.

## 한계

32클립(셀 단위 주장 없음, marginal만), `dual_u40_v2` 하나, teacher-forcing 조건,
마스크 계측(물리 slim의 gather 경로 차이 배제), 개루프·표현 수준.

## 다음

`dualq_u40_v2`(Q만)·`dualm_u40_v2`(MLP만)에 같은 계측을 걸면 TV와 쓰기 크기 감소를 축별로
분리할 수 있다. 재실행 없이 러너의 `--config`만 바꾸면 된다.

## 파일

| 파일 | 역할 |
|---|---|
| `experiments/head_analysis/run_prunedvlm.py` | 러너(`ResidualTaps`, `BlockTaps`, `ExpertAttnCapture`, `fine_spans`) |
| `experiments/head_analysis/analyze_prunedvlm.py` | G0–G4 판정, 클립 부트스트랩, 그림 4장 |
| `experiments/head_analysis/prunedvlm_report_template.html` | 보고서 템플릿 |
| `outputs/prunedvlm_{smoke,nomask,dual}`, `outputs/prunedvlm_analysis` | 산출물 |
