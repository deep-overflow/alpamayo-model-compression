# expert 축 분해: Q head만 자를 때 vs MLP 채널만 자를 때 (동일 비율)

날짜: 2026-08-28. 브랜치: `expert-axis-ablation` (승인 후 생성).

## 가설

스텝별 중요도 분석(`2026-08-21_denoise-step-importance.html` §4, 재계산 2026-08-28)에서
expert **Q head의 중요도 랭킹은 디노이징 스텝에 강하게 의존**하고(스텝 간 ρ 0.717, 바닥
0.929; step 0 vs step 9 = 0.14) **MLP 채널은 약하게만 의존**한다(ρ 0.877, 바닥 0.945;
step 0 vs 9 = 0.74). `dualexp` e10/e15 스윕에서는 expert 손상이 전부 Q head 축에서 나왔고
MLP 폭(하위 15%)은 공짜였다 — 단 그 스윕은 dual VLM 위였고 MLP 축만 움직였다.

**H1**: 같은 비율로 자르면 **Q head만 자른 쪽이 MLP 채널만 자른 쪽보다 더 비싸다**, 제거
파라미터는 MLP 쪽이 6.05배 많음에도 불구하고. 근거는 헤드가 스텝-특화라 어느 헤드를 잘라도
어떤 스텝은 자기 헤드를 잃는 반면, MLP 채널의 하위권은 모든 스텝이 동의한다는 것.

**H0 (기각 시)**: 비용은 축이 아니라 파라미터 수에 비례한다 — 그러면 MLP-only가 더 비싸고
e10/e15의 "MLP 공짜"는 절단량이 작아서 생긴 바닥 효과였다는 뜻.

## 설계

expert 타워만 자르고 VLM·KV는 무접촉. 기준은 znorm `traj_exp_q` / `traj_exp_mlp`
(`importance_stepexp_znorm` — 출시 `slim_expert_znorm_r25`와 같은 파일). 레이어당 균일
비율, `mask_lib.select_mask`(k = round(n·r)). 순수 선택 config이므로 `--no-state`로 빌드
(slim_meta.json에서 bit-identical 복원, 디스크 무비용 — `dualexp` 계획과 같은 이유).

### arm과 파라미터 수 (expert: hidden 2048, head_dim 128, 16 heads, MLP 8256, 36층)

헤드 1개 = q_proj 행 + o_proj 열 = 2·2048·128 = **524,288**;
채널 1개 = gate/up 행 + down 열 = 3·2048 = **6,144** (85.3배 차이).
expert 타워 ≈ 2,279,236,608 (전체 11,078,526,194의 20.6%): Q(q+o) 301,989,888 (13.2%),
MLP 1,826,095,104 (80.1%).

| arm | config | 층당 제거 | 제거 파라미터 | 전체 대비 | expert 대비 |
|---|---|---|---|---|---|
| **q25** | `expertq_u25` | 4 / 16 heads | **75,497,472** | 0.681% | 3.31% |
| **m25** | `expertm_u25` | 2064 / 8256 ch | **456,523,776** | 4.121% | 20.03% |
| (기존) both25 | `expert_u25` = `slim_expert_znorm_r25` | 4 heads + 2064 ch | 532,021,248 | 4.802% | 23.34% |
| q50 | `expertq_u50` | 8 / 16 | 150,994,944 | 1.363% | 6.62% |
| m50 | `expertm_u50` | 4128 / 8256 | 913,047,552 | 8.242% | 40.06% |
| m_pm (선택) | `expertm_c341` | 341 ch (r=0.0413) | 75,423,744 | 0.681% | 3.31% |

- q25/m25가 주 비교(사용자 요청: **비율 동일**, 파라미터 수는 별도 보고). 같은 비율에서
  MLP-only가 Q-only의 **6.05배** 파라미터를 제거한다.
- both25 = q25 ∪ m25 가 정확히 성립하므로(같은 importance, 같은 select_mask) 기존
  `expert_znorm_r25_ps_indist`(val500, 500 rows)로 **가산성**을 공짜로 얻는다.
- q50/m50: MLP가 50%에서도 공짜인지, Q 비용이 헤드 수에 선형인지.
- m_pm (선택): Q25와 **파라미터 수를 맞춘** MLP-only 대조군(차이 −73,728, 0.1%). H1이
  성립할 때 "축 효과"와 "파라미터 수 효과"를 최종 분리하는 arm — 이것 없이는 q25가 더
  비싸도 "파라미터가 6배 적은데도"라는 정성 진술까지만 가능하다.

### 평가

고정 개루프 프로토콜: rollout-only, **indist val500 / test500 / OOD-val 262**, minADE@6 ·
minFDE@6, 클립 유도 시드, **Ada 4–7**, `launch_arms.sh` 큐(arm당 세트당 2 shard).
대조군은 기존 `baseline_ada_ps_{indist,test,oodval}`.

주의(기존 관찰): expert-only slim vs 무압축 baseline은 slim attention 경로(gather K/V)의
커널 드리프트를 문다(gen_coc 12/500 상이). **q25 vs m25 직접 비교(둘 다 slim)는 완전히
깨끗**하고, 이것이 주 판정이다.

### 사전 등록 게이트

- **G0 빌드 무결성**: 제거 파라미터가 위 표의 값과 정확히 일치; 반대 축·VLM·KV kept = 전부;
  q25 ∪ m25 의 kept set이 `slim_expert_znorm_r25`의 `slim_meta.json`과 bit-identical;
  expert arm 사이 gen_coc가 500/500 동일(expert는 CoC에 영향 못 줌).
- **G1 주 판정 (val500, paired 중앙값 ΔminADE@6 + 부트스트랩 CI, 평균 병기, Wilcoxon)**:
  Δ(q25 − m25)의 CI가 0 위 → **H1 채택: 축이 파라미터 수를 이긴다**. 0 아래 → H0.
  0 포함 → n=500으로 미결(중앙값 CI 폭 ≈ ±0.02–0.03).
- **G2 가산성 (val500)**: Δ(both25) − [Δ(q25) + Δ(m25)] 의 CI. 0 포함이면 두 축은 독립.
- **G3 스케일링**: m50 vs m25 (MLP가 40% expert-MLP 제거에서도 공짜인가), q50 vs q25.
- **G4 (m_pm 포함 시)**: Δ(q25 − m_pm)의 CI가 0 위 → 파라미터 수를 맞춰도 Q가 더 비쌈 =
  축 효과가 본질적.
- 세트 간 일관성: test500·OOD-val에서 G1 부호가 유지되는지(부호 뒤집히면 보고에 명시).

### 실행 중 변경 (2026-08-28 16:19 KST, 사용자 결정): 50% arm은 Blackwell

Ada 4장(7번이 비어 워커 추가)으로는 완료가 21:15 KST라, **q50·m50을 Blackwell 0–3으로**
옮겼다. 주 판정(G1·G2·G4)에 걸린 q25·m25·m_pm·both25·baseline_ada는 전부 Ada에 남는다.
Blackwell에는 per-sample 배열이 있는 baseline이 없어(`baseline_test/ood`는 minADE@8만)
**`baseline_bw_ps_{indist,test,oodval}`를 새로 측정**하고, q50·m50의 Δ는 이 baseline 대비로
잰다. 따라서 **G3는 Δ-대-Δ 비교**가 된다: 클립별 (q50 − base_bw) − (q25 − base_ada).
아키텍처 편향은 실측 ≈0(+0.0000/+0.0001, p=0.82)이므로 잡음만 늘고 편향은 없다.
q50 − m50(G3x)은 같은 아키텍처라 직접 대비 그대로. 논문 표의 q50/m50 행에는 "Blackwell,
Blackwell baseline 대비" 각주. 예상 완료 ~19:05 KST.

### 결과의 쓰임

H1이 서면 expert 예산은 MLP에 쓰는 것이 맞다: `dualexp_u40_e25`의 expert 절반(both25,
−532M)을 **m50(−913M, Q 무접촉)** 로 바꾸면 더 많이 제거하면서 비용은 낮을 수 있다 — 이는
별도 계획(dual + expert MLP-only)으로 검증한다. H0이면 e10/e15 결론을 "소량에서만 공짜"로
축소한다.

## 파일

| 파일 | 변경 |
|---|---|
| `experiments/head_analysis/make_slim.py` | 분기 추가: `^expert(q|m)_(u|c)(\d+)$` — `q`/`m`은 축, `u<N>`은 비율 %, `c<N>`은 층당 개수(m_pm용). 반대 축은 identity. **공유 파일 → 동료 세션에 공지** |
| `experiments/evaluation/paper_numbers.py` | ARMS에 `expertq25`/`expertm25`/`expertq50`/`expertm50`/(`expertm_pm`) 추가. **공유 파일** |
| `experiments/evaluation/analyze_expert_axis.py` (신규) | G0–G4 판정, 파라미터 수 표, `outputs/expert_axis_analysis/{metrics.json,summary.txt,plots/}` |
| `experiments/head_analysis/expert_axis_report_template.html` (신규) | 보고서 템플릿 → `reports/evaluation/2026-08-28_expert-axis.html` |

## 실행 · 자원

- 빌드: `make_slim.py --config expertq_u25 --importance importance_stepexp_znorm --no-state
  --out outputs/slim_expertq_u25` 등, arm당 ~15–30분(모델 로드 + 수술 + 검증, 상태 저장 없음).
- 평가: arm당 1,262클립 ≈ 3.2 GPU-h (indist/test 8 s/clip, OOD-val 13 s/clip).
  4 arm = 12.7 GPU-h, 5 arm(m_pm 포함) = 15.9 GPU-h. 지금 Ada 4·5·6 유휴, 7은 타인의
  alpasim 렌더러(root, 14 h 경과) → 3장 기준 **약 5.5 h** (7번이 비면 ~4 h).
- 디스크: `--no-state`라 체크포인트 무게 없음(/mnt/nvme1n1 167 GB 여유, 98%).

## 결과 (2026-08-28, 19:16 KST 완료)

보고서 `reports/evaluation/2026-08-28_expert-axis.html`, 분석 `outputs/expert_axis_analysis/`.
G0 PASS (제거 파라미터 6 arm 정확 일치, q25∪m25 == both25 bit-identical, nesting, 같은 아키텍처의
expert arm 간 gen_coc 1.000). minADE@6 paired 중앙값 [95% CI], 자기 아키텍처 baseline 대비:

| arm | 제거 | val500 | test500 | OOD-val 262 |
|---|---|---|---|---|
| q25 (Q 4/16) | 75.5M | **+0.0222 [+0.0103, +0.0372]\*** | +0.0286 [+0.0182, +0.0432]\* | +0.0192 [+0.0040, +0.0370]\* |
| m_pm (MLP 341, 파라미터 매칭) | 75.4M | −0.0000 [−0.0001, +0.0001] | −0.0000 [−0.0001, +0.0001] | −0.0000 [−0.0002, +0.0000] |
| m25 (MLP 2064) | 456.5M | +0.0000 [−0.0001, +0.0001] | +0.0000 [−0.0001, +0.0001] | −0.0000 [−0.0002, +0.0001] |
| q50 (Q 8/16) [BW] | 151.0M | **+0.1160 [+0.0848, +0.1830]\*** | +0.1394 [+0.1008, +0.1789]\* | +0.1645 [+0.1036, +0.2231]\* |
| m50 (MLP 4128) [BW] | 913.0M | +0.0001 [−0.0001, +0.0003] | −0.0001 [−0.0002, +0.0001] | +0.0001 [−0.0002, +0.0003] |

| 게이트 | val500 | test500 | OOD-val | 판정 |
|---|---|---|---|---|
| G1 q25 − m25 | +0.0205 [+0.0103, +0.0380]\* | +0.0282 [+0.0174, +0.0436]\* | +0.0188 [+0.0039, +0.0369]\* | **H1 채택** |
| G4 q25 − m_pm | +0.0217 [+0.0103, +0.0375]\* | +0.0283 [+0.0179, +0.0437]\* | +0.0193 [+0.0053, +0.0375]\* | **축 효과가 본질적** |
| G2 both25 − (q25+m25) | +0.0001 [−0.0001, +0.0002] | — | — | 완전 가산: both25의 비용은 전부 Q head |
| G3 Δ(q50) − Δ(q25) | +0.0997 [+0.0686, +0.1398]\* | +0.0867\* | +0.1281\* | Q 비용은 헤드 수에 초선형 (2배 → 5배) |
| G3 Δ(m50) − Δ(m25) | +0.0000 [−0.0001, +0.0002] | −0.0001 | +0.0002 (0.2 mm) | MLP는 50%에서도 공짜 |
| G3x q50 − m50 | +0.1171\* | +0.1399\* | +0.1664\* | 같은 방향 |

기전: 1차 중요도 질량(|Σ_s|)에서 하위 50% 유닛이 든 몫이 **MLP 0.44% vs Q head 24.96%**
(하위 25%: 0.16% vs 9.24%). MLP 중요도는 소수 채널에 몰려 있어 하위권은 불활성이고, Q head는
질량이 퍼져 있는 데다 스텝마다 다른 헤드를 쓴다(step 0 vs 9 랭킹 상관 0.14).

**결론**: expert의 프루닝 비용은 전부 Q head 축에서 나온다. MLP-only 50%(−913M, 전체 8.2%)는
세 세트에서 무압축과 구별되지 않는다 — both25(−532M, +0.023)보다 더 많이 지우고 더 싸다.
후속 후보: `dualexp`의 expert 절반을 m50으로 바꾼 dual + expert-MLP-only(−3.57B, 32.2%).
개루프 판정이며 폐루프는 재지 않았다.
