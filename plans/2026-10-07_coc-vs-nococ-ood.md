# CoC가 액션에 도움이 되는가 — OOD train + val (2026-10-07)

## 가설

무압축 Alpamayo-1.5-10B에서, 모델이 CoC를 먼저 쓰고 액션을 생성할 때(릴리스 경로)가
CoC 없이 액션을 생성할 때보다 개루프 궤적이 더 정확하다.

## 조건 (모두 같은 클립 · 같은 디노이징 시드 → 페어드)

expert는 VLM의 KV 캐시를 읽으므로, 조건은 "캐시에 무엇이 들어 있는가"로만 다르다.

| 조건 | VLM 시퀀스 | 의미 |
|---|---|---|
| `rollout` | prompt(`…<|cot_start|>`) + 모델이 쓴 CoC + `<|cot_end|>` + `<|traj_future_start|>` | 릴리스 추론 (기준) |
| `empty` | prompt(`…<|cot_start|>`) + `<|cot_end|>` + `<|traj_future_start|>` | **주 조건.** 같은 프롬프트, CoC 0토큰. 프루닝된 arm이 빈 출력으로 붕괴할 때와 같은 캐시 |
| `skip` | prompt에서 `<|cot_start|>` 제거 + `<|traj_future_start|>` | CoC 구획 자체가 없음 |
| `trajprompt` | 지시문을 "output the future trajectory."로 교체, assistant는 `<|traj_future_start|>`로 시작 | 탐색용. 추론을 요구하지도 않음. 학습 분포 안인지 확인 불가(이 커밋의 `create_message`는 `use_nav_prompt`를 받기만 하고 쓰지 않음) |

GT CoC teacher-forcing 조건은 2026-08-19 프로토콜에 따라 새로 돌리지 않는다. 참고값이
필요하면 저장된 `baseline_ada_ood`의 `minADE_tf`(@8)를 인용한다.

## 설정

- 세트: `outputs/eval_sets/ood.parquet` 1,533 클립 = train 1,271 + val 262. 둘 다 zero-shot.
- 모델: baseline (rev `7aba8293`), Ada 4–7에서 4-way strided shard, k=8, seed 42,
  `clip_seed(seed, clip_id) + k`. 디노이징 노이즈는 시드만으로 정해지므로 조건 간 동일.
- 지표: minADE@6 / minFDE@6 **평균 우선**, 중앙값·Wilcoxon 병기. 지평 1.6 / 3.2 / 6.4 s.
  보조: 샘플 평균 ADE(다양성과 분리), 버킷·클러스터·split별, CoC 길이·퇴화 여부별.
- 교란 점검: rollout 조건은 같은 프로세스에서 다시 돌린다(저장본은 per-sample 배열이
  없어 @6 축소가 불가). 저장된 `baseline_ada_ood`의 `minADE_rollout`(@8)과 비트 일치하는지로
  재현성을 확인한다.

## 사전 등록 게이트

`Δ = minADE@6(no-CoC) − minADE@6(rollout)`, 주 조건 `empty`, 부트스트랩 95% CI(평균).

- **G1 (CoC가 돕는다)**: train·val **둘 다**에서 CI 하한 > 0.
- **G2 (실질적 크기)**: 전체 1,533에서 Δ 평균 ≥ 0.05 m (이 레포의 결정 임계).
- **G3 (조건 정의에 강건)**: `skip`의 Δ가 `empty`와 같은 부호이고 CI가 0을 배제.
- G1 통과·G2 실패 → "돕지만 작다". G1 실패 → "개루프 궤적 정확도에는 차이 없음".
- 하위집단(버킷·클러스터·CoC 길이)은 탐색용이며 게이트가 아니다.

## 산출물

`outputs/coc_ablation_ood/` (config.json, metrics.json, summary.txt, plots/),
리포트 `reports/evaluation/2026-10-07_coc-vs-nococ-ood.html`.

## 한계 (미리 적음)

- 개루프다. CoC의 폐루프 가치(재계획 일관성 등)는 측정하지 않는다.
- "CoC 없음"은 학습 때 본 적 없는 시퀀스일 수 있다. no-CoC가 나쁘면 CoC의 내용이
  아니라 분포 이탈 때문일 수 있다 — 그래서 조건을 셋 두고, 저장된 GT-CoC 값과 함께 읽는다.

## 수정 (2026-10-07, 4클립 스모크 직후 · 전체 결과를 보기 전)

`trajprompt`를 "탐색용"으로 적은 것은 틀렸다. 8클립에서 다음 토큰 분포를 확인하니
지시문이 "output the future trajectory."일 때 모델은 assistant 턴 첫 토큰으로
`<|traj_future_start|>`를 **p=1.0**으로 낸다(8/8; CoC 지시문에서는 `<|cot_start|>`가 p=1.0).
즉 이것은 모델이 학습한 네이티브 no-CoC 모드다. 반대로 `empty`는 강제 토큰 NLL이 6–8,
`skip`은 34–38로 모델이 스스로는 가지 않는 시퀀스다.

- 게이트 G1–G3는 등록한 대로 `empty` 기준으로 판정한다(바꾸지 않는다).
- 단 해석은 둘로 나눠 적는다: **`trajprompt` − `rollout`** = "CoC 모드 대 네이티브 no-CoC
  모드" (분포 안끼리의 비교, 사용자 질문에 가장 직접 대응), **`empty` − `rollout`** =
  "같은 프롬프트에서 CoC 토큰만 제거" (분포 밖 교란 포함).
- 같은 G1/G2 규칙을 `trajprompt`에도 적용해 G1'/G2'로 병기한다.
