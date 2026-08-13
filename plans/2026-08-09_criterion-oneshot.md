# 기준 1요인 비교: CoC Taylor vs J-lens (2026-08-09)

## 가설

폭 프루닝의 "추론 보호" 항을 **참조 CoC 텍스트 없이도** 대체할 수 있는가.

두 config는 `max(rank I_traj, rank X)` 구조를 공유하고 `X` 하나만 다르다.

| arm | X | 라벨 |
|---|---|---|
| `dual_u40_v2` | `I_CoC` — CoC NLL Taylor | 필요 |
| `j_traj_u40_v2` | `J` — J-lens 점수 | **불필요** |

**H0**: 두 arm의 개루프 성능 차이가 없다 → J가 CoC 라벨을 대체한다.

## 왜 지금 이 설계인가

기존 `j_traj_full_r20` vs `dual_uniform` 비교는 **세 요인이 동시에** 달랐다:
criterion(J vs I_CoC), VLM 비율(20% vs 39.9%) + KV 그룹 1개 드롭, expert(magnitude
early40/late10 vs 미개입). 그래서 폐루프 격차(0.451 vs 0.610)를 귀속시킬 수 없었다.

여기서는 criterion을 제외한 전부를 `dual_uniform` 쪽으로 고정한다.

또한 기존 비교는 **캘리브레이션 출처도 어긋나 있었다** — `I_CoC`는 `importance_v1`(구 50클립),
`J`는 `jlens_coc`(8클립, ~25% pick churn). 이번에는 양쪽 모두 `calib_100`에서 나온
`importance_v2` / `jlens_v2`(각 100클립, 동일 model revision)를 쓴다. 남는 차이는 수식뿐이다.

개루프 r20/r30에서는 이 비교가 이미 무승부다(`jsweep32_summary`:
`j_traj_r20 − cocsafe_r20` dADE +0.006 / dNLL −0.000 p=0.45). 새 정보는 **39.9%라는 더
공격적인 비율**과 **baseline 평가 3개 세트 전체(2,533 클립)** 라는 검정력이다.

## 설계

### 공통 (두 arm 동일)

- 기준 구조: `max(rank_norm(I_traj), rank_norm(X))`, 레이어 내부 랭크 정규화
- 배분: **uniform, target = 0.3985632694** (`run_grid.allocations()`가 `slim_integrated_mag`의
  실현 예산에서 역산한 값 — 0.40으로 반올림하면 MLP가 레이어당 17채널 어긋난다)
  → Q head 13/32 제거(19 유지), MLP 4898/12288 제거(7390 유지), 36층 전부 동일
- **expert 미개입** (Q 16/16, MLP 8256/8256)
- **KV 그룹 드롭 없음**, kv-only 레이어 없음
- 중요도: `importance_v2` (calib_100, 100클립) / J: `jlens_v2` (calib_100, 100클립)
- 예상 제거: 2.657 B / 11.078 B = **24.0%**

마스크 생성은 `run_grid.allocations()` + `mask_lib.select_mask_ratios()`를 그대로 재사용한다
(수식 중복 없음). `--importance importance_v1`로 부르면 `dual_u40_v2`는 shipped
`slim_dual_uniform`과 bit-identical해야 한다 — 빌드 시 검증한다.

### 평가

baseline이 평가된 3개 세트 전부, 동일 프로토콜(`run_baseline.py`, k=8, seed 42,
clip_id 유래 시드, 페어드):

| set | n | 조건 |
|---|---:|---|
| in-dist val | 500 | own rollout |
| in-dist test | 500 | own rollout |
| OOD | 1,533 | own rollout + GT-CoC teacher-forced |

arm당 2,533 클립 × 2 arm.

## 사전 등록 게이트

**Gate C1 (주 판정)** — in-dist val+test 1,000 클립에서 두 arm의 페어드 minADE 차이:
`j_traj_u40_v2 − dual_u40_v2`의 중앙값이 **±0.05 m 이내이고 Wilcoxon p > 0.05**면
"J가 CoC 라벨을 대체한다"로 판정. (0.05 m은 이 프로토콜의 분해능 하한 —
깊이 ablation의 Gate D와 동일 기준.)

**Gate C2 (추론 채널)** — OOD 1,533에서 `nll_gtcoc` 페어드 차이. J-lens는 정의상
언어로 읽히는 write만 재므로, 여기서 뒤지면 "라벨-프리의 대가"가 추론 쪽에 있다는 뜻이다.

**Gate C3 (붕괴 없음)** — 두 arm 모두 CoC degeneracy < 0.05 (baseline 0.006~0.008).
어느 한쪽이 넘으면 그 arm은 40%에서 붕괴한 것이고 C1은 무효.

**부정 결과의 의미**: C1에서 `j_traj`가 유의하게 나쁘면, 20%에서 무승부였던 것이
39.9%에서 갈린 것이므로 "J는 저압축에서만 CoC를 대체한다"는 조건부 결론이 된다.

## 실행

```bash
# 1) 체크포인트 2개 (~1.5 h each, 17 GB each, 병렬)
python make_slim.py --config dual_u40_v2   --importance importance_v2 --gpu 5 \
    --out outputs/slim_dual_u40_v2
python make_slim.py --config j_traj_u40_v2 --importance importance_v2 --jlens jlens_v2 \
    --gpu 6 --out outputs/slim_jtraj_u40_v2

# 2) 개루프 3세트 x 2 arm (run_baseline.py --model <ckpt dir>)
# 3) analyze_baseline.py --compare 로 페어드 델타
```

## 리스크

1. **GPU 아키텍처 혼입** — baseline은 Blackwell(0–3)에서 돌았는데 지금 비어 있는 카드는
   Ada(5–7)뿐이다. 결정성은 아키텍처 내부에서만 성립한다(관측 예: 0.286 vs 0.291).
   두 arm을 모두 Ada에 올리면 **주 판정(C1, arm 간 비교)은 깨끗하다**. vs-baseline 수치는
   부차적 맥락으로만 쓴다. 착수 직후 baseline 20클립을 Ada에서 재실행해 아키텍처 효과의
   크기를 실측하고, 무시할 수 없으면 baseline arm도 Ada에서 재실행한다.
2. **디스크** — 17 GB × 2. nvme 1.3 TB 여유이므로 문제 없음.
3. **`make_slim.smoke()`가 `load_physical_aiavdataset`을 호출** — val 카메라 청크는
   2026-08 초에 삭제됐으므로 재다운로드를 시도한다. 캐시(`sample_cache`) 경로로 교체한다.
4. **model revision 미고정** — `make_slim.py`는 `from_pretrained`에 revision을 넘기지 않는다.
   평가 스크립트는 고정하므로 맞춰서 고정한다.
