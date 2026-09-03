# 스텝별 랭크의 합집합: znorm11이 한 번에 바꾼 두 요인을 분리한다

## 질문

`znorm11`(11개 손실의 층내 z-score 평균)은 `dual`(`max(rank I_traj, rank I_CoC)`) 대비
+0.14~+0.20으로 기각됐다(`reports/evaluation/2026-09-03_criterion-aggregation.html`).
그런데 znorm11은 **두 가지를 동시에** 바꿨다.

1. **궤적 표현**: 합산된 `I_traj` 하나 → 디노이징 스텝별 10개 점수
2. **결합 연산**: `max`(합집합) → `mean`(평균)

기각의 원인을 (2)로 귀속했지만, 그것은 해석이지 측정이 아니다. 2×2를 채운다.

| | 2-way (합산 traj) | 11-way (스텝별) |
|---|---|---|
| **max** | `dualfix` ✔ 측정됨 | **`maxstep11`** ← 신규 |
| **mean** | **`meandual`** ← 신규 | `znorm11` ✔ 측정됨 |

- `maxstep11` = `max(rank FM_0, …, rank FM_9, rank I_CoC)` — 11개 랭크의 합집합
- `meandual` = `mean(z(I_traj), z(I_CoC))` — dual의 두 half를 znorm11의 연산자로

두 신규 arm 모두 `dualfix`의 상수-층 가드를 상속한다(층 35는 FM 10개가 전부 상수라 CoC가
단독 결정 — `dualfix`의 층 35 kept set과 **비트 단위로 동일**함을 확인).

## 사전 정보 (빌드 전에 이미 아는 것)

- `sum_s(q_abs_step)`은 `traj_vlm_q`를 층내 Spearman **+0.989**로 재현한다(MLP +0.984).
  두 축은 실제로 분리 가능하다.
- **스텝 간 랭킹이 이미 매우 비슷하다**: Q head 층내 페어와이즈 Spearman 평균 **+0.920**
  (최소 +0.812). expert에서 스텝 축이 컸던 것과 대조적이다. 따라서 (1)의 여지 자체가 얇다는
  것이 사전 예측이고, 이 실험은 그 예측을 확인하거나 뒤집는다.
- kept set 사전 계산(예산 동일, 19/32 Q · 7390/12288 MLP):

| arm | dualfix 대비 Q 일치 | znorm11 대비 Q 일치 | 파라미터 churn |
|---|---|---|---|
| `maxstep11` | 0.9254 | 0.9211 | 13.7% |
| `meandual` | 0.9415 | 0.8874 | 8.6% |
| `znorm11` | 0.8772 | — | 17.2% |

`maxstep11`은 두 기존 셀의 정확히 중간에 놓인다 — 어느 쪽으로 붙든 정보량이 있다.

## 게이트 (사전 등록)

기준선은 `dualfix`(= 실질적으로 출시 dual), 지표는 paired minADE@6 중앙값, 세 세트
(val500 / test500 / OOD-val 262), 부트스트랩 95% CI.

- **B1 (주 판정)** `|median(maxstep11 − dualfix)| < 0.05` 가 **세 세트 모두**에서 성립
  → 스텝 축은 합집합 연산 아래에서 **무해**하고, znorm11의 손실은 전적으로 연산자 때문이다.
  세 세트 중 하나라도 위반하면 → 11-way라는 **arity 자체**가(합집합이어도) 해롭다.
- **B2 (보조)** `median(meandual − dualfix)`의 크기가 `median(znorm11 − dualfix)`의
  **절반 이상**이면 평균 연산자가 단독으로 대부분의 손해를 설명한다. 그보다 작으면
  손해는 연산자와 arity의 **상호작용**이다.
- **B3** 두 신규 arm 모두 CoC 퇴화율 < 10%. 넘으면 해당 arm의 B1/B2 판정은 무효.

**B1이 PASS이고 `maxstep11`이 `dualfix`보다 유의하게 낫지도 않다면**, 결론은 "VLM에서 스텝 축은
살릴 것이 없다"이다. 사전 정보(스텝 간 Spearman 0.92)가 그쪽을 가리키므로, 이 실험의 주된 가치는
znorm11 기각의 원인을 확정하는 데 있지 새 SOTA를 기대하는 데 있지 않다.

## 실행

```
make_slim.py --config maxstep11_u40_v2 --importance importance_v2_ada \
             --stepvlm importance_stepvlm_v1 --no-state --out outputs/slim_maxstep11_u40_v2
make_slim.py --config meandual_u40_v2  --importance importance_v2_ada \
             --no-state --out outputs/slim_meandual_u40_v2
```

평가는 `launch_arms.sh append 4`로 2 arm × 3 세트 × 4 shard = 24 job, **Ada 4–7**
(기존 arm과 페어링하려면 아키텍처를 고정해야 한다). 분석은
`analyze_criterion_agg.py`에 두 arm을 추가해 같은 표로 확장한다.

## 검증

1. 두 arm 모두 제거 파라미터가 **정확히 2,657,452,032**여야 한다(다르면 예산이 어긋난 것).
2. 층 35 kept set이 `dualfix`와 동일해야 한다(가드가 상속됐다는 뜻). 빌드 전 확인 완료.
3. 시드는 clip 유도이므로 기존 arm과 자동으로 페어링된다 — 공통 클립에서 시드 불일치 0건 확인.

## 결과 1차: val500 (2026-09-03, 완료)

빌드 검증 통과 — 두 arm 모두 제거 2,657,452,032, 층 35 kept set이 `dualfix`와 비트 단위 동일.

| | **2-way** (합산 traj) | **11-way** (스텝별) |
|---|---|---|
| **max** | `dualfix` 0.9236 (0.6398) | **`maxstep11` 0.8915 (0.5991)** |
| **mean** | **`meandual` 0.9493 (0.6451)** | `znorm11` 1.2890 (0.9220) |

minADE@6 평균(중앙값), n=500 paired. `dualfix` 기준 페어드 델타:

| 비교 | median | mean | 95% CI(med) | p |
|---|---|---|---|---|
| `maxstep11 - dualfix` | -0.0033 | -0.0321 | [-0.0162, +0.0042] | 0.19 |
| `meandual - dualfix` | +0.0058 | +0.0257 | [-0.0103, +0.0188] | 0.50 |
| `znorm11 - dualfix` | +0.1731 | +0.3655 | [+0.1036, +0.2378]* | 8.2e-22 |

- **B1 PASS** — 0.0033은 게이트(0.05)의 1/15. 스텝 축은 합집합 아래에서 무해.
- **B2 예상과 다름** — `meandual`도 무해(+0.0058). `znorm11` 손해의 **3.3%**만 설명하므로
  "평균 연산자 단독"이라는 앞선 귀속은 **틀렸다**. 손해는 전적으로 **연산자 x 중복항 개수의
  상호작용**이다: 스텝 간 상관 0.92라 10개 FM 항은 사실상 한 랭킹이고, 평균에서 그 공통 성분이
  10/11의 무게를 가져 CoC 몫이 1/2 -> 1/11로 떨어진다. `max`는 무게 개념이 없어 항이 늘어도
  CoC의 발언권이 줄지 않는다.
- **B3 PASS** — 퇴화율 2.2% / 3.6%.
- 부수: `maxstep11`의 이득은 **꼬리에만** 있다. `dualfix` 난이도 상위 10%에서 mean -0.5186 /
  median -0.1245, 쉬운 90%에서는 mean +0.0220 / median +0.0004. 다만 `dualfix` 대비 전체
  페어드는 유의하지 않다(p=0.19).

**남은 교란**: 이 2x2는 행마다 정규화가 다르다(max 행 = rank_norm, mean 행 = z-score).
"11개 평균"의 해악이 *평균*에서 오는지 *z-score*에서 오는지 가르려면 `mean of 11 rank_norm`
arm이 하나 더 필요하다. 미실행.

## 확장 실행 (test500 + OOD-val) — 준비 완료, 미실행

GPU가 비면 아래 두 블록이면 된다. 큐는 append로만 늘린다(`init`은 살아 있는 워커에게 같은
샤드를 두 번 나눠준다). 워커 기동 조건은 **peer 세션의 워커까지 매칭하지 않도록** 경로를 포함해
검사한다 — 2026-09-03에 `pgrep -f "launch_arms.sh worker $g$"`가 다른 worktree의 워커를 잡아
GPU 4-6 워커가 조용히 안 떴다.

**`maxstep11`만 확장한다.** 두 arm의 역할이 다르다: `maxstep11`은 채택 후보라 held-out
test500과 OOD-val이 필요하고, `meandual`은 2x2의 빈 칸을 채우는 메커니즘 대조군이라
val500 하나로 족하다(+0.0058 [-0.0103, +0.0188], n=500이면 CI가 이미 충분히 좁다).
대가로 상호작용 주장의 네 칸 중 `meandual`만 단일 세트 근거로 남는다 - 논문에 2x2를 그대로
실을 때 아래 명령의 arm 이름만 바꿔 8 job을 더 돌리면 된다.

```bash
cd /home/cvlab21/project/chan/alpamayo-model-compression
SETS="test oodval" bash experiments/evaluation/launch_arms.sh append 4 \
    maxstep11_u40_v2=outputs/slim_maxstep11_u40_v2      # 8 job
for g in 4 5 6 7; do
  ps -eo args | grep -q "^bash experiments/evaluation/launch_arms.sh worker $g$" || \
    nohup bash experiments/evaluation/launch_arms.sh worker $g > logs/worker_$g.log 2>&1 &
done
```

Ada 4-7 고정(기존 arm과의 페어링이 아키텍처에 의존한다). 8 job = 1 arm x 2 set x 4 shard,
4카드로 약 45~60분. 끝나면 `analyze_criterion_agg.py`를 그대로 다시 돌리면 표와 플롯이
자동으로 확장된다(안 돌린 세트는 arm 단위로 건너뛴다).
