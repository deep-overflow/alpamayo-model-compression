# hard100 폐루프 평가 — 어려운 씬에서도 dual이 최고인가

## 가설

지금까지의 모든 폐루프 결론은 `public_2601`의 첫 150씬에서 나왔다. 그런데 그 표본은 자기가
뽑혀 나온 스위트보다 **유의하게 쉽다** — sangoh님의 913씬 비압축 baseline 런
(`eval2601_a1_5`)으로 재면 첫 150씬 0.742 vs 나머지 763씬 0.660 (Mann-Whitney p=0.039),
offroad 9.3% vs 13.0%, **과실 충돌 2.0% vs 5.5%**.

그래서 두 가지가 미지수다.

- **H1 (순위 보존).** 150씬에서 `slim_dual_u40_v2`는 baseline 대비 +0.079 (p<0.001)로 15개 arm
  중 1위였다. 이 이득이 더 가혹한 씬에서도 유지되는가, 아니면 쉬운 표본의 산물인가.
- **H2 (안전 프로파일).** 150씬에서 dual의 과실 충돌은 2.3% (7/300)로 baseline 4.0%보다 낮았지만
  Fisher p=0.351로 유의하지 않았다 — 바닥 효과로 검정력이 없었다. hard100은 baseline이 5.5%대일
  것으로 예상되므로 **처음으로 충돌 축에 검정력이 생긴다**. dual의 낮은 충돌률이 실재하는가.

H2가 이 실험의 진짜 이유다. H1은 확인 사살에 가깝다.

## 씬 집합

`experiments/head_analysis/make_hard_suite.py`가 생성한 `public_2601_hard100` (100씬).

- 모집단은 `public_2601` 913씬에서 **매트릭스 첫 150씬을 제외**한 나머지. 스크립트가
  `--exclude-runs`로 실제 평가된 씬 집합과 정의를 교차 확인하고, 교집합이 0이 아니면 중단한다.
  검증 완료: 교집합 **0개**.
- GT 주행 5 m 미만인 8개는 추가 제외 (`progress_score`가 1.0으로 덮어써져 정보량이 0).
  후보 풀 755개.
- 랭킹은 `hard_score = z(v_mean) + z(yaw_total_deg)`, usdz의 에고 GT 궤적만으로 계산 —
  모델을 전혀 돌리지 않는다. 상위 100개를 취한다 (스위트 87퍼센타일부터).
- 중앙값: v_mean 12.0 m/s, 누적 선회 78°, 경로 241 m (스위트 중앙값 8.7 / 3 / 174).
- uuid는 `public_2601` 소속 100/100, 로컬 usdz 100/100 존재 → **다운로드 0**.

### 예상 baseline 점수는 이미 안다

sangoh님 913씬 런에서 이 100씬의 실측 baseline은 **0.499** (1 rollout 기준). 이건 사전 정보이지
검정 대상이 아니다 — 우리 런의 baseline이 0.45~0.55를 크게 벗어나면 셋업을 의심해야 한다는
sanity check로 쓴다.

## 사전등록 게이트

| 게이트 | 기준 | 해석 |
|---|---|---|
| **S1** (sanity) | baseline 점수 0.42–0.58 | 벗어나면 셋업 이상 — 결과 해석 전에 원인 규명 |
| **G1** (H1) | dual − baseline ≥ 0, Wilcoxon | 음수이고 p<0.05면 "dual이 최고"는 쉬운 표본의 산물 |
| **G2** (H2) | 과실 충돌 OR(dual/baseline) ≤ 1.0 | 1을 넘으면 dual의 안전 우위는 150씬 바닥효과의 산물 |
| **G3** (보조) | `analyze_longitudinal.py` 연속 지표에서 dual이 baseline 대비 악화 없음 | 충돌 카운트의 검정력 부족을 보완 |

**무엇이 가설을 죽이는가**: G1이 뒤집히면 (dual이 hard100에서 baseline보다 유의하게 나쁨)
"dual_u40_v2가 최고 폐루프 config"라는 주장은 철회해야 한다. G2가 뒤집히면 리포트
`2026-09-03_difficulty-stratified-arms.html`의 "우리 arm은 충돌을 피해서 점수를 번다" 서술을
수정해야 한다.

## 검정력

150씬에서 씬 단위 페어드 델타의 σ ≈ 0.30이고, 정적으로 고른 어려운 부분집합에서는 σ ≈ 0.335
(1.12배)였다. N=100이면 1.96·0.335/√100 = **0.066**까지 분해한다. dual의 150씬 효과 +0.079는
검출 가능하고, 어려운 씬에서 효과가 커진다면(표 B에서 어려움 계층 +0.175) 여유가 있다.

충돌은 다르다. baseline 5.5% × 200 rollout ≈ 11건, dual이 절반이면 5–6건. Fisher로는
**유의에 못 미친다**. G2는 방향 확인용이고, 결론은 G3의 연속 대리지표에 의존한다. 이 한계는
사전에 인정하고 시작한다 — 충돌 카운트로 안전성을 논증하려면 N을 몇 배로 키워야 한다.

## 실행

arm 순서는 **baseline 먼저**(기준이 없으면 dual 결과를 읽을 수 없다), 그 다음 `dual`.
Ada 4–7 네 장, 씬당 2 rollout, `DRIVER_OMP_THREADS=8`.

```bash
# 0) 씬 목록 (겹침 0 재확인 포함)
python experiments/head_analysis/make_hard_suite.py \
    --out outputs/scene_difficulty/hard100_suite.csv \
    --exclude-runs /home/cvlab21/project/chan/alpasim-runs/m2601_merged_baseline

# 1) 실행 전 점검 — 컨테이너를 띄우지 않고 샤드 분할만 확인
DRY_RUN=1 SUITE=public_2601_hard100 PREFIX=h100_ \
SCENES_CSV=$PWD/outputs/scene_difficulty/hard100_suite.csv \
  bash experiments/head_analysis/launch_alpasim_shards.sh baseline 100 2 "4 5 6 7"

# 2) baseline (~5.2 h)
SUITE=public_2601_hard100 PREFIX=h100_ DRIVER_OMP_THREADS=8 \
SCENES_CSV=$PWD/outputs/scene_difficulty/hard100_suite.csv \
  bash experiments/head_analysis/launch_alpasim_shards.sh baseline 100 2 "4 5 6 7"

# --shards / --out 은 --runs-root 기준 *이름*이지 경로가 아니다
python experiments/head_analysis/merge_alpasim_shards.py \
    --runs-root /home/cvlab21/project/chan/alpasim-runs \
    --shards h100_baseline_sh0 h100_baseline_sh1 h100_baseline_sh2 h100_baseline_sh3 \
    --out h100_merged_baseline --expect-scenes 100

# 3) dual (~4.8 h) — 위와 동일, config만 slim_dual_u40_v2
```

### 예약 실행 (2026-09-04 00:38 KST 설정됨)

위 순서를 `/home/cvlab21/project/chan/alpasim-runs/hard100/run_hard100.sh`가 대신 수행한다.
그 디렉터리는 리포 밖(공유 결과 마운트)에 있고 런처와 씬 목록 사본을 함께 두므로, 워크트리가
지워지거나 브랜치가 바뀌어도 영향받지 않는다.

Ada 4–7에는 사용자 작업이 두 개 줄 서 있다 — 진행 중인 `calib-draw-variance` 개루프 평가와,
그 다음 `dual` 세션의 실험 하나. **단순히 "GPU가 비면 시작"하면 두 실험 사이의 빈 틈에 잘못
발사되므로**, 트리거를 명시적 바통으로 만들었다:

1. `hard100/GO` 파일이 생길 때까지 대기 (`dual` 세션이 자기 실험을 마치고 touch)
2. 그 다음 Ada 4–7이 **3분 연속** 2 GB 미만인지 확인 — GO가 일러도 남의 작업을 밟지 않는다
3. baseline → 병합 → `slim_dual_u40_v2` → 병합. 앞 단계가 실패하면 뒤는 실행하지 않는다

프로세스는 `setsid`로 분리돼 있어 (PPID 1) 요청한 세션이 사라져도 살아남는다.
최대 대기 36시간, 그 안에 GO가 없으면 아무것도 실행하지 않고 종료한다.

| | |
|---|---|
| 시작 신호 | `touch /home/cvlab21/project/chan/alpasim-runs/hard100/GO` |
| 취소 | `touch .../hard100/ABORT` 또는 `kill $(cat .../hard100/scheduler.pid)` |
| 진행 | `tail -f .../hard100/run.log` |

`PREFIX=h100_`이라 로그 디렉터리가 매트릭스(`m2601_`)와 섞이지 않는다. 드라이버
`/mnt/nvme1n1/ad_vla/data/alpasim/drivers/slim_dual_u40_v2`는 이미 있고 `outputs/`와 하드링크로
inode를 공유하므로 추가 용량은 0이다 (디스크 여유 481 GB).

비용: config당 ~5 h, 두 arm 합 **~10 h**. 개루프로 환산하면 같은 시간에 2,533클립 × 4 config를
돌릴 수 있는 양이라, arm을 더 추가하기 전에 이 둘의 결과를 보고 결정한다.

### baseline을 다시 안 돌리는 선택지

sangoh님 913씬 런이 이 100씬을 이미 포함한다. 그걸 baseline으로 쓰면 5.2 h를 아끼지만
(a) 씬당 1 rollout이라 노이즈가 2배, (b) 다른 시점·다른 런이라 스택 드리프트를 배제할 수 없다.
겹치는 150씬에서 우리 런과 Wilcoxon p=0.697로 일치하므로 **교차 검증용으로는 충분**하다.
헤드라인은 우리 baseline으로 내고, sangoh님 런은 재현성 확인에 쓴다.

## 분석

`analyze_alpasim.py`(alpasim venv에서), `analyze_collisions.py`, `analyze_longitudinal.py`를
`h100_merged_*`에 대해 실행. 그 다음 `analyze_difficulty_strat.py`에 hard100 arm을 추가해
150씬 결과와 나란히 놓는다 — **합치지 않고 별도 스플릿으로 보고한다.** 두 집합은 난이도가
다르고, 150씬 쪽은 스위트 대비 쉬운 표본이다.

## 열린 문제

- 정적 난이도는 과실 충돌과 무관하다 (ρ=+0.008). hard100은 progress·차선유지를 스트레스하지
  충돌 회피를 스트레스하지 않는다. H2에 대한 검정력이 생기는 이유는 "충돌이 많은 씬을 골라서"가
  아니라 단지 "전반적으로 더 어려운 씬이라 baseline 충돌률이 높아서"다. 충돌을 직접 겨냥한
  씬 선별 기준은 아직 없다.
- 정적 점수는 80퍼센타일 위에서 포화한다 (80–90% 0.496, 90–100% 0.511). N을 100 이상으로 키울
  때는 상위 100을 넘겨 뽑지 말고 `--band 80 100`으로 183씬 밴드에서 뽑아야 난이도가 유지된다.
