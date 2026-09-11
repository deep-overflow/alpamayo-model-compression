# action 층화 캘리브레이션 추출 — 3규칙 × 3seed, test500

날짜: 2026-09-11. 브랜치: `worktree-worktree-action-strat-calib`. 상태: **승인됨, 1단계 진행 중**
(승인 2026-09-11 20:45 KST; 진행 기록은 문서 끝).

## 왜

캘리브레이션 추출(`make_eval_sets.py` / `make_calib_blocks.py`)은 country · platform_class ·
time_of_day · season · month · radar_config 6축을 그리디로 맞출 뿐 **action은 보지 않는다.**
그 상태에서 n=100 추출을 바꾸면 test500 minADE@6가 0.95–1.77로 흔들리고(SD 0.363,
`2026-09-03_calib-draw-variance`), 기전 후보 21개가 전부 기각됐다(`2026-09-07_draw-order`).

이 세션(2026-09-11)에서 저장된 레코드만으로 두 가지를 더 쟀다
(`outputs/calib_buckets/action_mix_2026-09-11.{parquet,py}`, 이 플랜의 §부록):

1. **추출 분산은 cruise 버킷의 현상이다.** 자연 추출 5개의 unpruned 대비 손해를 버킷별로
   나누면 cruise +0.736 (추출 간 SD 0.569), decel_stop +0.228 (0.112), accel +0.230 (0.043),
   turn +0.296 (0.089). 나쁜 추출 c/d/e는 cruise에서 +0.65/+1.38/+1.27인데 나머지 버킷은
   +0.23–0.40에 머문다.
2. **`tr` 계열은 버킷 균형이 아니었다.** 회복 풀은 (clip, t0) 행 기준 20%씩이지만
   `make_calib_blocks`가 5.1 s에 가장 가까운 창을 다시 고르면서 그 t0에서는 cruise 35–47%가
   됐다(자연 49–59%, test500 56%). tr을 구분하는 건 균형이 아니라 **난이도의 모양**이다 —
   느리고(v p90 16–19 vs 23–29 m/s), 세게 서고(감속 p90 2.3–2.9 vs 1.8–2.1 m/s²), 크게
   돈다(|회전| p90 46–66° vs 33–51°). 그런데도 cruise 손해가 +0.401 (SD 0.072)로 자연 추출보다
   작고 6–8배 안정적이며, accel에서는 +0.134 vs +0.230으로 낫다.
3. 자연 추출 안에서는 **버킷 조성이 결과를 설명하지 못한다** — 가장 다양한 nt_e(cruise 49%,
   turn_left 14%)가 최악(1.72), 가장 편중된 nt_b(59%)가 멀쩡(1.06). 4×4 감속×회전 커버리지도
   무상관.

그래서 "action 다양성이 캘리브레이션을 돕는가"는 아직 **한 번도 깨끗하게 측정된 적이 없다**:
균형 × 자연 난이도 칸이 비어 있고, tr은 균형이 아니었으며, 자연 추출의 조성 범위(cruise
49–59%)는 너무 좁아 검정이 안 된다. 이 플랜은 그 칸을 채운다.

6축 매칭은 버린다. 근거 둘:

- 매칭이 뭔가를 사줬다는 증거가 없다. 여덟 세트가 6축 전부 n=100 이산화 하한에 붙어
  있는데(draw-order §랜덤 대조) 결과는 0.83 벌어졌다. 랜덤 대조 `calib_rd100_a/b`는 캐시까지
  만들어 놓고 `dual`로 평가된 적이 없다.
- 그리디는 복제를 망친다. `calib_100 → nt_a → … → nt_e`는 같은 seed로 앞 세트를 빼고 다시
  그리디한 순차 계열이라 교환 가능하지 않고 결과가 순서를 따른다(ρ +0.943). 지금까지의
  "추출 SD 0.363"은 그래서 분산 추정으로 쓸 수 없다. 층화 **무작위** 추출은 seed만 바꾸면
  iid 복제가 나온다.

층화 안에서 무작위로 뽑으면 6축은 기대값에서 편향 없이 따라온다(랜덤 추출의 6축 weighted L1은
0.19–0.22로 그리디 0.03보다 크지만, 그 차이가 결과를 움직인다는 측정은 없다 — H1이 그것을
잰다).

## action 라벨은 무엇을 보는가

두 층위를 모두 계산해 저장한다. 둘 다 **GT egomotion**(카메라나 모델 출력이 아닌 실제
주행 기록)의 함수이고, 모델 예측은 어디에도 들어가지 않는다.

- **bucket5** (`make_train_set.bucket5`): GT 미래 경로 64점(10 Hz, 6.4 s)의 xy에서 속도와
  heading을 유한차분해 규칙으로 나눈다 — 끝속도 < 0.5 m/s 또는 초기의 절반 미만이면
  `decel_stop`, |순회전| ≥ 30°면 `turn_left`/`turn_right`, 끝속도 > 1.5×초기면 `accel`, 나머지
  `cruise`. 우선순위 stop > turn > accel > cruise. 평가 레코드의 `bucket`(`eval_lib.bucket`,
  turn 부호만 합침)과 같은 정의라 **캘리브레이션 층과 평가 층이 같은 자로 잰다.**
- **raw action** (`model.action_space.traj_to_action`): `I_traj`의 flow-matching 타깃 그
  자체 — history/future xyz·rot에서 unicycle (accel, curvature) (64, 2)를 만들고 config의
  `accel_mean/std = 0.0290/0.681`, `curvature_mean/std = 0.00027/0.0261`로 정규화한 것.
  `UnicycleAccelCurvatureActionSpace`는 순수 `nn.Module`이고 입력 넷이 전부 캐시 npz
  (`ego_history_xyz/rot`, `ego_future_xyz/rot`)에 있으므로 **모델 없이 CPU에서** 계산된다.
  클립당 기술량: mean|a|, max 감속, mean|κ|, max|κ| (물리 단위) 와 정규화 채널별 RMS.

**층화 단위는 bucket5**로 한다 — 평가 판독이 버킷별이고, 기존 nt/tr 조성표와 같은 자이며,
stop과 accel처럼 |a|는 같고 부호가 다른 경우를 구분한다. raw action은 G0에서 "bucket5 층이
실제 action 공간에서 갈라지는가"를 검증하는 데 쓰고, 클립별로 저장해 두어 **bucket5가 신호를
보이면 raw action 셀(mean|a| × mean|κ| 3분위 3×3) 층화가 다음 1요인 실험**이 된다. bucket5는
raw action의 적분량(속도비·순회전)에 문턱을 건 것이므로 두 라벨이 크게 어긋나면 그 자체가
결과다(G0-d).

## 질문과 가설

모든 arm은 `dual_u40_v2` 레시피 — 같은 기준 `max(rank I_traj, rank I_CoC)`, 같은 예산
(uniform 0.3985632694, 정확히 2,657,452,032 파라미터 제거), VLM만, expert·KV 불변,
`jlens_v2`. **오직 캘리브레이션 100클립을 어떤 규칙으로 뽑았는가만 다르다.**

- **H1 (6축 매칭은 무의미)**: 랜덤 규칙(매칭 없음, 층화 없음)은 자연 그리디 추출 6개
  (`calib_100` + `nt_a..e`, 평균 1.312)와 구별되지 않는다.
  - 반증: 랜덤이 유의하게 나쁘면 매칭이 돕는 것이고, 층화는 그리디 위에 7번째 속성으로 얹는
    후속으로 간다.
- **H2 (다양성은 비-cruise를 산다)**: 균등 층화(20%×5)는 랜덤보다 decel_stop / accel / turn
  버킷에서 손해가 작다.
- **H3 (대가는 cruise)**: 같은 비교에서 cruise 손해는 커진다. H2·H3가 같이 성립하면 다양성은
  "평균"이 아니라 "분포 모양"의 노브이고, 어느 쪽을 택할지는 평가셋 조성(개루프 56% cruise)이
  아니라 폐루프가 정해야 한다.
- **H4 (평가셋 매칭 층화는 무효)**: test500 조성(56/13/15/9/7)으로 층화한 세트는 랜덤과
  구별되지 않는다 — §왜 3의 "조성은 결과를 설명 못 한다"의 전향적 확인.
- **H5 (분산, 기술적)**: 규칙별 seed 간 SD. n=3이라 구간이 넓어(σ의 95% CI가 추정치의
  0.5–6배) 판정 게이트로는 쓰지 않고 보고만 한다.

## 설계

### 풀과 라벨링

- 풀: 공식 train 153,625 − OOD 1,740 − `calib_100` − `nt_a..e` − `tr_a..e` − `rd_a/b` −
  `calib_val100` − `st4000` 계열 − 평가셋 4종. `make_calib_blocks.py --exclude`가 이미 하는
  홀드아웃에 st 계열을 더한 것. **t0 = 5,100,000 µs 고정** (`calib_100`과 같음; tr이 잃은
  균형은 t0 재선택에서 사라졌으므로 라벨과 캐시의 t0가 같아야 한다 — G0-c).
- 후보: 풀에서 무작위 3,000클립(seed 20260911). 이 후보에 대해 egomotion zip만 읽어
  (`bucket_calib_sets.py`의 경로, 비디오 디코딩 없음; 621 로컬 청크 밖은 스트리밍) bucket5를
  붙인다. 자연 풀의 turn_left ≈ 7–9%이므로 후보 3,000에 ~240개 — 아래 9세트가 필요로 하는
  turn_left 합계 60+27+21 = 108개의 2배 여유. 층화 표본은 후보가 무작위인 한 풀 전체에서의
  층화와 분포가 같다.
- raw action은 캐시 빌드 후 npz에서 계산한다(라벨 시점에는 rot이 없음). 후보 3,000 전체가
  아니라 선택된 700 + 기존 rd_a/b 200 + nt/tr 1,000 + test500에 대해 계산해 저장.

### 세 규칙 × 세 seed = 9 arm, 전부 서로소

| 규칙 | 태그 | 정의 | 세트 |
|---|---|---|---|
| random | `rd` | 풀에서 균등 무작위 100 (층화 없음, 매칭 없음) | 기존 `calib_rd100_a`, `calib_rd100_b` (캐시 있음, `dual` 미평가) + 신규 `calib_rd100_c` |
| strat-eval | `se` | bucket5 층화, 쿼터 = test500 조성 **56/13/15/9/7** (cruise/stop/accel/turnL/turnR), 층 안 무작위 | `calib_se100_a/b/c` |
| strat-uniform | `su` | bucket5 층화, 쿼터 **20/20/20/20/20**, 층 안 무작위 | `calib_su100_a/b/c` |

- seed는 규칙별 a/b/c = 1/2/3에 기본 20260911을 더한 값. 9세트는 서로·기존 전 세트·평가셋과
  **서로소**(assert). 층화 규칙에서 순차 배제는 층 안 무작위 순열의 분할과 같아 교환
  가능성을 깨지 않는다.
- `rd_a/b`는 `--select random --seed 42`로 같은 풀(t0 5.1 s)에서 뽑힌 것이라 rd_c와 같은
  분포다. draw-order 플랜이 이 둘을 "랜덤 축"으로 사전 등록해 두었으므로 그 미해결 축이
  여기서 닫힌다.
- 캐시: 신규 700클립 → `pre_processed/calib_strat` (`build_cache.py --manifest … --cache
  calib_strat`). 클립당 ~5.2 MB(`calib_rd_a` 524 MB/100)이므로 ~3.7 GB. rd_a/b는 기존 캐시.

### 파이프라인 (arm당, 기존 nt arm과 동일)

```bash
# 1. 라벨 + 추출 (CPU)
python experiments/evaluation/label_actions.py --candidates 3000 --seed 20260911 \
    --out outputs/strat_calib/labels.parquet            # bucket5 @ t0=5.1s, 후보 3,000
python experiments/evaluation/make_calib_strat.py --labels outputs/strat_calib/labels.parquet \
    --rules rd:1 se:3 su:3 --seed 20260911              # 7 신규 manifest, 서로소 assert
python experiments/evaluation/build_cache.py --manifest calib_strat700 --cache calib_strat
python experiments/evaluation/label_actions.py --raw-from-cache …   # traj_to_action, CPU

# 2. 중요도 → slim → 평가 (Ada 4–7, 빈 카드만)
bash experiments/head_analysis/run_retry_host.sh 120 experiments/head_analysis/run_importance.py \
    --calib-manifest calib_se100_a --cache calib_strat --num-clips 100 \
    --exp-id importance_se100_a --gpu <N>
python experiments/head_analysis/make_slim.py --config dual_u40_v2 \
    --importance importance_se100_a --jlens jlens_v2 --out outputs/slim_dual_se_a --no-state
python experiments/evaluation/run_baseline.py --set test --model outputs/slim_dual_se_a \
    --exp-id dual_se_a_test --shard {0,1} --n-shards 2 --gpu <N>
```

- `--no-state`: `dual`은 선택 전용이라 `slim_meta.json`만으로 재구성이 비트 동일
  (`2026-09-03` §7에서 1,159 텐서 전수 확인). arm당 17 GB를 안 쓴다 — `/mnt/nvme1n1`이
  92%라 이것이 없으면 9 arm이 못 돈다.
- 평가는 `baseline_test`(Ada, 저장됨)와 페어링. 모든 arm을 **Ada 4–7**에서, 카드는 실행
  직전 `nvidia-smi`로 빈 것만.
- 분석: `analyze_strat_calib.py` (신규) — 레코드에서 클립별 Δ = arm − baseline, 규칙별
  **클립별 3-seed 평균**을 만든 뒤 규칙 쌍을 500클립 페어드로 비교(부트스트랩 CI + Wilcoxon),
  같은 것을 버킷별로, arm 단위 minADE 9개와 규칙별 SD, 기존 nt/tr 계열과의 비교표,
  6축 weighted L1·bucket5 조성·raw action 기술량 per set. `outputs/strat_calib/`에
  `config.json` / `metrics.json` / `summary.txt` / `plots/`.

### 변하지 않는 것

기준·예산·expert·KV·`jlens_v2`·평가 프로토콜(k=8, seed 42, `sha256(f"{seed}:{clip_id}")`,
deterministic, Ada)·baseline 레코드. 새 코드는 추출·라벨·분석 셋뿐이고 `run_importance` /
`make_slim` / `run_baseline`은 건드리지 않는다.

## 검정력 (실측)

기존 n=100 arm 쌍(nt_a..e, 10쌍)의 test500 **클립별 Δ SD 중앙값 0.808**(범위 0.34–1.27).
규칙 쌍 비교는 클립별로 3-seed 평균을 내므로 SE ≈ 0.808/√1500 = **0.021 → MDE(2·SE) ≈ 0.04**
(전체). 버킷별 SE: cruise 0.029 (n=278), decel_stop 0.028 (65), accel 0.035 (75), turn 0.050
(82) → **버킷 MDE 0.06–0.10**. §왜 1의 버킷 간 차이(cruise +0.74 vs 나머지 +0.23–0.30,
tr accel +0.134 vs +0.230)는 이 해상도 안이다. arm 단위 SD(H5)는 n=3이라 해상도가 없다 —
그래서 SD는 게이트가 아니다.

## 사전 등록 게이트

- **G0 (파이프라인 무결성)**, 전부 통과해야 arm 빌드 시작:
  - (a) `se`/`su` 세트의 실현 조성이 쿼터와 **정확히** 일치(정의상), `rd_c`의 6축 L1은
    `rd_a/b`(0.19/0.22)와 같은 크기.
  - (b) 9세트 서로소, 기존 전 세트·평가셋 4종과 겹침 0 (assert).
  - (c) **t0 함정**: 캐시 npz의 `ego_future_xyz`에서 다시 계산한 bucket5가 egomotion 라벨과
    700/700 일치. 하나라도 어긋나면 tr이 겪은 창 재선택이 일어난 것.
  - (d) raw action에서 층이 갈라진다: mean|κ| 중앙값이 turn 층 > cruise 층, max 감속 중앙값이
    stop 층 > cruise 층, 정규화 RMS(a)가 accel/stop 층 > cruise 층. 갈라지지 않으면 bucket5는
    action 라벨이 아니고 raw action 셀로 층화 단위를 바꿔 다시 뽑는다.
- **G1 (H1, 매칭)**: 클립별 [rd 3-seed 평균 − nt 5블록 평균], 500클립 페어드 부트스트랩.
  CI가 0을 포함하거나 |Δ| < 0.04 → **매칭 무의미** (그리디를 버린다). Δ > +0.04이고 CI가 0을
  배제 → 매칭이 돕는다 → 층화는 그리디의 7번째 속성으로 재설계. Δ < −0.04 → 매칭이 해롭다.
  보조: rd 3 arm 대 nt 6 arm의 arm 단위 평균·범위.
- **G2 (H2, 다양성→비-cruise)**: 클립별 [su − rd] 3-seed 평균, decel_stop / accel / turn
  세 버킷. **≥2 버킷에서 Δ ≤ −0.05이고 CI가 0을 배제** → 채택.
- **G3 (H3, cruise 대가)**: 같은 비교의 cruise Δ. 보고하고, 전체 Δ(= test500 조성 가중)와
  같이 읽는다. G2 채택 + G3 ≥ +0.05 → "분포 모양의 노브" 판정, 폐루프 후속.
- **G4 (H4, 평가셋 매칭)**: 클립별 [se − rd] 전체와 버킷별. 전부 |Δ| < 0.04이고 CI가 0 포함
  → 자연 비율의 조성 통제는 무효(H4 채택).
- **G5 (H5)**: 규칙별 arm 단위 SD와 범위. 보고만.

### 결정 규칙 (표준 추출을 무엇으로 바꿀 것인가)

1. G1 무의미 + G2 채택 → `su`(균등 층화 무작위)를 표준으로 **제안**하되, 개루프 전체 Δ가
   +0.04 이상 나쁘면 채택 전에 hard100 폐루프 1회(`launch_alpasim_shards.sh`, 100씬 × 2
   rollout, `su_a` vs `rd_a`) — 이 레포에서 개루프와 폐루프가 갈린 전례(head vs MLP)가 있다.
2. G1 무의미 + G2 기각 + G4 채택 → `rd`(무작위)를 표준으로. 그리디와 층화 모두 버린다 —
   가장 단순한 결과이고, "조성으로는 안 된다"가 전향적으로 확정된 것.
3. G1 "돕는다" → 그리디 유지, bucket5를 7번째 속성으로 얹는 후속 플랜.
4. 어느 경우든 **raw action 셀 층화**는 G2 채택 시에만 다음 실험으로 올린다.

## 비용

| 단계 | 자원 | 시간 |
|---|---|---|
| 후보 3,000 라벨링 (egomotion 스트리밍, 16 workers) | CPU/네트워크 | ~15분 (1,100클립 선례 기준 추정; 실측 기록) |
| 캐시 700클립 | CPU/네트워크, 3.7 GB | ~26분 (nt 500클립 18.7분 비례) |
| 중요도 9 × 100클립 | Ada 1장 | 9 × 9분 ≈ 1.4 GPU-h |
| slim 빌드 9 (`--no-state`) | Ada 1장 | 9 × ~1.0 h ≈ 9 GPU-h |
| test500 평가 9 × 2 shard | Ada | 9 × 1.35 ≈ 12 GPU-h |
| **합계** | | **≈ 23 GPU-h**, 디스크 ~4 GB |

`rd_a/b`가 캐시·매니페스트 재사용이라 실제 신규 캐시는 700클립. GPU는 빈 카드만
(`only-empty-gpus`), 평가는 Ada 4–7(`gpus-0-3-off-limits`).

## 한계

- **seed 3개**는 규칙의 기대값 비교(클립별 3-seed 평균, MDE 0.04)에는 충분하지만 arm 단위
  SD에는 부족하다. "층화가 분산을 줄이는가"는 이 실험이 답하지 않는다 — 부호가 보이면 seed
  5로 늘리는 후속.
- **test500만** 본다(val500 생략, 사용자 결정). test500은 56% cruise라 전체 Δ는 cruise
  가중이다 — 그래서 게이트가 버킷별이고, 전체 Δ는 결정 규칙 1의 안전장치로만 쓴다.
- 후보 3,000이 풀 전체가 아니므로 turn 층의 표본 공간이 ~240개다. 층화 통계에는 문제없지만
  "그 240개 중 어느 100개"는 여전히 잡음이다 — §왜 3이 말하는 클립 단위 잡음은 층화가
  없애 주지 않는다.
- bucket5는 t0 = 5.1 s의 6.4 s 창만 본다. 클립의 다른 구간은 무관하다(캘리브레이션도 그
  창만 쓰므로 일관되지만, "클립의 action"이 아니라 "창의 action"이다).
- 개루프 결론이 폐루프로 옮겨간다는 보장은 없다(head vs MLP 전례). 결정 규칙 1이 그 때문에
  폐루프 1회를 조건으로 건다.

## 실행 순서

1. 코드 셋: `label_actions.py`, `make_calib_strat.py`, `analyze_strat_calib.py`
   (전부 `experiments/evaluation/`). 기존 `make_calib_blocks.py`의 풀·배제·assert 로직을
   import해 재사용하고 그리디는 부르지 않는다.
2. 라벨링 → 추출 → G0 (a)(b) → 캐시 → G0 (c)(d). 여기까지 GPU 0.
3. 중요도 9개(빈 Ada 1장, 순차 ~1.4 h) → slim 9개 → 평가 18 shard를 `launch_arms.sh` 큐로.
4. `analyze_strat_calib.py` → `outputs/strat_calib/` → 보고서
   `reports/evaluation/2026-09-11_action-stratified-calib.html`
   (`experiments/evaluation/strat_calib_report_template.html`).
5. 결정 규칙에 따라 표준 추출 변경 여부를 CLAUDE.md에 기록.

## 부록 — 기존 세트의 action 분포 (2026-09-11 측정)

`outputs/calib_buckets/action_mix_2026-09-11.parquet`. 캐시 npz의 `ego_future_xyz`(64점)에서
계산. 최다 셀 = 감속 4구간 × |회전| 4구간 히스토그램의 최대 점유율.

| set | cruise | stop | accel | turn L | turn R | v p90 | 감속 p90 | \|회전\| p90 | 최다 셀 | test500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| test_500 | 56 | 13 | 15 | 9 | 7 | 23.1 | 2.23 | 48 | 32% | — |
| calib_100 | 49 | 18 | 16 | 8 | 9 | 27.5 | 2.01 | 50 | 29% | 0.950 |
| nt_a | 54 | 13 | 17 | 8 | 8 | 26.0 | 2.02 | 44 | 32% | 1.022 |
| nt_b | 59 | 13 | 16 | 4 | 8 | 27.9 | 2.11 | 33 | 35% | 1.060 |
| nt_c | 57 | 15 | 15 | 6 | 7 | 28.7 | 2.02 | 35 | 39% | 1.348 |
| nt_d | 52 | 16 | 15 | 9 | 8 | 27.2 | 2.07 | 51 | 42% | 1.774 |
| nt_e | 49 | 14 | 19 | 14 | 4 | 22.7 | 1.79 | 51 | 37% | 1.718 |
| tr_a | 42 | 16 | 14 | 12 | 16 | 19.4 | 2.32 | 63 | 21% | 1.214 |
| tr_b | 43 | 19 | 21 | 9 | 8 | 18.5 | 2.52 | 47 | 29% | 1.146 |
| tr_c | 35 | 13 | 23 | 16 | 13 | 15.9 | 2.14 | 64 | 29% | 1.101 |
| tr_d | 47 | 21 | 15 | 8 | 9 | 18.2 | 2.90 | 46 | 25% | 1.169 |
| tr_e | 43 | 20 | 15 | 8 | 14 | 18.0 | 2.34 | 66 | 18% | 1.251 |
| rd_a | 61 | 13 | 17 | 5 | 4 | 23.3 | 2.07 | 33 | 34% | 미평가 |
| rd_b | 51 | 12 | 18 | 3 | 16 | 20.0 | 2.02 | 71 | 36% | 미평가 |

버킷별 손해(unpruned 대비 클립별 Δ 평균, 5추출 평균과 추출 간 SD). 이 표는 `baseline_test`와
페어링한 값이고, `analyze_strat_calib.py`는 calib_variance 관례대로 `baseline_ada_ps_test`와
페어링한다 — 두 baseline 런은 다른 실행이라 값이 ≤0.02 다르다(nt cruise +0.736 vs +0.746);
게이트 판정은 분석기 쪽 값으로 한다:

| bucket (n) | calib_100 | nt ×5 | tr ×5 |
|---|---:|---:|---:|
| cruise (278) | +0.050 | +0.736 (0.569) | +0.401 (0.072) |
| decel_stop (65) | +0.163 | +0.228 (0.112) | +0.210 (0.055) |
| accel (75) | +0.168 | +0.230 (0.043) | +0.134 (0.048) |
| turn (82) | +0.147 | +0.296 (0.089) | +0.285 (0.094) |
| all (500) | +0.098 | +0.522 (0.350) | +0.317 (0.058) |

## 진행 기록

### 1. 승인, 1단계 (2026-09-11 20:45–21:30 KST)

코드 셋 + 런처: `experiments/evaluation/label_actions.py` (후보 라벨링 + 캐시 raw action 검증),
`make_calib_strat.py` (규칙별 추출, 서로소·쿼터 assert), `analyze_strat_calib.py` (G1–G5,
arm 없으면 pending), `launch_strat_calib.sh` (중요도 → slim → 평가 큐; **빈 Ada 카드**를
60초 폴링으로 기다려 그 카드에 고정 — `reserve_gpu`의 "여유 있음"은 "빔"이 아니다).

**라벨링**: 공식 train 153,625 − 홀드아웃 7,004(기존 calib_* 21개 + 평가셋 + OOD) = 146,621에서
3,000 무작위(seed 20260911), egomotion만 읽어 t0=5.1 s bucket5. 16 workers **16.1분, 실패 0**.
조성 cruise 56.3 / stop 13.1 / accel 15.1 / turnL 7.7 / turnR 7.8% — **test500(56/13/15/9/7)과
같다**, 즉 자연 풀 = 평가셋 조성. turn 층 230 / 234개.

**추출 (G0-a, G0-b 통과)**: 7세트 전부 쿼터 정확 실현(se 56/13/15/9/7, su 20×5, assert), 9세트
서로소 + 기존 전 세트·평가셋·OOD와 겹침 0 (assert). 6축 weighted L1: rd_c 0.137, se 0.151 /
0.192 / 0.183, su 0.205 / 0.160 / 0.132 — 전부 랜덤 수준(rd_a/b 0.190 / 0.225, 그리디 0.031).
seed 규칙 `default_rng([20260911, rule_idx, letter_idx])`.

**캐시 함정 하나**: `build_cache.py`는 토큰을 환경에서만 읽는데 맨 셸은 `HF_TOKEN` 미설정 +
`HF_HOME`이 공용 캐시라 익명 요청이 되어 `refs` 엔드포인트 **429**로 700/700 실패(0.2분,
exit 0 — `errors.json`을 봐야 보인다). 토큰·`HF_HOME`·`HF_HUB_CACHE`를 설정하는 파이썬
래퍼로 재실행: 25클립/1.5분 → 700클립 ~27분, 클립당 ~5.1 MB.

**G0-d 사전 확인 (참조셋 1,800클립, 새 세트 캐시 전)**: `traj_to_action`을 CPU에서 돌린 raw
action 공간에서 bucket5 층이 갈라진다 — mean|κ| turn 0.033 vs cruise 0.0008 (40배), 최대 감속
stop 1.91 vs cruise 0.41 m/s², 정규화 RMS(a) accel 1.64 / stop 1.65 vs cruise 0.51. 부수:
버킷이 속도도 층화한다(v0 중앙값 cruise 13.3, stop 6.8, turn 5.5, accel 1.3 m/s).

GPU: Ada 4–7은 `m2601_slim_dualexp_u40_em87p5` 150씬 폐루프(5시간째)가, Blackwell 0–3은 다른
멤버의 cosmos 학습이 점유. 중요도 런처는 백그라운드에서 빈 Ada 카드를 기다리는 중.

### 2. 캐시·G0 완료, GPU 단계를 cvlab20으로 (2026-09-11 21:30–22:20 KST)

캐시 `pre_processed/calib_strat` **700/700, 3.74 GB, 오류 0** (25.5분). 전체 21세트(신규 7 +
rd_a/b + calib_100 + nt/tr + test500 = 2,600클립)에서 raw action 재계산:

- **G0-c PASS — 700/700.** 추출에 쓴 egomotion 라벨과 캐시 npz에서 다시 계산한 bucket5가 전부
  일치. t0 재선택은 일어나지 않았다.
- **G0-d PASS.** 풀 전체 중앙값: mean|κ| turn 0.035 vs cruise 0.0008 (44배), 최대 감속 stop
  1.90 vs cruise 0.41 m/s², 정규화 RMS(a) accel 1.61 / stop 1.65 vs cruise 0.51.
  `outputs/strat_calib/raw_actions.parquet`, `g0_raw.json`.

**GPU 단계는 cvlab20으로** (사용자 지시). cvlab21의 Ada 4–7은 우리 폐루프가, Blackwell은 다른
멤버가 점유. cvlab20은 `/mnt/dataset1` 79%(1.5 TB 여유), 카드 8장 중 0–3·6·7은 31 GB씩,
4·5는 다른 사람의 소규모 프로세스(0.3 / 2.5 GB) — 완전히 빈 카드는 아직 없어 런처가 대기 중.
옮긴 것: 코드 트리(`experiments/ src/ paper/`, 러너 3종 체크섬 일치), 매니페스트 8종,
`calib_strat` 캐시 4.0 GB (97 MB/s). `calib_rd_a/b`·`test` 캐시, `jlens_v2`,
`slim_integrated_mag/slim_meta.json`은 이미 있었다. 런처는 `DIRECT_ENV=…/env.sh`로
`run_retry_host.sh` 대신 `.venv/bin/python`을 직접 부르고 로그는 `/mnt/dataset1/chan/logs`.
cvlab21 쪽 런처는 중복 방지를 위해 중지 — 9 arm 전부 한 박스(cvlab20, Ada)에서 잰다.
분석은 cvlab21의 `baseline_ada_ps_test`와 페어링(두 박스 비트 동일, 2026-09-08 slim 경로 검증).

실수 하나: 첫 rsync를 `experiments/ src/`(끝 슬래시)로 보내 내용물이 cvlab20 repo 루트에
풀렸다. 올바른 경로로 재동기화했고 동작에는 영향 없음(임포트는 `experiments/…` 기준). 루트의
잔재(`alpamayo_r1 evaluation head_analysis lingoqa llm_pruner recovery transfer`, 그리고
`paper/`에 섞인 `experiments/paper` 파일)는 공용 계정이라 자동 삭제가 거부됨 — 수동 정리 필요.
