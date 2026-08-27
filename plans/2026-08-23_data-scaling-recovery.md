# 회복 학습 데이터 확대 — 9,021 샘플 / 1,689 스텝

작성 2026-08-23 (KST), 실측값으로 갱신 2026-08-24. 승인·실행됨 (NEURON job 892052).

## 1. 배경

`traj_u55` 실험(2026-08-23)에서 u55 세 arm이 회복 후 풀링 minADE@6 0.9539 / 0.9683 /
0.9713으로 0.017 안에 모였고, 무손상 baseline과는 test_500에서 **+0.096**이 남았다.
회복 레시피가 그 격차를 못 메우고 있으니, 무엇이 병목인지가 다음 질문이다.

**현재 데이터가 병목이라는 증거는 약하다.** 기존 런의 `--val-fm-clips 100` held-out
고정격자 FM loss(`recover_dual_u55_v2`):

| step | 0 | 150 | 300 | 600 | 900 | 1200 |
|---|---:|---:|---:|---:|---:|---:|
| held-out FM mean | 0.3494 | 0.2715 | 0.2656 | 0.2640 | 0.2633 | 0.2630 |
| t≈0.05–0.15 (어려운 구간) | 0.5114 | 0.3761 | 0.3616 | 0.3599 | 0.3595 | 0.3585 |

train `L_fm`(마지막 100스텝) 0.2776 vs held-out 0.2630 — **과적합 격차가 없고**,
행동 채널은 150스텝에 수렴한 뒤 1,050스텝 동안 3%만 움직인다. 반면 `L_ce`는 0.015까지
떨어져 언어 채널은 명백히 암기했으나 degen은 이미 0.000이라 더 얻을 것이 없다.

즉 사전 예측은 **"데이터를 늘려도 minADE는 거의 안 움직인다"** 이다. 이 계획은 그
예측을 반증 가능한 형태로 검정한다. 예측이 맞으면 병목은 LoRA 용량 또는 프루닝 손상
자체로 좁혀지고, 틀리면 데이터가 실제 레버였음이 확인된다. 어느 쪽이든 다음 실험의
방향이 정해진다.

## 2. 설계

`recover_dual_u55_v2`와 **학습 데이터 구성만** 다르다.

| 축 | 기존 (`recover_dual_u55_v2`) | 신규 (`recover_dual_u55_d5`) |
|---|---|---|
| 체크포인트 | `slim_dual_u55_v2` | 동일 |
| 학습 샘플 | 2,471 (official 1,200 + ood 1,271) | **9,021** (official 7,750 + ood 1,271) |
| 스텝 | 1,200 | **1,689** |
| 에폭 | 7.8 | **3.0** |
| probe 주기 | 150 (= 1 에폭) | **563** (= 1 에폭) |
| global batch | 16 | 동일 |
| lr / warmup / λ_traj / LoRA r,α | 1.4e-4 cosine / 50 / 0.5 / 32,64 | 동일 |
| probe 세트 | ood_val 262 + official 238 | 동일 (파일 복원으로 고정) |
| 학습 하드웨어 | Blackwell | **NEURON A100** |

1,689스텝 × 16 / 9,021 = 2.996 에폭. 기존 런의 best 지점(step 900)보다 1.9배 많은
스텝이므로 학습 부족 위험은 없다.

### 제안 대비 조정 3건

**① 버킷은 4개가 아니라 5개.** 기존 `train_official_1200`은 turn_left / turn_right /
decel_stop / accel / cruise를 240개씩 균등 추출한다. 4분류(직진·감속·좌·우)는 `accel`을
빼는데, 그러면 데이터 양과 **구성**이 동시에 바뀌어 원인 귀속이 불가능해진다.

**② OOD는 1,271 전량.** `ood.parquet`의 train split이 정확히 1,271행이고 자를 이유가 없다.

**③ 규모는 12,000이 아니라 9,021.** `balanced_draw`가 **클립당 최대 1샘플**이라
고유 클립 수(9,874)가 하드 상한이고, 버킷끼리 클립을 공유해 균등 5분할의 실제 상한은
버킷당 1,550이다(turn_left 1,852클립 → turn_right는 겹침 제외 후 1,627만 남음).
`--n-train 10729`를 그냥 돌리면 에러 없이 버킷이 크게 치우친 9,874개가 나온다.
3.65배로 낮추는 대신 추출 성질(클립당 1윈도우·완전 균등)을 보존해 데이터 양만
단일 요인으로 남겼다.

### 실측 추출 결과

```
train official: 7,750 samples, 7,750 clips
  turn_left=1550  turn_right=1550  decel_stop=1550  accel=1538  cruise=1562
```

클립당 윈도우 **1.000**, CoC 보유 7,750/7,750, minADE max 3.0,
ood/calib_100/val_500/test_500/probe238과 교집합 **전부 0**. `accel` 12개 부족분은
`cruise`로 재분배됐다(0.8% 불균형).

## 3. 사전등록 게이트

**주 판정 (test_500, minADE@6, rollout, Ada).** `d(d5 − dual_u55_v2)` 페어드,
부트스트랩 95% CI:

- CI 상한 < 0 → **H1 채택**: 데이터가 레버였다
- CI가 0을 포함하고 |평균| < 0.05 → **H0 채택**: 데이터는 병목이 아니다
- 그 사이 → inconclusive로 명시

**보조 (기전 확인).** held-out FM plateau(마지막 probe의 `fm.mean`)가 기존 **0.2630** 대비:

- 0.005 이상 낮아지면 → 행동 채널이 실제로 데이터를 더 썼다
- 0.002 이내면 → §1의 진단이 확인됨

**감시 지표 (dilution).** OOD 비중이 51% → 14.1%로 떨어진다. OOD-val minADE@6이
기존 1.0739 대비 0.05 이상 나빠지면 증가 효과와 희석 효과를 분리해 서술한다.

**게이트 (기존과 동일).** G1 test_500 ≤ 1.6 및 제로샷 대비 CI < 0, G2 degen ≤ 0.05.

## 4. 실행 단계 (실측 소요)

### S0. 학습셋 생성 — 완료, 2분
```bash
python experiments/recovery/make_train_set.py --n-train 7750 --n-val 238 --max-ade 3.0
```

**`rng` 공유 함정 (실제로 발동).** `rng = np.random.default_rng(SEED)`가 한 번만
시드되고 train 추출이 먼저 소비하므로, `--n-train`을 바꾸면 뒤이은 val 추출이 **다른
238개**를 뽑고 `val_official_238.parquet`을 **덮어쓴다**. 실측 확인: 새 파일 sha256
`54b67411…` vs 원본 `0b944a3a…`. probe는 arm 간 비교의 고정축이므로 백업본으로
복원했다. `config.json`·`summary.txt`도 `*_1200.*.bak`으로 보존.

**`glob` 정렬은 이번엔 안전.** `max(glob(...))`가 문자열 비교라
`train_official_7750` > `train_official_1200`으로 신규 파일이 선택된다. 다만 12000
같은 숫자에서는 역전되므로 함정은 남아 있다.

### S1. npz 캐시 빌드 — 완료, 4시간 18분, 42.9 GB
```bash
python experiments/evaluation/build_cache.py \
    --manifest outputs/recovery_sets/train_official_7750.parquet --cache train
```
7,750개 중 417개는 기존 캐시 재사용, 7,333개 신규. 에러 0건. train 네임스페이스
총 8,533 샘플 / 47.3 GB. 체인 zip이 로컬에 있어 HF 스트리밍(500클립 24분)보다 빨랐다.

### S2. 뉴론 전송 — 완료, 22분, 44 GB
```bash
bash experiments/transfer/push_neuron.sh e1997a06 data recipes
```
원격 train 캐시 8,533개, 매니페스트 sha256 일치, 7,750개 전수 해석 확인.

### S3. 학습 — 제출됨 (job 892052)
```bash
sbatch experiments/transfer/train_recover.sbatch \
    --ckpt outputs/slim_dual_u55_v2 --exp-id recover_dual_u55_d5 \
    --steps 1689 --lr 1.4e-4 --warmup 50 --val-every 563 \
    --lambda-traj 0.5 --val-fm-clips 100
```
기동 게이트: `train samples: 9021 (official 7750, ood 1271)` — **2471이면 즉시 취소**.
그 외 KI CHECK PASS, `gpus 4 accum 4 global_batch 16`, `trainable_params 110755584`.
예상 1,689 × 5.3s ≈ 2.5시간 + probe 4회(step 0은 제로샷이라 36.8분, 이후 12.9분)
≈ 1.3시간 → **약 3.8시간**.

### S4. 회수 · 병합 · 평가 (Ada, ~3시간)
adapter 회수 → Ada에서 `rebuild_merged.py` → `run_baseline.py` 3세트. 제로샷은
`dual_u55_{indist,test,oodval}`로 이미 측정돼 있으므로 재실행하지 않는다.

### S5. 분석
```bash
python experiments/recovery/analyze_recovery.py --rec slim_recover_dual_u55_d5 \
    --zs dual_u55 --peer slim_recover_dual_u55_v2 --out outputs/recovery_eval_dualD5
```

## 5. 위험과 대응

- **OOD 희석 (구조적).** OOD-train이 1,271 상한이라 official을 늘리면 비중이 반드시
  떨어진다(51% → 14.1%). 감시 지표로 분리 서술한다.
- **스텝 수 동반 변경.** 에폭 고정이 의도이므로 수용. 결과가 애매하면 9,021 샘플 +
  1,200 스텝(계산량 고정) arm을 추가해 분리한다.
- **하드웨어 교란.** peer는 Blackwell, 신규는 A100. `traj_u55`에서 수용한 것과 같은
  성질이며 보고 시 병기한다.
- **뉴론 큐.** 2026-08-24 11:30 UTC 제출 시 `squeue --start` 추정이 08-26 05:15로
  나왔다(백필 상한이라 실제로는 더 빠를 수 있음 — 전날은 추정 22:13 → 실제 17:16).
  다른 사용자의 잡이 안 보여 대기열 깊이는 확인 불가.

## 6. 범위 밖

- LoRA rank (r=32 → 128), lr 축 — §1의 진단이 가리키는 1순위지만 단일 요인 유지를 위해 분리
- `traj` / `coc` arm의 데이터 스케일 재현
- 폐루프 3-way
- micro-batch > 1 배칭 (KI 절연 경로 수정 필요)
