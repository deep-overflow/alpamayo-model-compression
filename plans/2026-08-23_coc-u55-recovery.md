# coc_u55 회복 학습 — 프루닝 criterion 차이는 회복 후에도 남는가

작성 2026-08-23 (KST). 승인 전 계획서.

## 1. 배경과 가설

지금까지의 회복 실험은 전부 `dual`(= `max(rank I_traj, rank I_CoC)`) 기준으로 잘린
체크포인트에서만 돌았다. u55에서 "회복하면 dual_u40 제로샷과 구분 불가"라는 결론이 나왔지만,
그 결론이 **criterion 자체에 대해** 말해주는 바는 없다. 회복 학습이 붙는 순간
어떤 기준으로 잘랐는지가 지워지는지, 아니면 구조적으로 남는지가 미지수다.

제로샷에서는 criterion 효과가 분명히 존재한다. 같은 예산(u40, VLM only, expert/KV 무손상,
동일 calibration)에서 같은 지표(minADE@6, 페어드 시드)로 재환원하면:

| arm | val_500 | test_500 | degen (val/test) |
|---|---:|---:|---|
| `dual_u40_v2` (`max(rank I_traj, rank I_CoC)`) | 0.8904 | 0.9498 | 0.014 / 0.030 |
| `coc_u40_v2` (단일 `I_CoC`) | 1.5577 | 1.5837 | 0.142 / 0.150 |

궤적 반쪽(`I_traj`)을 빼면 −24% 예산에서도 minADE가 +0.67 벌어지고 degeneracy가 10배가
된다. u55에서는 더 벌어질 것으로 본다.

**주 가설 H1 (criterion 잔존):** u55 예산에서 `coc` 기준으로 자른 모델은 동일 레시피의
KI-LoRA 회복 후에도 `dual` 기준 회복본보다 유의하게 나쁘다. 즉 선택 기준의 손상은
LoRA로 메워지지 않는 구조적 손상이다.

**대립 H0 (회복이 지운다):** 회복 후 두 arm의 페어드 차이가 0을 포함한다.
이 경우 "criterion 연구는 zero-shot 아티팩트"라는 강한 함의가 생기고, 이후 압축
파이프라인은 선택 기준보다 회복 예산에 투자해야 한다.

어느 쪽이 나와도 보고 가치가 있다. H0은 우리 자신의 선택-기준 트랙(dual / j_traj /
tyr / wanda 계열 전체)의 유효 범위를 규정하고, H1은 그 트랙을 정당화한다.

부가 가설 **H2:** `coc_u55` 제로샷은 `dual_u55`(degen 0.694/0.714/0.676)와 같거나 더 심한
붕괴를 보인다. 회복 메커니즘(언어 채널 교정 → 궤적 개선)이 재현되는지 확인용.

## 2. 일요인 설계 (one-factor)

`dual_u55_v2`와 **레이어 내 점수만** 다르고 나머지는 전부 동일하다.
`make_slim.build_masks`의 `^(.+)_u(\d+)_v2$` 분기가 이 성질을 보장한다:

| 축 | 값 | dual_u55_v2와 |
|---|---|---|
| 예산 | uniform 0.55 (Q heads, MLP channels), 36개 층 전부 | 동일 |
| expert tower | 무손상 (`eq, em = ones`) | 동일 |
| KV groups | 무손상 (`kvonly = ()`) | 동일 |
| calibration | `importance_v2` (calib_100, 100 클립) | 동일 |
| 점수 | `I_CoC` 단일 (`coc_vlm_q`, `coc_vlm_mlp`) | **다름** (dual은 `max(rank I_traj, rank I_CoC)`) |
| 제거 파라미터 | 3,669,000,192 (예상) | 동일해야 함 → 빌드 시 assert |

단일 기준이라 `rank_norm`은 적용되지 않는다(레이어 내 argsort에 대해 단조 사상이므로
무효). `traj/coc/j` 단일-기준 대조군 관례와 일치한다.

회복 레시피도 u55 v2와 **완전 동일**하다: 1,200 step, accum 4(글로벌 배치 16),
lr 1.4e-4 cosine, warmup 50, λ_traj 0.5, LoRA r=32/α=64, KI 절연(FM→expert, CE→VLM),
학습 데이터 2,471(OOD train 1,271 + official 버킷균형 1,200), probe 500, seed 42,
`--val-every 150`, `--probe-k 6`, `--val-fm-clips 100`.

## 3. 실행 단계

### S0. 빌드 (Blackwell GPU 0, ~2분)
```bash
bash experiments/head_analysis/run_retry_host.sh 20 experiments/head_analysis/make_slim.py \
    --config coc_u55_v2 --out outputs/slim_coc_u55_v2 --importance importance_v2 --gpu 0
```
확인: `summary.txt`의 removed == 3,669,000,192, 그리고 `dual_u55_v2`와의 kept-set 중첩률
(Q head / MLP)을 보고한다 — 중첩이 100%면 실험 자체가 무의미하므로 진행 전 게이트다.

### S1. 회복 학습 (Blackwell 0–3, ~4.8시간)
```bash
nohup bash experiments/recovery/run_ddp_retry.sh 60 "0 1 2 3" \
    experiments/recovery/train_recover.py --ckpt outputs/slim_coc_u55_v2 \
    --exp-id recover_coc_u55 --steps 1200 --accum 4 --lr 1.4e-4 --warmup 50 \
    --val-every 150 --lambda-traj 0.5 --val-fm-clips 100 \
    > logs/recover_coc_u55.log 2>&1 &
```
step-0 probe가 곧 제로샷 참조값(minADE@6, degen)이다. KI CHECK와 rank 배치(0–3만)를
기동 직후 확인한다. `--resume auto`가 기본이므로 kill 시 `state_last.pt`에서 이어진다.

### S2. merged 재구성 (Ada, ~5분)
```bash
python experiments/recovery/rebuild_merged.py --adapter outputs/recover_coc_u55/adapter_best.pt \
    --out outputs/slim_recover_coc_u55 --gpu <free ada>
```
LoRA 병합은 bf16 matmul이라 아키텍처 간 반올림이 다를 수 있다. 기존 u55/u70 merged가
전부 Ada에서 만들어졌으므로 **Ada에서 병합**해 전례와 맞춘다.

### S3. 제로샷 앵커 3세트 (Ada, 페어드 시드)
```bash
python experiments/evaluation/run_baseline.py --set indist --model outputs/slim_coc_u55_v2 \
    --exp-id coc_u55_indist --k 8 --gpu <ada>
python experiments/evaluation/run_baseline.py --set test   --model outputs/slim_coc_u55_v2 \
    --exp-id coc_u55_test   --k 8 --gpu <ada>
python experiments/evaluation/run_baseline.py --set ood --manifest ood_val \
    --model outputs/slim_coc_u55_v2 --exp-id coc_u55_oodval --k 8 --gpu <ada>
```

### S4. 회복본 3세트 (Ada) — S3와 동일, `--model outputs/slim_recover_coc_u55`,
`--exp-id slim_recover_coc_u55_{indist,test,oodval}`.

S3+S4는 총 2,524 클립 × 2 arm. 4카드가 나면 `--shard/--n-shards`로 쪼개 채운다.
붕괴한 제로샷은 CoC가 max_gen 256까지 도는 경우가 많아 in-dist 8s/clip보다 느리다 —
u55 제로샷 전례 기준 넉넉히 3–4시간으로 잡는다.

### S5. 분석
```bash
python experiments/recovery/analyze_recovery.py --rec slim_recover_coc_u55 --zs coc_u55 \
    --peer slim_recover_dual_u55_v2 --out outputs/recovery_eval_cocu55
```
`--peer`는 신규 인자다(작은 수정 1건): 5번째 arm으로 dual u55 v2 회복본을 표에 넣고,
`recovered − peer` 페어드 델타를 세트별로, 그리고 세 세트 **풀링(1,262 클립)**으로 낸다.
풀링은 u55 v1↔v2 비교에서 이미 쓴 방식이고 세트별로는 잡히지 않는 0.03급 차이를 잡는다.

## 4. 사전 등록 게이트

- **G1 (회복 성립, test_500):** recovered minADE@6 ≤ 1.6 **AND** `d(recovered − zeroshot)`
  부트스트랩 CI 상한 < 0.
- **G2 (degeneracy):** recovered degen ≤ 0.05 (세 세트 전부).
- **G3 (주 판정, criterion 잔존):** 풀링 1,262 클립의 `d(coc_rec − dual_rec)`
  - CI 하한 > 0 → **H1 채택** (criterion 차이가 회복 후에도 남는다)
  - CI가 0을 포함하고 |mean| < 0.05 → **H0 채택** (회복이 지운다)
  - 그 사이(CI가 0을 포함하나 |mean| ≥ 0.05) → **inconclusive**로 명시하고, 판정을
    폐루프(150 scenes)로 넘길지 별도 판단.
- 해상도: 같은 풀링 설계에서 u55 v1↔v2의 δ=0.0341이 유의하게 잡혔다. 즉 이 검정의
  실효 해상도는 0.03 부근이고, G3의 H0 임계 0.05는 그보다 보수적이다.

## 5. 일정과 GPU 정책

현재(09:51 KST) Ada 4–7은 `slim_tyr_u40_r` alpasim 150-scene 4-shard가 09:32부터 점유 중
(~17:10 KST 종료 예상). Blackwell 0–3은 완전 유휴.

| 시각(KST) | 작업 | GPU |
|---|---|---|
| ~10:00 | S0 빌드 | 0 |
| 10:00–14:50 | S1 학습 1,200 step | 0–3 (Blackwell) |
| ~17:10 | alpasim 종료 대기 | — |
| 17:10–17:20 | S2 merged | Ada 1장 |
| 17:20–21:00 | S3+S4 평가 6런 | Ada 4장 |
| 이후 | S5 분석 | CPU |

**정책 준수:** 학습·빌드만 Blackwell 0–3, **모든 평가는 Ada 4–7**. 결정성이 아키텍처
내에서만 bitwise이고 published arm(baseline_ada_*, dual_u40, dual_u55, u70)이 전부
Ada 측정값이라 페어드가 성립하려면 Ada여야 한다. 학습 중 probe는 Blackwell이어도 되고
(체크포인트 선택용, 절대값 미보고) 이는 u55 v2·u70과 동일 조건이다.

## 6. 위험과 대응

- **디스크.** `/mnt/nvme1n1` 172 GB 여유(98% 사용), alpasim이 동시에 쓴다.
  필요량 ~28 GB(slim 14 + merged 14). 부족해지면 `outputs/slim_recover_dual_u55`(v1, 14 GB,
  adapter에서 재구성 가능)를 후보로 삼되, alpasim driver 하드링크 여부를 먼저 확인한다.
- **완전 붕괴.** `coc_u55` 제로샷이 u70처럼 degen 1.000일 수 있다. u70에서도 150 step에
  degen 0으로 회복됐으므로 차단 사유는 아니지만, S3 평가 시간이 늘어난다.
- **probe = ood_val.** 체크포인트 선택 probe가 보고 세트 ood_val 262와 동일 클립이다.
  u55·u70과 **동일한 기존 성질**이라 arm 간 비교는 일요인이 유지되지만, ood_val은
  선택-오염된 수치로 읽어야 한다. 깨끗한 게이트는 test_500이며 G1을 거기 건 이유다.
- **한 arm만 늘어난 학습 예산이 아님.** 두 arm 모두 1,200 step 동일 레시피이므로
  u55 v1(600 step)과 섞어 비교하지 않는다. 비교 대상은 `recover_dual_u55_v2`뿐이다.

## 7. 범위 밖(이번엔 안 함)

- 폐루프(alpasim 150 scenes): open-loop 판정 후 별도 판단. Ada 8시간 × arm이 든다.
- `j_u55` 등 다른 단일 기준 arm.
- 귀속 대조군(λ_traj=0), CE 데이터 5배 — 기존 백로그 유지.
