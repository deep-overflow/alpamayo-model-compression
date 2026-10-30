# KISTI(NEURON) 학습 계획: 네 가지 변경을 한 번의 학습에 (2026-10-26, 사용자 결정)

**결정 경과.**
1. 첫 안은 arm A/B였다.
2. 사용자가 점진 사다리를 요청했다("한번에 너무 많이 바꾸는 건 좋지 않으니까").
3. 최종 결정: **"2번이야. 한번의 학습에 모두 넣어줘."** 네 가지 변경을 한 run(`mtw24_kd2`)에 모두 넣는다.

**해석의 한계.** 결과는 출발점 R0(`mtw24_kd`)와 비교한 **네 변경의 합산 효과**로만 해석한다. 어느 변경이 효과를
냈는지는 이 run만으로 가를 수 없다. 원인을 나눠야 하면 하나씩 빼는 ablation을 나중에 따로 돌린다.

## 1. 출발점과 변경

**R0** = `mtw24_kd`(cvlab21).
- 학생은 `slim_mtw24`이고, 데이터는 `kd_r1` × 3 epoch다.
- 손실은 모든 지도 위치 평균 KL + 속도 MSE이고, KI를 쓴다. teacher target은 사전 계산했다.
- 결과: test_2000 0.992, OOD-Val 1.124, LingoQA 57.4%.

| # | 변경 | `mtw24_kd2` |
|---|---|---|
| 1 | 플랫폼 | KISTI A100 80GB × 4, FSDP(`--shard-group 0`), **온라인 teacher**(사전 계산 target 없음) |
| 2 | 손실 가중 | **그룹별 평균 KL**: 궤적 그룹(궤적 어휘 + `traj_future_start`) 평균 + 텍스트 그룹(CoC + `cot_end`, VQA 답 + `<\|im_end\|>`) 평균, 1:1. 공식 `sft_base_model.py:388-407`과 같은 분할이다. 속도 MSE는 그대로 |
| 3 | VQA 데이터 | **LingoQA train** 약 16,000개 QA(세그먼트 단위로 고르게). 문맥은 GT 답이고, 답 토큰에만 KL을 건다. 속도 손실은 없다(expert 미사용). 주행 대 VQA는 샘플 수 기준 약 3:1 |
| 4 | 주행 데이터량 | **`kd_r5` × 1 epoch**: 9,905개 클립 × R1 window 최대 5개 = 49,033개. `kd_r1` 9,905개를 모두 포함하고, 나머지는 같은 클립의 다른 시점이다(seed 0). GT 버킷: cruise 51.9 / turn 19.6 / decel_stop 16.7 / accel 11.8% |

- **바꾸지 않는 것.**
  - 학생 초기화(`slim_mtw24`, `calib_1024`로 잰 중요도, 24.0%)
  - KI
  - 속도 KD 규칙(D1: GT 보간점, t는 추론 격자에서 균등, `kd_draw`)
  - 옵티마이저(AdamW 0.9/0.95, wd 0.1, 탑별 clip 1.0)
  - 학습률(1e-4 → 1e-5 cosine, warmup 40)
  - global batch 64, seed 0
- **held-out(손실 모니터링).** 주행은 기존 64개 클립의 모든 window, VQA는 세그먼트 32개의 QA다. 모두 학습에서 뺀다.
- **학습량.** 주행 샘플 약 4.9만 + VQA 약 1.6만 = 약 6.5만 샘플, 64개씩 약 1,020 step이다. R0는 약 2.9만 샘플, 459
  step이었다.
- **체크포인트.** 학습의 25/50/75/100% 지점에 bf16 전체 state로 저장한다.
- **재시작(2026-10-26 추가).** cvlab21 smoke(Blackwell 4장, PCIe, micro-batch 1)가 241 s/step이라 1,015 step이면
  68시간이다. A100 NVLink에서는 빨라질 것이지만 2일 안이라는 보장이 없어, 위 단서대로 재시작 기능을 넣었다.
  - 25/50/75% 지점에서 `<exp>/state`(fp32 가중치 + AdamW 모멘트, 샤드별 DCP, 약 100 GB, 한 벌만 유지)를 함께 저장한다.
  - `--deadline`(작업 종료 시각 − 1시간, `scontrol`의 EndTime; 없으면 시작 + 47시간)을 넘길 step 직전에 state를 저장하고 멈춘다. `sbatch --time`을 짧게 줘도(backfill로 일찍 시작) 그에 맞춰 멈춘다. 멈춤 판단은 전 rank가 all-reduce로 합의한다.
  - 두 번째 작업을 `sbatch --dependency=afterany:<jobid>`로 미리 걸어 두면 state를 찾아 smoke 없이 같은 micro-batch로
    이어 가고, 첫 작업이 `ckpt_q100`까지 끝냈으면 바로 종료한다.
  - 데이터 순서·학습률·속도 KD 추출은 모두 step 번호의 결정적 함수라서 이어 가도 같은 학습이다.

## 2. 데이터 준비

| 데이터 | 상태 |
|---|---|
| `kd_r5` 매니페스트 | `minitron/code/make_kd_r5.py` → `outputs/minitron/kd_sets/kd_r5.parquet`(+ `kd_r5_config.json`) 완료 |
| `kd_r5` 카메라 캐시 | NEURON에 이미 27,540개가 있음(예전 학습 세트, sha256 동일 확인). **NEURON에 없는 21,493개만** cvlab21에서 만들어 보낸다(`outputs/minitron/kd_r5_neuron_missing.parquet`). `build_cache.py`를 클립 단위로 묶어 한 클립의 영상을 한 번만 받도록 고쳤다(window 10개 바이트 동일 확인, 신규 window 분당 31개 → 137개). 로그: `outputs/minitron/kd_r5_cache_build.log` |
| teacher CoC | `coc_train.parquet`에 모든 window가 있음(R1 통과 window의 CoC) |
| LingoQA train | 로컬 `/mnt/nvme1n1/ad_vla/data/lingoqa/train`(QA 148,506개, 세그먼트 3,508개, 각 5프레임). 이 중 약 16k개를 고르는 매니페스트를 만든다 |

## 3. 코드 작업

1. **`train_kd.py`.**
   - (a) 온라인 teacher 경로를 재검증한다. rope_deltas 수정 이후 처음 실행하는 것이다.
   - (b) `--kl-groups`를 추가한다.
   - (c) VQA 샘플 경로를 추가한다. 입력은 `run_lingo_vqa.py`와 같은 메시지와 프레임 선택을 쓴다. micro-batch는 종류별로
     같은 것끼리 묶고, 손실은 global step의 샘플 수로 정규화한다.
   - (d) 주행 매니페스트를 `kd_r5`로 바꾸고, held-out을 클립 단위로 분리한다.
   - (e) 25/50/75/100% 지점에서 체크포인트를 저장한다.
2. **`make_vqa_set.py`**(가칭). LingoQA에서 약 16k개를 고르고, held-out 세그먼트를 지정한다.
3. **`kd_job.sbatch`.**
   - 온라인 teacher, world 4, micro-batch smoke(주행 6/4/3/2/1)를 넣는다. 재시작 연쇄(위 1절)도 처리한다.
   - wandb 오프라인 fallback, `--comment field=efficientai;appl=pytorch`, `amd_a100nv_8`, 최대 2일.
4. **cvlab21 smoke.** Blackwell 0–3에서 온라인 teacher + 그룹 KL + VQA 혼합 경로를 몇 step 돌린다. **완료(2026-10-26)**: 2 step, 오류·멈춤 없음, step 2 주행 KL 1.218 / v_mse 0.047 / VQA KL 0.516, held-out 1.266 / 0.038 / 0.456, 최대 42 GiB/rank, 241 s/step. 로그 `outputs/minitron/mtw24_kd2_smoke.log`.
5. **`eval_kd.sh`.** 선택 단계가 `ckpt_epoch{1,2,3}` 대신 25/50/75/100% 체크포인트를 받도록 일반화한다. **완료(2026-10-26)**: `CKPTS="ckpt_q25 ckpt_q50 ckpt_q75 ckpt_q100" bash minitron/code/eval_kd.sh mtw24_kd2`, `select_ckpt.py --ckpts ...`. R0 rows로 돌려 기존 선택(epoch 3)과 수치가 소수 넷째 자리까지 같음을 확인했다.

## 4. KISTI 전송과 실행

- **인증.** OTP 소켓을 연다(`! ssh -fN neuron-dm`). 사용자 조작이 필요하다.
- **전송.** `push_neuron.sh <user> code kd2`. 코드(작업 트리, 141 MB), NEURON에 없는 `kd_r5` npz 21,493개(약 115 GB,
  `--ignore-existing`), LingoQA train 이미지 전체(7.9 GB), `kd_sets` 매니페스트, `slim_mtw24/slim_meta.json`.
  검증은 파일 수, `--apparent-size`, sha256으로 한다.
- **작업 제출.** Datamover에서 `ssh glogin01`이 막혀 있어(2026-10-26 timeout) 로그인 노드 OTP(`ssh -fN neuron`)가 따로 필요하다.
- **실행 순서.** KISTI smoke(3 step, micro-batch 결정, s/step 측정) → 본 학습 → 체크포인트를 cvlab21로 회수(약 17 GB × 4).
- **실행 기록(2026-10-27).** 전송 검증(kd_r5 49,033/49,033, 크기 전부 일치, sha256 6개 일치, LingoQA 17,540개 바이트
  일치, 로그인 노드 preflight 누락 0) 후 `READY_KD2` 생성.
  - job **948566** 11:23 KST 시작(gpu39, 대기 없음). smoke에서 micro-batch 6이 바로 통과(최대 60.6 GB/rank),
    **약 31 s/step**. 본 학습 11:35 KST 시작: 주행 48,716 + VQA 16,239, held-out 316 + 256, 캐시 누락 0, **1,014 step**,
    deadline 10-29 10:23 KST. 예상 종료 약 21:30–22:00 KST(8.7시간 + held-out 평가·저장).
  - job **948567**은 `afterany:948566` 후속. 948566이 `ckpt_q100`까지 쓰면 바로 종료한다.
  - cvlab21 Blackwell(PCIe, micro-batch 1) 241 s/step 대비 약 7.8배 빠르다.
  - **불안정 사건(step 223–247).** VLM gradient norm이 4 → 11 → 22 → 8 → **101**(step 227)로 튄 뒤 학습 KL 중앙값이
    0.29(step 151–200)에서 0.43으로, held-out KL이 0.316(step 200)에서 0.515(step 250)로 올랐고 step 350에도 0.488이다.
    v_mse도 함께 나빠졌다(held-out 0.082 → 0.199 → 0.140). 사건 step들의 샘플을 재구성해 봤지만 길이·문자 모두
    평범해, 데이터 한 건이 아니라 최적화 불안정으로 본다. R0는 같은 최고 학습률에서 이런 사건이 없었다. 추정 원인은
    더 긴 고학습률 구간(step 225에서 R0 약 6.3e-5 vs 9.2e-5)과, 그룹 KL·VQA 혼합으로 커진 gradient(중앙값 약 2 vs 1)지만
    합산 실행이라 분리할 수 없다. 첫 저장(step 254)부터 모두 사건 이후 상태다.
  - **결정(2026-10-27 14시경 KST, 사용자): 끝까지 진행.** 보고서에서 R0와 비교할 때 이 사건을 함께 적는다.

## 5. 평가 (cvlab21 Ada)

- **선택.** 4개 체크포인트 중 val500 minADE@6가 가장 낮은 것을 고른다.
- **보고.**
  - test_2000, OOD-Val262(저장 rows와 페어드)
  - LingoQA(전체, "현재 행동" 질문, 궤적 토큰 누출 수)
  - CoC 퇴화율
- **비교 대상.** R0(`mtw24_kd`), dense, MoOP zs, MT-W zs.

## 6. 판정 (사전 등록, R0 대비 합산 효과)

- **주행.** test_2000 `mtw24_kd2` − R0 페어드 차이의 95% CI 상한이 0보다 작으면 "개선"이다.
- **LingoQA.** `mtw24_kd2` − R0의 segment-clustered CI 하한이 0보다 크면 "개선"이다.
- 두 지표가 반대로 움직이면 그대로 보고한다. 이 경우 원인은 ablation으로만 가를 수 있다.
