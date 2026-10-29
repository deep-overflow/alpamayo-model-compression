# KISTI(NEURON) 학습 계획: MT-W + KD 데이터 확대 (2026-10-26, 승인 대기)

## 0. 목적과 가설

- **H1(데이터).** MT-W + KD에는 dense 대비 +0.115 m(test_2000 minADE@6) 격차가 남아 있다. 이 격차의 상당 부분이
  데이터 부족 때문이라면, 주행 데이터를 약 5배(클립당 window 1개 → 5개) 늘릴 때 격차가 줄어야 한다.
  - 근거: val500이 epoch 2 → 3에서 0.967 → 0.867로 좋아졌고, held-out KL도 끝까지 내려갔다. 학습률 감쇠와 겹치므로
    이것만으로는 원인이 분리되지 않는다.
  - D10-1에 사전 등록한 확장 조건("곡선이 끝까지 오르면 클립당 5 시점, 1 epoch")이 충족됐다.
- **H2(레시피).** LingoQA 하락(57.4%, "현재 행동" 질문에서 3%)은 두 원인에서 왔다. 공식 레시피대로 고치면 LingoQA가
  회복되고 주행 지표는 나빠지지 않아야 한다.
  - 원인 1: 궤적 토큰이 KL 위치의 약 88%를 차지했다.
  - 원인 2: VQA 데이터가 없었다.
  - 고치는 방법: 궤적·텍스트 그룹별 평균 KL, 그리고 LingoQA train 추가.
- **학습 장소.** KISTI NEURON A100 80GB × 4(계정 한도). 평가는 지금처럼 cvlab21 Ada에서 한다. 저장된 비교
  rows와 같은 아키텍처여야 하기 때문이다.

## 1. arm (KISTI에서 순차 실행)

| arm | 데이터 | 손실 | 답하는 질문 |
|---|---|---|---|
| **B (본안, 먼저)** | 주행 R1 window 클립당 5개(약 4.9만) + LingoQA train 일부(§2) | **그룹별 평균 KL**(궤적 그룹 + 텍스트 그룹, 1:1) + 속도 MSE | 공식 레시피에 맞춘 최종 KD |
| **A (ablation)** | 주행 R1 window 클립당 5개만 | 현재와 같음(모든 위치 평균 KL) + 속도 MSE | H1: 데이터만 늘렸을 때의 효과 |

- **같게 두는 것:**
  - 학생 초기화 `slim_mtw24`. `calib_1024`로 잰 중요도, 24.0%를 그대로 쓴다.
  - KI: KV stop-gradient.
  - 속도 KD는 D1 규칙(GT 보간점, t는 추론 격자에서 균등)을 따른다.
  - 옵티마이저(AdamW 0.9/0.95, wd 0.1)와 학습률(1e-4 → 1e-5 cosine).
  - global batch 64.
- **기존 run과의 비교.** 기존 `mtw24_kd`(클립당 1개 × 3 epoch, 샘플 약 2.9만 회)가 기준이다.
  - A와 B는 1 epoch로, 샘플이 약 4.9만~6.5만 회라 연산량이 1.7~2.2배다.
  - A에서 효과가 크게 나오면, 연산량을 맞춘 대조군(클립당 1개 × 5 epoch)을 따로 검토한다.
- **MoOP + KD.** 공정 비교를 위해 확정된 레시피(B 또는 A)로 나중에 같은 파이프라인에서 돌린다. 이 계획의 범위 밖이다.

## 2. 데이터

| 데이터 | 규모 | 상태 |
|---|---|---|
| 주행 `kd_r5` | R1 통과 window 106,291개(9,905개 클립) 중 클립당 최대 5개를 무작위로 뽑음(seed 0) → **49,033개**. GT 버킷: cruise 51.9 / turn 19.6 / decel_stop 16.7 / accel 11.8% | 7,746개는 캐시돼 있음, **41,287개를 스트리밍해야 함**(약 240 GB). teacher CoC는 `coc_train`에 이미 있음 |
| LingoQA train | QA 148,506개, 세그먼트 3,508개(각 5프레임) | 로컬 `/mnt/nvme1n1/ad_vla/data/lingoqa/train` (16 GB) |
| held-out(손실 모니터링) | 주행: 기존 64개 클립의 모든 window. VQA: 세그먼트 32개의 QA 전부 | — |

- **LingoQA 섞는 비율(B만):** 주행 대 VQA를 샘플 기준 약 3:1로 한다. 즉 QA 약 16,000개를 세그먼트 단위로 고르게 뽑는다.
  공식 1.5 학습은 LingoQA를 포함했고, Bolt는 SFT blend를 잃은 능력 쪽으로 재가중했다. ShortOPD에서는 일반 프롬프트를
  빼면 성능이 48.5 → 39.0으로 떨어졌다.
- **VQA 입력.** `run_lingo_vqa.py`와 같은 `helper.create_vqa_message`(전방 카메라, 세그먼트 프레임)를 쓴다.
- **VQA KL 위치.** 답 토큰 + `<|im_end|>`, teacher-forced다. 문맥은 **GT 답**이다(D12-3). VQA 샘플에는 속도 손실을
  걸지 않는다(expert 미사용).
- **보류.** 다음 데이터는 효과가 우리 지표에 잘 잡히지 않거나 OOD 지표를 흐리므로 이번 범위에서 뺐다.
  - 클립 수 확대(teacher CoC 생성 필요)
  - 공개 주행 QA 데이터셋(DriveLM, nuScenesQA 등)
  - OOD-train 1,271개

## 3. 코드 작업 (cvlab21에서 구현하고 smoke까지 확인)

1. **`make_kd_set.py` 확장.** `kd_r5` 매니페스트를 만든다(클립당 5개, seed 고정). held-out은 클립 단위로 분리한다.
2. **`train_kd.py`.**
   - **온라인 teacher 모드 재검증.** A100 80GB에서는 teacher bf16을 4장에 샤딩해 카드당 약 5 GB면 된다. rope_deltas
     수정 이후 온라인 경로는 아직 돌려 보지 않았으므로 smoke로 확인한다. KISTI는 NVLink라 HSDP가 필요 없다
     (`--shard-group 0`).
   - **`--kl-groups`.** KL = 궤적 그룹 위치 평균 + 텍스트 그룹 위치 평균이다. 공식 `sft_base_model.py:388-407`과 같은
     분할이다. 궤적 그룹은 궤적 어휘 + `traj_future_start`이고, 텍스트 그룹은 CoC + `cot_end` + VQA 답이다.
   - **VQA 샘플 경로.** LingoQA 로더를 붙이고, micro-batch는 종류별로 같은 것끼리 묶는다(주행끼리, VQA끼리). VQA는
     KL만 쓴다.
   - **주기적 체크포인트.** 학습의 25/50/75/100% 지점에 저장한다. 체크포인트 선택과 2일 제한, 재시작에 대비한 것이다.
     옵티마이저 상태도 함께 저장해 이어서 학습할 수 있게 한다.
3. **`kd_job.sbatch` 갱신.**
   - 온라인 teacher, world 4, micro-batch를 smoke로 정한다(후보 8/4/2/1).
   - 주기적 체크포인트, wandb 오프라인 fallback.
   - `--comment field=efficientai;appl=pytorch`, 파티션 `amd_a100nv_8`, 최대 2일.
4. **cvlab21 smoke.** 온라인 모드 + 그룹 KL + VQA 경로를 Blackwell 0–3에서 몇 step 돌려 확인한다(micro 1).

## 4. 데이터 준비와 전송

1. **카메라 캐시 스트리밍(cvlab21, GPU 불필요).** 41,287개를 `build_cache.py`(토큰 env 래퍼)로 받는다. 이전 속도
   (8,600개에 약 7시간, 16 worker)로 보면 약 34시간이다. worker를 늘려 줄이는 것을 시도한다. **계획을 승인받으면 가장
   먼저 시작한다**(병목).
2. **전송(Datamover `neuron-dm`, OTP 필요).**
   - 대상: 코드(`push_neuron.sh` minitron tier), 새 npz 약 240 GB, LingoQA 이미지와 parquet, 매니페스트.
   - 검증: 메모리 규칙대로 파일 수, `du --apparent-size`, sha256으로 한다.
3. **KISTI에 이미 있는 것.** `kd_r1` npz 9,905개(57 GB, 2026-10-25 검증), `slim_mtw24/slim_meta.json`, venv(torch 2.8.0,
   transformers 4.57.1).

## 5. 실행

1. **KISTI smoke.** 3 step, micro-batch를 정하고 s/step을 측정한다.
2. **arm B 본 학습**, 이어서 **arm A**. 계정 한도가 4장이라 순차로 돌린다.
   - 시간은 smoke 실측으로 정한다. 추정치는 B가 약 1,000 step, A가 약 770 step이다.
3. **체크포인트 회수.** cvlab21로 가져온다(arm당 약 17 GB × 4).

## 6. 평가 (cvlab21 Ada, `eval_kd.sh` 확장)

- **선택.** 주기적 체크포인트 4개 중 val500 minADE@6가 가장 낮은 것을 고른다(사전 등록 규칙과 같음).
- **보고.**
  - test_2000, OOD-Val262(개루프, 저장 rows와 페어드 비교)
  - LingoQA(전체 + "현재 행동" 질문 + 궤적 토큰 누출 수)
  - CoC 퇴화율
- **비교 대상.** dense, MoOP zs, MT-W zs, 기존 MT-W + KD(`mtw24_kd`).
- **폐루프.** 이번에는 하지 않는다(사용자 결정). 필요하면 패치된 하네스로 나중에 돌린다.

## 7. 판정 기준 (사전 등록)

- **H1(A − 기존 `mtw24_kd`, test_2000).** 페어드 평균 차이의 95% CI 상한이 0보다 작으면 "데이터 확대가 주행을
  개선한다"로 본다.
- **H2(B).** 두 조건을 모두 만족하면 "레시피 수정이 VQA를 회복하고 주행을 해치지 않는다"로 본다.
  - LingoQA: B − 기존 `mtw24_kd`의 segment-clustered CI 하한이 0보다 크다.
  - 주행: B − A(test_2000)의 CI 상한이 +0.03 m보다 작다.

## 8. 결정 필요 (D12)

1. **D12-1 데이터 구성.** 주행 클립당 5개 + LingoQA 약 16k(3:1)로 할지. 추천은 그렇게 하는 것이다. 클립 확대와
   공개 QA는 다음 단계로 미룬다.
2. **D12-2 손실.** arm B에 그룹별 평균 KL을 쓸지. 추천은 쓰는 것이다(공식 레시피와 같음).
3. **D12-3 VQA 문맥.** GT 답(추천)과 teacher 생성 답 중 무엇을 쓸지. teacher 생성 답은 생성 패스가 추가로 든다.
4. **D12-4 arm 순서.** B 먼저, A 다음(추천). 또는 둘 중 하나만 돌린다.
5. **D12-5 체크포인트.** 25/50/75/100% 지점의 4개 중 val500으로 선택한다(추천).
