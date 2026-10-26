# Alpamayo 공식 학습 레시피 확인 (NVlabs/alpamayo-recipes, 2026-10-26)

소스는 `/home/cvlab21/project/chan/alpamayo-recipes`다. 아래 내용은 모두 코드와 설정 파일에서 직접 확인했다.

## 1. 레시피 구성

| 레시피 | 내용 |
|---|---|
| `alpamayo1_5_sft` | 1.5 SFT. HF Trainer + DeepSpeed ZeRO-2, 2단계 |
| `alpamayo1_x_rl` | 1/1.5 open-loop RL. Cosmos-RL + GRPO, 이산 궤적 토큰 VLM |
| `alpamayo1_5_quant` | 1.5 PTQ. ModelOpt FP8 / AutoQuant(FP8 + NVFP4) |
| `alpamayo1_sft`, `alpamayo2_sft` | Alpamayo 1 / 2 SFT |

## 2. Alpamayo 1.5 SFT (`recipes/alpamayo1_5_sft`)

**2단계 파이프라인** (README).

1. **Stage 1: VLM만 학습.** 궤적 과제는 미래 궤적 토큰, VQA 과제는 답 텍스트를 지도한다.
2. **Stage 2: VLM을 얼리고 action expert(flow matching)만 학습.**
   - 구현은 `models/sft_alpamayo_r1.py`다. `cotrain_vlm=False`면 VLM forward를 `torch.no_grad()`로 돌린다.
   - KV 캐시를 마지막 `<traj_future_start>`+1까지 자른 뒤 expert가 읽는다.
   - `stop_grad_from_vlm=True`(기본)면 KV를 detach한다.
   - `cotrain_vlm=True`면 손실은 `FM 손실 + VLM CE`의 단순 합이다(가중치 없음).

**Stage-1 지도 위치** (`configs/vla_processor/*.yaml`). 입력 요소는 image, traj_history, route, question, prompt이고,
출력 요소는 cot, meta_action, traj_future, answer다.

| 설정 | 입력 순서 | 지도(label) |
|---|---|---|
| `default` (A1 궤적 SFT) | image, traj_history, prompt, traj_future | `traj_future` |
| `nav` | image, traj_history, route, prompt, traj_future | `traj_future` |
| `vqa` | image, question, answer | `answer` |

공개 설정 중 CoC(`cot`)를 지도하는 것은 없다.

**Stage-1 손실** (`models/sft_base_model.py:388-407`).
- 두 그룹으로 나눠 **그룹마다 평균한 CE를 같은 가중치로 더한다**.
  - `future_traj`: 궤적 어휘(`future_token_start_idx` ~ +`traj_vocab_size`) + `traj_future_start` + `traj_future_end`
  - `others`: 그 밖의 지도 토큰(CoC, 답 등)
- 즉 손실은 `mean CE(궤적 토큰) + mean CE(나머지)`다.

**Flow matching**(Stage 2, `alpamayo_r1/diffusion/flow_matching.py`; 1.5 config의 `diffusion_cfg`는 sampler를 따로
지정하지 않아 기본값을 쓴다).
- `t = 0.999 − 0.999·u`, `u ~ Beta(1.5, 1.0)`. 연속값이고, 노이즈 쪽(t → 0)으로 치우친다.
- `x_t = t·x + (1−t)·noise`, 목표 `v = x − noise`, MSE.

**하이퍼파라미터**(`configs/sft_base.yaml`, `sft_stage1_*.yaml`).

| | base | Stage 1 (nav / lingoqa) |
|---|---|---|
| LR | 1e-4 | **1e-5** |
| warmup | 100 | 500 |
| 스케줄 | cosine, 최소 1e-6 | 같음 |
| visual LR 배율 | — | **0.1** (nav) |
| GPU당 배치 × 누적 | 1 × 1 | 1 × 4 |
| epoch | 3 | (nav 데모 700) |
| 정밀도·메모리 | bf16, ZeRO-2, gradient checkpointing | 같음 |

- 옵티마이저는 HF Trainer 기본값인 AdamW(β = 0.9, 0.999)다.
- weight decay는 설정에 없어 0이다(lr_multiplier 경로도 `args.weight_decay` 사용).
- gradient clipping은 HF 기본 1.0이다.

**데이터.** README에 "LingoQA was included in training for the released Alpamayo 1.5 model"이라고 적혀 있다. LingoQA
Scenery train은 QA 148k개, 세그먼트 3.5k개다.

## 3. RL (`recipes/alpamayo1_x_rl`)

- "VLM backbone with discrete action-token variant"를 GRPO(Cosmos-RL)로 학습한다.
- 보상은 궤적 L2(motion)이고, 선택적으로 Lingo-Judge로 매긴 추론 보상(`reward.reasoning_weight`)을 더한다.
- 즉 1.5는 VLM의 **이산 궤적 토큰 출력이 RL로 다듬어진** 모델이다.

## 4. 양자화 (`recipes/alpamayo1_5_quant`)

- ModelOpt **PTQ**만 있고 재학습은 없다.
  - calibration은 `max`(또는 smoothquant)이고, `data/0417_5k_train_set_for_calibration_25.10.parquet`에서 100개 클립을
    쓴다.
- 파라미터 메모리: BF16 약 22 GB, FP8 약 11 GB(2.00×), AutoQuant 6.5 bits 약 9 GB(2.44×).
- 평가 지표는 minADE다.

## 4b. Alpamayo 1.5 학습 데이터

출처는 HF 모델 카드다(revision `7aba8293`, 로컬 `cache/hub/models--nvidia--Alpamayo-1.5-10B/.../README.md`의
§Training Dataset). 카드는 데이터 묶음을 단계(SFT, RL)별로 나누지 않는다.

- **초기화**: Cosmos-Reason2-8B(VLM backbone).
- **구성**: CoC 추론 trace, Cosmos-Reason Physical AI 데이터셋, NVIDIA 내부 자율주행 데이터, 공개 주행 데이터.
- **주행 데이터**:
  - 다중 카메라 영상 80,000시간(이미지 10억 장 이상). 카메라는 차량당 RGB 2–6대이고 IMU와 GPS가 있다.
  - egomotion과 궤적(미래 waypoint)은 10 Hz다.
- **CoC trace 300만 개**(영어, 구조화 형식). 사람이 단 구조화 CoC와, VLM 자동 라벨·휴리스틱으로 만든 것이 섞여 있다.
- **텍스트**: 10억 토큰 미만이다(CoC trace, Cosmos-Reason 학습 데이터, 공개 데이터셋).
- **공개 데이터셋 17개**: CODA-LM, Drive-Action, DriveGPT4, DriveLM, **LingoQA**, MapLM, MM-AU, NAVSIM-ReCogDrive,
  NAVSIM-Traj, nuInstruct, nuScenesQA, Omnidrive, Roadwork, Senna, SUTD, Talk2Car, W3DA.
- **HF 메타데이터의 datasets**: `nvidia/PhysicalAI-Autonomous-Vehicles`, `nvidia/PhysicalAI-Autonomous-Vehicles-NuRec`.
- **보고 벤치마크**:
  - LingoQA Lingo-Judge 74.2.
  - AlpaSim 913개 장면(PhysicalAI-AV-NuRec) 1.37 ± 0.10.
  - PhysicalAI-AV의 어려운 샘플 1,434개에서 개루프 minADE_6(6.4 s) 0.916 m.
- **레시피 레포의 데이터 파일**(`data/`):
  - `0417_5k_train_set_for_calibration_25.10.parquet`: 5,000개 클립, 모두 공식 train split. quant 레시피의
    calibration용이다.
  - `1005_7cam_gold_eval_metadb_public.parquet`: 644개 클립, 이벤트 t0 667개, quant 평가용이다. 공식 split이
    train 325 / val 179 / test 140으로 섞여 있다. 따라서 PAI의 split 라벨만으로는 학습 사용 여부를 알 수 없다.

## 5. 우리 KD 레시피(`mtw24_kd`)와 다른 점

| 항목 | 공식 레시피 | 우리 KD | 시사점 |
|---|---|---|---|
| VLM 손실 가중 | 궤적 그룹 평균 + 텍스트 그룹 평균(1:1) | 모든 지도 위치 평균(궤적 토큰이 약 88%) | **텍스트가 과소 가중됐다.** LingoQA의 궤적 토큰 누출과 맞물린다 → 그룹 평균 KL로 바꾸는 것이 1순위 수정 후보 |
| VQA 데이터 | LingoQA train 포함 | 없음 | LingoQA 하락의 데이터 원인이다. KD 데이터에 LingoQA train(teacher 답 logit)을 섞는다 |
| 단계 | VLM 먼저, 그다음 VLM을 얼리고 expert | 동시 학습 + KV stop-gradient(= `cotrain_vlm=True`, `stop_grad_from_vlm=True`) | 손실 합·KI는 공식 옵션과 같다. 2단계로 하면 expert가 최종 VLM 캐시에 맞춰진다(검토 후보) |
| FM t 분포 | 연속, 0.999·(1−Beta(1.5,1)) | 추론 격자 {0, …, 0.9}에서 균등(D1) | 다르다. D1은 추론 지점을 직접 맞추려는 선택이다. 공식 분포는 ablation 후보 |
| LR | Stage 1: 1e-5, visual 0.1배 | 1e-4 → 1e-5(Minitron), visual도 같은 LR | 우리는 프루닝 복구라 높은 LR이 맞다(Minitron). visual 0.1배는 검토 가능 |
| 옵티마이저 | AdamW β(0.9, 0.999), wd 0 | AdamW β(0.9, 0.95), wd 0.1 (Megatron 기본) | Minitron 쪽 설정을 유지한 것이다 |
| expert 문맥 | `<traj_future_start>`까지 자른 캐시 | CoC 끝(= `cot_end`, `traj_future_start` 포함)까지 | 같다 |
