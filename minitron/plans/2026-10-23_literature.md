# Minitron 베이스라인 — 결정 항목별 문헌 조사와 방법 추천, 그리고 VLA 전용 방법론 (2026-10-23)

계획서 `minitron/plans/2026-10-23_plan.md`의 D1–D11을 문헌으로 다시 본 결과다.
**추천은 여전히 논의 출발점이며 결정이 아니다.**

- 조사 방식: 주제별 조사 4갈래(확산·flow 증류 / LLM 프루닝+증류 / VLM 압축·증류 / VLA·주행 VLA 압축).
  모든 인용은 arXiv 초록·본문 또는 학회 페이지에서 실존을 확인했다.
- 직접 재확인한 것: 결론을 좌우하는 9편은 내가 원문으로 다시 봤다 — Shallow-π, Multi-Teacher OPD,
  Offline Hidden-State Distillation, Alpamayo-R1의 KV 캐시 stop-gradient, RLRC, MuCRASP, SPAR,
  FlashDrive, SlimVLM Table 4.
- 2026년 논문 다수는 초록 기준이다. 본문에서 확인하지 못한 세부는 "(초록 기준)" 또는 "미확인"으로 적었다.
- 표기: **확립** = 논문에 있는 결과, **추론** = 우리 설정에 대한 내 해석.

---

## 0. 요약 — 문헌을 본 뒤 추천이 바뀐 곳

| 항목 | 이전 추천 | 문헌 반영 추천 | 바뀐 이유 |
|---|---|---|---|
| D1 비교 지점 | (ii) teacher 추론 경로 | **(i) 학습식 보간점, t는 추론 격자 10개에서 균등** | 구조를 줄인 denoiser를 증류로 회복한 논문(BK-SDM, TinyFusion, SnapGen, NanoFLUX, Shallow-π)이 **전부 (i)** 이다. (ii)의 선례는 데이터 없는 증류(DKDM) 하나뿐이고 실데이터 학습보다 못했다 |
| D1 손실·가중 | 속도 MSE, 동적 α | 그대로 | 분산이 같은 가우시안의 KL은 MSE와 같다(DiffusionOPD). 특징·attention 손실은 불안정하다(BK-SDM 악화, TinyFusion NaN, Shallow-π 붕괴) |
| D2 KL 위치 | CoC + 종료 토큰, teacher 생성 CoC | 그대로, **이산 궤적 토큰 위치 KL을 선택지로 추가** | 응답 토큰만 > 전체 토큰(LLaVADI, LLaVA-MoD). Alpamayo-R1은 VLM을 이산 궤적 토큰 CE로 학습했고, 1.5에도 그 경로가 남아 있다 |
| D3 기울기 경로 | end-to-end | **stop-gradient (expert → VLM 차단)** | **Alpamayo-R1이 바로 이렇게 학습됐다** (원문 확인: "we apply a stop-gradient to the KV-cache produced by the VLM"). π0.5-KI도 같은 설계에서 언어 능력이 나아졌다 |
| D4 hidden 경계 | merger 출력 절단, norm 보정 없음 | 그대로, norm 이득 보정은 1요인 선택지 | MLLM에서 hidden 차원을 자르고 projector를 맞춘 **선례가 없다**. norm 보정의 측정된 이득은 Pangu Light의 +0.7점(21B 토큰 후) 하나뿐 |
| D5 depth 방식 | VLM KV-only 층 | **두 탑에서 짝 층을 함께 제거** | π 계열의 층별 결합 구조에서 두 탑을 함께 줄인 선례가 있다(Shallow-π 18→6층, CLP). KV-only는 선례가 없고 K/V 연산이 남는다 |
| D6 teacher correction | 생략 | 생략 + **teacher 손실 진단** | 근거가 Minitron 한 편뿐이고, 이후 NVIDIA 보고서(Nemotron-H, Nano 2)도 쓰지 않았다 |
| D8 hidden:MLP | W-b (3200/9856) | **W-a (3328/9216)** | Llama-3.1-Minitron-4B-Width의 절단 비율(MLP 절단 = hidden 절단 × 1.44)을 24% 예산에 맞추면 정확히 W-a가 나온다. 논문의 대표 width 모델 패턴이다 |
| D9 캘리브레이션 | 모든 위치 | 그대로(충실 재현), **위험 명시** | vision 토큰이 지배하는 중요도는 VLM zero-shot에서 크게 나쁘다(SlimVLM 73.3% vs 89.5–93.8%). 다만 모두 텍스트 답변 기준이다 |
| D10 예산 | ~1만 샘플 | 그대로, **"Minitron 선택 + 경량 증류"로 명명** | 약 10⁷ 토큰은 문헌상 "경량 회복" 영역이다(약 20–25% 제거까지 작동) |
| D11 비교 | MoOP+KD 추가 | 그대로 + **무작위 선택 + 같은 KD 대조군** | Liu 2019, Blalock 2020: 재학습 방식의 분산이 프루닝 방식 간 분산과 맞먹는다 |

---

## 1. 결정 항목별 근거

### D1. action 출력 증류

| 논문 | 무엇을 했나 | 비교 지점 / 손실 | 수치 |
|---|---|---|---|
| BK-SDM (Kim, ECCV 2024, [2305.15798](https://arxiv.org/abs/2305.15798)) | U-Net 블록 제거 + 증류 | 같은 (z_t, t), t ~ U; 출력 MSE + 특징 MSE, λ=1 | v2-Small FID: task만 16.71 → +출력 KD **14.27** → +특징 KD 16.61 (특징 KD가 악화) |
| TinyFusion (Fang, CVPR 2025, [2412.01199](https://arxiv.org/abs/2412.01199)) | DiT 깊이 프루닝 | 출력 MSE 0.9 + GT 0.1 | FID: 단순 미세조정 5.79 vs 출력 KD **4.66**; 표현 KD는 NaN |
| SnapGen (Hu, CVPR 2025, [2412.09619](https://arxiv.org/abs/2412.09619)) | flow DiT teacher → 작은 U-Net | 같은 x_t, t ~ logit-normal; 속도 MSE + 특징; t별로 KD 항 크기를 task 항에 맞춤 | GenEval 0.61 → 0.66 |
| NanoFLUX (Chavhan, ICML 2026, [2602.06879](https://arxiv.org/abs/2602.06879)) | FLUX 12B → 2.4B (폭 + 블록 병합) | 같은 입력에서 속도 MSE, GT 속도 없음; 학습 이미지를 teacher가 재생성 ((i)과 (iii)의 혼합) | — |
| **Shallow-π** (Jeon, [2601.20262](https://arxiv.org/abs/2601.20262), 2026) | π0/π0.5 **VLM과 action expert를 함께** 18→6층 | task FM + **GT 보간점에서 속도 MSE** + 중간층 action→VL attention KL | LIBERO(6층): task만 93.0 → +KD 93.9 → +attention 94.6. 전 토큰 attention KD는 붕괴(61.5 / 0.0). 2배 이상 빠름 |
| Multi-Teacher OPD (Ni, [2610.21522](https://arxiv.org/abs/2610.21522), 2026) | π0.5 같은 크기 학생, 다중 teacher | **student 자신의 flow 상태**에서 속도 MSE | 끝점(action) 매칭과 teacher rollout SFT보다 LIBERO-40 +6.4–9.5점. 단 환경 rollout도 on-policy라 flow 상태 효과가 분리되지 않음 |
| FastOPD (Oh, [2610.02832](https://arxiv.org/abs/2610.02832)) | π0.5 → SmolVLA, VLM 동결 | student 상태에서 teacher 속도, λ=0.1 (0.01은 붕괴) | 1스텝 77.3% (teacher 97.5%) |
| DKDM (Xiang, CVPR 2025, [2409.03550](https://arxiv.org/abs/2409.03550)) | 데이터 없는 증류 | **teacher 역방향 경로 상태** ((ii)) | CIFAR-10 FID 9.56 vs 실데이터 일반 학습 7.84 |
| DMD2 ([2405.14867](https://arxiv.org/abs/2405.14867)), Imagine Flash ([2405.05224](https://arxiv.org/abs/2405.05224)) | 소수 스텝 학생 | student 자기 출력에서 질의 ((iv)) | 소수 스텝 학생에서는 (iv)가 확실히 도움 |
| InstaFlow ([2309.06380](https://arxiv.org/abs/2309.06380)) / Rectified Flow ([2209.03003](https://arxiv.org/abs/2209.03003)) | teacher 노이즈→샘플 쌍 학습 ((iii)) | — | 같은 노이즈 쌍 학습은 속도장을 **다시 만든다(직선화)** — teacher 압축이 아니다 |
| DiffusionOPD ([2605.15055](https://arxiv.org/abs/2605.15055)) | — | 분산이 같은 가우시안 KL = 평균 MSE / (2σ²) | KL 헤드를 따로 둘 이유가 없음 |

- **확립.**
  - 출력 증류는 GT만 쓰는 회복보다 매번 나았다(BK-SDM, TinyFusion, Shallow-π, SnapGen; Minitron 자신도 KD 37.81 vs 일반 학습 24.57 MMLU).
  - 구조 압축 논문은 모두 (i)에서 자기 파라미터화(ε 또는 속도)로 맞췄다.
  - 특징·attention 손실은 깨지기 쉽다.
  - GT 항을 크게 섞으면 해롭거나 중립이다(Minitron: CE + KL이 KL만보다 약간 나쁨; SnapFusion γ=1이 지배).
- **증거가 없는 곳.**
  - 샘플러(10스텝)를 그대로 두고 **구조만 줄인 학생**에서 (i)/(ii)/(iv)를 비교한 통제 실험은 없다.
  - 학생의 오차가 expert가 아니라 **상류(VLM 캐시)**에 있는 경우도 연구되지 않았다.
- **추론 (가중).** 10스텝 Euler에서 최종 action 오차는 1차 근사로 각 격자점 속도 오차의 합이다.
  그래서 t를 추론 격자 {0, 0.1, …, 0.9}에서 **균등**하게 뽑는 것이 원칙에 맞다. 학습 때 쓴 shifted beta의
  파라미터가 공개돼 있지 않다는 문제도 이 선택으로 피해 간다.
- **추천.**
  - action 증류: 한다.
  - 손실: 속도 MSE만 쓴다(특징·attention 항 없음, GT 항 없음).
  - 비교 지점: (i) GT 보간점, t ~ 격자 균등.
  - 가중: Minitron(2407)의 동적 α = sg(L_KL)/sg(L_v). 감도는 ×0.3 / ×3로 확인한다.
  - (ii)와 (iv)는 ablation으로 둔다.

### D2. 텍스트 logit KL의 위치와 문맥

| 논문 | 결과 |
|---|---|
| Minitron 2407 ([2407.14679](https://arxiv.org/abs/2407.14679)) | 순방향 KL > 역방향(LM loss 2.155 vs 2.665), τ=1 최선, top-K(K≤100) 근사는 "크게 손해". 고정 텍스트의 **모든 위치**에서 측정 |
| LLaVADI (Xu, [2407.19409](https://arxiv.org/abs/2407.19409)) | **답변 토큰 KL은 효과, 전체 토큰 KL은 모든 벤치마크에서 악화.** 마지막 두 층 hidden 매칭은 붕괴. teacher가 다시 생성한 답변은 대체로 도움(MME만 하락), student 생성 답변은 혼재 |
| LLaVA-MoD (Shu, ICLR 2025, [2408.15881](https://arxiv.org/abs/2408.15881)) | 응답만 KL 60.2 > 응답 + 지시문 58.4; KD 60.2 > SFT 55.7 |
| LLaVA-KD (Cai, ICCV 2025, [2410.16236](https://arxiv.org/abs/2410.16236)) | 응답만 57.2, + vision 위치 KL 57.7(+0.5), + 프롬프트 56.9 |
| Seq-level KD (Kim & Rush, EMNLP 2016, [1606.07947](https://arxiv.org/abs/1606.07947)) | teacher 생성 출력 학습 +4.2 BLEU |
| GKD (Agarwal, ICLR 2024, [2306.13649](https://arxiv.org/abs/2306.13649)) | 생성형 학생에서는 on-policy > supervised KD, seq-level KD. greedy 디코딩에서는 발산 종류의 차이가 작다 |
| Qwen3 ([2505.09388](https://arxiv.org/abs/2505.09388)) / **Qwen3-VL** ([2511.21631](https://arxiv.org/abs/2511.21631)) | 작은 모델을 off-policy → on-policy logit 증류로 만든다. Qwen3-VL은 **텍스트 데이터만으로** LLM을 증류 |
| **Alpamayo-R1** ([2511.00088](https://arxiv.org/abs/2511.00088)) | VLM을 **추론 텍스트와 이산 궤적 토큰의 CE**로 학습(이산 토큰은 추론에 미사용) |

- **추론.** 1.5 체크포인트에도 같은 경로가 있다. `traj_tokenizer`가 action을 3000 bin × 128토큰으로 바꾸고,
  어휘에 그 자리가 예약돼 있으며, 릴리스의 logits processor가 CoC 디코딩 때 그 범위를 마스킹한다.
  따라서 **이산 궤적 토큰 위치의 logit KL**은 Minitron 원칙(logit KL) 안에 머물면서 VLM에 action 정보를
  주는 유일한 경로다. D3에서 stop-gradient를 고르면 특히 중요해진다.
  단, dense teacher가 이 위치에서 의미 있는 분포를 내는지는 S0에서 teacher NLL로 확인해야 한다.
- **추천.**
  - 위치: CoC + 종료 토큰. 이산 궤적 128토큰 위치는 선택지로 추가.
  - 문맥: teacher가 생성한 CoC. 이산 궤적 위치는 GT를 이산화한 토큰(Minitron이 실제 텍스트 위에서 KL을 거는 것과 같은 구조).
  - 발산: 순방향 KL, τ=1, 전체 어휘.
  - on-policy 단계(Qwen3식)는 CoC 퇴화가 나타날 때만 고려한다. Minitron 밖의 요소이므로 별도 표기한다.

### D3. 기울기 경로

| 논문 | 결과 |
|---|---|
| **Alpamayo-R1** (원문 확인) | "During training, we apply a stop-gradient to the KV-cache produced by the VLM" — expert 기울기가 VLM 가중치에 닿지 않게 학습 |
| π0.5-KI (Driess, NeurIPS 2025, [2505.23705](https://arxiv.org/abs/2505.23705)) | expert가 읽는 모든 층 K/V에 sg. 이유: "무작위 초기화된 로봇 전용 모듈의 기울기가 사전학습 VLM 가중치와 나쁘게 상호작용". 절연 없는 π0는 테이블 정리에서 7.5배 많은 스텝 필요, backbone 동결은 셔츠 개기 0%, 절연 시 언어 지시 따르기 향상 |
| Offline HSD (Kim, [2609.19579](https://arxiv.org/abs/2609.19579)) | 압축 회복에서 action 손실을 LoRA 단 backbone으로 흘려도 작동 — 단 이 모델들은 언어를 출력하지 않는다 |

- **확립.** 압축·증류 상황에서 KI(절연)와 end-to-end를 비교한 연구는 **없다**(VLA 조사의 공백 5번).
- **추론.** KI의 전제는 "무작위 초기화된 expert"이고, Minitron 학생은 두 탑 다 학습된 teacher에서 출발하므로
  전제는 약해진다. 그러나 이 모델의 VLM은 **한 번도 expert 기울기를 받은 적이 없다.**
  end-to-end는 원 학습에 없던 종류의 업데이트를 새로 들이는 것이다.
- **추천.** stop-gradient(원 모델의 학습 방식 그대로)를 기본으로 한다. VLM에는 텍스트 KL과 (D2의) 이산 궤적
  KL만 들어간다. end-to-end는 ablation으로 둔다.
  이전에 end-to-end를 추천했던 근거("Minitron은 모델 전체를 손실 하나로 학습")는 이렇게 다시 읽는다:
  Minitron은 원래 학습 손실의 구조를 증류 손실로 바꿀 뿐이고, 이 모델의 원래 손실 구조에는 stop-gradient가 들어 있다.

### D4. hidden 채널 축의 경계

| 논문 | 결과 |
|---|---|
| Minitron 2407/2408 | norm 출력으로 채널 순위, norm 가중치는 잘라내기만 함, 보정 없음. 채널 집합은 전 층 공통 1개 |
| Sheared-LLaMA (Xia, ICLR 2024, [2310.06694](https://arxiv.org/abs/2310.06694)) | 전 층 공통 hidden 마스크 |
| SliceGPT (Ashkboos, ICLR 2024, [2401.15024](https://arxiv.org/abs/2401.15024)) | 층별 PCA 회전, skip 연결마다 D×D 행렬 필요 — 축 정렬 방식이 아님 |
| DISP-LLM (Gao, NeurIPS 2024, [2410.11988](https://arxiv.org/abs/2410.11988)) | 층마다 다른 residual 채널 부분집합 |
| Pangu Light ([2505.20155](https://arxiv.org/abs/2505.20155)) | RMSNorm 이득 재조정 γ_new = γ_pruned·‖γ_orig‖/‖γ_pruned‖ — 21B 토큰 후 33.1 → 33.8 (+0.7) |
| Huang (GCPR 2025, [2507.20749](https://arxiv.org/abs/2507.20749)) | MLLM 프루닝에서 ViT는 안 자름(미세조정해도 이득 없음). 15% 압축에서 **projector만 회복해도 95% 이상 유지** |
| LLaVA-KD | projector만 학습하는 증류 단계 하나로 +0.9 |
| ECoFLaP ([2310.02998](https://arxiv.org/abs/2310.02998)), MULTIFLOW ([2404.05621](https://arxiv.org/abs/2404.05621)), UPop ([2301.13741](https://arxiv.org/abs/2301.13741)) | vision 쪽 프루닝이 더 취약하다 — 전역 배분은 vision에 낮은 희소도를 줌, 예산 없이 자르면 초기 vision 층이 사라져 붕괴 |

- **확립.**
  - MLLM에서 **LLM hidden 차원을 자르고 projector를 맞춰 자른 연구는 찾지 못했다.** 증류 논문들은 학생에게
    새 projector를 만들어 준다.
  - ViT는 어떤 MLLM 압축에서도 건드리지 않는다.
- **추론.**
  - Minitron과 일관된 처리는 하나다: merger와 DeepStack merger 3개의 출력 행을 embedding·norm과 **같은 채널 인덱스**로 자르고, ViT는 그대로 둔다.
  - hidden 축소는 k/v_proj의 **입력** 열을 바꾸므로, 캐시 모양은 같아도 expert가 읽는 모든 캐시 값이 바뀐다.
  - 우리 예산(~10⁷ 토큰)에서는 norm 보정의 상대적 가치가 94B 토큰일 때보다 클 수 있다.
- **추천.** 위의 절단(merger + DeepStack 3개, ViT 무손상), norm 보정 없음(충실 재현). Pangu Light식 이득 재조정은
  1요인 선택지로 둔다.

### D5. depth 변형

| 논문 | 결과 |
|---|---|
| Minitron | 연속 블록을 다운스트림 지표로 선택(층 16–31 연속 제거 Winogrande 0.595 vs 층별 중요도 0.50). width > depth 정확도(MMLU 60.5 vs 58.7, GSM8K 41.2 vs 16.8), depth가 더 빠름(2.7× vs 1.8×) |
| Gromov (ICLR 2025, [2403.17887](https://arxiv.org/abs/2403.17887)) | 각거리로 연속 블록 선택 + QLoRA 회복(0.16–0.33B 토큰). 회복은 loss를 되돌리지만 QA는 못 되돌림 |
| Lu ([2411.15558](https://arxiv.org/abs/2411.15558)) | 마지막 층 제거 > Taylor > PPL > BI > 무작위. 1,270–1,500만 토큰에서 lm_head + 마지막 3층 부분 미세조정 > LoRA |
| **SmolVLA** ([2506.01844](https://arxiv.org/abs/2506.01844)) | VLM 앞 절반만 사용: LIBERO 평균 N=16 78.5 vs N=32 80.3 |
| GR00T N1 ([2503.14734](https://arxiv.org/abs/2503.14734)) | LLM 12번째 층 특징 사용(수치 없음) |
| **Shallow-π** | 층별 결합 구조(π0/π0.5)에서 **두 탑을 함께** 18→6층 |
| CLP (Nguyen, [2606.20246](https://arxiv.org/abs/2606.20246)) | CKA로 "쌍둥이" 층을 찾아 VLM과 action head 양쪽에서 제거 후 미세조정. π0 LIBERO 94.6→93.9, 211→152 ms |
| EfficientVLA (Yang, NeurIPS 2025, [2506.10100](https://arxiv.org/abs/2506.10100)) | 1−cos로 비연속 10/32층 제거 + 다른 기법. head가 특징 하나만 읽어서 인터페이스가 자명하게 유지됨 |

- **확립.** expert 깊이를 유지한 채 VLM 층만 뺀 연구는 없다. SmolVLA와 GR00T는 추론 때 텍스트를 생성하지
  않는데, Alpamayo의 CoC는 LM head 앞의 후반 층이 필요하다.
- **추천.** Width를 주 arm으로 한다(Minitron 권장 + 추론 텍스트 출력).
  depth arm을 넣는다면 **층 l을 두 탑에서 함께 제거**한다(Shallow-π, CLP 선례). 블록은 연속으로, probe
  minADE로 고른다. 24% 예산이면 짝 층당 약 2.56억이라 10쌍 = 23.1%다.

### D6. teacher correction

- **확립.**
  - 근거는 Minitron 2408 한 편이다: LM val loss 6% 이상 감소, 증류와 병행해도 동급.
  - 그 뒤 Minitron 레시피를 쓴 NVIDIA 보고서들(Nemotron-H [2504.03624](https://arxiv.org/abs/2504.03624),
    Nemotron Nano 2 [2508.14444](https://arxiv.org/abs/2508.14444))의 압축 절은 teacher correction을 언급하지 않는다.
  - 독립 재현은 없다.
  - Distillation Scaling Laws (Busbridge, ICML 2025, [2502.08606](https://arxiv.org/abs/2502.08606)): teacher는
    자기 CE를 통해서만 학생에 영향을 준다.
- **추천.** 생략하되, teacher의 증류 세트 손실(CoC NLL, 속도 FM)을 held-out과 비교하는 **진단**을 S0에 넣는다.
  correction을 하게 되면 dense 기준 행도 correction된 teacher로 바꿔야 한다.

### D7. 예산 — 24.0% (VLM only)를 주 실험으로 유지. 문헌상 바뀔 근거 없음.

### D8. hidden:MLP 비율

- **확립.**
  - 2408은 수동으로 골랐다(Llama-4B-Width: hidden −25% / MLP −36%; MN-8B: 둘 다 약 −20%).
  - 이후 NVIDIA 보고서는 모두 후보를 짧게 증류해 골랐다(후보당 1.8–19B 토큰). zero-shot 순위가 초반에 뒤집히기 때문이다(2407: 약 300 step부터 안정).
  - 우리 전체 예산은 후보 1개 run의 약 1/60이라, 그 방식의 탐색은 감당할 수 없다.
- **추천.** 수동 선택 **W-a (hidden 3328 / MLP 9216, 제거 2,659,847,424, +0.09%)**. Llama-4B-Width의 절단 비율을
  예산에 맞춘 결과다. W-b(MN-8B식 같은 비율)가 대안이다. "탐색하지 않은 shape"임을 논문에 명시한다.

### D9. 캘리브레이션 위치

| 논문 | 결과 (모두 **텍스트 답변**을 채점) |
|---|---|
| SlimVLM (Wen, [2608.03580](https://arxiv.org/abs/2608.03580)) Table 4 (본문 확인) | LLaVA-1.5-7B, 20% 구조 프루닝, 회복 없음: **모든 vision 토큰 73.30%** / 무작위 부분집합 89.51% / attention 선택 93.80% (FLAP: 76.26 → 89.37) |
| TAMP (Lee, ACL Findings 2025, [2504.09897](https://arxiv.org/abs/2504.09897)) | Wanda 50%: 언어 토큰만으로 잰 활성이 vision + 언어보다 좋음 |
| ATV-Pruning ([2603.16001](https://arxiv.org/abs/2603.16001)) | 60%: 텍스트 토큰 캘리브레이션 61.58 / 혼합 35.85 / vision만 11.11 |
| Huang 2507.20749, Shortened LLaMA ([2402.02834](https://arxiv.org/abs/2402.02834)) | 캘리브레이션 크기는 약 10개에서 포화 |

- **추론.** Minitron의 sequence-mean 집계를 3,086토큰 프롬프트에 그대로 쓰면 vision 위치가 지배한다.
  다만 우리 expert는 vision 위치의 KV를 읽으므로, 텍스트만으로 잰 중요도는 expert가 읽는 채널을 덜 보호할 수 있다.
  위 결과들을 그대로 옮길 수 없는 이유다.
- **추천.** 충실 재현이 기본(모든 위치)이다. zero-shot이 크게 무너지면 "캘리브레이션이 베이스라인을 불리하게
  만든 것"과 구분하기 위해, vision / 프롬프트 텍스트 / CoC 위치를 따로 잰 점수를 함께 저장해 둔다
  (추가 비용 거의 없음).

### D10. 데이터·예산

- **확립.**
  - Minitron: logit KD로 18.9B 토큰에서 MMLU 58.3, 94B에서 62.8.
  - 경량 회복 영역(LLM-Pruner 5만 샘플 LoRA, SliceGPT 약 5천 시퀀스, Lu 1,270–1,500만 토큰, Gromov 0.16–0.33B)은
    **약 20–25% 제거까지** 작동하고, 40% 이상에서는 실패한다(Shortened LLaMA: 60%에서 LoRA PPL 68.9 vs 연속 사전학습 17.1).
  - MLLM: 50% 미만 압축에서는 데이터 5%로 95% 이상 회복(Huang).
- **추론.** 우리 24%는 경량 회복이 통하는 경계 안이다. 다만 LR 1e-4로 적은 데이터에 전체 파라미터를
  학습하면 망각 위험이 있다(Lu는 이 영역에서 부분 층 미세조정을 선호).
- **추천.** 결과는 "Minitron 선택 + 경량 증류"로 부른다(Minitron 회복의 재현이라고 하지 않는다).
  예산은 두 KD arm에 동일하게 주고, 학습 곡선(step별 probe)을 둘 다 기록한다.

### D11. 공정 비교

| 논문 | 결과 |
|---|---|
| Liu, Rethinking the Value of Network Pruning (ICLR 2019, [1810.05270](https://arxiv.org/abs/1810.05270)) | 같은 연산으로 처음부터 학습하면 물려받은 가중치의 미세조정과 같거나 낫다(비정형 ImageNet은 예외) |
| Blalock, State of Neural Network Pruning (MLSys 2020, [2003.03033](https://arxiv.org/abs/2003.03033)) | "미세조정 방식 간 변동이 프루닝 방식 간 변동과 거의 같다", 같은 초기 모델 필수, 대조군 보고 |
| Sheared-LLaMA vs LLM-Pruner | 토큰·데이터·희소도를 맞춰 비교(PPL 6.85 vs 7.09) |
| Minitron 2407 Table 4 | 반대 사례: 94B 토큰 Minitron을 LoRA 회복 LLM-Pruner와 예산 불일치로 비교(MMLU 63.8 vs 25.2) |

- **추천.** {Minitron 선택, MoOP 선택} × {zero-shot, 같은 KD} 2×2에, **무작위 선택 + 같은 KD** 대조군을 하나
  더한다(Minitron shape에서). 무작위 대조군은 "증류 후 결과 중 선택 기준이 기여한 몫"을 잰다.

---

## 2. 가장 가까운 선행 연구 (상세)

### Shallow-π: Knowledge Distillation for Flow-based VLAs (2026, arXiv 2601.20262)
- **Core Contribution.** π0/π0.5의 VLM과 flow action head를 **함께** 18→6층으로 줄이는 증류 프레임워크.
- **Method.** task FM + GT 보간점에서 teacher·학생 속도 MSE + 중간층 action query → VL key attention KL.
- **Results.** 2배 이상 빠름, 성공률 하락 1%p 미만(초록). LIBERO 6층: 93.0 → 93.9(KD) → 94.6(+attention).
- **Relevance.** 층별 결합 구조에서 두 탑을 함께 줄인 유일한 증류 선례 → D1(비교 지점), D5(짝 층 제거).
- **Limitations.** 텍스트 출력 없음, 텍스트 KD 없음, 가중치·t 분포 미기재, 깊이 축만.

### SPAR: Structural Pruning and Action Recovery for VLA Models (2026, arXiv 2610.26487)
- **Core Contribution.** π0.5 LLM backbone의 **hidden 차원과 FFN 차원**을 자른다 — Minitron-Width와 가장 비슷한 VLA 축.
- **Method.** pruning-aware tuning → RL 회복(초록 기준; 증류 미언급, expert 처리 미언급).
- **Results.** LLM 2.34B → 0.09B, 전체 3.37B → 1.12B, LIBERO 84.55% → RL 후 93.45%, 1.98× 빠름.
- **Relevance.** VLA에서 hidden 축 절단이 실제로 작동한다는 근거(D4). 회복은 증류가 아니라 RL이다.
- **Limitations.** 텍스트 생성·추론 보존을 보지 않음. 세부는 본문 미확인.

### Recovering Aggressively Pruned VLA Models with Offline Hidden-State Distillation (2026, arXiv 2609.19579)
- **Core Contribution.** residual 폭을 유지한 width 프루닝 후, teacher hidden state 캐시로 오프라인 증류.
- **Results.** 45% 제거까지는 증류와 지도 회복이 유의차 없음. 63–87%에서 OFT +2.1–4.5점, CogACT +9.4–22.1점.
  63%에서 teacher 대비 3.5점 이내를 약 8 GPU-시간에 회복.
- **Relevance.** (a) 우리 24%는 "증류와 지도 회복이 갈리지 않는" 영역일 수 있다 → D11 대조군이 중요하다.
  (b) residual 폭이 같으면 hidden state를 직접 맞출 수 있다 → §4의 캐시 증류 제안의 근거.
- **Limitations.** 언어 출력 없음. action head는 증류하지 않음.

### Knowledge Insulating VLA Models (π0.5-KI, NeurIPS 2025, arXiv 2505.23705) + Alpamayo-R1 (arXiv 2511.00088)
- **Core Contribution.** expert → VLM 기울기 차단과 이산 action 토큰으로 VLM에 행동 신호를 주는 설계.
  **Alpamayo-R1은 같은 설계로 학습됐다**(KV 캐시 stop-gradient, 추론 텍스트 + 이산 궤적 토큰 CE,
  shifted beta t, 추론 Euler 0.1 간격).
- **Relevance.** D3의 결정적 근거다. D2의 이산 궤적 토큰 KL은 이 설계를 따르는 것이다.

### Minitron (2407.14679, 2408.11796) — 계획서 §1 참조.

---

## 3. VLA 전용 압축·효율 방법론 정리 (2024–2026)

| 범주 | 논문 | 무엇을 줄이나 | 재학습 | 보고 수치 |
|---|---|---|---|---|
| 조기 종료·층 건너뛰기 | DeeR-VLA (Yue, NeurIPS 2024, [2411.02359](https://arxiv.org/abs/2411.02359)) | 동적 다중 출구 LLM | 예 | LLM 연산 5.2–6.5배 감소, CALVIN 4.13 @10 GFLOPs vs 4.07 @31.2 |
| | MoLe-VLA ([2503.20384](https://arxiv.org/abs/2503.20384)) | router로 LLM 층 약 50% 건너뜀 + 자기 증류(CogKD) | 예 | RLBench 57.2 → 60.8. router만 쓰면 56.3(증류 필요) |
| | ActDistill ([2511.18082](https://arxiv.org/abs/2511.18082)) | teacher → 동적 층 라우팅 학생 | 예 | 연산 50% 이상 감소, 1.67× |
| 구조적 프루닝 | EfficientVLA ([2506.10100](https://arxiv.org/abs/2506.10100)) | 층 10/32 + MLP 25% + 토큰 + DiT 캐시 | 아니오 | CogACT 74.8 → 74.2%, 1.93× |
| | CLP ([2606.20246](https://arxiv.org/abs/2606.20246)) | 두 탑의 쌍둥이 층 | 미세조정 | π0 94.6 → 93.9, 211 → 152 ms |
| | **SPAR** ([2610.26487](https://arxiv.org/abs/2610.26487)) | π0.5 LLM hidden + FFN | 튜닝 + RL | LLM 2.34 → 0.09B, 93.45%, 1.98× |
| | **Offline HSD** ([2609.19579](https://arxiv.org/abs/2609.19579)) | LLM head·MLP (residual 유지) | LoRA + hidden KD | §2 |
| | **RLRC** (Chen, RA-L 2026, [2506.17639](https://arxiv.org/abs/2506.17639)) | OpenVLA LLM 구조 프루닝 → SFT + RL → 양자화 | 예 | 메모리 최대 8배 감소, 2.3× |
| | Zhang (ECCV 2026, [2606.31382](https://arxiv.org/abs/2606.31382)) | OpenVLA·π0.5 모듈 간 공동 프루닝 | 없음 | 12–30% 제거에 약 90% 유지; 기존 기준들은 회복 없이 붕괴 |
| | EcoVLA ([2602.00780](https://arxiv.org/abs/2602.00780)), AdaVLA ([2608.29208](https://arxiv.org/abs/2608.29208)), GLUESTICK ([2510.08464](https://arxiv.org/abs/2510.08464)) | 채널 / flow 곡률 기반 스텝·MLP / 비정형 | 아니오 | 1.6–2.24× |
| 토큰 프루닝·캐시 | VLA-Cache ([2502.02175](https://arxiv.org/abs/2502.02175)), SP-VLA ([2506.12723](https://arxiv.org/abs/2506.12723)), SpecPrune-VLA ([2509.05614](https://arxiv.org/abs/2509.05614)), CogVLA ([2508.21046](https://arxiv.org/abs/2508.21046)) | 시각 토큰 재사용·절단 | 대부분 아니오 | 1.5–2.8× |
| | Prune2Drive (CVPR 2026, [2508.13305](https://arxiv.org/abs/2508.13305)) | 주행 VLM 다중 시점 토큰 | 아니오 | 토큰 10% 유지, prefill 6.40×, DriveLM 약 −3% |
| 양자화 | QAIL ([2412.01034](https://arxiv.org/abs/2412.01034)), BitVLA ([2506.07530](https://arxiv.org/abs/2506.07530)), SQAP-VLA ([2509.09090](https://arxiv.org/abs/2509.09090)), QuantVLA ([2602.20309](https://arxiv.org/abs/2602.20309)) | QAT / 1.58bit / W4A4 / DiT 포함 PTQ | 혼재 | 주행 W4A4 3.7×(QAIL) 등 |
| action head 스텝 축소 | SnapFlow ([2604.05656](https://arxiv.org/abs/2604.05656)), FastOPD ([2610.02832](https://arxiv.org/abs/2610.02832)), Coda ([2609.21216](https://arxiv.org/abs/2609.21216)) | 10 → 1–2스텝 | 예 | π0.5 1스텝 98.75% vs teacher 97.75%(SnapFlow) |
| 디코딩 가속 | Spec-VLA ([2507.22424](https://arxiv.org/abs/2507.22424)), PD-VLA ([2503.02310](https://arxiv.org/abs/2503.02310)), CEED-VLA ([2506.13725](https://arxiv.org/abs/2506.13725)) | 추측·병렬 디코딩 | 일부 | 1.4–4× |
| **주행 VLA (Alpamayo)** | **FlashDrive** (Li, [2608.12932](https://arxiv.org/abs/2608.12932)) | **Alpamayo 1.5-10B**: 스트리밍 KV 재사용, diffusion drafter 추측 디코딩, flow 스텝 캐싱, W4A8, CUDA Graph | 일부 | **717 → 151 ms (4.7×)**, minADE6 0.08 m 변화, 폐루프 충돌·이탈 개선 |
| | Dinh ([2606.31160](https://arxiv.org/abs/2606.31160)) | Alpamayo CoC 추측 디코딩 + RL | 예 | 추론 단계 약 4× |
| 작은 학생으로 증류 | Shallow-π, VLA-AD ([2605.16241](https://arxiv.org/abs/2605.16241), OpenVLA-7B → 158M), RT-VLA ([2606.14010](https://arxiv.org/abs/2606.14010), 주행 SimLingo → 소형) | — | 예 | VLA-AD 3.28×, RT-VLA 1544 → 196 ms |
| | DiMA ([2501.09757](https://arxiv.org/abs/2501.09757)), VLM-AD ([2412.14446](https://arxiv.org/abs/2412.14446)), VERDI ([2505.15925](https://arxiv.org/abs/2505.15925)), DistillDrive ([2508.05402](https://arxiv.org/abs/2508.05402)) | MLLM이 학습 때만 비-VLM planner를 가르침 | 예 | DiMA L2 −37%, 충돌 −80% |
| 처음부터 작게 | TinyVLA, SmolVLA, NORA, EdgeVLA, VLA-Adapter, GR00T N1 | — | — | — |
| 추론 인지 프루닝 (VLM) | MuCRASP (Dutta, [2605.25842](https://arxiv.org/abs/2605.25842)) | CoT 핵심 토큰 기반 구조 프루닝 | 없음 | Qwen2.5-VL-7B 30%: judge 8.87 vs 7.32 |

참고: NeurIPS 2026 목록의 ["One Loss Is Not Enough: Cross-Objective Structured Pruning for Reasoning VLA Models"](https://neurips.cc/virtual/2026/161992)
(COSP)는 Alpamayo-1.5-10B 24.0%에서 폐루프 0.750 → 0.828, 과실 충돌 0.040 → 0.023을 보고한다.
MoOP 수치와 같아서 **본인 논문으로 보인다.**

FlashDrive는 같은 모델에서 시스템 수준으로 4.7×를 낸다. 논문의 지연시간 표(Table 5)와 나란히
읽힐 수 있으니 인용·비교 여부를 따로 정해야 한다.

### 문헌의 공백 (조사 4갈래가 공통으로 찾지 못한 것)
1. **Minitron 레시피를 VLA에 적용한 사례가 없다.** 폭 프루닝 + 텍스트 logit KL + action 증류의 조합은 아무도 하지 않았다. 기존 회복은 hidden MSE, RL, SFT다.
2. flow-matching **action expert의 폭(head·채널) 프루닝이 없다.** expert는 얕게 만들거나 스텝만 줄였다.
3. **생성 추론을 측정·보존하는 VLA 프루닝이 없다.** 가까운 것은 LLM/VLM 전용(MuCRASP 등)이다.
4. 층별 KV 결합을 가진 **주행 VLA의 구조적 프루닝·증류와 그 폐루프 평가가 없다.** 공개된 Alpamayo 작업은 시스템 최적화(FlashDrive)와 추측 디코딩이다.
5. 압축·증류 중 **KI 대 end-to-end 기울기 경로**를 비교한 연구가 없다.
6. MLLM에서 **hidden 차원 + projector 동반 절단**의 선례가 없다.

---

## 4. VLA를 위한 새 방법론 제안 (내 제안 — 문헌으로 검증되지 않음)

아래는 위 공백에서 나온 아이디어다. **베이스라인에 넣으면 더 이상 Minitron이 아니게 되므로**, 넣는다면
"우리 방법" 또는 "강화된 Minitron" 변형으로 분리해야 한다. 각 항목에 해결하는 결정 항목을 적었다.

| # | 제안 | 무엇을 하나 | 근거와 위험 | 관련 D |
|---|---|---|---|---|
| P1 | **KV 캐시 인터페이스 증류** | 층별 K/V를 teacher와 MSE(또는 cosine)로 맞춘다. hidden 축소에도 k/v_proj 출력 폭(8×128)은 그대로라 투영 없이 바로 비교된다 | expert가 읽는 바로 그것을 VLM에 직접 가르친다. stop-gradient(D3)와 양립한다. residual 폭이 같을 때 hidden 매칭이 통했다(2609.19579). 위험: MLLM에서 여러 층 매칭은 해로웠다(LLaVADI, Huang), 캐시 수준 매칭은 미검증(공백 5) | D1·D3·D4 |
| P2 | **인터페이스 인지 hidden 채널 중요도** | residual 채널 점수에 LM head 경로뿐 아니라 k/v_proj로 가는 기여(‖W_k[:,c]‖·\|x_c\|를 expert가 읽는 위치에서 집계)를 넣는다 | vision 위치를 지배적으로 세는 대신 "expert가 읽는 것"으로 가중한다. SlimVLM·TAMP의 "텍스트 기준이 낫다"를 VLA에 맞게 바꾼 형태 | D9·D4 |
| P3 | **이산 궤적 토큰 logit KD** | stop-gradient 아래에서 VLM이 받는 유일한 action 신호. AR1의 학습 설계를 증류로 옮긴 것 | 로짓 KL이라 Minitron 원칙 안에 있다. **베이스라인에 넣어도 되는 유일한 제안** | D2·D3 |
| P4 | **결합 깊이 프루닝의 짝 선택 기준** | 두 탑에서 함께 뺄 층을 고를 때 VLM 블록 중요도와 expert의 층별 캐시 의존도를 함께 본다 | Shallow-π·CLP는 짝 제거를 했지만 결합을 고려한 선택 기준은 없다 | D5 |
| P5 | **같은 샘플러 학생을 위한 질의점 혼합** | (i) 보간점 + (iv) 학생 자기 flow 상태를 섞는다(DAgger식) | 소수 스텝 학생에서만 검증된 (iv)를 구조 압축 학생에 처음 적용. 통제 비교 자체가 기여(공백) | D1 |
| P6 | **2단계 회복** | merger·DeepStack merger만 먼저 짧게 증류한 뒤 전체 증류 | projector만 회복하는 단계가 싸고 유효(LLaVA-KD +0.9, Huang 95%+). hidden 절단으로 시각 경계가 바뀌는 우리 경우에 특히 맞음 | D4·D10 |
| P7 | **추론 보존 on-policy CoC 증류** | 학생 자신의 CoC rollout 위에서 KL(GKD / Qwen3-VL식) | 압축 후 CoC 퇴화(빈 출력·반복)는 exposure bias 구조다. on-policy가 그 처방 | D2 |
| P8 | **RMSNorm 이득 닫힌꼴 보정** | 캘리브레이션 통계로 남은 채널의 RMS 비를 계산해 norm 가중치를 한 번 재조정 | Pangu Light +0.7(21B 토큰 후). 우리처럼 예산이 작을수록 상대 가치가 클 수 있다 | D4 |

---

## 5. 다음 논의

D1부터 다시 본다. 문헌 반영 후 D1 추천은 다음과 같다.

- action 증류: 한다.
- 손실: 속도 MSE만.
- 비교 지점: (i) GT 보간점, t는 추론 격자 균등.
- 가중: 동적 α.
- ablation: (ii)와 (iv).

D3(stop-gradient로 바뀜)과 D2의 이산 궤적 토큰 선택지는 D1과 묶여 있어 함께 보는 게 좋다.
