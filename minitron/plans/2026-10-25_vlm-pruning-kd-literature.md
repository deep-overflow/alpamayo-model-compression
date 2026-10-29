# VLM 구조적 프루닝 + KD 문헌 조사 — Minitron을 넘는 방향 고르기 (2026-10-25)

**목적.** MT-W + KD 베이스라인 다음에 시험할 방향을 정하려고, 다음 세 가지에서 무엇이 Minitron(활성 중요도 + logit KL)을
**복구 뒤에도** 이겼는지 찾았다.
- (A) 기준: 출력이나 토큰 종류를 반영한 중요도
- (B) 예산 배분: 층·모듈 사이 비균등 배분
- (C) 복구: on-policy, 중간 특징, KV 증류

2026-10-23 조사(`2026-10-23_literature.md`)에 있는 논문은 제외했다.

**방법.** 세 갈래(VLM 프루닝, Minitron 대비 LLM 방법, 압축 모델 복구 증류)를 병렬로 검색했다. 모든 논문은 arXiv
초록·본문(또는 학회 페이지)을 열어 확인했다. 숫자는 그 논문의 표에서 옮긴 것이다. 확인하지 못한 항목은
**[미확인]**으로 표시했다. 우리 논문(COSP, NeurIPS 2026)은 검색에 걸렸지만 제외했다.

## 0. 결론 — 추천이 바뀐 곳

| 방향 | 이전 추천(10-25 오전) | 문헌을 본 뒤 | 근거 |
|---|---|---|---|
| (A) 기준 교체 | 1순위(싸고 빠름) | **단독으로는 하위.** 복구 뒤 효과로만 판정 | 통제된 비교마다 기준 차이가 복구 뒤 사라짐: L2 vs FLAP 3.69/5.85 → 1.380/1.380(Minitron-SSM); 구조적 방법 30점 차 → 2.1점(2606.14150). 우리 u55(KI-LoRA)와 같은 현상 |
| (B) 예산 배분 | 2순위 | **유지·강화.** 후보는 짧은 KD 뒤에 순위를 매김 | 같은 크기 후보가 KD 19B 토큰 뒤에도 59.07 vs 63.02(Nemotron Nano 2). width vs depth의 차이도 복구 뒤에 남음(2606.14150, Huang IJCV) |
| (C) 복구 | 베이스라인 결과를 보고 | **1순위.** 특히 on-policy CoC 증류 | Minitron의 width 설정 그대로, 같은 step에서 on-policy가 teacher-forced forward KL을 +6.4–9.0점 이김(ShortOPD). 실패 양상은 반복으로, 우리 CoC 퇴화와 같음 |
| KV 캐시 증류 | C1 | **새 기여 후보.** 보조 손실로 | 두 번째 소비자(expert)가 있는 KV를 증류한 압축 연구는 없음. 압축 밖에서 KaVa +3–4점. hidden/attention 증류는 보조 손실일 때만 +1–5점 |
| 전체 어휘 logit 저장 | (결정됨) | **옳았음** | 캐시된 top-K12/50 목표는 CE만 쓴 것보다 나쁨(2.500/2.396 vs 2.371, Sparse Logit Sampling) |

## 1. 비교표

### (A) 기준 — 복구 뒤에도 남는가

| 논문 | 모델 | 결과 |
|---|---|---|
| **Minitron-SSM** ([2504.11409](https://arxiv.org/abs/2504.11409)) | Nemotron-H 8B → 4B | L2 활성 vs FLAP: zero-shot LM loss 3.690 vs 5.854 → 짧은 KD(3.8B 토큰) 뒤 **1.380 vs 1.380**. 저자 결론: "FLAP은 뚜렷한 이점 없음". 후보 순위는 짧은 KD 뒤에 매김 |
| **Small LLMs: Pruning vs Scratch** ([2606.14150](https://arxiv.org/abs/2606.14150)) | Llama-3.1-8B 구조, 50–81% 프루닝 | 재학습 전 평균 33.5–63.3(Minitron-W 최하 33.5, Sheared 최고 63.3) → LM 손실 50B 토큰 뒤 64.4–66.5. 순서도 뒤집힘(Minitron-W 66.3, FLAP 66.5 > Sheared 65.9). 81%에서 Minitron-W는 처음부터 학습한 모델보다 +5.1, Minitron-D는 이점 없음(48.1 vs 48.3) |
| **SlimQwen** ([2605.08738](https://arxiv.org/abs/2605.08738)) | Qwen3-Next 80A3B → 23A2B | depth 기준 간 차이 32–38점(zero-shot) → 2.4–4.0점(120B 토큰). expert 병합 방법 9종은 400B 토큰 뒤 "미미". 점진 vs 일괄: +0.35 이하 |
| **Short-LVLM** (ACM MM 2025, [2507.23362](https://arxiv.org/abs/2507.23362)) | LLaVA-1.5 7B/13B 등 | 층 중요도를 텍스트만 또는 비전만으로 재면 **무작위보다 나쁨**(20% 프루닝에서 상대 약 30% 하락). 전체 토큰은 소폭만 개선. attention 상위 10% VL 토큰으로 한정하면 해결. 단, zero-shot(학습 없는 SVD 보정)만 측정 |
| **Huang et al.** (IJCV 2026, [2604.24380](https://arxiv.org/abs/2604.24380), GCPR 2025판 확장) | LLaVA-1.5-7B, Mini-InternVL-4B, Bunny-3B | 출력 기반 Taylor(CE)는 캘리브레이션 약 10개에서 포화. width 15%: 멀티모달 −9.13% vs 텍스트 −1.07% — 저비율 손상은 주로 비전-언어 정렬 |
| **Free Lunch** ([2510.14444](https://arxiv.org/abs/2510.14444)) | Qwen2.5 7–72B (주로 비구조) | magnitude vs Wanda 약 18점 차가 블록 단위 재구성 뒤 0. 데이터 64배 적게 MaskLoRA 재학습과 동급 |
| DarwinLM (COLM 2026, [2502.07780](https://arxiv.org/abs/2502.07780)) | Llama-2-7B → 2.6B 등 | 학습을 반영한 진화 탐색. 10B 토큰에서 62.8 vs Sheared 61.9(같은 10B). Sheared 30B 토큰은 63.0. 학습 뒤 Minitron과는 비교 없음 |

### (B) 예산 배분 — 복구 뒤에도 남는가

| 논문 | 모델 | 결과 |
|---|---|---|
| **Nemotron Nano 2** ([2508.14444](https://arxiv.org/abs/2508.14444)) | 12B → 9B (hybrid) | 같은 크기(8.89–8.97B) 후보 3개가 FFN과 Mamba head 사이 예산만 다름. KD 19B 토큰 뒤 **59.07 / 63.02 / 62.94**. depth 52/54/56층 → 6B 토큰 뒤 44.9/47.4/51.5 |
| **Puzzle / Nemotron-51B** ([2411.19146](https://arxiv.org/abs/2411.19146)) | Llama-3.1-70B → 51B | 블록 단위 국소 증류(BLD) 라이브러리 + MIP 탐색. 탐색 구조 MMLU 79.7 vs 같은 라이브러리의 무작위 구조 66.0(같은 후속 학습인지 [미확인]). BLD만으로 KD 전 78.39. 전역 KD: cosine(hidden)+KL 84.16 > KL만 83.07 > cosine만 83.40. LM 손실을 더하면 MT-Bench 하락 |
| LLaMaFlex (ICLR 2025, [proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/685d249ad59836727be209032f082bd7-Abstract-Conference.html)) | Llama-3.1-8B → 1.8–7B 묶음 | 3.5B 평균 70.6 / MMLU 60.3 vs Minitron-Width 3.7B 70.5 / 60.5 — **동급**. 이득은 토큰 비용 |
| Nemotron Elastic ([2511.16664](https://arxiv.org/abs/2511.16664)) | Nano-V2 12B → 9B/6B | 9B 75.95 vs Minitron-SSM 계열 75.99 — **동급**. 토큰 110B vs 750B |
| **HiPP-Prune** ([2603.06270](https://arxiv.org/abs/2603.06270)) | LLaVA-1.5-7B, Qwen2.5-VL-3B | MLP 희소도 합은 고정하고 층별 배분을 GRPO 정책으로 정함. 신호는 언어→비전 attention 질량. 약 22.5%에서 POPE 72.89 vs LLM-Pruner 49.27 / SliceGPT 52.56(dense 82.43). 무작위 배분은 분산이 큼. 복구가 약해 SQA는 39 vs dense 64 |
| **INTERLACE** ([2511.19676](https://arxiv.org/abs/2511.19676)) | **Qwen3-VL-8B/4B (36층 — 우리 VLM과 같은 폭·깊이)** | 층 25% 제거 + CE 미세조정(FineVision 1%, 약 6시간, KD 없음): 긴 CoT에서 86.1%(8B) / 81.7%(4B) 유지. 짧은 CoT 88.9% vs SLEB 60.5%. TTFT 1.18× |
| Týr-the-Pruner (NeurIPS 2025, [2503.09657](https://arxiv.org/abs/2503.09657)) | Llama-3.1-8B 50% | 균일 40.74 vs 탐색한 층별 희소도 47.79 — zero-shot만 |
| UKMP (AAAI 2025, [페이지](https://ojs.aaai.org/index.php/AAAI/article/view/32923)) | BLIP-2 | 모달리티별 정규화 Taylor + 점진 증류. 수치 **[미확인]** |
| EfficientLLaVA (CVPR 2025, [2503.15369](https://arxiv.org/abs/2503.15369)) | LLaVA-1.5-7B | 구성요소별 비율을 탐색. ScienceQA 83.05%, 1.8×. 어떤 구성요소를 자르는지 **[미확인]** |

### (C) 복구 — logit-only KD를 이기는 것

| 논문 | 설정 | 결과 |
|---|---|---|
| **ShortOPD** ([2607.13124](https://arxiv.org/abs/2607.13124)) | Qwen3-4B/8B, depth 25% 및 **Minitron ModelOpt width/hybrid 설정** | 같은 710 step. width 29.5%: forward-KL KD 43.70 → **on-policy(JSD β 0.5) 51.46**; hybrid 46.15 → 55.16; depth 30.5 → 48.5(4B). SFT 21.2, SeqKD 28.6, RLVR ≤7.7. depth 25%에서 롤아웃의 84%가 접미 반복으로 끝남. 반복 토큰의 JSD는 정상 토큰의 약 1/35(0.0014 vs 0.051). 일반 지시문 프롬프트를 빼면 48.5 → 39.0 |
| **Revisiting OPD** ([2603.25562](https://arxiv.org/abs/2603.25562)) | Qwen2.5-7B ← OpenThinker3-7B | teacher가 student의 반복 루프 안 토큰에 높은 점수를 줌. 수학 평균: 샘플 토큰 OPD 36.4 → 특수 토큰 마스크 40.7 → **teacher top-K(32) 지지 집합 매칭 41.7** |
| TIDE ([2608.09836](https://arxiv.org/abs/2608.09836)) | Qwen3 쌍 | "degenerate agreement": student 루프가 teacher와 토큰마다 일치하는 현상. 불일치가 크면 6.9 → 20.3%, 응답 길이 ÷3.6 |
| SimpleOPD ([2608.14277](https://arxiv.org/abs/2608.14277)) | 장문 teacher | 순수 OPD는 반복·잘림을 늘림. student 기준 KL(계수 1.0) + 종결 토큰 마스킹으로 잘림 ≈ 0. ProofBench 21.7 → 38.5 |
| SKD (ICLR 2025, [2410.11325](https://arxiv.org/abs/2410.11325)) | Gemma-2B ← 7B | GSM8K: supervised KD 22.5, on-policy 25.3, **SKD(교차 샘플링) 29.1**. 번역에서 순수 on-policy는 반복으로 붕괴(COMET 36.1 vs SKD 75.3) |
| DistiLLM-2 (ICML 2025, [2503.07067](https://arxiv.org/abs/2503.07067)) | TinyLLaVA-1.4B ← LLaVA-1.5-7B | OK-VQA/TextVQA 평균: student 32.6, DistiLLM 35.2, GKD 37.8, **DistiLLM-2 39.9** |
| **Huang et al.** (IJCV 2026) | 프루닝한 LVLM | Bunny 층 15/30/45/60%: **SFT+hidden L2 99.6/95.0/90.2/72.6** > SFT+RKL 98.7/93.8/88.5/69.9 > SFT 96.3/94.3/86.7/69.4. logit KD만으로는 고비율에서 SFT를 못 넘음(RKL만 60%에서 12.6). L2만은 고비율에서 최하 |
| **Offline HSD** (Kim, [2609.19579](https://arxiv.org/abs/2609.19579)) | OpenVLA-OFT, CogACT; head + MLP(residual 유지) | 출력 Taylor(action 손실) + action 토큰 hidden MSE를 오프라인 캐시로. SFT 대비 +2.1–4.5(OFT 63–85%), +10.3–22.1(CogACT 63–81%). **45% 이하에서는 유의한 이득 없음** |
| **Drive-KD** ([2601.21288](https://arxiv.org/abs/2601.21288)) | InternVL3-1B ← 8B, DriveBench | 인지/추론/계획: CE 40.9/29.1/45.6, **CE+KL 39.6/28.2/43.4(KL이 해침)**, attention 증류 43.1/34.3/53.0. 다중 teacher를 그냥 섞으면 추론 25.7(기울기 투영 시 33.2) |
| CompoDistill (ICLR 2026, [2510.12184](https://arxiv.org/abs/2510.12184)) | 1.8B ← LLaVA-4B | 중간층(깊이 30–70%) 시각 key attention, cosine 손실: logit-only 56.8/62.9 → 62.9/66.7 |
| **KaVa** (ICLR 2026, [2510.02312](https://arxiv.org/abs/2510.02312)) | 잠재 추론 학생 ← 명시 CoT teacher | **KV 캐시**를 층·head마다 L1/MSE로 맞춤: Llama-3.2-1B GSM8k 56.5 vs KV 손실 없이 53.4. 압축 연구는 아님 |
| Sparse Logit Sampling (ACL 2025, [2503.16870](https://arxiv.org/abs/2503.16870)) | 8B → 3B 사전학습 KD, 100B 토큰 | LM loss: top-K12 2.500, top-K50 2.396, CE만 2.371, 샘플 12개 2.350, 전체 KD 2.344. top-K는 ECE도 9–23배 나쁨 |
| Self-Data Distillation (MLSys 2025, [2410.09982](https://arxiv.org/abs/2410.09982)) | Llama-3.1-8B −6층 | teacher가 다시 생성한 데이터로 SFT 91.2% vs SFT 81.7% 유지 |
| RAC ([2509.12464](https://arxiv.org/abs/2509.12464)) | SparseGPT 50%, Qwen3-8B | 자기 CoT로 캘리브레이션한 층 재구성: MATH500 0.564(C4) → 0.862(dense 0.962). 프루닝 모델은 장황해짐(실행 시간 258.8 vs 17.1분) |
| OPD for driving planners ([2604.07944](https://arxiv.org/abs/2604.07944)) | Qwen3-1.7B ← 8B, nuScenes(텍스트 CoT + waypoint) | GKD(JSD, λ 0.5): L2 0.373 vs teacher 0.355 vs RL 0.579. SFT·off-policy KD 기준선 없음 |
| Reflect-VLA ([2610.13453](https://arxiv.org/abs/2610.13453)) | Qwen3.5 0.8–4B, NAVSIM | 궤적 토큰에 on-policy 자기증류(teacher top-64 역KL): PDMS +2.6–3.1(4B 88.4 → 91.5), GRPO보다 약간 낫고 학습 시간 약 1/14 |

### VLM에서 Minitron 계열 직접 사례

**Nemotron 3.5 Bolt 74B-A9B**(NVIDIA, 2026-10-15). Star Elastic([2605.07182](https://arxiv.org/abs/2605.07182))을 썼다.
**기술 보고서 원문으로 확인**(2026-10-25)한 내용은 다음과 같다.
- **텍스트 탑만 압축했다.** MoE layer마다 routed expert 수와 중간 폭을 줄였다. hidden size는 "부모의 vision head와
  호환되도록" **유지**했다(p.1: "preserving the hybrid architecture and hidden size to retain compatibility with the
  parent's vision head").
- **1단계: 압축 중 KD 120B 토큰.** 데이터는 부모 SFT 데이터 70% + 사전학습 데이터 30%이고, 비전 탑을 뗀 상태였다.
  보고서 표현은 "knowledge distillation applied to recover **text-only** benchmarks during compression"이다.
- **2단계: vision adapter만 학습.** 그 뒤 부모의 비전 탑을 다시 붙이고 adapter만 학습했다(이미지 전용 약 6.6M
  샘플/epoch).
- **3단계: 비전 중심 SFT.** LLM, adapter, 비전 인코더를 모두 학습했다(원래 blend 54.1B 토큰을 비전 쪽으로 재가중).
  이어서 adapter만 학습한 체크포인트와 SLERP 병합(80:20)을 했다.
- 보고서 결론: 2·3단계가 비전 복구에 "critical"이고, 텍스트는 "little to no degradation"이다.
- 단계별 수치는 Figure 7에만 있어 텍스트로는 읽지 못했다.
- **순서가 핵심이다: 디코더를 먼저(텍스트 KD 120B 토큰) 복구한 뒤에 접점을 학습했다.**
- 출처: [기술 보고서](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-Technical-Report.pdf),
  [모델 카드](https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-GGUF)

**같은 예산에서 VLM에 Minitron(활성 중요도 + logit KD)을 직접 비교한 논문은 찾지 못했다.**

## 2. 핵심 논문 상세

### ShortOPD (2026, arXiv 2607.13124)
**Core Contribution**
- 구조적 프루닝 뒤의 복구에서 on-policy 증류가 Minitron식 teacher-forced KD를 크게 이긴다.

**Method**
- student가 생성하고, 프루닝 전 모델이 teacher로 채점한다.
- 손실은 generalized JSD(β 0.5), teacher top-100 logit + 꼬리 질량이다.
- 프롬프트는 수학, 코드, 일반을 섞는다.

**Results**
- 위 표 참조.
- 실패 양상은 greedy pass@1 붕괴(pass@k는 유지)와 접미 반복이다.

**Relevance to Our Work**
- MT-W zs의 CoC 퇴화 24%가 같은 현상일 가능성이 크다.
- Alpamayo의 CoC는 평균 약 16토큰이라 student 롤아웃이 싸다.

**Limitations / Open Questions**
- 우리는 훨씬 덜 망가진 상태(퇴화 24%)라 격차가 작을 수 있다.
- teacher가 루프를 보상하는 함정이 있다. 그래서 top-K 지지 집합 매칭, 종결 토큰 처리, λ 혼합(GKD)이나 SKD가 필요하다.
- 우리 종결 토큰(cot_end)이 바로 지도 위치이므로 주의가 필요하다.

### Minitron-SSM (2025, arXiv 2504.11409) + Small LLMs: Pruning vs Scratch (2026, arXiv 2606.14150)
**Core Contribution**
- 기준 간 차이가 KD·재학습 뒤 사라진다. NVIDIA 자체 ablation과 독립 통제 실험이 같은 결론을 낸다.

**Method**
- 같은 파이프라인에서 기준만 바꿨다.
- 짧은 KD 또는 LM 재학습 뒤에 비교했다.

**Results**
- 1.380 = 1.380이다.
- 30점 차가 2.1점이 되고 순서도 뒤집혔다.

**Relevance to Our Work**
- 우리 u55 결과(zero-shot 기준 차이 최대 +1.78 → 복구 뒤 ±0.05)와 일치한다.
- 그래서 MoOP + KD vs MT-W + KD의 차이는 대부분 형태(head+MLP vs hidden+MLP)에서 올 것으로 예상된다.
- (A)는 zero-shot 승리로는 근거가 되지 않는다.

**Limitations / Open Questions**
- 2606.14150은 KD가 아닌 LM 재학습이고, 부모 모델은 200B 토큰만 사전학습했다.
- 멀티모달에서의 통제 실험은 없다.

### Nemotron Nano 2 (2025, arXiv 2508.14444) + Puzzle (2024, arXiv 2411.19146)
**Core Contribution**
- 같은 크기에서도 모듈·층 사이 배분이 KD 뒤까지 남는 차이를 만든다.

**Method**
- Nano 2: 후보를 각각 짧게 KD한 뒤 고른다.
- Puzzle: 블록 단위 국소 증류 + MIP로 이질적 구조를 탐색한다.

**Results**
- Nano 2: 59.07 vs 63.02(4점)이다.
- Puzzle: 탐색 79.7 vs 무작위 66.0(MMLU)이다.

**Relevance to Our Work**
- (B)를 지지한다. VLM↔expert 예산과 층별 MLP 폭이 대상이다.
- 후보 순위는 짧은 KD(1 epoch) 뒤에 매긴다. 우리 선별 절차와 같다.

**Limitations / Open Questions**
- LLaMaFlex와 Elastic은 탐색한 구조로도 잘 조정된 Minitron 형태와 동급이었다.
- 그래서 "아무 탐색"이 아니라 배분 자체가 문제인 축(우리는 두 탑)에서 이득을 기대해야 한다.

### Huang et al. (IJCV 2026, arXiv 2604.24380)
**Core Contribution**
- LVLM 프루닝 × 복구 방식을 가장 크게 통제 비교했다.

**Method**
- width는 Taylor(CE)로 head와 뉴런을 자른다. residual 폭을 유지했는지는 원문에 명시되지 않았다.
- 층은 Block Influence로 고른다.
- 복구는 projector만 SFT, LoRA, logit KD, hidden L2를 비교했다.

**Results**
- SFT + 최종층 hidden L2가 최선이다.
- logit KD만으로는 고비율에서 SFT를 못 넘는다.
- 데이터 5%로 전체 결과의 90% 이상을 복구한다.
- 30%까지 width가 depth보다 낫다.

**Relevance to Our Work**
- hidden 증류는 "보조 손실로 +1–3점" 수준이다.
- 저비율 손상이 비전-언어 정렬에 몰린다는 점이 hidden을 자르는 MT-W에 시사하는 바가 있다.

**Limitations / Open Questions**
- 활성 기준 arm이 없다.
- hidden 폭을 자르는 경우를 다루지 않는다.

### KaVa (ICLR 2026, arXiv 2510.02312) + Offline HSD (2026, arXiv 2609.19579)
**Core Contribution**
- KaVa: teacher KV 캐시를 직접 맞추면 출력 증류 위에 신호가 더해진다.
- Offline HSD: VLA에서 action head가 읽는 상태를 증류하면 고압축에서 크게 돕는다.

**Results**
- KaVa: +3–4점.
- Offline HSD: +2–22점, 45% 이하에서는 유의하지 않다.

**Relevance to Our Work**
- MT-W는 head와 head_dim을 유지하므로 **KV 캐시 모양이 teacher와 같다.** 투영 없이 맞출 수 있다.
  - 조사 보고서 하나가 "hidden을 자르면 투영 없는 매칭이 불가능하다"고 썼는데, 이는 hidden state에만 해당한다.
- 두 번째 소비자(expert)가 있는 KV를 증류한 압축 연구는 없다. 그래서 새 기여 후보다.
- KI 아래에서는 KV 손실이 VLM을 expert와 호환되게 붙잡는 유일한 신호가 된다.

**Limitations / Open Questions**
- 24% 예산은 Offline HSD에서 이득이 없던 구간이다.
- 전 층·전 위치 teacher KV는 window당 약 472 MB다(36층 × K/V 2 × 8 head × 128 × 약 3,200 위치 × bf16). 사전 계산은 불가능하고(9,905개 ≈ 4.7 TB), 온라인 teacher가 필요하다.

## 2b. VLM 복구 연구 모음 (사용자 질문 "VLM에서 복구하는 연구는 없어?"에 대한 답)

복구까지 다룬 VLM 연구는 있다. 세 부류로 나뉜다.

**(1) 프루닝한 VLM을 복구한 연구**

| 논문 | 무엇을 잘랐나 | 복구 방법 | 결과 |
|---|---|---|---|
| Huang (IJCV 2026, 2604.24380) | LLaVA·InternVL·Bunny LLM의 head·뉴런 또는 층 | projector만 SFT **또는** joint SFT(projector + LLM LoRA) — 단계가 아니라 대안. logit KD / hidden L2 추가 | 원문 확인: projector만 SFT는 15%에서 95% 이상 유지, 30% 미만에서 joint와 비슷, 60%에서 60–80%. 45% 이상은 LLM SFT가 필요 |
| Nemotron 3.5 Bolt (NVIDIA, 2026) | 텍스트 탑, hidden 유지 | **텍스트 KD 120B 토큰(디코더 먼저)** → adapter만 학습 → 비전 중심 SFT + SLERP | 원문 확인: 뒤 두 단계가 비전 복구에 "critical", 텍스트는 거의 무손실 |
| INTERLACE (2511.19676) | Qwen3-VL-8B/4B 층 25% | CE 미세조정, FineVision 1%, 약 6시간 | 긴 CoT 86.1% / 81.7%, 짧은 CoT 88.9%(SLEB 60.5%) |
| Eddy-VL (2607.16316) | Qwen3-VL-Embedding-2B 층 28 → 24 | 층별 KD: 중간층 attention(1−CKA) + 최종층 MSE/cosine | 잃은 12.1점 중 6.4점 회복(56.8 → 63.2, teacher 68.9) |
| HiPP-Prune (2603.06270) | LLaVA·Qwen2.5-VL MLP(층별 배분) | LM head·norm·projector + 마지막 층 LoRA, 800 step | POPE 72.89(dense 82.43), SQA 39(dense 64) — 복구는 약함 |
| MoP (2602.06127) | LLaVA-1.5-7B depth + width | **텍스트만** LoRA(Alpaca) | 시각 벤치마크 4개 평균 46.70 / 44.61 / 40.86(20/30/40%) vs dense 52.44 |
| UKMP (AAAI 2025) | BLIP-2 ViT + LLM | LoRA 점진 증류 + 잘린 가중치 "recall" | 수치 [미확인] |
| Short-LVLM (2507.23362) | LLaVA 등 층 | 학습 없는 SVD 보정 | LLaVA-7B 30%: 70.1 → 68.4 |

**(2) 작은 VLM 학생으로 증류한 연구** — 프루닝 후손은 아니지만 손실 설계 근거로 쓸 수 있다.
- LLaVA-KD / LLaVA-MoD / LLaVADI(기존 조사): 응답 토큰 KL이 최선이고, projector만 증류하는 단계가 +0.9.
- DistiLLM-2: 대조형 KD로 GKD보다 +2.1.
- CompoDistill: 중간층 attention 증류.
- Drive-KD: 주행 출력에 KL을 걸면 CE보다 나쁘고, attention 증류가 나음.

**(3) VLA 복구** — Offline HSD, Shallow-π, SPAR, RLRC, CLP(대부분 기존 조사). 이 가운데 Offline HSD만 구조적
프루닝 뒤 hidden 증류를 썼다.

**모달리티 단계 복구(C3)는 MT-W에서는 후보가 아니라 진단이다**(2026-10-25 사용자 지적으로 정정, 원문 확인).

처음 조사 요약은 "세 연구가 '접점을 먼저 복구하는 단계'가 효과가 크다고 보고한다"였다. 원문을 보면
**손상된 디코더를 둔 채 접점부터 복구한 연구는 하나도 없다.**
- **Huang (IJCV 2026, 2604.24380).**
  - projector만 학습하는 것은 첫 단계가 아니라 joint SFT의 **대안**이다.
  - 디코더가 대부분 살아 있는 저비율에서만 충분하다. 저자 결론은 "15% 안팎이면 projector만으로 대부분 회복,
    40% 이상이면 LLM SFT가 필요"다.
  - Table 4(LLaVA-1.5-7B width)의 텍스트 성능 하락은 15% 1.07%, 30% 10.38%, 45% 56.92%다.
    - 15% 값은 표의 평균(40.30 → 38.87)으로 계산하면 3.5%라 표 안에서 어긋난다.
  - hidden 폭 유지 여부는 원문에 명시되지 않았다.
  - 표 5 레시피는 45% 미만에서 layerwise, 45% 이상에서 widthwise다.
- **LLaVA-KD (2410.16236).**
  - DPT 단계는 projector만 학습한다(+0.9: 54.7 → 55.6).
  - student LLM은 프루닝한 모델이 아니라 기성 Qwen1.5/2.5 0.5–1.8B이고, 동결 상태다. 표준 LLaVA 정렬 단계일 뿐이다.
  - 언어 능력이 온전한 LLM에 새 projector를 맞추는 것이라 손상된 디코더와는 무관하다.
- **Nemotron 3.5 Bolt.**
  - **디코더를 먼저** 텍스트 KD 120B 토큰으로 복구했다.
  - 그 뒤에 adapter만 학습하고, 이어서 전체 SFT를 했다.
  - hidden size는 비전 헤드 호환을 위해 유지했다.

- MT-W에서는 이 전제가 성립할 이유가 더 약하다.
  1. hidden 절단은 텍스트 임베딩과 비전 특징(merger·DeepStack 출력)에 **똑같이** 적용된다. 비전 쪽 입력만 따로
     어긋난 것이 아니다.
  2. 유지 채널 순위가 사실상 비전 위치 순위였다(CoC 위치와의 top-k 겹침 hidden 0.866, MLP 0.807). 그래서
     **잘려 나간 쪽은 오히려 언어(CoC) 생성에 중요한 채널**일 가능성이 크다. Huang과 반대 방향의 손상이다.
  3. zero-shot CoC 퇴화 24%(MoOP 3%)도 언어 쪽 손상과 맞는다. 다만 시각 근거가 무너져 생긴 반복과는 아직 구분하지
     못한다.
- 결론: C3는 방법 후보에서 내리고, 손상 위치를 가리는 **진단**으로만 둔다.
  - merger·DeepStack만 학습하는 짧은 KD(50–100 step)를 돌린다.
  - 같은 step의 전체 파라미터 KD 곡선(KL 1.0 → 0.15, 50 step)과 비교한다.
  - KL이 거의 안 내려가면 손상은 디코더 쪽이다.

## 3. 문헌의 공백 (세 갈래 공통)

- VLM width 프루닝에서 활성(Minitron) vs 출력 기반 Taylor vs 토큰 종류 한정 통계를, 같은 예산과 같은 KD로 비교한 연구는 없다.
- **hidden 폭을 자른 VLM width 연구는 찾지 못했다.**
  - Bolt(원문 확인: 비전 헤드 호환을 위해 hidden 유지), Kim(residual 유지), HiPP(MLP만)는 hidden을 유지했다.
  - Huang은 head·뉴런을 잘랐고 hidden에 대한 명시는 없다.
  - merger, DeepStack, expert 인터페이스와 결합된 VLM hidden 폭 절단을 복구까지 본 연구는 없다. 우리 MT-W가 처음이다.
- 하류 모듈이 읽는 전 층 KV를 증류한 복구는 없다.
- VLM 구조적 프루닝 뒤의 on-policy 증류, prune + KD에서의 CoT 보존도 없다. 예외는 MuCRASP(기존 조사)다.
- 모듈 간 예산 배분(ViT vs merger vs LLM vs action head)을 복구까지 본 연구는 BLIP-2(UKMP)뿐이다.

## 4. 이 조사로 바뀐 실험 우선순위 (제안)

1. **C2 on-policy CoC 증류.**
   - 현재 KD에 student가 생성한 CoC에 대한 teacher 점수를 더한다.
   - 처음에는 GKD λ≈0.5 혼합으로 한다. 반복 방지로 top-K 지지 집합 매칭, cot_end 처리, student 기준 KL을 쓴다.
   - teacher는 온라인으로 둔다. CoC가 짧아 롤아웃 비용이 작다.
   - MT-W + KD 결과에서 CoC 퇴화가 남는지 먼저 본다.
2. **B 예산 배분.**
   - VLM↔expert(expert MLP 50%까지 거의 무손실이었던 기존 결과)와 층별 MLP 폭.
   - 후보는 KD 1 epoch 뒤 val500으로 순위를 매긴다.
3. **C1 KV 캐시 증류(보조 손실).**
   - 새 기여 후보다. 온라인 teacher가 필요하다.
   - ablation으로 판정한다: student KV vs teacher KV에서 expert FM 손실을 비교한다.
4. **A 기준.**
   - 단독 실험은 하지 않는다. B의 후보를 만드는 수단으로만 쓰고, 판정은 KD 뒤에 한다.
   - 쓸 기준은 Short-LVLM식 출력 관련도 가중이고, 토큰 종류 한정은 쓰지 않는다.
5. 참고 — 국소 재구성 + 전역 KD(Puzzle BLD, Free Lunch).
   - KD 전에 블록마다 최소제곱 재구성을 하면 복구 비용이 줄 수 있다.
   - 우리 기존 refit 경험(dualr Hessian)과 이어진다.
