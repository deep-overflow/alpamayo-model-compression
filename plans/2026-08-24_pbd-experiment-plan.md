# Progressive Block-wise Self-Distillation (PBD)

VLA 구조적 프루닝 회복 학습 실험 계획

> 개정 2026-08-24 (rev. 1). 원안 대비 바뀐 곳은 §7 개정 이력 참조. 이 저장소의 기존 결과와
> 고정 프로토콜에 맞춰 수정했다.

---

## 0. 포지셔닝

| 구분 | 내용 |
|---|---|
| **기여로 주장할 것** | (a) VLA 압축에서 `p(action \| CoC)` 조건부 경로가 marginal 지표로는 안 보이게 붕괴한다는 현상 규명<br>(b) 이를 명시적으로 타깃하는 recovery 목적함수<br>(c) reasoning + driving 이중축 평가 |
| **기여로 주장하지 않을 것** | progressive / iterative pruning 자체, teacher-free 구현, block-wise reconstruction |
| **주 운영점** | **~33%** (one-shot이 무너지는 구간). ~24%는 대조용 세컨더리 |

### 반드시 diff를 명시할 선행연구

| 논문 | 겹치는 지점 |
|---|---|
| Iterative Layer-wise Distillation (arXiv:2511.05085) | 반복 레이어 제거 + distillation, "원본에서만 중요도 평가하는 건 문제" 논리까지 동일 |
| SlimQwen | progressive structured pruning + KD recovery, progressive > one-shot 보고 |
| ActDistill | VLA + self-derived distillation + 레이어별 VL→Action 변환 |
| Drop-Then-Recovery | VLA compress-then-recover 체제, action 오차의 long-horizon 누적 |
| BERT-of-Theseus | module replacement (= B2 타깃) |
| BRECQ / AdaRound | 블록 재구성, dense trajectory 타깃 (= B1) |
| Minitron, LoRAPrune, Týr-the-Pruner | prune + distill, 반복적 채널 제거 |

### 이 저장소의 선행 결과 (반드시 대조할 것)

| 결과 | 수치 | PBD에 대한 함의 |
|---|---|---|
| `slim_recover_dual_u55` — 33.1% one-shot 프루닝 + KI-LoRA 600step 회복 | val 0.924 / test **1.008** / OOD-val 1.156, degen **0.000**, 폐루프 0.823 (+0.0736*) | **one-shot 회복만으로 이미 24% dual과 대등.** PBD가 이겨야 할 상대는 붕괴한 모델이 아니라 이 수치 |
| `dual_u40_v2` — 24% one-shot, 회복 없음 | val 0.890 / test 0.950 / OOD-val 1.119, 폐루프 0.828 (+0.0787*) | 24% 운영점의 기준선 |
| `dual_u55_v2` 제로샷 | test 4.264, degen 0.714 | 33%가 "one-shot이 무너지는 구간"이라는 §0 전제의 근거 |
| `it3` — 3단계 재캘리브레이션 (증류 **없이** 단계마다 importance 재측정) | 원샷 대비 test **+0.0302\*** [+0.0101, +0.0422], val +0.0219\*, OOD-val +0.0261\* | **P2(= importance 재추정)는 증류 없는 형태로 이미 기각됐다.** §3-1 게이트가 이걸 정면으로 다뤄야 함 |
| kept-set 겹침 잡음 바닥 (같은 기준, calib_100 50:50 분할) | Q **0.860**, MLP **0.782** | drift 지표의 영점. 이 아래로 내려가야 "선택이 실제로 바뀌었다" |
| LingoQA 퇴화 하한 (모든 질문에 "No.") | **37.0** | reasoning 지표는 이 하한 대비로만 해석 가능 |

---

## 1. 메서드 정의

**세팅**: VLM 텍스트 타워 **L = 36층**, 블록 크기 **6 → 6 스테이지**. Q head + MLP channel
구조적 제거 (expert·KV 인터페이스는 불변 — `slim_lib` 규약).

> 원안의 `L=24, 블록 4`는 다른 모델 기준이다. Alpamayo 1.5는 VLM·expert 모두 36층이다.
> 블록 4를 유지하려면 9스테이지가 되고 §5의 예산 매칭이 함께 바뀐다.

### 1-1. 손실 함수

```
L = alpha*L_local + beta*L_KD_coc + gamma*L_KD_act   (+ delta*L_CE, 최종 phase만)
```

| 항 | 내용 | 라벨 | 적용 구간 |
|---|---|---|---|
| `L_local` | 블록 k의 **교정 업데이트분**에 대한 토큰별 정규화 MSE (아래 정의) | ✗ | 전 스테이지 |
| `L_KD_coc` | 원본 모델 최종 logit과의 KL, CoC 토큰 위치 | ✗ | 전 스테이지, beta ≈ 0.1–0.3 |
| `L_KD_act` | 원본 모델 action 출력 매칭 | ✗ | 전 스테이지, gamma ≈ 0.1–0.3 |
| `L_CE` | GT CoC + GT action | ✓ (1,200개) | 최종 phase만 |

**`L_local` 정의 (원안 수정)**

```
d_h_hat_k = Block_k^{pruned+LoRA}( h_hat_{k-1} )      # student 블록 출력 업데이트분
target    = h_star_k - h_hat_{k-1}                    # 오염된 입력에서 깨끗한 출력까지의 교정량
L_local   = || d_h_hat_k - target ||  , 토큰별 정규화
```

- **왜 `Δh`에 거는가** — residual stream에서 `h_out`은 `h_in`이 지배하므로 `h_out` MSE는
  자동으로 작아지고 gradient가 죽는다. 블록의 업데이트분에 걸어야 신호가 산다.
- **왜 빼는 기준이 `h_hat_{k-1}`인가 (원안의 모순 수정)** — 원안처럼
  `Δh*_k = h*_k − h*_{k-1}`를 타깃으로 잡으면 최적점이 `ĥ_k = ĥ_{k-1} + Δh*_k = h*_k + δ_{k-1}`이
  되어 **누적 드리프트가 그대로 통과한다**(`δ_k = δ_{k-1}`). 즉 §1-2가 B1에 부여한 "자기
  교정적" 성질이 사라진다. 빼는 기준을 student의 실제 입력 `ĥ_{k-1}`로 두면 최적점이
  `ĥ_k = h*_k`가 되어 **교정이 일어나면서도 손실은 업데이트 공간에 남는다.**
- **전역항 상시 소량 적용** — `L_local`만으로는 각 스테이지가 로컬 최적일 뿐이고 drift가
  `δ_k ≈ J_k·δ_{k-1} + ε_k`로 누적된다. 마지막에만 켜면 이미 나쁜 basin에 진입해 있고
  freeze 때문에 교정 자유도도 없다.
- **KD는 라벨 불필요** — teacher가 타깃을 생성하므로 대량 unlabeled AV 클립 사용 가능
  (`/mnt/nvme1n1/ad_vla/data/coc_generated/train/coc_train.parquet`: 119,260 샘플 /
  10,040 클립). 1,200 라벨은 최종 phase에만 소비.

### 1-2. 타깃 정의

drift를 `δ_k = ĥ_k − h*_k`로 둘 때 (student 입력은 세 옵션 모두 pruned trajectory `ĥ_{k-1}`):

| 옵션 | 타깃 | 최적점에서의 drift | prefix 공유 |
|---|---|---|---|
| **B1 (기본값)** | `h*_k` — **dense 모델이 자기 궤적 `h*_{k-1}`에서 만든** hidden | `δ_k → 0` (교정) | **불가** (§1-3) |
| **B2 (ablation)** | `Block_k^orig(ĥ_{k-1})` — 원본 블록을 **같은 오염 입력**에 적용 | `δ_k ≈ (I+J_k)·δ_{k-1}` (전달·증폭) | 가능 |
| **blend** | `λ·B1 + (1−λ)·B2`, λ=0.5 | — | 불가 |

> B1과 B2의 차이는 **teacher가 어느 궤적 위에 있는가**다. B2는 타깃이 realizable하지만
> 드리프트 교정항이 없다. `L_local`을 §1-1의 형태로 적지 않으면 B1도 교정성을 잃어 두 arm이
> 사실상 같아지므로, 이 ablation의 전제 조건이 §1-1의 정의다.

### 1-3. 구현

- dense weight 1벌 + structured mask 토글 + **LoRA 어댑터 1벌**(전 층에 부착, 스테이지마다
  해당 블록 슬라이스만 unfreeze).
  - **teacher = mask off + `disable_adapter()`** → 매 스테이지 정확히 원본.
  - **merge하지 않는다** (원안 §1-4 수정). merge하면 스테이지 2부터 mask-off가
    `원본 + 누적 LoRA Δ`가 되고, 그 Δ는 "채널이 없는 상태"를 보상하도록 학습된 것이라
    채널이 되살아난 상태에서 **이중 계상**된다 → teacher가 원본도 student도 아닌
    과보상 하이브리드가 되어 §1-1의 drift 교정 설계와 정면 충돌.
  - 어댑터 스택 우려(원안 §5)는 r=32 기준 한 벌 ~0.3 GB로, 1벌 유지 + 부분 unfreeze면
    스택 자체가 생기지 않는다.
  - **검증**: 스테이지 시작 시 `mask off + adapter off`의 hidden이 원본과 일치하는지
    한 위치에서 assert.
- **teacher와 student는 블록 1부터 궤적이 갈라지므로 prefix를 공유할 수 없다** (원안 수정).
  "블록 k에서만 분기"는 **B2 한정** 최적화다.
- **스텝당 연산**: `teacher 블록 1..k (no-grad)` + `student 블록 1..L (fwd+bwd)`.
  teacher를 L까지 돌릴 필요가 없다 — 최종 출력 타깃은 아래처럼 미리 캐시하기 때문.
- **캐시 전략** (원안에 없음):

  | 대상 | 샘플당 | 10k 샘플 | 처리 |
  |---|---|---|---|
  | teacher 최종 CoC logit(top-k) + action | ~0.12 MB | ~1.2 GB | **최초 1회 전수 캐시** (스테이지 불변) |
  | `h*_k`, `ĥ_{k-1}` hidden 전체 (3,086토큰) | 25.3 MB x2 | 253 GB x2 | **캐시 불가 → 재계산** |
  | 위 hidden, 512토큰 부분표집 | 4.2 MB x2 | 84 GB | 데이터셋이 작고(≈2.5k) **2 epoch 이상일 때만** 검토 |

  CoC 스팬이 클립당 15–25토큰뿐이라 KD 타깃은 사실상 공짜인 반면, hidden은 vision 토큰이
  93%라 200배 크다. `/mnt/nvme1n1`은 98% 사용(여유 158 GB)이므로 hidden 전수 캐시는 배제.
- 실제 채널 materialize는 최종에만.

### 1-4. 스케줄

| phase | 내용 |
|---|---|
| Stage 1–6 | 블록 k prune → 해당 슬라이스 unfreeze → 학습 → **freeze (merge 안 함)** |
| Final | **전체 해동** + on-policy GKD (student가 CoC free-running 생성, teacher가 채점) + `L_CE` |

순서는 **front-to-back 고정**. back-to-front는 앞을 자르는 순간 이미 복구한 뒤쪽이 stale해지고
보상 여지가 사라진다.

---

## 2. 실험 arm

### Phase A — 코어 (33%, **토큰 예산 T 고정 + 데이터셋 x epoch 명시**)

| arm | prune | recovery | 분리하는 것 |
|---|---|---|---|
| **B0** | one-shot | one-shot LoRA, 예산 T, **PBD와 동일 목적함수** | 예산 매칭 대조군 |
| **P1** | one-shot | progressive block-wise, 총 T | recovery 스케줄만의 효과 |
| **P2** | progressive (importance 재추정) | progressive, 총 T | 전체 제안 |
| *(앵커)* | *one-shot* | *KI-LoRA (CE+FM), 4,800 sample-pass* | *`slim_recover_dual_u55` — 재실행 불필요, 기존 최고 회복 결과* |

- `P1 − B0` = recovery 스케줄의 기여
- `P2 − P1` = importance 재추정의 기여
- **B0는 새로 돌려야 한다.** `slim_recover_dual_u55`는 구조(one-shot 프루닝 + one-shot
  LoRA 회복 @33%)는 같지만 **목적함수가 CE+FM(KI-LoRA)이고 KD가 아니다.** 따라서 깨끗한
  `P1 − B0` 대조에는 쓸 수 없고, **"이미 도달된 수준"을 보여주는 앵커**로만 쓴다.
- **예산 T 명시 필수**: 총 sample-pass 수와 (데이터셋 크기 x epoch) 분해를 둘 다 적는다.
  참고로 앵커는 2,471샘플 x ≈2 epoch = 4,800 sample-pass다. epoch ≥ 2면 §1-3의 부분표집
  hidden 캐시가 이득이고, 만 단위 코퍼스 1-pass면 캐시는 순손해다.

### Phase B — P2 위 ablation

| 축 | 변이 |
|---|---|
| **타깃 (T)** | T-a: dense trajectory `h*` / T-b: `Block^orig(ĥ)` / T-c: blend λ=0.5 |
| **전역항 (G)** | G0: 스테이지 중 로컬만, 끝에만 전역 / G1: 스테이지마다 전역항 소량 |
| **freeze (F)** | F0: prefix 계속 frozen / F1: 최종 해동 |
| **최종 phase (O)** | O0: teacher-forced KD / O1: on-policy GKD |
| **granularity (S)** | 블록 3 / 6 / 12 / 36(= one-shot) |

---

## 3. 진단 지표

### 3-1. Ranking drift — **게이트**

정적 importance(M0에서 1회) vs progressive importance(현재 pruned+recovered prefix 위) 비교.

- Spearman, Kendall, top-k IoU, RBO — **depth별로** 플롯
- 24% / 33% / 40% 세 지점
- **잡음 바닥 대비로 판정한다**: 같은 기준으로 calib_100을 50:50 분할만 해도 kept 겹침이
  Q 0.860 / MLP 0.782다. drift가 이 바닥 안쪽이면 "선택이 바뀌었다"고 말할 수 없다.

> **이 게이트는 drift 유무를 묻는 것이 아니다.** `it3`가 이미 보여준 것은 (a) 재측정하면
> 선택이 실제로 바뀌고(kept 겹침 89.9/88.7%), (b) 그런데도 결과는 **유의하게 나빠진다**는
> 것이다(test +0.0302*). 즉 재측정된 그래디언트는 정보가 아니라 잡음이었다. 따라서 P2가
> 정당화되려면 **"증류가 그 잡음을 신호로 바꾼다"**는 가설을 명시하고, 무엇이 관측되면
> 기각인지 사전 등록해야 한다. 예: P2−P1이 잡음 바닥을 넘는 drift를 보이면서도 개루프
> paired Δ의 CI가 0을 포함하면 → P2 폐기, 프레이밍을 "recovery schedule + conditional
> pathway"로 전환.

### 3-2. CoC-Action 조건부 민감도 (핵심 신규 지표) — **원안 수정**

```
S(model) = E || a_model(c_ref) - a_model(c_cf) || ,   (c_ref, c_cf)는 arm과 무관하게 고정
```

- `c_ref` = OOD 셋의 `gt_coc` 또는 dense teacher가 생성해 **한 번 고정한** CoC
- `c_cf` = 같은 클립에 대한 **의미 반전 CoC**(감속↔가속, 직진↔차선변경) 또는 다른 클립의 CoC
- 원본 모델의 `S0`로 정규화해 보고. **`S/S0`가 크게 감소 = action이 CoC를 무시 = 조건부 경로 붕괴**
- **크기와 함께 부호 있는 응답을 병기**한다(감속 counterfactual에 대한 속도 프로파일 변화).
  L2만 보면 "엉뚱하게 반응"과 "올바르게 반응"이 구분되지 않는다.
- CoC 자체 품질(Lingo-Judge, **하한 37.0 대비**)과 독립적으로 움직이는지가 논지의 핵심.

> **원안이 쓴 `c_student` vs `c_teacher` 쌍은 arm 간 비교에서 무너진다.** `c_student`가
> arm마다 다르므로, 회복이 잘된 arm일수록 두 입력이 서로 가까워져 `S`가 자동으로 0으로
> 붕괴한다(우리 회복 모델은 degen 0.000이라 이미 그 영역이다). 그러면 "조건부 경로가 살아
> 있는 정도"가 아니라 "CoC가 teacher와 얼마나 비슷해졌는가"를 재게 된다. 한 arm 안에서
> 원본 대비 비교하는 데는 문제가 없지만, Phase A의 B0/P1/P2 줄세우기에는 쓸 수 없다.
>
> **프로토콜 주의**: `S`는 본질적으로 주어진 CoC를 teacher-forcing해야 측정된다. 고정
> 프로토콜의 "TF 금지"는 **보고 지표**에 대한 규칙이므로, `S`는 **진단 프로브이며 개루프
> 성능 수치로 보고하지 않는다**고 명시하면 충돌이 없다. 구현은 `run_baseline.py`의 OOD TF
> 경로(`--no-tf`로 끄는 그 분기)에서 조건 CoC만 바꿔 끼우면 된다.

### 3-3. 스테이지별 괴리 추적

매 스테이지 종료 시 동시 기록: 로컬 ΔMSE / 전역 CoC KL / **free-running** Lingo-Judge
(하한 37.0 병기) / NLI entailment / action MSE.

> MSE는 낮은데 생성 지표가 떨어지는 패턴 = exposure bias의 증거. O0 vs O1 arm이 이걸
> 해결하는지로 닫는다.

### 3-4. 최종 평가 — **고정 프로토콜 준수 (원안 수정)**

| 축 | 프로토콜 | 역할 |
|---|---|---|
| **개루프** | 고정 프로토콜: **rollout-only(TF 금지)**, val 500 / test 500 / OOD-val 262, **minADE@6·minFDE@6 평균**, clip_id 유도 paired seed, K=8 실행 후 앞 6개 축소, **Ada 카드** | **주 판별 도구.** n=500 paired 해상도 ≈ **0.02 m** |
| **폐루프** | alpasim `public_2601`, 150–300 씬 x 2 롤아웃, paired bootstrap | **확인 도구.** Phase A 승자 + Phase B 유의 arm만 |

> **원안의 "937 chunk 전수, chunk당 2–3 샘플"은 우리 평가셋이 아니다.** 그 프로토콜로는
> baseline(test 0.842)·dual(0.950)·회복 앵커(1.008) 등 **기존 모든 arm과 페어링이 깨져**
> 비교가 불가능하다.
>
> **"개루프는 sign reversal 전력이 있어 결정 근거로 쓰지 않는다"도 뒤집어야 한다.** 검정력이
> 정반대이기 때문이다:
>
> | | 해상도 | 근거 |
> |---|---|---|
> | 개루프 n=500 paired | **≈0.02 m** | dualr−dual −0.052* 급 차이를 잡음 |
> | 폐루프 n=150 | 0.067 | 씬별 σ 0.28–0.30, 동점 49–55%, 같은 config 재롤아웃 차이 0.145 |
> | 폐루프 n=300 | 0.047 | 여전히 0.05급 차이를 못 봄 |
>
> 올바른 서술은 **"개루프 단독으로 배포를 판단하지 않는다(폐루프 확인 필수)"**이지
> "결정 근거로 쓰지 않는다"가 아니다. 개루프를 버리면 판별 도구가 남지 않는다.
> 앵커(test 1.008) 대비 남은 헤드룸이 ≈0.06 m이므로, **폐루프로는 원리적으로 판별 불가**,
> 개루프로는 판별 가능하다.

---

## 4. 진행 순서

| # | 작업 | 게이트 조건 |
|---|---|---|
| 0 | Ranking drift 진단 (3-1) | drift가 **잡음 바닥(Q 0.860 / MLP 0.782) 안쪽** → P2 폐기, 프레이밍 전환 |
| 1 | mask-toggle + `disable_adapter` teacher + §1-1 `L_local` 구현, 24%에서 sanity check | `mask off + adapter off ≡ 원본` assert 통과 & 로컬 MSE 하강 & free-running 지표 비악화 |
| 2 | **B0 / P1 @ 33%, 예산 T 매칭** (P2는 게이트 0 통과 시에만) | P1 ≈ B0면 스케줄 효과 없음 → 조건부 손실항으로 무게중심 이동 |
| 3 | Phase B ablation (T → G → F → O 순) | — |
| 4 | 전 arm에 3-2 민감도 지표 적용 (고정 대조 쌍) | — |
| 5 | 승자만 closed-loop | — |

> **레이어 제거는 별도 트랙으로 분리.** 블록 전체를 제거하면 학습할 파라미터가 없으므로
> compensation window(예: 6개 중 1개 제거 + 나머지 5개 학습) 정의가 선행돼야 한다.
> 추가 제약: **expert가 VLM의 per-layer KV 캐시를 읽는다**(expert 층 i ← VLM 층 i, 양 타워
> 모두 36층). 따라서 VLM 블록을 없애려면 expert 블록도 **쌍으로** 제거하거나 캐시 매핑을
> 재색인해야 한다.

---

## 5. 리스크 체크리스트

- [ ] **예산 confound** — progressive는 스테이지 수만큼 step이 늘어난다. 총 토큰/step 고정
      안 하면 "더 오래 학습했다"를 측정하게 된다. **가장 큰 함정**
- [ ] **데이터셋 x epoch 미명시** — 예산 T만으로는 부족하다. (샘플 수 x epoch)를 적어야
      §1-3 캐시 전략과 앵커 비교가 결정된다
- [ ] **teacher 오염** — merge 금지, `disable_adapter()` 경로 assert (§1-3)
- [ ] **타깃 정의 오독** — `L_local`의 빼는 기준을 `h*_{k-1}`로 잡으면 B1의 교정성이 사라져
      B1/B2 ablation이 무의미해진다 (§1-1)
- [ ] **프로토콜 비대칭** — 일부 arm만 recovery 포함하면 arm 간 해석 불가
- [ ] **평가셋 오염** — KD용 unlabeled 클립과 val_500 / test_500 / OOD-val 262의 분리 확인.
      `outputs/recovery_sets/`의 기존 매니페스트가 이미 calib_100 겹침 9클립과 val_500을
      제외해 두었으므로 그대로 재사용
- [ ] **CoC 망각** — beta/gamma 비율이 action 쪽으로 기울면 CoC 품질이 조용히 하락.
      스테이지마다 free-running Lingo-Judge 동시 모니터 (**하한 37.0 대비로 해석**)
- [ ] **디스크** — `/mnt/nvme1n1` 98% 사용(여유 158 GB). hidden 전수 캐시 불가, 슬림
      체크포인트는 16.8 GB/개
- [ ] **LoRA 용량** — 1벌 유지 + 부분 unfreeze면 스택 없음. rank는 ablation 대상
- [ ] **순서** — front-to-back 고정

---

## 6. 비용 (원안에 없음)

| 항목 | 단가 | 비고 |
|---|---|---|
| 개루프 3셋 (val 500 + test 500 + OOD-val 262) | **≈3 GPU-시간 / arm** | 주 판별 도구 |
| 폐루프 150씬 x 2롤아웃 | **≈8 GPU-시간 / arm** (Ada 4장) | 300씬이면 ≈16 |
| LingoQA 500문항 (답변 + judge) | ≈40분 / arm | 하한 37.0 대비 |
| 회복 학습 (앵커 기준 600step x accum 8) | ≈4 GPU-시간 + probe 3시간 | PBD는 스테이지 수만큼 분할 |

Phase A를 B0/P1 두 arm으로 시작하면 학습 ≈8시간 + 개루프 6시간이고, 폐루프는 승자 1개만
투입해 8시간이다. P2까지 포함하고 Phase B를 폐루프로 확장하면 폐루프만 40시간을 넘으므로
**§4의 "승자만 closed-loop"를 반드시 지킨다.**

---

## 7. 개정 이력 (rev. 1, 2026-08-24)

| # | 원안 | 수정 | 이유 |
|---|---|---|---|
| 1 | `L=24`, 블록 4 → 6스테이지 | **L=36, 블록 6 → 6스테이지** | Alpamayo 1.5는 VLM·expert 모두 36층 |
| 2 | `L_local`을 `Δh`에 걸되 타깃은 `h*_k` | **타깃 업데이트 = `h*_k − ĥ_{k-1}`** | 원안 조합은 최적점에서 `δ_k = δ_{k-1}`로 드리프트가 통과 → B1의 "자기 교정" 성질이 사라짐 |
| 3 | merge 후 freeze, "mask off = teacher" | **merge 금지, `disable_adapter()` = teacher** | merge 시 mask-off가 원본+누적 LoRA(이중 보상)가 되어 타깃이 매 스테이지 이동 |
| 4 | "prefix 공유, 블록 k에서만 분기" | **B2 한정으로 격하**, B1은 두 궤적 분리 | B1의 `h*_k`는 dense 자기 궤적에서만 나옴 |
| 5 | (캐시 전략 없음) | **teacher 최종 출력만 전수 캐시(~1.2 GB), hidden은 재계산, teacher는 블록 k까지만** | hidden 전수는 10k 샘플에 253 GB로 불가 (여유 158 GB) |
| 6 | `S = ||a(c_student) − a(c_teacher)||` | **arm 무관 고정 쌍 `(c_ref, c_cf)`, `S/S0` 정규화, 부호 병기** | `c_student`가 arm마다 달라 회복이 잘될수록 `S`가 자동 붕괴 → arm 간 비교 불가 |
| 7 | 개루프 "937 chunk 전수", "결정 근거로 쓰지 않음" | **고정 프로토콜(val500/test500/OOD-val262, minADE@6) + 개루프를 주 판별 도구로** | 937 chunk는 우리 셋이 아니라 기존 arm과 페어링 불가. 해상도는 개루프 0.02 m vs 폐루프 0.067 |
| 8 | B0를 새 실험으로 | **B0는 KD 목적함수로 신규 실행, `slim_recover_dual_u55`는 앵커로 병기** | 구조는 같지만 목적함수가 CE+FM이라 깨끗한 대조군이 아님. 다만 "이미 test 1.008/degen 0.000 도달"이라는 기준선을 세움 |
| 9 | 게이트 0 = "drift 없음 → 폐기" | **"잡음 바닥 안쪽 → 폐기" + it3 기각 사실 명시** | drift가 있다는 건 이미 앎(it3). 문제는 그 drift가 잡음이었다는 것 |
| 10 | (비용 없음) | **§6 신설** | 폐루프 8시간/arm이 설계를 지배 |
| 11 | 레이어 제거 별도 트랙 | **+ expert-VLM per-layer 캐시 결합 제약 명시** | 블록을 쌍으로 제거하지 않으면 expert 층이 캐시 소스를 잃음 |
