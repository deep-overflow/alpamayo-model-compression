# Dual 주요 실험 코드 검토 — 2026-09-10

**결론: 중요도 gate 구현과 expert MLP znorm/채널 선택에서는 오류를 발견하지 않았다.
다만 논문 해석 또는 재현을 바꾸는 문제 4개를 확인했다.** 우선순위는 dualr_wl의
Hessian 혼합 계수와 재구성 checkpoint 로더이며, 기존 dual의 상수 rank와 예전
 dualr의 수치 불안정은 이미 알려진 문제의 잔존 여부·실제 영향을 재확인했다.

실험 코드나 원본 결과는 변경하지 않았다. 신규 GPU 측정·빌드·평가도 실행하지 않았다.

## 검토 범위와 근거

- 핵심 계열: dual / traj / coc, dualr / dualr_rep / dualr_wl,
  dualrwl+expert MLP 50/75/87.5/93.75%, dual+expert MLP 93.75%, 공통 baseline.
- 12 arms × val500/test500/OOD-val262 = **36개 결과 묶음, 15,144개 저장 row**.
- Expert 96.875/98.4375/100%는 점수·mask·state 보관 상태만 추가 확인.
- LingoQA 5개 arm × 500문항의 저장된 judge 결과 및 평가 키 검증.
- 기존 m2601 폐루프 5개 arm, 각 150 scenes × 2 rollouts의 집계 완전성 검증.
- CPU에서 실제 함수 추출 실행, 원본 NPZ 재집계, recipe 대조, mmap checkpoint
  tensor 대조. 후속 탐색 전체, hard100/새 search, quantization/recovery는 범위 밖이다.

재실행 가능한 [검토 스크립트](../../experiments/evaluation/analyze_dual_code_audit.py)와
[전체 측정 JSON](2026-09-10_dual-code-audit.json)을 함께 보관한다.

## 1. [P1] dualr_wl의 전체 Hessian 혼합 계수가 문서의 0.68/0.04/0.16/0.12와 다름

**위치:** `run_cache_recon.py:116–139`, `:279–294`, `:103–110`.
[토큰 가중치 코드](../../experiments/head_analysis/run_cache_recon.py#L116),
[실제 실험 계획](../../plans/2026-08-30_dualr-w-lingo.md).

`token_weights`는 각 sample 내부에서 token 평균을 만들지만 데이터 소스별 sample
수로는 나누지 않는다. 주행 100 clips에는 각각 0.68/0.04, QA 600 samples에는 각각
0.16/0.12를 부여하고, `WeightedHessianHook`이 모든 sample을 단순 합한다.

각 스트림의 sample-mean covariance를 H̄라 하면 실제 H는 다음과 같다.

```text
H = 68 H̄_driving_prefill + 4 H̄_ownCoC
    + 96 H̄_QA_prefill + 72 H̄_QA_answer
```

전체 공통 배율을 제거했을 때의 계수:

| 스트림 | 문서에서 전체 몫으로 제시 | 실제 sample-mean covariance 계수 |
|---|---:|---:|
| 주행 prefill | 68% | 28.33% |
| Own-CoC | 4% | 1.67% |
| QA prefill | 16% | 40.00% |
| QA answer | 12% | 30.00% |
| QA 합계 | 28% | **70%** |

이 값은 covariance의 trace/energy 비중이 아니라 **각 데이터 분포 평균 앞의 혼합
계수**다. 데이터별 activation 크기가 달라 실제 energy 비중과는 다르다.

**영향:** 현재 `dualr_wl` 및 이를 상속한 모든 `dualrwl_em*`는 QA 쪽에 훨씬 더 큰
계수가 적용된 refit이다. 기록된 LingoQA 72.6%와 주행 성능은 실제 checkpoint의
결과로 남지만, “QA를 28%만 섞었다” 및 “decode 총 몫 16% 유지”라는 해석은 틀리다
(실제 own-CoC+QA answer 계수 합은 약 31.67%).

**조치:** 기존 결과를 유지한다면 방법을 실제 sample 가중치/전체 계수로 정정한다.
의도한 전체 혼합을 실험하려면 각 데이터 소스의 sample 수로 나누어 H를 구성하고
별도 arm으로 다시 refit·평가한다. 기존 checkpoint 이름이나 결과 파일을 덮어쓰면 안 된다.
기존 wl+MLP의 부모-자식 비교는 동일한 wl 가중치를 공유하므로 이 문제만으로
expert MLP의 추가 효과 비교가 무효가 되는 것은 아니다.

## 2. [P1] 재구성 state가 없어도 load_slim이 selection-only 모델로 조용히 대체

**위치:** [slim_lib.py:250–258](../../experiments/head_analysis/slim_lib.py#L250).

`load_slim`은 먼저 원본 가중치를 recipe로 slice하고, `slim_state.pt`가 있을 때만
재구성 가중치를 load한다. 파일이 없으면 오류 없이 원본을 slice한 모델을 반환한다.
이 동작은 dual 같은 selection-only recipe에는 맞지만 **refit된 dualr/dualr_wl에는
맞지 않는다**. `make_slim`의 `--no-state` 차단은 빌드 단계에만 있으며 로더를 보호하지 않는다.

현재 없는 state:

```text
slim_dualrwl_em50_u40
slim_dualrwl_em75_u40
slim_dualrwl_em87p5_u40
slim_dualrwl_em96p875_u40
slim_dualrwl_em98p4375_u40
slim_dualrwl_em100_u40
```

**영향:** 현재 이 디렉터리를 `--model`로 지정하면 이름은 dualrwl이어도 VLM refit이
빠진 모델을 재평가한다. Meta의 kept indices가 동일해도 가중치가 같다는 보장은 없다.
실제로 `recipe_gates()`의 `vlm_identical_to_wl` 검사는 indices만 비교하므로 이
차이를 검출하지 못한다(`analyze_dualrwl_em.py:137–139`).

**과거 결과와 구분:** 파일이 현재 없다는 사실은 과거 평가 시점에도 없었다는 증거가
아니다. 확인한 em50/75/87.5/93.75 저장 결과는 각각 1,262 clips의 CoC가 wl과 전부
같고, 보관된 em93.75 state의 VLM refit matrices 72개는 wl과 tensor 단위로 같다.
과거 사다리 실험을 잘못 실행했다고 단정하지 않는다.

**조치:** refit recipe에는 `requires_state`/가중치 출처를 저장하고 state가 없으면
즉시 실패시킨다. 재현용 가중치는 원래 wl checkpoint/supernet에서 복구해야 한다.
새 추론을 실행해 이 fallback을 시험하지 않고 로더 코드와 실제 파일 상태로 확인했다.

## 3. [P1, 기존 알려진 문제] 예전 dualr checkpoint에 수치 불안정한 refit 잔존

**위치:** [tyr_lib.py:185–197](../../experiments/head_analysis/tyr_lib.py#L185),
[run_tyr_supernet.py](../../experiments/head_analysis/run_tyr_supernet.py).

예전 경로는 FP32 Hessian solve 결과를 품질 검사 없이 저장한다. `safe_refit`을
적용한 새 경로와 달리 mask-only보다 reconstruction error가 큰 해도 받아들인다.

현재 `slim_dualr_u40` state의 layer 35 `o_proj`에서 max|W|=**46.25**를 확인했다.
원본 slice인 dual에서는 0.59765625, wl에서는 2.453125다. 큰 norm 자체만을 오류의
근거로 삼지 않고, 보관된 `dualr_rep_supernet_u40/g0_function_space.json`도 대조했다:
같은 dense calibration input에서 해당 모듈의 relative reconstruction error는
old dualr **0.398497**, safe-refit rep **0.127960**이다.

이 문제는 [기존 진단](../../plans/2026-08-30_dualr-weighted-hessian.md)에 이미
기록되어 있다. 저장된 LingoQA 점수도 old dualr 41.8% / rep 52.2%로 재확인했다.
다만 이 10.4pp 전체를 이번 CPU 검토만으로 한 모듈의 인과 효과라고 주장하지 않는다.

**영향:** old dualr를 유일한 재구성 대조군으로 삼으면 재구성 방식의 효과에 수치
실패가 섞인다. `dualr_wl` 보고서도 old dualr와의 폐루프 비교에 safe_refit 변경이
섞였음을 인정한다. 현재 old checkpoint를 수정된 dualr로 간주하면 안 된다.

**조치:** 재구성의 원리를 비교하는 대조는 수정된 경로의 `dualr_rep`로 맞추고,
old dualr는 역사적/결함 있는 구현 결과로 구분한다. 기존 폐루프 결과를 새로운
rep 결과인 것처럼 옮기지 않는다. wl에는 safe_refit이 적용되어 있으며 metadata의
72개 모듈 모두 기록된 `fit_obj_refit <= fit_obj_mask` 조건을 만족한다.

## 4. [P2, 기존 알려진 문제] dual의 0-vector rank가 마지막 레이어 선택에 개입

**위치:** [run_cocsafe.py:43–49](../../experiments/head_analysis/run_cocsafe.py#L43),
[tyr_lib.py:163–169](../../experiments/head_analysis/tyr_lib.py#L163).

`rank_norm`은 `argsort(argsort(x))/(U−1)`을 사용한다. Layer 35의 trajectory
Q/MLP score가 전부 0이어도 서로 다른 0–1 rank를 부여한다. 정보가 없는 trajectory
branch가 CoC rank와 max 경쟁을 하는 실제 선택 오류다. 동률의 순서는 NumPy
argsort에 의존하므로 반드시 단조 인덱스 순서라고도 일반화하면 안 된다.

원본 Blackwell importance로 recipe가 정확히 재현됨을 확인했다. 상수 trajectory
branch를 중립화해 CoC 단독으로 선택하면 마지막 레이어에서 다음이 바뀐다.

- Q heads: retained 19개 중 **4개 교체**.
- MLP channels: retained 7,390개 중 **1,764개 교체**.
- 다른 35개 레이어는 이 상수-branch 수정으로 변하지 않는다.

이미 `dualfix` 별도 arm이 있고, 해당 구현과 평가를 대조했다. 현재 main dual 및
wl의 dual selection에는 예전 rank 규칙이 남아 있다. 기존 dualfix는 Ada importance
기반 대조이므로 위 Blackwell recipe와 동일한 변경 실험이라고 혼동하면 안 된다.

**기존 “영향 0” 기록의 정확한 범위:** raw rows에서 `dualfix - dual_ada`를 재계산했다.

| 세트 | Paired median ΔminADE@6 | Mean ΔminADE@6 | 값이 달라진 clips | CoC 동일 |
|---|---:|---:|---:|---:|
| val500 | 0 | −0.000732 m | 72/500 | 428/500 |
| test500 | 0 | −0.004003 m | 65/500 | 435/500 |
| OOD-val262 | 0 | −0.012567 m | 39/262 | 223/262 |

중앙값 0은 모든 clip의 결과가 같다는 뜻이 아니다. 코드에서 median을 계산한 것은
오류가 아니지만 “결과가 정확히 같음/영향 없음”으로 확장한 문장은 정정해야 한다.
위 mean의 유의성을 이번 표만으로 주장하지 않는다.

**조치:** 논문에 기존 rank 구현과 상수 처리 대조를 공개하고, 향후 새 recipe는
상수 objective를 제외하는 정의로 고정한다. 기존 dual/wl 결과를 바꾸지 않고
모든 과거 checkpoint를 교체하는 것은 별도 실험이 필요하다.

## 문제가 확인되지 않은 핵심 부분

1. **Q-head/MLP gate Taylor 식:** 실제 `UnitGates` hook을 작은 CPU 모델에서 실행해
   gate gradient가 activation-gradient contraction과 일치함을 검증했다. Activation을
   추가로 곱하는 중복 연산 오류는 없다. 이는 hook 검증이며 전체 GPU FM backward의
   end-to-end 검증을 대신하지 않는다.
2. **Expert znorm:** 원본 `mlp_abs_step` → layer별 z-score → step 평균이 저장된
   `traj_exp_mlp`와 완전히 같았다. 7개 pruning 비율의 36개 레이어에서 kept channels와
   Q-head 보존이 recipe와 일치했다.
3. **실제 남아 있는 두 타워 결합:** em93.75의 VLM refit matrices 72개가 wl과 같고,
   dual+em93.75의 대응 matrices 72개가 dual과 같았다. 두 em93.75 모델의 expert
   state tensors 397개도 전부 같았다. 다른 VLM tensors 전체를 비교한 검사는 아니다.
4. **개루프 데이터 선택:** 36개 결과 묶음에서 clip 누락/추가/중복, calibration 중복,
   잘못된 clip seed, 6개 미만 sample, 비유한 ADE/FDE를 발견하지 못했다.
   First-six minADE 계산도 현재 보고 경로와 일치한다.
5. **CoC 보존:** expert 추가 5개 arm-parent 비교 각각 1,262 clips에서 생성 CoC가
   전부 일치했다. Expert 절단이 VLM 생성 경로를 바꾼 흔적은 없다.
6. **LingoQA:** QA calibration 300 segments와 평가 segments의 교집합 0.
   5개 arm 모두 고유 평가 키 500개와 일치하고 저장 logit>0 판정으로 정확도가
   재계산된다(73.2/68.8/41.8/52.2/72.6%). Judge 자체의 신규 추론은 하지 않았다.
7. **폐루프 집계:** 5개 주요 arm은 공통 150 scenes, 각각 scene당 2회, score 누락 0.
   `analyze_alpasim_pairs.py`는 scene 평균 후 paired 비교하므로 300 rollouts를
   독립 scene처럼 세는 오류는 보이지 않았다. 시뮬레이터 내부 scoring과 runtime
   model hash/seed 동일성까지 검증한 것은 아니다.

## 논문 작성 및 재실험 우선순위

| 항목 | 지금 가능한 처리 | 추가 실행이 필요한 경우 |
|---|---|---|
| dualr_wl 혼합 | 방법에 실제 전체 계수 28.33/1.67/40/30 명시 | 의도한 68/4/16/12의 효과를 주장하려면 refit·평가 |
| state 누락 | 로더의 조용한 fallback을 차단하고 provenance 복구 | 누락된 wl+MLP checkpoint 재사용 전 가중치 복구 |
| old dualr | 결함 있는 과거 대조로 표시, rep와 구분 | 수정된 재구성의 폐루프 효과를 주장하려면 matched rep 평가 |
| dual 0-rank | 기존 dualfix 대조의 median/mean/변경 clip을 함께 보고 | 새 canonical recipe의 성능을 주장하려면 별도 평가 |
| expert MLP znorm | 현재 식과 채널 선택을 방법에 사용 가능 | 이번 검토에서 score/mask 자체 재측정 필요는 발견하지 못함 |

## 검증의 한계와 재실행

실행 당시의 코드 commit와 모든 shard별 설정이 완전히 보관된 것은 아니다.
Open-loop runner는 여러 shard가 같은 `config.json`을 덮어쓰므로 현재 config의
GPU는 마지막 writer의 기록일 수 있다. 전부 Ada로 기록되어 있지만 그것만으로
모든 과거 shard의 GPU를 독립적으로 증명하지는 못한다. 현재 코드와 저장 artifact가
일치하는 범위까지 검증했으며, 확인되지 않은 과거 상태는 추정으로 채우지 않았다.

```bash
.venv/bin/python experiments/evaluation/analyze_dual_code_audit.py
```

코드는 원본 `outputs/` 및 checkpoint를 읽기만 하고 위 JSON만 갱신한다.
새 스크립트 Ruff lint/format 검사 및 CPU 실행을 완료했다. 공유 `.venv`/`outputs`
파일, 모델 가중치, 기존 보고서와 논문 그래프는 이 검토에서 수정하지 않았다.
