# OOD-train 개루프 + OOD CoC 품질 (A100, 2026-11-02)

## 0. 목적

- **주행:** OOD-Val(262클립)만으로는 long-tail 결론의 CI가 넓다(MoOP l0keep − MoOP OOD +0.063이 클립 5개에 좌우됨).
  OOD-train 1,271클립을 두 번째 long-tail 세트로 더해, OOD에서의 arm 순위가 세트를 바꿔도 유지되는지 본다.
- **CoC 품질:** 지금 A100 표의 언어 지표는 LingoQA와 CoC 퇴화율뿐이다. OOD 두 세트에는 사람이 큐레이션한 정답 CoC
  (`gt_coc`, 한 문장 평균 12단어: "Stop at the stop sign while waiting for pedestrians to cross the street.")가 있으므로,
  arm이 **맞는 추론**을 내는지를 두 방식으로 잰다.
  - ① 정답 CoC NLL(`nll_gtcoc`): 모델이 정답 추론을 얼마나 그럴듯하게 보는가. 생성과 무관하고 seed도 없다.
  - ② Claude 채점: 모델이 실제로 생성한 CoC가 정답과 같은 행동·원인을 말하는가.
- 가설: 언어가 무너진 arm(Wanda-sp 39.0%, Tyr 38.0%, LLM-Pruner 53.4%)은 두 지표에서 모두 MoOP·MoOP l0keep보다 나쁘다.
  Tyr처럼 주행은 dense와 같고 언어만 무너진 arm에서 ①과 ②가 갈리는지(정답은 알아보지만 스스로 말하지 못하는지)가
  관심 지점이다.

## 1. 사용자 결정 (2026-11-02)

| 항목 | 결정 |
|---|---|
| OOD-train 주행 평가 arm | 핵심 + 외부 기준선 7개: dense, MoOP, MoOP l0keep, LLM-Pruner `lp_m24`, Tyr `tyr_off`, Wanda-sp, `tyr_vt` |
| CoC 품질 지표 | 정답 CoC NLL + Claude 채점 (둘 다) |
| CoC 품질 세트 | OOD-Val, OOD-train (같은 7개 arm) |

## 2. 세트

- `ood_train.parquet` = `ood.parquet`에서 `split == "train"`인 1,271클립(= OOD 1,533 − OOD-Val 262, 겹침 0 확인).
  클러스터: 공사 구간 621, 보행자 300, 특이 차량 171, 자전거 59, 교차로 33, 기타 25, 비상 25, 동물 21 등.
  로컬 `outputs/eval_sets/`에 만들어 NEURON `/scratch/e1997a06/outputs/eval_sets/`로 보낸다. 해시를 대조한다.
- **오염 없음:** 7개 arm은 모두 `calib_100`(공식 train, OOD와 겹침 0)으로 보정됐다. OOD-train에서 뽑은 `calib_ood_100`과
  OOD-train을 쓰는 회복 학습은 이 arm들과 무관하다.
- 데이터: NEURON `datasets/physicalai_av/pre_processed/ood/samples`에 1,533개 전부 있다(8.5 GB). 전송은 필요 없다.

## 3. 주행 (OOD-train)

- 프로토콜은 OOD-Val과 같다: `run_baseline.py --set ood --manifest ood_train --no-tf`, k=8, 클립별 seed 42, max_gen 256.
  지표는 minADE@6 / minFDE@6을 1.6 / 3.2 / 6.4 s에서 잰다.
- `eval_a100.sbatch`에 세트 `oodtrain`(`--set ood --manifest ood_train`)을 추가한다. arm마다 4샤드라 작업 28개다.
- exp-id `cvpr27/a100/<arm>_oodtrain`. `analyze_a100.py`의 `SETS`에 `oodtrain`을 넣어 같은 표와 짝지은 Δ에 나오게 한다.

## 4. CoC 품질 ① — 정답 CoC NLL

- 새 스크립트 `cvpr27/code/run_nll_gtcoc.py`로 NLL만 잰다. 롤아웃과 궤적 샘플링이 없어 클립당 약 1–2 s다.
  - 구간은 `run_baseline`의 TF 경로와 같다: `gt_coc_seq` = `[prompt] + tokenize(gt_coc) + <cot_end> + <traj_future_start>`.
  - forward 한 번에 그 구간의 cross-entropy 평균(nats)을 낸다. 계산은 `run_eval.eval_config_samples`의 `nll`과 같다.
  - `run_nll_dense.py`의 `span_nll`을 그대로 쓰고 기준 텍스트만 `gt_coc`로 바꾼다.
  - NEURON kd2 트리의 기존 파일은 고치지 않는다. 새 파일만 `cvpr27/code/`에 둔다.
- 이 NLL은 언어 지표다. 정답 CoC를 문맥으로 넣고 주행을 재는 TF 주행 지표는 프로토콜상 계속 쓰지 않는다.
- `eval_a100.sbatch`에 작업 종류 `nll <arm> <set> <shard> <n>`을 추가한다.
  - OOD-Val은 arm당 1샤드, OOD-train은 2샤드다. 작업 21개다.
  - exp-id는 `cvpr27/a100/nll_<arm>_<set>`이다.
- **G1 (NLL 게이트, 큐 맨 앞).** 비교 대상은 저장된 Ada 행 `baseline_pred_oodval`(평균 3.139)과
  `dual_u40_v2_pred_oodval`(3.185)이다. A100에서 잰 dense·MoOP의 OOD-Val NLL이 다음을 모두 만족해야 한다.
  - 클립별 |Δ| 평균 ≤ 0.02
  - arm 평균 차 ≤ 0.01
  - 클립 262개 모두 존재
  - 근거: 같은 계산이면 하드웨어 차이만 남고, 이전 Blackwell–Ada 차는 클립당 최대 0.03이었다.
  - 구간 규칙이 틀리면 평균이 크게 어긋나므로 이 게이트로 잡힌다.
  - **실패하면 나머지 `nll` 줄을 멈추고 원인부터 본다.**

## 5. CoC 품질 ② — Claude 채점

- 새 스크립트 `cvpr27/code/grade_coc_claude.py`. GPU 없이 로컬의 가벼운 venv(`venvs/longtail`, anthropic SDK)로 돌린다.
  API 키 로딩, 프롬프트 캐싱, 비용 집계는 `long-tail-detection/label_longtail.py`의 방식을 그대로 쓴다.
- 모델은 `claude-opus-5-5`다. **텍스트만 보낸다**(정답 CoC + arm의 생성 CoC). 이미지는 보내지 않는다.
- **블라인드.** 요청 하나에 (정답, 후보) 한 쌍만 넣는다. arm 이름은 보내지 않고, 요청 순서는 arm·세트를 섞는다.
- **루브릭(JSON 출력).**

  | 항목 | 값 | 뜻 |
  |---|---|---|
  | `action` | 0 / 1 / 2 | 후보의 주행 결정이 정답과 반대 / 부분 일치 또는 불명확 / 일치 |
  | `cause` | 0 / 1 / 2 | 정답이 든 핵심 원인(물체·행위자·조건)을 놓침 / 범주만 맞음 / 일치 |
  | `contradiction` | bool | 정답과 양립할 수 없는 요소를 단정 |
  | `coherent` | bool | 읽을 수 있는 문장인가(단어 나열이면 false) |
  | `score` | 1–5 | 종합 일치도 |

  주 지표는 `score` 평균과 `score ≥ 4` 비율이다. 보조 지표는 `action == 2` 비율과 `cause == 2` 비율이다.
  퇴화한 CoC도 건너뛰지 않고 똑같이 채점한다.
- **대조.**
  - (i) 잡음 바닥: 전체 쌍의 10%를 한 번 더 채점해 `score`의 반복 일치도를 잰다.
  - (ii) 우연 바닥: dense CoC를 다른 클립의 정답과 짝지은 OOD-Val 262쌍. 무관하지만 그럴듯한 추론이 받는 점수다.
- **비용(사전 추정).** 시스템 프롬프트 약 700토큰(캐시), 요청당 입력 약 150, 출력 약 120토큰이다.
  - 가격은 repo에 기록된 $4 / $20 / 캐시 읽기 $0.20 per MTok를 쓴다. 요청당 약 $0.003이다.
  - 요청 수: 7 arm × 1,533쌍 = 10,731 + 반복 1,073 + 우연 바닥 262 ≈ 12,100건이다.
  - 표준 요금으로 **약 $35–40**, Batch API를 쓰면 그 절반이다.
  - **파일럿 50쌍**으로 실제 토큰을 잰다. 추정의 2배를 넘으면 멈추고 보고한다.
- 순서: OOD-Val은 6개 arm의 생성 CoC가 이미 있으니 바로 시작한다. `tyr_vt`는 현재 평가가 끝나면(약 23:00 KST) 넣는다.
  OOD-train은 3절의 행이 생기는 대로 채점한다.

## 6. 분석

- 주행: `analyze_a100.py --stage table`에 `oodtrain` 행과 짝지은 Δ(부트스트랩 200,000회)를 추가한다.
  비교 기준은 dense, MoOP, MoOP l0keep이다.
- CoC 품질: 새 스크립트 `cvpr27/code/analyze_coc_quality.py`.
  - arm × 세트별 `nll_gtcoc` 평균, 채점 지표, 퇴화율을 낸다.
  - 짝지은 Δ는 dense·MoOP 대비 부트스트랩 200,000회로 낸다.
  - 클러스터별로 나눠 본다.
  - 채점 점수와 minADE@6의 클립 단위 상관을 본다.
- 리포트: `cvpr27/reports/<날짜>_oodtrain-coc-quality.md`(사실/해석 분리, 절댓값 우선).

## 7. 실행·비용

| 작업 | 줄 수 | A100-시간 |
|---|---|---|
| G1 (dense·MoOP OOD-Val NLL) | 2 | 0.4 |
| NLL 나머지 (OOD-Val 5 + OOD-train 14) | 19 | 약 6 |
| OOD-train 주행 (7 arm × 4) | 28 | 약 20 (Wanda-sp는 CoC가 256토큰이라 약 2배) |
| 합계 | 49 | **약 27** |

- 홀더는 `eval_a100.sbatch` 4장짜리 2개다.
  - 계정 한도는 실행 2개·제출 4개다. tyr_vt 홀더 2개가 끝나는 약 23:00 KST에 시작된다.
  - 시작 후 약 4시간 걸려 **11-03 03:00–04:00 KST 완료**로 예상한다.
- 큐 순서: G1 → OOD-train 주행 → NLL OOD-train. G1 결과를 확인한 뒤 NLL 나머지 줄을 연다.
- Claude 채점: 약 $35–40. 로컬 CPU만 쓴다.

## 8. git

- 기존 A100 트랙의 후속이므로 main에 바로 커밋한다.
