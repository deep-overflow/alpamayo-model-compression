# tyr_t: Tyr 공식 절차에서 OSSCAR 헤시안을 텍스트 토큰만으로 (A100 빌드·평가)

날짜: 2026-11-03. 상태: **취소** (사용자, 2026-11-03: "아니야 이제 pruning ratio 실험 진행하자"). 빌드·평가 없음.

## 0. 목적

- 같은 절차에서 OSSCAR 재구성 헤시안의 비전:텍스트 비중만 바꿔 온 두 점이 있다.

  | arm | 비전:텍스트 | test2000 minADE@6 | LingoQA | OOD CoC 채점 (OOD-train) |
  |---|---|---|---|---|
  | `tyr_off` | 토큰 수 그대로 (약 95:5) | 0.882 | 38.0% | 2.75 |
  | `tyr_vt` | 1:1 | 0.931 | 62.8% | 2.91 |

- 세 번째 점 `tyr_t`(0:1, 텍스트만)를 더해 맞바꿈 곡선을 완성한다. 비전 쪽 비중을 0으로 하면 두 가지를 본다.
  - 언어(LingoQA, CoC)가 dense·MoOP l0keep 수준까지 더 오르는지
  - 주행이 얼마나 더 나빠지는지
- 탐색 적합도(dense에 대한 top-8,192 KL)는 `tyr_off`부터 이미 텍스트 위치(클립당 프롬프트 157 + CoC 16)에서만 잰다.
  그러니 "텍스트에만 적용"에서 바뀌는 것은 헤시안 하나다.

## 1. 사용자 결정 (2026-11-03)

| 항목 | 결정 |
|---|---|
| 적용 범위 | OSSCAR 헤시안을 프롬프트 텍스트 토큰만으로 만든다. 나머지는 공식 절차 그대로이고, 감쇠 보강은 하지 않는다 |
| 평가 | test2000, OOD-Val, LingoQA + OOD-train 주행 + 정답 CoC NLL + Claude CoC 채점 |
| 빌드 | KISTI A100 |

## 2. 설정

- `tyr_off`·`tyr_vt`와 같은 것:
  - 출발점 s = 0.39856(제거 2,657,452,032 = MoOP)
  - supernet·탐색 각 4회, 수준 격자, 논문 식 5 오차 누적, calib_100
  - 탐색 50세대 × 128, 1/5/42 클립, 생존 16/4/2
  - teacher 재사용(`cvpr27/tyr_off/teacher`)
- 다른 것 하나: `supernet.py --hessian-mix t_only`(새 선택지).
  - `StreamHessianHook`에 텍스트 스트림 하나만 둔다. `mix_hessians(H, n, {T: 1})`이므로 H = H_T / N_T다.
  - 마스크는 `tyr_vt`와 같은 `compute_spans(...)["text"]`다.
- 표본이 적다. 텍스트 토큰은 15,700개(클립당 157)이고 down_proj 입력은 12,288차원이라 헤시안의 조건수가 나쁠 수 있다.
  - 감쇠는 공식값(1e-2 × 평균 대각)을 그대로 쓴다.
  - 8월 dualr 실험에서도 텍스트 단독 헤시안은 조건수 때문에 탈락했다. 그래서 아래 G2로 확인한다.
- 이름: `TYR_OUT=cvpr27/tyr_t`, 레시피 `outputs/cvpr27/slim_tyr_t`.

## 3. 게이트

- **G0 (기존).**
  - 매 supernet 실행의 0번 블록에서 조기 종료 헤시안이 전체 prefill 헤시안과 같아야 한다(혼합 후, 상대차 ≤ 1e-6).
  - 라운드마다 유지 총량이 불변(684 head / 266,040 채널)이어야 한다.
  - 최종 구성에 head·채널이 0개인 층이 없어야 한다.
- **G2 (새로 추가, 빌드 잡 안).** `check_level_norms.py`로 최종 구성(R=4)의 모듈별 가중치 크기를 dense와 비교한다.
  - 최대 비율이 **10× 이하**여야 한다. 참고로 `tyr_off` 최종 구성은 최대 3.45×였다.
  - 넘으면 잡이 실패로 끝나고, 연결된 평가 홀더는 시작하지 않는다. 보고하고 멈춘다.
  - 모든 라운드의 상위 비율은 로그로 남긴다.
- 스모크: make_slim 스모크에서 CoC 길이와 NLL을 기록한다.

## 4. 평가

- A100 트랙 arm `tyr_t`. 별도 큐 `queue_tyrt`를 쓰고 홀더는 빌드에 afterok로 연결한다.
- 작업 14개:
  - 주행: test2000 4샤드, OOD-Val 2샤드, OOD-train 4샤드
  - LingoQA 1개
  - 정답 CoC NLL: OOD-Val 1샤드, OOD-train 2샤드
- 비교 기준(REFS): MoOP, MoOP l0keep, tyr_off, tyr_vt. 짝지은 Δ는 부트스트랩 200,000회다.
- Claude CoC 채점: `grade_coc_claude.py --exp-id ood_coc_v1 --arms tyr_t --sets oodval oodtrain`.
  - 같은 루브릭·모델·effort로 기존 7개 arm과 같은 파일에 쌓는다.
  - 약 1,690건, 약 $4.4.
- 분석: `analyze_a100.py`, `analyze_coc_quality.py`에 arm을 추가한다.

## 5. 비용·일정

- 빌드: 약 22 A100-시간(A100 4장, 벽시계 약 6시간). 지금 NEURON 계정에 돌아가는 잡은 없다.
- 평가: 약 10 A100-시간. 4장 홀더 2개로 벽시계 약 1.5–2시간.
- Claude API: 약 $4.4.

## 6. 산출물·git

- 리포트: `cvpr27/reports/<날짜>_tyr-t-text-hessian.md`. tyr_off → tyr_vt → tyr_t 세 점의 주행·언어 맞바꿈과 층별 배분 비교를 담는다.
- 기존 Tyr 비교 라인의 후속이므로 main에 바로 커밋한다.
