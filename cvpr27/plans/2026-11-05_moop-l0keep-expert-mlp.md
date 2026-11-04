# MoOP l0keep + expert MLP 절단 (A100 개루프)

날짜: 2026-11-05. 상태: 계획 (승인 대기).

## 0. 목적

- MoOP l0keep(VLM 24% 절단, 0번 층 Q head 보존) 위에 action expert의 MLP 폭을 더 잘랐을 때 개루프 주행이 유지되는지 잰다.
- 근거: 예전 MoOP(`dual_u40_v2`, l0keep 전) 위에서는 결과가 이랬다.
  - expert MLP만 93.75%까지 잘라도 개루프·폐루프 손실이 없었다(`dualexp_u40_em<M>`).
  - 메모리는 8.8 GB 줄었다(`dualexp-latency-memory-result`).
  - expert Q head를 자르면 비용이 컸다(r25 위 +0.069 m, 기각).
- 이번에는 출하 레시피인 l0keep 위에서 이 결과가 유지되는지 A100 표로 확인한다.

## 1. 사용자 결정 (2026-11-05)

| 항목 | 결정 |
|---|---|
| expert 절단 | MLP만, 75 / 87.5 / 93.75% (Q head·KV는 그대로) |
| 평가 | test_2000, OOD-Val262, OOD-train1271 (개루프 주행) |

LingoQA는 재지 않는다. expert만 자르면 VLM의 언어 출력이 MoOP l0keep과 같다.

## 2. Arm

| arm | VLM | expert MLP (층당 남는 채널 / 8,256) | 레시피 |
|---|---|---|---|
| `moop_l0keep` (기준) | `dual_u40_l0keep_v2` | 8,256 | A100에 있음 (3세트 모두) |
| `l0keep_em75` | 같음 | 2,064 | `dualexp_u40_l0keep_em75` (새 이름) |
| `l0keep_em87p5` | 같음 | 1,032 | `dualexp_u40_l0keep_em87p5` |
| `l0keep_em93p75` | 같음 | 516 | `dualexp_u40_l0keep_em93p75` |

- **expert 선택:** 예전 em 사다리와 같다. 중요도는 `importance_stepexp_znorm`(스텝별 z-정규화 FM Taylor)이고, 모든 층에서 같은 비율을 자른다.
- **제거량:** MoOP l0keep 2,657,447,936에 expert MLP 몫을 더한다. 채널당 3 × 2,048 = 6,144 파라미터다.

  | arm | 층당 절단 채널 | 추가 제거 | 총 제거 (모델 대비) |
  |---|---|---|---|
  | em75 | 6,192 | 1,369,571,328 | 4,027,019,264 (36.3%) |
  | em87p5 | 7,224 | 1,597,833,216 | 4,255,281,152 (38.4%) |
  | em93p75 | 7,740 | 1,711,964,160 | 4,369,412,096 (39.4%) |

## 3. 구현

- `make_slim.py`에 `dualexp_u40_l0keep_em<M>` 분기를 추가한다.
  - VLM 마스크는 `build_masks("dual_u40_l0keep_v2", ...)`를 그대로 불러 받는다. 출하 l0keep 선택을 재현하는 코드를 다시 쓰지 않는다.
  - expert MLP는 기존 `dualexp_u40_em<M>` 분기와 같은 식이다.
- 레시피는 이 서버 빈 카드 1장에서 `--importance importance_v2 --no-state`로 만든다. 3개에 약 2분이다.
- **게이트:**
  - VLM 유지 집합이 `slim_dual_u40_l0keep_v2`와 층마다 같아야 한다.
  - expert 유지 집합이 기존 `slim_dualexp_u40_em<M>`(l0keep 전, 같은 expert 선택)와 층마다 같아야 한다.
  - 제거량이 위 합과 같아야 한다.
  - 스모크(CoC·궤적)를 기록한다.

## 4. 평가

- A100(KISTI): test_2000(4샤드), OOD-Val262(2샤드), OOD-train1271(4샤드). arm 3개 × 10작업 = 30작업이다.
  - 프로토콜: `run_baseline.py --no-tf`, k=8, 클립별 seed 42, 1.6/3.2/6.4 s minADE@6·minFDE@6.
- 비교:
  - MoOP l0keep 대비 짝지은 Δ(부트스트랩 200,000회)
  - 사다리 단계 간 Δ
  - CoC 퇴화율. VLM이 같으므로 CoC 텍스트는 거의 같아야 한다.
- 큐 `queue_exp`, 4장·2장 홀더 각 1개.
- 비용: 약 18 A100-시간. 홀더가 시작한 뒤 벽시계 약 3시간이다.

## 5. 산출물·git

- 리포트: `cvpr27/reports/<날짜>_moop-l0keep-expert-mlp.md`. 절댓값 우선으로 쓰고, 메모리 절감(bf16 산술)도 함께 적는다.
- 기존 라인의 후속이므로 main에 바로 커밋한다.
