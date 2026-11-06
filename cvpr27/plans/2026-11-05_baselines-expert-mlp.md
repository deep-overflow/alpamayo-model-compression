# 외부 기준선 + expert MLP 절단: LLM-Pruner·Tyr (A100 개루프)

날짜: 2026-11-05. 상태: 완료 (2026-11-06, 리포트 `cvpr27/reports/2026-11-06_baselines-expert-mlp.md`).

## 0. 목적

- MoOP l0keep 위에서는 expert MLP를 93.75%까지 잘라도 개루프가 그대로였다(`2026-11-05_moop-l0keep-expert-mlp.md`).
- 같은 expert 절단을 **다른 방법으로 자른 VLM 위**에 얹어도 성능이 유지되는지 본다.
  - 유지된다: expert MLP 절단은 VLM 압축 방법과 독립인 공짜 추가 압축이라고 말할 수 있다.
  - 깨진다: VLM이 이미 망가진 경우(LLM-Pruner)나 재구성된 경우(Tyr), expert가 가진 여유를 그 VLM이 쓰고 있다는 뜻이다.
- 이번에는 LLM-Pruner와 Tyr에 적용한다. Wanda-sp 등 나머지 방법은 결과를 보고 정한다.

## 1. 사용자 결정 (2026-11-05)

| 항목 | 결정 |
|---|---|
| expert 선택 | MoOP와 같은 선택. `importance_stepexp_znorm`(스텝별 z-정규화 FM Taylor)로, 모든 층에서 같은 비율 |
| 비율 | expert MLP 75 / 87.5 / 93.75% (Q head·KV 그대로) |
| VLM | LLM-Pruner `lp_m24`, Tyr `tyr_off` (둘 다 u40 예산) |
| 평가 | test_2000, OOD-Val262, OOD-train1271 (MoOP l0keep + expert와 같음). LingoQA는 VLM이 같아 재지 않음 |

## 2. Arm

| arm | VLM | expert MLP 남는 채널/층 | 총 제거 (모델 대비) |
|---|---|---|---|
| `lp_m24` (기준, 있음) | LLM-Pruner u40 | 8,256 | 2,657,452,032 (23.99%) |
| `lp_m24_em75` | 같음 | 2,064 | 4,027,023,360 (36.35%) |
| `lp_m24_em87p5` | 같음 | 1,032 | 4,255,285,248 (38.41%) |
| `lp_m24_em93p75` | 같음 | 516 | 4,369,416,192 (39.44%) |
| `tyr_off` (기준, 있음) | Tyr 공식 | 8,256 | 2,657,452,032 (23.99%) |
| `tyr_off_em75` | 같음 (재구성 가중치 포함) | 2,064 | 4,027,023,360 |
| `tyr_off_em87p5` | 같음 | 1,032 | 4,255,285,248 |
| `tyr_off_em93p75` | 같음 | 516 | 4,369,416,192 |

expert 유지 집합은 세 방법(MoOP l0keep, LLM-Pruner, Tyr)에서 단계마다 같다. 그래서 VLM 방법만 다르다.

## 3. 구현

- **`lp_prune.py`:** `--expert-mlp-cut <pct>`(와 `--expert-importance`)를 추가한다.
  - 지금은 "expert는 범위 밖"이라 마스크가 전부 1이다. 이 옵션을 주면 expert MLP 마스크를 위 기준으로 만든다.
  - 예상 제거량(`--expect-removed`·내부 assert)에 expert 몫을 더한다.
  - 레시피는 `lp_m24`와 같은 인자에 `--expert-mlp-cut`만 더해 `--no-state`로 만든다.
- **`make_slim.py`:** `--expert-mlp-cut <pct>`를 추가한다.
  - `build_masks`가 돌려준 expert MLP 마스크가 전부 1일 때만 덮어쓴다. 이미 expert를 자르는 설정과 겹치면 거부한다.
  - Tyr는 `--config tyr_u40_r --tyr-supernet cvpr27/tyr_off/supernet_r4 --tyr-config cvpr27/tyr_off/search_r4/final_config.json`(tyr_off와 같음)에 이 옵션을 더한다.
  - 재구성 가중치를 쓰므로 `slim_state.pt`를 포함한다(레시피당 약 13–15 GB).
- **게이트(하나라도 틀리면 중단):**
  - VLM 유지 집합이 기준 레시피(`slim_lp_m24`, `slim_tyr_off`)와 층마다 같아야 한다.
  - Tyr: VLM 가중치(`slim_state.pt`의 VLM 텐서)가 `slim_tyr_off`와 비트 단위로 같아야 한다.
  - expert 유지 집합이 `slim_dualexp_u40_em<M>`(= MoOP l0keep + expert)과 층마다 같아야 한다.
  - 제거량이 위 표와 정확히 같아야 한다.
  - 스모크(CoC 길이·NLL)가 기준 레시피와 같아야 한다(VLM 동일).
- 빌드는 이 서버의 빈 카드 1장에서 한다. LLM-Pruner 3개에 약 5분, Tyr 3개에 약 10분(가중치 저장 포함)이다.

## 4. 평가

- A100(KISTI): test_2000(4샤드), OOD-Val262(2샤드), OOD-train1271(4샤드). arm 6개 × 10작업 = 60작업이다.
  - 프로토콜: `run_baseline.py --no-tf`, k=8, 클립별 seed 42, minADE@6·minFDE@6 × 1.6/3.2/6.4 s.
- 비교:
  - 각 방법의 기준 arm(`lp_m24`, `tyr_off`) 대비 짝지은 Δ를 6개 지표 모두에 대해 낸다(부트스트랩 200,000회).
  - MoOP l0keep 사다리 결과와 같은 표에 놓는다.
- 큐 `queue_expx`, 4장·2장 홀더 각 1개. Tyr 레시피 3개(가중치 포함)는 NEURON으로 보내야 한다(약 45 GB).
- 비용:
  - 약 36 A100-시간. LLM-Pruner는 CoC가 길어 arm당 더 걸린다.
  - 벽시계는 홀더가 시작한 뒤 약 6시간이다.

## 5. 산출물·git

- 리포트: `cvpr27/reports/<날짜>_baselines-expert-mlp.md`. 6개 지표 × 3세트로 쓰고, MoOP l0keep 사다리와 나란히 둔다.
- 기존 라인의 후속이므로 main에 바로 커밋한다.
