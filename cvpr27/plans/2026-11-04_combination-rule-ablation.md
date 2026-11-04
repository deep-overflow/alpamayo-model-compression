# 중요도 결합 규칙 ablation (u40, l0keep, A100)

날짜: 2026-11-04. 상태: 계획 (승인 대기).

## 0. 목적

- 논문 ablation 표에 들어갈 행을 만든다. MoOP는 두 Taylor 중요도를 하나로 합쳐 유닛을 고른다.
  - 궤적 FM 손실의 I_traj
  - CoC CE 손실의 I_CoC
  - MoOP의 결합 규칙: 층 안 순위로 바꾼 뒤 max = 합집합
- **결합 규칙만 바꾸고** 나머지는 MoOP l0keep과 똑같이 둔다.
  - 같은 것: 중요도 값, 예산, 층별 균일 배분, 0번 층 Q head 보존, expert·KV 미절단
  - 그래서 어떤 결합이 주행·언어를 지키는지가 바로 비교된다.

## 1. 사용자 결정 (2026-11-04)

| 항목 | 결정 |
|---|---|
| 규칙 | rank-mean, raw mean, raw max (3개) |
| raw 규칙의 FM 점수 | 10개 FM 스텝 각각의 Taylor 절댓값을 평균 (raw mean·raw max 둘 다) |
| 비율 | u40만 (MoOP와 같은 예산 2,657,447,936, l0keep) |
| 가중치 민감도 | 넣지 않음 |

## 2. Arm

| ablation 행 | 결합 | FM 점수 | CoC 점수 | 정규화 | 레시피 (`make_slim --config`) |
|---|---|---|---|---|---|
| MoOP l0keep (기준) | max | I_traj = 스텝 합의 절댓값 (`importance_v2`) | I_CoC (`importance_v2`) | 층 안 순위 | `dual_u40_l0keep_v2` (A100에 있음) |
| FM-only / CE-only (참고) | 한쪽만 | — | — | — | A100에 있음 |
| **rank-mean** | 평균 | I_traj (MoOP와 같음) | I_CoC | 층 안 순위 | `dualsum_u40_l0keep_v2` (기존 이름, 순위 합 = 평균 ×2라 선택이 같음) |
| **raw mean** | 평균 | 스텝별 \|Taylor\|의 평균 (`importance_stepvlm_v1`) | I_CoC | 없음 (원래 크기) | `rawabsavg_u40_l0keep_v2` (**새 이름**) |
| **raw max** | max | 스텝별 \|Taylor\|의 평균 | I_CoC | 없음 | `rawabsmean_u40_l0keep_v2` (기존 이름) |

- **스텝별 FM 중요도의 출처:** `importance_stepvlm_v1`은 `importance_v2`와 같은 calib_100, 모델 리비전 `7aba829`, FM 10스텝, seed 42로 쟀다(config 확인). FM 점수의 집계 방식만 다르다.
- **비교 구조:**
  - rank-mean ↔ MoOP는 연산자(평균 vs max)만 다르다.
  - raw max ↔ raw mean도 연산자만 다르다.
  - raw max ↔ MoOP는 정규화(원래 크기 vs 순위)와 FM 집계가 함께 다르다. 이 점은 표에 적는다.
- **마지막 층(35번):** I_traj가 정확히 0이다(그 출력이 expert가 읽는 캐시에 닿지 않는다).
  - 순위 규칙(MoOP, rank-mean)에서는 0의 순위가 인덱스 순서가 된다. MoOP와 똑같이 두려고 보정하지 않는다.
  - raw 규칙에서는 I_CoC가 그대로 정한다.

## 3. 구현

- `make_slim.py`에 새 이름 `rawabsavg`를 추가한다.
  - FM 쪽은 `trajabs`(이미 있음: `step_importance_vlm.npz`의 스텝별 |grad| 평균), CoC 쪽은 `coc`
  - 연산자 `np.add`(평균과 선택이 같음), 정규화 없음(raw)
  - `parts`·`op`·`rank` 세 표에 한 줄씩 넣는다.
- 레시피는 이 서버의 빈 카드 1장에서 `--importance importance_v2 --no-state`로 만든다. 3개에 약 10분이다.
- 게이트:
  - 제거량이 MoOP l0keep과 같아야 한다(2,657,447,936).
  - 0번 층 Q 32개가 유지돼야 한다.
  - expert가 그대로여야 한다.
  - `dual_u40_l0keep_v2`를 다시 만들어 출하본과 유닛 단위로 같아야 한다(빌드 경로 확인).
  - MoOP l0keep과의 유닛 겹침 비율(Q head·MLP)을 기록한다. 거의 같으면 결과 차이도 작을 것이라 해석에 쓴다.

## 4. 평가

- A100(KISTI): test_2000(4샤드), OOD-Val262(2샤드), LingoQA 500문항. arm 3개 × 7작업 = 21작업이다.
  - 프로토콜은 `run_baseline.py --no-tf`, k=8, 클립별 seed 42, 1.6/3.2/6.4 s minADE@6·minFDE@6이다.
- 비교: MoOP l0keep 대비 짝지은 Δ(부트스트랩 200,000회). FM-only·CE-only도 같은 표에 둔다.
- 큐 `queue_comb`, 홀더 2장·4장 각 1개.
- 비용: 약 15 A100-시간. 벽시계는 NEURON 대기에 달려 있고, 홀더가 시작한 뒤 약 3시간이다.

## 5. 산출물·git

- 리포트: `cvpr27/reports/<날짜>_combination-rule-ablation.md`. 논문 ablation 표 형식(절댓값 우선, Δ·CI 열)으로 쓴다.
- 기존 기준 ablation 라인의 후속이므로 main에 바로 커밋한다.
