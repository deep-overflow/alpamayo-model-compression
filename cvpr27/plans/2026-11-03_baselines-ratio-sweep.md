# 외부 기준선 비율 실험: LLM-Pruner·Tyr·Wanda-sp를 VLM 35/45/50%로 (A100)

날짜: 2026-11-03. 상태: **승인됨** (사용자, 2026-11-03: "일단 wanda-sp, llmpruner, ce-only, fm-only, tyr 이렇게 진행해줘").
다섯 방법 × u35/u45/u50 = 15개 arm. 단일 기준 대조 CE-only·FM-only(l0keep)를 추가했다(§3.4).

## 0. 목적

- 지금까지 외부 기준선 세 개는 MoOP u40과 같은 예산(제거 2,657,452,032, 모델의 23.99%) 한 점에서만 쟀다.
  MoOP l0keep은 u35/u40/u45/u50 네 점이 A100에 있다(`2026-11-01_l0keep-ratio-sweep.md`).
- 기준선도 MoOP u35/u45/u50과 같은 예산으로 만들어 비율 곡선 위에서 비교한다.
  - 순위가 비율에 따라 바뀌는가
  - 어느 방법이 먼저 무너지는가
- 모든 arm은 지금처럼 **VLM 텍스트 타워만** 자른다. expert와 k/v는 그대로 둔다(확인: MoOP·lp_m24·tyr_off·Wanda-sp 모두 expert 576/576 head 유지).

## 1. 사용자 결정 (2026-11-03)

- 외부 기준선(LLM-Pruner, Tyr, Wanda-sp)을 VLM 35%, 45%, 50%로 진행한다("외부 기준선에 대해서 진행해줘").

## 2. 비율별 예산 (MoOP `dual_u{N}_v2` = 모든 층 같은 비율)

| 비율 | 층당 Q head 절단 | 층당 MLP 절단 | 제거 파라미터 | 모델 대비 | A100 기준 arm |
|---|---|---|---|---|---|
| u35 | 11 / 32 | 4,301 / 12,288 | 2,317,860,864 | 20.92% | `l0keep_u35` |
| u40 (기존) | 13 | 4,898 | 2,657,452,032 | 23.99% | `moop`, `moop_l0keep` |
| u45 | 14 | 5,530 | 2,974,777,344 | 26.85% | `l0keep_u45` |
| u50 | 16 | 6,144 | 3,321,888,768 | 29.98% | `l0keep_u50` |

l0keep 레시피는 0번 층 Q head를 남기고 MLP 채널로 갚는다. 그래서 같은 예산에서 4,096 파라미터 이하로만 모자란다.

## 3. 기준선별 빌드 (u40 때와 같은 규칙)

### 3.1 LLM-Pruner (`lp_u35`, `lp_u45`, `lp_u50`)
- `lp_m24`와 같은 방식이다(`cvpr27/plans/2026-10-30_llm-pruner-budget-matched.md`).
  - 원본 층 범위 4–33번(30개 층), Q head 단위
  - 중요도는 soowon `lp_importance_v1`의 `coc_param_first`를 그대로 쓰고 새로 재지 않는다.
- 예산 맞추기:
  - 원본 반올림 규칙으로 MoOP 예산에 가장 가까운 비율 r을 고르고, 그 r의 층당 head 수를 쓴다.
  - MLP 수는 예산에 정확히 맞도록 푼다. 나머지 채널은 앞쪽 층에 +1씩 배분한다(`--units qhead:H,mlp:M --mlp-extra-layers E --expect-removed B`).
  - 대략적인 값:

    | 비율 | r | 층당 head | 층당 MLP |
    |---|---|---|---|
    | u35 | 약 0.42 | 13 | 약 5,178 |
    | u45 | 약 0.54 | 17 | 약 6,619 |
    | u50 | 약 0.60 | 19 | 약 7,390 |

    정확한 값은 `lp_prune.py`의 규칙으로 계산해 빌드 로그와 리포트에 남긴다.
- 빌드: 로컬 빈 카드 1장, arm당 약 5분, `--no-state`. 내장 스모크를 거친 레시피를 NEURON으로 보낸다.
- 게이트:
  - 총 제거량이 MoOP u{N}과 정확히 같아야 한다(`--expect-removed`).
  - 0–3·34–35번 층과 expert는 그대로여야 한다.
  - 비율이 커질수록 남는 유닛은 작은 비율의 부분집합이어야 한다(같은 점수에서 더 지움). u40의 `lp_m24`와도 포함 관계를 확인한다.

### 3.2 Tyr-the-Pruner, 공식 절차 (`tyr_u35`, `tyr_u45`, `tyr_u50`)
- `tyr_off`와 같은 절차다.
  - supernet·탐색 각 4회, 수준 격자, 논문 식 5, top-8,192 텍스트 위치 KL, calib_100
  - 헤시안은 토큰 수 그대로, teacher 재사용
- 다른 것은 R=1 출발점 하나다.
  - supernet은 유지 수를 `int(n·(1−s))`로 정한다. 이 규칙에서는 u35·u45의 head 수와 MLP 수를 동시에 만드는
    s 값 하나가 존재하지 않는다(u40의 0.39856은 우연히 겹쳤다).
  - 그래서 R=1에만 유형별 출발점 `--sparsity-mha`, `--sparsity-mlp`를 준다. 값은 `(절단 수 − 0.5) / n`이다.
    R≥2는 원래 코드가 층·유형별 비율을 따로 들고 간다.

    | 비율 | `--sparsity-mha` | `--sparsity-mlp` | 출발점 절단 (head / MLP) | 제거 파라미터 |
    |---|---|---|---|---|
    | u35 | 0.328125 | 0.3499755859375 | 11 / 4,301 | 2,317,860,864 |
    | u45 | 0.421875 | 0.4499918619791667 | 14 / 5,530 | 2,974,777,344 |
    | u50 | 0.484375 | 0.4999593098958333 | 16 / 6,144 | 3,321,888,768 |

    같은 식에 u40을 넣으면 MLP 쪽이 0.39856으로 기존 값과 같다.
  - 탐색은 유형별 총량을 보존하므로 최종 예산이 MoOP u{N}과 정확히 같다(`TYR_BUDGET` assert).
  - make_slim 설정 이름은 그대로 `tyr_u40_r`다. 이 이름은 분기만 고르고, 예산은 supernet이 정한다.
- `build_tyr_off.sbatch` 수정: 예산 assert(지금 2,657,452,032 고정)를 `TYR_BUDGET` 환경 변수로 바꾼다.
- 빌드 위치:
  - u35 → u45: KISTI A100, 4장 잡을 순서대로(afterany) 돌린다. 각각 약 5.5시간·22 A100-시간 걸린다.
  - u50: cvlab21 Blackwell(`build_tyr_off_local.sh`, 빈 카드만)에서 돌리고 약 6–7시간 걸린다. 레시피는 NEURON으로 보낸다.
  - 이렇게 나누는 이유: 계정의 동시 실행 한도가 잡 2개라서, 다른 한 슬롯은 평가 홀더가 바로 써야 한다.
- 게이트: tyr_off와 같다.
  - 0번 블록 헤시안이 비트 동일
  - 라운드마다 유지 총량 불변
  - 빈 층 없음
  - 스모크
  - 최종 구성의 가중치 크기 최대 비율도 기록한다(참고: tyr_off 3.45×).

### 3.3 Wanda-sp, FLAP 공식 (`wandasp_u35`, `wandasp_u45`, `wandasp_u50`)
- `run_wanda_sp.py`(층별 순차, `|W|·sqrt(scaler_row)`, calib_100 prefill, 0번 층 포함 모든 층 균일)와 같다.
- 층별 순차라 L층 통계가 앞 층의 절단에 따라 달라진다. 그래서 비율마다 통계를 다시 모은다.
  - `run_wanda_sp.py`의 고정 상수 `CUT_Q, CUT_MLP = 13, 4898`을 인자 `--cut-q`, `--cut-mlp`로 바꾼다. 기본값은 그대로다.
- 빌드:
  - 로컬 빈 카드 3장에서 동시에, 각 약 35분.
  - `make_slim --config wanda_u{N}_v2 --wanda cvpr27/wandasp_u{N} --importance importance_v2 --no-state`
- 게이트(u40과 같음):
  - 빌드 유닛이 순차 수집 때 고른 유닛과 층마다 같아야 한다.
  - 제거량이 MoOP u{N}과 같아야 한다.
  - expert가 그대로여야 한다.
  - 0번 층 Q head가 11/14/16개 잘려야 한다.

### 3.4 단일 기준 대조: FM-only·CE-only (l0keep) (`fm_only_u35/45/50`, `ce_only_u35/45/50`)
- u40 arm(`fm_only_l0keep` = `trajlastcoc_u40_l0keep_v2`, `ce_only_l0keep` = `coc_u40_l0keep_v2`)과 같은 레시피에 비율만 바꾼다.
  - `make_slim --config trajlastcoc_u{N}_l0keep_v2` / `coc_u{N}_l0keep_v2 --importance importance_v2 --no-state`
  - MoOP l0keep u{N}과 예산이 같다.
- 빌드: 로컬 빈 카드 1장, 6개 레시피에 약 30분. 레시피 파일만 NEURON으로 보낸다.
- 게이트:
  - u40을 다시 만들어 기존 `slim_trajlastcoc_u40_l0keep_v2` / `slim_coc_u40_l0keep_v2`와 유닛 단위로 같아야 한다.
  - 제거량이 MoOP l0keep u{N}과 같아야 한다.

## 4. 평가 (A100, l0keep 비율 실험과 같은 세트)

- 세트: test_2000(4샤드), OOD-Val262(2샤드), LingoQA 500문항(Lingo-Judge).
  - arm당 7작업, 15개 arm이면 105작업이다.
  - 프로토콜은 `run_baseline.py --no-tf`, k=8, 클립별 seed 42, 1.6/3.2/6.4 s minADE@6·minFDE@6이다.
- 큐: 새 큐 `queue_ratio`, `eval_a100.sbatch` 4장 홀더 2개. 레시피가 아직 없는 줄은 기존 대기 규칙대로 건너뛴다.
- 비교:
  - 각 비율에서 MoOP l0keep u{N} 대비 짝지은 Δ(부트스트랩 200,000회)
  - 같은 방법의 비율 간 곡선
  - u40 점(lp_m24, tyr_off, wandasp)도 곡선에 넣는다.
- 분석: `analyze_a100.py`에 arm·REFS를 추가하고, 비율별 곡선 그림(minADE@6, LingoQA)을 그린다.

## 5. 비용·일정 (대략)

| 항목 | 비용 |
|---|---|
| LLM-Pruner·Wanda-sp 빌드 | 로컬 GPU 약 1시간 |
| Tyr 빌드 | A100 약 44 A100-시간 + 로컬 Blackwell 약 6–7시간 |
| 평가 | 약 95 A100-시간(기준선 9개 약 65 + FM/CE 6개 약 30). 퇴화 CoC가 긴 Wanda-sp·LLM-Pruner는 arm당 2배쯤 |
| 벽시계 | 약 15–18시간. NEURON 실행 슬롯 2개 중 하나는 Tyr u35 → u45 빌드에 쓰고, 다른 하나는 평가 홀더가 바로 쓴다 |

## 6. 산출물·git

- 리포트: `cvpr27/reports/<날짜>_baselines-ratio-sweep.md`(사실/해석 분리, 절댓값 우선). 결과 모음과 README도 갱신한다.
- 후속 실험이므로 main에 바로 커밋한다.

## 7. 하지 않는 것

- **MoOP 출하 방식(0번 층 포함 절단) u35/u45/u50은 재지 않는다.**
  - MoOP 쪽 비율 점은 이미 A100에 있는 MoOP l0keep u35/u40/u45/u50을 쓴다(사용자: "MoOP는 이미 한거 아니야?").
  - CLAUDE.md도 dual 계열 arm을 `_l0keep`으로 만들라고 한다(0번 층 head 선택 = 붕괴·추출 분산).
  - u40에서만 출하 방식 MoOP가 함께 있다.

## 8. 추가: OOD-train1271 (사용자, 2026-11-05)

- 사용자 요청: "OOD train에 대해서 u35, u40, u45, u50을 테스트 진행해줘". 6개 방법 전부를 대상으로 한다.
- 새로 잴 arm은 20개다.
  - u35/u45/u50 × {MoOP l0keep, FM-only, CE-only, Tyr, LLM-Pruner, Wanda-sp} = 18개
  - u40의 FM-only, CE-only = 2개
  - u40의 MoOP l0keep, tyr_off, lp_m24, wandasp는 이미 OOD-train 행이 있다.
- 프로토콜은 OOD-Val과 같다(`--set ood --manifest ood_train --no-tf`, k=8, seed 42, 4샤드). 작업은 80개다.
- 큐 `queue_ratio_oodtrain`, 홀더 4장·2장 각 1개(10시간)다. 비용은 약 70 A100-시간이다.
- 비교: 같은 비율의 MoOP l0keep 대비 짝지은 Δ를 6개 지표(minADE@6·minFDE@6 × 1.6/3.2/6.4 s) 모두에 대해 낸다.
