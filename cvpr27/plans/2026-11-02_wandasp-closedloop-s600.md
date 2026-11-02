# Wanda-sp 기준선 빌드 + s600 폐루프 (2026-11-02)

## 0. 목적

- 구조적 Wanda(Wanda-sp, FLAP 논문의 기준선)를 MoOP와 같은 제거량으로 만들고, 같은 600씬 폐루프에서 짝지어 잰다.
  LLM-Pruner·Tyr에 이어 세 번째 외부 기준선이다.
- 기존 `wanda_u40_v2`(2026-08)는 우리 식 적응판이다. 열 점수를 L2로 집계했고, 무압축 모델에서 통계를 한 번에 쟀다.
  공식 Wanda-sp 절차와는 이 두 가지가 다르다.

## 1. 공식 절차 (CASIA-IVA-Lab/FLAP `lib/prune.py`, `prune_wanda_sp`, 2026-11-02 확인)

- 점수: `W_metric = |W| · sqrt(scaler_row)`. o_proj는 출력 행 평균 후 head(128열)마다 합하고, down_proj는 출력 행 평균이다.
- 모든 층을 같은 비율로 자르고, bias 보정은 없다.
- **층별 순차.** 층 L을 자른 뒤 그 층을 다시 돌려 다음 층의 입력을 만든다. 그래서 층 L의 통계는 0…L−1번 층을 자른 모델에서 잰다.
- 보정 데이터: C4 128샘플.

## 2. 사용자 결정 (2026-11-02)

| 항목 | 결정 |
|---|---|
| 점수 계산 | 공식대로 층별 순차 |
| 층 처리 | 공식대로 0번 층을 포함한 모든 층 균일 |
| 보정 데이터 | calib_100 (MoOP·LLM-Pruner·Tyr와 같음). fused 프롬프트의 prefill 토큰 전체를 쓰고 라벨은 쓰지 않는다 |

## 3. 구현

- `cvpr27/code/run_wanda_sp.py`
  - 통계는 `run_wanda.py`와 같은 WrappedGPT 러닝 평균(fp32)이다.
  - 층 L마다 100클립을 0…L번 층까지만 forward한다(L번 층 뒤에서 멈춤).
  - 이미 고른 0…L−1번 층 유닛은 o_proj·down_proj 입력을 0으로 만들어 정확히 제거한 상태로 둔다.
  - 층마다 MoOP와 같은 수를 자른다: Q head 13/32, MLP 4,898/12,288. 총 2,657,452,032개.
  - k/v와 expert는 자르지 않는다. expert가 VLM의 KV 8개를 읽기 때문이다.
  - smoke(2클립): 0.7분, 유지 Q 684 / MLP 266,040으로 MoOP와 같다. 100클립이면 약 35분이다.
- 빌드: `make_slim.py --config wanda_u40_v2 --wanda cvpr27/wandasp_v1`. 가중치를 포함하고, 결과는 `outputs/cvpr27/slim_wandasp_u40`다.
- **G0.** 아래를 모두 확인한다. 하나라도 틀리면 중단한다.
  - 빌드 유닛이 순차 수집 때 고른 유닛과 층마다 같다
  - 제거량이 2,657,452,032개다
  - expert meta가 MoOP와 같다
  - 0번 층에서 Q head 13개를 잘랐다

## 4. 폐루프

- `cvpr27/code/run_wandasp_s600.sh`. 수집 → 빌드 → G0 → 드라이버 하드링크(`slim_wandasp_u40`) → smoke → 게이트 → 본 런을 자동으로 잇는다.
- **카드:** GPU 0–7, `scenes_smoke8.csv` / `scenes_d600.csv`. 각 씬은 dense 기준선의 카드에서 돈다.
- **이름:** TAG `wsp`. 결과는 `outputs/cvpr27/alpasim_wandasp_s600/`다.
- **시간:** 일반 arm이면 약 7.5시간이다.
  - 8월 Wanda(L2판)는 폐루프에서 CoC가 무너지면서(퇴화 46–56%) 디코딩이 길어졌다. 같은 일이 생기면 더 걸린다.
- **분석 (자동).**

  | 비교 대상 | 600씬 | 같은 카드 |
  |---|---|---|
  | MoOP | O | 600씬 전부 |
  | MoOP l0keep | O | 471씬 |
  | dense | O | 600씬 전부 |
  | lp_m24 | O | 600씬 전부 |
  | tyr_off | O | 517씬 |

- 리포트: `cvpr27/reports/<날짜>_wandasp-closedloop-s600.md`

## 5. git

- 기준선 비교 라인의 후속이라 main에 바로 커밋한다.
