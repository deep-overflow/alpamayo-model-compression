# LLM-Pruner 두 판(`lp_m24`, `lp_r50`) 폐루프: s600, 4장씩 동시 (2026-10-31)

## 0. 목적

- **출발점.** A100 개루프(`cvpr27/reports/2026-10-31_llm-pruner-budget-matched.md`) 결과:
  - `lp_m24`(MoOP와 같은 제거량 2,657,452,032개): test_2000 MoOP 대비 +0.409 m, LingoQA −13.2 pp
  - `lp_r50`(원래 설정, 24.99%): `lp_m24`보다 주행 +0.04 m, LingoQA −8.0 pp 더 나쁨
- **질문.**
  - 같은 예산의 LLM-Pruner 기준선이 폐루프에서 MoOP보다 얼마나 나쁜가.
  - 예산 1%p 차이(`lp_r50` − `lp_m24`)가 폐루프에서 보이는가.
- 기존 폐루프 `lp_r50` 수치는 비교에 바로 쓸 수 없다. 패치 전 하네스에서 150씬으로만 돌린 것이다(soowon `runs_root`).
- **사용자 지시 (2026-10-31).**
  - 1–4번 층 MLP 큐가 끝난 뒤, **사용자가 시작하라고 할 때** 두 판을 GPU 4장씩 동시에 돌린다.
  - 생성 상한은 다른 arm과 같은 128토큰으로 둔다.

## 1. 체크포인트

| arm | 드라이버 폴더 | 구성 | 제거 |
|---|---|---|---|
| `lp_m24` | `slim_lp_m24` | 4–33번 층에서 층마다 Q head 15개, MLP 5,928개(4–27번 층은 +1) | 2,657,452,032 (23.99%) |
| `lp_r50` | `lp_r50` | 4–33번 층에서 층마다 Q head 16개, MLP 6,144개 | 2,768,240,640 (24.99%) |

- **`lp_m24`.**
  - A100 세션의 빌드 옵션에서 `--no-state`만 빼고 가중치 포함으로 다시 빌드했다(2026-10-31, GPU 3). 산출물은 `outputs/cvpr27/slim_lp_m24_state`다.
    ```
    experiments/llm_pruner/lp_prune.py --importance cvpr27/lp_importance_v1 --arm coc_param_first \
      --units qhead:15,mlp:5928 --mlp-extra-layers 24 --expect-removed 2657452032
    ```
  - 선택 전용 빌드 `outputs/cvpr27/slim_lp_m24`와 `vlm`·`expert`·`kvonly_layers`·`params`가 모두 같다.
  - 드라이버 폴더에 하드링크했다.
- **`lp_r50`.** 기존 드라이버 폴더(soowon, 8월)를 그대로 쓴다. A100 세션이 다시 빌드한 `outputs/cvpr27/slim_lp_r50`과 유닛 단위로 같다.
- **공통.** 0–3·34–35번 층, expert, KV는 손대지 않았다.

## 2. 실행

- **스크립트:** `cvpr27/code/run_lp_pair_s600.sh`. 사용자가 지시하면 시작한다.
- **카드:** 판마다 Blackwell 2장 + Ada 2장이다.

  | arm | GPU | TAG |
  |---|---|---|
  | `lp_m24` | 0, 1, 4, 5 | `p24` |
  | `lp_r50` | 2, 3, 6, 7 | `r50` |

- **씬 배정:** `outputs/cvpr27/lp_s600_scenes/scenes_{smoke8,d600}_4c.csv`
  - `shard`는 각 판의 GPU 목록(B, B, A, A)에서의 위치다.
  - 각 씬은 dense 기준선과 같은 카드 종류에 놓인다. 카드당 150씬이다.
  - 그래서 두 판 모두 dense·출하본 MoOP와 600씬 전부 카드 종류가 같다.
  - 한 판을 Blackwell에만 두면 카드 종류와 arm이 겹치므로 피했다.
- **판마다의 순서:**
  1. smoke 8씬(카드당 2씬)을 돌리고 게이트 G1·G2를 본다.
  2. 본 런 592씬을 돌린다.
  3. 짝 분석을 낸다.
  - 게이트가 실패하면 그 판만 멈춘다.
- **설정:** 패치된 alpasim, `pair_a15`, `DRIVER_OMP_THREADS=8`, CoC 생성 상한 128토큰. 다른 arm과 같다.
- **시간:** 일반 arm을 4장으로 돌리면 약 14–15시간이다.
  - LLM-Pruner는 CoC가 평균 2배 길다(개루프 생성 34.5 대 17.3토큰, 상한 128 기준). 그래서 드라이버가 20–30% 느려질 것으로 보고, 두 판 모두 **약 17–19시간**으로 잡는다.

## 3. 분석 (판마다 자동, 둘 다 끝나면 교차 비교)

| 비교 | 600씬 | 같은 카드 |
|---|---|---|
| − 출하본 MoOP (주 비교) | O | 600씬 전부 |
| − MoOP l0keep | O | 471씬 (`outputs/alpasim_l0keep_s600/card_mismatch.csv` 제외) |
| − dense | O | 600씬 전부 |
| `lp_r50` − `lp_m24` (예산 1%p) | O | 600씬 전부 |

- 지표: score, progress, 과실 충돌, offroad, wrong lane, plan deviation, open loop collision, 충돌 유형, CoC 퇴화율, 중단 rollout
- 리포트: `cvpr27/reports/<날짜>_lp-closedloop-s600.md`. 개루프 A100 결과와 나란히 놓는다.

## 4. git

- 기존 LLM-Pruner 기준선 라인의 후속이라 main에 바로 커밋한다.

## 5. 변경 (2026-10-31, 사용자 요청): Ada 절반 먼저, Blackwell 절반은 0–3번이 빈 뒤

- **바꾼 이유.** GPU 0–3은 다른 작업이 11-01 03시 KST쯤까지 쓴다. 사용자가 "일단 4–7번만 사용해서 진행"하라고 했다.
- **방식.** 판, 카드, 씬별 카드 종류 배정은 §2와 같다. Blackwell 쪽 절반만 늦게 시작한다.
  - lp_m24: Ada 4, 5에서 바로 시작하고, Blackwell 0, 1은 빈 뒤에 시작한다.
  - lp_r50: Ada 6, 7에서 바로 시작하고, Blackwell 2, 3은 빈 뒤에 시작한다.
- **판마다의 순서.**
  1. Ada에서 smoke 4씬(기준선 Ada 씬, 카드당 2씬)을 돌리고 게이트 G1·G2를 본다.
  2. Ada 본 런 296씬(`main_ada296.csv`)을 돌린다.
  3. 같은 시각부터 Blackwell 카드가 비기를 기다렸다가 Blackwell 본 런 300씬(`main_bw300.csv`)을 돌린다.
  4. 셋을 600씬으로 병합하고 짝 분석을 낸다.
- **스크립트:** `cvpr27/code/run_lp_pair_split_s600.sh`. 결과는 `outputs/cvpr27/alpasim_lp_{m24,r50}_s600/`, 로그는 `outputs/cvpr27/lp_s600.log`.
- **시작:** 2026-10-31 21:0x KST.
- **시간:** 카드당 약 150씬이다.
  - Ada 절반은 11-01 오후에 끝난다.
  - Blackwell 절반은 03시에 시작하면 11-01 저녁(약 20시 KST)에 끝난다.

## 6. 사고와 조치 (2026-10-31 21:30 KST): 포트 충돌

- **무슨 일.** 두 판의 Ada 본 런이 거의 동시에 떴다. 그 순간 lp_r50 shard 1과 lp_m24 shard 0의 wizard가 1초 차이로 같은 호스트 포트(6006)를 골랐다. lp_r50 shard 1의 runtime이 "port is already allocated"로 뜨지 못했다.
  - 나머지 세 shard는 정상이다.
  - 실패한 shard의 컨테이너 일부(렌더러, 드라이버, physics)가 GPU 7에 남아 있어서 지웠다.
- **조치 1.** 그 shard의 148씬(`main_ada296_sh1_r50c.csv`)을 prefix `r50c_`로 GPU 7에서 다시 돌린다. 로그는 `outputs/cvpr27/alpasim_lp_r50_s600/r50c.log`다.
- **조치 2.** Blackwell 절반에서 같은 충돌이 다시 나지 않게 한다.
  - pair 스크립트의 lp_r50 Blackwell 대기 서브셸을 종료했다.
  - 그 일은 `cvpr27/code/finish_lp_r50_s600.sh`가 맡는다. GPU 2, 3이 빈 뒤 5분을 더 기다려 lp_m24보다 늦게 띄운다.
  - 이 스크립트는 lp_r50의 600씬 병합(smoke + r50a sh0 + r50c + r50b), 짝 분석, lp_r50 − lp_m24까지 이어서 한다.
  - pair 스크립트의 lp_r50 병합은 r50a sh1이 비어서 실패한다. 이건 예상한 동작이다.
- lp_m24는 pair 스크립트가 그대로 끝까지 처리한다.
