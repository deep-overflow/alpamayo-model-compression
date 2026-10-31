# LLM-Pruner 제거량 맞춤판(`lp_m24`) 폐루프: s600 (2026-10-31)

## 0. 목적

- **출발점.** A100 개루프(`cvpr27/reports/2026-10-31_llm-pruner-budget-matched.md`)에서 `lp_m24`는 MoOP와 같은 제거량(2,657,452,032개)인데도:
  - test_2000 1.365, MoOP 대비 +0.409 m
  - OOD-Val +0.456 m
  - LingoQA 53.4%, −13.2 pp
- **질문.** 같은 예산의 LLM-Pruner 기준선이 폐루프에서도 MoOP보다 나쁜가, 얼마나 나쁜가.
  - 기존 폐루프 LLM-Pruner 수치(`lp_r50`)는 다음 이유로 이 비교에 바로 쓸 수 없다.
    - 제거량이 24.99%로 MoOP보다 많다.
    - 패치 전 하네스에서 150씬으로만 돌렸다(soowon `runs_root`).
- **사용자 지시 (2026-10-31).** 1–4번 층 MLP 큐(l14m95/90/80)가 끝나면 `lp_m24`로 s600 폐루프를 돌린다.

## 1. 체크포인트

- A100 세션의 빌드 옵션을 그대로 쓰고, `--no-state`만 빼서 가중치를 포함한다.
  ```
  experiments/llm_pruner/lp_prune.py --importance cvpr27/lp_importance_v1 --arm coc_param_first \
    --units qhead:15,mlp:5928 --mlp-extra-layers 24 --expect-removed 2657452032 \
    --out outputs/cvpr27/slim_lp_m24_state --gpu 0
  ```
- **구성.**
  - 4–33번 층(30개)에서 층마다 Q head 15개와 MLP 5,928개를 지운다. 4–27번 층은 MLP를 하나 더 지운다.
  - 0–3번 층, 34–35번 층, expert, KV는 그대로 둔다.
- **검증.**
  - 선택 전용 빌드 `outputs/cvpr27/slim_lp_m24`와 `vlm`·`expert`·`kvonly_layers`·`params`가 같아야 한다. 다르면 중단한다.
  - 통과하면 드라이버 폴더 `slim_lp_m24`에 하드링크한다.

## 2. 실행

- **스크립트:** `cvpr27/code/run_lp_m24_s600.sh`. 다음 순서로 돈다.
  1. l14m 큐가 "queue finished"를 남기고 GPU 8장이 모두 빌 때까지 기다린다.
  2. 빌드하고 유닛을 대조한 뒤 하드링크한다.
  3. smoke(8씬)를 돌리고 게이트 G1·G2를 본다.
  4. 본 런(592씬)을 돌린다.
  5. 짝 분석을 낸다.
- **설정과 카드:** 기준선과 같다. 패치된 alpasim, `pair_a15`, `DRIVER_OMP_THREADS=8`이고, 씬마다 dense 기준선의 카드를 쓴다(`scenes_{smoke8,d600}.csv`).
- **이름:** TAG `p24`. 런은 `alpasim-runs/p24_merged_slim_lp_m24`, 결과는 `outputs/cvpr27/alpasim_lp_m24_s600/`.
- **시간:** 약 7.5시간 이상 걸린다.
  - 개루프 CoC 퇴화율이 17%로 높아서, 디코딩이 길어지면 더 걸릴 수 있다. wanda는 생성이 붕괴해 shard가 14.3시간 걸렸다.

## 3. 분석

| 비교 대상 | 600씬 | 같은 카드 |
|---|---|---|
| 출하본 MoOP (주 비교, 논문 기준선 비교) | O | 600씬 전부 같은 카드 |
| MoOP l0keep | O | 471씬 (`outputs/alpasim_l0keep_s600/card_mismatch.csv` 제외) |
| dense | O | 600씬 전부 같은 카드 |

- 지표: score, progress, 과실 충돌, offroad, wrong lane, plan deviation, open loop collision, 충돌 유형, CoC 퇴화율, 중단 rollout
- 리포트: `cvpr27/reports/<날짜>_lp-m24-closedloop-s600.md`. 개루프 A100 결과와 나란히 놓는다.

## 4. git

- 기존 LLM-Pruner 기준선 라인의 후속이라 main에 바로 커밋한다.
