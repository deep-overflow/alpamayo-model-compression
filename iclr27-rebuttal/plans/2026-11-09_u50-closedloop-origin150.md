# 계획: u50(29.98% 제거) MoOP · Tyr · LLM-Pruner 폐루프 — Origin150 × 2 rollout (cvlab21)

2026-11-09 · 상태: **승인 대기** · 결과는 `iclr27-rebuttal/`에 정리

**변경 이력(11-09, 사용자):**
- 씬 세트를 s600(600씬 × 1)에서 **Origin150 × 2 rollout(arm당 300 rollout)**으로 바꿨다.
- MoOP arm은 l0keep판이 아니라 **기본 MoOP u50(`dual_u50_v2`)**이다. MoOP l0keep u50은 나중에 따로 한다.

## 목적

- **배경:**
  - u40(23.99%) 폐루프에서 MoOP는 논문 Tyr 행·LLM-Pruner와 scene score로 구별되지 않았다.
    - Origin150: Tyr +0.042 [+0.000, +0.085]*, LLM-Pruner +0.018 [−0.029, +0.065]
    - s600: Tyr −0.017, LLM-Pruner +0.011, 둘 다 n.s.
  - u50 개루프(`cvpr27/reports/2026-11-05_baselines-ratio-sweep.md`, A100)에서는 MoOP 계열이 두 기준선보다 확실히 낫다.
    - Tyr: test_2000 +0.044*, OOD-Val +0.224*
    - LLM-Pruner: +1.461*, CoC 퇴화 74.5%
- **가설:**
  - H1: u50에서는 폐루프에서도 MoOP u50의 scene score가 Tyr u50보다 높다. 씬 클러스터 짝 부트스트랩 CI가 0을 포함하지 않는다고 본다.
  - H2: 같은 조건에서 MoOP u50이 LLM-Pruner u50보다도 높다.
- **반증 조건:** CI가 0을 포함하면 "u50에서도 폐루프 score 차이는 보이지 않는다"로 보고한다.
- **분해능:** Origin150 × 2의 짝 차이 CI 반폭은 u40에서 ±0.043–0.047이었다(`2026-11-07_closedloop-ci-origin150-hard100.md`). 약 0.05보다 작은 차이는 잡기 어렵다.

## arm (사용자 범위: MoOP, Tyr, LLM-Pruner)

| arm | 레시피 | 제거 파라미터 | 상태 |
|---|---|---|---|
| MoOP u50 | `dual_u50_v2` (`make_slim.py`, calib_100 중요도, 0번 층 포함 층당 50% 균일) | 3,321,888,768 (29.98%) | **처음 빌드한다.** 개루프 수치 없음 |
| Tyr u50 | `outputs/cvpr27/slim_tyr_u50` (공식 절차, KL 적합도) | 3,321,888,768 | `slim_state.pt` 15 GB 있음. 비율 실험의 A100 개루프 행과 같은 빌드 |
| LLM-Pruner u50 | `outputs/cvpr27/slim_lp_u50` (Q head 단위, 4–33번 층, `coc_param_first`) | 3,321,888,768 | 레시피만 있음 → `slim_state.pt` 생성. 비율 실험과 같은 빌드 |

- Tyr u50과 LLM-Pruner u50의 예산은 원래 `dual_u50_v2`의 제거량에 맞춰 만들었다. 그래서 세 arm의 제거량이 정확히 같다.
- 대조군 dense는 `hofix_merged_baseline`이다(패치된 하네스, Origin150 × 2, 300 rollout, scene score 0.780).

**주의할 교란 요인:**
- **MoOP u50은 0번 층 Q head도 50% 자른다.**
  - 0번 층에서 어느 head를 자르느냐가 cruise 붕괴를 가른다는 것이 확인돼 있다(CLAUDE.md, 17개 arm SD 0.260 → 0.018).
  - u40 출하본(`dual_u40_v2`)은 그 추첨에서 운이 좋았다. u50이 어떨지는 모른다.
  - 개루프 수치가 없으므로, 폐루프 결과가 나쁘면 0번 층 때문인지 비율 때문인지 이 실험만으로는 가를 수 없다. 그 구분은 나중의 l0keep u50 런이 준다.
- **Tyr u50은 공식 절차판이다.** u40 `tyr_off`와 같은 계열이고, 논문 Tyr 행 `tyr_u40_r`(간이 포트)과는 다르다.
  - u40 s600에서 공식판은 MoOP l0keep보다 −0.035*였고, 포트판은 +0.009(n.s.)였다.
- **LLM-Pruner u50은 예산을 맞춘 판이다(`lp_m24` 계열).** 논문 행 `lp_r50`과는 다르다. u40 s600에서 두 판의 차이는 −0.011(n.s.)이었다.
- **하네스가 다르다.** 논문 표 2의 Origin150 행(`m2601_merged_*`)은 9월 런으로, 인계 감속 수정 전 하네스에서 돌았다.
  - 새 arm은 패치된 하네스에서만 돌린다(10-26 이후 규칙).
  - 그래서 u50 행은 표 2의 u40 행과 같은 표에 바로 놓을 수 없다. 패치 전 dense는 0.750, 패치 후는 0.780이다.
  - u40 행과 나란히 놓으려면 u40 MoOP·Tyr·LLM-Pruner를 패치된 하네스의 Origin150 × 2로 다시 돌려야 한다(선택 사항, 아래 질문 3).

## 절차

0. **준비.**
   - `make_slim.py --config dual_u50_v2`로 MoOP u50을 빌드하고 `slim_state.pt`를 쓴다.
   - LLM-Pruner u50은 `slim_lib.load_slim` → `save_slim(write_state=True)`로 `slim_state.pt`를 만든다.
   - 둘 다 CPU에서 할 수 있으면 CPU로 한다. GPU가 필요하면 빈 카드가 날 때 한다.
   - **G0:**
     - 같은 코드로 `dual_u40_v2`를 다시 빌드해 기존 `outputs/slim_dual_u40_v2`와 유닛 단위로 같은지 본다.
     - `dual_u50_v2`의 제거 파라미터가 3,321,888,768로 Tyr u50·LLM-Pruner u50과 같은지 본다.
     - 세 드라이버 모두 AlpaSim 로더(`alpamayo1_5_slim.load_slim`)로 읽혀야 한다.
   - 세 체크포인트를 `alpasim/data/drivers/`에 하드링크한다(`cp -al`).
1. **대기.** 지금 Ada 4–7을 쓰는 ada-fill 런처가 끝나고 Ada 4–7이 모두 비어야 한다(10 MiB 이하). Blackwell은 쓰지 않는다.
2. **arm마다 진행.** `launch_alpasim_shards.sh <cfg> 150 2 "4 5 6 7"`, PREFIX `u50o_`로 띄운다.
   - dense `hofix`와 같은 4샤드 분할(38/38/37/37씬)이 되고, 각 씬이 dense와 같은 Ada 카드에서 돈다.
   - 설정(pair_a15, 카드당 샤드 1개, ASL 이미지 생략)도 `hofix`와 같다.
   - **smoke:** 본 런 전에 카드마다 1씬 × 1 rollout을 돌린다. 다음을 확인한다.
     - rollout이 끝까지 도는지
     - `Rollout FAILED`·재시도가 없는지
     - CoC 텍스트가 나오는지
     - score가 dense 범위인지
   - 순서는 MoOP → Tyr → LLM-Pruner다. LLM-Pruner는 CoC 퇴화가 커서 가장 느릴 것이라 마지막이다.
   - 런마다 `merge_alpasim_shards.py`로 병합하고, 우리 docker 컨테이너와 네트워크를 정리한다.
   - 중단 rollout이 있으면 개수를 기록한다. 표 2 규약대로 score 0, 사건 없음으로 처리하고 /300으로 나눈다.
3. **분석.** `cvpr27/code/closedloop_ci.py` 방식으로 씬 클러스터 부트스트랩을 한다. 0 근처 칸은 20만 회 × 시드 3으로 수렴시킨다.
   - 주 비교: Tyr u50 − MoOP u50, LLM-Pruner u50 − MoOP u50
   - 보조: 각 u50 − dense(`hofix`)
   - 지표:
     - scene score와 함께 progress, 과실 충돌, 전체 충돌, offroad, plan deviation, CoC 퇴화, 중단 rollout 수
     - AlpaSim score(km/사건)
     - score 분산(씬 간·씬 내)
   - score만 단독으로 보고하지 않는다.
4. **보고.**
   - `iclr27-rebuttal/reports/2026-11-1x_u50-closedloop-origin150.md`
   - 결과 `outputs/iclr27-rebuttal/u50_origin150/`
   - 런 `alpasim-runs/u50o_*`
   - README 갱신, main 커밋

## 비용과 일정

- 준비: 빌드 2개, 약 30–60분. 디스크는 약 31 GB가 늘어난다(현재 1.2 TB 여유).
- GPU(Ada 4–7):
  - arm당 약 8시간이다. `hofix` dense는 7.5–7.8시간, 9월 가지치기 arm들은 7.3–7.5시간이었다.
  - LLM-Pruner는 CoC 퇴화 때문에 12–20시간으로 본다.
  - 합계 약 28–36시간이다.
- 시작은 ada-fill 작업이 끝난 뒤다.

## 나중에 할 것 (이번 범위 아님)

- MoOP l0keep u50(`dual_u50_l0keep_v2`)을 같은 설정으로 돌린다(사용자, 11-09).
  MoOP u50과 짝지으면 u50에서 0번 층 보존의 효과가 나온다.

## 사용자에게 확인할 것

1. **Tyr u50 판.**
   - 기본은 이미 있는 공식 절차판 `slim_tyr_u50`이다.
   - 논문 Tyr 행(간이 포트)과 맞추려면 포트판 u50을 새로 빌드해야 한다. 빌드 시간과 개루프 평가가 추가로 든다.
2. **MoOP u50 개루프 행(선택).**
   - 비율 실험의 Tyr u50·LLM-Pruner u50 행은 A100이다. MoOP u50을 같은 A100 표에 넣으려면 NEURON에서 test_2000·OOD-Val·LingoQA를 돌려야 하고, NEURON OTP 소켓이 필요하다.
   - 하지 않으면 폐루프 결과만 남는다.
3. **u40 행의 패치 하네스 재실행(선택).**
   - u40 MoOP·Tyr·LLM-Pruner를 패치된 하네스의 Origin150 × 2로 다시 돌린다. arm당 약 8시간이다.
   - 그러면 u40과 u50을 같은 하네스·같은 표에서 비교할 수 있다.
