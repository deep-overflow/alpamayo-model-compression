# 계획: u50 arm open loop + LingoQA + CoC NLL (Blackwell 0–3)

2026-11-12 · 상태: **진행 중(사용자 지시)** · 결과 `outputs/iclr27-rebuttal/u50_openloop_bw/`

**사용자 결정(11-12):**
- u50 closed loop에 쓴 모델로 open loop를 잰다.
- GPU 0–3을 쓴다.
- 세트는 Test500과 OOD-Val262다.
- LingoQA와 CoC NLL도 함께 잰다.

## 목적

- u50 closed loop(`reports/2026-11-12_u50-closedloop-origin150-ci.md`)에서 MoOP u50은 모든 압축 기준선보다 score가 유의하게 높고 과실 충돌이 유의하게 적었다.
- 같은 모델로 논문 표 1·4·9·12와 같은 open loop와 언어 지표를 내서, closed loop 결과와 짝을 맞춘다.

## arm

모두 u50이고, 논문(ICLR) 레시피를 따른다. MoOP u50 l0keep은 사용자 요청으로 뺀다.

| arm | 모델 |
|---|---|
| dense | `baseline` |
| MoOP u50 | `outputs/slim_dual_u50_v2` |
| Tyr u50 (논문 포트) | `outputs/slim_tyr_u50_r` |
| LLM-Pruner u50 | `outputs/slim_lp_r60` |
| FM-only u50 | `outputs/slim_traj_u50_v2` |
| CE-only u50 | `outputs/slim_coc_u50_v2` |

## 평가 (논문 표와 같은 명령)

| 항목 | 명령 |
|---|---|
| Test500 | `run_baseline.py --set test`, 4샤드 |
| OOD-Val262 | `run_baseline.py --set ood --manifest ood_val --no-tf`, 2샤드 |
| 지표 | k=8 중 앞 6개로 minADE@6·minFDE@6을 1.6/3.2/6.4 s에서 내고, CoC 퇴화율도 낸다 |
| CoC NLL | `run_nll_dense.py --set test`. 기준 문장은 표 9와 같은 Ada dense CoC(`baseline_ada_ps_test`) |
| LingoQA | `run_lingo_vqa.py` + `score_lingo_vqa.py` (500문항, Lingo-Judge) |

## 하드웨어

- RTX PRO 5000 Blackwell 0–3, 카드당 worker 하나가 공용 대기열에서 작업을 가져간다.
- 실행 스크립트: `code/run_u50_openloop.sh`
- **open loop는 표 하나에 GPU 종류 하나만 둔다.** 그래서 dense도 Blackwell에서 새로 재고, 이 표는 논문의 Ada 표와 섞지 않는다.

## 분석

- `code/u50_openloop_ci.py`
- `ci_openloop_lingoqa.py`의 open_loop_block·lingo_block·nll_block을 그대로 쓴다.
  - 부트스트랩: 클립 단위(LingoQA는 세그먼트 단위), 20만 회, 시드 0, 시드 1로 확인
  - 기준: MoOP u50과 dense
- 표는 절댓값을 먼저 싣고, 짝 차이 CI를 붙인다.

## 비용

- 카드당 클립 하나에 약 9.5초(Ada 기준)가 걸린다. arm당 open loop 762클립이 4장에서 약 30분이다.
- NLL은 arm당 약 10–15분, LingoQA는 답변과 채점을 합쳐 arm당 약 30–40분이다.
- 6개 arm 합계는 4장에서 약 5–6시간으로 본다.
