# VLM MLP 1–4번 층 추가 절단: 편중된 층을 더 깊게 자르면 공짜인가 (2026-10-29)

## 0. 목적

- **출발점.** 층별 분석(`cvpr27/reports/2026-10-29_vlm-mlp-layer-concentration.md`)에서 VLM MLP의 1차 Taylor 중요도는
  1–4번 층에서만 expert MLP처럼 몰려 있었다. 12,288개 채널 중 40–151개가 질량의 절반을 가졌다.
  - l0keep(`dual_u40_l0keep_v2`)의 균일 절단은 이 층들에서 질량을 0.1–1.8%만 지운다. 다른 층에서는 약 22%를 지운다.
- **가설 H1.** 1–4번 층에서 Taylor 상위 10%(또는 5%) 밖의 채널은 기능이 거의 없다.
  - 그래서 l0keep 위에서 그 채널들을 더 지워도, 파라미터당 비용이 기존 24% 절단보다 크지 않다.
  - 이것을 세 수준에서 차례로 확인한다: 손실(①) → 개루프 minADE(②) → 폐루프 점수(③).
- **H1이 틀리는 두 가지 방식.**
  1. 편중이 Taylor의 인공물이다. MLP 채널 하나하나의 1차 순위는 부정확했다(expert probe r=0.33).
     이 경우 상위를 남기든 무작위로 남기든 결과가 같다.
  2. 편중은 진짜지만 제거가 공짜가 아니다. 초기 층의 변화는 뒤의 모든 층과 KV 캐시로 전파된다.
     0번 층 Q head 선택이 cruise 붕괴를 일으킨 전례가 있다.
- **범위 밖.**
  - 같은 예산 안에서의 재배분(1–4번 층을 더 자르고 다른 층을 덜 자르기). 기대 효과가 약 0.003 m 이하라 이번에 하지 않는다.
  - activation 쪽 분석(massive activation 여부, 토큰 종류별 활성).

## 1. 사용자 결정 (2026-10-29)

- 추가 압축 판(A)으로 간다. 순서는 ① 실제 제거로 손실 검증 → ② 개루프 → ③ 폐루프.

## 2. 공통 설정

- **기준 arm B0 = l0keep.** `outputs/masks_u40_l0keep/dual.npz`(Q 697, MLP 264,931 유지).
  - `slim_dual_u40_l0keep_v2`와 같은 선택이다. 제거 2,657,447,936개(전체 11,078,526,194개의 23.99%).
  - 1–4번 층 MLP는 층마다 7,359개를 유지한다. 이 집합은 dual 점수(max(rank_norm I_traj, rank_norm I_CoC)) 상위 7,359개와
    정확히 같다(계산으로 확인).
- **처치.** 1–4번 층 MLP만 dual 점수 상위 K개로 줄인다. 그 밖(Q head, 0번·5–35번 층 MLP, expert, KV)은 B0와 비트 단위로 같다.
  K개 집합은 B0 유지 집합의 부분집합이다.

  | arm | 1–4번 층 유지 | 추가 제거 | 총 제거 | 추가 ÷ B0 제거 |
  |---|---|---|---|---|
  | B0 (l0keep) | 7,359 | — | 2,657,447,936 (23.99%) | — |
  | **A10** | 1,229 (10%) | 301,301,760 | 2,958,749,696 (26.71%) | 0.1134 |
  | **A05** | 614 (5%) | 331,530,240 | 2,988,978,176 (26.98%) | 0.1248 |

- **비례 문턱 (저장소 관례, `plans/2026-08-26_dual-plus-znorm.md`와 같은 형태).**
  - 문턱 = (B0 − dense) × (추가 제거 ÷ B0 제거). "추가로 지운 파라미터가 기존 절단의 평균 단가보다 비싸지 않다"는 뜻이다.
  - 판정은 평균 Δ(arm − B0)의 95% CI로 한다.
    - **FREE**: 상한 < 문턱
    - **REJECT**: 하한 ≥ 문턱
    - 그 외 **INCONCLUSIVE**
  - CI가 문턱에 걸치면 부트스트랩을 수렴시킨다(20만 회). 중앙값과 절사평균도 함께 보고한다.
- **결과 위치.** 코드는 `cvpr27/code/`, 결과는 `outputs/cvpr27/early_mlp/`, 리포트는 `cvpr27/reports/<날짜>_early-mlp-cut.md`.

## 3. ① 실제 제거로 손실 검증 (GPU 1장, 약 1.5시간)

- **클립.** `calib_rd100_a` 100개(캐시 `calib_rd_a`, 100/100 존재).
  - 무작위 추출이다. 평가셋 3개(val500·test500·OOD-Val262)와도, 중요도를 잰 calib_100과도 겹치지 않는다.
- **러너.** 기존 `experiments/head_analysis/run_token_ablation.py --arm-masks`를 그대로 쓴다.
  - dense가 만든 CoC를 클립마다 한 번 생성한다.
  - config마다 마스크를 씌운 teacher-forced forward 한 번으로 다음을 읽는다:
    - 그 텍스트의 CoC NLL
    - GT 앵커 FM 손실(10스텝, `run_importance`와 같은 노이즈)
    - minADE/minFDE(K=8)
  - seed는 클립에서 유도한다.
- **config (dense 포함 11개).** 모두 B0에서 출발하고 1–4번 층 MLP만 바꾼다.

  | 이름 | 1–4번 층 MLP | 묻는 것 |
  |---|---|---|
  | `b0` | B0 그대로 | 기준 |
  | `e10`, `e05` | 상위 1,229 / 614개만 유지 (= A10, A05) | 나머지가 정말 불활성인가 |
  | `r10` | B0 유지 집합에서 무작위 1,229개 유지 (seed 0) | 같은 깊이에서 Taylor 순위가 무작위보다 나은가 |
  | `t64` | B0에서 dual 점수 상위 64개만 제거 | 상위 소수가 기능을 지니는가 |
  | `r64` | B0 유지 집합에서 무작위 64개 제거 (seed 0) | `t64`의 대조군 |
  | `e05_l1` … `e05_l4` | 한 층만 상위 614개 유지, 나머지 층은 B0 | 층별 내성 (서술용) |

- **게이트.**
  - **G1 (서술): Taylor 순위가 인과적인가.**
    - `t64 − r64`와 `r10 − e10`이 FM 손실·NLL 모두에서 양수이고 CI가 0을 배제하는지 본다.
    - 실패하면 H1의 첫 번째 대안(인공물)을 지지한다. 그래도 G2만으로 ②의 진행 여부를 정한다. 제거가 싸다면 그 자체로 유용하기 때문이다.
  - **G2 (②로 갈지 결정).**
    - `e10`, `e05`의 Δ(arm − `b0`)를 FM 손실과 NLL 각각에서 비례 문턱(= (`b0` − dense) × 0.1134 / 0.1248)과 비교한다.
    - 두 손실 중 하나라도 REJECT인 arm은 ②에서 뺀다. 둘 다 REJECT면 ②로 가지 않고 리포트로 끝낸다.
    - minADE@6(100클립)는 서술용으로만 쓴다.
- **산출물.** `outputs/cvpr27/early_mlp/gate_loss/`(러너의 per-clip 결과), 분석 `outputs/cvpr27/early_mlp/gate_analysis/`
  (`config.json`, `metrics.json`, `summary.txt`, 그림).

## 4. ② 개루프 (Ada 4–7)

- **빌드.** `make_slim.py --config maskfile_cvpr27/early_mlp/m_<arm> --out outputs/cvpr27/slim_early_<arm> --no-state`.
  `--out`은 레포 루트 기준이다. 마스크는 `outputs/cvpr27/early_mlp/m_<arm>/masks.npz`에서 읽는다.
  - 순수 선택이라 가중치를 저장하지 않는다. expert와 KV는 이 경로에서 원래대로 남는다.
- **G0 (빌드 게이트, 평가 전에 기계로 확인).**
  - (a) 같은 경로로 B0 마스크를 빌드하면 `slim_dual_u40_l0keep_v2`의 `slim_meta.json` 유지 인덱스와 유닛 단위로 같고,
    제거 수가 2,657,447,936개다.
  - (b) A arm은 B0와 1–4번 층 MLP에서만 다르고, 그 층의 유지 집합이 B0의 부분집합이다.
  - (c) 제거 수가 위 표와 정확히 같다.
  - (d) 빈 카드인지 실행 직전 `nvidia-smi`로 확인한다.
- **평가.** 기준 행(`swap_ship_L0fullM_*`, `baseline_ada_ps_*`)과 같은 프로토콜이다: `run_baseline.py`, k=8, seed 42, max_gen 256, Ada.
  - `--set test --manifest test_500` → `cvpr27/early_<arm>_test500`
  - `--set indist` → `cvpr27/early_<arm>_val500`
  - `--set ood --manifest ood_val` → `cvpr27/early_<arm>_oodval`
- **분석 (`cvpr27/code/analyze_early_mlp_openloop.py`, 새로 작성).**
  - 클립 단위로 B0(`swap_ship_L0fullM_{500,val500,oodval}`)와 dense(`baseline_ada_ps_{test,indist,oodval}`)에 짝짓는다.
  - minADE@6 / minFDE@6을 보고한다. 각 arm의 실제 평균·중앙값을 먼저 쓰고, Δ와 CI(20,000회)를 옆에 붙인다.
  - 버킷별(특히 cruise), CoC 퇴화율도 같이 낸다.
  - 기존 `minitron/code/compare_openloop.py`는 기준 arm의 이름 규칙이 달라서 l0keep 행과 짝짓지 못한다.
- **문턱 (minADE@6, 지금 저장된 행으로 계산; 분석 때 같은 식으로 다시 계산한다).**

  | 세트 | B0 평균 (중앙값) | dense 평균 | B0 − dense | A10 문턱 | A05 문턱 | B0 CoC 퇴화 |
  |---|---|---|---|---|---|---|
  | test500 | 0.9557 (0.6708) | 0.8417 | +0.1140 | +0.0129 | +0.0142 | 3.2% |
  | val500 | 0.8756 (0.6421) | 0.8236 | +0.0520 | +0.0059 | +0.0065 | 1.8% |
  | OOD-Val262 | 1.1142 (0.7106) | 0.9995 | +0.1146 | +0.0130 | +0.0143 | 3.4% |

- **검정력 한계를 미리 인정한다.** 평균 Δ의 CI 반폭은 0.01–0.03 m 정도다. 그래서 val500 문턱(+0.006)에서 FREE는 나오기 어렵고
  INCONCLUSIVE가 예상된다.
- **③으로 보낼 arm (사전 등록).**
  - 조건: 세 세트 모두 REJECT가 아니고, CoC 퇴화율이 세트마다 B0 + 1 pp 이하일 것.
  - 그 조건을 만족하는 가장 깊은 arm 하나만 보낸다. A05가 되면 A05, 아니면 A10이다.
- **보조 (판정에 안 씀).** LingoQA(`experiments/lingoqa/eval_lingo_arm.sh`)를 B0와 ③으로 가는 arm에 대해 Ada에서 잰다.
  - l0keep의 기존 LingoQA는 A100에서만 쟀으므로 B0도 새로 잰다.

## 5. ③ 폐루프 (s600, 8장, 약 7시간 15분)

- **빌드.** 가중치 포함(`--no-state` 없이, `--out outputs/cvpr27/slim_early_<arm>_state`)으로 만든다(15–17 GB, `/mnt` 여유 1.1 TB).
  alpasim driver 폴더에 하드링크한다.
- **실행.** `run_dual_s600.sh`를 `CFG=<driver 폴더> TAG=e600 OUT=$PWD/outputs/cvpr27/alpasim_early_s600`로 돌린다.
  - 순서는 `STAGE=smoke` → `STAGE=main`이다.
  - 씬과 카드는 `outputs/alpasim_dual_s600/scenes_d600.csv`(600씬 × 1)의 `shard` 열을 따른다. dense·MoOP와 같은 카드다.
  - 스크립트는 모든 카드가 비어 있어야 시작하고, 끝나면 컨테이너와 네트워크를 정리한다.
- **smoke 게이트.** 출하본 MoOP와 같다. 8/8 채점, slim 인식, 인계 시 |차 − 사람| 속도 < 0.1 m/s, 포화 0.
- **분석.** `analyze_dual_s600.py --base l600_merged_slim_dual_u40_l0keep_v2 --base-label "MoOP l0keep"
  --arm e600_merged_<cfg> --arm-label "l0keep+early<K>" --out outputs/cvpr27/alpasim_early_s600 --no-pre-fix`.
  - dense(`s600_merged_baseline`) 대비도 같은 방식으로 낸다.
  - B0 런은 129씬이 기준선과 다른 카드 종류(Blackwell)에서 돌았다. 그래서 600씬 전체와 같은 카드 471씬을 둘 다 보고한다.
  - 보는 지표: score(경로 끝 인지 4 m 변형), progress, offroad, 과실 충돌, CoC 퇴화율, 중단 rollout.
- **판정 (사전 등록, 서술형).** s600의 분해능은 약 ±0.02다(l0keep − MoOP CI 반폭 0.021). 그래서 0.02보다 작은 차이는 가리지 못한다.
  - "폐루프에서 검출 가능한 비용 없음": Δscore(arm − B0)의 CI가 0을 포함하고, offroad·과실 충돌이 B0보다 나빠지지 않을 때.
  - "비용 있음": CI 상한 < 0일 때.
  - 개루프에서 "MLP 폭은 공짜"였던 판정은 지금까지 폐루프에서 뒤집히지 않은 적이 없다(`dual+h4`, expert MLP 사다리 등).
    그래서 ③ 없이는 공짜라고 주장하지 않는다.

## 6. 새로 쓸 코드

| 파일 | 내용 |
|---|---|
| `cvpr27/code/make_early_mlp_masks.py` | B0에서 ①의 config 10개와 A arm 마스크를 만든다. `q_mask`/`mlp_mask`(러너용)와 `qhead`/`mlp`(maskfile용) 키를 둘 다 저장한다. 유지 수·부분집합·제거 수를 검사하고 `config.json`에 남긴다 |
| `cvpr27/code/analyze_early_mlp_gate.py` | ①의 페어드 Δ, 비례 문턱 판정(G1·G2), 층별 내성 그림 |
| `cvpr27/code/analyze_early_mlp_openloop.py` | ②의 B0·dense 페어드 비교, 문턱 판정, 버킷·CoC 퇴화 |

나머지는 기존 코드를 그대로 쓴다: `run_token_ablation.py`, `make_slim.py`(maskfile), `run_baseline.py`, `eval_lingo_arm.sh`,
`run_dual_s600.sh`, `analyze_dual_s600.py`.

## 7. 시간과 자원

| 단계 | GPU | 예상 시간 |
|---|---|---|
| ① 손실 검증 | Ada 1장 (2장으로 나누면 약 45분) | 약 1.5시간 |
| ② 빌드 + 개루프 (2 arm × 3세트) + LingoQA | Ada 4–7 | 반나절 정도 |
| ③ 폐루프 (1 arm) | 8장 전부 | 약 7시간 15분 |

## 8. git

- 브랜치 `worktree-cvpr27-early-mlp`(main `9c029af`에서 분기). 계획서, 코드, 리포트를 이 브랜치에 쌓고, 끝나면 PR을 연다.
- worktree에는 `outputs`, `.venv`, `cvpr27/outputs` 심링크를 다시 만들어야 한다.

## 9. 진행 기록

- **2026-10-29 승인.** 사용자가 "4번 GPU만 사용"을 지시했다. 그래서 ①은 GPU 4 한 장, 샤드 1개로 돈다.
  이후 단계도 따로 말이 없으면 4번만 쓴다. ③(8장)은 그 전에 다시 묻는다.
- **마스크.** `make_early_mlp_masks.py` → `outputs/cvpr27/early_mlp/m_*/masks.npz` 10세트를 만들었다.
  추가 제거 수가 §2 표와 정확히 같고, 1–4번 층 밖은 B0와 같으며, 모두 B0의 부분집합이다.
- **① 실행.** `run_token_ablation.py --arm-masks ... --manifest calib_rd100_a --cache calib_rd_a --num-clips 100 --k 8 --gpu 4`
  → `outputs/cvpr27/early_mlp/gate_loss/`. 클립당 약 49초, 100클립에 약 82분이다.
- **G2 규칙 보완 (5클립 점검 때, 최종 데이터를 보기 전).**
  - 비례 문턱은 "기존 절단의 평균 단가"라서, B0가 그 손실에서 비용을 낼 때(B0 − dense > 0)만 정의된다.
  - 5클립에서 FM 손실의 B0 − dense가 −0.0024로 음수였다.
  - 그래서 B0 − dense ≤ 0인 손실은 판정을 UNDEFINED로 두고 CI만 보고한다. UNDEFINED는 ②로 가는 것을 막지 않는다(막는 것은 REJECT뿐).
  - 5클립 값으로는 아무 판정도 내리지 않는다.
- **① 결과 (100클립, 2026-10-30 01:22 KST 완료, `outputs/cvpr27/early_mlp/gate_analysis/summary.txt`).**
  - 기준값: dense / B0의 FM 손실 평균은 0.2674 / 0.2783, CoC NLL 평균은 0.1929 / 0.2631이다.
  - G2 (판정은 사전 등록 규칙 그대로):

    | arm | 손실 | Δ(arm − B0) [95% CI] | 문턱 | 판정 |
    |---|---|---|---|---|
    | A10 | FM | +0.0004 [−0.0002, +0.0009] | +0.0012 | FREE |
    | A10 | NLL | +0.0169 [+0.0020, +0.0445] | +0.0080 | INCONCLUSIVE |
    | A05 | FM | +0.0005 [−0.0003, +0.0014] | +0.0014 | FREE |
    | A05 | NLL | +0.0097 [+0.0027, +0.0200] | +0.0088 | INCONCLUSIVE |

    REJECT가 없어서 두 arm 모두 ②로 간다.
  - G1:
    - `t64 − r64`: FM +0.0413, NLL +0.0507로 둘 다 유의하게 양수다. 상위 64개는 기능을 지닌다.
    - `r10 − e10`: FM +0.0236은 유의하게 양수이지만, NLL +0.0082 [−0.0094, +0.0249]는 유의하지 않다.
  - NLL 꼬리 (해석, 판정에는 반영하지 않음):
    - A10 NLL 증가분 합계의 77%가 클립 `da2131b8` 하나에서 나온다(accel, CoC 12토큰, 0.253 → 1.555). 그 클립의 minADE는 변하지 않았다.
    - 그 클립을 빼면 A10 +0.0039, A05 +0.0055로 문턱 아래다. 양끝 5개씩 자른 평균은 +0.0025 / +0.0031, 중앙값은 +0.0008 / +0.0006이다.
    - 이 클립에서는 A05(더 많이 지움)가 A10보다 덜 나빠(0.671) 비선형적이다.
  - 층별 (서술): 한 층만 상위 614개로 줄였을 때 NLL은 1번 층 +0.0027 [+0.0011, +0.0045], 4번 층 +0.0016(유의하지 않음)이고,
    2·3번 층은 0이다. FM은 네 층 모두 0에 가깝다.
  - minADE@8 (서술, 100클립): B0 0.922, A10 +0.011, A05 +0.016으로 CI가 0을 포함한다. r10 +0.164, t64 +0.383이다.
- **② 시작 (GPU 4).** `cvpr27/code/run_early_openloop.sh`가 빌드 3개, G0(`check_early_builds.py`), 개루프 6회를 차례로 돈다.
  로그는 `outputs/cvpr27/early_mlp/openloop.log`다.
  - G0 10개 항목을 모두 통과했다(B0 재현, smoke NLL 0.2245로 l0keep 빌드와 같음).
- **② 중단 (2026-10-30 13:2x KST, 사용자 요청).** "개루프는 여기서 안해도 될 것 같아" → "중단해줘".
  - 큐의 프로세스 그룹을 종료했고, GPU 4는 비었다.
  - 남은 것:
    - A10 test500·val500은 완료다(`outputs/cvpr27/early_a10_{test500,val500}`).
    - A10 OOD-Val262는 약 111/262에서 멈춘 부분 결과다(요약 파일 없음).
    - A05는 돌리지 않았다.
  - ②의 사전 등록 판정(③으로 보낼 arm 선정)은 내리지 않았다.
