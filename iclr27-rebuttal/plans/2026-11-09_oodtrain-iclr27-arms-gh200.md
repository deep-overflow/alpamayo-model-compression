# 계획: ICLR27 arm의 OOD-train 개루프 평가 — KISTI 한강(kisti6, GH200)

2026-11-09 · 상태: **완료(2026-11-09, 48/48 태스크, holder 1243307)** · 보고서 `iclr27-rebuttal/reports/2026-11-09_oodtrain-iclr27-arms-gh200.md` · 결과는 `iclr27-rebuttal/`에 정리

**사용자 결정(11-09):**
- MoOP + expert 포함(기본값).
- OOD CoC 품질은 정답 CoC NLL만 본다(Claude 채점은 하지 않음).
- 결과 경로는 `outputs/iclr27-rebuttal/gh200/`.

**구현:**
- `cvpr27/code/eval_a100.sbatch`: arm 4개와 `EXP_ROOT`를 추가했다.
- `iclr27-rebuttal/code/`에 네 스크립트를 두었다.
  - `push_oodtrain_gh200.sh`: 전송과 G0 해시 확인
  - `launch_oodtrain_gh200.sh`: G0 점검, 큐 48줄(주행 32 + NLL 16), holder 제출
  - `pull_oodtrain_gh200.sh`: 회수
  - `analyze_oodtrain_gh200.py`: `--stage gate`(G1), `--stage table`
- 분석 스크립트는 A100 OOD-train 행(dense, MoOP)으로 시험했고, 11-03 A100 보고서 수치를 그대로 재현했다(dense 0.922, MoOP 1.017, Δ +0.095 [+0.043, +0.149]).
- NLL 태스크는 arm당 2샤드다. 1,271클립을 한 번의 forward로 채점하므로 주행 태스크보다 짧다.

## 1. 목적

- 논문의 OOD 평가(OOD-Val262)는 262클립이다. 같은 OOD 풀의 train 쪽 1,271클립(OOD-train, 5배)에서 **arm 순위가 유지되는지** 본다. OOD 결론이 표본 크기 탓이라는 지적에 답하는 근거다.
- 가설: MoOP가 FM-only·CE-only·Wanda·LLM-Pruner보다 낫고, Tyr와는 비슷하며, 무압축보다는 나쁘다(OOD-Val에서 본 순서).
- 참고(A100, 이미 있음): OOD-train에서 MoOP − dense +0.095 [+0.043, +0.149]* (6.4 s minADE@6). l0keep 판과 tyr_off·LLM-Pruner(lp_m24)·Wanda-sp도 있지만, **ICLR27 표의 빌드로는 dense와 MoOP 두 개만 있다.**

## 2. 대상 arm — ICLR27 표의 빌드(0번 층 Q head 비보존판)

| 논문 행 | 빌드 | 큐 이름 | kisti6에 있나 | 보낼 것 |
|---|---|---|---|---|
| dense | 무압축 | `dense` | 있음 | — |
| MoOP | `slim_dual_u40_v2` | `moop` | 있음(`slim_meta.json`) | — |
| MoOP + expert | `slim_dualexp_u40_em93p75` | `moop_em93p75` (새로 추가) | 없음 | `slim_meta.json` 외 작은 파일 3개 |
| FM-only | `slim_traj_u40_v2` | `fm_only` (새로 추가) | 없음 | 같음 |
| CE-only | `slim_coc_u40_v2` | `ce_only` (새로 추가) | 없음 | 같음 |
| Wanda | `slim_wanda_u40_v2` | `wanda` (새로 추가) | 없음 | 같음 |
| LLM-Pruner | `cvpr27/slim_lp_r50` (soowon `lp_r50`과 유닛 단위 동일) | `lp_r50` | 없음 | 작은 파일 3개(3.2 MB) |
| Tyr | `slim_tyr_u40_r` | `tyr_u40_r` | 없음 | `slim_meta.json` + **`slim_state.pt` 16 GB** |

- 마스크만 쓰는 arm은 기준 가중치와 `slim_meta.json`으로 다시 빌드되므로 `slim_state.pt`를 보내지 않는다(A100 트랙과 같은 방식).
- Tyr는 재구성으로 가중치를 다시 쓰기 때문에 `slim_state.pt`가 있어야 한다. 이 파일은 NEURON에 있을 것으로 보이며(A100 트랙의 arm 표에 `tyr_u40_r`이 있음), NEURON Datamover → 한강 Datamover 내부망으로 보낸다(`cvpr27/code/neuron_to_kisti6.sh` 방식, 22 ms 구간).
  - NEURON에 없으면 cvlab21에서 직접 보내야 하는데, 이 회선은 20–50 kB/s까지 떨어진 적이 있다(1단계에서 확인).
- l0keep 판, 비율 실험 arm은 이번 범위가 아니다.

## 3. 고정 조건 (A100 OOD-train과 같게)

| 항목 | 값 |
|---|---|
| 세트 | OOD-train 1,271클립 = `outputs/eval_sets/ood_train.parquet`. kisti6 `/scratch/e1997a06/datasets/physicalai_av/pre_processed/ood`에 이미 있음(11-06 전송) |
| 실행 | `experiments/evaluation/run_baseline.py --set ood --manifest ood_train --no-tf`, k=8 표본 저장, 클립별 seed 규칙, arm당 4샤드 |
| 지표 | minADE@6·minFDE@6을 1.6 / 3.2 / 6.4 s에서(평균 우선, 중앙값 병기), CoC 퇴화율 |
| 모델·환경 | `nvidia/Alpamayo-1.5-10B` rev `7aba8293`, kisti6 venv(torch 2.8.0+cu129, flash-attn 2.8.3) |
| 하드웨어 | **GH200만.** dense도 GH200에서 다시 돌려 같은 카드 종류로만 비교한다. Ada(논문 Test500·OOD-Val262)나 A100 표와 섞지 않는다 |
| 통계 | arm 간 짝지은 Δ(클립 단위), 부트스트랩 200,000회, 95% CI. 기준은 dense와 MoOP 둘 다 |

## 4. 코드 변경 (최소)

1. `cvpr27/code/eval_a100.sbatch`
   - arm 표(`MODEL`)에 `moop_em93p75`, `fm_only`, `ce_only`, `wanda` 4개를 추가한다.
   - 결과 경로 앞부분을 정하는 `EXP_ROOT` 환경변수를 추가한다(기본값 `cvpr27`이라 기존 큐에는 영향 없음). 이번 실행은 `EXP_ROOT=iclr27-rebuttal`로 `outputs/iclr27-rebuttal/gh200/<arm>_oodtrain/`에 쓴다.
2. `iclr27-rebuttal/code/push_oodtrain_gh200.sh` — 작은 recipe 6개와 코드 묶음(`experiments/`, `cvpr27/code`, `iclr27-rebuttal/code`)을 hangang-dm으로 보낸다. `BatchMode=yes`, 전송 후 sha256 대조.
3. `iclr27-rebuttal/code/launch_oodtrain_gh200.sh` — `launch_gh200_gate.sh` 방식이다.
   - 큐 `outputs/iclr27-rebuttal/gh200/queue_oodtrain`에 32줄(8 arm × 4샤드)을 쓴다. dense를 맨 앞에 둔다.
   - holder를 `sbatch -p 5thEx --exclusive --gres=gpu:4 --time=03:00:00 --comment="field=efficientai;appl=pytorch" --export=ALL,OUT_SITE=gh200,A100_QUEUE=<q>,EXP_ROOT=iclr27-rebuttal`로 제출한다.
4. `iclr27-rebuttal/code/analyze_oodtrain_gh200.py` — arm별 절댓값 표(세 시간대), dense·MoOP 대비 짝지은 Δ, CoC 퇴화율, A100과의 교차 확인을 낸다. 가능한 부분은 `cvpr27/code/analyze_a100.py`의 로더를 import한다.

## 5. 단계와 게이트

| 단계 | 내용 | 통과 기준 |
|---|---|---|
| 0. 사전 점검 | `hangang-status.bash`: 소켓, 내 잡, **그룹 파일 수 쿼터**. OOD-train npz 1,271개·매니페스트·hub 스냅샷이 남아 있는지(15일 미접근 삭제 규칙) | 파일 여유 ≥ 20,000개, 입력 전부 존재 |
| 1. 전송 | 작은 recipe(cvlab21 → hangang-dm), Tyr `slim_state.pt`(NEURON dm → hangang-dm), 코드 묶음 | — |
| **G0 recipe** | 보낸 파일 sha256 = 원본. `slim_meta.json`의 제거 파라미터: u40 arm 2,657,452,032, MoOP+E 4,369,416,192, lp_r50 2,768,240,640. kisti6 코드의 `run_baseline.py`·`slim_lib.py` md5 = main | 전부 일치 |
| 2. 제출 | 큐 32줄, holder 4개(4 GPU, 3 h). 동시 실행 한도 5개 중 1개는 남긴다 | — |
| **G1 첫 샤드** | dense OOD-train 샤드 0(318클립)이 끝나면 A100 dense 샤드 0(같은 클립)과 비교 | 6.4 s minADE@6 평균 차이 ≤ 0.02 m, CoC 퇴화율 비슷. 넘으면 나머지를 멈추고 원인 확인 |
| **G2 완료** | 8 arm × 1,271클립, 실패 태스크 0, 샤드 요약 파일 32개 | 전부 존재 |
| 3. 회수 | kisti6-dm → cvlab21 `outputs/iclr27-rebuttal/gh200/`(작은 파일, rsync `BatchMode=yes`). 15일 규칙 때문에 끝나는 대로 | 클립 수 일치 |
| 4. 분석·보고 | `iclr27-rebuttal/reports/YYYY-MM-DD_oodtrain-iclr27-arms-gh200.md`(사실과 해석 분리) | — |

교차 확인(보고서에 별도 행): dense·MoOP의 GH200 OOD-train을 A100 결과(dense 0.922, MoOP 1.017)와 비교한다. 사이트 게이트에서 본 차이(test_2000 +0.004)와 비슷한지 보는 건전성 확인일 뿐, 표를 합치지는 않는다.

## 6. 비용

- **GPU:** A100에서 318클립 샤드 하나가 32.5분 걸렸다(로그). 압축 arm은 로드·수술에 약 8분이 더 든다. GH200은 2–5% 느리다.
  - 샤드당 약 40분 × 32샤드 ≈ **22 GH200-시간**.
  - holder 4개(16장)면 두 바퀴로 **약 1.5–2시간**이 걸린다.
- **저장:** /scratch 새 파일은 약 200개다(recipe 약 20 + 태스크당 결과 약 5 × 32 + 로그). 용량은 Tyr 16 GB 외에는 MB 단위다.
- **전송:** 작은 recipe 6개는 MB 단위이고, Tyr 16 GB는 NEURON 내부망으로 보낸다.

## 7. 위험

- **사용자 조치가 필요하다.** kisti6 로그인·Datamover 소켓(OTP)과 NEURON → 한강 소켓을 열어 주셔야 한다. 스크립트는 `BatchMode=yes`라 소켓이 없으면 로그인 시도 없이 멈춘다.
- **그룹 파일 수 쿼터(11-06 99.2%).** 다른 그룹원이 채우면 결과 쓰기가 실패한다. 제출 직전에 다시 확인한다.
- **15일 미접근 삭제.** 입력은 11-06에 올렸으므로 11-21 전후로 지워질 수 있다. 그 전에 돌리거나 다시 보낸다.
- **Tyr `slim_state.pt`가 NEURON에 없을 수 있다.** 그러면 느린 직접 회선을 쓰거나, 사용자 판단에 따라 Tyr를 빼거나 미룬다.
- **해석 한계.** ICLR27 빌드는 0번 층 head 선택에 따른 분산을 안고 있다. 결과는 "이 빌드들"의 OOD-train 성능이지 기준 일반의 성질이 아니다.

## 8. 결정 필요

1. **MoOP + expert(`dualexp_u40_em93p75`)를 포함할지.** 기본은 포함(논문 본표에 있는 행). 비용은 샤드 4개, 약 2.7 GH200-시간이다.
2. **OOD CoC 품질까지 볼지.** 기본은 주행 지표와 CoC 퇴화율만 본다.
   - 정답 CoC NLL(`nll` 태스크)은 GPU 시간이 더 든다.
   - Claude 채점은 8 arm × 1,271클립 × 약 $0.024 ≈ $240이다.
3. **결과 경로.** 기본은 `outputs/iclr27-rebuttal/gh200/`이다(`EXP_ROOT` 추가). 기존처럼 `outputs/cvpr27/gh200/`에 둘 수도 있다.
