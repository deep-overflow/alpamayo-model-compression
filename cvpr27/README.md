# cvpr27/ — CVPR 2027 실험·분석

2026-10-29부터 진행하는 실험과 분석을 모아 두는 작업 폴더다. 계획서, 코드, 리포트를 모두 여기에 두고,
기존 `plans/`, `reports/`, `experiments/`에는 새로 추가하지 않는다.

## 폴더 구조

```
cvpr27/
├── README.md
├── plans/      # 계획서 (YYYY-MM-DD_name.md)
├── code/       # 이 트랙 전용 스크립트
├── reports/    # 리포트 (YYYY-MM-DD_name.md, 그림은 reports/figures/)
└── outputs     # → /mnt/nvme1n1/ad_vla/outputs/chan/cvpr27 (심볼릭 링크, git 무시)
```

`outputs`는 루트 파티션이 아니라 `/mnt`에 둔다. 같은 데이터가 레포의 `outputs/cvpr27/`로도 보이므로,
기존 러너는 고치지 않고 `--exp-id cvpr27/<실험명>`으로 쓰면 된다. 클론이나 worktree마다 한 번 만든다:

```bash
ln -s /mnt/nvme1n1/ad_vla/outputs/chan/cvpr27 cvpr27/outputs
```

실험마다 `outputs/cvpr27/<exp_id>/`에 `config.json`, `metrics.json`, `summary.txt`, 그림(`.png`)을 남긴다.
`collect_results.py`(결과 시트)는 `outputs/` 바로 아래만 훑으므로, 여기 결과는 시트에 자동으로 오르지 않는다.

## 공용 코드 (복사하지 않고 import)

| 용도 | 파일 |
|---|---|
| Taylor 중요도 (dual-objective) | `experiments/head_analysis/prune_lib.py` |
| 마스크·물리 제거·체크포인트 | `experiments/head_analysis/mask_lib.py`, `slim_lib.py`, `make_slim.py` |
| 개루프 평가 (페어드 seed) | `experiments/evaluation/run_baseline.py`, `experiments/head_analysis/eval_lib.py` |
| 프롬프트·토큰 span | `experiments/head_analysis/analysis_lib.py` |

스크립트는 레포 루트에서 `.venv/bin/python cvpr27/code/<script>.py`로 실행한다.

## 분석 목록

| 날짜 | 주제 | 리포트 | 코드 | 상태 |
|---|---|---|---|---|
| 2026-10-29 | VLM MLP 중요도 편중의 층별 분해 — 1–4번 층만 expert처럼 편중, 나머지는 평평 | [리포트](reports/2026-10-29_vlm-mlp-layer-concentration.md) | `code/analyze_layer_concentration.py` | 완료 (CPU, 저장된 중요도) |
| 2026-10-30 | MoOP l0keep + VLM 1–4번 층 MLP 추가 제거(층별 질량 95/90/80%, A100) — 주행 +0.006/+0.010/+0.016, LingoQA −3.6/−4.0/−6.0 pp(MoOP l0keep 대비) → 사전 등록 기준 "공짜" 아님 | [리포트](reports/2026-10-30_l0keep-l14-mlp-cut.md) · [계획](plans/2026-10-30_l0keep-l14-mlp-cut.md) | `experiments/head_analysis/make_slim.py`(`_l14m<XX>`) | 완료 (약 15 A100-시간) |
| 2026-10-31 | LLM-Pruner를 MoOP와 같은 제거량(2,657,452,032)으로(A100) — `lp_m24` test_2000 1.365(MoOP 대비 +0.409*), OOD-Val +0.456*, LingoQA 53.4%(−13.2 pp*); 기존 `lp_r50`(24.99%)은 주행 +0.04 m, LingoQA −8.0 pp 더 나쁨 | [리포트](reports/2026-10-31_llm-pruner-budget-matched.md) · [계획](plans/2026-10-30_llm-pruner-budget-matched.md) | `experiments/llm_pruner/lp_prune.py`(`--mlp-extra-layers`), `code/a100_queue.sh arm` | 완료 (약 11 A100-시간) |
| 2026-10-31 | 1–4번 층 MLP 추가 절단 세 판(l14m95/90/80)의 s600 폐루프 — MoOP l0keep 대비 score −0.017 / −0.002 / −0.014, 모두 CI가 0 포함(검출 가능한 비용 없음), 용량-반응 없음; 늘어난 충돌은 뒤 충돌(기준 런이 이례적으로 낮음) | [리포트](reports/2026-10-31_l14m-closedloop-s600.md) · [계획](plans/2026-10-30_l14m-closedloop-s600.md) | `code/run_l14m_s600.sh` | 완료 (s600, 8장 × 3판, 약 22시간) |
| 2026-11-02 | LLM-Pruner 두 판(`lp_m24`, `lp_r50`) s600 폐루프 — lp_m24 score 0.766 = 출하본 MoOP(−0.000)이지만 과실 충돌 2배(+0.031*)·offroad −0.044*로 상쇄; lp_r50 − lp_m24 −0.011 n.s.; 개루프 +0.409 m와 어긋남 | [리포트](reports/2026-11-02_lp-closedloop-s600.md) · [계획](plans/2026-10-31_lp-closedloop-s600.md) | `code/run_lp_pair_split_s600.sh`, `code/finish_lp_pair_s600.sh` | 완료 (s600, 판마다 B2+A2, 기준선 카드 종류 일치) |
| 2026-11-02 | Tyr 공식 재빌드판(`tyr_off`) s600 폐루프 — score 0.738, 출하본 MoOP 대비 −0.028*(같은 카드 −0.037*), dense와 같음(+0.009); plan deviation·offroad·충돌이 dense와 같은 'dense 재현' 프로파일 — 개루프(MoOP보다 −0.075 m*)와 순위 반대 | [리포트](reports/2026-11-02_tyr-off-closedloop-s600.md) · [계획](plans/2026-11-01_tyr-off-closedloop-s600.md) | `code/run_tyr_off_s600.sh`, `code/resume_tyr_off_8gpu.sh` | 완료 (4장 164씬 → 8장 436씬) |
| 2026-11-02 | Wanda-sp(FLAP 공식, 층별 순차, 제거량 = MoOP) A100 개루프 — test_2000 2.232(MoOP 대비 +1.276*), OOD-Val +1.106*, LingoQA 39.0%(−27.6 pp*), CoC 퇴화 75%; 네 버킷 모두 2–3배로 무너짐 | [리포트](reports/2026-11-02_wandasp-a100-openloop.md) · [계획](plans/2026-11-02_wandasp-closedloop-s600.md) | `code/run_wanda_sp.py`, `code/eval_a100.sbatch` | 완료 (4장 × 2시간 46분 ≈ 11 A100-시간; 퇴화 CoC가 256토큰까지 늘어 느림) |
| 2026-11-02 | Tyr 공식 절차 + 헤시안 V:T = 1:1(`tyr_vt`, A100 빌드·평가) — LingoQA 38.0 → 62.8%(tyr_off 대비 +24.8 pp*), 주행은 tyr_off보다 +0.049 m*(test_2000 0.931, MoOP 대비 −0.026 n.s.); head가 0–17층에서 18–35층으로 63개 이동 | [리포트](reports/2026-11-02_tyr-vt-equal-hessian.md) · [계획](plans/2026-11-01_tyr-vt-equal-hessian.md) | `code/tyr_off/supernet.py`(`--hessian-mix vt_equal`) | 완료 (빌드 약 22 A100-시간, 평가 약 5 A100-시간) |
| 2026-11-05 | **A100 개루프 결과 정리 (최신)** — u40 예산의 모든 arm(기준·외부 기준선·기준 ablation·추가 절단) test_2000·OOD-Val·OOD-train × minADE@6·minFDE@6 1.6/3.2/6.4 s + MoOP l0keep 대비 짝지은 CI, LingoQA, 비율 곡선, OOD CoC 품질, 핵심 결론 | [리포트](reports/2026-11-05_a100-results-summary.md) | `code/analyze_u40_summary.py`, `code/analyze_ratio_sweep.py` | 완료 |
| 2026-11-02 | A100 개루프 결과 모음 (11-03까지) — 11-05 정리본으로 대체됨 | [리포트](reports/2026-11-02_a100-results-summary.md) | `code/analyze_a100.py` | 대체됨 |
| 2026-11-05 | MoOP l0keep + expert MLP 절단(75/87.5/93.75%, A100: test_2000·OOD-Val·OOD-train) — 93.75%(층당 516채널)까지 6.4 s minADE@6 변화 ±0.003 m 이내(모두 n.s.), CoC 100% 동일; 제거 24.0% → 39.4%, bf16 −3.42 GB | [리포트](reports/2026-11-05_moop-l0keep-expert-mlp.md) · [계획](plans/2026-11-05_moop-l0keep-expert-mlp.md) | `experiments/head_analysis/make_slim.py`(`dualexp_u40_l0keep_em<M>`), `code/analyze_exp_mlp.py` | 완료 (약 18 A100-시간) |
| 2026-11-05 | 중요도 결합 규칙 ablation (u40, l0keep, A100) — rank-mean은 주행 = MoOP(+0.005 n.s.), LingoQA −4.4 pp(경계); raw mean·raw max는 주행 +0.05 m*·LingoQA −10~11 pp*(원래 크기 결합은 깊이별로 FM/CoC 한쪽만 대변); 순위 정규화가 결정적 | [리포트](reports/2026-11-05_combination-rule-ablation.md) · [계획](plans/2026-11-04_combination-rule-ablation.md) | `experiments/head_analysis/make_slim.py`(`rawabsavg`), `code/analyze_comb_ablation.py` | 완료 (약 15 A100-시간) |
| 2026-11-05 | 외부 기준선·단일 기준 비율 실험 (u35/u40/u45/u50, 15개 새 arm, A100: test_2000·OOD-Val·LingoQA) — MoOP l0keep만 비율이 커져도 주행·언어 함께 유지(test 0.933→1.079, LingoQA ≥68% to u45); Tyr는 u45까지 주행 최고·언어 38–47%, u50에서 역전; FM-only·CE-only 격차는 비율과 함께 커짐(u50 +0.57*/+0.20*); LLM-Pruner·Wanda-sp는 u45부터 CoC 퇴화와 함께 붕괴 | [리포트](reports/2026-11-05_baselines-ratio-sweep.md) · [계획](plans/2026-11-03_baselines-ratio-sweep.md) | `code/analyze_ratio_sweep.py`, `code/plot_ratio_sweep.py`, `code/build_ratio_local.sh` | 완료 (평가 약 95 A100-시간) |
| 2026-11-06 | KISTI 한강(kisti6, GH200 120GB, `5thEx`)을 두 번째 평가 사이트로 구축 — aarch64 환경(torch 2.8.0+cu129, flash-attn 소스 빌드); 게이트: GH200 재실행 125/125 bit 동일, A100 대비 6.4 s minADE@6 dense +0.004 · MoOP l0keep +0.006 · tyr_off 0.000 m(CoC 2–7% 변화) → A100 표와 섞지 않음; GPU당 속도 ≈ A100(2–5% 느림), 이점은 동시 20장·짧은 대기 | [리포트](reports/2026-11-06_kisti6-gh200-site.md) · [계획](plans/2026-11-05_kisti6-gh200-site.md) | `experiments/transfer/build_venv_kisti6.sh`, `build_flash_attn_kisti6.sbatch`, `code/push_kisti6.sh`, `code/launch_gh200_gate.sh`, `code/analyze_gh200_gate.py` | 완료 (게이트 약 1 GH200-시간) |
| 2026-11-03 | OOD-train(1,271클립) 개루프 + OOD CoC 품질(정답 CoC NLL, Claude 채점), 7개 arm — tyr_off의 주행 = dense는 in-distribution 한정(OOD-train dense 대비 +0.113*, MoOP와 같음); Claude 채점 dense 3.05, MoOP·l0keep·tyr_vt −0.12~−0.14*, tyr_off −0.30*, LLM-Pruner −0.41*, Wanda-sp −1.7*; Tyr의 낮은 정답 CoC NLL은 분포 평탄화 탓; 채점 반복 일치 92.6% | [리포트](reports/2026-11-03_oodtrain-coc-quality.md) · [계획](plans/2026-11-02_oodtrain-coc-quality.md) | `code/run_nll_gtcoc.py`, `code/grade_coc_claude.py`, `code/analyze_coc_quality.py` | 완료 (약 27 A100-시간, Claude API $31.42) |
| 2026-11-03 | 논문 Tyr 행(`tyr_u40_r`, 간이 포트) s600 폐루프 — score 0.783로 13개 arm 중 수치상 1위(MoOP +0.017, l0keep +0.009, 둘 다 n.s.; tyr_off 대비 +0.044*); 과실 충돌 0.032 최저; LingoQA 34% | [리포트](reports/2026-11-03_tyr-u40-r-closedloop-s600.md) · [계획](plans/2026-11-02_tyr-u40-r-closedloop-s600.md) | `code/run_tyr_u40_r_s600.sh` | 완료 |
| 2026-11-03 | Wanda-sp(FLAP 공식, 층별 순차, 0번 층 포함 균일) 빌드 + s600 폐루프 — score 0.683, 전 arm 중 꼴찌(dense −0.046*, MoOP −0.083*); CoC 퇴화 39.7%, 뒤 충돌 16%·offroad 6%, 17.6시간 | [리포트](reports/2026-11-03_wandasp-closedloop-s600.md) · [계획](plans/2026-11-02_wandasp-closedloop-s600.md) | `code/run_wanda_sp.py`, `code/run_wandasp_s600.sh` | 완료 |
| 2026-11-02 | s600 폐루프 11개 arm 정리 + AlpaSim score(km/사건, R1 논문 정의) — scene score와 서열이 다름: at-fault는 offroad 적은 LLM-Pruner 1위, all은 MoOP l0keep 1위; CI ±0.2–0.3 km | [리포트](reports/2026-11-02_s600-closedloop-all-arms.md) | `code/alpasim_score_s600.py` | 완료 (Wanda-sp·tyr_u40_r 추가 예정) |
| 2026-11-02 | Claude API(`claude-opus-5-5`)로 long-tail 씬 탐지 → 저장된 A100 개루프 결과(15개 arm)를 long-tail / NOMINAL로 분할 — test_2000 15.3%(305클립: 사건형 213, 조건형 92)를 long-tail로 판정, MoOP l0keep 비용 long-tail +0.105 vs NOMINAL +0.064(DiD +0.039 [−0.029, +0.108], 기동 맞춤); 14개 arm 모두 양수지만 CI가 0을 벗어난 건 L1-4 MLP 80%·`tyr_u40_r`뿐, 순위 유지(ρ 0.86); OOD-val에선 방향 반대; 라벨링 $53.59 | [리포트](reports/2026-11-02_long-tail-openloop.md) · [계획](long-tail-detection/plan.md) | `long-tail-detection/label_longtail.py`, `long-tail-detection/analyze_longtail_openloop.py` | 완료 (GPU 없음, API $56) |
| 2026-11-01 | Tyr-the-Pruner를 공식 절차대로 다시 빌드(`tyr_off`: supernet·탐색 4회, 논문 식 5 오차 누적, top-8,192 KL, calib_100; 제거량 = MoOP) → A100 — 주행은 dense와 같음(test_2000 0.882, MoOP 대비 −0.075*), LingoQA는 38.0%(MoOP 대비 −28.6 pp*, 답변이 한 단어로 끊김) | [리포트](reports/2026-11-01_tyr-official-a100.md) · [계획](plans/2026-10-31_tyr-a100.md) | `code/tyr_off/` | 완료 (빌드 Blackwell 약 6시간, 평가 약 5 A100-시간) |
| 2026-10-31 | KISTI NEURON 폐루프 스모크 (dense, 장면 1개) — docker 없이 AlpaSim SLURM 모드(pyxis+enroot)로 A100에서 rollout 완주, NuRec 렌더러 A100 동작, 지표가 로컬 dense rollout과 같은 범위 | [리포트](reports/2026-10-31_kisti-closedloop-smoke.md) · [계획](plans/2026-10-31_kisti-closedloop-smoke.md) | `code/alpasim_neuron/` | 통과 (잡 956432) |
| 2026-10-30 | A100 FM-only·CE-only(0번 층 Q 보존) — test_2000에서 MoOP l0keep 대비 FM-only +0.034*, CE-only +0.085*; LingoQA FM-only −27.2 pp*, CE-only −5.0 pp* | [리포트](reports/2026-10-30_a100-fm-ce-only.md) · [계획](plans/2026-10-29_a100-fm-ce-only.md) | `experiments/head_analysis/make_slim.py`(`trajlastcoc`), `code/a100_queue.sh arm` | 완료 (약 10 A100-시간) |
| 2026-10-29 | KISTI A100 개루프 트랙 — dense·MoOP·MoOP l0keep을 A100에서 test_2000·OOD-Val·LingoQA로 재측정(Ada 표와 섞지 않음) | [리포트](reports/2026-10-29_kisti-a100-openloop.md) · [계획](plans/2026-10-29_kisti-a100-openloop.md) | `code/eval_a100.sbatch`, `code/a100_queue.sh`, `code/a100_pull.sh`, `code/analyze_a100.py` | 완료 (게이트 PASS, 18 A100-시간) |
| 2026-10-29 | VLM MLP 1–4번 층 추가 절단 (l0keep 위 +2.7%p) — 손실 검증 → 개루프 → 폐루프 | [계획서](plans/2026-10-29_early-mlp-cut.md) | `code/make_early_mlp_masks.py` 외 | ① 완료 (FM FREE, NLL INCONCLUSIVE), ② 이 서버에서 중단 — 개루프·폐루프는 다른 세션에서 |
| 2026-10-30 | VLM MLP 중요도의 깊이 구조 — 편중(Gini)·두 목적의 순위 일치(ρ)·calibration 안정성이 함께 바뀜 | [그림](reports/figures/2026-10-30_depth_structure.png) | `code/plot_depth_structure.py` | 완료 (CPU, 저장된 중요도) |
| 2026-10-30 | 분석에서 나온 후속 실험 후보 (재배분·층별 한계·기전·질량 기반 폭·적은 클립·백본 비교) | [후보 목록](plans/2026-10-30_followup-experiments.md) | — | 후보 (선택 대기) |
