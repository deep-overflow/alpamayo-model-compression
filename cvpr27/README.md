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
| 2026-10-30 | A100 FM-only·CE-only(0번 층 Q 보존) — test_2000에서 MoOP l0keep 대비 FM-only +0.034*, CE-only +0.085*; LingoQA FM-only −27.2 pp*, CE-only −5.0 pp* | [리포트](reports/2026-10-30_a100-fm-ce-only.md) · [계획](plans/2026-10-29_a100-fm-ce-only.md) | `experiments/head_analysis/make_slim.py`(`trajlastcoc`), `code/a100_queue.sh arm` | 완료 (약 10 A100-시간) |
| 2026-10-29 | KISTI A100 개루프 트랙 — dense·MoOP·MoOP l0keep을 A100에서 test_2000·OOD-Val·LingoQA로 재측정(Ada 표와 섞지 않음) | [리포트](reports/2026-10-29_kisti-a100-openloop.md) · [계획](plans/2026-10-29_kisti-a100-openloop.md) | `code/eval_a100.sbatch`, `code/a100_queue.sh`, `code/a100_pull.sh`, `code/analyze_a100.py` | 완료 (게이트 PASS, 18 A100-시간) |
| 2026-10-29 | VLM MLP 1–4번 층 추가 절단 (l0keep 위 +2.7%p) — 손실 검증 → 개루프 → 폐루프 | [계획서](plans/2026-10-29_early-mlp-cut.md) | `code/make_early_mlp_masks.py` 외 | ① 완료 (FM FREE, NLL INCONCLUSIVE), ② 이 서버에서 중단 — 개루프·폐루프는 다른 세션에서 |
| 2026-10-30 | VLM MLP 중요도의 깊이 구조 — 편중(Gini)·두 목적의 순위 일치(ρ)·calibration 안정성이 함께 바뀜 | [그림](reports/figures/2026-10-30_depth_structure.png) | `code/plot_depth_structure.py` | 완료 (CPU, 저장된 중요도) |
