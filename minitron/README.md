# minitron/ — Minitron 베이스라인 (Alpamayo 1.5 10B)

Minitron(arXiv 2408.11796, 방법 원전 2407.14679)을 Alpamayo 1.5 10B에 **외부 베이스라인으로 충실하게
재현**하는 작업 폴더다. 논문에 답이 있는 곳은 논문대로 하고, VLA 구조 때문에 답이 없는 곳만 사용자와
합의해 바꾼다(코드에는 `PORT:` 주석).

- 계획서: [`plans/2026-10-23_plan.md`](plans/2026-10-23_plan.md) — 맨 앞의 "1단계 실행 요약"과 §3 "확정된 결정" 표
- 문헌 조사: [`plans/2026-10-23_literature.md`](plans/2026-10-23_literature.md) — 결정 항목별 근거, VLA 압축 방법론 정리

## 폴더 구조

```
minitron/
├── README.md
├── plans/      # 계획서, 문헌 조사
├── code/       # Minitron 전용 스크립트 (구현하면서 추가)
├── reports/    # HTML 리포트 (git 추적)
└── outputs     # → /mnt/nvme1n1/ad_vla/outputs/chan/minitron (심볼릭 링크, git 무시)
```

`outputs`는 체크포인트(15–17 GB) 때문에 루트 파티션이 아닌 `/mnt`에 둔다. 같은 데이터가 레포의
`outputs/minitron/`으로도 보이므로 기존 러너는 고치지 않고 `--exp-id minitron/<실험명>`으로 쓴다.
클론마다 한 번 만든다:

```bash
ln -s /mnt/nvme1n1/ad_vla/outputs/chan/minitron minitron/outputs
```

`collect_results.py`(결과 시트)는 `outputs/` 바로 아래만 훑으므로 여기 결과는 시트에 자동으로 오르지 않는다.

## 공용 코드 (복사하지 않고 import)

MoOP 행과 같은 코드로 비교되도록 다음은 원래 위치에서 가져다 쓴다.

| 용도 | 파일 |
|---|---|
| 프루닝 수술·저장·로드 | `experiments/head_analysis/slim_lib.py` |
| MoOP 체크포인트 빌드 | `experiments/head_analysis/make_slim.py` |
| 개루프 평가 (페어드 seed) | `experiments/evaluation/run_baseline.py`, `eval_lib.py`, `sample_cache.py` |
| KI 절연·학습 인프라 참고 | `experiments/recovery/recover_lib.py`, `train_recover.py` |
| GPU 확보·hub 패치 | `experiments/head_analysis/expert_per_clip.py` |

## 1단계 진행 상태

| # | 단계 | 상태 |
|---|---|---|
| 1 | `minitron/` 폴더와 확정 계획서 | 완료 (PR #89) |
| 2 | teacher 진단 G0-T (이산 궤적 토큰 NLL, FM 손실; train vs val) | 완료 — (a) PASS, (b) 차이 3% 안팎 |
| 3 | hidden 수술 코드 + G0a–c | 완료 — G0a·G0c PASS, G0b는 허용치 초과를 사용자가 수용 |
| 4 | 증류 캐시 보충 (약 9,200 샘플) | 보정 1,024개 완료, 나머지 진행 중 |
| 5 | 중요도 측정 (1,024 샘플) | 완료 — `outputs/minitron/slim_mtw24` |
| 6 | MT-W zs 평가 | 완료 — test500 minADE@6 1.413 (dense 0.842, MoOP 0.950), CoC 퇴화 24.2% |
| 7 | `train_kd.py` smoke (G0d, GPU당 배치) | 이 박스에서 통과 (GPU당 1샘플); A100에서 배치 크기 결정 필요 |
| 8 | MT-W + KD 학습 (NEURON A100 × 4) | |
| 9 | 평가 (test500 / val500 / OOD-Val262, LingoQA, 지연시간) | |
| 10 | 리포트 | |
