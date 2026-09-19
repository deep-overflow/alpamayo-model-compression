# graphed fast path 재측정 — 한 카드 고정, 24클립 × 5라운드

날짜: 2026-09-19. 브랜치: `worktree-fastpath-rounds`. 상태: **승인 대기**.
사용자 결정(2026-09-19): 경로 **graphed fast path만**, **한 Ada 카드에서 5 arm 순차 실행**,
규모 **arm당 24클립 × 5라운드**(총 25 run). 선행: `plans/2026-09-15_dualexp-latency-memory.md`,
`outputs/profile_dualexp/`, `reports/evaluation/2026-09-16_dualexp-latency-memory.html`.

## 질문

선행 측정의 fast-path 표에서 **em 사다리(em75 / em87.5 / em93.75)의 e2e 순서가 단조가 아니다**
(976.0 / 964.1 / 984.5 ms). 이것이 실제 효과인가, 측정 구조의 산물인가 — 그리고 산물이라면
같은 카드·반복 run으로 걷어냈을 때 사다리의 실제 간격은 얼마이고 분해 가능한가.

## 배경 — 선행 데이터를 다시 읽어 확인한 것 (GPU 0)

- 선행 fast-path run은 arm당 12클립 × 1 run이고 **arm마다 다른 카드**에서 돌았다(빈 카드 풀:
  base 4·5, dual 7·6, em75 7·5, em87p5 6·7, em93p75 4·6 — profile shard 기준, fastpipe도 같은 풀).
- **run 안의 클립 간 흔들림은 이미 작다.** `prompt_len`이 전 클립 3086으로 같아(입력 shape 불변)
  클립은 사실상 같은 계산의 반복이다: fast denoise (max−min)/median **0.3–1.2%**, fast prefill
  1.4–3.7%(em75의 이상치 한 클립 564.6 ms 제외), fast dec/tok 4–8%.
- **오차는 run(카드) 사이에 있다.** dual / em75 / em87p5 / em93p75는 **VLM이 동일**하므로 prefill과
  dec/tok이 같아야 한다. dec/tok은 실제로 같다(19.49 / 19.41 / 19.41 / 19.41). 그런데 fast prefill
  중앙값은 505.6 / 505.5 / 499.7 / **525.0** — em93p75의 **최소값 520.7이 dual의 최대값 508.1보다
  크다**. 클립 표본이 아니라 run 단위의 체계적 오프셋(+4–5%, ≈20–25 ms)이다.
- e2e 비단조는 전부 이 오프셋이다: denoise만 보면 203.7 → 160.0 → 153.8 → 148.9로 단조이고,
  em87p5와 em93p75의 denoise 차(4.9 ms)보다 prefill 오프셋(25 ms)이 5배 크다.
- 따라서 **클립 수 확대만으로는 이 오차가 줄지 않는다.** 필요한 것은 (a) 카드 고정, (b) run 간
  분산의 직접 추정(라운드 반복)이다. 클립 12 → 24는 릴리스 경로와 수를 맞추는 의미.
- 클립 라이브 로드는 **25–42 s/클립**(실측, CPU)이고 payload ≈ 100 MB/클립. 캐시 없이 25 run ×
  25클립이면 로드만 ~6 h, 그동안 경합이 심한 Ada 카드를 놀린다.

## 가설 (사전 등록)

- **H1 (음성 대조 = 잡음 바닥)**: 같은 카드에서는 VLM이 같은 네 arm의 fast prefill·fast dec/tok이
  일치한다 — 라운드 평균의 쌍별 차이 |Δ| < 1%이고 CI가 0을 포함. 선행의 em93p75 prefill +4–5%는
  사라진다. 반증: 같은 카드에서도 em93p75 prefill이 > 2% 높게 **재현**되면 카드 탓이 아니라
  체크포인트/경로의 성질이므로 별도 조사 대상.
- **H2 (사다리, denoise)**: fast denoise는 dual > em75 > em87p5 > em93p75로 **엄격히 단조**이고
  인접 쌍 세 개의 CI가 전부 0을 배제한다(선행 203.7 / 160.0 / 153.8 / 148.9 ms).
- **H3 (사다리, e2e)**: H1이 성립하면 fast e2e(16토큰 정규화)도 expert 폭에 단조가 된다. em87p5 −
  em93p75 ≈ 5 ms(0.5%)가 분해되는지는 run 간 SD에 달렸다(분해하려면 SD ≲ 2 ms). 분해 안 되면
  "em87p5 ≈ em93p75"를 **이번엔 잡음 바닥 수치와 함께** 확정한다.
- **H4 (run 간 변동)**: 같은 카드·같은 arm의 5 run에서 e2e 중앙값의 CV < 1%.

## 설계

### arm (5) — 선행과 동일

`base`(HF 무압축) / `dual`(`slim_dual_u40_v2`) / `em75` / `em87p5` / `em93p75`
(`slim_dualexp_u40_em{75,87p5,93p75}`). 네 체크포인트 모두 `slim_state.pt` 존재 확인(2026-09-19).

### 프로토콜

- **카드**: cvlab21 Ada 4–7 중 **먼저 완전히 비는 한 장**(< 100 MiB)을 잡아 25 run 전부를 그
  카드에 고정. 매 run 직전에 그 카드가 다시 빈 상태인지 확인하고 아니면 대기.
- **순서**: 5 × 5 순환 라틴 방진 — 라운드 r의 순서는 arm 목록을 r칸 회전. 각 arm이 각 위치(1–5번째)를
  정확히 한 번씩 차지하므로 발열·시간 드리프트가 arm과 교락되지 않는다.
- **run**: `PYTORCH_CUDA_ALLOC_CONF= bench_fastpipeline.py --gpu N --num-clips 24 --clip-offset 0
  --reserve-gb 8 --clip-cache outputs/clip_cache_fastpipe [--slim-ckpt …]`. 클립 0은 warmup(+graph
  capture), 1–24가 live. 앞 12개 live 클립은 선행 run과 **동일 클립**(같은 seed 42 순열·offset 0).
  seed는 `42 + ci`로 라운드 간 동일 → 같은 아키텍처에서 CoC 길이가 같으므로 라운드는 순수한
  타이밍 반복이다.
- **공통**: 배치 1, bf16, sdpa, `max_gen 256`, cap 3456. stock 경로도 같은 run에서 공짜로 같이
  재지므로 부차 결과로 보고(주 결과는 fast).
- **동거 프로세스 감시**: run 동안 20 s마다 `nvidia-smi --query-compute-apps`를 기록, 우리 PID가
  아닌 compute 프로세스가 찍힌 run은 무효 처리하고 재실행.
- **메모리**: `peak_gb`(run 전체 `max_memory_allocated`)는 그대로 기록 — 결정론적이라 라운드 간
  동일해야 하며(G0에서 확인) 선행 값(22.49 / 17.45 / 14.90 / 14.47 / 14.26 GiB)과 대조.

### 코드

- `bench_fastpipeline.py`(수정, 작게):
  - `--clip-cache DIR`: `DIR/<clip_id>.pt`가 있으면 `torch.load`, 없으면 라이브 로드 후 `torch.save`.
    로드는 타이밍 구간 밖이라 측정에 영향 없음(입력 텐서 동일).
  - `--prefetch-only`: GPU 예약·모델 로드 **전에** 클립만 캐시에 받아 두고 종료(CPU 전용). 클립 선택
    로직을 한 곳에 두려고 별도 스크립트 대신 플래그로.
  - `config.json`에 `gpu_index`, `clip_ids` 추가(지금은 GPU 이름만 있어 카드 고정을 사후 검증 못 함).
- `launch_fastpath_rounds.sh`(신규): prefetch → 카드 1장 claim → 라틴 방진 순서로 25 run,
  완료된 run은 skip(재개 가능), 동거 감시 로그 `logs/<exp>.cotenant`. `run_retry_host.sh` 경유,
  `ALPAMAYO_REPO` = 이 워크트리(`.venv`·`outputs` 심볼릭 링크를 워크트리에 생성 — 둘 다 gitignore).
- `analyze_fastpath_rounds.py`(신규): `--arms name=prefix`로 `fastpipe_<arm>_ada1c_r{0..4}` 읽기.
  - run 내부: 24클립 중앙값(선행과 같은 정의, e2e는 16토큰 정규화).
  - 라운드 간: arm × 단계(prefill / dec·tok / denoise / e2e; fast·stock)별 5 run의 평균·SD·CV·범위.
  - arm 비교: 같은 (라운드, 클립)끼리 짝지은 비 a/b, **계층 부트스트랩**(라운드 재표집 → 클립
    재표집, 10,000회) 95% CI. 대상 쌍: 각 arm/base, emX/dual, 인접 사다리 쌍.
  - 음성 대조표(H1), 게이트 판정, 플롯 3장(단계별 라운드 점도표, 사다리 CI, 선행 12클립 값 대조).
  - `outputs/fastpath_rounds/{config.json, metrics.json, summary.txt, plots/}`.
- 보고서 `reports/evaluation/2026-09-19_fastpath-single-card-rounds.html`
  (`head_analysis/fastpath_rounds_report_template.html`, `build_report.py`로 플롯 인라인),
  `CLAUDE.md` 리포트 표에 한 줄, 선행 리포트의 fast-path 표에는 "후속 측정 참조" 한 줄.

### 산출물 이름

`fastpipe_{base,dual,em75,em87p5,em93p75}_ada1c_r{0,1,2,3,4}`(ada1c = Ada one card),
집계 `outputs/fastpath_rounds/`, 캐시 `outputs/clip_cache_fastpipe/`(~2.5 GB, 완료 후 삭제).

## 게이트 (사전 등록)

- **G0 (완결·위생)**: 25/25 run exit 0; 25개 `config.json`의 `gpu_index`·GPU 이름이 전부 동일;
  run마다 live ≥ 22클립이고 arm 안에서 live 클립 집합이 라운드 간 동일; 동거 compute 프로세스
  0건; arm별 `peak_gb`가 5 라운드에서 동일(±0.01 GiB); 같은 arm·클립의 `fast_steps`가 라운드 간 동일.
- **G1 (H1, 음성 대조)**: dual·em75·em87p5·em93p75의 fast prefill, fast dec/tok — 여섯 쌍 전부
  |비 − 1| < 1%이고 CI가 1을 포함 → PASS. 어느 쌍이든 > 2%로 CI가 1을 배제하면 FAIL(조사 대상으로
  보고, 그 단계가 섞인 e2e 비교는 해석 보류).
- **G2 (H2)**: fast denoise 인접 쌍 em75/dual, em87p5/em75, em93p75/em87p5의 CI 상한이 전부 < 1.
- **G3 (H3)**: fast e2e — em93p75/base, 그리고 em93p75/em87p5의 비와 CI를 보고. CI가 1을 배제하면
  "분해됨", 포함하면 "구분 불가(잡음 바닥 = G4의 CV)".
- **G4 (H4)**: arm별 e2e fast의 라운드 간 CV 보고, 전 arm < 1%이면 PASS.

## 비용

- prefetch: 25클립 × ~33 s ≈ **15분, CPU만**(카드를 잡기 전에 끝냄).
- run 1개: 모델 로드 ~2–3분 + graph capture ~1분 + 25클립 × 2경로 ≈ **5–6분**. 25 run ≈ **2–2.5 h**,
  Ada 1장. 첫 run의 실제 소요를 재서 보고.
- 대기: 2026-09-19 현재 Ada 4–7 전부 점유(24–28 GB씩, 다른 작업) → 카드가 빌 때까지 60 s 폴링.
- 디스크: 캐시 ~2.5 GB(`/mnt/nvme1n1` 여유 346 GB, 95%) — 완료 후 삭제. 새 체크포인트 없음.

## 한계

- 한 카드의 결과다 — 카드 간 절대값 차이는 이 실험이 **제거하는** 대상이지 재는 대상이 아니다.
- 배치 1 단일 추론 latency, 처리량 아님. 릴리스 경로(`profile_stages`)는 이번 범위 밖(사용자 결정).
- 라운드 5개는 run 간 SD의 추정으로는 작다(자유도 4) — CV는 점추정과 범위를 같이 보고.
- 같은 박스의 다른 카드에서 도는 남의 작업이 CPU·PCIe를 공유한다(선행과 같은 조건, 통제 불가).
