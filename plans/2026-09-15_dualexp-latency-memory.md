# dual + expert MLP 절단 사다리 — 무압축 Alpamayo 대비 latency·memory (Ada, 릴리스 경로 + fast path)

날짜: 2026-09-15. 브랜치: `worktree-worktree-dualexp-latency`. 상태: **승인 대기**.
사용자 결정(2026-09-15): 하드웨어 **Ada만**(cvlab21 4–7이 빌 때까지 대기), 경로 **릴리스 + CUDA-graph
fast path 둘 다**, arm **5개**(baseline / dual_u40_v2 / dualexp em75 / em87p5 / em93p75).

## 질문

`slim_dualexp_u40_em93p75`(VLM `dual_u40_v2` 24.0% + expert MLP 8256→516; 11.08B → 6.71B,
−39.4%)가 무압축 `nvidia/Alpamayo-1.5-10B` 대비 한 번의 추론에서 **얼마나 빠르고 얼마나 적은
메모리를 쓰는가** — 단계별(ViT / VLM prefill / CoC decode / expert denoise)로, 릴리스 경로와
graphed fast path 각각에서. 그리고 expert MLP 비율 사다리(75 → 87.5 → 93.75%)로 그 이득이 expert
가중치 바이트에 **비례하는지**(대역폭 지배) 아니면 **무관한지**(step-bound).

## 배경 (이 레포가 이미 아는 것)

- `profile_stages.py`: 릴리스 추론 경로를 CUDA 이벤트로 단계 타이밍, rollout/denoise peak
  `max_memory_allocated`, KV 캐시 MB, 파라미터 수 기록. `bench_fastpipeline.py`: 같은 클립에서
  stock 경로와 graphed fast path(eager ViT+prefill → graphed decode 루프 → graphed expert
  denoise)를 나란히 타이밍. 둘 다 `notebooks/clip_ids.parquet` seed 42 순열의 클립을
  `load_physical_aiavdataset(clip_id, t0=5.1 s)`로 라이브 로드(cvlab21 호스트에서 검증된 경로).
- 선행 결과(Blackwell, `analyze_slim.py`, decode는 baseline 중앙 CoC 길이 16토큰으로 정규화):
  baseline 1346 ms / peak 21.5 G; identity(무압축·gather 경로) 1433 ms(**경로 대가 +87 ms**:
  prefill +8.6, dec/tok +3.25, expert +13.0); slim_int(−3.25B) 1312 ms = 1.03×, peak 15.4 G.
  **expert 절단은 expert 단계를 거의 안 줄였다**(378→377 ms) — expert는 FLOPs 5.5%인데
  wall-clock 22–28%, 10회 순차 패스에 매번 가중치를 읽는 step-bound(CLAUDE.md). 단 slim_int의
  expert 절단은 graded 30/50이고 이번 사다리의 끝은 expert 2.30B → 0.59B(가중치 바이트 −74%).
- fast path 선행(Blackwell): slim_int stock 1322 → fast 871 ms(1.52×), decode/tok 35.9 → 18.4.
- 메모리는 산술로 예측된다(bf16 가중치): 무압축 22.16 GB, dual 16.84(−5.32), em75 14.10(−8.06),
  em87p5 13.65(−8.51), em93p75 13.42(−8.74). slim_int는 −6.5 GB 예측에 peak −6.1 G 실측.
- `profile_stages.py`·`bench_fastpipeline.py`·`analyze_slim.py`는 `REPO=/workspace/...`
  하드코딩(CLAUDE.md "stale paths") — 호스트에서 돌리려면 `__file__` 기준으로 바꿔야 한다.

## 가설 (사전 등록)

- **H1 (메모리)**: 각 arm의 rollout peak 감소가 가중치 산술(위)의 **±1 GB 안**. KV 캐시 MB는 5 arm
  동일(k/v_proj·8×128 인터페이스 불변). 반증: 감소가 산술보다 2 GB 이상 작으면 gather 경로의
  활성화가 이득을 잠식.
- **H2 (expert 단계, 사다리)**: expert 가중치 바이트가 −53 / −70 / −74%(em75/87.5/93.75, dual 대비)
  인데 expert denoise 시간은 dual 대비 **−15% 미만이고 사다리 간 차이 없음**(step-bound).
  반증: em93p75가 −40% 이상 줄고 사다리가 단조면 대역폭 지배.
- **H3 (VLM 단계)**: `dual / baseline` prefill 0.70–0.85(VLM −24% 이득에서 경로 대가 ~9 ms를
  뺀 것), dec/tok +0~15%(런치 지배 + gather 대가). 릴리스 경로 end-to-end는 **1.0–1.15×**.
- **H4 (fast path)**: graphed 경로에서 decode/tok이 절반이 되어 `dualexp/baseline` 비가 릴리스
  경로보다 커진다 — 단 그 이득의 대부분은 CUDA graph 자체(baseline도 받는다)이므로 **같은 경로
  안의 비교**(fast-vs-fast)로 읽는다.

## 설계

### arm (5)

| arm | 체크포인트 | 파라미터 | 가중치 bf16 | 역할 |
|---|---|---:|---:|---|
| `baseline` | HF 무압축, stock 경로 | 11.079B | 22.16 GB | 기준 |
| `dual` | `slim_dual_u40_v2` (VLM −24.0%) | 8.421B | 16.84 GB | VLM 절단(+경로 대가) |
| `em75` | `slim_dualexp_u40_em75` | 7.052B | 14.10 GB | expert MLP 사다리 1 |
| `em87p5` | `slim_dualexp_u40_em87p5` | 6.823B | 13.65 GB | 사다리 2 |
| `em93p75` | `slim_dualexp_u40_em93p75` | 6.709B | 13.42 GB | **대상**, 사다리 3 |

identity(무압축·gather 경로)는 사용자 결정으로 제외 — 경로 대가는 선행 +87 ms(Blackwell)를
참고치로만 인용하고, `dual − baseline`은 "VLM 절단 + 경로 대가"의 합으로 읽는다.
`dualexp_emX − dual`이 expert MLP 절단의 순효과.

### 프로토콜

- 하드웨어: **cvlab21 Ada 4–7**(폐루프 드라이버 카드). 빈 카드(<100 MiB)만, 카드가 날 때까지
  60 s 폴링 대기(`launch_strat_calib.sh`의 pool 로직 재사용). 같은 박스 동시 실행은 선행과 같은
  조건(arm 간 공정, 절대값은 보수적).
- 릴리스 경로: `profile_stages.py --gpu N --num-clips 12 --warmup 1 --clip-offset {0,13}
  [--slim-ckpt outputs/slim_<arm>]`, FLOPs는 arm당 shard 0만. → arm당 24클립.
- fast path: `PYTORCH_CUDA_ALLOC_CONF= … bench_fastpipeline.py --gpu N --num-clips 12
  --reserve-gb 8 [--slim-ckpt …]` (graph capture에 expandable_segments 금지), arm당 1 run(선행
  관례, 12클립 + warmup). peak 메모리 기록 한 줄 추가.
- 공통: 배치 1, bf16, sdpa, `max_gen 256`, seed 42.
- 정규화: decode는 baseline의 중앙 CoC 길이(REF 16토큰)로, 두 경로 모두.

### 코드

- `profile_stages.py`, `bench_fastpipeline.py`: `REPO = Path(__file__).resolve().parents[2]`.
  `bench_fastpipeline.py`에 `peak_gb`(`max_memory_allocated`) 기록 추가. 그 외 불변.
- `launch_profile_arms.sh`(신규): 15 job(5 arm × {profile s0, profile s13, fastpipe}) 큐, K워커,
  빈 Ada 카드 claim/release, `run_retry_host.sh` 경유(`ALPAMAYO_REPO` = 이 워크트리).
- `analyze_profile_arms.py`(신규, `analyze_slim.py`의 `stages()` 로직 이식): `--arms
  name=run_s0,run_s13 --fast name=run …`. 산출: 릴리스 단계표(정규화 total·speedup), 메모리표
  (파라미터·가중치 GB·rollout/denoise peak·KV MB·산술 예측 대비), fast-path 표(stock/fast, 정규화),
  사다리·귀속표(dual−baseline, emX−dual), 스택 플롯 2장. `outputs/profile_dualexp/`.
- 보고서 `reports/evaluation/2026-09-15_dualexp-latency-memory.html`
  (`head_analysis/profile_dualexp_report_template.html`).

### 산출물 이름

`profile_{base,dual,em75,em87p5,em93p75}_ada2_s{0,13}`, `fastpipe_{…}_ada2`.

## 게이트 (사전 등록)

- **G0**: 15 run 전부 완료(profile 24/24 클립, fastpipe 12/12), GPU 이름 전부 "RTX 5880 Ada",
  shard별 clip_ids 동일, 각 arm 파라미터 수가 `config.json`의 `params.slim`과 정확히 일치.
- **G1 (H1)**: 각 arm의 (baseline rollout peak − arm rollout peak)가 가중치 산술 ±1 GB 안,
  KV MB 5 arm 동일(±1%).
- **G2 (H2)**: expert 단계 중앙값 `em93p75 / dual` > 0.85이고 세 사다리 값의 범위 < 10% → step-bound
  채택; `em93p75 / dual` < 0.60이고 단조 → 대역폭 지배. 24클립 페어드(같은 클립·seed) 부트스트랩 CI.
- **G3 (H3)**: `dual / baseline` prefill ∈ [0.70, 0.85], dec/tok ∈ [1.00, 1.15]; 릴리스 e2e
  `em93p75 / baseline` 보고(예측 0.87–1.0 = 1.0–1.15×).
- **G4 (H4)**: fast path에서 fast e2e(16토큰 정규화) `em93p75 / baseline` 보고; stock→fast 이득이
  5 arm에서 같은 크기인지(CUDA graph 이득은 arm 무관이어야 함).

## 비용

profile run ~3분(FLOPs run ~6분), fastpipe run ~15분(capture 포함). 15 run을 Ada 4장에 풀로 →
**~40분**(카드가 다 비었을 때). 디스크 0(em93p75·dual은 `--no-state` 재구성, em87p5·em75는 state
있음 — 둘 다 `load_slim`이 같은 결과). 대기 시간은 Ada 4–7 점유(현재 다른 작업 27 GB × 4)에 달림.

## 한계

- 배치 1 단일 추론의 latency이지 처리량이 아니다.
- identity가 없어 경로 대가를 이번 측정 안에서 분리하지 못한다(선행 Blackwell +87 ms 참고).
- 4 run 동시 실행의 CPU·PCIe 공유(선행과 동일 조건).
- CoC 길이가 arm마다 다르므로 정규화 없는 wall-clock은 비교하지 않는다.
