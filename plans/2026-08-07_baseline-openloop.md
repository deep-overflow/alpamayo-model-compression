# Baseline open-loop 평가: in-distribution vs OOD

## 목적

Alpamayo-1.5-10B(무손상)의 open-loop 성능을 **두 평가셋에서 따로** 측정해 기준선을 확정한다.
이후 모든 pruning config는 이 숫자와의 페어드 델타로 보고된다. 지금은 비교가 아니라
**기준선 확정**이 목적이므로 config는 baseline 하나뿐이다.

두 셋을 분리하는 이유: 압축이 in-distribution 성능은 유지하면서 롱테일에서만 무너지는지가
핵심 질문인데, 섞어서 평균 내면 그 구분이 사라진다.

## 평가셋 정의 (사전 등록)

### A. In-distribution

- 풀: 공식 `val` split ∩ `pre_processed/eval` (18,868클립)
- **제외**: `ood_reasoning.parquet`의 1,740개 전부 + `split.json`의 calib 50개
  (OOD 290개만이 아니라 1,740개 전부를 빼야 두 셋이 완전 배타적)
- 추출: **N=500**, full-val 클립 분포를 타깃으로 greedy 매칭
  (country 4.0 / platform_class 2.0 / time_of_day 2.0 / season 1.5 / month 1.0 / radar_config 1.0)
- t0 = 5,100,000 µs (캐시 고정값, 기존 러너와 동일)
- 품질 보고: 속성별 L1 / JSD를 `quality.csv`로 저장 — 청크 단계(200청크, 가중 L1 0.0144)에
  이어지는 2단계 매칭이므로 두 단계 모두 수치를 남긴다

### B. OOD

- 풀: `pre_processed/ood` 1,533개 (1,740 중 events NaN 9개, 이벤트가 클립 경계 밖 198개 제외)
- **primary: split=val 262개**, secondary: split=train 1,271개
  (우리는 학습을 하지 않으므로 train 소속도 zero-shot 평가로 유효하지만, 보고는 분리)
- t0 = 클립별 `event_start_timestamp` (OOD 상황이 발생하는 시점)
- 9개 `event_cluster`별로 분해 보고

## 측정 항목

| 항목 | in-dist | OOD |
|---|---|---|
| minADE / minFDE @ K=8 (자체 롤아웃 CoC 조건) | ✅ | ✅ |
| minADE / minFDE (GT CoC teacher-forcing 조건) | — (GT 없음) | ✅ |
| CoC NLL (GT CoC 기준) | — | ✅ |
| CoC self-NLL (자체 생성 CoC) | ✅ | ✅ |
| 시나리오 버킷 (decel_stop / turn / accel / cruise) | ✅ | ✅ |
| CoC 퇴화율 (`coc_degenerate`) | ✅ | ✅ |
| 생성 CoC 텍스트 저장 (정성 비교·향후 LLM judge용) | ✅ | ✅ |

in-dist에 GT CoC가 없으므로 "추론 품질"은 OOD에서만 직접 측정된다. 이것이 축 4(Driving QA)가
필요한 이유이고, OOD의 `gt_coc` vs `gen_coc` 쌍이 그 입력이 된다.

## 시드 (사전 등록)

- CoC 롤아웃: `torch.manual_seed(42 + i)` — `run_rollout`이 `do_sample=True, temp 0.6, top_p 0.98`로
  확률적이므로 호출 직전 고정 필수
- 궤적 K개: `seed = 42 + i*100 + k`, k=0..7
- 두 평가셋 모두 동일 규칙. 이후 pruning config가 같은 시드를 재사용해 페어드 비교가 성립한다

## 구현

`experiments/evaluation/`에 새로 작성:

- `make_eval_sets.py` — 위 정의대로 두 매니페스트 + `quality.csv` 생성
- `run_baseline.py` — 캐시(`sample_cache.load_cached`)에서 로드해 baseline 평가.
  `--set indist|ood`, `--shard`로 분할 실행
- `analyze_baseline.py` — 버킷별·cluster별 집계, 부트스트랩 CI, 플롯

`head_analysis`의 `analysis_lib`(프롬프트 구성/스팬), `eval_lib`(minADE, 버킷, CI, 퇴화 판정)는
재사용하되 import만 하고 수정하지 않는다.

## 비용 추정

클립당 롤아웃 + 8 denoise ≈ 20~30초 (GPU 1장). in-dist 500 + OOD 1,533 = 2,033클립 →
**약 11~17시간**. `--shard`로 카드 4장에 나누면 3~4시간. 캐시 로드가 0.37초라 데이터 I/O는
병목이 아니다.

## 검증

1. 스모크: 각 셋 5클립으로 형상·NaN·CoC 길이 확인
2. 재현성: 같은 시드로 2회 실행 시 minADE 완전 일치
3. 기존 결과와의 정합: in-dist baseline minADE가 기존 `jsweep` baseline(0.891, 80클립)과
   같은 자릿수인지 — 크게 다르면 캐시 경로나 프롬프트 구성에 문제가 있다는 신호
4. ruff 통과

## Risks

| Risk | Mitigation |
|---|---|
| JPEG 재인코딩이 성능에 영향 | 검증 3번이 1차 감지. 필요시 직접 로드와 20클립 비교 |
| OOD t0가 이벤트 시점이라 ego history가 짧을 수 있음 | 빌드 단계에서 이미 걸러짐(198개 제외) |
| GPU 점유 | `run_retry_host.sh`로 큐잉 |
| in-dist N=500이 부족 | 페어드 델타 기준이므로 500이면 충분. 부족하면 매니페스트 확장만 하면 됨 |
