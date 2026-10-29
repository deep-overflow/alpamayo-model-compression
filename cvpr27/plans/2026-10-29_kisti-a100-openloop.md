# KISTI(NEURON) A100 개루프 평가 트랙: 게이트 런과 기준 arm 재측정

날짜: 2026-10-29. 상태: **승인됨(사용자 결정 2026-10-29), 구현 중.** 브랜치 `worktree-kisti-a100-eval`. 위치는 `cvpr27/`(2026-10-29 규칙)이다.

## 1. 목적

앞으로 개루프 평가를 KISTI A100에서 돌린다. 이번 범위는 두 가지다.
1. A100 결과를 믿고 써도 되는지 확인하는 게이트 런
2. A100 전용 비교표의 기준 arm을 다시 재는 일

## 2. 근거: 하드웨어는 arm의 일부다

2026-10-29에 기존 런에서 확인했다. 비교는 같은 체크포인트끼리이고, 프로토콜은 k=8, 클립별 seed 42, max_gen 256,
결정론 모드로 같다. 수치는 minADE@6(m)이다.

| arm | 세트 | Ada | A100 | 차이 | CoC 문장 일치율 |
|---|---|---|---|---|---|
| dense | test500 | 0.842 | 0.844 | +0.003 | 97.6% |
| dense | OOD-Val262 | 1.000 | 1.001 | +0.001 | 96.6% |
| MoOP (`slim_dual_u40_v2`) | test500 | 0.950 | 0.960 | +0.010 | 95.8% |
| MoOP | OOD-Val262 | 1.119 | 1.101 | −0.018 | 95.0% |

- **같은 카드 재실행:** 100% 비트 동일이다. 카드 종류가 다르면 비트 동일 클립은 0%다.
- **LingoQA도 움직인다:** dense가 Ada 73.2%, A100 72.6%(9월)였다.
- **결론:** 한 비교표의 arm은 모두 같은 종류의 GPU에서 잰다. A100 행은 Ada 표와 섞지 않고 별도 블록으로 보고한다.

## 3. 범위 (사용자 결정 2026-10-29)

### arm

| 이름 | 모델 경로 | NEURON에 필요한 것 |
|---|---|---|
| `dense` | `baseline` | 기본 가중치(rev `7aba829`) |
| `moop` | `outputs/slim_dual_u40_v2` | `slim_meta.json`(선택 전용, 기본 가중치에서 재구성) |
| `moop_l0keep` | `outputs/slim_dual_u40_l0keep_v2` | `slim_meta.json`(선택 전용, `make_slim.py --config dual_u40_l0keep_v2`) |

MT-W zs와 R0는 이번에 제외한다.

### 세트

| 세트 | 호출 | 클립 수 |
|---|---|---|
| test_2000 | `run_baseline.py --set test --n-indist 2000 --no-tf` | 2,000 |
| OOD-Val262 | `run_baseline.py --set ood --manifest ood_val --no-tf` | 262 |
| LingoQA | `run_lingo_vqa.py` → `score_lingo_vqa.py`(표준 프로토콜, Lingo-Judge) | 500문항 |

val500(체크포인트 선택)은 이번에 하지 않는다.

- **지표:** minADE@6과 minFDE@6의 평균(1.6 / 3.2 / 6.4 s), CoC 퇴화율, LingoQA 정확도.
- **보고 방식:** 각 arm의 실제 평균을 먼저 쓰고, dense와 짝지은 Δ와 95% 부트스트랩 CI(20,000회)를 덧붙인다.

### 이름 규칙

`--exp-id cvpr27/a100/<arm>_<set>` 형식이다(예: `outputs/cvpr27/a100/moop_l0keep_test2000/`). LingoQA는 `cvpr27/a100/lingo_vqa_<arm>`,
`cvpr27/a100/lingo_vqa_scores_<arm>`이다. GPU 종류는 경로의 첫 단에만 쓴다. `cvpr27/a100/` 아래에 두므로 `collect_results.py`의
Ada 시트에도 섞이지 않는다.

## 4. 실행 방식: 4장을 붙잡아 두는 잡 + 작업 큐 (사용자 제안 2026-10-29)

`eval_a100.sbatch`는 A100 4장짜리 잡 하나로 시작해, 카드마다 워커 하나를 띄운다.
워커는 `/scratch/<user>/outputs/cvpr27/a100/queue/tasks.txt`에서 한 줄씩 꺼내 실행한다.
작업 줄은 다음 두 형식이다.

```
ol <arm> <set> <shard> <n_shards> [rerun]
lingo <arm>
```

- **작업 추가:** 잡이 도는 중에 `a100_queue.sh <stage>`가 다음 작업을 큐에 붙인다(Datamover에서 실행 가능).
  그래서 게이트 결과를 본 뒤 다음 단계를 넣을 때 SLURM 대기열을 다시 타지 않는다.
- **선점:** 줄 번호로 `queue/claimed/<n>` 디렉토리를 만들어 작업을 선점한다(`mkdir`은 Lustre에서 노드를 넘어서도
  원자적이다). 선점 디렉토리에는 소유 잡 ID를 남긴다.
- **홀더 여러 개(사용자 요청 2026-10-29):** 백필을 노리는 작은 잡(예: GPU 2장, 3시간)이 같은 큐를 함께 읽을 수 있다.
  잡은 시작할 때와 대기 중 10분마다 `squeue`로 확인해, 이미 끝난 잡의 미완료 선점만 풀어 다시 돌린다.
  시간 상한까지 75분이 안 남으면 새 작업을 집지 않는다(가장 긴 작업은 test_2000 shard 하나, 약 1시간).
- **자동 종료:** 집을 작업이 없고 그 잡이 아무것도 돌리지 않은 상태가 `IDLE_MAX`(기본 45분) 이어지면 끝난다.
  `queue/STOP`을 만들면, 그보다 먼저 시작한 잡들이 돌던 작업을 마치고 끝난다.
- **건너뛰기:** 출력 JSON이 이미 있는 작업은 건너뛴다. 같은 단계를 다시 넣어도 안전하다.
- **환경 기록:** 시작할 때 `queue/env_<jobid>.txt`에 G0용 정보를 남긴다
  (노드, `nvidia-smi -L`, 드라이버, torch·CUDA 버전, `alpamayo1_5` direct_url).
- **`salloc`을 쓰지 않는 이유:** 대화형 할당은 상한이 8시간이고, OTP 소켓이 끊기면 할당도 같이 죽는다.
- **대가:** 잡이 도는 동안 계정 상한 4장을 다 쓰므로 KISTI 학습 잡과 동시에 돌 수 없다.
  단계 사이의 빈 시간도 사용량으로 잡힌다.
- **시간 상한:** 4장 잡은 `--time=07:00:00`, 백필용 2장 잡은 `--time=03:00:00`으로 제출한다. 짧게 잡아야 백필이 잘 붙는다.

## 5. 전송 (`push_neuron.sh <user> a100eval lingoqa weights`)

| 항목 | 크기 | 비고 |
|---|---|---|
| test_2000 캐시(`test` 네임스페이스) | 11.2 GB | `--ignore-existing`, 8월 test_500(2.8 GB)이 남아 있으면 그만큼 빠짐 |
| OOD-Val 캐시(`ood` 네임스페이스 262개) | 1.6 GB | `--ignore-existing` |
| 매니페스트 | <1 MB | `eval_sets/test_2000.parquet`, `ood_val.parquet` |
| recipe 2개 | 6.7 MB | `slim_dual_u40_v2`, `slim_dual_u40_l0keep_v2`의 `slim_meta.json` + `config.json` |
| 새 스크립트 2개 | <20 KB | `eval_a100.sbatch`, `a100_queue.sh` (`a100_pull.sh`는 cvlab21에서만 씀) |
| LingoQA val + benchmark + Lingo-Judge | 약 1.9 GB | 기존 `lingoqa` 티어 |
| 기본 가중치 | 약 22 GB | 기존 `weights` 티어, 이미 있으면 크기·시각 비교로 건너뜀 |

- **전체 코드 트리는 보내지 않는다.** NEURON의 코드는 10-27 kd2 전송분(minitron 브랜치 작업 트리)이고 KD 잡이 그것에 기댄다.
  `run_baseline.py`와 LingoQA 코드는 main과 minitron 브랜치가 같아 그대로 쓴다.
- **검증:** 파일 수, `du --apparent-size`, 큰 파일 sha256으로 확인한다.
- **필요한 사용자 조작:** OTP 소켓 두 개(`! ssh -fN neuron-dm`, `! ssh -fN neuron`).

## 6. 단계와 예산

**클립당 시간:** 9월 실측(TF 포함)이 test 6.8 s, OOD 11.4 s다. 이번에는 `--no-tf`라 이보다 짧을 것이다.

| 단계 | 큐에 넣는 작업 | 4카드 wall(상한 추정) |
|---|---|---|
| `gate` | dense test_2000 4 shard, dense test_2000 shard 0 재실행, dense OOD-Val 2 shard, LingoQA dense | 약 2 h |
| (게이트 판정) | cvlab21로 결과를 받아 `analyze_a100.py --stage gate` | 수 분 |
| `refs` | {moop, moop_l0keep} × (test_2000 4 shard + OOD-Val 2 shard), LingoQA 2 arm | 약 2.5 h |

dense의 OOD-Val과 LingoQA는 게이트 판정과 상관없이 표에 필요하다. 그래서 게이트 단계의 빈 카드에 함께 넣는다.

## 7. 판정 (사전 등록)

- **G0 환경:** `config.json`과 `env_<jobid>.txt`에서 다음을 확인한다. 어긋나면 중단한다.
  - `gpu`가 "NVIDIA A100-SXM4-80GB"
  - `model_revision`이 `7aba829…`
  - `alpamayo1_5`가 커밋 `f42e594`
- **G1 A100 결정론:** shard 0 재실행이 원 실행과 500/500 비트 동일해야 한다(궤적 8개, 구간별 배열, CoC 문장).
  실패하면 트랙 전체를 멈춘다.
- **G2 버전 일치:** A100 dense test_2000을 Ada dense(0.878 / 2.393)와 비교한다.
  기준은 minADE@6 평균 차이 ±0.005 m 이내, CoC 문장 일치율 95% 이상이다. 실패하면 중단한다.
- **G3 일관성(보고만):** A100의 MoOP − dense가 Ada(test_2000 +0.075)와 부호가 같은지 본다.

## 8. 코드

| 파일 | 역할 |
|---|---|
| `experiments/transfer/push_neuron.sh` | 새 티어 `a100eval`(§5). 기존 티어는 바꾸지 않음 |
| `cvpr27/code/eval_a100.sbatch` | 홀더 잡(카드별 워커), `mkdir` 선점 큐, 여러 잡 공유, 유휴·시간 상한 처리 |
| `cvpr27/code/a100_queue.sh` | `gate` / `refs` 단계 작업 추가(긴 작업 먼저, 중복 방지), `status`, `stop` |
| `cvpr27/code/a100_pull.sh` | `outputs/cvpr27/a100/`를 cvlab21로 회수 |
| `cvpr27/code/analyze_a100.py` | `--stage gate`(G0–G2), `--stage table`(A100 표·그림). 출력은 `outputs/cvpr27/a100/{gate,table}/` |

## 9. 하지 않는 것

- Ada에서 잰 기존 표의 수정이나 대체
- 폐루프(AlpaSim)
- teacher-forcing 조건, val500
