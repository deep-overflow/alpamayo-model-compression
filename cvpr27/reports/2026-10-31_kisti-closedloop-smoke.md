# Experiment: KISTI NEURON 폐루프 스모크 (dense, 장면 1개)

2026-10-31 · 모델 `nvidia/Alpamayo-1.5-10B`(rev `7aba829`) · KISTI NEURON A100-SXM4-80GB (gpu40)
계획 `cvpr27/plans/2026-10-31_kisti-closedloop-smoke.md` · 코드 `cvpr27/code/alpasim_neuron/` · 결과 `outputs/cvpr27/kisti_closedloop_smoke/`

## Purpose
- NEURON에는 docker가 없다. AlpaSim 폐루프가 AlpaSim 자체의 SLURM 모드(`wizard.run_method=SLURM`)로 NEURON A100에서
  끝까지 도는지 확인한다. 서비스마다 `srun --overlap --container-image=<sqsh>`(pyxis + enroot)로 띄우는 방식이다.
- 측정이 아니라 동작 확인이다(장면 1개, rollout 1회).

## Setup
- **하네스:** 로컬 s600 실행과 같은 AlpaSim 트리(인계 제동 수정, ASL 이미지 생략, slim 로더 포함)와 같은 override.
  `topology=pair_a15 driver=alpamayo1_5`, 모든 서비스가 잡의 A100 1장에 올라감, 드라이버 `HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8`,
  런타임 `ALPASIM_ASL_SKIP_IMAGES=1`, `n_rollouts=1`, 영상 없음. 생성된 driver/eval/controller 설정은 로컬 s600과 바이트 단위로 같다.
- **장면:** `clipgt-01d503d4-449b-46fc-8d78-9085e70d3554`(public_2601, uuid `4f6bf6bb…`), 로컬 s600 sh0의 첫 장면.
- **이미지:** `alpasim_base_0.111.0.sqsh` 11 GB, `nre_ga_26.04.sqsh` 12 GB(로컬에서 `enroot import dockerd://`, zstd 3, 양쪽 sha256 일치).
- **잡:** `956432`, 1 GPU · CPU 8 · 120 GB · 2 h 한도.

### 돌리기까지 고친 것
| 문제 | 원인 | 수정 |
|---|---|---|
| `enroot import` 실패(`mkdir rootfs/app/run.runfiles: File exists`) | nre-ga의 WorkingDir가 절대 심볼릭 링크(`/app/run.runfiles`)를 지나감. enroot가 호스트에서 `mkdir -p rootfs<WorkingDir>`를 하다 링크를 따라감. 레이어 평탄화로는 안 풀림(확인) | `FROM nre-ga:26.04` + `WORKDIR`를 같은 디렉터리의 실제 경로로(`make_nre_sqsh.sh`, 메타데이터만 변경) |
| 1 GPU에 CPU 16 요청 거절 | `amd_a100nv_8`은 GPU당 8코어까지 | `--cpus-per-task=8` |
| 드라이버가 모델을 못 찾음 (1차, 잡 956427) | pyxis가 기본으로 홈을 마운트하고, root 리매핑 시 enroot가 홈을 `/root`에 덮어 `/root/.cache/huggingface` 볼륨을 가림 | AlpaSim `deployment/slurm.py`(SLURM 경로만)에 `--no-container-mount-home` 추가. 컨테이너의 `~/.cache` 쓰기가 NEURON 홈(파일 10만 개 한도)으로 가는 것도 막음 |
| 드라이버가 모델을 못 찾음 (2차, 잡 956430) | 서비스는 root 리매핑 없이 사용자로 실행되어(`remap_root` 기본 false) HOME이 `/root`가 아님 | 드라이버 환경에 `HOME=/root`, `HF_HOME=/root/.cache/huggingface` |

## Results (Facts)
- **통과:** 잡 `COMPLETED`, wizard 종료 코드 0(17:43:37 KST), `rollouts/<scene>/<id>/metrics.parquet`과 `_complete` 생성, 집계 완료.
- **소요:** 컨테이너 기동 17:34:08 → 모든 포트 열림 17:36:21 → 세션 시작 17:37:02 → 종료 17:43:37.
  드라이버 모델 로딩 약 80초, 첫 렌더는 커널 JIT로 약 50초, 이후 시뮬레이션 0.1 s 한 스텝당 약 1.6–1.9초.
- **렌더러:** NuRec(nre-ga 26.04)이 A100에서 커널을 JIT 컴파일한 뒤 정상 렌더. 4카메라 렌더 요청 198회, 렌더러·드라이버 로그 에러 0, 검은 프레임 0.
- **드라이버:** 180 스텝 모두 CoC 생성(예: "Keep distance to the lead vehicle since it is moving slowly ahead" 151회).

같은 장면의 로컬 dense rollout과 나란히 (AlpaSim은 seed를 고정하지 않고 GPU도 달라 비트 일치는 기대하지 않음):

| 지표 (rollout 1회) | 로컬 s600 (Ada/Blackwell) | NEURON A100 |
|---|---|---|
| 시뮬레이션 길이 | 19.9 s | 19.9 s |
| 충돌 / 이탈 / 역차선 | 0 / 0 / 0 | 0 / 0 / 0 |
| progress | 1.000 | 1.000 |
| 주행 거리 (GT 73.8 m) | 89.8 m | 90.5 m |
| plan_deviation | 0.164 | 0.190 |
| 장애물 최소 거리 | 0.85 m | 0.96 m |
| 검은 프레임 | 0 | 0 |

## Analysis (Opinion)
- **NEURON에서 폐루프는 돌아간다.** docker 없이 AlpaSim의 SLURM 모드로 로컬과 같은 하네스·설정이 그대로 실행되고, 결과도 로컬 rollout과 같은 범위다.
  가장 컸던 불확실성(RT 코어·NVENC가 없는 A100에서 NuRec 렌더러)은 해소됐다.
- **본 실행 전에 정할 것:**
  1. **하드웨어 블록:** A100 폐루프 점수는 기존 Ada/Blackwell 폐루프 표와 섞을 수 없다. 비교할 arm을 모두 A100에서 다시 돌려야 한다.
  2. **처리량 (대략):** 장면당 rollout 1회에 약 6–7분(1 GPU, 동시 rollout 1개). pair_a15는 동시 rollout 2개를 허용하므로
     s600 × 1이면 arm당 약 30–35 A100-시간, 계정 한도 4장으로 arm당 8–9시간 정도로 본다(측정 아님, 동시 2개 효율 미확인).
     GPU당 CPU 8코어 제한이 동시 실행에서 병목이 될 수 있다.
  3. **장면 전송:** 장면당 약 2 GB. s600이면 약 1.2 TB를 Datamover로 보내거나 계산 노드에서 HF로 직접 받는다.
  4. **잡 구성:** 한 잡에 여러 장면 shard를 돌리려면 shard마다 `wizard.baseport`를 달리해야 한다(같은 노드에서 포트 충돌).
- **다음 단계 제안:** dense와 MoOP를 같은 장면 소수(예: 8개 × 2 rollout)로 돌려 처리량과 동시 실행 효율을 실측한 뒤 본 실행 규모를 정한다.
