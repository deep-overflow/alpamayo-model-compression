# cvlab20 서버 사용법

연구실 공용 GPU 서버. cvlab21에서 SSH로 붙어 쓴다. **실제로 돌려보며 확인한 것만** 적는다.

최종 검증: 2026-09-05 — 이 프로젝트 전체를 이관하고 중요도 측정 4,000클립을 완주.

---

## 1. 한눈에

```bash
ssh cvlab20@cvlab20                       # Tailscale, 키 인증

# 실행 전 반드시 카드 확인 (공용 서버)
ssh cvlab20@cvlab20 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader'

ssh cvlab20@cvlab20 'bash -lc "
  cd /home/cvlab20/project/chan/alpamayo-model-compression
  . /home/cvlab20/project/chan/cvlab20-server/env.sh
  .venv/bin/python <스크립트> --gpu <비어있는 카드>
"'
```

## 2. 절대 규칙 — 무거운 것은 `/mnt/dataset1/chan`

| 무엇 | 어디에 |
|---|---|
| **코드만** | `/home/cvlab20/project/chan/` |
| **나머지 전부** — 데이터·캐시·모델·outputs·venv·로그 | **`/mnt/dataset1/chan/`** |

홈(`/`)은 879 GB인데 **2026-09-05 기준 26 GB만 남았다 (97%)**. 이 프로젝트를 이관하기
전에도 53 GB였고, 줄어든 27 GB는 다른 사람들 작업이다(홈 상위 소비자: `.cache/wandb` 9.9 GB,
`.cache/uv` 8.1 GB, `.cache/torch` 6.7 GB — 전부 우리 것이 아니다). **여기에 뭔가 더 쌓으면
서버 전체가 멈춘다.**

우리가 홈에 쓰는 것은 **104 MB**(repo 86 MB + LingoQA 9.6 MB + 이 스크립트들)뿐이다.

`/mnt/dataset1`은 7.0 TB 중 ~680 GB 여유(91%). 여기도 공용이니 큰 걸 올리기 전에
`df -h /mnt/dataset1`을 본다.

repo가 홈을 가리키는 지점은 **심링크로 우회**한다. 그래야 코드를 한 줄도 안 고치고 돌아간다:

```
<repo>/.venv    -> /mnt/dataset1/chan/venvs/alpamayo-mc
<repo>/outputs  -> /mnt/dataset1/chan/outputs
```

`build_venv_cvlab20.sh`가 둘 다 만들어 준다.

## 3. 접속

- `cvlab20`은 Tailscale 이름(100.94.229.67). cvlab21의 `~/.ssh/id_ed25519.pub`을
  2026-09-05에 등록해 키 인증이 된다.
- **계정 `cvlab20`은 공용이다.** `authorized_keys`에 연구실 13명의 키가 있다.
  남의 프로세스·파일을 건드리지 않고, 작업은 우리 디렉터리 안에서만 한다.

## 4. GPU

**RTX 5880 Ada × 8, 각 48 GB.** 드라이버 570.133.07 / CUDA 12.8 —
`torch 2.8.0+cu128`과 정확히 맞는다.

cvlab21의 평가용 카드(4–7번)와 **같은 아키텍처**다. 이 프로젝트의 결정론은 한 아키텍처
안에서만 성립하므로(`CLAUDE.md`), cvlab20은 *평가*를 옮겨도 커널 교락이 안 생기는 유일한
원격 후보다.

**2026-09-05 검증 완료 — 비트 동일하다.** 무압축 baseline을 test500에서 재측정해
cvlab21의 `baseline_ada_ps_test`와 대조한 결과:

| | |
|---|---|
| minADE@6 | 양쪽 **0.841711**, 페어드 차이 +0.000000 |
| 차이 나는 클립 | **0 / 500**, max \|diff\| 0.000e+00 |
| CoC 텍스트 일치 | **100.0%** |

Ada↔Blackwell이 0.286 vs 0.291로 갈리고 텍스트도 3–4% 달랐던 것과 대비된다. **cvlab20의 평가
결과는 cvlab21 결과와 같은 표에 넣어도 된다.**

**범위 주의:** baseline 경로에서만 검증됐다. slim 체크포인트는 `slim_lib`의 다른 attention
forward를 탄다(kept Q head마다 K/V gather, `num_key_value_groups=1`). 프루닝 arm을 여기서
돌려 cvlab21 표에 넣을 거면 그때 `dual` 대조를 한 번 더 해야 한다.

**공용이므로 실행 전 반드시 확인한다.** `run_importance_st.sh`처럼 러너 안에 사전 점검을
넣는 편이 낫다 — 카드마다 여유 메모리를 확인하고 모자라면 `REFUSING`으로 거부한다:

```bash
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$g")
[ $((total - used)) -lt 42000 ] && { echo "REFUSING: cuda:$g 사용 중"; exit 1; }
```

중요도 패스는 peak **40.5 GB**를 쓴다.

## 5. `run_retry_host.sh`를 쓰지 말 것

cvlab21 전용이다. 두 가지가 cvlab20에서 해롭다:

1. `HF_HOME=$HOME/.cache/huggingface`를 강제 export → **94% 찬 홈 파티션**을 가리킨다.
2. 그 경로의 `stored_tokens`에서 토큰을 읽는다 → **공용 계정이라 남의 토큰이 있을 수 있다.**

대신 `env.sh`를 source 하고 `.venv/bin/python`을 직접 부른다. 토큰은 애초에 필요 없다 —
가중치·config를 전부 옮겼고 실행 중 네트워크에 접근하지 않는다.

## 6. 환경변수 (`env.sh`)

```bash
export ALPAMAYO_REPO=/home/cvlab20/project/chan/alpamayo-model-compression
export AD_VLA_DATA=/mnt/dataset1/chan/data          # sample_cache / build_cache가 읽음
export HF_HUB_CACHE=/mnt/dataset1/chan/cache/hub    # blob 위치
export HF_HOME=/mnt/dataset1/chan/cache
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8              # 평가 결정론, CUDA 초기화 전에 필요
```

`AD_VLA_DATA`가 `os.environ.get`으로 읽히므로 **코드 수정이 필요 없다.**

## 7. 무엇을 옮겼나 (2026-09-05)

Tailscale rsync **실측 105 MB/s** — 50 GB가 10분 남짓이다. 걱정했던 것보다 훨씬 빠르다.

| 항목 | 크기 | 목적지 |
|---|---|---|
| repo (코드) | 86 MB | `/home/cvlab20/project/chan/alpamayo-model-compression` |
| 모델 `7aba8293` | 21 GB | `/mnt/dataset1/chan/cache/hub` |
| **프로세서 체인 3종 (메타데이터만)** | **35 MB** | 〃 — §8 참조 |
| 캘리브레이션 캐시 `calib_st` | 21 GB | `/mnt/dataset1/chan/data/physicalai_av/pre_processed/` |
| val_500 / test_500 / ood_val | 1.4 / 2.7 / 1.5 GB | 〃 |
| LingoQA 벤치마크 | 448 MB | `/mnt/dataset1/chan/data/lingoqa` |
| LingoQA judge repo | 9.6 MB | `/home/cvlab20/project/chan/LingoQA` (코드) |
| 매니페스트 `eval_sets` | 1.7 MB | `/mnt/dataset1/chan/outputs/eval_sets` |

**평가 세트는 필요한 클립만 보낸다.** 네임스페이스 전체는 `eval` 51 GB / `ood` 8.5 GB인데
val_500은 그중 500개, ood_val은 262개다. `--files-from`으로 `<clip_id>__t0_<t0_us>.npz`
목록을 만들면 정확히 필요한 것만 간다 — **62 GB → 5.4 GB**.

**옮기지 않는 것:**
- `slim_state.pt` (arm당 17 GB). `slim_meta.json` 3 MB면 `slim_lib.load_slim`이 베이스
  가중치에서 비트 동일 복원한다 — 텐서 1159개 전수 비교로 검증했다.
- `.venv` 8 GB. 현지에서 빌드한다(§9).
- `lingoqa_train` 16 GB. Hessian 캘리브레이션용이고 벤치마크에는 안 쓴다
  (`lingo_lib.DATA`는 `lingoqa`를 가리킨다).

## 8. 함정 — 모델만 옮기면 로드가 안 된다

모델 전송 후 첫 스모크 테스트가 이렇게 죽었다:

```
OSError: We couldn't connect to 'https://huggingface.co' to load the files,
and couldn't find them in the cached files.
```

Alpamayo의 `_build_processor`가 `AutoProcessor.from_pretrained`로 **VLM repo의
config/tokenizer**를 읽는다. 그 repo들의 safetensors는 절대 안 읽히므로 잊기 쉽다:

- `models--nvidia--Cosmos-Reason2-8B`
- `models--Qwen--Qwen3-VL-8B-Instruct`
- `models--Qwen--Qwen3-VL-2B-Instruct`

**safetensors·png를 제외한 스냅샷 파일 + `refs/` + `.no_exist/`** 만 보내면 된다(총 35 MB).
`experiments/transfer/lists.sh`의 `_repo_meta`가 정확히 그 목록을 만든다.

**또 하나: 스냅샷 디렉터리는 심링크 farm이다.** `snapshots/<rev>/`를 그냥 rsync하면
24 KB만 간다 — 실제 21 GB는 `blobs/`에 있다. **`rsync -aL`로 역참조**해야 한다. cvlab20에는
공유할 blob 저장소가 없으니 실파일로 보내는 게 맞고, `huggingface_hub`는 일반 파일로 된
스냅샷도 문제없이 읽는다.

## 9. venv는 복사하지 말고 빌드한다

`bash cvlab20-server/build_venv_cvlab20.sh` — 약 5분.

- 시스템 파이썬이 **3.10**인데 `pyproject`는 `==3.12.*`를 요구한다 → uv가 관리
  인터프리터를 받는다.
- `uv`가 설치돼 있지 않다 → `pip install --user uv`.
- **uv의 캐시와 관리 파이썬을 반드시 `/mnt/dataset1`로 돌린다.** 기본값은 홈이고 torch만
  1만 개 파일이다:
  ```bash
  export UV_CACHE_DIR=/mnt/dataset1/chan/.uv-cache
  export UV_PYTHON_INSTALL_DIR=/mnt/dataset1/chan/.uv-python
  ```
- **flash-attn은 prebuilt wheel로.** `config.json`이 `attn_implementation=flash_attention_2`를
  요구하므로 필수이고, sdist에서 빌드하면 30분+ 와 CUDA 툴체인이 필요하다.
  `cp312 / cu12torch2.8 / cxx11abiTRUE`가 이 venv와 맞는다.
- `alpamayo1_5`는 `pyproject.toml`에 없다 → `git+…/alpamayo1.5.git@f42e594` 별도 설치.

검증된 결과: python 3.12.13 / torch 2.8.0+cu128 / transformers 4.57.1 / flash_attn 2.8.3,
venv 8.1 GB(전부 `/mnt/dataset1`), 홈 증가분 0.

## 10. 스모크 테스트

`cvlab20-server/smoke_test.py` — 다섯 단계를 순서대로 확인한다. venv가 임포트된다고
모델이 로드된다는 보장은 없으므로 따로 만들었다.

1. 환경변수 (`AD_VLA_DATA` 없으면 엉뚱한 파일시스템을 본다)
2. torch·GPU 이름
3. 캐시 4종 매니페스트와 첫 npz 실존
4. 40 GB 예약
5. 모델 로드 + **파라미터 수 11,078,526,194 대조** — 다른 스냅샷이 왔으면 여기서 잡힌다

2026-09-05 통과: 로드 6.2초, 파라미터 수 일치.

## 11. 이 디렉터리의 스크립트

| 파일 | 역할 |
|---|---|
| `env.sh` | 환경변수. 무엇을 하든 먼저 source |
| `build_venv_cvlab20.sh` | venv 현지 빌드 + 심링크 |
| `smoke_test.py` | 5단계 점검 |
| `run_importance_st.sh` | 중요도 샤딩 실행 (카드 사전 점검 포함) |
| `parity_check.py` | cvlab21과 클립 단위 대조 — 평가를 섞어 쓰기 전 필수 |

## 12. 아직 확인 안 한 것

- **slim 체크포인트 경로의 비트 동일성** — baseline은 PASS(§4)지만 프루닝 arm은 미검증.
- 캐시 빌드(`build_cache.py`)를 여기서 돌릴 수 있는지 — HF 게이트 접근이 필요하고,
  공용 계정이라 토큰을 어떻게 넘길지 정하지 않았다.
- alpasim(폐루프)은 전혀 이관하지 않았다.
