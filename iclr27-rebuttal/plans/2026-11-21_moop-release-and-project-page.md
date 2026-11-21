# Plan: MoOP 공개 코드 저장소 + GitHub Pages 프로젝트 페이지 (2026-11-21)

사용자 요청(11-21):
- "이제 코드 정리하는 거랑 project page 만드는 거 하려고 하는데 도와줘."
- "공개용 새 저장소로 MoOP 코드만 정리하고, 페이지는 GitHub Pages로"
- "https://cvlab-kaist.github.io/DreamX/ 구성을 참고해줘."

## 0. 먼저 정할 것 (각 항목의 첫 선택지가 추천)

| # | 결정 | 선택지 | 이유 |
|---|---|---|---|
| D1 | **공개 시점과 익명 처리** | (a) 지금은 로컬과 비공개 저장소로만 만들고, 공개는 심사 결과 또는 arXiv 이후. (b) 심사용 익명 버전(anonymous.4open.science 등)을 지금 내고, 실명 버전은 나중에. (c) 지금 실명으로 공개 | 논문 PDF가 "Under review … ICLR 2027, Anonymous authors" 상태다. 실명 저장소나 페이지를 지금 공개하면 double-blind 원칙과 부딪힐 수 있다 |
| D2 | **GitHub 위치** | 예: `cvlab-kaist/MoOP`. 코드는 `main`, 페이지는 `gh-pages` 브랜치 → `cvlab-kaist.github.io/MoOP/`. 또는 개인 계정 | 지금 쓸 수 있는 push 권한은 `deep-overflow` 개인 토큰뿐이다. 조직 저장소라면 사용자가 저장소를 만들거나 권한을 줘야 한다. push는 승인 후에만 한다 |
| D3 | **코드 범위** | (a) MoOP 핵심(중요도 → 마스크 → 가지치기) + 개루프(표 1) + LingoQA(표 4) + AlpaSim 폐루프용 패치. (b) 핵심 + 개루프만 | 논문 표를 재현할 수 있어야 공개 의미가 있다. Tyr와 LLM-Pruner 포트, A100·GH200 도구, 분석용 스크립트는 넣지 않는다 |
| D4 | **배포물** | (a) 마스크(`slim_meta.json`, arm당 1.7–3.4 MB) + 중요도 파일(`importance.npz` 12 MB 등) + 평가 manifest(clip ID 목록). 체크포인트는 스크립트로 기본 가중치에서 다시 만든다. (b) 가지치기한 체크포인트(arm당 15–17 GB)도 HF Hub에 올린다 | MoOP는 가중치를 고쳐 쓰지 않는다. 그래서 마스크와 기본 가중치만 있으면 체크포인트가 그대로 다시 만들어진다. 기본 가중치 라이선스(OpenMDW-1.1)상 (b)도 가능하지만 필요하지 않다 |
| D5 | **공개 레시피** | (a) 논문 그대로 `dual_u40_v2`. `_l0keep`은 "안정적인 변형"으로 문서에 적는다. (b) `_l0keep`을 기본으로 | 논문 표와 같아야 재현이 된다. 다만 plain 레시피는 0번 층 head 동점 처리에 민감하다(같은 레시피를 다시 측정하면 Test500 SD 0.26). 공개 배포는 논문 마스크 파일을 그대로 쓰므로 문제가 없고, 다시 측정할 때만 영향이 있다 |
| D6 | **페이지 미디어** | (a) 우리가 만든 그림과 표만 쓴다. 방법 그림은 인라인 SVG로 다시 그린다. 영상 자리는 비워 둔다. (b) 폐루프 렌더와 카메라 프레임 영상도 넣는다 | 주행 영상과 카메라 프레임은 NVIDIA PhysicalAI-AV와 NuRec(gated) 데이터에서 나온다. 지금까지 공개물에 넣지 않았다. (b)를 고르려면 데이터 이용 약관을 사용자가 확인해야 한다 |
| D7 | **저자·소속·링크** | 실명이면 저자 목록, 소속, 논문과 arXiv 링크가 필요하다. 익명이면 자리표시자로 둔다 | 히어로 영역에 들어간다 |

## 1. 코드 저장소

위치: `/home/cvlab21/project/chan/MoOP`. 연구 저장소 밖의 새 git 저장소로, 승인 전에는 push하지 않는다.

### 구성
```
MoOP/
  README.md            설치 · 데이터 준비 · 빠른 시작 · 표 재현 명령
  LICENSE              Apache-2.0
  NOTICE               Alpamayo 1.5 코드 Apache-2.0, 가중치 OpenMDW-1.1, 출처 표기
  pyproject.toml       uv. torch 2.8.0, transformers 4.57.1,
                       alpamayo1_5 @ git+NVlabs/alpamayo1.5@f42e594, physical_ai_av,
                       numpy, pandas, pyarrow, pillow
  moop/
    model_io.py        입력 구성, CoC rollout, KV 캐시 denoise, GT action   ← analysis_lib 일부
    gates.py           Q head·MLP 채널 gate, CE NLL, FM grad(합산·step별)    ← prune_lib 일부
    importance.py      VLM CE + FM Taylor 중요도                            ← run_importance
    expert.py          expert step별 중요도, znorm 집계                      ← run_step_importance + make_stepexp_importance
    masks.py           층 안 rank 정규화, max 결합, 층별 균일 예산, expert 선택
                       ← make_slim(uni/dualexp 분기) + mask_lib + tyr_lib.dual_scores + run_cocsafe.rank_norm
    slim.py            물리적 제거, 저장·불러오기                            ← slim_lib
    data.py            manifest, 캐시 빌드·로드                              ← sample_cache + build_cache + build_ood_cache
    metrics.py         minADE/minFDE@6, 클러스터 bootstrap CI               ← eval_lib + ci_openloop_lingoqa 일부
  scripts/             compute_importance.py, compute_expert_importance.py, build_slim.py,
                       eval_openloop.py, eval_lingoqa.py, score_lingoqa.py, summarize.py
  configs/             moop_u40, moop_e_u40(em93.75), fm_only_u40, ce_only_u40, wanda_u40, moop_u50 …
                       예산은 상수로 둔다(u40 = 층당 head 13개·채널 4,898개 절단, u50 = 16·6,144)
  recipes/             논문 arm별 slim_meta.json + importance.npz
  data/manifests/      calib_100, test_500, ood_val (parquet, clip ID와 t0)
  alpasim/             driver 패치(slim 로더 + 10줄 hook), 기준 upstream 커밋, 인계 감속 수정 패치, 실행 안내
```

### 옮기면서 고칠 것 (탐색 결과)
- **경로:** 하드코딩된 경로(`/mnt/nvme1n1/…`, `/home/cvlab21/…`, `/workspace/…`)를 인자나 환경 변수로 바꾼다. 연구실 전용 GPU 예약 코드(`reserve_gpu`)와 Cosmos-Reason2 `local_files_only` 패치는 뺀다.
- **기본값:** 잘못된 arm을 조용히 만드는 기본값을 바로잡는다.
  - `--importance` 기본값이 v1이다.
  - `--num-clips` 기본값이 50이다.
  - `--noise-mode` 기본값이 shared다.
  - znorm 집계가 존재하지 않는 per-clip 파일을 읽는다.
- **예산:** u40 예산을 다른 체크포인트(`slim_integrated_mag`)에서 읽어 오던 부분을 상수로 바꾼다. 이 경로를 따라 들어오던 matplotlib과 scipy 의존성도 사라진다.
- **리비전 고정:** 모델 리비전(`7aba829…`)을 모든 곳에 고정한다. 지금 고정되지 않은 AlpaSim 로더도 포함한다.
- **정리:** 연구용 분기와 플래그(`make_slim`의 수십 개 config 분기, `prune_lib`의 사용하지 않는 gradient 경로 등)를 지운다. 주석·docstring에 남은 날짜, 보고서 경로, 한국어, GPU 이름도 지운다.
- **패키지 범위:** 표 1, 4 재현에 필요한 것만 남긴다. 그림 분석 코드는 넣지 않는다(필요하면 나중에 추가).

### 동등성 검증 (gate)

연구 코드와 결과가 같은지 확인한다.

| gate | 내용 |
|---|---|
| G1 마스크 | 공개 코드로 `importance_v2`에서 다시 만든 마스크가 연구 쪽 `slim_meta.json`과 유닛 단위로 같다. 대상: dual / traj / coc / wanda u40, dualexp_u40_em93p75, dual_u50_v2 |
| G2 체크포인트 | 공개 코드로 만든 slim 모델의 state_dict가 연구 체크포인트와 텐서 단위로 같다 |
| G3 개루프 | Test500 20 clip을 같은 카드에서 돌린 결과가 연구 `run_baseline` 결과와 bit 단위로 같다 |
| G4 중요도 | calib 3 clip에서 다시 잰 중요도가 연구 코드와 같다(같은 카드). 비결정성이 있으면 차이를 문서화한다 |
| G5 위생 | ruff 통과. 절대경로, 토큰, 한국어, 사내 경로를 grep으로 확인한다 |
| G6 새 환경 | 깨끗한 uv venv에 설치한 뒤 build와 smoke가 돈다. 이 venv는 11-17에 지워진 메인 venv를 대신하는 역할도 한다 |

G3와 G4는 GPU가 필요하다. 그때 빈 카드를 확인하고 사용자 규칙에 따라 띄운다.

## 2. 프로젝트 페이지 (DreamX 구성 참고)

DreamX(`cvlab-kaist.github.io/DreamX/`)의 구조:
- 상단 고정 내비게이션과 주간·야간 전환
- 히어로: 워드마크, 부제, 저자, 소속, 버튼
- 티저, 한 줄 요약, 핵심 수치 타일 4개
- 섹션은 배경을 번갈아 쓰는 band 구조
- 그림은 인라인 SVG, 색은 토큰으로 관리
- 정적 3파일(index.html, style.css, main.js)

**구성은 따르되 코드는 복사하지 않는다.** DreamX 저장소에 라이선스 파일이 없으므로, HTML과 CSS는 새로 쓴다.

| 섹션 | 내용 |
|---|---|
| 히어로 | "MoOP" 워드마크 + 논문 제목, 저자와 소속(D7), Paper / arXiv / Code / Results 버튼 |
| 티저 | 논문 그림 1(Reasoning only / Action only / Reasoning & Action)을 인라인 SVG로 다시 그린다 |
| 요약 + 수치 | 24.0% 감소 시 LingoQA 94.0% 유지 · expert 포함 39.4% 감소 · 1.37× 가속 · peak memory 36.6% 감소 |
| Findings | 카드 세 장. (1) 추론 목적과 궤적 목적이 같은 VLM 구조에 다른 중요도를 준다. (2) expert MLP 채널은 denoising step 사이에 안정적이다(Q head는 아니다). (3) 소수의 채널이 중요도 대부분을 차지한다. 각 카드에 논문 그림(우리 플롯)을 넣는다 |
| Method | 3단계. CE와 FM Taylor 중요도 측정 → 층 안 rank 결합 → expert step별 중요도를 모아 고정 채널 부분집합 선택 |
| Results | 표 1(개루프), 표 2(폐루프 Origin150/Hard100), 표 4(LingoQA), 표 5(지연·메모리). u50 rebuttal 결과를 넣을지는 사용자 결정 |
| BibTeX | |

- **호스팅:** 코드 저장소의 `gh-pages` 브랜치(D2).
- **미리보기:** 공개 전에 비공개 Artifact로 띄워 검토받을 수 있다.

## 3. 순서
1. 이 문서로 D1–D7 결정
2. 저장소 뼈대와 환경 구성(G6)
3. 핵심 코드 포팅 → G1, G2
4. 평가 코드 포팅 → G3, G4
5. AlpaSim 패치와 README
6. 페이지 뼈대 → 내용 → 그림 → 미리보기
7. 사용자 검토 후 공개(D1, D2 조건에 따라 push)
