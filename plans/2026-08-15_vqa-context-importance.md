# VQA-문맥 중요도: 관측 문맥을 넓히면 추론이 보호되는가

2026-08-15 · 선행: `reports/evaluation/2026-08-14_lingoqa-zeroshot-benchmark.html`

## 가설

기존 두 언어 기준(`I_CoC`, `J`)은 **사전 어휘가 아니라 관측 문맥이 좁아서** 실패한다.
관측 문맥에 VQA 응답 위치를 추가하면 같은 예산에서 LingoQA가 유의하게 개선된다.

### 근거 (2026-08-14 측정)

동일 예산(−24.0%)에서 LingoQA 제로샷:

| arm | 기준 | LingoQA |
|---|---|---:|
| baseline | — | 73.2% |
| `dual_u40_v2` | max(traj, CoC) | 68.8% |
| `j_traj_u40_v2` | max(traj, J) | 67.8% |
| `traj_u40_v2` | 궤적 단독 | 37.0% |
| `j_u40_v2` | J 단독 | 32.2% |
| `coc_u40_v2` | CoC 단독 | 30.2% |

가설을 세우기 전에 두 개의 경쟁 설명을 배제했다.

**사전 어휘 편향이 아니다.** `jlens_v2`의 사전은 714 토큰 중 CoC 빈출이 202개(28%)고
나머지 512개(72%)는 무작위 어휘다. `select_tokens`가 *"the random half keeps the score
from being defined only on terms we already expect to matter"*라고 명시한 대로,
무작위 쪽이 오히려 다수다.

**표본 수 부족이 아니다.** `importance_v1`(50클립)과 `importance_v2`(100클립)는
**공유 클립이 0개**인 완전 분리 보정셋인데, 같은 비율에서 선택이 크게 일치한다:

| 기준 | 유닛 | v1 vs v2 Jaccard |
|---|---|---:|
| coc | Q head / MLP | 0.803 / 0.707 |
| traj | Q head / MLP | 0.786 / 0.693 |

표본을 통째로 갈아끼워도 70–80% 같은 유닛을 고른다. 표본을 늘리면 추정 분산은 줄지만
**목적함수의 적용 범위는 넓어지지 않는다.**

**남는 설명은 문맥이다.** `prune_lib.coc_nll`은 CoC 토큰 위치의 cross-entropy이고,
`jlens_lib`은 source position을 *"text and generated-CoC tokens"*로 제한한다
(`jlens_v2`: 1,676 위치 / 100 클립). 생성되는 CoC는 중앙값 14토큰의 "행동 + 근거" 한 줄이다.
"신호등이 무슨 색인가", "보행자가 몇 명인가"를 답할 때만 켜지는 유닛은 그 위치들에서
활성화되지 않으므로 점수가 0에 가깝고 잘려나간다. 실제로 LingoQA 손실은
**존재/예-아니오(−11.8pp, n=178)**와 **개수 세기**에 몰려 있다.

## 설계

`prune_lib`에 세 번째 목적함수 `I_VQA`를 추가한다. `coc_nll`과 구조가 같고
**관측 위치만 다르다** — CoC 토큰 대신 VQA 답변 토큰이다.

```
coc_nll : create_message      → <|cot_start|>  ... CoC 토큰 위치의 CE
vqa_nll : create_vqa_message  → <|answer_start|> ... 답변 토큰 위치의 CE
```

`helper.create_vqa_message`가 이미 있으므로 프롬프트 포크가 필요 없다.
`experiments/lingoqa/lingo_lib.py::load_segment`가 LingoQA 이미지를 Alpamayo 입력 규격으로
변환하는 일을 이미 하므로 그대로 재사용한다.

### 데이터: LingoQA **train** split

평가셋은 절대 쓰지 않는다. 중요도는 train에서만 계산하고 `val.parquet` 500 질문은 held-out으로 남긴다.

| split | 크기 | 내용 | 사용 |
|---|---:|---|---|
| scenery train | 7.7 GB | 3.5k 비디오 / 267.8k QA, **fine-grained perception 중심** | **주 사용** |
| action train | 53 GB | 24.5k 비디오 / 152.5k QA, 행동·판단 중심 | 보류 |
| val (평가) | 232 MB | 100 세그먼트 / 500 질문 | **손대지 않음** |

scenery를 고르는 이유는 두 가지다. 손상이 집중된 범주(존재/개수 = 지각)와 정확히 맞고,
53 GB 대신 7.7 GB로 끝난다. 중요도 계산에는 기존 프로토콜과 동일하게 **100 클립**만 쓰므로
전체를 받을 필요는 없지만, Drive zip은 부분 추출이 어려워 통째로 받은 뒤 100개를 뽑는다.

보정 클립 선정은 `make_split.py`의 관례를 따라 시드 고정으로 뽑고 `outputs/vqa_calib_100.json`에 기록한다.

## 만들 arm (전부 예산 0.3985632694, VLM-only — v2 계열과 동일)

| arm | 기준 | 검정하는 것 |
|---|---|---|
| `vqa_u40_v2` | VQA 단독 | VQA 문맥만으로 충분한가 (단일 기준 대조군) |
| `trajvqa_u40_v2` | max(rank traj, rank VQA) | **주 arm.** `dual`/`j_traj`의 직접 대응물 |
| `coclingo_u40_v2` | CoC 단독, **LingoQA 이미지에서 계산** | 도메인 통제 (아래) |

### 도메인 통제가 필요한 이유

LingoQA train은 Wayve 영국 영상이고 기존 보정셋은 PhysicalAI-AV 미국 영상이다.
`trajvqa`가 이기더라도 원인이 **VQA 문맥**인지 **영국 도메인 노출**인지 구분되지 않는다.
`coclingo_u40_v2`는 목적함수를 CoC로 두고 이미지만 LingoQA로 바꾼 arm이라, 이 둘의 차이가
문맥 효과를 분리해 준다.

```
coc_u40_v2      : CoC 목적 · PhysicalAI-AV 이미지   → 30.2% (측정됨)
coclingo_u40_v2 : CoC 목적 · LingoQA 이미지          → 도메인 효과만
vqa_u40_v2      : VQA 목적 · LingoQA 이미지          → 도메인 + 문맥
```

## 사전 등록 게이트

**G1 (주 가설).** `trajvqa_u40_v2` − `dual_u40_v2`의 LingoQA 페어드 델타가
95% 클러스터 부트스트랩 CI로 **0을 초과**해야 한다. 못 넘으면 문맥 가설은 기각한다.

**G2 (문맥 vs 도메인).** `vqa_u40_v2` − `coclingo_u40_v2` > 0 (CI가 0 배제).
이게 성립해야 이득이 도메인이 아니라 문맥에서 온 것이다. 성립하지 않으면
"LingoQA 이미지를 본 것"이 원인이며 일반화되지 않는 결과로 보고한다.

**G3 (반증 조건).** `vqa_u40_v2`가 단독 기준으로 30%대에 머물면 —
즉 다른 단일 기준들과 구분되지 않으면 — **문맥이 아니라 기준 결합이 지배 요인**이라는 뜻이다.
이 경우 G1이 성립하더라도 "VQA 문맥이 특별하다"고 주장하지 않고
"세 번째 독립 관점을 추가한 효과"로 보고한다.

**G4 (주행 능력 통제, 필수).** `trajvqa_u40_v2`의 in-dist minADE가 `dual_u40_v2`(0.7766) 대비
**+10% 이내**여야 한다. 언어를 얻고 주행을 잃으면 이 방향은 실패다.
`nll_gtcoc`와 OOD minADE도 함께 보고한다.

## 실행 순서

| # | 단계 | 산출물 | 비용 |
|---|---|---|---|
| 0 | scenery train 다운로드 + 100 클립 보정셋 확정 | `vqa_calib_100.json` | ~30분, 7.7 GB |
| 1 | `prune_lib.vqa_nll` + `run_importance.py --objective vqa` | `outputs/importance_vqa/` | 계산 필요 |
| 2 | `coclingo` 중요도 (동일 이미지, CoC 목적) | `outputs/importance_coclingo/` | 1과 동일 |
| 3 | `make_slim.py --config {vqa,trajvqa,coclingo}_u40_v2` | slim 3개 (17 GB씩) | ~1.5h × 3 |
| 4 | LingoQA 평가 (`run_lingo_vqa.py` + `score_lingo_vqa.py`) | 점수 3개 | ~15분 × 3 |
| 5 | 개루프 평가 (`run_baseline.py --set indist,ood`) — G4 | minADE / nll | ~2h × 3 |
| 6 | 분석 + 리포트 | `reports/evaluation/` | — |

**1단계 소요는 미지수다.** `run_importance.py`의 100클립 실측 시간을 먼저 확인하고,
디스크(현재 932 GB 여유, 체크포인트 3개 = 51 GB)를 재확인한 뒤 3단계에 들어간다.

## 위험

- **`vqa_nll`의 타깃.** LingoQA train은 질문당 정답이 1개다(평가셋은 2개).
  레퍼런스 답변을 teacher-forcing해 CE를 재므로 `coc_nll`과 구조는 같지만,
  답변 길이 분포가 CoC(14토큰)보다 길어 위치 수가 달라진다. 목적함수 간 스케일이 달라지면
  `rank_norm`이 흡수하지만, 단일 기준 arm의 절대 스케일 비교는 하지 않는다.
- **제로샷 주장 소멸.** 이 arm들의 LingoQA 수치는 중요도 계산에 LingoQA train을 썼으므로
  **더 이상 제로샷이 아니다.** 논문에 실을 때 기존 5개 arm과 별도 블록으로 분리해야 한다.
  train/eval 분리는 지켜지므로 벤치마크 결과로서는 유효하다.
- **비용.** 체크포인트 3개 재빌드 + 개루프 평가로 하루 이상. 공유 GPU 사정에 따라 더 길어진다.
- 도메인이 바뀌므로 `analysis_lib.build_inputs`의 토큰 span 가정(카메라 4대 × 4프레임)이
  LingoQA(1대 × 4프레임)에서 달라진다. 중요도 계산 코드가 span에 의존하는지 1단계에서 확인한다.

## 축소 대안

비용이 부담이면 **1–2단계까지만 하고 멈추는 것**이 가능하다. 중요도끼리의 겹침만으로도
가설의 예측을 부분 검증할 수 있다:

```
문맥 가설이 맞다면  I_VQA와 I_CoC의 선택 겹침은 낮아야 한다 (< 0.62, coc↔traj 수준)
문맥이 무관하다면   I_VQA ≈ I_CoC (Jaccard > 0.8, v1↔v2 재현성 수준)
```

체크포인트를 만들지 않고 중요도 계산만으로 끝나므로 비용이 크게 준다.
겹침이 높게 나오면 3단계 이후를 진행할 이유가 없다.
