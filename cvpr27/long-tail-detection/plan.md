# 계획: Claude API로 long-tail 씬 탐지 → 그 씬에서 개루프 평가 (2026-11-02)

## 가설

압축 비용(Δ minADE@6 vs dense)은 평균 세트보다 long-tail 씬에 몰려 있고, 방법 간 순위(MoOP vs LLM-Pruner vs Tyr)가
long-tail 부분집합에서 달라질 수 있다. NVIDIA의 큐레이션 long-tail(OOD, 9개 `event_cluster`)은 train/val에만
있으므로, **held-out test 분할(test_2000)에서** long-tail을 Claude로 찾아 같은 질문을 검증한다.

출발점(A100, minADE@8 평균): dense / MoOP l0keep — OOD-val 0.920 / 1.071(+0.151), test_2000 0.795 / 0.850(+0.055).

혼동 변수: long-tail 씬은 GT 기동(정지·회전)이 더 많을 수 있다 → `eval_lib.bucket`(행마다 저장됨) 안에서 비교하고
long-tail의 기동 분포로 가중한 DiD를 함께 보고한다.

## 결정 사항

| 항목 | 결정 |
|---|---|
| 이미지 전송 | 사용자가 허용(2026-11-02) — 카메라 4대 t0 프레임, 긴 변 1280 px |
| Claude 입력 | 모델 입력과 같은 정보만: 4개 카메라 t0 프레임 + ego 과거 1.5 s 요약. GT 미래·OOD의 `gt_coc`/`cluster`는 보내지 않음 |
| 모델 | `claude-opus-5-5`, adaptive thinking(기본), effort `medium`, structured output, `fallbacks: "default"` |
| 범주 | NVIDIA 9개 `event_cluster` + `NOMINAL`, 난이도 1–5 |
| 라이선스 §4.5 | 프롬프트에서 사람의 개인 속성 기술 금지, 번호판 언급 금지 |
| 라벨러 환경 | `/mnt/nvme1n1/ad_vla/venvs/longtail`(uv, torch 없음, `anthropic` 2.3.0) — 공용 `.venv`는 그대로 |

## 단계

| 단계 | 내용 | 상태 |
|---|---|---|
| S0 파일럿 | OOD-val 25(범주 라운드로빈) + test_2000 25, 각 2회 | 완료 — $2.39/100건, 반복 일치 96–100%, OOD long-tail 재현 60% |
| S1+S2 라벨링 | OOD-val 262 전부 + test_2000 전부, 1회 (`label_v1`) | 진행 |
| S3 개루프 분석 | 저장된 A100 클립별 결과 15개 arm을 long-tail / NOMINAL로 분할(GPU 없음) | 스크립트 준비 |
| S4 (선택) 확장 | test_2000 long-tail이 너무 적으면 Hub 스트리밍으로 test 분할을 더 훑어 캐시 빌드 후 주요 arm 재평가 | 결과 보고 결정 |

## 파일럿에서 본 한계

- OOD-val에서 놓친 것의 상당수는 NVIDIA 라벨 자체가 시각적으로 평범한 경우("급커브 앞 감속"이 `OTHER_LONGTAIL`,
  "신호 바뀌어 출발"이 `CYCLISTS`).
- 진짜 놓친 것은 단일 프레임으로 안 보이는 동적 사건(끼어들기·합류·양보). 캐시 프레임은 0.3 s 구간뿐이라 보강 불가 —
  이 라벨은 "t0 입력에서 보이는 long-tail"로 해석한다.
- 프롬프트는 파일럿 25클립에 맞춰 고치지 않았다(과적합 방지). 본 실행의 OOD-val 262 전체가 검증이 된다.

## 산출물

- 코드: `label_longtail.py`(라벨러), `analyze_longtail_openloop.py`(S3)
- `outputs/cvpr27/long_tail_detection/{pilot50,label_v1,openloop_v1}/`: `config.json`, `labels.parquet`/`metrics.json`,
  `summary.txt`, 그림
- 리포트: `cvpr27/reports/2026-11-02_long-tail-openloop.md`
