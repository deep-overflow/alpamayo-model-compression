# Wanda 베이스라인 — LingoQA 언어 채널 평가 (2026-08-20)

## 목적과 가설

개루프에서 Wanda(wanda_u40_v2)는 dual 대비 ΔminADE@6 +1.06* 열세에 CoC degen 86–88%를
보였다(`plans/2026-08-20_wanda-baseline.md`). 추론 채널 붕괴가 **일반 언어 능력**
(LingoQA, 표준 프로토콜)에서도 나타나는지 측정해 T3(`tab:lingoqa`)의 베이스라인 행을
채운다.

**H-WL**: Wanda의 LingoQA 정확도는 dual(68.8)보다 유의하게 낮고, 단일 기준 arm들
(traj 37.0 / j 32.2 / coc 30.2) 수준 이하다 — gradient-free 선택은 언어 유닛도 지키지
못한다.

## 설정 (기존 코드 그대로, 신규 로직 없음)

- **런**: `experiments/lingoqa/run_lingo_vqa.py --arm outputs/slim_wanda_u40_v2` —
  표준 LingoQA VQA 프로토콜 (`helper.create_vqa_message`, 합성 ego 없음), 500문항,
  greedy, max-gen 128, style unprompted, 프레임 4/카메라 1, Ada 카드, 결정론 플래그.
  T3의 모든 arm(`lingo_vqa_slim_*_u40_v2`)과 동일 설정. 출력
  `outputs/lingo_vqa_slim_wanda_u40_v2/{rows.json, predictions.csv, config.json}`.
- **채점**: `score_lingo_vqa.py --runs lingo_vqa_slim_wanda_u40_v2 ...` — LingoQA의
  `LingoJudge` verbatim, logit > 0, `matched == 500` 필수. 요약 `outputs/lingo_vqa_scores_wanda`.
- **비교**: `analyze_lingo.py --runs <wanda> <dual> <baseline> <traj> <coc> <j>
  --baseline lingo_vqa_slim_dual_u40_v2` — 세그먼트(5문항/클러스터) 재표집 paired
  bootstrap(주) + McNemar(보조). 출력 `outputs/lingo_wanda_vs_dual`.
- 드라이버: `experiments/lingoqa/eval_lingo_arm.sh <slim_dir_name> <gpu>` (신규, 위 세
  단계를 run_retry_host로 체인; 어느 arm에나 재사용 가능).

## 사전 등록 게이트

- **L0 (무결성)**: predictions 500행, judge `matched == 500`, config가 T3 arm과 동일
  (greedy/128/unprompted/4프레임).
- **L1 (주 판정)**: paired Δacc (wanda − dual), 세그먼트 군집 bootstrap 95% CI.
  - CI 전체 < 0 → H-WL 확인 (언어 채널도 붕괴), T3에 베이스라인 행 수록.
  - CI가 0 포함 → "추론 붕괴(degen 86%)에도 VQA 능력은 보존" — 흥미로운 분리 결과로
    보고, CoC 경로 vs VQA 경로 차이를 별도 논의.
- **L2 (부기록)**: Δ vs baseline(73.2), 단일 기준 arm 대비 위치, 답변 길이·truncation
  비율(dual 34.4%), 퇴화 답변(빈/반복) 비율.

## 예산

런 ~45–60분(Ada 1장) + 채점 ~5분(judge, GPU) + 분석(CPU) — 총 ~1.2 h.

## 상태

작성·승인("계획을 세우고 코드 작성부터 진행해줘", 2026-08-20) — 드라이버 작성 후 실행.

**완료 — H-WL 확인 (2026-08-20).** L0: predictions 500, judge matched 500, 설정 동일.
L1: wanda **9.2%** vs dual 68.8% → paired Δ **−59.6pp [−64.2, −54.8]**, McNemar p≈0
(b/c 11/309) — CI 전체 < 0. L2: vs baseline 73.2 → −64.0pp; 단일 기준 arm(30–37%)보다
20pp+ 아래; 답변 중앙값 1단어, ≤2단어 69.0%, 토큰 쓰레기 포함 56.2%, trunc 25.8%.
산출물: outputs/lingo_vqa_slim_wanda_u40_v2, lingo_vqa_scores_wanda_u40_v2,
lingo_wanda_vs_dual. 해석: 개루프 degen 86%와 같은 붕괴가 VQA 경로에서도 재현 —
gradient-free 활성값×크기 선택은 주행·추론·일반 언어 유닛 어느 것도 지키지 못한다.
