# 재구성 오차를 토큰 타입별로 분해한다

## 왜

`analyze_dualrw.py:202`의 "rel err on own-CoC tokens" 그래프는 정보가 없다. 최소제곱 refit은
`H_fit`으로 가중된 오차를 **정의상** 최소화하므로, own-CoC를 `H_fit`에 넣은 arm이 own-CoC
오차에서 이기는 것은 결과가 아니라 항등식이다. 게다가 그 값은 **in-sample**이었다
(`G3a in-sample reconstruction error`).

정보가 나오는 질문은 이것이다: **fit에 넣지 않은 토큰 타입에서는 무슨 일이 일어나는가.**
재구성은 예산이 정해진 게임이므로, 한 타입을 사면 다른 타입을 판다. 그 교환율을 재본 적이 없다.

## 무엇을 재는가 (용어 정정)

`tyr_lib.recon_error`는 `rel = ‖(W − Ŵ)X‖_F / ‖W X‖_F`, `X`는 **그 층 o_proj / down_proj의
입력**이다. 따라서 이 값은

- **KV cache 변화가 아니다.** 층 l의 o_proj 출력은 residual → 층 l+1의 k/v_proj → 층 l+1
  캐시로 간다. 캐시 변화의 상류 원인이지 캐시 변화 자체가 아니다.
- **CoC 위치 hidden state의 변화도 아니다.** 층 하나·서브블록 하나의 **증분**이고, dense
  입력에서 층별로 독립 측정된다(오차가 층을 타고 누적되지 않는다).
- 정확히는 **"층 l의 어텐션(또는 MLP) 출력이 해당 토큰 위치에서 얼마나 달라졌는가"** 이다.
  토큰 타입은 측정 대상이 아니라 **오차를 읽는 위치를 고르는 마스크**다.

## 설계

기존 V/T/D 3분할(V = vision+hist+sink)을 이 저장소의 표준 5분할로 쪼갠다
(`expert_per_clip.REGIONS`, `analysis_lib.compute_spans`):

    vision · prompt_text(=instruction) · hist(ego history) · sink · coc(own rollout)

arm 4종은 이미 `slim_state.pt`가 있어 재빌드가 필요 없고, 각자의 fit 구성이 metadata에
숫자로 남아 있다:

| arm | prefill 가중 | CoC 몫 | LingoQA |
|---|---|---|---|
| `dualr` | 균일 | 0.0 | 없음 |
| `dualr_rep` | expert 어텐션 (vision .7223 / text .1656 / hist .0418 / sink .0110) | 0.0 | 없음 |
| `dualr_w` | 동일 | 0.16 | 없음 |
| `dualr_wl` | 동일 | 0.04 | 있음 |

**측정 방법 — Hessian을 쌓지 않는다.** `H`는 d×d(down_proj는 12288² = 604 MB/스트림/층)라
5분할이면 저장이 부담이다. 대신 dense forward의 forward hook에서 `y = Wx`(모듈 출력)와
`ŷ = Ŵ x_kept`를 직접 만들어 타입별로 `‖y − ŷ‖²`, `‖y‖²`를 누적한다. 스칼라만 쌓으므로
메모리가 상수이고, `rel = sqrt(Σ‖Δy‖² / Σ‖y‖²)`는 `recon_error`와 **정확히 같은 양**이다
(H = X Xᵀ를 거치지 않을 뿐).

**held-out**: fit에 쓰인 `racfit_v1` 캘리브 클립이 아니라 `indist_500`의 앞 N개를 쓴다.
in-sample이었던 기존 그래프의 두 번째 결함을 함께 고친다.

## 가설과 게이트

- **H1 (교환)** CoC 몫이 0 → 0.16으로 커질 때(`dualr_rep` → `dualr_w`, 다른 요인 동일)
  coc 오차는 내려가고 **vision 오차는 올라간다**. 두 변화가 모두 CI로 0을 제외하면 교환이
  실재한다. 올라가지 않으면 CoC 추가는 공짜이고, 그때는 "왜 LingoQA는 나빠졌나"가 다시 열린다.
- **H2 (가중 추종)** arm 내에서 타입별 오차 순위가 그 타입의 fit 가중치 순위와 음의 상관을
  갖는다(가중치가 클수록 오차가 작다). Spearman으로 확인. 성립하면 오차는 fit 가중치의
  직접적 함수이고, 기준 설계는 "무엇을 넣느냐"가 아니라 "얼마나 넣느냐"의 문제가 된다.
- **H3 (비대칭)** hist·sink는 토큰 수가 적고(≈0.5%, 1개) 가중치도 작아 어느 arm에서도
  오차가 크다 — 즉 재구성이 구조적으로 포기하는 스트림이 있는지 확인한다.

## 실행

`experiments/head_analysis/run_streamerr.py` (신규) — dense 모델 1회 로드, 클립마다
롤아웃 1회(고정 시드) + teacher-forced forward 1회, 72개 모듈 × 4 arm × 5 타입 누적.
`outputs/streamerr_v1/{config.json,metrics.json,summary.txt,plots/*.png}`.
분석은 `analyze_streamerr.py`.

비용: 클립당 롤아웃 ~8 s + forward ~1 s, arm별 추가 matmul은 클립당 ~0.5 s. N=24면 약 5분.
GPU 1장(Ada 또는 Blackwell 무관 — 이 측정은 arm 간 페어드 비교가 아니라 절대 오차라
아키텍처 고정이 필요 없다. 다만 한 실행 안에서 모든 arm을 같은 카드로 잰다).
