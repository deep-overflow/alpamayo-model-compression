# Plan (draft): MoOP 레이어 예산을 중요도 불일치로 배분 — Test500 개루프

2026-11-10 · status: **done 2026-11-11 13:03 KST**. 보고서 `iclr27-rebuttal/reports/2026-11-11_moop-disagreement-allocation-test500.md`. 승인됨(2026-11-10), 승인 때 D5(초반 층 과다 절단 방지)가 더해졌다.
- 결과: 두 arm 모두 MoOP l0keep보다 **악화**. 6.4 s minADE@6 짝 차이는 dis050 +0.035 [+0.005, +0.065], dis100 +0.103 [+0.064, +0.143]이다.
- 실행: soowon 큐가 빈 11-11 12:21 KST에 대기 스크립트가 워커를 띄웠고, 13:03에 끝났다.

## 진행 기록

- **D5 해석.** 사용자 요청은 "초반 레이어들이 너무 많이 잘리진 않게"였다.
  - 구현: 1–35번 층 어디서도 유닛의 절반 넘게 자르지 않는다(Q head ≤ 16/32, MLP ≤ 6,144/12,288). MoOP는 층마다 약 40%를 자른다.
  - 상한을 넘는 몫은 상한에 걸리지 않은 층에, 그 층들의 절단 수에 비례해 넘긴다. 총량은 그대로다.
  - 값을 바꾸려면 `make_disagree_masks.py`의 `CAP_FRAC` 한 줄을 고치면 된다.
- **G0 통과** (`code/make_disagree_masks.py`, `outputs/iclr27-rebuttal/disagree_alloc/alloc_summary.json`).
  - 강도 0은 `slim_dual_u40_l0keep_v2`를 유닛 단위로 재현한다.
  - 두 arm 모두 제거 파라미터가 2,657,447,936개로 MoOP l0keep과 같고, 0번 층도 같다. 층 안에서 남긴 유닛은 모두 자른 유닛보다 점수가 높다.
  - 상한이 걸린 층과 실제 배분:

    | arm | Q 상한 걸린 층 | MLP 상한 걸린 층 | 층별 Q 절단 | 층별 MLP 절단 |
    |---|---|---|---|---|
    | dis050 | 3, 5, 11, 13, 17 | 1–5 | 7–16 | 3,881–6,144 |
    | dis100 | 1–5, 8–13, 16, 17, 19–21, 25, 32 (18개 층) | 1–7 | 1–16 | 2,930–6,144 |

  - 층 구간별 평균 (Q / MLP):

    | arm | 1–5 | 6–17 | 18–29 | 30–35 |
    |---|---|---|---|---|
    | dis050 | 15.2 / 6,144 | 14.2 / 5,247 | 12.1 / 4,574 | 10.5 / 3,989 |
    | dis100 | 16.0 / 6,144 | 14.8 / 5,777 | 12.1 / 4,455 | 8.8 / 3,168 |

  - 상한 때문에 두 arm은 1–5번 층 MLP가 같다(6,144). 차이는 주로 중간 · 후반 층에서 난다.
- **실행.** `code/run_disagree_alloc.sh`(카드당 워커 1개, mkdir 큐: 빌드 2개 → Test500 4묶음 × 2 arm), Ada 4–7번, main 체크아웃에서 띄움.
- **대기 (21:46 KST).** 4–7번 카드가 비어 있지 않았다.
  - soowon의 GPU 큐(`/mnt/nvme1n1/ad_vla/results/soowon/rebuttal_expert/queue_1110`)가 4–7번을 쓰고 있다. 작업 76개 중 68개가 대기였고, 거의 전부 OOD-train 묶음(묶음당 약 32분)이다.
  - 4장이면 약 10시간이라 11-11 08:00 KST 전후에 빈다. 0–3번(Blackwell)도 다른 작업으로 차 있었다. cvlab20은 키 인증으로 접속되지 않는다.
  - 남의 큐를 건드리지 않으려고 카드별 대기 스크립트 4개를 띄웠다(job tmp의 `start_when_free.sh`). 각 스크립트는 그 큐에 시작 안 된 작업이 0개이고 그 카드가 5분 연속 10 MiB 이하일 때만 워커를 띄운다. 로그는 `outputs/iclr27-rebuttal/disagree_alloc/logs/gate_{4..7}.log`.

## 목적 · 가설

- **가설.** MoOP는 층마다 max(rank I_traj, rank I_CoC)로 남길 유닛을 고른다. 두 기준의 상위 집합이 많이 다른 층(불일치가 큰 층)에서는 두 기준의 상위 유닛을 예산 안에 다 담지 못해, 한쪽에만 중요한 유닛을 버리게 된다. 불일치가 큰 층은 덜 자르고 작은 층은 더 자르면, 같은 총 제거량에서 개루프 궤적 오차가 줄어든다.
- **비교 기준.** MoOP l0keep(`dual_u40_l0keep_v2`, 층별 균일 배분), 같은 Test500 500클립, 같은 RTX 5880 Ada.

## 결정 (2026-11-10 사용자 답변)

| | 결정 |
|---|---|
| D1 불일치 지표 | 상위 집합 겹침: 1 − \|T ∩ C\| / \|T ∪ C\| |
| D2 반영 강도 | 두 강도 모두: ±50% arm과 ±100% arm |
| D3 깊이 대조군 | 넣지 않음(해석 한계는 아래 "위험"에 적음) |
| D4 0번 층 | MoOP l0keep과 같게 고정: Q head 32개 전부 보존, MLP는 MoOP와 같은 4,929채널 절단. 배분은 1–35번 층에만 적용 |

D1의 T와 C는 한 층에서 I_traj와 I_CoC가 각자 남기려는 상위 유닛 집합이다. 크기는 MoOP가 그 층에서 남기는 개수다.

내가 정한 기본값(질문하지 않음, 이견 있으면 알려주세요):

- **축별 총량 고정.** Q head 절단 총수 455개(1–35번 층 × 13), MLP 절단 총수 172,508채널(1–35번 층)을 MoOP l0keep과 같게 둔다. 둘 다 같으면 제거 파라미터도 정확히 같다: 2,657,447,936개(VLM의 23.99%).
  - head ↔ MLP 비율을 그대로 두는 이유가 있다. 폐루프는 예산이 아니라 자른 head 수를 따라가고, 이 비율을 움직이면 그것이 또 다른 교락이 된다.
- **중요도.** `importance_v2`(calib_100, MoOP와 같은 파일). 층 안 선택 점수도 MoOP와 같은 max(rank_norm(I_traj), rank_norm(I_CoC))다.
- **35번 층.** I_traj가 0이라(expert KV로 가는 경로 없음) 불일치를 정의할 수 없다. 7월 격자와 같이 30–34번 층 평균으로 채운다.
- **기준 행은 다시 돌리지 않는다.**
  - MoOP l0keep의 Ada Test500 = `outputs/swap_ship_L0fullM_500`. 마스크가 `dual_u40_l0keep_v2`와 같음을 11-09에 확인했다.
  - Dense = `outputs/baseline_pred_test`.

## 설계

1. **불일치 계산.** 층 ℓ = 1..35에서 Q head와 MLP 채널을 따로 계산한다.
   - d_ℓ = 1 − \|T ∩ C\| / \|T ∪ C\|.
   - 상위 집합 크기는 MoOP가 그 층에서 남기는 개수다: Q 19개, MLP 7,359/7,360개.
2. **절단 수 배분.** 층 ℓ의 절단 수 = 평균 절단 수 × (1 − 강도 × z_ℓ).
   - z_ℓ는 1–35번 층의 불일치를 평균 0으로 옮기고, 가장 먼 값이 ±1이 되게 나눈 값이다.
   - 강도 0.5(arm `dis050`): 불일치가 가장 큰 층은 평균의 50%, 가장 작은 층은 150%를 자른다.
   - 강도 1.0(arm `dis100`): 같은 기준으로 0%와 200%를 자른다.
   - 정수화는 최대 나머지 방식(동률은 앞 층부터)으로 해서 총수를 정확히 맞춘다.
3. **층 안 선택.** 각 층에서 MoOP 점수가 가장 낮은 유닛부터 그 층의 절단 수만큼 자른다(`mask_lib.select_mask_ratios`와 같은 규칙).
4. **나머지.** expert, KV group, vision tower, embedding은 건드리지 않는다.

**미리 계산한 배분** (`importance_v2`, 층 구간별 평균 절단 수; MoOP는 Q 13, MLP 약 4,929)

| 층 구간 | Q head dis050 | Q head dis100 | MLP dis050 | MLP dis100 |
|---|---|---|---|---|
| 1–5 | 16.2 | 18.8 | 7,100 | 9,272 |
| 6–17 | 14.2 | 15.6 | 5,070 | 5,212 |
| 18–29 | 11.8 | 10.6 | 4,419 | 3,910 |
| 30–35 | 10.2 | 7.8 | 3,855 | 2,781 |
| 층별 범위 | 7–20 | 1–26 | 3,751–7,393 | 2,572–9,858 |

불일치는 깊이에 따라 거의 단조롭게 커진다. MLP에서 두 기준의 순위 상관은 1–5번 층 0.96–0.99에서 28–34번 층 0.22–0.33으로 떨어진다. 그래서 두 arm 모두 "앞 층을 더, 뒤 층을 덜 자르는" 배분이 된다.

## 구현

| 단계 | 파일 / 명령 | 자원 |
|---|---|---|
| 마스크 | `iclr27-rebuttal/code/make_disagree_masks.py` (새 파일). 강도 0 / 0.5 / 1.0의 마스크와 층별 불일치 · 절단 수를 `outputs/iclr27-rebuttal/disagree_alloc/{s000,s050,s100}/`(`masks.npz` = vq (36, 32), vm (36, 12288), 1 = 남김; `alloc.json`)에 쓴다 | CPU, 수 분 |
| 빌드 | `make_slim.py --config maskfile_iclr27-rebuttal/disagree_alloc/s050 --importance importance_v2 --no-state --out outputs/iclr27-rebuttal/disagree_alloc/slim_dis050` (s100도 같음). `--no-state`라 레시피만 저장한다(평가 시 기본 가중치에서 재구성, 디스크 거의 0) | Ada 1장, arm당 약 5분 |
| 평가 | `iclr27-rebuttal/code/run_disagree_alloc.sh` (새 워커, `run_ada_fill.sh`와 같은 mkdir 큐). 빌드 2개 → `run_baseline.py --set test --model <slim> --exp-id iclr27-rebuttal/disagree_alloc/<arm>_test500 --shard i --n-shards 4 --gpu 0 --reserve-gb 40` × 8. k = 8, seed 42, max_gen 256, rollout만. 카드당 프로세스 1개, `HF_HUB_OFFLINE=1` | **Ada 4–7번**, 약 45분 |
| 분석 | `iclr27-rebuttal/code/analyze_disagree_alloc.py` (새 파일, `ci_openloop_lingoqa`의 부트스트랩 재사용). 아래 판정 항목과 층별 절단 수 그림 | CPU |

## 게이트

- **G0 (GPU 전, CPU).**
  - 강도 0 마스크가 `outputs/slim_dual_u40_l0keep_v2/slim_meta.json`의 남김 집합을 유닛 단위로 재현한다.
  - dis050 · dis100 모두 다음을 만족한다.
    - 0번 층이 MoOP와 같다.
    - Q 절단 455개, MLP 절단 177,437개(0번 층 포함), 제거 파라미터 2,657,447,936개다.
    - 어느 층도 전부 잘리지 않는다.
  - 하나라도 어긋나면 빌드하지 않는다.
- **G1 (빌드).** make_slim이 출력한 제거 파라미터 수가 G0과 같고, `slim_meta.json`의 남김 집합이 마스크와 같다. `importance_from`이 `importance_v2`인지 확인한다(make_slim 기본값은 v1이다).
- **G2 (실행 초기).** 각 arm 첫 10클립이 Ada 카드에서 돌고, NaN이 없고, ADE가 상식 범위다.
- **G3 (완료).** arm마다 500/500클립.

## 판정 (사전 등록)

- **1차 지표.** MoOP l0keep 대비 짝 차이 Δ minADE@6, 6.4 s. 95% 클립 부트스트랩 CI, 200,000회, 시드 1로 재확인.
  - 개선: CI 상한 < 0
  - 동등: CI가 ±0.03 m 안. ±0.03 m는 0번 층 보존 arm들의 흔들림(SD 0.018)으로 정한 기존 기준이다
  - 악화: CI 하한 > 0
  - 그 밖: 불확정
- **2차 지표.**
  - 1.6 · 3.2 · 6.4 s의 minADE@6 · minFDE@6(평균, CI, 중앙값). Dense 대비 차이도 함께 본다.
  - CoC 퇴화율, 주행 상황별(cruise · decel · accel · turn) 차이, 두 강도 사이의 용량-반응.
- **주의.** 11-09 재분석에서 Test500은 MoOP l0keep에 불리한 클립 구성으로 보였다. rank-mean · FM-only가 Test500에서만 좋아 보였다. 그래서 Test500에서 "개선"이 나오면 논문 주장 전에 val500이나 A100 test_2000에서 재현을 확인해야 한다.

## 이전 증거 · 위험

- **7월 격자의 "agree" 배분.**
  - 설계: 남김 비율을 상위 집합 합집합 크기에 비례시켰다. 비례 상수는 전체 파라미터에 하나만 썼다.
  - 결과: dual에서 uniform보다 나빴다(minADE 중앙값 Δ +0.179 대 +0.098).
  - 조건이 지금과 다르다: 7월 50클립 중요도, 0번 층 절단, val 50클립. 같은 결론이 나올지는 이번 실험이 답한다.
- **깊이 교락 (D3).** 불일치가 깊이와 거의 같이 움직인다. 그래서 결과는 "이 배분"의 효과로만 해석하고, "불일치 정보 자체"의 효과라고는 말하지 않는다.
- **강한 arm의 앞 층 절단.** dis100은 1–5번 층 MLP를 평균 약 75%(9,272채널) 자른다.
  - 1–4번 층 MLP를 더 자른 l14m arm은 주행에서 무해했지만 LingoQA를 3.6–6.0 pp 잃었다. 이번 계획은 LingoQA를 재지 않는다.
  - expert는 0–15번 층 캐시에 민감하다(메모리 기록). 앞 층 절단이 그쪽으로 해가 될 수 있다.
- **층별 head 분포.** 총 head 수는 같지만 층별 분포가 크게 바뀐다(dis100에서 33번 층 1개, 3번 층 26개). 폐루프 영향은 이번 범위 밖이다.

## 비용 · 일정

- 승인 후 약 1시간이다: G0 수 분, 빌드 약 10분, 평가 약 45분, 분석 수 분.
- 띄우기 직전에 `nvidia-smi`로 4–7번이 비었는지(2 MiB) 확인한다. 디스크는 레시피만 쓰므로 거의 0이다.

## 산출물

- `outputs/iclr27-rebuttal/disagree_alloc/`: 마스크, `alloc.json`, Test500 런, `analysis/`(`config.json`, `metrics.json`, `summary.txt`, 그림)
- 보고서 `iclr27-rebuttal/reports/2026-11-10_moop-disagreement-allocation-test500.md`, README 목록 갱신
